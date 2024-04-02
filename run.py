import argparse
import ipdb
import os
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


"""
Run DepthAnything single image depth prediction model on an image 
folder and store the result including a visualization.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--outdir", type=str, default="./vis_depth")
    parser.add_argument(
        "--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"]
    )
    parser.add_argument(
        "--save_array",
        action="store_true",
        help="save the raw depth array for later usage",
    )
    parser.add_argument(
        "--pred-only",
        dest="pred_only",
        action="store_true",
        help="only display the prediction when saving",
    )
    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="do not apply colorful palette",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="normalize the depth map to [0, 1.0]",
    )
    return parser.parse_args()


def visualize(depth: torch.Tensor, raw_image: np.ndarray, args, filename) -> np.ndarray:
    """Plot the depth map and maybe the raw image side by side."""
    margin_width, caption_height = 50, 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 1, 2
    h, w = raw_image.shape[:2]

    if args.grayscale:
        depth_rgb = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth_rgb = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    if args.pred_only:
        cv2.imwrite(
            os.path.join(args.outdir, filename[: filename.rfind(".")] + "_depth.png"),
            depth_rgb,
        )
    else:
        split_region = (
            np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
        )
        combined_results = cv2.hconcat([raw_image, split_region, depth_rgb])

        caption_space = (
            np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8)
            * 255
        )
        captions = ["Raw image", "Depth Anything"]
        segment_width = w + margin_width

        for i, caption in enumerate(captions):
            # Calculate text size
            text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

            # Calculate x-coordinate to center the text
            text_x = int((segment_width * i) + (w - text_size[0]) / 2)

            # Add text caption
            cv2.putText(
                caption_space,
                caption,
                (text_x, 40),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

        final_result = cv2.vconcat([caption_space, combined_results])

        cv2.imwrite(
            os.path.join(
                args.outdir, filename[: filename.rfind(".")] + "_img_depth.png"
            ),
            final_result,
        )


def preprocess(raw_image: np.ndarray):
    """Preprocess RGB Image in [0, 255] value range"""
    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    return image


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""

    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred["metric_depth"] if "metric_depth" in pred else pred["out"]
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    # Average over flipped predictions
    return 0.5 * (pred1 + pred2)


def get_images(input_path):
    if os.path.isfile(input_path):
        if input_path.endswith("txt"):
            with open(input_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            filenames = [input_path]
    else:
        filenames = os.listdir(input_path)
        filenames = [
            os.path.join(input_path, filename)
            for filename in filenames
            if (not filename.startswith(".") and not filename.endswith(".png"))
        ]
        filenames.sort()
    return filenames


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """Colorize depth maps."""
    import matplotlib

    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def plot_2d(rgb: np.ndarray, depth: np.ndarray, cmap: str = "Spectral") -> None:
    import matplotlib.pyplot as plt

    # Colorize manually to appealing colormap
    percentile = 0.03
    min_depth_pct = np.percentile(depth, percentile)
    max_depth_pct = np.percentile(depth, 100 - percentile)
    depth_colored = colorize_depth_maps(
        depth, min_depth_pct, max_depth_pct, cmap=cmap
    ).squeeze()  # [3, H, W], value in (0, 1)
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Plot the Image, Depth, and Uncertainty side-by-side in a 1x2 grid
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)
    ax[0].set_title("Image")
    ax[1].imshow(chw2hwc(depth_colored))
    ax[1].set_title("Depth")
    ax[0].axis("off"), ax[1].axis("off")

    plt.show()


def load_hf_model(args):
    # Load from huggingface model hub
    modelstr = "LiheYoung/depth_anything_{}14"
    depth_anything = DepthAnything.from_pretrained(modelstr.format(args.encoder))
    return depth_anything.to(DEVICE).eval()


def main():
    args = parse_args()

    depth_anything = load_hf_model(args)
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    filenames = get_images(args.img_path)

    os.makedirs(args.outdir, exist_ok=True)

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        h, w = raw_image.shape[:2]
        image = preprocess(raw_image)

        with torch.no_grad():
            # Inference with flip augmentation
            depth = infer(depth_anything, image)

        # depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=False)[0, 0]
        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]

        if args.normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy()

        plot_2d(raw_image[..., ::-1], depth)

        # Save and visualize the depth map
        filename = os.path.basename(filename)
        visualize(depth, raw_image, args, filename)
        np.save(
            os.path.join(args.outdir, filename[: filename.rfind(".")] + "_depth.npy"),
            depth,
        )


if __name__ == "__main__":
    main()
