import argparse
import ipdb
import os
from pathlib import Path
import glob
from typing import Union, Optional

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from metric_depth.zoedepth.models.builder import build_model
from metric_depth.zoedepth.utils.config import get_config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def preprocess(image_raw: np.ndarray) -> torch.Tensor:
    image_tensor = transforms.ToTensor()(image_raw).unsqueeze(0).to(DEVICE)
    return image_tensor


def postprocess(
    pred: torch.Tensor, ht: int, wd: int, normalize: bool = False
) -> torch.Tensor:

    if pred.ndim == 3:
        pred = F.interpolate(pred[None], (ht, wd), mode="bilinear", align_corners=False)
    else:
        pred = F.interpolate(pred, (ht, wd), mode="bilinear", align_corners=False)
    if normalize:
        pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0
    return pred[0, 0]


def get_depth_from_prediction(pred) -> torch.Tensor:
    if isinstance(pred, torch.Tensor):
        pass
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    elif isinstance(pred, dict):
        pred = pred["metric_depth"] if "metric_depth" in pred else pred["out"]
    else:
        raise NotImplementedError(f"Unknown output type {type(pred)}")
    return pred


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
            if not filename.startswith(".")
            # if (not filename.startswith(".") and not filename.endswith(".png"))
        ]
        filenames.sort()
    return filenames


@torch.no_grad()
def infer(model, images: torch.Tensor, **kwargs) -> torch.Tensor:
    """Inference with flip augmentation"""

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    # Average over flipped predictions
    return 0.5 * (pred1 + pred2)


# Depth Anything sadly has some artifacts at the borders of the image for indoor scenes
def process_images(
    model,
    image_paths,
    output_dir,
    normalize: bool = False,
    dataset: Optional[str] = None,
):
    for image_path in tqdm(image_paths, desc="Processing Images"):

        image_raw = Image.open(image_path).convert("RGB")
        original_width, original_height = image_raw.size
        image_tensor = preprocess(image_raw)

        # 2 forwards with flip augmentation
        pred = infer(model, image_tensor, dataset=dataset)
        # Single forward
        # pred = model(image_tensor, dataset=dataset)

        pred = get_depth_from_prediction(pred)
        pred = postprocess(pred, original_height, original_width, normalize)
        pred = pred.squeeze().detach().cpu().numpy()

        # plot_2d(image_raw, pred, cmap="Spectral")

        # Save depth map with numpy
        fname = Path(image_path).stem + ".npy"
        output_path = os.path.join(output_dir, fname)

        np.save(output_path, pred)


def print_num_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))


def main(args):
    # Lets not pick a fight with the model's dataloader
    if "indoor" in args.weights:
        dataset = "nyu"
    elif "kitti" in args.weights:
        dataset = "kitti"
    else:
        dataset = None

    config = get_config(args.model, "infer", dataset)
    config.pretrained_resource = "local::./" + args.weights
    # config.dataset = dataset

    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print_num_parameters(model)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    image_paths = get_images(args.input_dir)
    process_images(model, image_paths, args.output_dir, args.normalize, dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="zoedepth", help="Name of the model to test"
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="checkpoints/depth_anything_metric_depth_indoor.pt",
        help="Pretrained resource to use for fetching weights.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="./data",
        help="Directory containing images to process",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save processed point clouds",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="normalize the depth map to [0, 1.0]",
    )

    args = parser.parse_args()
    main(args)
