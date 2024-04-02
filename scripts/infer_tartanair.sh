#!/bin/bash

# List of scenes
scenes=("P001" "P002" "P004" "P005" "P006" "P008" "P009" "P010" "P011")

# Loop over the scenes
for scene in "${scenes[@]}";do
	echo "Running inference on ${scene} ..."
	python demo.py -w checkpoints/metric/outdoor/depth_anything_metric_depth_outdoor.pt -i /media/data/TartanAir/abandonedfactory/Easy/${scene}/image_left/ -o /media/data/TartanAir/abandonedfactory/Easy/${scene}/depthany-vitl-outdoor_left/
done
