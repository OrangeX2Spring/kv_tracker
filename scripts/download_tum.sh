#!/bin/bash

dest="datasets/tum_rgbd"
mkdir -p "$dest"
cd "$dest"

base="https://cvg.cit.tum.de/rgbd/dataset/freiburg1"
scenes=(360 floor desk desk2 room plant teddy xyz rpy)

for scene in "${scenes[@]}"; do
    file_name="rgbd_dataset_freiburg1_$scene.tgz"
    echo "Downloading $file_name..."
    wget "$base/$file_name"
    echo "Extracting $file_name..."
    tar -xzf "$file_name"
done
