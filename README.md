# [CVPR 2026] KV-Tracker: Real-Time Pose Tracking with Transformers

### 🏆 Best Demo Award

### [Project Page](https://marwan99.github.io/kv_tracker/) | [arXiv](https://arxiv.org/abs/2512.22581)

**Marwan Taher, Ignacio Alzugaray, Kirill Mazur, Xin Kong, Andrew J. Davison**

> Accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026.

## Installation

```
git clone --recursive git@github.com:Marwan99/kv_tracker.git
cd kv_tracker
uv venv --python 3.11
source .venv/bin/activate
# Install the correct torch for your CUDA
uv pip install torch torchvision setuptools --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-build-isolation -e .
```

### Object level

Download the SAM2.1 checkpoint for object level usage.

```
wget -P thirdparty/segment-anything-2-real-time/checkpoints \
 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

use `--obj_mode` to run in object mode. 

## Live demo with real-sense camera

```
python main.py config/live.yaml --cam_only --resize_dim 308 --rerun
```

## Run on captured video

Modify `scene_dir` in `config/video.yaml` to point to your video.

```
python main.py config/video.yaml --cam_only --resize_dim 308 --rerun
```

## Eval


### Preparing datasets

The eval scripts expect datasets under `datasets/`. Use the helper scripts to
download and extract them into the expected layout:

```
# 7-Scenes
bash scripts/download_7_scenes.sh

# TUM RGB-D
bash scripts/download_tum.sh
```

If you already have a copy, you can symlink it instead:

```
mkdir -p datasets
ln -s /path/to/7-scenes datasets/7-scenes
ln -s /path/to/tum_rgbd datasets/tum_rgbd
```

### Running evaluation

```
# 7-scenes with main pipeline
python run_dataset.py --cam_only --resize_dim 308 --dataset 7scenes --results test
python eval.py --dataset 7scenes --results results_test

# TUM with depth anything pipeline
python run_dataset.py --dataset tum --pipeline depth_anything --results test
python eval.py --dataset tum --results results_test
```

## Acknowledgements

Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:

- [Pi3](https://github.com/yyfz/Pi3)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)

## Citation

```
@InProceedings{Taher_2026_CVPR,
    author    = {Taher, Marwan and Alzugaray, Ignacio and Mazur, Kirill and Kong, Xin and Davison, Andrew},
    title     = {KV-Tracker: Real-Time Pose Tracking with Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {28990-28999}
}
```