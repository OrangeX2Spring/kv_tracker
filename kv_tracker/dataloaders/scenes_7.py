import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import torch.multiprocessing as mp

from pathlib import Path
from glob import glob

from kv_tracker.sam_interface import SAMInterface
from kv_tracker.image import pi3_resize_image

def get_all_scenes_dir(dataset_dir):
    scenes = sorted(glob(f"{dataset_dir}/*"))

    scenes_list = []
    for scene_path in scenes:
        if not os.path.isdir(scene_path):
            continue

        scenes_list.append(scene_path)

    print(f"Found {len(scenes_list)} scenes")
    return scenes_list

class scenes7Loader(SAMInterface):

    def __init__(self, device, **cfg):
        super().__init__(device, **cfg)

        self.scene_dir = cfg["scene_dir"]

        self.rgb_paths = sorted(glob(f"{self.scene_dir}/seq-01/*.color.png"))
        self.offset = cfg.get("offset", 0)

        self.rgb_paths = self.rgb_paths[self.offset :]
        self.length = len(self.rgb_paths)

        gt_pose_dir = f"{self.scene_dir}/seq-01"
        print(f"Loading gt poses from: {gt_pose_dir}")
        # self.gt_poses_np = self.load_gt(gt_pose_dir)

        self.intrinsics = 0

        frame0 = self.get_rgb_frame(0)
        self.height = frame0.shape[0]
        self.width = frame0.shape[1]

        self.init_models()

    def get_rgb_frame(self, idx=0):
        bgr = cv2.imread(self.rgb_paths[idx])
        rgb = bgr[:, :, ::-1]
        return rgb
    
    @staticmethod
    def load_gt(gt_pose_dir):
        # gt_poses_paths = sorted(Path(gt_pose_dir).glob("*.pose.txt"), key=lambda x: int(x.stem))
        gt_poses_paths = sorted(Path(gt_pose_dir).glob("*.pose.txt"))

        gt_poses_list = []
        for pose in gt_poses_paths:
            temp_pose = np.loadtxt(pose)
            gt_poses_list.append(temp_pose)

        gt_poses_np = np.array(gt_poses_list)

        return gt_poses_np


    def get_gt_pose(self, i):
        return np.eye(4)