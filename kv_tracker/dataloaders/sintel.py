import os
import cv2
import struct
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

class SintelLoader(SAMInterface):

    def __init__(self, device, **cfg):
        super().__init__(device, **cfg)

        self.scene_dir = cfg["scene_dir"]

        self.rgb_paths = sorted(glob(f"{self.scene_dir}/*.png"))
        self.offset = cfg.get("offset", 0)

        self.rgb_paths = self.rgb_paths[self.offset :]
        self.length = len(self.rgb_paths)

        gt_pose_dir = self.scene_dir.replace("final", "camdata_left")
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
        gt_poses_paths = sorted(Path(gt_pose_dir).glob("*.cam"))

        gt_poses_list = []
        for cam_file_path in gt_poses_paths:

            # with open(cam_file_path, 'rb') as f:
            #     # Read and verify the tag (4 bytes as float32)
            #     tag_bytes = f.read(4)
            #     tag = struct.unpack('<f', tag_bytes)[0]  # little-endian float
                
            #     if not np.isclose(tag, 202021.25):
            #         raise ValueError(f"Invalid tag: {tag}, expected 202021.25")
                
            #     # Read intrinsic matrix (3x3 = 9 float64 values = 72 bytes)
            #     intrinsic = np.fromfile(f, dtype='<f8', count=9).reshape(3, 3)
                
            #     # Read extrinsic matrix (3x4 = 12 float64 values = 96 bytes)
            #     extrinsic = np.fromfile(f, dtype='<f8', count=12).reshape(3, 4)
            #     temp_tf = np.eye(4)
            #     temp_tf[:3, :4] = extrinsic

            # gt_poses_list.append(temp_tf)

            TAG_FLOAT = 202021.25

            f = open(cam_file_path, "rb")
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert (
                check == TAG_FLOAT
            ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
                TAG_FLOAT, check
            )
            intrinsics = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
            extrinsic = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
            temp_tf = np.eye(4)
            temp_tf[:3, :4] = extrinsic
            gt_poses_list.append(temp_tf)

        gt_poses_np = np.array(gt_poses_list)

        return gt_poses_np


    def get_gt_pose(self, i):
        return np.eye(4)