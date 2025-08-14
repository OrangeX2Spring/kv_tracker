import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation as R

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

class TUMLoader(SAMInterface):

    def __init__(self, device, **cfg):
        super().__init__(device, **cfg)

        self.scene_dir = cfg["scene_dir"]

        self.rgb_paths = sorted(glob(f"{self.scene_dir}/rgb/*.png"))
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

        trans_and_quat = np.loadtxt(gt_pose_dir + "/groundtruth.txt")#[:, 1:]  # skip timestamp
        all_gt_poses = []
        timestamps = []
        for pose in trans_and_quat:
            tx, ty, tz = pose[1:4]
            qx, qy, qz, qw = pose[4:8]

            rot = R.from_quat([qx, qy, qz, qw])
            rot_mat = rot.as_matrix()

            gt_pose = np.eye(4)
            gt_pose[0:3, 0:3] = rot_mat
            gt_pose[0:3, 3] = [tx, ty, tz]

            all_gt_poses.append(gt_pose)
            timestamps.append(pose[0])

        timestamps = np.array(timestamps)


        # strip path to get timestamp
        rgb_paths = sorted(glob(f"{gt_pose_dir}/rgb/*.png"))
        rgb_timestamps = [Path(p).stem for p in rgb_paths]
        
        gt_poses = []
        for rgb_ts in rgb_timestamps:
            rgb_ts_float = float(rgb_ts)
            # find closest timestamp in gt timestamps
            idx = (np.abs(timestamps - rgb_ts_float)).argmin()
            gt_poses.append(all_gt_poses[idx])
        gt_poses = np.array(gt_poses)
        return gt_poses

    def get_gt_pose(self, i):
        return np.eye(4)