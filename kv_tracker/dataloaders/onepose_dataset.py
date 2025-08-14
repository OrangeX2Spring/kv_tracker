import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import torch.multiprocessing as mp
import imageio
from pathlib import Path
import os


from glob import glob

from kv_tracker.sam_interface import SAMInterface
from kv_tracker.image import pi3_resize_image

def get_all_scenes_dir(dataset_dir):
    objs_dirs = sorted(glob(f"{dataset_dir}/*"))

    scenes_list = []
    for obj_dir in objs_dirs:
        if not os.path.isdir(obj_dir):
            continue

        scenes = sorted(os.listdir(obj_dir))
        valid_scenes = [s for s in scenes if os.path.isdir(os.path.join(obj_dir, s))]

        # for scene in scenes:
        scene_path = os.path.join(obj_dir, valid_scenes[-1])
        # print(scene_path)
        # only add if it's a directory (filter out files like box3d_corners.txt)
        if os.path.isdir(scene_path):
            scenes_list.append(scene_path)

    print(f"Found {len(scenes_list)} scenes")
    return scenes_list

class onePoseLoader(SAMInterface):

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)

        self.scene_dir = kwargs.get("scene_dir", "")
        self.scene_dir = Path(self.scene_dir)

        print(f"Running: {self.scene_dir}")

        video_path = self.scene_dir / "Frames.m4v"

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        gt_pose_dir = self.scene_dir / "poses_ba"
        print(f"Loading gt poses from: {gt_pose_dir}")
        self.gt_poses_np = self.load_gt(gt_pose_dir)
        self.length = self.gt_poses_np.shape[0] - 2

        frame0 = self.get_rgb_frame(0)
        self.height = frame0.shape[0]
        self.width = frame0.shape[1]

        bbox_center = self.get_bb_center(idx=1)
        self.init_mask_coords = [
            bbox_center, 
            bbox_center+50,
            bbox_center-50,
            bbox_center-100,
            bbox_center+100
            ]

        self.init_models()
        # self.init_bbox_segmentation()
        # self.init_segmentation()
        init_mask = self.get_init_mask()
        self.init_SAM_w_mask(init_mask)

    @staticmethod
    def load_gt(gt_pose_dir):
        gt_poses_paths = sorted(Path(gt_pose_dir).glob("*.txt"), key=lambda x: int(x.stem))

        gt_poses_list = []
        for pose in gt_poses_paths:
            temp_pose = np.loadtxt(pose)
            gt_poses_list.append(temp_pose)

        gt_poses_np = np.array(gt_poses_list)

        return gt_poses_np

    def get_rgb_frame(self, idx=0):

        ret, bgr = self.cap.read()
        if not ret:
            raise "End of video"

        rgb = bgr[:, :, ::-1]
        return rgb

    def get_gt_pose(self, idx):
        return self.gt_poses_np[idx]

    def get_bbox(self, idx):
        bbox_path = self.scene_dir / f"reproj_box/{idx}.txt"
        return np.loadtxt(bbox_path)
    
    def init_bbox_segmentation(self):
        bbox = self.get_bbox(0)

        min_x = bbox[:, 0].min()
        min_y = bbox[:, 1].min()
        max_x = bbox[:, 0].max()
        max_y = bbox[:, 1].max()

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(self.width, max_x)
        max_y = min(self.height, max_y)

        bbox_extent = np.array([[min_x, min_y], [max_x, max_y]])
        self.init_SAM_w_bbox(bbox_extent)

    def get_bb_center(self, idx):
        bbox_path = self.scene_dir / f"reproj_box/{idx}.txt"
        bbox_np = np.loadtxt(bbox_path)
        return bbox_np.mean(axis=0)
    
    def get_init_mask(self):
        mask_np = cv2.imread(str(self.scene_dir / "init_mask.png"), cv2.IMREAD_GRAYSCALE)
        binary_mask = mask_np > 127
        return binary_mask
    
    def close_cap(self):
        self.cap.release()
