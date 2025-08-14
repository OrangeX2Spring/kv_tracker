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



class phoneLoader(SAMInterface):

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)

        self.scene_dir = kwargs.get("scene_dir", "")
        self.scene_dir = Path(self.scene_dir)

        print(f"Running: {self.scene_dir}")

        video_path = self.scene_dir

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.length = 100000

        frame0 = self.get_rgb_frame(0)
        self.height = frame0.shape[0]
        self.width = frame0.shape[1]

        self.init_models()
        # self.init_segmentation()
        # self.init_segmentation_interactive_plt()


    def get_rgb_frame(self, idx=0):

        ret, bgr = self.cap.read()
        if not ret:
            raise "End of video"

        rgb = bgr[:, :, ::-1]
        return rgb

    def get_gt_pose(self, idx):
        return np.eye(4)

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
