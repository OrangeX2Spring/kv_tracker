import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
from pathlib import Path
import torch.multiprocessing as mp

from glob import glob

from kv_tracker.sam_interface import SAMInterface
from kv_tracker.image import pi3_resize_image

class arcticLoader(SAMInterface):

    def __init__(self, device, **cfg):
        super().__init__(device, **cfg)

        self.scene_dir = Path(cfg["scene_dir"])
        
        # if cfg.get("use_cached_masked_imgs", False):
        #     rgb_dir = str(self.scene_dir / "sam_segmented" / "*.png")
        #     self.rgb_paths = sorted(glob(rgb_dir))
        # else:
        rgb_dir = str(self.scene_dir / "0" / "*.jpg")
        self.rgb_paths = sorted(glob(rgb_dir))

        mask_dir = str(self.scene_dir / "mask/*.png")
        self.masks_paths = sorted(glob(mask_dir))

        self.offset = cfg["offset"]
        self.rgb_paths = self.rgb_paths[self.offset :]
        self.masks_paths = self.masks_paths[self.offset :]

        # self.masks_paths = []
        self.length = len(self.rgb_paths)
        self.intrinsics = 0

        frame0 = self.get_rgb_frame(0)
        self.height = frame0.shape[0]
        self.width = frame0.shape[1]

        self.init_models()
        # self.init_segmentation_interactive()
        # exit()

        # init_mask = self.get_gt_mask(0)
        # self.init_SAM_w_mask(init_mask)

        init_mask = self.get_init_mask()
        self.init_SAM_w_mask(init_mask)

        # x, y = np.where(init_mask)
        # coords = np.stack([y, x], axis=-1)
        # # get 10 coords evenly spaced
        # coords_subset = coords[:: max(1, len(coords) // 5)]
        # self.init_SAM_w_points_from_mask(init_mask)

        # get 2d bbox from mask
        # ys, xs = np.where(init_mask)
        # x_min, x_max = xs.min(), xs.max()
        # y_min, y_max = ys.min(), ys.max()
        # # inflate bbox a bit
        # inflate = 10
        # x_min = max(0, x_min - inflate)
        # x_max = min(self.width - 1, x_max + inflate)
        # y_min = max(0, y_min - inflate)
        # y_max = min(self.height - 1, y_max + inflate)
        # bbox = [[x_min, y_min], [x_max , y_max]]
        # self.init_SAM_w_bbox(bbox)

    def get_rgb_frame(self, idx=0):
        bgr = cv2.imread(self.rgb_paths[idx])
        rgb = bgr[:, :, ::-1]
        return rgb
    
    def get_gt_mask(self, idx=0):
        mask_img = cv2.imread(self.masks_paths[idx])
        binary_mask = mask_img.sum(axis=-1) == 300

        binary_mask = binary_mask.astype(np.uint8) * 255

        # return binary_mask
        # Step 1: Morphological opening to remove small white specs
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Step 2: Remove remaining small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Find the largest component (assuming it's the main object)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask = (labels == largest_component).astype(np.uint8) * 255
        
        # Step 3: Morphological closing to smooth and fill small holes
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return clean_mask.astype(bool)

    def get_init_mask(self):
        mask_np = cv2.imread(str(self.scene_dir / "init_mask.png"), cv2.IMREAD_GRAYSCALE)
        binary_mask = mask_np > 127
        return binary_mask

    def get_gt_pose(self, i):
        return np.eye(4)