import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
from tqdm import tqdm
import torch.multiprocessing as mp

from kv_tracker.sam_interface import SAMInterface
from kv_tracker.image import pi3_resize_image

class liveData(SAMInterface):

    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)

        self.init_rs_cam()
        self.init_models()

        # Set the length of the dataset to be infinite
        self.length = 10**10

    def init_rs_cam(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Create streams
        self.width = 640
        self.height = 480
        self.fps = 30
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        # config.enable_stream(
        #     rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        # )
        # self.frame_aligner = rs.align(rs.stream.color)

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        intrin = rs.video_stream_profile(
            pipeline_profile.get_stream(rs.stream.color)
        ).get_intrinsics()
        print(f"Realsense intrinsics: {intrin}")
        self.intrinsics = torch.tensor(
            [
                [intrin.fx, 0.0, intrin.ppx],
                [0.0, intrin.fy, intrin.ppy],
                [0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        # Sanity Check
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break
        if not found_rgb:
            raise ValueError("Live Capture requires Depth camera with Color sensor")

        # Start streaming
        self.profile = self.pipeline.start(config)
        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        # self.rgb_sensor.set_option(rs.option.exposure, 200)
        for _ in tqdm(range(60)):  # Warmup to avoid initial adjustment
            self.get_rgb_frame()

    def get_rgb_frame(self, idx=0):
        frames = self.pipeline.wait_for_frames()
        rgb_frame = np.asanyarray(frames.get_color_frame().get_data())
        return rgb_frame.copy()

    def next_rgb_depth_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.frame_aligner.process(frames)
        color_frame = np.asanyarray(frames.get_color_frame().get_data())
        depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
        return color_frame, depth_frame

    def get_gt_pose(self, idx=None):
        return np.eye(4)


if __name__ == "__main__":

    live_data = liveData("cuda:0")
    live_data.init_segmentation()
    while True:
        live_data.get_frame(debug=True)
