import os
import cv2
import time
import torch
import numpy as np

from kv_tracker.image import pi3_resize_image
from sam2.build_sam import build_sam2_camera_predictor


class SAMInterface:

    def __init__(self, device, **kwargs):
        self.device = device
        self.resize_dim = kwargs.get("resize_dim", 308)
        self.cfg = kwargs

        # camera initializations; to be populated by initialised camera/data source
        self.intrinsics = None
        self.height = None
        self.width = None
        self.init_mask_coords = None

    def init_models(self, model_cfg=None, checkpoint=None):
        print("Dataset initialized....")

        # --------------------- SAM2 Initializations ---------------------
        # checkpoint = "/home/marwan/track3r/third_party/segment-anything-2-real-time/checkpoints/sam2_hiera_base_plus.pt"
        # model_cfg = "sam2_hiera_b+.yaml"

        if model_cfg is None and checkpoint is None:
            model_cfg = "sam2.1_hiera_s.yaml"
            checkpoint = "thirdparty/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"

        self.seg_predictor = build_sam2_camera_predictor(model_cfg, checkpoint)
        self.seg_init = False

    def get_rgb_frame(self, idx=0):
        """
        Output shape: (H, W, 3),
        Output dtype: uint8
        Output format: RGB
        """
        raise NotImplementedError

    def close_cap(self):
        pass
    
    def init_SAM(self, points, labels, rgb_frame):
        """
        Initialize SAM with given points and labels on the first frame
        points: Nx2 numpy array of (x, y) coordinates
        labels: N numpy array of labels (1 for foreground, 0 for background)
        rgb_frame: HxWx3 numpy array of the RGB image
        """

        self.seg_predictor.load_first_frame(rgb_frame)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.seg_predictor.add_new_prompt(
                frame_idx=0, obj_id=1, points=points, labels=labels
            )

        self.seg_init = True

    def init_segmentation(self):
        """
        create opencv window to annotate the object to be tracked,
        get clicks, stores them in a list and space bar to confirm
        """
        if self.seg_init is True:
            return

        points = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])


        scale = 1.0
        if self.init_mask_coords is None:
            cv2.namedWindow('Init Segmentation')
            cv2.setMouseCallback("Init Segmentation", mouse_callback)
            while len(points) == 0:
                rgb_frame = self.get_rgb_frame(0)

                # if frame is too large, resize for visualization
                vis_frame = rgb_frame.copy()
                vis_height, vis_width = vis_frame.shape[:2]
                max_dim = 800
                if max(vis_height, vis_width) > max_dim:
                    scale = max_dim / max(vis_height, vis_width)
                    vis_frame = cv2.resize(vis_frame, (int(vis_width * scale), int(vis_height * scale)))

                cv2.imshow("Init Segmentation", vis_frame[..., [2, 1, 0]])
                cv2.waitKey(1)

            # get click points
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if len(points) == 0:
                        print("Initializing object at center of the frame")
                        height, width = rgb_frame.shape[:2][::-1]
                        points = np.array([[height / 2, width / 2]], dtype=np.float32)
                    break
        else:
            print("Using pre-defined mask coordinates")
            rgb_frame = self.get_rgb_frame(0)
            points = self.init_mask_coords

            # visualise points
            # temp_img = rgb_frame.copy()
            # temp_img[points[:, 1], points[:, 0]] = [0, 0, 255]
            # cv2.imshow("Init Segmentation", temp_img[..., [2, 1, 0]])
            # cv2.waitKey(100000)

        points = np.array(points, dtype=np.float32) / scale
        labels = np.array([1] * points.shape[0], dtype=np.int32)

        self.seg_predictor.load_first_frame(rgb_frame)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.seg_predictor.add_new_prompt(
                frame_idx=0, obj_id=1, points=points, labels=labels
            )

        self.seg_init = True

        cv2.destroyAllWindows()

    def init_segmentation_interactive(self):
        """
        Interactive segmentation with dynamic mask preview.
        Click points on the image, then press:
        - 'p': mark last point as positive (foreground)
        - 'n': mark last point as negative (background)
        - space: finish and confirm segmentation
        """
        if self.seg_init is True:
            print("SAM already initialized.")
            return

        points = []
        labels = []
        next_label = 1  # Default to positive
        clicked_point = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_point
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point = [x, y]

        # Get the initial frame
        rgb_frame = self.get_rgb_frame(0)

        # Calculate scale for visualization if needed
        vis_height, vis_width = rgb_frame.shape[:2]
        max_dim = 800
        scale = 1.0
        if max(vis_height, vis_width) > max_dim:
            scale = max_dim / max(vis_height, vis_width)

        # Load first frame into SAM predictor
        self.seg_predictor.load_first_frame(rgb_frame)

        # Setup window
        cv2.namedWindow('Interactive Segmentation')
        cv2.setMouseCallback("Interactive Segmentation", mouse_callback)

        print("\n=== Interactive Segmentation ===")
        print("Click on image to add points")
        print("Press 'p' to mark last point as POSITIVE (foreground)")
        print("Press 'n' to mark last point as NEGATIVE (background)")
        print("Press SPACE to finish")
        print("================================\n")

        mask = None

        while True:
            # Create visualization frame
            vis_frame = rgb_frame.copy()
            if scale != 1.0:
                vis_frame = cv2.resize(vis_frame, (int(vis_width * scale), int(vis_height * scale)))

            # Overlay mask if available
            if mask is not None:
                mask_np = mask.cpu().numpy()
                if scale != 1.0:
                    mask_np = cv2.resize(mask_np.astype(np.uint8),
                                        (int(vis_width * scale), int(vis_height * scale)))

                # Create colored overlay (green for mask)
                overlay = vis_frame.copy()
                overlay[mask_np > 0] = overlay[mask_np > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                vis_frame = overlay.astype(np.uint8)

            # Draw points on visualization
            for i, (pt, lbl) in enumerate(zip(points, labels)):
                # Scale point coordinates for display
                display_pt = (int(pt[0] * scale), int(pt[1] * scale))
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255)  # Green for positive, red for negative
                cv2.circle(vis_frame, display_pt, 5, color, -1)
                cv2.circle(vis_frame, display_pt, 7, (255, 255, 255), 2)
                # Add label number
                cv2.putText(vis_frame, str(i+1), (display_pt[0] + 10, display_pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show status
            status = f"Points: {len(points)} | Next: {'POSITIVE' if next_label == 1 else 'NEGATIVE'}"
            cv2.putText(vis_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Interactive Segmentation", vis_frame[..., [2, 1, 0]])

            key = cv2.waitKey(1) & 0xFF

            # Handle clicked point
            if clicked_point is not None:
                # Unscale coordinates back to original frame
                original_pt = [clicked_point[0] / scale, clicked_point[1] / scale]
                points.append(original_pt)
                labels.append(next_label)

                # Update SAM with new points
                points_array = np.array(points, dtype=np.float32)
                labels_array = np.array(labels, dtype=np.int32)

                with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                    # Reset predictor state and reload frame
                    self.seg_predictor.reset_state()
                    self.seg_predictor.load_first_frame(rgb_frame)

                    # Get prediction with current points
                    self.seg_predictor.add_new_prompt(
                        frame_idx=0, obj_id=1, points=points_array, labels=labels_array
                    )
                    # Track to get the mask
                    _, mask_logits = self.seg_predictor.track(rgb_frame)
                    mask = (mask_logits[0] > 0.0).squeeze()

                print(f"Added {'POSITIVE' if next_label == 1 else 'NEGATIVE'} point at ({int(original_pt[0])}, {int(original_pt[1])})")
                clicked_point = None

            # Handle key presses
            if key == ord('p'):
                next_label = 1
                print("Next point will be POSITIVE (foreground)")
            elif key == ord('n'):
                next_label = 0
                print("Next point will be NEGATIVE (background)")
            elif key == ord(' '):
                if len(points) == 0:
                    print("No points added. Please add at least one point.")
                else:
                    print(f"Segmentation confirmed with {len(points)} points")
                    break
            elif key == 27:  # ESC key
                print("Segmentation cancelled")
                cv2.destroyAllWindows()
                return

        # mask_dir = self.scene_dir / "init_mask.png"
        # cv2.imwrite(str(mask_dir), (mask.cpu().numpy().astype(np.uint8)) * 255)
        # return

        self.seg_init = True
        cv2.destroyAllWindows()
    
    def init_segmentation_interactive_plt(self):
        """
        Interactive segmentation with dynamic mask preview using matplotlib.
        Click points on the image, then press:
        - 'p': mark last point as positive (foreground)
        - 'n': mark last point as negative (background)
        - space: finish and confirm segmentation
        - 'esc': cancel
        """
        if self.seg_init is True:
            print("SAM already initialized.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        points = []
        labels = []
        next_label = 1  # Default to positive
        mask = None

        # Get the initial frame
        rgb_frame = self.get_rgb_frame(0)

        # Load first frame into SAM predictor
        self.seg_predictor.load_first_frame(rgb_frame)

        # Setup matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        
        # Initial display
        im_display = ax.imshow(rgb_frame)
        ax.set_title('Interactive Segmentation - Click to add points')
        ax.axis('off')
        
        # Store plot elements for points
        point_plots = []
        
        # Status text
        status_text = fig.text(0.5, 0.05, '', ha='center', fontsize=12, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        print("\n=== Interactive Segmentation ===")
        print("Click on image to add points")
        print("Press 'p' to mark next point as POSITIVE (foreground)")
        print("Press 'n' to mark next point as NEGATIVE (background)")
        print("Press SPACE to finish")
        print("Press ESC to cancel")
        print("================================\n")

        def update_display():
            """Update the visualization with current mask and points"""
            vis_frame = rgb_frame.copy()
            
            # Overlay mask if available
            if mask is not None:
                mask_np = mask.cpu().numpy()
                # Create colored overlay (green for mask with transparency)
                overlay = vis_frame.copy().astype(np.float32)
                overlay[mask_np > 0] = overlay[mask_np > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                vis_frame = overlay.astype(np.uint8)
            
            im_display.set_data(vis_frame)
            
            # Clear old point markers
            for plot_elem in point_plots:
                plot_elem.remove()
            point_plots.clear()
            
            # Draw points
            for i, (pt, lbl) in enumerate(zip(points, labels)):
                color = 'lime' if lbl == 1 else 'red'
                edge_color = 'white'
                
                # Draw point with white border
                circle = plt.Circle((pt[0], pt[1]), 5, color=color, zorder=10)
                border = plt.Circle((pt[0], pt[1]), 7, color=edge_color, fill=False, linewidth=2, zorder=9)
                
                point_plots.append(ax.add_patch(circle))
                point_plots.append(ax.add_patch(border))
                
                # Add label number
                text = ax.text(pt[0] + 10, pt[1] - 10, str(i+1), 
                            color=color, fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                            zorder=11)
                point_plots.append(text)
            
            # Update status text
            status = f"Points: {len(points)} | Next: {'POSITIVE (green)' if next_label == 1 else 'NEGATIVE (red)'} | Press 'p'/'n' to toggle, SPACE to finish"
            status_text.set_text(status)
            
            fig.canvas.draw_idle()

        def on_click(event):
            """Handle mouse clicks"""
            if event.inaxes != ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            
            # Add point
            x, y = int(event.xdata), int(event.ydata)
            points.append([x, y])
            labels.append(next_label)
            
            # Update SAM with new points
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            
            nonlocal mask
            
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                # Reset predictor state and reload frame
                self.seg_predictor.reset_state()
                self.seg_predictor.load_first_frame(rgb_frame)
                
                # Get prediction with current points
                self.seg_predictor.add_new_prompt(
                    frame_idx=0, obj_id=1, points=points_array, labels=labels_array
                )
                # Track to get the mask
                _, mask_logits = self.seg_predictor.track(rgb_frame)
                mask = (mask_logits[0] > 0.0).squeeze()
            
            print(f"Added {'POSITIVE' if next_label == 1 else 'NEGATIVE'} point {len(points)} at ({x}, {y})")
            update_display()

        def on_key(event):
            """Handle key presses"""
            nonlocal next_label
            
            if event.key == 'p':
                next_label = 1
                print("Next point will be POSITIVE (foreground)")
                update_display()
            elif event.key == 'n':
                next_label = 0
                print("Next point will be NEGATIVE (background)")
                update_display()
            elif event.key == ' ':
                if len(points) == 0:
                    print("No points added. Please add at least one point.")
                else:
                    print(f"Segmentation confirmed with {len(points)} points")
                    plt.close(fig)
            elif event.key == 'escape':
                print("Segmentation cancelled")
                plt.close(fig)
                return

        # Connect event handlers
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Initial display update
        update_display()
        
        # Show the plot (blocks until closed)
        plt.show()
        
        # Check if user cancelled
        if len(points) == 0 or not plt.fignum_exists(fig.number):
            print("Segmentation not completed")
            return
        
        self.seg_init = True
        print("Segmentation complete!")

    def init_SAM_w_bbox(self, bbox):
        rgb_frame = self.get_rgb_frame(0)
        self.seg_predictor.load_first_frame(rgb_frame)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.seg_predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox
            )

        self.seg_init = True

    def init_SAM_w_mask(self, mask):
        rgb_frame = self.get_rgb_frame(0)
        self.seg_predictor.load_first_frame(rgb_frame)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.seg_predictor.add_new_mask(
                frame_idx=0, obj_id=1, mask=mask
            )
        self.seg_init = True

    def init_SAM_w_points_from_mask(self, mask):
        rgb_frame = self.get_rgb_frame(0)
        self.seg_predictor.load_first_frame(rgb_frame)

        x, y = np.where(mask)
        pos_coords = np.stack([y, x], axis=-1)
        # get 20 coords evenly spaced
        N_point_samples = 5
        pos_coords_subset = pos_coords[:: max(1, len(pos_coords) // N_point_samples)]
        pos_labels = np.array([1] * pos_coords_subset.shape[0], dtype=np.int32)

        # x, y = np.where(~mask)
        # neg_coords = np.stack([y, x], axis=-1)
        # # get 20 coords evenly spaced
        # neg_coords_subset = neg_coords[:: max(1, len(neg_coords) // 20)]
        # points = np.concatenate([pos_coords_subset, neg_coords_subset], axis=0)

        # neg_labels = np.array([0] * neg_coords_subset.shape[0], dtype=np.int32)
        # labels = np.concatenate([pos_labels, neg_labels], axis=0)

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.seg_predictor.add_new_prompt(
                # frame_idx=0, obj_id=1, points=points.astype(np.float32), labels=labels
                frame_idx=0, obj_id=1, points=pos_coords_subset.astype(np.float32), labels=pos_labels
            )

        self.seg_init = True

    def get_segmentation(self, frame):
        if self.seg_init is False:
            # self.init_segmentation_interactive()
            
            self.init_segmentation_interactive_plt()
            self.seg_init = True
            # self.init_segmentation()

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            out_obj_ids, out_mask_logits = self.seg_predictor.track(frame)

        mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0)

        return mask[..., 0].detach().clone()

    def get_frame(self, idx=None, run_seg=True, debug=False, output_queue=None):
        s_time = time.perf_counter()

        frame_data = {}

        rgb_frame_np = self.get_rgb_frame(idx)
        frame_data["rgb_np"] = rgb_frame_np

        if self.cfg["obj_mode"]:
            mask = self.get_segmentation(rgb_frame_np)
            mask_np = mask.cpu().numpy()

            # Optionally Expand the mask into a bbox with an offset
            if self.cfg.get("use_bbox", 0) > 0:
                offset = self.cfg["use_bbox"]
                min_x = np.min(np.where(mask_np)[1]) - offset
                max_x = np.max(np.where(mask_np)[1]) + offset
                min_y = np.min(np.where(mask_np)[0]) - offset
                max_y = np.max(np.where(mask_np)[0]) + offset

                min_x = max(0, min_x)
                min_y = max(0, min_y)
                max_x = min(rgb_frame_np.shape[1], max_x)
                max_y = min(rgb_frame_np.shape[0], max_y)
                mask_np[min_y:max_y, min_x:max_x] = True

            # frame_data["mask_np"] = mask_np
            # frame_data["rgb_masked_np"] = rgb_frame_np * mask_np[..., None]
            rgb_masked_np = rgb_frame_np * mask_np[..., None]

            if self.cfg.get("export_masked_imgs", False):
                export_dir = self.scene_dir / f"sam_segmented"
                os.makedirs(export_dir, exist_ok=True)
                file_uri = export_dir / f"{idx:05d}.png"
                cv2.imwrite(str(file_uri), frame_data["rgb_masked_np"][..., ::-1])

                export_dir = self.scene_dir / f"sam_masks"
                os.makedirs(export_dir, exist_ok=True)
                file_uri = export_dir / f"{idx:05d}.png"
                cv2.imwrite(str(file_uri), (mask_np.astype(np.uint8)) * 255)
        else:
            mask_np = np.ones_like(rgb_frame_np[:, :, 0], dtype=bool)
            # frame_data["mask_np"] = mask_np
            # frame_data["rgb_masked_np"] = rgb_frame_np * mask_np[..., None]
            rgb_masked_np = rgb_frame_np.copy()

        e_time = time.perf_counter()

        # print(f"Live data FPS: {1/(e_time-s_time):0.1f}", end='\r')
        if debug:
            bgr_frame = cv2.cvtColor(rgb_frame_np, cv2.COLOR_RGB2BGR)
            # add an X in the middle of the frame
            cv2.line(bgr_frame, (320, 240), (280, 200), (0, 0, 255), 2)
            cv2.line(bgr_frame, (320, 240), (360, 200), (0, 0, 255), 2)
            cv2.imshow("RGB Frame", bgr_frame)

            if run_seg:
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                mask_np = cv2.cvtColor(mask_np.squeeze(), cv2.COLOR_GRAY2RGB)
                cv2.imshow("Segmentation Mask", mask_np)

            cv2.waitKey(1)

        resize_dim = self.resize_dim
        frame_data["resized_rgb_masked_np"] = pi3_resize_image(rgb_masked_np.copy(), (resize_dim, resize_dim))
        frame_data["resized_mask_np"] = pi3_resize_image(mask_np.copy(), (resize_dim, resize_dim))
        frame_data["resized_mask"] = torch.tensor(frame_data["resized_mask_np"], device=self.device)

        frame_data["resized_rgb_masked"] = torch.tensor(frame_data["resized_rgb_masked_np"], device=self.device, dtype=torch.float32) / 255.0
        frame_data["resized_rgb_masked"] = frame_data["resized_rgb_masked"][None, None]

        if output_queue is not None:
            try:
                # output_queue.put(frame_data)
                output_queue.put_nowait(frame_data)
                return
            except:
                pass
    
        return frame_data
