import os
import cv2
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.multiprocessing as mp
# import open3d as o3d

mp.set_start_method("spawn", force=True)

from tqdm import tqdm
from pathlib import Path
from time import perf_counter
from kv_tracker.rerun_tools import *
from kv_tracker.live_cap import liveData
from kv_tracker.geometry import umeyama_alignment
from kv_tracker.dataloaders.onepose_dataset import onePoseLoader
from kv_tracker.dataloaders.scenes_7 import scenes7Loader
from kv_tracker.dataloaders.tum import TUMLoader
from kv_tracker.dataloaders.phone import phoneLoader
from kv_tracker.dataloaders.sintel import SintelLoader
from kv_tracker.dataloaders.arctic_loader import arcticLoader
from kv_tracker.pi3_utilts import (
    load_pi3_from_pretrained,
    pi3_inference,
    move_pi3_mlps_to_bfloat32,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def capture_keyframes(frames_que, w_seg=True):
    keyframes = []
    print("Press 'c' to capture a keyframe, 'q' to quit.")
    while True:
        if frames_que.full():
            frame_data = frames_que.get()
        else:
            continue

        if not w_seg:
            frame_data["rgb_masked_np"] = frame_data["rgb_np"]
        rgb_masked_np = frame_data["rgb_masked_np"]

        # overlay N of frames captured
        cv2.putText(
            rgb_masked_np.copy(),
            f"Captured: {len(keyframes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
        cv2.imshow("Live", rgb_masked_np)

        key = cv2.waitKey(10) & 0xFF
        if key == ord(" "):
            keyframes.append(frame_data)
            print(f"Captured {len(keyframes)} keyframe(s)")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return keyframes


def visualise_pi3(
    namespace: str,
    pts3d: torch.tensor,
    T_wc: torch.tensor,
    rgb_np: np.array,
    masks_np: np.array,
    color=[241, 211, 2],
):

    if pts3d is not None and masks_np is not None:
        pts3d_np = pts3d.cpu().numpy()

        visable_pts3d = pts3d_np[masks_np]
        visable_colors = rgb_np[masks_np]
        rr.log(f"{namespace}/pts3d", rr.Points3D(visable_pts3d, colors=visable_colors))

    pred_T_wc_np = T_wc.cpu().numpy()
    for i in range(pred_T_wc_np.shape[0]):
        rr_viz_cam(
            f"{namespace}/cam_{i}",
            pred_T_wc_np[i],
            scale=0.1,
            fx=380.121,
            fy=380.065,
            w=640,
            h=480,
            color=color,
        )

def compute_elevation_azimuth(view_dir):
    z_axis = view_dir * -1 # T_wc[:, :3, 2] * -1  # [N, 3]

    z_axis_norm = torch.norm(z_axis, dim=1, keepdim=True)
    z_axis = z_axis / z_axis_norm

    # Azimuth
    azimuth_rad = torch.asin(torch.clamp(z_axis[:, 0] / z_axis[:, 2], -1.0, 1.0))
    azimuth = torch.rad2deg(azimuth_rad)

    # Elevation
    elevation = torch.rad2deg(torch.atan2(z_axis[:, 1], z_axis[:, 2]))

    return elevation, azimuth

def check_if_keyframe(obj_center, cur_T_wc, keyframes_T_wc, elevation_threshold=10.0, azimuth_threshold=10.0):

    # Compute angles for current frame
    cur_view_dir = obj_center - cur_T_wc[:3, 3]  # [3]
    # cur_view_dir = cur_T_wc[:3, 2]  # [3]

    cur_elevation, cur_azimuth = compute_elevation_azimuth(cur_view_dir[None])

    # Compute angles for all keyframes
    kf_view_dirs = obj_center - keyframes_T_wc[:, :3, 3]  # [N, 3]
    # kf_view_dirs = keyframes_T_wc[:, :3, 2]  # [N, 3]
    kf_elevations, kf_azimuths = compute_elevation_azimuth(kf_view_dirs)

    # Compute angular differences
    elevation_diffs = torch.abs(cur_elevation - kf_elevations)

    # Azimuth difference handling wrap-around at 180/-180 degrees
    azimuth_diffs = torch.abs(cur_azimuth - kf_azimuths)
    azimuth_diffs = torch.min(azimuth_diffs, 360.0 - azimuth_diffs)

    # Check if minimum differences exceed thresholds
    min_elevation_diff = elevation_diffs.min()
    min_azimuth_diff = azimuth_diffs.min()

    # Trigger keyframe if either angle exceeds threshold
    if min_elevation_diff > elevation_threshold or min_azimuth_diff > azimuth_threshold:
        return True

    return False

def data_stream_proc(frames_que, device, **kwargs):
    datasource = kwargs.get("datasource", "liveData")
    live_data = eval(datasource)(device, **kwargs)

    for i in range(live_data.length):
        frame_data = live_data.get_frame(i)
        frame_data["idx"] = i
        frame_data["gt_pose"] = live_data.get_gt_pose(i)

        frames_que.put(frame_data)

    live_data.close_cap()
    print("Data stream process is done.")

def dump_data(data, dir):
    db_file = open(dir, 'wb')
    pickle.dump(data, db_file)
    db_file.close()

    print(f"saved pickle: {dir}")


def follower_cam(cur_T_wc, offset=np.array([0.0, 0.0, 0.5])):
    """
    Create a follower camera with offset in the current camera's local frame.

    Args:
        cur_T_wc: Current camera pose (4x4 transformation matrix)
        offset: Offset in camera's local frame [x, y, z]
                e.g., [0, 0, 0.5] means 0.5m behind the camera
    """
    follower_T_wc = cur_T_wc.copy()

    # Apply offset in the camera's local coordinate system
    # Transform offset from camera frame to world frame
    offset_world = cur_T_wc[:3, :3] @ offset
    follower_T_wc[:3, 3] = cur_T_wc[:3, 3] + offset_world

    rr_viz_cam(
        f"follower_cam",
        follower_T_wc,
        scale=0.0,
        fx=380.121,
        fy=380.065,
        w=640,
        h=480,
        color=[0, 255, 0],
    )


def run_track3r(cfg = None, args = None, frame_source=None, snapshot_callback=None,
               keyframe_indices=None, keyframe_selector=None, keyframe_cache=None):

    assert keyframe_indices is None or keyframe_selector is None

    device = "cuda:0"
    tracking_frame_type = "resized_rgb_masked" # "rgb_crop", "resized_rgb_masked"

    parser = argparse.ArgumentParser()
    if cfg is None:
        parser.add_argument('config', default='config/live.yaml')
    parser.add_argument('--obj_mode', default=False, action='store_true')
    parser.add_argument('--cam_only', default=False, action='store_true')
    parser.add_argument('--manual_kf', default=False, action='store_true')
    parser.add_argument('--dump_data', default=False, action='store_true')
    parser.add_argument('--rerun', default=False, action='store_true')
    parser.add_argument('--crop_kf', default=False, action='store_true')
    parser.add_argument('--mesh', default=False, action='store_true')
    parser.add_argument('--export_pcd', default=False, action='store_true')
    parser.add_argument('--resize_dim', default=518) # 308, 518, 630
    parser.add_argument('--kf_auto', default=50)
    parser.add_argument('--sim3', default=False, action='store_true')
    args = parser.parse_args(args)

    if keyframe_cache is not None:
        # The evaluated scene protocol uses first-frame gauge, not optional Sim(3).
        assert keyframe_selector is not None and frame_source is not None
        assert not args.sim3 and (args.cam_only or args.obj_mode)
        assert not args.obj_mode or keyframe_cache.supports_object_mode
        assert not args.manual_kf and not args.crop_kf

    if cfg is None:
        cfg = yaml.safe_load(open(args.config, 'r'))

    resize_dim = int(args.resize_dim)

    if args.obj_mode:
        cfg["obj_mode"] = True
    else:
        cfg["obj_mode"] = False

    if args.crop_kf:
        mapping_frame_type = "rgb_crop_np"
        mapping_frame_mask_type = "resized_mask_crop_np"
    else:
        mapping_frame_type = "resized_rgb_masked_np"
        mapping_frame_mask_type = "resized_mask_np"

    if args.rerun:
        rr.init("KV-Track3r goes brrrr", spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("fps_plot", rr.SeriesLines(colors=[0, 255, 0], widths=[2.5], names="FPS"), static=True)

    if "results_path" in cfg:
        results_path = Path(cfg["results_path"])
    else:
        results_path = Path("debug_dumps")
    os.makedirs(results_path, exist_ok=True)

    print(f"Using resize dim: {resize_dim}")
    print(f"Saving results to {results_path}")

    # ----------------------
    # Start Data Stream Process
    # ----------------------
    if frame_source is None:
        frames_que = mp.Queue(maxsize=cfg["que_size"])
        data_proc = mp.Process(
            target=data_stream_proc,
            args=(frames_que, device),
            kwargs={"resize_dim": resize_dim, **cfg}
        )
        data_proc.start()
    else:
        assert not args.manual_kf
        frame_source = iter(frame_source)

    model = load_pi3_from_pretrained(device).eval()
    model = move_pi3_mlps_to_bfloat32(model)
    if keyframe_selector is not None:
        keyframe_selector.attach(model)
    if keyframe_cache is not None:
        keyframe_cache.attach(model)

    # Get Keyframes
    # =============
    if args.manual_kf:
        print("Manual Keyframe Capture Mode")
        keyframes = capture_keyframes(frames_que)
    else:
        keyframes = []
        while len(keyframes) < 1:
            if frame_source is None:
                if frames_que.empty():
                    continue
                frame_data = frames_que.get()
            else:
                frame_data = next(frame_source)

            keyframes.append(frame_data)
            keyframes.append(frame_data)

    if len(keyframes) == 0:
        print("No keyframes captured, exiting.")
        exit(0)

    # ----------------------
    # Inference on Keyframes
    # ----------------------
    kf_rgb_np_list = []
    kf_masks = []
    for kf in keyframes:
        kf_rgb_np_list.append(kf[mapping_frame_type])
        kf_masks.append(kf[mapping_frame_mask_type])

    kf_rgb_np = np.stack(kf_rgb_np_list, axis=0)  # [N_keyframes, H, W, 3]
    kf_masks_np = np.stack(kf_masks, axis=0)  # [N_keyframes, H, W]
    capture_frame_ids = [kf["idx"] for kf in keyframes]

    print(f"Frames shapes: {kf_rgb_np_list[0].shape}")

    if args.rerun:
        kf_rgb_viz_np = kf_rgb_np.transpose(1, 0, 2, 3).reshape(kf_rgb_np.shape[1], -1, 3)
        rr.log("keyframes/images", rr.Image(kf_rgb_viz_np))

    batch_pts3d, batch_pred_T_wc, batch_conf, batch_images_np, local_pts3d, origin_offset = pi3_inference(
        model, [kf_rgb_np], device, cam_only=False, store_cache=True, tokens_mask=None
    )
    if keyframe_cache is not None:
        keyframe_cache.after_rebuild(capture_frame_ids, batch_conf,
                                     points=batch_pts3d, masks=kf_masks_np)
    if keyframe_selector is not None:
        keyframe_selector.bootstrap(keyframes[0])

    kf_masks = torch.tensor(kf_masks_np, device=device).bool()
    obj_center = batch_pts3d[0][kf_masks].mean(dim=0)

    scene_origin = obj_center.clone()
    batch_pts3d[..., :] -= scene_origin
    batch_pred_T_wc[..., :3, 3] -= scene_origin

    if keyframe_selector is not None:
        selector_obj_center = batch_pts3d[0][kf_masks].mean(dim=0)

    # offset the scene to be centered around the object

    initial_point_conf = batch_conf.clone() if snapshot_callback is not None else None
    batch_conf = batch_conf.squeeze()
    batch_conf[~kf_masks] = 0.0
    batch_conf = batch_conf.sum(dim=(-2, -1)) / kf_masks.sum(dim=(-2, -1))

    kf_conf_thresh = batch_conf[0] * 0.6
    pts3d_conf_thresh = kf_conf_thresh * 1.15
    if snapshot_callback is not None:
        snapshot_callback("keyframes", capture_frame_ids, batch_pts3d,
                          batch_pred_T_wc, initial_point_conf, kf_rgb_np,
                          kf_masks_np, pts3d_conf_thresh, keyframes[-1]["rgb_np"])
        del initial_point_conf

    if args.rerun:
        visualise_pi3(
            "keyframes",
            batch_pts3d.squeeze(),
            batch_pred_T_wc.squeeze(),
            batch_images_np.squeeze(),
            kf_masks_np,
            color=[255, 0, 0],
            # (batch_conf.squeeze() > pts3d_conf_thresh).cpu().numpy(),
        )

    # Canonical keyframe positions will be initialized after kf_init
    # (when we have 2 real keyframes from the same reconstruction)
    canonical_kf_positions = None

    # Initialize Sim(3) as identity (no transformation initially)
    current_sim3_R = torch.eye(3, device=device, dtype=torch.float64)
    current_sim3_t = torch.zeros(3, device=device, dtype=torch.float64)
    current_sim3_s = 1.0

    print("------------------")

    # -------------
    # Tracking Loop
    # -------------
    fps_rolling = []
    kf_init = False
    idx = 2

    pcd_list = [batch_pts3d[0, 0].cpu().numpy()]
    poses_np_list = [batch_pred_T_wc[0, 0].cpu().numpy()]
    good_idx = [0]
    kf_idx = [0]
    kf_poses = [batch_pred_T_wc[0, 0].cpu().numpy()]
    assert len(poses_np_list) == len(good_idx)

    prev_pred = torch.linalg.inv(origin_offset).clone()
    while frame_source is not None or data_proc.is_alive():
        start_time = perf_counter()

        # Get next frame if available
        if frame_source is None:
            if frames_que.empty():
                continue
            current_frame = frames_que.get_nowait()
        else:
            current_frame = next(frame_source, None)
            if current_frame is None:
                break


        segmentation_time = perf_counter()

        # if args.rerun:
        #     rr.log("latest_rgb", rr.Image(current_frame["rgb_np"]))
        #     rr.log("latest_rgb_cropped", rr.Image(current_frame[tracking_frame_type]))

        pi3_start_time = perf_counter()

        # ----------------------
        # Latest Frame Inference
        # ----------------------
        if keyframe_cache is not None:
            keyframe_cache.begin_query(current_frame['idx'])
        inference_ret = pi3_inference(
            model,
            current_frame[tracking_frame_type].clone(),
            device,
            cam_only=args.cam_only,
            store_cache=False,
            use_cache=True,
        ) 

        if args.cam_only:
            pred_T_wc = inference_ret
        else:
            pred_pts3d , pred_T_wc, pred_conf, pred_images_np, _, _ = inference_ret

            conf = pred_conf.squeeze()[current_frame["resized_mask"]].mean()
            if conf < kf_conf_thresh * 0.3: # 0.4
                print(f"low conf detected: {conf.item():.2f}, limit is {kf_conf_thresh * 0.4:.2f}")
                pred_T_wc[0, 0] = prev_pred
            else:
                prev_pred = pred_T_wc[0, 0].clone()

        pi3_inference_time = perf_counter()

        # Transform to original coord system
        with torch.autocast("cuda", dtype=torch.float64):
            pred_T_wc = origin_offset @ pred_T_wc
            if not args.cam_only:
                pred_pts3d = origin_offset[:3, :3] @ pred_pts3d[..., None] + origin_offset[:3, 3:4]
                pred_pts3d = pred_pts3d.squeeze(-1)

        pred_T_wc[..., :3, 3] -= scene_origin

        # Apply Sim(3) alignment to match canonical coordinate system
        pred_T_wc = pred_T_wc.to(torch.float64)
        pred_T_wc[..., :3, 3] = current_sim3_s * (pred_T_wc[..., :3, 3] @ current_sim3_R.T) + current_sim3_t
        pred_T_wc[..., :3, :3] = current_sim3_R @ pred_T_wc[..., :3, :3]
        pred_T_wc = pred_T_wc.to(torch.float32)

        if not args.cam_only:
            pred_pts3d[..., :] -= scene_origin
            pred_pts3d = pred_pts3d.to(torch.float64)
            pred_pts3d = current_sim3_s * (pred_pts3d @ current_sim3_R.T) + current_sim3_t
            pred_pts3d = pred_pts3d.to(torch.float32)

        T_w2c = pred_T_wc.squeeze().cpu().numpy()
        if snapshot_callback is not None and not args.cam_only:
            snapshot_callback("queries", [current_frame["idx"]], pred_pts3d,
                              pred_T_wc, pred_conf,
                              current_frame["resized_rgb_masked_np"][None],
                              current_frame["resized_mask_np"][None],
                              pts3d_conf_thresh, current_frame["rgb_np"])

        # Export pose estimates
        poses_np_list.append(T_w2c)
        with open(results_path / "traj.npy", 'wb') as f:
            np.save(f, np.array(poses_np_list))

        if idx % 40 == 0 and not args.cam_only:
            pcd_list.append(pred_pts3d.squeeze().cpu().numpy())
            with open(results_path / "pcd.npy", 'wb') as f:
                np.save(f, np.array(pcd_list))

            visualise_pi3(
                f"frames/idx_{idx}",
                pred_pts3d[0],
                pred_T_wc[0],
                current_frame["resized_rgb_masked_np"][None],
                (pred_conf[0,..., 0] > pts3d_conf_thresh).cpu().numpy(),
                color=[255, 0, 0],
            )
        
        # -----------------
        # Keyframe Decision
        # -----------------
        should_add_kf = False
        if args.obj_mode:
            # mask_np = current_frame["resized_mask_np"]
            # fill_ratio = mask_np.sum() / (mask_np.shape[0] * mask_np.shape[1])
            # if fill_ratio > 0.1:
            #     should_add_kf = check_if_keyframe(obj_center, pred_T_wc.squeeze(), batch_pred_T_wc.squeeze())
            should_add_kf = check_if_keyframe(obj_center, pred_T_wc.squeeze(), batch_pred_T_wc.squeeze())
        else:
            should_add_kf = (int(args.kf_auto) > 0 and (idx % int(args.kf_auto) == 0))
            should_add_kf = should_add_kf and (kf_rgb_np.shape[0] < 20)

        if keyframe_indices is not None:
            should_add_kf = current_frame["idx"] in keyframe_indices
        if keyframe_selector is not None:
            should_add_kf = keyframe_selector.select(
                current_frame, selector_obj_center, pred_T_wc[0, 0],
                batch_pred_T_wc[0], capture_frame_ids, bool(should_add_kf))

        if should_add_kf:

            if not kf_init and kf_rgb_np.shape[0] == 2:
                kf_init = True
                kf_rgb_np = kf_rgb_np[0][None]
                kf_masks_np = kf_masks_np[0][None]
                capture_frame_ids = capture_frame_ids[:1]

            if keyframe_cache is not None:
                keep = [i for i, frame_id in enumerate(capture_frame_ids)
                        if frame_id in keyframe_cache.keep_frame_ids]
                kf_rgb_np = kf_rgb_np[keep]
                kf_masks_np = kf_masks_np[keep]
                capture_frame_ids = [capture_frame_ids[i] for i in keep]

            kf_rgb_np = np.concatenate([kf_rgb_np, current_frame[mapping_frame_type][None]], axis=0)  # [N_keyframes, H, W]
            kf_masks_np = np.concatenate([kf_masks_np, current_frame[mapping_frame_mask_type][None]], axis=0)  # [N_keyframes, H, W]
            capture_frame_ids.append(current_frame["idx"])

            batch_pts3d , batch_pred_T_wc, batch_conf, batch_images_np, local_pts3d, origin_offset = pi3_inference(
                model, [kf_rgb_np], device, cam_only=False, store_cache=True
            )
            if keyframe_cache is not None:
                keyframe_cache.after_rebuild(capture_frame_ids, batch_conf,
                                             points=batch_pts3d, masks=kf_masks_np)

            kf_masks = torch.tensor(kf_masks_np, device=device).bool()
            obj_center = batch_pts3d[0][kf_masks].mean(dim=0)

            batch_pts3d[..., :] -= scene_origin
            batch_pred_T_wc[..., :3, 3] -= scene_origin

            # -----------------
            # Sim(3) Alignment
            # -----------------
            if args.sim3:
                # Initialize canonical from first real 2-keyframe reconstruction (after kf_init)
                if canonical_kf_positions is None:
                    # First real reconstruction with 2 distinct keyframes
                    # Store ALL positions from this reconstruction as the canonical reference
                    canonical_kf_positions = batch_pred_T_wc[0, :, :3, 3].clone().to(torch.float64)
                    print(f"Initialized canonical with {canonical_kf_positions.shape[0]} keyframes")
                else:
                    # Align new reconstruction to canonical coordinate system
                    n_canonical = canonical_kf_positions.shape[0]
                    n_new = batch_pred_T_wc.shape[1]

                    if n_canonical == 2 and n_new > n_canonical:
                        # Scale-only alignment with 2 points (can't do full Sim(3))
                        # Compute scale from distance ratio between first two keyframes
                        dist_canonical = torch.norm(
                            canonical_kf_positions[1] - canonical_kf_positions[0]
                        ).to(torch.float64)
                        dist_new = torch.norm(
                            batch_pred_T_wc[0, 1, :3, 3] - batch_pred_T_wc[0, 0, :3, 3]
                        ).to(torch.float64)

                        s_align = (dist_canonical / dist_new).item()

                        # For scale-only: R = I, t = translation to align first keyframe
                        R_align = torch.eye(3, device=device, dtype=torch.float64)

                        # Scale first, then compute translation to align first keyframe
                        batch_pred_T_wc = batch_pred_T_wc.to(torch.float64)
                        batch_pred_T_wc[..., :3, 3] = s_align * batch_pred_T_wc[..., :3, 3]

                        # Translation to align first keyframe position
                        t_align = canonical_kf_positions[0] - batch_pred_T_wc[0, 0, :3, 3]
                        batch_pred_T_wc[..., :3, 3] = batch_pred_T_wc[..., :3, 3] + t_align
                        batch_pred_T_wc = batch_pred_T_wc.to(torch.float32)

                        # Apply to 3D points
                        batch_pts3d = batch_pts3d.to(torch.float64)
                        batch_pts3d = s_align * batch_pts3d + t_align
                        batch_pts3d = batch_pts3d.to(torch.float32)

                        # Update obj_center after alignment
                        obj_center = batch_pts3d[0][kf_masks].mean(dim=0)

                        # Store for live predictions
                        current_sim3_R = R_align
                        current_sim3_t = t_align
                        current_sim3_s = s_align

                        print(f"Scale-only alignment: scale={s_align:.4f}")

                    elif n_canonical >= 3 and n_new > n_canonical:
                        # Full Sim(3) alignment (need ≥3 non-collinear points)
                        # Use positions of existing keyframes (exclude newly added one)
                        new_positions = batch_pred_T_wc[0, :n_canonical, :3, 3].to(torch.float64)  # [n_canonical, 3]

                        # Compute Sim(3): canonical ≈ s * R @ new + t
                        R_align, t_align, s_align = umeyama_alignment(
                            new_positions.cpu().numpy().T,      # [3, n_canonical] source
                            canonical_kf_positions.cpu().numpy().T,  # [3, n_canonical] target
                            with_scale=True
                        )

                        R_align = torch.tensor(R_align, device=device, dtype=torch.float64)
                        t_align = torch.tensor(t_align, device=device, dtype=torch.float64)

                        # Apply Sim(3) to keyframe poses
                        batch_pred_T_wc = batch_pred_T_wc.to(torch.float64)
                        batch_pred_T_wc[..., :3, 3] = s_align * (batch_pred_T_wc[..., :3, 3] @ R_align.T) + t_align
                        batch_pred_T_wc[..., :3, :3] = R_align @ batch_pred_T_wc[..., :3, :3]
                        batch_pred_T_wc = batch_pred_T_wc.to(torch.float32)

                        # Apply Sim(3) to 3D points
                        batch_pts3d = batch_pts3d.to(torch.float64)
                        batch_pts3d = s_align * (batch_pts3d @ R_align.T) + t_align
                        batch_pts3d = batch_pts3d.to(torch.float32)

                        # Update obj_center after alignment
                        obj_center = batch_pts3d[0][kf_masks].mean(dim=0)

                        # Store Sim(3) for live predictions
                        current_sim3_R = R_align
                        current_sim3_t = t_align
                        current_sim3_s = s_align

                        # print(f"Sim(3) alignment: scale={s_align:.4f}")

                    # Add new keyframe position to canonical (after alignment if it happened)
                    canonical_kf_positions = torch.cat([
                        canonical_kf_positions,
                        batch_pred_T_wc[0, -1:, :3, 3].to(torch.float64)
                    ], dim=0)

            if args.rerun:
                # show all keyframes
                kf_rgb_viz_np = kf_rgb_np.transpose(1, 0, 2, 3).reshape(kf_rgb_np.shape[1], -1, 3)
                rr.log("keyframes/images", rr.Image(kf_rgb_viz_np))

                # show obj center
                temp = np.eye(4)
                temp[:3, 3] = obj_center.cpu().numpy()
                rr_viz_pose(f"obj_center", temp)

                pixels_to_viz = kf_masks_np.copy()
                pixels_to_viz = pixels_to_viz & (batch_conf.squeeze() > pts3d_conf_thresh).cpu().numpy()

                visualise_pi3(
                    "keyframes",
                    batch_pts3d.squeeze(),
                    batch_pred_T_wc.squeeze(),
                    batch_images_np.squeeze(),
                    pixels_to_viz,
                    color=[255, 0, 0],
                )

            if keyframe_selector is not None:
                selector_obj_center = batch_pts3d[0][kf_masks].mean(dim=0)
            if snapshot_callback is not None:
                snapshot_callback("keyframes", capture_frame_ids, batch_pts3d,
                                  batch_pred_T_wc, batch_conf, kf_rgb_np,
                                  kf_masks_np, pts3d_conf_thresh, current_frame["rgb_np"])

            batch_conf = batch_conf.squeeze()
            batch_conf[~kf_masks] = 0.0
            batch_conf = batch_conf.sum(dim=(-2, -1)) / kf_masks.sum(dim=(-2, -1))

            # while in obj_mode check if frame is actually a keyframe 
            # since a keyframe addition might have been triggered
            # by a rogue live prediction
            revert_kf = False # no reverts in scene mode for now
            is_kf = True
            if args.obj_mode:
                is_kf = check_if_keyframe(
                    obj_center,
                    batch_pred_T_wc.squeeze()[-1],
                    batch_pred_T_wc.squeeze()[:-1],
                )

                # Fall back if the last keyframe is of low confidence
                revert_kf = (batch_conf[-1] < kf_conf_thresh) or (not is_kf)
                revert_kf = revert_kf and (kf_rgb_np.shape[0] > 2)
                # revert_condition = revert_condition

            revert_kf = False
            if revert_kf:
                kf_rgb_np = kf_rgb_np[:-1]
                kf_masks_np = kf_masks_np[:-1]

                # Remove the reverted keyframe from canonical positions
                if canonical_kf_positions is not None:
                    canonical_kf_positions = canonical_kf_positions[:-1]

                # Reset kv cache
                # model.kv_cache = last_kv
                batch_pts3d , batch_pred_T_wc, batch_conf, batch_images_np, local_pts3d, origin_offset = pi3_inference(
                    model, [kf_rgb_np], device, cam_only=False, store_cache=True
                )

                kf_masks = torch.tensor(kf_masks_np, device=device).bool()
                obj_center = batch_pts3d[0][kf_masks].mean(dim=0)

                batch_pts3d[..., :] -= scene_origin
                batch_pred_T_wc[..., :3, 3] -= scene_origin

                # Recompute Sim(3) alignment after revert
                if args.sim3 and canonical_kf_positions is not None:
                    n_canonical = canonical_kf_positions.shape[0]
                    n_new = batch_pred_T_wc.shape[1]

                    if n_canonical >= 3 and n_new == n_canonical:
                        new_positions = batch_pred_T_wc[0, :, :3, 3].to(torch.float64)

                        R_align, t_align, s_align = umeyama_alignment(
                            new_positions.cpu().numpy().T,
                            canonical_kf_positions.cpu().numpy().T,
                            with_scale=True
                        )

                        R_align = torch.tensor(R_align, device=device, dtype=torch.float64)
                        t_align = torch.tensor(t_align, device=device, dtype=torch.float64)

                        batch_pred_T_wc = batch_pred_T_wc.to(torch.float64)
                        batch_pred_T_wc[..., :3, 3] = s_align * (batch_pred_T_wc[..., :3, 3] @ R_align.T) + t_align
                        batch_pred_T_wc[..., :3, :3] = R_align @ batch_pred_T_wc[..., :3, :3]
                        batch_pred_T_wc = batch_pred_T_wc.to(torch.float32)

                        batch_pts3d = batch_pts3d.to(torch.float64)
                        batch_pts3d = s_align * (batch_pts3d @ R_align.T) + t_align
                        batch_pts3d = batch_pts3d.to(torch.float32)

                        obj_center = batch_pts3d[0][kf_masks].mean(dim=0)

                        current_sim3_R = R_align
                        current_sim3_t = t_align
                        current_sim3_s = s_align

                # Reset object center
                # obj_center = last_obj_center.clone()

                if args.rerun:
                    # Show previous keyframes
                    kf_rgb_viz_np = kf_rgb_np.transpose(1, 0, 2, 3).reshape(kf_rgb_np.shape[1], -1, 3)
                    rr.log("keyframes/images", rr.Image(kf_rgb_viz_np))

                    pixels_to_viz = kf_masks_np.copy()
                    pixels_to_viz = pixels_to_viz & (batch_conf.squeeze() > pts3d_conf_thresh).cpu().numpy()

                    # Show previous pts3d
                    visualise_pi3(
                        "keyframes",
                        batch_pts3d.squeeze(),
                        batch_pred_T_wc.squeeze(),
                        batch_images_np.squeeze(),
                        kf_masks_np,
                        color=[255, 0, 0],
                        # (batch_conf.squeeze() > pts3d_conf_thresh).cpu().numpy(),
                    )

            else:
                # last_kv = model.cache.copy()
                # last_obj_center = obj_center.clone()
                # last_batch_pts3d = batch_pts3d.clone()
                # last_batch_pred_T_wc = batch_pred_T_wc.clone()
                # last_batch_images_np = batch_images_np.copy()
                
                kf_idx.append(current_frame["idx"])
                kf_poses = batch_pred_T_wc.squeeze().cpu().numpy()

                with open(results_path / "kf_poses.npy", 'wb') as f:
                    np.save(f, np.array(kf_poses))
                
                with open(results_path / "kf_idx.npy", 'wb') as f:
                    np.save(f, np.array(capture_frame_ids if keyframe_cache is not None else kf_idx))
                if keyframe_cache is not None:
                    np.save(results_path / 'inserted_kf_idx.npy', np.array(kf_idx))
                
                if args.export_pcd:
                    pcd_path = results_path / f"pcd_{idx}.ply"
                    pcd = o3d.geometry.PointCloud()

                    # import pdb; pdb.set_trace()
                    pts3d_all = batch_pts3d.squeeze().cpu().numpy()
                    pts3d_all = pts3d_all[pixels_to_viz]
                    pcd.points = o3d.utility.Vector3dVector(pts3d_all)
                    colors_all = kf_rgb_np.squeeze().copy()
                    colors_all = colors_all[pixels_to_viz]
                    pcd.colors = o3d.utility.Vector3dVector(colors_all/ 255.0)
                    o3d.io.write_point_cloud(str(pcd_path), pcd)
                    
                    # alpha blending rgb frames with mask show a bit of background
                    pixels_to_viz = kf_masks_np.copy()
                    blended_rgb = kf_rgb_np.squeeze().copy()
                    blended_rgb[~pixels_to_viz] = (blended_rgb[~pixels_to_viz] * 0.5).astype(np.uint8)
                    # save blended rgb
                    for i in range(blended_rgb.shape[0]):
                        cv2.imwrite(str(results_path / f"kf_{i:03d}.png"), blended_rgb[i, :, :, ::-1])


        # -----------------
        # Runtime stats
        # -----------------
        end_time_with_viz = perf_counter()
        # fps_rolling.append(1.0 / (end_time_with_viz - start_time))
        fps_rolling.append(1.0 / (pi3_inference_time - start_time))
        if len(fps_rolling) > 30:
            fps_rolling.pop(0)
        
        if args.rerun:
            rr.log("latest_rgb", rr.Image(current_frame["rgb_np"]))
            follower_cam(T_w2c, offset=np.array([0.0, 0.0, -0.3]))

            # Visualize trajectory as a line
            if len(poses_np_list) > 1:
                trajectory_points = np.array([pose[:3, 3] for pose in poses_np_list])
                rr.log("trajectory/path", rr.LineStrips3D(trajectory_points, colors=[0, 127, 255]))

            rr_viz_cam(
                f"current_frame/T_wc",
                T_w2c,
                scale=0.1,
                fx=380.121,
                fy=380.065,
                w=640,
                h=480,
                color=[255, 255, 0]
                )

            if current_frame['idx'] % 4 == 0:
                rr_viz_cam(
                    f"all_frames/T_wc_{current_frame['idx']:06d}",
                    T_w2c,
                    scale=0.1,
                    fx=380.121,
                    fy=380.065,
                    w=640,
                    h=480,
                    color=[0, 127, 255]
                    )

            if idx>10:
                rr.set_time("FPS", sequence=idx)
                rr.log("fps_plot", rr.Scalars(np.mean(fps_rolling)))
                rr.log("fps_text", rr.TextDocument(f"""
                                                   
# FPS: {np.mean(fps_rolling):.1f}
# Keyframes: {kf_rgb_np.shape[0]}
""", media_type=rr.MediaType.MARKDOWN))

        del current_frame

        metrics = (
            f"Seg: {segmentation_time-start_time:.03f}s | "
            f"Resize: {pi3_start_time-segmentation_time:.03f}s | "
            f"Pi3: {pi3_inference_time-pi3_start_time:.03f}s | "
            f"Viz: {end_time_with_viz-pi3_inference_time:.03f}s | "
            f"Total: {end_time_with_viz-start_time:.03f}s | "
            f"FPS: {np.mean(fps_rolling):.1f} | "
            f"N: {kf_rgb_np.shape[0]} "
        )
        print(metrics, end="\r")
        idx +=1


if __name__ == "__main__":
    run_track3r()
