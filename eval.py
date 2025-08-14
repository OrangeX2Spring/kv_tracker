import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from kv_tracker.eval_tools.evo_utils import align_pair

ARCTIC_SCENES = [
    "espressomachine_grab_01",
    "ketchup_grab_01",
    "microwave_grab_01",
    "box_grab_01",
    "laptop_grab_01",
    "waffleiron_grab_01",
    "scissors_grab_01",
    "capsulemachine_grab_01",
    "phone_grab_01",
    "mixer_grab_01",
]

SINTEL_SCENES = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4", "market_2", "market_5", "market_6",
    "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3",
]


def load_pairs_7scenes(dataset_dir, results_dir):
    from kv_tracker.dataloaders.scenes_7 import get_all_scenes_dir, scenes7Loader
    scenes_list = get_all_scenes_dir(dataset_dir)
    pairs = []
    for scene in scenes_list:
        traj_gt = scenes7Loader.load_gt(scene + "/seq-01")
        file_url = f"{scene}/{results_dir}/traj.npy"
        if not os.path.exists(file_url):
            print(f"Skipping {scene}, no results found")
            continue
        traj_est = np.load(file_url)
        # traj_est = np.linalg.inv(traj_est)
        assert traj_gt.shape == traj_est.shape, f"Shape mismatch: {traj_gt.shape} vs {traj_est.shape}"
        pairs.append({"traj_gt": traj_gt, "traj_est": traj_est, "name": Path(scene).name})
    return pairs


def load_pairs_tum(dataset_dir, results_dir):
    from kv_tracker.dataloaders.tum import get_all_scenes_dir, TUMLoader
    scenes_list = get_all_scenes_dir(dataset_dir)
    pairs = []
    for scene in scenes_list:
        traj_gt = TUMLoader.load_gt(scene)
        file_url = f"{scene}/{results_dir}/traj.npy"
        if not os.path.exists(file_url):
            print(f"Skipping {scene}, no results found")
            continue
        traj_est = np.load(file_url)
        assert traj_gt.shape == traj_est.shape, f"Shape mismatch: {traj_gt.shape} vs {traj_est.shape}"
        pairs.append({"traj_gt": traj_gt, "traj_est": traj_est, "name": Path(scene).name})
    return pairs


def load_pairs_sintel(dataset_dir, results_dir):
    from kv_tracker.dataloaders.sintel import SintelLoader
    pairs = []
    for scene in SINTEL_SCENES:
        scene_path = os.path.join(dataset_dir, scene)
        gt_path = scene_path.replace("final", "camdata_left")
        traj_gt = SintelLoader.load_gt(gt_path)
        traj_gt = np.linalg.inv(traj_gt)
        file_url = f"{scene_path}/{results_dir}/traj.npy"
        if not os.path.exists(file_url):
            print(f"Skipping {scene}, no results found")
            continue
        traj_est = np.load(file_url)
        assert traj_gt.shape == traj_est.shape, f"Shape mismatch: {traj_gt.shape} vs {traj_est.shape}"
        pairs.append({"traj_gt": traj_gt, "traj_est": traj_est, "name": scene})
    return pairs


def load_gt_arctic(sequence_name):
    from scipy.spatial.transform import Rotation as R
    obj_traj = np.load(
        f"datasets/arctic_data/data/raw_seqs/s01/{sequence_name}.object.npy",
        allow_pickle=True)
    obj_traj = obj_traj[:, 1:]  # remove articulation

    cam_traj = np.load(
        f"datasets/arctic_data/data/raw_seqs/s01/{sequence_name}.egocam.dist.npy",
        allow_pickle=True)

    batch_size = cam_traj.item()["T_k_cam_np"].shape[0]
    T_w2c = np.zeros((batch_size, 4, 4))
    T_w2c[:, 3, 3] = 1.0
    T_w2c[:, :3, 3] = cam_traj.item()["T_k_cam_np"].squeeze()
    T_w2c[:, :3, :3] = cam_traj.item()["R_k_cam_np"]

    T_w2obj = np.zeros((batch_size, 4, 4))
    T_w2obj[:, 3, 3] = 1.0
    T_w2obj[:, :3, 3] = obj_traj[:, 3:] * 1e-3
    T_w2obj[:, :3, :3] = R.from_rotvec(obj_traj[:, :3]).as_matrix()

    T_c2obj = np.linalg.inv(T_w2c) @ T_w2obj
    T_obj2c = np.linalg.inv(T_c2obj)
    return T_obj2c[2:]  # skip init frames


def load_pairs_arctic(dataset_dir, results_dir):
    pairs = []
    for scene in ARCTIC_SCENES:
        traj_gt = load_gt_arctic(scene)
        file_url = f"{dataset_dir}/{scene}/{results_dir}/traj.npy"
        if not os.path.exists(file_url):
            print(f"Skipping {scene}, no results found")
            continue
        traj_est = np.load(file_url)
        assert traj_gt.shape == traj_est.shape, f"Shape mismatch: {traj_gt.shape} vs {traj_est.shape}"
        pairs.append({"traj_gt": traj_gt, "traj_est": traj_est, "name": scene})
    return pairs


def load_pairs_onepose(dataset_dir, results_dir):
    from kv_tracker.dataloaders.onepose_dataset import get_all_scenes_dir, onePoseLoader
    scenes_list = get_all_scenes_dir(dataset_dir)
    pairs = []
    for scene in tqdm(scenes_list, desc="Loading"):
        gt_np = onePoseLoader.load_gt(Path(scene) / "poses_ba")
        gt_np = gt_np[2:]  # skip init frames
        gt_np = np.linalg.inv(gt_np)
        results_path = Path(scene) / results_dir
        if not results_path.is_dir() or not (results_path / "traj.npy").exists():
            print(f"Skipping {Path(scene).name}, no results found")
            continue
        kf_idx = np.load(results_path / "kf_idx.npy")
        pairs.append({
            "traj_gt": gt_np,
            "traj_est": np.load(results_path / "traj.npy"),
            "kf_gt": gt_np[kf_idx],
            "kf_est": np.load(results_path / "kf_poses.npy"),
            "name": Path(scene).name,
        })
    return pairs


def eval_evo(pairs, dataset_name):
    ate_list, rpe_t_list, rpe_rot_list = [], [], []
    for pair in tqdm(pairs, desc="Evaluating"):
        ate, rpe_t, rpe_rot = align_pair(pair, "traj", ret_np=False)
        ate_list.append(ate)
        rpe_t_list.append(rpe_t)
        rpe_rot_list.append(rpe_rot)

    print(f"\n=== {dataset_name} Results ({len(pairs)} scenes) ===")
    print(f"{'Scene':<30} {'ATE (m)':>10}")
    print("-" * 42)
    for pair, ate in zip(pairs, ate_list):
        print(f"{pair['name']:<30} {ate:>10.3f}")
    print("-" * 42)
    print(f"{'Mean ATE':<30} {np.mean(ate_list):>10.3f} m")
    print(f"{'Mean RPE_t':<30} {np.mean(rpe_t_list):>10.3f} m")
    print(f"{'Mean RPE_R':<30} {np.mean(rpe_rot_list):>10.3f}")
    print(f"\nCompact: ATE={np.mean(ate_list):.3f}  RPE_t={np.mean(rpe_t_list):.3f}  RPE_R={np.mean(rpe_rot_list):.3f}")


def eval_onepose(pairs):
    from kv_tracker.eval_tools.onepose_evaluator import BatchEvaluator
    evaluator = BatchEvaluator()
    for pair in tqdm(pairs, desc="Evaluating"):
        try:
            est, gt = align_pair(pair, "traj")
            if est is None:
                continue
        except Exception as e:
            print(e)
            continue
        evaluator.evaluate(poses_pred=est, poses_gt=gt)

    print(f"\n=== OnePose Results ({len(pairs)} scenes) ===")
    evaluator.summarize()


DATASETS = {
    "7scenes": {
        "loader": load_pairs_7scenes,
        "dir": "datasets/7-scenes",
        "eval": "evo",
    },
    "tum": {
        "loader": load_pairs_tum,
        "dir": "datasets/tum_rgbd",
        "eval": "evo",
    },
    "sintel": {
        "loader": load_pairs_sintel,
        "dir": "datasets/sintel/training/final",
        "eval": "evo",
    },
    "onepose": {
        "loader": load_pairs_onepose,
        "dir": "datasets/OnePose/test_data",
        "eval": "onepose",
    },
    "arctic": {
        "loader": load_pairs_arctic,
        "dir": "datasets/arctic_data/data/cropped_images_grab_only_subset/s01",
        "eval": "evo",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--results", required=True, help="Results subdirectory name inside each scene folder")
    parser.add_argument("--tless", default=False, action="store_true", help="Use OnePose low-texture split")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    dataset_dir = ds["dir"]
    if args.dataset == "onepose" and args.tless:
        dataset_dir = "datasets/OnePose/lowtexture_test_data"

    pairs = ds["loader"](dataset_dir, args.results)
    print(f"Loaded {len(pairs)} scenes with results")

    if ds["eval"] == "evo":
        eval_evo(pairs, args.dataset)
    else:
        eval_onepose(pairs)
