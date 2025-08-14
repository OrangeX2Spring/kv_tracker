import os
import copy
import pickle
import rerun as rr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt

from kv_tracker.eval_tools.onepose_evaluator import Evaluator, BatchEvaluator
from kv_tracker.rerun_tools import rr_viz_pose

from evo.tools import plot
from evo.core import lie_algebra as lie
from evo.core import sync, metrics
from evo.core.units import Unit
from evo.core.trajectory import PoseTrajectory3D


def align_pair(pair, prefix="traj", ret_np=True, show_plot=False):
    test_gt = pair[f"{prefix}_gt"]
    test_est = pair[f"{prefix}_est"]

    # test_est = np.linalg.inv(test_est)

    # test_gt = np.linalg.inv(test_gt)  # Needed?

    if test_gt.shape != test_est.shape:
        print(
            f"Pair not equal in length, gt: {test_gt.shape[0]}, est: {test_est.shape[0]}"
        )
        return None, None

    N_poses = test_gt.shape[0]
    time_stamps = np.arange(N_poses).astype(np.float32) * 0.1

    traj_ref = PoseTrajectory3D(
        poses_se3=test_gt,
        timestamps=time_stamps,
    )

    traj_est = PoseTrajectory3D(poses_se3=test_est, timestamps=time_stamps)

    max_diff = 0.01

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

    traj_est_aligned = copy.deepcopy(traj_est)
    R, t, s = traj_est_aligned.align(
        traj_ref, correct_scale=True, correct_only_scale=False
    )

    if show_plot:
        fig = plt.figure()
        traj_by_label = {
            # "estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref
        }

        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()

    if ret_np:
        return np.array(traj_est_aligned.poses_se3), test_gt

    tum_ate = metrics.APE(metrics.PoseRelation.translation_part)
    tum_ate.process_data((traj_ref, traj_est_aligned))
    ate_rmse = tum_ate.get_statistic(metrics.StatisticsType.rmse)

    delta = 1
    rpe_trans = metrics.RPE(
        metrics.PoseRelation.translation_part, delta, Unit.frames, all_pairs=True
    )
    rpe_trans.process_data((traj_ref, traj_est_aligned))
    rpe_trans_rmse = rpe_trans.get_statistic(metrics.StatisticsType.rmse)

    rep_rot = metrics.RPE(
        metrics.PoseRelation.rotation_part, delta, Unit.frames, all_pairs=True
    )
    rep_rot.process_data((traj_ref, traj_est_aligned))
    rpe_rot_rmse = rep_rot.get_statistic(metrics.StatisticsType.rmse)

    return ate_rmse, rpe_trans_rmse, rpe_rot_rmse

    # print(f"ATE RMSE: {ate_rmse:.4f} m")
    # print(f"RPE RMSE: {rpe_rmse:.4f} m")

    # print(f"{ate_rmse:.4f}")
    # print(f"{rpe_rmse:.4f}")


def viz_pair_stats(pairs, prefix, save_dir=None):

    res_trans = []
    res_R = []
    for pair in tqdm(pairs):
        try:
            est, gt = align_pair(pair, prefix)
            if est is None:
                continue
        except Exception as e:
            print(e)
        evaluator = BatchEvaluator()
        evaluator.evaluate(poses_pred=est, poses_gt=gt)

        res_trans.append(np.array(evaluator.error_trans).clip(max=20))
        res_R.append(np.array(evaluator.error_rot).clip(max=25))

    # plt.boxplot(res_trans)
    # plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Translation error box plot
    bp1 = ax1.boxplot(
        res_trans,
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
    )

    # Color the boxes
    for patch in bp1["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_edgecolor("darkblue")
        patch.set_linewidth(2)

    ax1.set_ylabel("Translation Error (cm)", fontsize=12)
    ax1.set_xlabel("Scene Index", fontsize=12)
    ax1.set_title(
        "Translation Error Distribution per Scene", fontsize=14, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Rotation error box plot
    bp2 = ax2.boxplot(
        res_R,
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
    )

    # Color the boxes
    for patch in bp2["boxes"]:
        patch.set_facecolor("lightcoral")
        patch.set_edgecolor("darkred")
        patch.set_linewidth(2)

    ax2.set_ylabel("Rotation Error (degrees)", fontsize=12)
    ax2.set_xlabel("Scene Index", fontsize=12)
    ax2.set_title(
        "Rotation Error Distribution per Scene", fontsize=14, fontweight="bold"
    )
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    # save the figure to pdf with high dpi
    if save_dir is not None:
        plt.savefig(f"{save_dir}", dpi=300)

    plt.show()


def align_pair_via_kf(pair):
    kf_gt = pair["kf_gt"]  # first 2 frames are skipped for init in pipeline
    kf_est = pair["kf_est"]

    # test_est = np.linalg.inv(test_est)
    kf_gt = np.linalg.inv(kf_gt)

    if kf_gt.shape != kf_est.shape:
        print(f"Pair not equal in length, gt: {kf_gt.shape[0]}, est: {kf_est.shape[0]}")
        return None, None

    N_poses = kf_gt.shape[0]
    time_stamps = np.arange(N_poses).astype(np.float32) * 0.1

    kf_ref = PoseTrajectory3D(
        poses_se3=kf_gt,
        timestamps=time_stamps,
    )

    kf_est = PoseTrajectory3D(poses_se3=kf_est, timestamps=time_stamps)

    max_diff = 0.01

    kf_ref, kf_est = sync.associate_trajectories(kf_ref, kf_est, max_diff)

    kf_est_aligned = copy.deepcopy(kf_est)
    R, t, s = kf_est_aligned.align(kf_ref, correct_scale=True, correct_only_scale=False)

    # fig = plt.figure()
    # traj_by_label = {
    #     # "estimate (not aligned)": traj_est,
    #     "estimate (aligned)": traj_est_aligned,
    #     "reference": traj_ref
    # }

    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    # plt.show()

    # tum_ate = metrics.APE(metrics.PoseRelation.translation_part)
    # tum_ate.process_data((kf_ref, traj_est_aligned))
    # ate_rmse = tum_ate.get_statistic(metrics.StatisticsType.rmse)

    # delta = 1
    # rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta, Unit.frames, all_pairs=True)
    # rpe.process_data((kf_ref, traj_est_aligned))
    # rpe_rmse = rpe.get_statistic(metrics.StatisticsType.rmse)

    # print(f"ATE RMSE: {ate_rmse:.4f} m")
    N_traj = pair["traj_est"].shape[0]
    traj_timestamps = np.arange(N_traj).astype(np.float32) * 0.1  # Use correct length

    traj_est_aligned = PoseTrajectory3D(
        poses_se3=pair["traj_est"].copy(), timestamps=traj_timestamps
    )
    traj_est_aligned.scale(s)
    traj_est_aligned.transform(lie.se3(R, t))

    traj_gt = np.linalg.inv(pair["traj_gt"])

    return np.array(traj_est_aligned.poses_se3), traj_gt
