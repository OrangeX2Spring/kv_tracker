import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

from kv_tracker.eval_tools.onepose_evaluator import BatchEvaluator
from kv_tracker.eval_tools.evo_utils import align_pair


def viz_pair_stats(pairs, prefix):

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
    plt.show()
