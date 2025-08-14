import os
import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime

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

DATASETS = {
    "7scenes": {
        "datasource": "scenes7Loader",
        "dataset_dir": "datasets/7-scenes",
        "dataloader": "kv_tracker.dataloaders.scenes_7",
    },
    "tum": {
        "datasource": "TUMLoader",
        "dataset_dir": "datasets/tum_rgbd",
        "dataloader": "kv_tracker.dataloaders.tum",
    },
    "sintel": {
        "datasource": "SintelLoader",
        "dataset_dir": "datasets/sintel/training/final",
        "dataloader": "kv_tracker.dataloaders.sintel",
    },
    "onepose": {
        "datasource": "onePoseLoader",
        "dataset_dir": "datasets/OnePose/test_data",
        "dataloader": "kv_tracker.dataloaders.onepose_dataset",
    },
    "arctic": {
        "datasource": "arcticLoader",
        "dataset_dir": "datasets/arctic_data/data/cropped_images_grab_only_subset/s01",
        "dataloader": None,
    },
}

PIPELINES = {
    "main": "main",
    "bidirectional": "bidirectional_ablation",
    "depth_anything": "depth_anything_attempt",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--pipeline", default="main", choices=PIPELINES.keys())
    parser.add_argument("--results", required=True)
    parser.add_argument("--resume", default=False, action="store_true")
    # onepose-specific
    parser.add_argument("--tless", default=False, action="store_true")
    parser.add_argument("--bbox", default=False, action="store_true")
    eval_args, tracker_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + tracker_args

    # Import the right pipeline
    import importlib
    pipeline_module = importlib.import_module(PIPELINES[eval_args.pipeline])
    run_track3r = pipeline_module.run_track3r
    dump_data = pipeline_module.dump_data

    # Import the right dataloader
    ds_cfg = DATASETS[eval_args.dataset]
    dataset_dir = Path(ds_cfg["dataset_dir"])
    if eval_args.dataset == "onepose" and eval_args.tless:
        dataset_dir = Path("datasets/OnePose/lowtexture_test_data")

    if eval_args.dataset == "arctic":
        scenes_list = [dataset_dir / s for s in ARCTIC_SCENES]
    else:
        dataloader_module = importlib.import_module(ds_cfg["dataloader"])
        get_all_scenes_dir = dataloader_module.get_all_scenes_dir
        scenes_list = get_all_scenes_dir(dataset_dir)

    cfg = {
        "datasource": ds_cfg["datasource"],
        "que_size": -1,
        "offset": 2 if eval_args.dataset == "arctic" else 0,
        "export_masked_imgs": False,
    }
    if eval_args.dataset == "onepose":
        cfg["use_bbox"] = 50 if eval_args.bbox else 0
        del cfg["export_masked_imgs"]

    skip = 0
    completed_scenes = []
    if eval_args.resume:
        pickle_path = dataset_dir / "completed_scenes_list.pkl"
        with open(pickle_path, "rb") as f:
            completed_scenes = pickle.load(f)
        skip = len(completed_scenes)
        print(f"Resuming from scene #{skip}")

    if skip > 0:
        print(f"Skipping the first {skip} scenes.")

    for i, scene in enumerate(scenes_list[skip:]):
        cfg["scene_dir"] = scene
        results_dir = Path(scene) / eval_args.results
        cfg["results_path"] = results_dir
        os.makedirs(results_dir, exist_ok=True)

        run_track3r(cfg)
        print("DONE scene:", scene)

        completed_scenes.append(scene)
        dump_data(completed_scenes, dataset_dir / "completed_scenes_list.pkl")

        print(f"Completed {i + 1 + skip}/{len(scenes_list)}.")
        print("-------------------------------\n")
