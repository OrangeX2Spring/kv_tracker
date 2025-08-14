import numpy as np


class Evaluator:
    def __init__(self):
        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        self.cmd7 = []
        self.add = []

    def cm_degree_1_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
        # self.cmd1.append(translation_distance < 1 and angular_distance < 1)
        self.cmd1.append(translation_distance < 1)

    def cm_degree_5_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
        # self.cmd5.append(translation_distance < 5 and angular_distance < 5)
        self.cmd5.append(translation_distance < 5)

    def cm_degree_3_metric(self, pose_pred, pose_target):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
        # self.cmd3.append(translation_distance < 3 and angular_distance < 3)
        self.cmd3.append(translation_distance < 3)

    def evaluate(self, pose_pred, pose_gt):
        if pose_pred is None:
            self.cmd5.append(False)
            self.cmd1.append(False)
            self.cmd3.append(False)
            self.cmd7.append(False)
        else:
            if pose_pred.shape == (4, 4):
                pose_pred = pose_pred[:3, :4]
            if pose_gt.shape == (4, 4):
                pose_gt = pose_gt[:3, :4]
            self.cm_degree_1_metric(pose_pred, pose_gt)
            self.cm_degree_3_metric(pose_pred, pose_gt)
            self.cm_degree_5_metric(pose_pred, pose_gt)

    def summarize(self):
        cmd1 = np.mean(self.cmd1)
        cmd3 = np.mean(self.cmd3)
        cmd5 = np.mean(self.cmd5)
        print("1 cm 1 degree metric: {}".format(cmd1))
        print("3 cm 3 degree metric: {}".format(cmd3))
        print("5 cm 5 degree metric: {}".format(cmd5))

        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        self.cmd7 = []
        return {"cmd1": cmd1, "cmd3": cmd3, "cmd5": cmd5}


class BatchEvaluator:
    def __init__(self):
        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        self.cmd7 = []
        self.add = []

        self.error_trans = []
        self.error_rot = []

    def cm_degree_metric(self, poses_pred, poses_gt, threshold_cm, threshold_deg):
        """Batched metric computation"""
        # Handle single pose or batch
        if poses_pred.ndim == 2:
            poses_pred = poses_pred[np.newaxis, ...]
            poses_gt = poses_gt[np.newaxis, ...]

        # Extract translations: (N, 3)
        trans_pred = poses_pred[:, :3, 3]
        trans_gt = poses_gt[:, :3, 3]

        # Translation distances in cm: (N,)
        translation_distances = np.linalg.norm(trans_pred - trans_gt, axis=1) * 100

        # Extract rotations: (N, 3, 3)
        rot_pred = poses_pred[:, :3, :3]
        rot_gt = poses_gt[:, :3, :3]

        # Rotation differences: (N, 3, 3)
        rotation_diffs = rot_pred @ np.transpose(rot_gt, (0, 2, 1))

        # Traces: (N,)
        traces = np.trace(rotation_diffs, axis1=1, axis2=2)
        traces = np.clip(traces, -1, 3)

        # Angular distances in degrees: (N,)
        cos_angles = (traces - 1.0) / 2.0
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angular_distances = np.rad2deg(np.arccos(cos_angles))

        # Check thresholds: (N,)
        success = (translation_distances < threshold_cm) & (
            angular_distances < threshold_deg
        )

        return success

    def error(self, poses_pred, poses_gt):
        if poses_pred.ndim == 2:
            poses_pred = poses_pred[np.newaxis, ...]
            poses_gt = poses_gt[np.newaxis, ...]

        # Extract translations: (N, 3)
        trans_pred = poses_pred[:, :3, 3]
        trans_gt = poses_gt[:, :3, 3]

        # Translation distances in cm: (N,)
        translation_distances = np.linalg.norm(trans_pred - trans_gt, axis=1) * 100

        # Extract rotations: (N, 3, 3)
        rot_pred = poses_pred[:, :3, :3]
        rot_gt = poses_gt[:, :3, :3]

        # Rotation differences: (N, 3, 3)
        rotation_diffs = rot_pred @ np.transpose(rot_gt, (0, 2, 1))

        # Traces: (N,)
        traces = np.trace(rotation_diffs, axis1=1, axis2=2)
        traces = np.clip(traces, -1, 3)

        # Angular distances in degrees: (N,)
        cos_angles = (traces - 1.0) / 2.0
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angular_distances = np.rad2deg(np.arccos(cos_angles))

        return translation_distances, angular_distances

    def evaluate(self, poses_pred, poses_gt):
        """Evaluate single pose or batch of poses"""
        if poses_pred is None:
            N = len(poses_gt) if poses_gt.ndim == 3 else 1
            self.cmd1.extend([False] * N)
            self.cmd3.extend([False] * N)
            self.cmd5.extend([False] * N)
            return

        # Convert 4x4 to ensure consistent shape
        if poses_pred.ndim == 2 and poses_pred.shape == (4, 4):
            poses_pred = poses_pred[:3, :4][np.newaxis, ...]
            poses_gt = poses_gt[:3, :4][np.newaxis, ...]
        elif poses_pred.ndim == 3:
            if poses_pred.shape[1:] == (4, 4):
                poses_pred = poses_pred[:, :3, :4]
                poses_gt = poses_gt[:, :3, :4]

        # Compute all metrics at once
        cmd1_results = self.cm_degree_metric(poses_pred, poses_gt, 1, 1)
        cmd3_results = self.cm_degree_metric(poses_pred, poses_gt, 3, 3)
        cmd5_results = self.cm_degree_metric(poses_pred, poses_gt, 5, 5)

        # Compute errors
        trans_errors, rot_errors = self.error(poses_pred, poses_gt)

        # Append results
        self.cmd1.extend(cmd1_results.tolist())
        self.cmd3.extend(cmd3_results.tolist())
        self.cmd5.extend(cmd5_results.tolist())
        self.error_trans.extend(trans_errors.tolist())
        self.error_rot.extend(rot_errors.tolist())

    def summarize(self):
        cmd1 = np.mean(self.cmd1)
        cmd3 = np.mean(self.cmd3)
        cmd5 = np.mean(self.cmd5)
        print(f"1 cm 1 degree metric: {cmd1*100:.2f}%")
        print(f"3 cm 3 degree metric: {cmd3*100:.2f}%")
        print(f"5 cm 5 degree metric: {cmd5*100:.2f}%")

        print(f"{cmd1*100:.2f},{cmd3*100:.2f},{cmd5*100:.2f}")

        print(f"{cmd1*100:.2f} & {cmd3*100:.2f} & {cmd5*100:.2f}")

        # # Compute translation error statistics
        # trans_mean = np.mean(self.error_trans)
        # trans_median = np.median(self.error_trans)
        # trans_std = np.std(self.error_trans)
        # print(f'\nTranslation Error (cm):')
        # print(f'  Mean:   {trans_mean:.2f}')
        # print(f'  Median: {trans_median:.2f}')
        # print(f'  Std:    {trans_std:.2f}')

        # # Compute rotation error statistics
        # rot_mean = np.mean(self.error_rot)
        # rot_median = np.median(self.error_rot)
        # rot_std = np.std(self.error_rot)
        # print(f'\nRotation Error (degrees):')
        # print(f'  Mean:   {rot_mean:.2f}')
        # print(f'  Median: {rot_median:.2f}')
        # print(f'  Std:    {rot_std:.2f}')

        # Reset all metrics
        self.cmd1 = []
        self.cmd3 = []
        self.cmd5 = []
        self.cmd7 = []
        self.error_trans = []
        self.error_rot = []

        # return {
        #     'cmd1': cmd1,
        #     'cmd3': cmd3,
        #     'cmd5': cmd5,
        #     'trans_mean': trans_mean,
        #     'trans_median': trans_median,
        #     'trans_std': trans_std,
        #     'rot_mean': rot_mean,
        #     'rot_median': rot_median,
        #     'rot_std': rot_std,
        # }
