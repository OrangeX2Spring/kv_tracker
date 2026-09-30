"""Passive causal routing diagnostic; never changes tracking or token choices."""
import json
import time

import numpy as np
import torch


def project_cache(points, confidence, pose, intrinsics, height, width):
    """One evidence sample per source patch; nearest target patch, two depths.

    points and camera-to-world pose must share a coordinate gauge. Missing
    support is unknown. Confidence is the front sample's confidence, not a sum.
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert confidence.shape == points.shape[:1] and pose.shape == (4, 4)
    assert points.dtype == pose.dtype == intrinsics.dtype == torch.float32
    assert height % 14 == width % 14 == 0
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    positive = z > 0
    safe_z = torch.where(positive, z, torch.ones_like(z))
    uv = camera[:, :2] / safe_z[:, None]
    uv = uv * intrinsics.diag()[:2] + intrinsics[:2, 2]
    valid = (torch.isfinite(camera).all(1) & torch.isfinite(uv).all(1) & positive &
             (uv[:, 0] >= -.5) & (uv[:, 0] < width - .5) &
             (uv[:, 1] >= -.5) & (uv[:, 1] < height - .5))
    cells = (height // 14) * (width // 14)
    # Sanitize invalid coordinates before integer conversion; sentinel is omitted.
    xy = torch.where(valid[:, None], uv, torch.zeros_like(uv))
    cell = ((xy[:, 1] + .5).floor().long() // 14 * (width // 14) +
            (xy[:, 0] + .5).floor().long() // 14)
    cell = torch.where(valid, cell, cells)
    depths = torch.where(valid, z, torch.full_like(z, float('inf')))
    front = z.new_full((cells + 1,), float('inf'))
    front.scatter_reduce_(0, cell, depths, reduce='amin', include_self=True)
    secondary = z.new_full((cells + 1,), float('inf'))
    back_depths = torch.where(valid & (z > 1.04 * front[cell]), depths,
                             torch.full_like(z, float('inf')))
    secondary.scatter_reduce_(0, cell, back_depths, reduce='amin', include_self=True)
    ids = torch.arange(len(points), device=points.device)
    first = torch.full((cells + 1,), len(points), device=points.device, dtype=torch.long)
    first.scatter_reduce_(0, cell, torch.where(valid & (z == front[cell]), ids,
                          len(points)), reduce='amin', include_self=True)
    counts = torch.zeros(cells + 1, device=points.device, dtype=torch.long)
    counts.scatter_add_(0, cell, valid.long())
    supported = counts[:cells] > 0
    conf = torch.cat([confidence, confidence.new_zeros(1)])[first[:cells]]
    gap = torch.where(torch.isfinite(secondary[:cells]),
                      secondary[:cells] / front[:cells] - 1, 0.)
    return dict(support=counts[:cells], front_depth=torch.where(supported, front[:cells], 0.),
                confidence=conf, secondary_gap=gap,
                source_index=torch.where(supported, first[:cells], -1),
                valid_samples=valid)


class SceneRoutingObserver:
    def __init__(self, directory, intrinsics, height, width):
        self.directory, self.intrinsics = directory, intrinsics
        self.height, self.width = height, width
        self.rebuild_id = 0
        self.rows = []

    def rebuild(self, frame_ids, points, confidence, poses):
        assert points.shape == (1, len(frame_ids), self.height, self.width, 3)
        assert confidence.shape == (1, len(frame_ids), self.height, self.width, 1)
        # Clone because the tracker later modifies its confidence/point tensors.
        self.points = points[0, :, 7::14, 7::14].reshape(-1, 3).clone().float()
        self.confidence = confidence[0, :, 7::14, 7::14, 0].reshape(-1).clone()
        self.pose = poses[0, -1].clone().float()
        self.frame_ids = list(frame_ids)
        self.rebuild_id += 1
        np.savez(self.directory / f'cache_{self.rebuild_id}.npz',
                 points=self.points.cpu().numpy(), confidence=self.confidence.cpu().numpy(),
                 frame_ids=frame_ids, pose=self.pose.cpu().numpy())

    def before_query(self, frame):
        assert max(self.frame_ids) < frame
        torch.cuda.synchronize()
        started = time.perf_counter()
        maps = project_cache(self.points, self.confidence, self.pose,
                             self.intrinsics, self.height, self.width)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        np.savez(self.directory / f'route_{frame:06d}.npz',
                 **{k: v.cpu().numpy() for k, v in maps.items()}, pose=self.pose.cpu().numpy())
        self.rows.append(dict(frame=frame, cache=self.rebuild_id, frame_ids=self.frame_ids,
            routing_seconds=seconds, supported_fraction=float((maps['support'] > 0).float().mean()),
            valid_fraction=float(maps['valid_samples'].float().mean()),
            metadata_bytes=sum(t.numel() * t.element_size() for t in
                               (self.points, self.confidence, self.pose, self.intrinsics))))
        (self.directory / 'routing.json').write_text(json.dumps(self.rows, indent=2))

    def after_query(self, poses):
        self.pose = poses[0, 0].clone().float()
