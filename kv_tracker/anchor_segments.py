"""Anchor-only reanchored segments: each segment tracks against its anchor image alone.

The user's segmentation + reanchor design without the in-segment rebuild. A segment is
bootstrapped on its anchor b ([b, b], as every map starts) and never rebuilt; saved
26159 office frames tracked this way (local frames < 49) matched native per-frame
error. A cut at frame f is made immediately: the old map reads f's dense geometry,
its KV and images are released, the new map bootstraps on f, and the connection pins
rotation and position at f with the point-fit scale of f's two pointmaps. No delayed
output and no minimum segment length tied to a rebuild.

Cut rules, all causal and from tracker outputs only (no ORB, no GT):
- 'schedule': every `length` frames.
- 'covisibility': when fewer than `covisibility` of the anchor's confident points lie
  inside the current camera's view (frustum from the anchor's own pixel rays).
- 'object_view': when the viewing direction about the object centre (masked anchor
  points) has turned more than `view_degrees` from the anchor's.
"""
import time
import weakref

import numpy as np
import torch

from .map_handoff import bridge, transform_pose
from .pi3_utilts import pi3_inference


class AnchorSegments:
    def __init__(self, model, log, cut, length=48, covisibility=.7, view_degrees=20.):
        assert cut in ('schedule', 'covisibility', 'object_view')
        assert length >= 2 and 0 < covisibility < 1 and 0 < view_degrees < 180
        self.model, self.log, self.cut = model, log, cut
        self.length, self.covisibility, self.view_degrees = length, covisibility, view_degrees
        self.device = next(model.parameters()).device
        self.world = (torch.tensor(1., dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                      torch.zeros(3, dtype=torch.float64))
        # Runner interface shared with ReanchorMaps: poses are returned already global.
        self.transforms, self.events, self.pending = [self.world], [], None
        self.cuts, self.start = [], 0

    def cache_bytes(self):
        return sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                    for layer in self.model.cache.values() for t in layer.values()}.values())

    def bootstrap(self, image, frame, mask):
        torch.cuda.synchronize()
        started = time.perf_counter()
        points, poses, conf, _, _, self.origin = pi3_inference(
            self.model, [np.stack([image, image])], self.device, store_cache=True)
        assert self.model.cache
        torch.cuda.synchronize()
        self.log(dict(kind='bootstrap', frame=frame, input_ids=[frame, frame], input_images=2,
                      seconds=time.perf_counter() - started, cache_bytes=self.cache_bytes()))
        pose = poses[0, 0].double().cpu()
        self.anchor_points = points[0, 0].double().cpu()
        self.anchor_conf = conf[0, 0, ..., 0].float().cpu()
        camera = (self.anchor_points - pose[:3, 3]) @ pose[:3, :3]
        front = camera[..., 2] > 0
        self.half_fov = (camera[front][:, :2] / camera[front][:, 2:]).abs().amax(0)
        confident = (self.anchor_conf >= self.anchor_conf.median())[::8, ::8]
        self.signal_points = self.anchor_points[::8, ::8][confident]
        self.anchor_position = pose[:3, 3]
        if self.cut == 'object_view':
            assert mask is not None and mask.shape == self.anchor_points.shape[:2] and mask.any()
            self.center = self.anchor_points[mask].mean(0)
        return pose

    def signal(self, local):
        """Cut signal for the current local pose; larger means further from the anchor."""
        if self.cut == 'covisibility':
            camera = (self.signal_points - local[:3, 3]) @ local[:3, :3]
            front = camera[:, 2] > 0
            inside = front.clone()
            inside[front] = (camera[front][:, :2] / camera[front][:, 2:]).abs().le(self.half_fov).all(1)
            return float(inside.double().mean())
        if self.cut == 'object_view':
            cosine = torch.nn.functional.cosine_similarity(
                self.center - self.anchor_position, self.center - local[:3, 3], dim=0)
            return float(torch.rad2deg(torch.acos(cosine.clamp(-1, 1))))
        return None

    def step(self, image, frame, mask=None):
        if frame == 0:
            return transform_pose(self.bootstrap(image, frame, mask), self.world).float().numpy()
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw = pi3_inference(self.model, [image[None]], self.device, cam_only=True, use_cache=True)
        local = (self.origin @ raw)[0, 0].double().cpu()
        pose = transform_pose(local, self.world)
        signal = self.signal(local)
        torch.cuda.synchronize()
        self.log(dict(kind='query', frame=frame, bank_ids=[self.start], input_images=1,
                      seconds=time.perf_counter() - started, cache_bytes=self.cache_bytes(),
                      signal=signal))
        cut = {'schedule': frame - self.start >= self.length,
               'covisibility': signal is not None and signal < self.covisibility,
               'object_view': signal is not None and signal > self.view_degrees}[self.cut]
        if cut:
            self.reanchor(image, frame, pose, mask, signal)
        return pose.float().numpy()

    def reanchor(self, image, frame, old_pose, mask, signal):
        torch.cuda.synchronize()
        started = time.perf_counter()
        # The old map reads frame's dense geometry (its global output frame).
        points, _, conf, _, _, origin = pi3_inference(
            self.model, [image[None]], self.device, use_cache=True)
        assert origin is None  # single-query outputs are in raw cache coordinates
        offset = self.origin.double().cpu()
        old_points = points[0, 0].double().cpu() @ offset[:3, :3].T + offset[:3, 3]
        scale, rotation, translation = self.world
        old_points = scale * (old_points @ rotation.T) + translation
        old_conf = conf[0, 0, ..., 0].float().cpu()
        refs = [weakref.ref(t) for layer in self.model.cache.values() for t in layer.values()]
        self.model.cache = {}
        assert all(ref() is None for ref in refs), 'Old KV tensor still referenced'
        previous = self.start
        new_pose = self.bootstrap(image, frame, mask)
        # Point fit supplies scale; its checks are diagnostics, not a gate.
        _, event, _ = bridge(old_points, self.anchor_points, old_conf, self.anchor_conf,
                             old_pose, new_pose)
        s = torch.tensor(event['scale'], dtype=torch.float64)
        assert s > 0
        r = old_pose[:3, :3] @ new_pose[:3, :3].T
        self.world = (s, r, old_pose[:3, 3] - s * (r @ new_pose[:3, 3]))
        self.start = frame
        self.cuts.append(dict(frame=frame, previous_anchor=previous, signal=signal, scale=event['scale'],
                              fit_accepted=event['accepted'], checks=event.get('checks'),
                              validation_median=event.get('validation_median'),
                              camera_rotation_deg=event.get('camera_rotation_deg')))
        torch.cuda.synchronize()
        self.log(dict(kind='update_total', frame=frame, bank_ids=[frame], seconds=time.perf_counter() - started,
                      cache_bytes=self.cache_bytes(), scale=event['scale'], signal=signal))

    def finish(self, image, frame):
        """Nothing is pending: every connection is made at its cut."""
