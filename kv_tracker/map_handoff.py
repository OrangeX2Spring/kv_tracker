"""Single causal two-image Pi3 map handoff. No ground truth or learned selector."""
import time
import weakref

import numpy as np
import torch

from .pi3_utilts import pi3_inference


def fit_similarity(source, target):
    """Float64 CPU points, mapping source into target; reject degenerate fits."""
    assert source.ndim == 2 and source.shape == target.shape and source.shape[1] == 3
    assert source.dtype == target.dtype == torch.float64
    x, y = source - source.mean(0), target - target.mean(0)
    u, d, vh = torch.linalg.svd(y.T @ x / len(x))
    if float(d[0]) <= 0 or float(d[1] / d[0]) < 1e-3:
        return None
    sign = torch.ones(3, dtype=source.dtype)
    sign[-1] = torch.linalg.det(u) * torch.linalg.det(vh)
    rotation = (u * sign) @ vh
    scale = (d * sign).sum() / x.square().sum(1).mean()
    translation = target.mean(0) - scale * (rotation @ source.mean(0))
    return scale, rotation, translation


def transform_pose(pose, transform):
    scale, rotation, translation = transform
    result = pose.double().clone()
    result[:3, :3] = rotation @ result[:3, :3]
    result[:3, 3] = scale * (rotation @ result[:3, 3]) + translation
    return result


def compose(outer, inner):
    so, ro, to = outer
    si, ri, ti = inner
    return so * si, ro @ ri, so * (ro @ ti) + to


def bridge(old_points, new_points, old_conf, new_conf, old_pose, new_pose):
    """Fit on checkerboard samples; validate disjoint samples and camera pose.

    Confidence uses each map's median, not a calibrated probability. Thresholds
    are preregistered engineering gates, not demonstrated uncertainty bounds.
    """
    assert old_points.shape == new_points.shape and old_points.shape[-1] == 3
    h, w = old_points.shape[:2]
    assert old_conf.shape == new_conf.shape == (h, w)
    y, x = torch.meshgrid(torch.arange(0, h, 4), torch.arange(0, w, 4), indexing='ij')
    old = old_points[y, x].double().reshape(-1, 3)
    new = new_points[y, x].double().reshape(-1, 3)
    ca, cb = old_conf[y, x].flatten(), new_conf[y, x].flatten()
    valid = torch.isfinite(old).all(1) & torch.isfinite(new).all(1)
    valid &= torch.isfinite(ca) & torch.isfinite(cb)
    if int(valid.sum()) < 256:
        return None, dict(accepted=False, reason='insufficient_finite_points'), {}
    valid &= (ca >= ca[valid].median()) & (cb >= cb[valid].median())
    train = valid & (((y // 4 + x // 4).flatten() % 2) == 0)
    test = valid & ~train
    evidence = dict(old_points=old.numpy(), new_points=new.numpy(),
                    sampled_pixels=torch.stack((y.flatten(), x.flatten()), 1).numpy(),
                    fit_mask=train.numpy(), validation_mask=test.numpy(),
                    old_pose=old_pose.numpy(), new_pose=new_pose.numpy())
    if min(int(train.sum()), int(test.sum())) < 128:
        return None, dict(accepted=False, reason='insufficient_confident_points'), evidence
    extent = (old[train] - old[train].mean(0)).square().sum(1).mean().sqrt()
    if float(extent) <= 0:
        return None, dict(accepted=False, reason='zero_extent'), evidence
    mask = train.clone()
    for _ in range(3):
        fitted = fit_similarity(new[mask], old[mask])
        if fitted is None:
            return None, dict(accepted=False, reason='degenerate_fit'), evidence
        scale, rotation, translation = fitted
        residual = (scale * (new @ rotation.T) + translation - old).norm(dim=1)
        mask = train & (residual <= max(float(3 * residual[train].median()), float(.01 * extent)))
        if int(mask.sum()) < 128:
            return None, dict(accepted=False, reason='insufficient_robust_inliers'), evidence
    fitted = fit_similarity(new[mask], old[mask])
    if fitted is None:
        return None, dict(accepted=False, reason='degenerate_final_fit'), evidence
    scale, rotation, translation = fitted
    residual = (scale * (new @ rotation.T) + translation - old).norm(dim=1) / extent
    validation = residual[test]
    inliers = test & (residual <= .05)
    bins = ((y.flatten() * 4 // h) * 4 + x.flatten() * 4 // w)
    coverage = int(bins[inliers].unique().numel())
    mapped_pose = transform_pose(new_pose, fitted)
    translation_error = float((mapped_pose[:3, 3] - old_pose[:3, 3]).norm() / extent)
    relative_rotation = mapped_pose[:3, :3].T @ old_pose[:3, :3].double()
    angle = float(torch.rad2deg(torch.acos(((torch.trace(relative_rotation) - 1) / 2).clamp(-1, 1))))
    checks = dict(scale_positive=bool(scale > 0),
                  validation_median=bool(validation.median() <= .02),
                  validation_p90=bool(torch.quantile(validation, .9) <= .05),
                  validation_inliers=bool((validation <= .05).double().mean() >= .8),
                  spatial_coverage=coverage >= 12,
                  camera_translation=translation_error <= .05,
                  camera_rotation=angle <= 5.)
    event = dict(accepted=all(checks.values()), checks=checks, scale=float(scale),
                 rotation=rotation.tolist(), translation=translation.tolist(),
                 extent=float(extent), fit_points=int(mask.sum()), validation_points=int(test.sum()),
                 validation_median=float(validation.median()),
                 validation_p90=float(torch.quantile(validation, .9)),
                 validation_inlier_fraction=float((validation <= .05).double().mean()),
                 spatial_bins=coverage, camera_translation_normalized=translation_error,
                 camera_rotation_deg=angle)
    evidence.update(normalized_residual=residual.numpy(), robust_fit_mask=mask.numpy(),
                    mapped_pose=mapped_pose.numpy())
    return fitted if event['accepted'] else None, event, evidence


class MapHandoff:
    def __init__(self, model, mode, log, save_bridge, query_executor=None, native_keyframe_cap=20,
                 local_keyframe_cap=2, pin_rebuilds=False, pin_scale=False, shared_scale=False):
        # 'reanchor' rebuilds like 'fixed' (anchor + latest) and keeps the first
        # rebuild's anchor geometry for the inter-map connection. Each later rebuild
        # re-normalizes Pi3's scale about the anchor camera; rebuild_scale restores
        # the first rebuild's scale from the anchor pointmap every rebuild shares.
        # pin_rebuilds additionally removes the rebuild's free rotation/translation
        # relative to the anchor: the new solution is rigidly moved so its pose of
        # the refresh frame equals the outgoing bank's pose of that frame.
        # pin_scale also replaces the anchor-pointmap scale: rebuild_scale makes the
        # new rebuild's depths of the refresh frame (in its own camera) match the
        # outgoing bank's, read with one dense query before the bank changes.
        # shared_scale (three-image banks) instead compares the previous keyframe,
        # a rebuild member of both the outgoing and the new bank, so both depths
        # are rebuild predictions and no extra forward is needed. The first
        # rebuild has no shared keyframe and keeps the anchor-defined scale.
        assert mode in ('native', 'fixed', 'handoff', 'oracle', 'reanchor')
        assert native_keyframe_cap >= 2
        assert mode == 'native' or native_keyframe_cap == 20
        assert local_keyframe_cap in (2, 3)
        assert mode == 'reanchor' or local_keyframe_cap == 2
        assert mode == 'reanchor' or not pin_rebuilds
        assert pin_rebuilds or not pin_scale
        assert not shared_scale or (pin_rebuilds and not pin_scale and local_keyframe_cap == 3)
        self.pin_rebuilds, self.pin_scale, self.shared_scale = pin_rebuilds, pin_scale, shared_scale
        self.local_keyframe_cap = local_keyframe_cap
        self.native_keyframe_cap = native_keyframe_cap
        self.model, self.mode, self.log, self.save_bridge = model, mode, log, save_bridge
        self.query_executor = query_executor
        self.device = next(model.parameters()).device
        self.ids, self.images = [], []
        self.events = []
        self.transform = (torch.tensor(1., dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                          torch.zeros(3, dtype=torch.float64))
        self.latest_points = self.latest_conf = None
        self.rebuild_scale = 1.

    def cache_bytes(self):
        storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                   for layer in self.model.cache.values() for t in layer.values()}
        return sum(storage.values())

    def query_geometry(self, image, frame):
        """Read shared-view geometry without rebuilding or changing the bank."""
        torch.cuda.synchronize()
        started = time.perf_counter()
        cache = [t for layer in self.model.cache.values() for t in layer.values()]
        points, poses, conf, _, _, origin = pi3_inference(
            self.model, [image[None]], self.device, use_cache=True)
        assert origin is None  # Single-query outputs are still in raw cache coordinates.
        assert all(a is b for a, b in zip(cache,
            [t for layer in self.model.cache.values() for t in layer.values()], strict=True))
        points = points[0, 0].double().cpu()
        offset = self.origin.double().cpu()
        points = points @ offset[:3, :3].T + offset[:3, 3]
        pose = (self.origin @ poses)[0, 0].cpu()
        if self.mode == 'reanchor':
            points = self.rebuild_scale * points
            pose[:3, 3] *= self.rebuild_scale
        scale, rotation, translation = self.transform
        torch.cuda.synchronize()
        self.log(dict(kind='shared_geometry', frame=frame, input_images=1,
                      seconds=time.perf_counter() - started, cache_bytes=self.cache_bytes()))
        return (scale * (points @ rotation.T) + translation,
                transform_pose(pose, self.transform),
                conf[0, 0, ..., 0].cpu())

    def reconstruct(self, images, ids, frame, kind):
        if self.query_executor is not None:
            self.query_executor.reset()
        torch.cuda.synchronize()
        started = time.perf_counter()
        points, poses, conf, _, _, origin = pi3_inference(
            self.model, [np.stack(images)], self.device, store_cache=True)
        h, w = images[0].shape[:2]
        expected_tokens = len(images) * (5 + (h // 14) * (w // 14))
        assert self.model.cache
        assert all(t.shape[2] == expected_tokens for layer in self.model.cache.values() for t in layer.values())
        torch.cuda.synchronize()
        self.log(dict(kind=kind, frame=frame, input_ids=list(ids), input_images=len(images),
                      seconds=time.perf_counter() - started, cache_bytes=self.cache_bytes()))
        if kind == 'rebuild' and (self.mode == 'oracle' or (self.mode == 'reanchor' and frame == 49)):
            self.anchor_points = points[0, 0].cpu().clone()
            self.anchor_conf = conf[0, 0, ..., 0].cpu().clone()
            self.anchor_pose = poses[0, 0].cpu().clone()
            self.bank_last_pose = poses[0, -1].cpu().clone()
        return points[0].cpu(), poses[0].cpu(), conf[0, ..., 0].cpu(), origin

    def bootstrap(self, image):
        points, poses, conf, self.origin = self.reconstruct([image, image], [0, 0], 0, 'bootstrap')
        self.transform = (self.transform[0], self.transform[1], -points.mean((0, 1, 2)).double())
        self.ids, self.images = [0], [image]
        self.latest_points, self.latest_conf = points[-1].clone(), conf[-1].clone()
        return transform_pose(poses[0], self.transform).float().numpy()

    def step(self, image, frame, update=True, dense_query=False):
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw = (pi3_inference(self.model, [image[None]], self.device, cam_only=True, use_cache=True)
               if self.query_executor is None or dense_query else
               self.query_executor.forward(self.model, image, self.device))
        local_pose = (self.origin @ raw)[0, 0].cpu()
        if self.mode == 'reanchor':
            local_pose[:3, 3] *= self.rebuild_scale
        global_pose = transform_pose(local_pose, self.transform).float().numpy()
        torch.cuda.synchronize()
        self.log(dict(kind='query', frame=frame, bank_ids=list(self.ids), input_images=1,
                      seconds=time.perf_counter() - started, cache_bytes=self.cache_bytes()))
        if not update or (self.mode == 'oracle' and frame != 49) or (frame + 1) % 50 or (
                self.mode == 'native' and len(self.ids) >= self.native_keyframe_cap):
            return global_pose
        started = time.perf_counter()
        if self.mode == 'handoff' and frame == 749:
            assert self.ids == [0, 699]
            old_cache = self.model.cache
            refs = [weakref.ref(t) for layer in old_cache.values() for t in layer.values()]
            old_storage = {t.untyped_storage().data_ptr() for layer in old_cache.values() for t in layer.values()}
            old_bytes = self.cache_bytes()
            self.model.cache = {}
            new_ids, new_images = [699, 749], [self.images[-1], image]
            points, poses, conf, origin = self.reconstruct(new_images, new_ids, frame, 'candidate')
            fitted, event, evidence = bridge(self.latest_points, points[0], self.latest_conf,
                                           conf[0], local_pose, poses[-1])
            event.update(frame=frame, shared_frame=699, old_ids=list(self.ids), candidate_ids=new_ids,
                         peak_logical_images=3, dual_cache_bytes=old_bytes + self.cache_bytes(),
                         transform_before=dict(scale=float(self.transform[0]),
                             rotation=self.transform[1].tolist(), translation=self.transform[2].tolist()),
                         old_geometry_bytes=sum(t.numel() * t.element_size()
                                                for t in (self.latest_points, self.latest_conf)),
                         candidate_geometry_bytes=sum(t.numel() * t.element_size()
                                                      for t in (points, poses, conf)))
            self.save_bridge(event, evidence)
            if fitted is not None:
                self.transform = compose(self.transform, fitted)
                self.ids, self.images, self.origin = new_ids, new_images, origin
                self.latest_points, self.latest_conf = points[-1].clone(), conf[-1].clone()
                assert not old_storage.intersection(t.untyped_storage().data_ptr()
                    for layer in self.model.cache.values() for t in layer.values())
                del old_cache
                assert all(ref() is None for ref in refs), 'Old KV tensor still referenced'
                event.update(retained_ids=list(self.ids), old_kv_released=True)
            else:
                self.model.cache = old_cache
                del old_cache
                event.update(retained_ids=list(self.ids), old_kv_released=False)
            event['transform_after'] = dict(scale=float(self.transform[0]),
                rotation=self.transform[1].tolist(), translation=self.transform[2].tolist())
            self.events.append(event)
            del evidence, points, poses, conf
            if fitted is not None:
                torch.cuda.synchronize()
                self.log(dict(kind='update_total', frame=frame, seconds=time.perf_counter()-started,
                              cache_bytes=self.cache_bytes(), bank_ids=list(self.ids)))
                return global_pose
        if self.mode == 'native':
            ids, images = self.ids + [frame], self.images + [image]
        else:
            ids, images = [self.ids[0], frame], [self.images[0], image]
            if self.local_keyframe_cap == 3 and len(self.ids) > 1:
                ids.insert(1, self.ids[-1])
                images.insert(1, self.images[-1])
        if self.pin_scale:
            old_points, old_pose, old_conf = self.query_geometry(image, frame)
            # Camera-frame points of the refresh frame, in the outgoing bank's units.
            old_camera = (old_points - old_pose[:3, 3]) @ old_pose[:3, :3] / self.transform[0]
        points, poses, conf, self.origin = self.reconstruct(images, ids, frame, 'rebuild')
        if self.pin_scale:
            new_pose = poses[-1].double()
            new_camera = (points[-1].double() - new_pose[:3, 3]) @ new_pose[:3, :3]
            assert old_camera.shape == new_camera.shape and old_conf.shape == conf[-1].shape
            valid = (old_conf >= old_conf.median()) & (conf[-1] >= conf[-1].median())
            self.rebuild_scale = float((old_camera[valid].norm(dim=-1) /
                                        new_camera[valid].norm(dim=-1)).median())
            assert np.isfinite(self.rebuild_scale) and self.rebuild_scale > 0
        elif self.shared_scale and len(ids) == 3:
            assert ids[1] == self.ids[-1]
            old_pose, new_pose = self.latest_pose.double(), poses[1].double()
            old_camera = (self.latest_points.double() - old_pose[:3, 3]) @ old_pose[:3, :3]
            new_camera = (points[1].double() - new_pose[:3, 3]) @ new_pose[:3, :3]
            valid = (self.latest_conf >= self.latest_conf.median()) & (conf[1] >= conf[1].median())
            self.rebuild_scale *= float((old_camera[valid].norm(dim=-1) /
                                         new_camera[valid].norm(dim=-1)).median())
            assert np.isfinite(self.rebuild_scale) and self.rebuild_scale > 0
        elif self.mode == 'reanchor' and frame != 49:
            # Both pointmaps are in the anchor camera's frame; only scale differs.
            valid = (conf[0] >= conf[0].median()) & (self.anchor_conf >= self.anchor_conf.median())
            self.rebuild_scale = float((self.anchor_points.double()[valid].norm(dim=-1) /
                                        points[0].double()[valid].norm(dim=-1)).median())
            assert np.isfinite(self.rebuild_scale) and self.rebuild_scale > 0
        pin = {}
        if self.pin_rebuilds:
            # Rebuild poses are already in the anchor-camera frame; queries apply origin.
            new_pose = poses[-1].double().clone()
            new_pose[:3, 3] *= self.rebuild_scale
            old_pose = local_pose.double()
            rotation = old_pose[:3, :3] @ new_pose[:3, :3].T
            translation = old_pose[:3, 3] - rotation @ new_pose[:3, 3]
            self.transform = compose(self.transform, (torch.tensor(1., dtype=torch.float64),
                                                      rotation, translation))
            angle = torch.acos(((torch.trace(rotation) - 1) / 2).clamp(-1, 1))
            # Disagreement between the two banks' poses of this frame (diagnostic).
            pin = dict(pin_rotation_deg=float(torch.rad2deg(angle)),
                       pin_position_step=float((old_pose[:3, 3] - new_pose[:3, 3]).norm()))
        self.ids, self.images = ids, images
        self.latest_points, self.latest_conf = points[-1].clone(), conf[-1].clone()
        self.latest_pose = poses[-1].clone()
        if self.events and self.events[-1]['frame'] == frame:
            self.events[-1]['retained_ids'] = list(self.ids)
        torch.cuda.synchronize()
        self.log(dict(kind='update_total', frame=frame, seconds=time.perf_counter()-started,
                      cache_bytes=self.cache_bytes(), bank_ids=list(self.ids),
                      rebuild_scale=self.rebuild_scale, **pin))
        return global_pose
