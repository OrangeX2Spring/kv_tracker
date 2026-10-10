"""Manual segments, reanchored on each segment's first frame; no GT in tracking.

At boundary b the old bank observes b (camera and dense geometry), then its KV and
images are deleted. A new map bootstraps on b and keeps b plus the latest keyframe.
At its first rebuild [b, b+49] the connection commits: rotation and position pin the
new camera of b onto the old camera of b, and scale comes from b's shared pointmap.
Global poses for b..b+48 are therefore a 49-frame delayed output.

overlap_stride keeps the old map (its KV swapped out of the model) until that first
rebuild and also locates every overlap_stride-th frame b, b+s, ... < b+49 with it.
The connection keeps rotation and position pinned at b but takes scale from the
camera displacements of those shared frames in both maps, not from one pointmap;
the point-fit scale is still computed and logged. Old KV and images are released
at the connection instead of at b.

two_way connects on two frames that each map actually covers: b (old map by query,
new map by its first rebuild, as before) and the old map's latest keyframe L just
before b (old map by its own rebuild, new map by one dense query at b+49 of a CPU
copy of L's image). One robust, confidence-weighted Sim(3) over both frames' points
supplies the scale (VGGT-Diff-style: weight by confidence, require the same points
to agree across views); rotation and position stay pinned at b. The two frames use
query-versus-rebuild in opposite directions. Per-frame scales are logged.
"""
import weakref

import torch

from .map_handoff import MapHandoff, bridge, compose


def weighted_similarity(source, target, weight):
    """Robust confidence-weighted Sim(3), source into target; float64 CPU points."""
    assert source.ndim == 2 and source.shape == target.shape and source.shape[1] == 3
    assert source.dtype == target.dtype == weight.dtype == torch.float64 and weight.shape == source.shape[:1]
    keep = torch.isfinite(source).all(1) & torch.isfinite(target).all(1) & (weight > 0)
    # Zero weight must not turn a non-finite point into NaN sums.
    source, target = torch.where(keep[:, None], source, 0.), torch.where(keep[:, None], target, 0.)
    extent = (target[keep] - target[keep].mean(0)).square().sum(1).mean().sqrt()
    for step in range(4):
        w = torch.where(keep, weight, 0.)
        w = w / w.sum()
        x_mean, y_mean = (w[:, None] * source).sum(0), (w[:, None] * target).sum(0)
        x, y = source - x_mean, target - y_mean
        u, d, vh = torch.linalg.svd((w[:, None] * y).T @ x)
        sign = torch.ones(3, dtype=torch.float64)
        sign[-1] = torch.linalg.det(u) * torch.linalg.det(vh)
        rotation = (u * sign) @ vh
        scale = (d * sign).sum() / (w * x.square().sum(1)).sum()
        if step < 3:  # three rejection rounds, then the final fit on the kept points
            residual = (scale * (x @ rotation.T) - y).norm(dim=1)
            keep = keep & (residual <= max(float(3 * residual[keep].median()), float(.01 * extent)))
    assert int(keep.sum()) >= 128 and scale > 0
    return float(scale), keep


class ReanchorMaps:
    def __init__(self, model, boundaries, log, save_bridge, query_executor=None,
                 local_keyframe_cap=2, pin_rebuilds=False, pin_scale=False, shared_scale=False,
                 fuse_scale=False, novelty_refresh=None, overlap_stride=None, two_way=False):
        self.model, self.boundaries = model, boundaries
        self.query_executor = query_executor
        self.local_keyframe_cap = local_keyframe_cap
        # Pinning changes the map transform after the anchor geometry used by
        # connect(); only the single-map (no retirement) configuration uses it.
        assert not pin_rebuilds or len(boundaries) == 2
        assert novelty_refresh is None or len(boundaries) == 2
        # Two maps share the model's single cache slot by swapping; no query executor.
        assert overlap_stride is None or (query_executor is None and not pin_rebuilds)
        self.overlap_stride = overlap_stride
        assert not two_way or (overlap_stride is None and not pin_rebuilds)
        self.two_way = two_way
        self.pin_rebuilds, self.pin_scale, self.shared_scale = pin_rebuilds, pin_scale, shared_scale
        self.fuse_scale, self.novelty_refresh = fuse_scale, novelty_refresh
        self.log, self.save_bridge = log, save_bridge
        self.start = 0
        self.tracker = self.new_map(0)
        self.world = (torch.tensor(1., dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                      torch.zeros(3, dtype=torch.float64))
        self.transforms = [self.world]
        self.pending = None
        self.events = []

    def new_map(self, start):
        return MapHandoff(self.model, 'reanchor',
            lambda row: self.log(dict(row, segment_start=start,
                                      global_frame=start + row['frame'])),
            self.save_bridge, query_executor=self.query_executor,
            local_keyframe_cap=self.local_keyframe_cap, pin_rebuilds=self.pin_rebuilds,
            pin_scale=self.pin_scale, shared_scale=self.shared_scale, fuse_scale=self.fuse_scale,
            novelty_refresh=self.novelty_refresh)

    def step(self, image, frame, mask=None):
        if frame == 0:
            return self.tracker.bootstrap(image, mask)
        if frame in self.boundaries[1:-1]:
            assert self.pending is None
            old = self.tracker
            # The retiring bank must not rebuild between camera and geometry reads.
            old_pose = torch.from_numpy(old.step(image, frame - self.start,
                                                 update=False, dense_query=True)).double()
            points, pose, conf = old.query_geometry(image, frame - self.start)
            torch.testing.assert_close(pose, old_pose, atol=1e-5, rtol=1e-4)
            old_ids = [self.start + i for i in old.ids]
            latest = None
            if self.two_way:
                # L in the old map's output frame, from its own rebuild; image copied to CPU.
                assert len(old.ids) >= 2
                s0, r0, t0 = old.transform
                latest = dict(frame=old_ids[-1], image=old.images[-1].copy(), conf=old.latest_conf.clone(),
                              points=s0 * ((old.rebuild_scale * old.latest_points.double()) @ r0.T) + t0)
            if self.query_executor is not None:
                self.query_executor.reset()
            overlap = None
            if self.overlap_stride is not None:
                # Keep the old map and its KV aside until the connection.
                overlap = dict(map=old, cache=self.model.cache, start=self.start)
            else:
                refs = [weakref.ref(t) for layer in self.model.cache.values() for t in layer.values()]
                image_refs = [weakref.ref(x) for x in old.images]
            # Delete all previous content before the new map exists.
            self.model.cache = {}
            self.tracker = old = None
            if overlap is None:
                assert all(ref() is None for ref in refs), 'Old KV tensor still referenced'
                assert all(ref() is None for ref in image_refs), 'Old image still referenced'
            self.start = frame
            self.tracker = self.new_map(frame)
            new_pose = self.tracker.bootstrap(image)
            self.pending = dict(old_pose=old_pose, points=points, conf=conf, old_ids=old_ids,
                                new_pose=torch.from_numpy(new_pose).double(), overlap=overlap,
                                latest=latest)
            if overlap is not None:
                overlap['pairs'] = [(old_pose, self.pending['new_pose'])]
            return new_pose
        local = frame - self.start
        pose = self.tracker.step(image, local, mask=mask)
        overlap = self.pending and self.pending['overlap']
        if overlap and local < 49 and local % self.overlap_stride == 0:
            new_cache, self.model.cache = self.model.cache, overlap['cache']
            old_pose = overlap['map'].step(image, frame - overlap['start'], update=False)
            self.model.cache = new_cache
            overlap['pairs'].append((torch.from_numpy(old_pose).double(), torch.from_numpy(pose).double()))
        if local != 49 or self.pending is None:
            return pose
        self.connect(frame)
        return pose

    def finish(self, image, frame):
        """Connect an unfinished final map at EOF, without dropping its poses."""
        if self.pending is None:
            return
        local = frame - self.start
        assert 0 <= local < 49
        images, ids = [self.tracker.images[0], image], [0, local]
        points, _, conf, self.tracker.origin = self.tracker.reconstruct(
            images, ids, local, 'rebuild')
        self.tracker.images, self.tracker.ids = images, ids
        self.tracker.anchor_points, self.tracker.anchor_conf = points[0], conf[0]
        self.connect(frame)

    def connect(self, frame):
        pending, self.pending = self.pending, None
        scale, rotation, translation = self.tracker.transform
        new_points = scale * (self.tracker.anchor_points.double() @ rotation.T) + translation
        # Point fit supplies scale; its checks are diagnostics, not a gate.
        _, event, evidence = bridge(pending['points'], new_points, pending['conf'],
                                    self.tracker.anchor_conf, pending['old_pose'],
                                    pending['new_pose'])
        s = torch.tensor(event['scale'], dtype=torch.float64)
        assert s > 0
        old, new = pending['old_pose'], pending['new_pose']
        r = old[:3, :3] @ new[:3, :3].T
        overlap = pending['overlap']
        if overlap is not None:
            # Least-squares scale of the new map's displacements from b onto the old map's,
            # with the rotation fixed by the pose anchor at b.
            old_c = torch.stack([a[:3, 3] for a, _ in overlap['pairs']])
            new_c = torch.stack([b[:3, 3] for _, b in overlap['pairs']])
            moved_old, moved_new = old_c[1:] - old_c[:1], (new_c[1:] - new_c[:1]) @ r.T
            assert len(moved_new) > 0 and float(moved_new.square().sum()) > 0
            s = (moved_old * moved_new).sum() / moved_new.square().sum()
            assert s > 0
            event.update(point_fit_scale=event['scale'], overlap_scale=float(s),
                         overlap_pairs=len(overlap['pairs']),
                         overlap_old_displacement=float(moved_old.norm(dim=1).max()))
            refs = [weakref.ref(t) for layer in overlap['cache'].values() for t in layer.values()]
            image_refs = [weakref.ref(x) for x in overlap['map'].images]
            overlap.clear()
            del overlap
            pending['overlap'] = None
            assert all(ref() is None for ref in refs), 'Old KV tensor still referenced'
            assert all(ref() is None for ref in image_refs), 'Old image still referenced'
        latest = pending['latest']
        if latest is not None:
            new_latest, _, new_latest_conf = self.tracker.query_geometry(latest['image'], frame - self.start)
            frames = ((pending['points'], new_points, pending['conf'], self.tracker.anchor_conf),
                      (latest['points'], new_latest, latest['conf'], new_latest_conf))
            sources, targets, weights, per_frame = [], [], [], []
            for old_points, frame_points, old_conf, new_conf in frames:
                assert old_points.shape == frame_points.shape and old_conf.shape == new_conf.shape
                # Same stride-4 pixel samples as bridge(); weight = joint confidence.
                sampled = [t[::4, ::4].reshape(-1, *t.shape[2:]).double()
                           for t in (frame_points, old_points, old_conf * new_conf)]
                per_frame.append(weighted_similarity(*sampled)[0])
                sources.append(sampled[0]), targets.append(sampled[1]), weights.append(sampled[2])
            two_way_scale, keep = weighted_similarity(torch.cat(sources), torch.cat(targets), torch.cat(weights))
            s = torch.tensor(two_way_scale, dtype=torch.float64)
            event.update(point_fit_scale=event['scale'], two_way_scale=two_way_scale,
                         frame_b_scale=per_frame[0], latest_frame=latest['frame'],
                         latest_frame_scale=per_frame[1], two_way_inliers=int(keep.sum()))
        anchored = (s, r, old[:3, 3] - s * (r @ new[:3, 3]))
        self.world = compose(self.world, anchored)
        self.transforms.append(self.world)
        event.update(boundary=self.start, decision_frame=frame, delay_frames=frame - self.start,
                     end_of_video=frame - self.start < 49,
                     old_ids=pending['old_ids'], new_ids=[self.start + i for i in self.tracker.ids],
                     anchored_rotation=r.tolist(), anchored_translation=anchored[2].tolist(),
                     old_kv_released=True, old_images_released=True,
                     connection='pose-anchored at b, ' + ('overlap-displacement scale'
                                if self.overlap_stride is not None else 'two-way two-frame weighted scale'
                                if self.two_way else 'point-fit scale'))
        self.save_bridge(event, evidence)
        self.events.append(event)
