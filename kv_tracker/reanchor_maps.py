"""Manual segments, reanchored on each segment's first frame; no GT in tracking.

At boundary b the old bank observes b (camera and dense geometry), then its KV and
images are deleted. A new map bootstraps on b and keeps b plus the latest keyframe.
At its first rebuild [b, b+49] the connection commits: rotation and position pin the
new camera of b onto the old camera of b, and scale comes from b's shared pointmap.
Global poses for b..b+48 are therefore a 49-frame delayed output.
"""
import weakref

import torch

from .map_handoff import MapHandoff, bridge, compose


class ReanchorMaps:
    def __init__(self, model, boundaries, log, save_bridge):
        self.model, self.boundaries = model, boundaries
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
            self.save_bridge)

    def step(self, image, frame):
        if frame == 0:
            return self.tracker.bootstrap(image)
        if frame in self.boundaries[1:-1]:
            assert self.pending is None
            old = self.tracker
            # The retiring bank must not rebuild between camera and geometry reads.
            old_pose = torch.from_numpy(old.step(image, frame - self.start, update=False)).double()
            points, pose, conf = old.query_geometry(image, frame - self.start)
            torch.testing.assert_close(pose, old_pose, atol=1e-5, rtol=1e-4)
            old_ids = [self.start + i for i in old.ids]
            refs = [weakref.ref(t) for layer in self.model.cache.values() for t in layer.values()]
            image_refs = [weakref.ref(x) for x in old.images]
            # Delete all previous content before the new map exists.
            self.model.cache = {}
            self.tracker = old = None
            assert all(ref() is None for ref in refs), 'Old KV tensor still referenced'
            assert all(ref() is None for ref in image_refs), 'Old image still referenced'
            self.start = frame
            self.tracker = self.new_map(frame)
            new_pose = self.tracker.bootstrap(image)
            self.pending = dict(old_pose=old_pose, points=points, conf=conf, old_ids=old_ids,
                                new_pose=torch.from_numpy(new_pose).double())
            return new_pose
        local = frame - self.start
        pose = self.tracker.step(image, local)
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
        anchored = (s, r, old[:3, 3] - s * (r @ new[:3, 3]))
        self.world = compose(self.world, anchored)
        self.transforms.append(self.world)
        event.update(boundary=self.start, decision_frame=frame, delay_frames=frame - self.start,
                     end_of_video=frame - self.start < 49,
                     old_ids=pending['old_ids'], new_ids=[self.start + i for i in self.tracker.ids],
                     anchored_rotation=r.tolist(), anchored_translation=anchored[2].tolist(),
                     old_kv_released=True, old_images_released=True,
                     connection='pose-anchored at b, point-fit scale')
        self.save_bridge(event, evidence)
        self.events.append(event)
