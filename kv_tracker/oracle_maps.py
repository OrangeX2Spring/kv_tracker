"""Manual local maps with a delayed geometric handoff; no GT in tracking."""
import weakref

import numpy as np
import torch

from .map_handoff import MapHandoff, bridge, compose, transform_pose


class OracleMaps:
    def __init__(self, model, boundaries, log, save_bridge):
        self.model, self.boundaries = model, boundaries
        self.log, self.save_bridge = log, save_bridge
        self.start = 0
        self.tracker = self.new_map(0)
        self.world = self.tracker.transform
        self.connected = True
        self.pending = None
        self.events = []
        self.transforms = [self.world]

    def new_map(self, start):
        return MapHandoff(self.model, 'oracle',
            lambda row: self.log(dict(row, segment_start=start,
                                      global_frame=start + row['frame'])),
            self.save_bridge)

    def step(self, image, frame):
        if frame == 0:
            return self.tracker.bootstrap(image)
        if frame in self.boundaries[1:-1]:
            assert self.pending is None
            # Observe the incoming anchor in the old, unchanged two-image bank.
            old_pose = self.tracker.step(image, frame - self.start)
            points, pose, conf = self.tracker.query_geometry(image, frame - self.start)
            np.testing.assert_allclose(pose.numpy(), old_pose, atol=1e-5, rtol=1e-4)
            self.pending = (self.tracker, self.model.cache, points, conf, self.start)
            self.model.cache = {}
            self.start = frame
            self.tracker = self.new_map(frame)
            return self.tracker.bootstrap(image)
        local = frame - self.start
        pose = self.tracker.step(image, local)
        if local != 49 or self.pending is None:
            return pose
        old, old_cache, old_points, old_conf, old_start = self.pending
        new_cache = self.model.cache
        dual_bytes = self.tracker.cache_bytes()
        self.model.cache = old_cache
        dual_bytes += old.cache_bytes()
        old_pose = torch.from_numpy(old.step(image, frame - old_start))
        self.model.cache = new_cache
        scale, rotation, translation = self.tracker.transform
        new_points = scale * (self.tracker.anchor_points.double() @ rotation.T) + translation
        new_pose = transform_pose(self.tracker.bank_last_pose, self.tracker.transform)
        fitted, event, evidence = bridge(old_points, new_points, old_conf,
                                        self.tracker.anchor_conf, old_pose, new_pose)
        event.update(boundary=self.start, decision_frame=frame, delay_frames=49,
                     old_ids=[old_start + i for i in old.ids],
                     new_ids=[self.start + i for i in self.tracker.ids],
                     dual_cache_bytes=dual_bytes, peak_unique_bank_images=4,
                     prior_connected=self.connected)
        if fitted is not None:
            self.world = compose(self.world, fitted)
        else:
            # Continue measuring local tracking, explicitly start an unconnected
            # component. Never manufacture a continuous score after rejection.
            self.connected = False
            self.world = (torch.tensor(1., dtype=torch.float64),
                          torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        self.transforms.append(self.world)
        refs = [weakref.ref(t) for layer in old_cache.values() for t in layer.values()]
        image_refs = [weakref.ref(x) for x in old.images]
        old_storage = {t.untyped_storage().data_ptr()
                       for layer in old_cache.values() for t in layer.values()}
        assert not old_storage.intersection(t.untyped_storage().data_ptr()
            for layer in new_cache.values() for t in layer.values())
        self.pending = None
        del old, old_cache, old_points, old_conf
        assert all(ref() is None for ref in refs)
        assert all(ref() is None for ref in image_refs)
        event.update(old_kv_released=True, old_images_released=True,
                     globally_connected=self.connected,
                     retirement='connected handoff' if fitted is not None else 'tracking loss; local reset')
        self.save_bridge(event, evidence)
        self.events.append(event)
        return pose
