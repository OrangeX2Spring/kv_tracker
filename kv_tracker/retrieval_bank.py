"""Bounded GPU bank retrieved from a CPU keyframe memory with fixed global poses.

Every admitted keyframe keeps its image, global pose and pointmap on the CPU and is
never re-solved. The GPU bank holds at most `capacity` images: the newest keyframe
plus the stored keyframes that best cover its view. Each rebuild is placed in the
global frame by one robust Sim(3) over all shared members' same-pixel pointmaps
(bridge() on the stacked maps), so no single image sets the gauge or the scale.
Its validation checks are logged as diagnostics, not used to reject a rebuild.
"""
import time

import torch

from .map_handoff import MapHandoff, bridge, transform_pose
from .pi3_utilts import pi3_inference


class RetrievalBank(MapHandoff):
    def __init__(self, model, log, capacity=4, novelty=None, interval=50):
        # novelty(center, pose, memory_poses) -> bool is native's object keyframe rule,
        # applied against the whole memory; without it a keyframe is admitted every
        # `interval` frames (native camera rule, uncapped because memory is on the CPU).
        super().__init__(model, 'fixed', log, None)
        assert capacity >= 2
        self.capacity, self.novelty, self.interval = capacity, novelty, interval
        self.memory, self.bank = [], []
        # Runner interface shared with ReanchorMaps: one map, poses already global.
        self.pending, self.events = None, []
        self.transforms = [self.transform]

    def bootstrap(self, image, mask=None):
        points, poses, conf, self.origin = self.reconstruct([image, image], [0, 0], 0, 'bootstrap')
        # The global frame is this solve: frame 0 is the identity camera and sets the scale.
        self.memory = [dict(frame=0, image=image, pose=poses[0].double(),
                            points=points[0].float(), conf=conf[0].float())]
        self.bank = [0]
        camera = points[0].double()
        front = camera[..., 2] > 0
        # Frustum half-widths (tan) from frame 0's own pixel rays; same camera throughout.
        self.half_fov = (camera[front][:, :2] / camera[front][:, 2:]).abs().amax(0)
        if self.novelty is not None:
            assert mask is not None and mask.shape == camera.shape[:2]
            self.center = camera[mask].mean(0)
        return transform_pose(poses[0], self.transform).float().numpy()

    def retrieve(self, pose):
        """Memory indices, best first; ties prefer newer keyframes."""
        if self.novelty is not None:
            # Objects: angle between viewing directions about the object centre.
            view = self.center - pose[:3, 3]
            score = [float(torch.nn.functional.cosine_similarity(
                view, self.center - m['pose'][:3, 3], dim=0)) for m in self.memory]
        else:
            # Scenes: fraction of a keyframe's confident points inside the current frustum.
            score = []
            for m in self.memory:
                points, conf = m['points'][::8, ::8].double(), m['conf'][::8, ::8]
                camera = (points[conf >= conf.median()] - pose[:3, 3]) @ pose[:3, :3]
                front = camera[:, 2] > 0
                inside = front.clone()
                inside[front] = (camera[front][:, :2] / camera[front][:, 2:]).abs().le(self.half_fov).all(1)
                score.append(float(inside.double().mean()))
        return sorted(range(len(self.memory)), key=lambda i: (-score[i], -i)), score

    def admit(self, image, frame, pose):
        started = time.perf_counter()
        ranked, score = self.retrieve(pose)
        members = ranked[:self.capacity - 1]
        ids = [self.memory[i]['frame'] for i in members] + [frame]
        images = [self.memory[i]['image'] for i in members] + [image]
        points, poses, conf, self.origin = self.reconstruct(images, ids, frame, 'rebuild')
        n, (h, w) = len(members), points.shape[1:3]
        _, event, _ = bridge(torch.cat([self.memory[i]['points'] for i in members]),
                             points[:n].reshape(n * h, w, 3),
                             torch.cat([self.memory[i]['conf'] for i in members]),
                             conf[:n].reshape(n * h, w),
                             self.memory[members[0]]['pose'], poses[0].double())
        scale, rotation, translation = (torch.tensor(event[k], dtype=torch.float64)
                                        for k in ('scale', 'rotation', 'translation'))
        assert scale > 0
        self.transform = (scale, rotation, translation)
        self.memory.append(dict(frame=frame, image=image, pose=transform_pose(poses[-1], self.transform),
                                points=(scale * (points[-1].double() @ rotation.T) + translation).float(),
                                conf=conf[-1].float()))
        self.bank = members + [len(self.memory) - 1]
        torch.cuda.synchronize()
        self.log(dict(kind='update_total', frame=frame, seconds=time.perf_counter() - started,
                      cache_bytes=self.cache_bytes(), bank_ids=ids, memory_size=len(self.memory),
                      scores=[score[i] for i in members], fit_scale=event['scale'],
                      fit_accepted=event['accepted'], fit_checks=event['checks'],
                      validation_median=event['validation_median'],
                      validation_inlier_fraction=event['validation_inlier_fraction'],
                      camera_rotation_deg=event['camera_rotation_deg'],
                      camera_translation_normalized=event['camera_translation_normalized']))

    def step(self, image, frame, mask=None):
        if frame == 0:
            return self.bootstrap(image, mask)
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw = pi3_inference(self.model, [image[None]], self.device, cam_only=True, use_cache=True)
        pose = transform_pose((self.origin @ raw)[0, 0].cpu(), self.transform)
        torch.cuda.synchronize()
        self.log(dict(kind='query', frame=frame, bank_ids=[self.memory[i]['frame'] for i in self.bank],
                      input_images=1, seconds=time.perf_counter() - started,
                      cache_bytes=self.cache_bytes()))
        if self.novelty is None:
            admit = (frame + 1) % self.interval == 0
        else:
            admit = bool(self.novelty(self.center, pose,
                                      torch.stack([m['pose'] for m in self.memory])))
        if admit:
            self.admit(image, frame, pose)
        return pose.float().numpy()

    def finish(self, image, frame):
        """Nothing is pending: every admission is registered when it is made."""
