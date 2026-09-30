"""Actual hard sparse actions for task-trained scene policies; Pi3 stays frozen."""
import math

import torch
import torch.nn.functional as F

from kv_tracker.adaptive_tokens import keep_set
from kv_tracker.atd_training import patch_features, scorer_keep
from kv_tracker.scene_routing import project_cache
from kv_tracker.token_drop import forward_kept


def ordered_log_probability(logits, order):
    """Ordered Plackett–Luce likelihood, without a surrogate through hard top-k."""
    assert logits.ndim == order.ndim == 1 and logits.dtype == torch.float32
    assert order.dtype == torch.long and 0 < len(order) <= len(logits)
    assert len(order.unique()) == len(order)
    removed = F.one_hot(order, len(logits)).cumsum(0).bool()
    previous = torch.cat((torch.zeros_like(removed[:1]), removed[:-1]))
    denominator = logits[None].expand(len(order), -1).masked_fill(previous, -torch.inf)
    return (logits[order] - denominator.logsumexp(1)).sum()


def sample_keep(logits, grid, generator):
    """Sample the scored half; fill the other half with existing spatial spread."""
    assert logits.shape == (math.prod(grid),) and logits.dtype == torch.float32
    count = (len(logits) + 1) // 2
    high = count // 2
    assert high > 0 and torch.isfinite(logits).all()
    uniform = torch.rand(logits.shape, device=logits.device, generator=generator)
    gumbel = -(-uniform.clamp_min(torch.finfo(logits.dtype).tiny).log()).log()
    order = (logits.detach() + gumbel).argsort(descending=True, stable=True)[:high]
    dense = torch.zeros_like(logits, dtype=torch.bool)
    dense[order] = True
    keep = keep_set(dense, count, grid)
    return keep, order, ordered_log_probability(logits, order)


class TaskTokenPolicy:
    """Own-cache maps in the raw Pi3 gauge; first-arrival choices survive rebuilds."""
    def __init__(self, mode, intrinsics, distortion=None, scorer=None, sample=False, seed=17):
        assert mode in ('all', 'uniform', 'actor')
        assert (scorer is not None) == (mode == 'actor')
        assert not sample or mode == 'actor'
        self.mode, self.scorer, self.sample = mode, scorer, sample
        self.intrinsics, self.distortion = intrinsics, distortion
        self.generator = torch.Generator(device=intrinsics.device).manual_seed(seed)
        self.saved, self.events, self.history = {}, [], []
        self.inputs, self.logits, self.orders, self.log_probabilities = [], [], [], []
        self.snapshots = []
        self.points = None
        self.bootstrap_depth = None

    def forward(self, model, imgs, masks, frame_ids, cam_only, store_cache, use_cache):
        assert imgs.shape[0] == 1 and imgs.shape[1] == len(frame_ids)
        assert torch.as_tensor(masks).all(), 'Scene-only task policy'
        assert not (store_cache and use_cache)
        assert not torch.is_grad_enabled(), 'Only the actor locally enables autograd'
        if not self.saved:
            model.requires_grad_(False)
        grid = (imgs.shape[-2] // 14, imgs.shape[-1] // 14)
        cells = math.prod(grid)
        for n, frame in enumerate(frame_ids):
            if frame in self.saved:
                continue
            assert not self.saved or max(self.saved) < frame
            score = imgs.new_zeros(cells, dtype=torch.float32)
            order = torch.empty(0, dtype=torch.long, device=imgs.device)
            if self.mode == 'actor':
                with torch.autocast('cuda', enabled=False):
                    if self.points is None:
                        maps = dict(support=torch.zeros(cells, dtype=torch.long, device=imgs.device),
                            front_depth=score, confidence=score, secondary_gap=score)
                    else:
                        assert max(self.cache_ids) < frame
                        maps = project_cache(self.points, self.confidence, self.pose,
                            self.intrinsics, *imgs.shape[-2:], self.distortion)
                    features = patch_features(imgs[0, n].float(), maps).detach()
                    with torch.enable_grad() if self.sample else torch.no_grad():
                        score = self.scorer(features[None])[0]
                        if self.sample:
                            keep, order, log_probability = sample_keep(score, grid, self.generator)
                            self.log_probabilities.append(log_probability)
                        else:
                            keep = scorer_keep(score, grid, frame)
                self.inputs.append(features.cpu())
                self.logits.append(score.detach().cpu())
                self.orders.append(order.cpu())
            else:
                keep = (torch.ones(cells, dtype=torch.bool, device=imgs.device)
                        if self.mode == 'all' else keep_set(
                            torch.zeros(cells, dtype=torch.bool, device=imgs.device),
                            (cells + 1) // 2, grid))
            self.saved[frame] = keep
            self.events.append(dict(frame=frame, selected=keep.nonzero().flatten().tolist(),
                budget=int(keep.sum()), target_budget=cells if self.mode == 'all' else (cells + 1) // 2,
                map_cache_ids=[] if self.points is None else list(self.cache_ids),
                map_pose=None if self.points is None else self.pose.cpu().tolist(),
                sampled_order=order.tolist(), log_probability=None if not self.sample else
                    float(self.log_probabilities[-1].detach())))
        keep = torch.stack([self.saved[f] for f in frame_ids])
        pointers = {i: {k: v.data_ptr() for k, v in layer.items()} for i, layer in model.cache.items()}
        output = forward_kept(model, imgs, keep, cam_only=cam_only,
                              store_cache=store_cache, use_cache=use_cache)
        if store_cache:
            # Dropped point-head output is zero, not evidence. Gather kept samples.
            self.points = output['points'][0, :, 7::14, 7::14].reshape(-1, 3)[keep.flatten()].float().clone()
            self.confidence = output['conf'][0, :, 7::14, 7::14, 0].reshape(-1)[keep.flatten()].sigmoid().float().clone()
            self.cache_ids = list(frame_ids)
            if self.bootstrap_depth is None:
                depth = output['local_points'][0, 0, 7::14, 7::14, 2].flatten()[keep[0]].float()
                positive = depth[depth > 0]
                assert len(positive) and torch.isfinite(positive).all()
                self.bootstrap_depth = float(positive.median())
            assert len(self.points) == int(keep.sum())
            self.snapshots.append(dict(points=self.points.cpu(), confidence=self.confidence.cpu(),
                poses=output['camera_poses'][0].float().cpu().clone(), frame_ids=list(frame_ids),
                keep=keep.cpu()))
            assert all(layer['k'].shape[2] == model.kept_cache['labels'].numel()
                       for layer in model.cache.values())
        if use_cache:
            assert pointers == {i: {k: v.data_ptr() for k, v in layer.items()}
                                for i, layer in model.cache.items()}
        # Clone before main.py rebases/mutates the returned camera tensor.
        self.pose = output['camera_poses'][0, -1].float().clone()
        self.history.append(dict(frame_ids=list(frame_ids), store_cache=store_cache,
            cached_samples=len(self.points), kept=[int(row.sum()) for row in keep]))
        return output
