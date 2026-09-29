"""Pre-encoder ATD and offline, dense-teacher diagnostics for gate 1.

All selections are made at first arrival and reused at rebuilds. Teacher state is
separate from the sparse tracker state. No training occurs in this module.
"""
import json
import math
import time

import torch
import torch.nn.functional as F

from kv_tracker.patch_select import cells, even, SEEDS
from kv_tracker.token_drop import PATCH, background_keep, forward_kept, patch_keep

ARMS = ('all', 'heuristic', 'oracle', 'uniform') + tuple(f'random{s}' for s in SEEDS) + ('come',)


def keep_set(dense, count, grid):
    """Keep the dense region and spread original samples up to an exact count."""
    assert dense.dtype == torch.bool and dense.shape == (math.prod(grid),)
    assert int(dense.sum()) <= count <= dense.numel()
    result = dense.clone()
    remaining = count - int(dense.sum())
    if remaining:
        result[even(~dense.cpu(), remaining, cells(grid)).to(dense.device)] = True
    assert int(result.sum()) == count
    return result


def energy(rgb):
    """Mean squared adjacent-pixel RGB difference inside each 14x14 patch."""
    assert rgb.ndim == 3 and rgb.shape[0] == 3
    c, h, w = rgb.shape
    patches = rgb.reshape(c, h // PATCH, PATCH, w // PATCH, PATCH).permute(1, 3, 0, 2, 4)
    return ((patches[..., 1:, :] - patches[..., :-1, :]).square().mean((2, 3, 4)) +
            (patches[..., 1:] - patches[..., :-1]).square().mean((2, 3, 4))).flatten() / 2


def select(score, object_patches, task, policy, grid, frame):
    p = math.prod(grid)
    assert score.shape == object_patches.shape == (p,)
    assert torch.isfinite(score).all()
    dense = object_patches.clone() if task == 'object' else torch.zeros_like(object_patches)
    count = min(p, int(dense.sum()) + 64) if task == 'object' else math.ceil(.5 * p)
    if policy == 'all':
        return torch.ones_like(dense)
    if policy == 'heuristic' and task == 'object':
        return background_keep(dense[None], 64)[0][0]
    if policy == 'uniform':
        return keep_set(dense, count, grid)
    eligible = (~dense).nonzero().flatten()
    remaining = count - int(dense.sum())
    if policy.startswith('random'):
        generator = torch.Generator().manual_seed(int(policy[6:]) + frame * 1009)
        chosen = eligible[torch.randperm(len(eligible), generator=generator).to(score.device)[:remaining]]
    else:
        assert policy in ('heuristic', 'oracle')
        # The heuristic reserves half its scene budget for spread context.
        high = remaining // 2 if policy == 'heuristic' else remaining
        chosen = eligible[score[eligible].argsort(descending=True, stable=True)[:high]]
    dense[chosen] = True
    return keep_set(dense, count, grid)


def merge_groups(confidence, object_patches, task, grid):
    """Low-confidence 3x3 merges; never drop source patches to force a budget.

    A partial final group reaches the target if feasible. Object patches remain
    singleton. When the minimum possible count is above target, report it.
    """
    p = math.prod(grid)
    eligible = ~object_patches if task == 'object' else torch.ones_like(object_patches)
    target = min(p, int(object_patches.sum()) + 64) if task == 'object' else math.ceil(.5 * p)
    ids = torch.arange(p, device=confidence.device).reshape(grid)
    candidates = []
    for y in range(0, grid[0], 3):
        for x in range(0, grid[1], 3):
            members = ids[y:y + 3, x:x + 3].flatten()
            members = members[eligible[members]]
            if len(members) > 1:
                candidates.append((float(confidence[members].mean()), int(members[0]), members))
    representatives = torch.arange(p, device=confidence.device)
    count = p
    for _, _, members in sorted(candidates, key=lambda item: item[:2]):
        if count == target:
            break
        members = members[:min(len(members), count - target + 1)]
        representatives[members] = members[0]
        count -= len(members) - 1
    unique, mapping = representatives.unique(sorted=True, return_inverse=True)
    keep = torch.zeros(p, dtype=torch.bool, device=confidence.device)
    keep[unique] = True
    assert count == int(keep.sum()) and count >= target
    assert task != 'object' or keep[object_patches].all()
    return keep, mapping


def sensitivity(model, imgs, use_cache):
    """Six local se(3) derivatives per frame, summed in absolute value.

    Log(T0^-1 T) has this differential at T=T0: body translation and the
    skew part of R0^T R. The detached reference avoids log singularities at 0.
    Translation is in Pi3 units, rotation in radians (no fitted normalization).
    """
    from kv_tracker.oracle_rope import OracleRoPE, cuRoPE2D

    if not isinstance(model.rope, OracleRoPE):
        original = model.rope
        assert isinstance(original, cuRoPE2D)
        replacement = OracleRoPE(freq=original.base, F0=original.F0)
        # Pi3 shares this module across decoder and head attention layers.
        for module in list(model.modules()):
            if getattr(module, 'rope', None) is original:
                module.rope = replacement
    n, p = imgs.shape[1], (imgs.shape[-2] // PATCH) * (imgs.shape[-1] // PATCH)
    with torch.enable_grad():
        gates = torch.ones(n, p, device=imgs.device, requires_grad=True)
        result = forward_kept(model, imgs, torch.ones_like(gates, dtype=torch.bool),
                              cam_only=True, use_cache=use_cache, gates=gates)
        pose = result['camera_poses'][0].float()
        r0 = pose[:, :3, :3].detach().transpose(-1, -2)
        r = r0 @ pose[:, :3, :3]
        t = (r0 @ (pose[:, :3, 3] - pose[:, :3, 3].detach())[..., None]).squeeze(-1)
        omega = torch.stack((r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0],
                             r[:, 1, 0] - r[:, 0, 1]), -1) / 2
        xi = torch.cat((t, omega), -1)
        score = torch.zeros_like(gates)
        for frame in range(n):
            for component in range(6):
                gradient, = torch.autograd.grad(xi[frame, component], gates,
                    retain_graph=not (frame == n - 1 and component == 5))
                score[frame] += gradient[frame].abs()
    assert torch.isfinite(score).all()
    return score.detach()


class AdaptiveTokens:
    def __init__(self, policy, task, log):
        assert policy in ARMS and task in ('object', 'scene')
        self.policy, self.task, self.log = policy, task, log
        self.saved = {}
        self.teacher_cache = {}
        self.teacher_metadata = None
        self.events = []
        self.teacher_seconds = 0.
        self.selection_seconds = 0.
        self.student_seconds = 0.
        self.frozen = False

    def forward(self, model, imgs, masks, frame_ids, cam_only, store_cache, use_cache):
        assert imgs.shape[0] == 1 and len(frame_ids) == imgs.shape[1]
        assert not (store_cache and use_cache)
        if not self.frozen:
            model.requires_grad_(False)
            self.frozen = True
        grid = (imgs.shape[-2] // PATCH, imgs.shape[-1] // PATCH)
        p = math.prod(grid)
        masks = torch.as_tensor(masks, device=imgs.device, dtype=torch.bool)
        object_patches = patch_keep(masks)
        torch.cuda.synchronize()
        started = time.perf_counter()
        old_cache = model.cache
        old_metadata = getattr(model, 'kept_cache', None)
        teacher = self.policy in ('oracle', 'come')
        scores = {}
        if teacher:
            model.cache = self.teacher_cache
            model.kept_cache = self.teacher_metadata
            # An arrival query uses only previously admitted, dense teacher frames.
            for n, frame in enumerate(frame_ids):
                if frame not in self.saved and frame not in scores:
                    one = imgs[:, n:n + 1]
                    if self.policy == 'oracle':
                        scores[frame] = sensitivity(model, one, use_cache)[0]
                    else:
                        dense = forward_kept(model, one, torch.ones(1, p, device=imgs.device,
                            dtype=torch.bool), use_cache=use_cache)
                        confidence = dense['conf'][0, 0, ..., 0].sigmoid()
                        scores[frame] = F.avg_pool2d(confidence[None, None], PATCH, PATCH).flatten()
            if store_cache:
                forward_kept(model, imgs, torch.ones(len(frame_ids), p, device=imgs.device,
                    dtype=torch.bool), cam_only=True, store_cache=True)
                self.teacher_cache = model.cache
                self.teacher_metadata = model.kept_cache
            model.cache = old_cache
            model.kept_cache = old_metadata
        torch.cuda.synchronize()
        self.teacher_seconds += time.perf_counter() - started
        started = time.perf_counter()
        for n, frame in enumerate(frame_ids):
            if frame in self.saved:
                continue
            gradient = energy(imgs[0, n])
            score = scores[frame] if teacher else gradient
            if self.policy == 'come':
                keep, groups = merge_groups(score, object_patches[n], self.task, grid)
            else:
                keep = select(score, object_patches[n], self.task, self.policy, grid, frame)
                groups = None
            self.saved[frame] = (keep, groups)
            event = dict(frame=frame, grid=list(grid), patches=p, object_patches=int(object_patches[n].sum()),
                budget=int(keep.sum()), target_budget=(p if self.policy == 'all' else
                    min(p, int(object_patches[n].sum()) + 64) if self.task == 'object' else math.ceil(.5 * p)),
                selected=keep.nonzero().flatten().tolist(),
                original_tokens=groups is None, representative_positions=True, energy_quantiles=torch.quantile(gradient.float(),
                    torch.tensor([0., .1, .25, .5, .75, .9, 1.], device=imgs.device)).tolist(),
                homogeneous_share={str(t): float((gradient <= t).float().mean())
                                   for t in (0., 1e-5, 1e-4, 1e-3, 1e-2)},
                score=score.float().tolist(), groups=None if groups is None else groups.tolist())
            self.events.append(event)
        keep = torch.stack([self.saved[f][0] for f in frame_ids])
        groups = [self.saved[f][1] for f in frame_ids] if self.policy == 'come' else None
        torch.cuda.synchronize()
        self.selection_seconds += time.perf_counter() - started
        cache_before = {i: (v['k'].data_ptr(), v['v'].data_ptr()) for i, v in model.cache.items()}
        expected_keys = sum(int(row.sum()) + model.patch_start_idx for row in keep)
        if use_cache:
            cached = model.kept_cache['labels'].numel()
            assert all(v['k'].shape[2] == cached for v in model.cache.values())
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = forward_kept(model, imgs, keep, cam_only=cam_only, store_cache=store_cache,
                              use_cache=use_cache, groups=groups)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        self.student_seconds += seconds
        if store_cache:
            assert all(v['k'].shape[2] == expected_keys for v in model.cache.values())
            assert model.kept_cache['labels'].numel() == expected_keys
        if use_cache:
            assert cache_before == {i: (v['k'].data_ptr(), v['v'].data_ptr()) for i, v in model.cache.items()}
        self.log.write(json.dumps(dict(kind='rebuild' if store_cache else 'query',
            frames=list(frame_ids), kept=keep.sum(1).tolist(), student_seconds=seconds,
            cache_tokens=model.kept_cache['labels'].numel(),
            cache_bytes=sum(t.numel() * t.element_size() for v in model.cache.values() for t in v.values()))) + '\n')
        return output
