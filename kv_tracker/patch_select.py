"""Quarter-patch selection of historical keyframe K/V (screen designed 2026-09-28,
object task amended 2026-09-29 after preflight 25993).

Protocol: tools/SEMANTIC_KV_OVERNIGHT_PLAN.md in the parent repository. Keyframes
are admitted by replaying the native schedule and every rebuild stays native and
dense. Afterwards the persistent cache keeps the anchor densely and, for every
later keyframe, its five register tokens plus the patches its policy chooses once
at admission:

- scene: K = ceil(P/4) of all P patches;
- object: K = ceil(P_obj/4) of the P_obj patches touching the SAM mask, chosen
  by the policy, plus BACKGROUND evenly spaced background patches that every
  object arm shares (token_drop.background_keep, as bg16 in job 25906).

Rows are gathered from the post-RoPE cache, so kept keys keep their positions;
nothing is averaged, rescaled or regenerated. Signals use the admitted frame and
already observed history only.
"""
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from kv_tracker.token_drop import background_keep

PATCH = 14
CELLS = 4
SEEDS = (17, 29, 41, 53, 67, 79, 97, 113)
CANDIDATES = tuple(f'K{i}' for i in range(1, 11))
# 'object_dense': every object patch plus the shared background; object task only.
CONTROLS = ('all', 'uniform', 'object_dense') + tuple(f'random{s}' for s in SEEDS)
POLICIES = CONTROLS + CANDIDATES
BACKGROUND = 16
MATCH_FLOOR = .9
CLUSTERS = 8
KMEANS_ITERATIONS = 5
# Pixels whose Sobel (1), structure window (1) or resize interpolation (1) reach
# outside the object mask carry the artificial black-cut edge.
MASK_MARGIN = 3


def allocate(count, capacity, weight):
    """Integer quotas summing to count, proportional to weight.

Largest remainders, clipped to capacity; clipped surplus is redistributed over
groups that still have room, by weight (by room if every such weight is zero).
Ties go to the lower group index.
"""
    capacity = torch.as_tensor(capacity, dtype=torch.long)
    weight = torch.as_tensor(weight, dtype=torch.float64)
    assert capacity.ndim == 1 and weight.shape == capacity.shape
    assert (capacity >= 0).all() and (weight >= 0).all()
    assert 0 <= count <= int(capacity.sum())
    quota = torch.zeros_like(capacity)
    while (remaining := count - int(quota.sum())) > 0:
        room = capacity - quota
        share = torch.where(room > 0, weight, 0.)
        if share.sum() == 0:
            share = (room > 0).double()
        share = share * remaining / share.sum()
        add = share.floor().long()
        extra = remaining - int(add.sum())
        add[(share - add).argsort(descending=True, stable=True)[:extra]] += 1
        quota += torch.minimum(add, room)
    return quota


def top(score, count, eligible):
    """Highest scores among eligible patches; the lower patch index wins ties."""
    candidates = eligible.nonzero().flatten()
    assert count <= len(candidates)
    return candidates[score[candidates].argsort(descending=True, stable=True)][:count]


def stratified(score, groups, count, eligible, weight=None):
    """Exact count: quotas over group IDs, then the top scores inside each group."""
    size = max(int(groups.max()) + 1, 0 if weight is None else len(weight))
    capacity = torch.bincount(groups[eligible], minlength=size)
    quota = allocate(count, capacity, capacity if weight is None else weight)
    picked = [top(score, int(q), eligible & (groups == g)) for g, q in enumerate(quota) if q]
    return torch.cat(picked) if picked else torch.zeros(0, dtype=torch.long)


def even(eligible, count, cell):
    """Cell quotas over eligible patches, then evenly spaced eligible patches
inside each cell in raster order (the spatial-uniform control)."""
    capacity = torch.bincount(cell[eligible], minlength=CELLS * CELLS)
    quota = allocate(count, capacity, capacity)
    picked = []
    for g, q in enumerate(quota.tolist()):
        if q:
            members = (eligible & (cell == g)).nonzero().flatten()
            picked.append(members[torch.linspace(0, len(members) - 1, q).round().long()])
    return torch.cat(picked) if picked else torch.zeros(0, dtype=torch.long)


def cells(grid):
    height, width = grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    return ((y * CELLS // height) * CELLS + x * CELLS // width).flatten()


def patches_of(values):
    height, width = values.shape[:2]
    assert height % PATCH == width % PATCH == 0
    return values.reshape(height // PATCH, PATCH, width // PATCH, PATCH).transpose(
        0, 2, 1, 3).reshape(-1, PATCH * PATCH)


def mask_fraction(mask):
    assert mask.dtype == bool and mask.ndim == 2
    return torch.from_numpy(patches_of(mask).mean(1))


def image_scores(rgb, mask):
    """Per patch: maximum Shi-Tomasi response and mean Sobel magnitude.

Only pixels at least MASK_MARGIN inside the mask count; a patch without such a
pixel scores -inf. Scene masks are all true, and erosion leaves image borders.
Also returns each patch's mask fraction.
"""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert mask.dtype == bool and mask.shape == rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = np.ones((2 * MASK_MARGIN + 1,) * 2, np.uint8)
    valid = patches_of(cv2.erode(mask.astype(np.uint8), kernel).astype(bool))
    corner = patches_of(cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3))
    magnitude = patches_of(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)))
    count = valid.sum(1)
    peak = np.where(valid, corner, -np.inf).max(1)
    mean = np.where(count > 0, np.where(valid, magnitude, 0).sum(1) / np.maximum(count, 1), -np.inf)
    return (torch.from_numpy(peak.astype(np.float64)), torch.from_numpy(mean.astype(np.float64)),
            mask_fraction(mask))


def farthest_points(distance_to, first, count, eligible):
    """Greedy farthest-point selection; `distance_to(i)` gives distances to patch i.
The lower patch index wins ties."""
    picked = [first]
    distance = distance_to(first)
    blocked = ~eligible.to(distance.device)
    blocked[first] = True
    for _ in range(1, count):
        index = int(distance.masked_fill(blocked, -math.inf).argmax())
        assert not blocked[index]
        picked.append(index)
        blocked[index] = True
        distance = torch.minimum(distance, distance_to(index))
    return torch.tensor(picked, dtype=torch.long)


def feature_clusters(features, corner, eligible):
    """Up to eight cosine clusters of the eligible patches: farthest-point
initialisation from the strongest corner, five assignment/update iterations; an
empty cluster keeps its centre. Labels (CPU) are meaningful on eligible patches."""
    count = min(CLUSTERS, int(eligible.sum()))
    first = int(top(corner, 1, eligible)[0])
    centres = features[farthest_points(lambda i: 1 - features @ features[i], first, count, eligible)]
    rows = eligible.nonzero().flatten()
    members = features[rows]
    for _ in range(KMEANS_ITERATIONS):
        labels = (members @ centres.T).argmax(1)
        for c in range(count):
            if (labels == c).any():
                centres[c] = F.normalize(members[labels == c].mean(0), dim=0)
    full = torch.zeros(len(features), dtype=torch.long)
    full[rows] = (members @ centres.T).argmax(1).cpu()
    return full


def label_boundary(labels, grid):
    """Patches with a four-neighbour of a different label."""
    labels = labels.reshape(grid)
    edge = torch.zeros(grid, dtype=torch.bool)
    vertical = labels[1:] != labels[:-1]
    horizontal = labels[:, 1:] != labels[:, :-1]
    edge[1:] |= vertical
    edge[:-1] |= vertical
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    return edge.flatten()


def lexicographic(candidates, keys):
    """Order by the keys, most significant first, each descending; stable."""
    for key in reversed(keys):
        candidates = candidates[key[candidates].argsort(descending=True, stable=True)]
    return candidates


def reciprocal_matches(features, previous):
    """Mutual cosine nearest neighbours at or above MATCH_FLOOR; CPU results."""
    similarity = features @ previous.T
    best, link = similarity.max(1)
    back = similarity.argmax(0)
    matched = (best >= MATCH_FLOOR) & (back[link] == torch.arange(len(features), device=link.device))
    return matched.cpu(), link.cpu(), best.cpu()


def register_attention(block, x, xpos, cache, special):
    """Softmax of the special (register) queries over every key of a cached pass.

Mirrors FlashAttentionRope.forward_w_cache up to the softmax: (heads, special, keys).
"""
    attn = block.attn
    y = block.norm1(x)
    batch, length, channels = y.shape
    assert batch == 1
    qkv = attn.qkv(y).reshape(batch, length, 3, attn.num_heads,
                              channels // attn.num_heads).transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    q, k = attn.q_norm(q).to(v.dtype), attn.k_norm(k).to(v.dtype)
    q, k = attn.rope(q, xpos), attn.rope(k, xpos)
    keys = torch.cat([cache['k'], k], dim=2)
    score = q[0, :, :special].float() @ keys[0].float().transpose(-1, -2) * attn.scale
    return score.softmax(dim=-1)


class PatchSelectCache:
    """One policy's persistent cache over a replayed native admission schedule."""

    def __init__(self, policy, task, schedule):
        assert policy in POLICIES and task in ('object', 'scene')
        assert policy != 'object_dense' or task == 'object'
        assert schedule[0] == 0 and schedule == sorted(set(schedule))
        self.policy = policy
        self.task = task
        self.schedule = list(schedule)
        self.selected = {}
        self.events = []
        self.previous = None  # K7/K8: the latest admission's eligible descriptors
        self.retained = []    # K8: descriptors of every retained patch
        self.hooks = []
        self.frame = None
        self.tokens = None
        self.demand = None
        self.demand_seconds = 0.
        self.needs_features = policy in ('K4', 'K5', 'K6', 'K7', 'K8') and not (
            policy == 'K5' and task == 'object')
        # The bootstrap is the first forward and holds the anchor.
        self.capturing = self.needs_features
        self.measuring = False

    def attach(self, model):
        assert model.patch_size == PATCH and model.patch_start_idx == 5
        assert len(model.decoder) == 36
        self.model = model
        self.hooks.append(model.encoder.register_forward_hook(self.capture))
        if self.policy == 'K9':
            self.hooks.append(model.decoder[-1].register_forward_pre_hook(
                self.read_demand, with_kwargs=True))

    def capture(self, module, args, output):
        if not self.capturing:
            return
        tokens = output['x_norm_patchtokens']
        assert tokens.ndim == 3 and tokens.shape[-1] == 1024
        # Bootstrap duplicates frame 0; a query holds exactly one frame.
        self.tokens = F.normalize(tokens[0].detach().float(), dim=-1)
        self.capturing = False

    def read_demand(self, module, args, kwargs):
        """K9: attention the admitted frame's patches receive from its own register
queries in the final global block, on its causal query pass. Timed."""
        if not self.measuring:
            return
        cache = kwargs['kv_cache']
        assert cache is self.model.cache[len(self.model.decoder) - 1]
        torch.cuda.synchronize()
        started = time.perf_counter()
        special = self.model.patch_start_idx
        x = args[0]
        probability = register_attention(module, x, kwargs['xpos'], cache, special)
        patches = x.shape[1] - special
        self.demand = probability[..., -patches:].mean(dim=(0, 1))
        torch.cuda.synchronize()
        self.demand_seconds = time.perf_counter() - started
        self.measuring = False

    def begin_query(self, frame_id):
        self.frame = frame_id
        if frame_id in self.schedule:
            self.capturing = self.needs_features
            self.measuring = self.policy == 'K9'

    def end_query(self):
        if self.frame in self.schedule:
            assert not self.capturing and not self.measuring
        self.capturing = self.measuring = False

    def eligible(self, mask):
        """Patches a policy chooses among: the object's (touching the mask) or all."""
        fraction = mask_fraction(mask)
        if self.task == 'scene':
            assert mask.all()
            return torch.ones(len(fraction), dtype=torch.bool)
        eligible = fraction > 0
        assert eligible.any()
        return eligible

    def after_rebuild(self, frame_ids, confidence, points, masks, rgb):
        started = time.perf_counter()
        count = len(frame_ids)
        assert confidence.shape[:2] == (1, count) and confidence.shape[-1] == 1
        assert points.shape == (*confidence.shape[:-1], 3)
        assert masks.shape == confidence.shape[1:-1] and rgb.shape == (*masks.shape, 3)
        height, width = masks.shape[1:]
        grid = height // PATCH, width // PATCH
        total = math.prod(grid)
        details = {}
        if not self.selected:
            assert frame_ids == [0, 0]
            new = 0
            picked = torch.arange(total)
            details = dict(budget=total)  # the anchor is dense
            if self.needs_features:
                assert self.tokens.shape == (total, 1024)
                if self.policy in ('K7', 'K8'):
                    rows = self.eligible(masks[0]).nonzero().flatten()
                    self.previous = dict(features=self.tokens[rows.to(self.tokens.device)],
                                         age=torch.zeros(len(rows), dtype=torch.long))
                if self.policy == 'K8':
                    self.retained.append(self.tokens)
        else:
            new = frame_ids[-1]
            assert frame_ids[:-1] == list(self.selected)
            assert new == self.schedule[len(self.selected)] == self.frame
            picked, details = self.choose(new, grid, rgb[-1], masks[-1],
                                          confidence[0, -1, ..., 0], points[0, -1])
            assert len(picked) == len(set(picked.tolist())) == details['budget']
            assert 0 <= int(picked.min()) and int(picked.max()) < total
        self.selected[new] = picked
        self.tokens = None
        self.demand = None
        special = self.model.patch_start_idx
        width_tokens = special + total
        # The anchor (both bootstrap slots) is dense: selected[0] is every patch.
        indices = torch.cat([torch.cat((torch.arange(special), self.selected[frame] + special))
                             + slot * width_tokens for slot, frame in enumerate(frame_ids)])
        dense_bytes = self.cache_bytes()
        assert set(self.model.cache) == set(range(1, len(self.model.decoder), 2))
        for layer in self.model.cache.values():
            for name in ('k', 'v'):
                tensor = layer[name]
                assert tensor.ndim == 4 and tensor.shape[2] == count * width_tokens
                # Full retention keeps the native tensors; otherwise the dense
                # rebuild rows are released when this reference is replaced.
                if len(indices) != tensor.shape[2]:
                    layer[name] = tensor.index_select(2, indices.to(tensor.device))
        self.events.append(dict(frame=new, policy=self.policy, retained_frame_ids=list(self.selected),
            patches=total, selected=picked.tolist(), retained_tokens=len(indices),
            dense_cache_bytes=dense_bytes, query_cache_bytes=self.cache_bytes(),
            state_bytes=self.state_bytes(),
            rebuild_input_bytes=int(masks.nbytes + rgb.nbytes),
            read_demand_seconds=self.demand_seconds if new else 0.,
            selector_host_seconds=time.perf_counter() - started, **details))
        self.demand_seconds = 0.

    def choose(self, frame, grid, rgb, mask, confidence, points):
        """Patch IDs (CPU, sorted) and diagnostics. Encoder features stay on
their device; image scores and index bookkeeping are on the CPU."""
        total = math.prod(grid)
        policy = self.policy
        if policy == 'all':
            return torch.arange(total), dict(budget=total)
        corner, gradient, fraction = image_scores(np.ascontiguousarray(rgb), mask)
        eligible = self.eligible(mask)
        budget = math.ceil(int(eligible.sum()) / 4)
        cell = cells(grid)
        details = dict(corner_scored_patches=int(torch.isfinite(corner).sum()),
                       eligible_patches=int(eligible.sum()), eligible_budget=budget)
        background = torch.zeros(total, dtype=torch.bool)
        if self.task == 'object':
            # Shared by every object arm, so arms differ only inside the object.
            background = background_keep(eligible[None], BACKGROUND)[1][0]

        def complete(picked, count):
            """Fill to count with K2's ranking over eligible patches not yet chosen."""
            free = eligible.clone()
            free[picked] = False
            details['k2_fill'] = details.get('k2_fill', 0) + count - len(picked)
            return torch.cat((picked, stratified(corner, cell, count - len(picked), free)))

        features = None
        if self.needs_features:
            assert self.tokens.shape == (total, 1024)
            features = self.tokens
        if policy == 'object_dense':
            picked = eligible.nonzero().flatten()
        elif policy == 'uniform':
            picked = even(eligible, budget, cell)
        elif policy.startswith('random'):
            generator = torch.Generator().manual_seed(int(policy[6:]) * 1_000_000 + frame)
            rows = eligible.nonzero().flatten()
            picked = rows[torch.randperm(len(rows), generator=generator)[:budget]]
        elif policy == 'K1':
            picked = top(corner, budget, eligible)
        elif policy == 'K2':
            picked = stratified(corner, cell, budget, eligible)
        elif policy == 'K3':
            first = stratified(corner, cell, budget - budget // 2, eligible)
            rest = eligible.clone()
            rest[first] = False
            picked = torch.cat((first, stratified(gradient, cell, budget // 2, rest)))
        elif policy == 'K4':
            # Equal-capacity-adjusted quotas over encoder-feature clusters.
            labels = feature_clusters(features, corner, eligible)
            details['cluster_sizes'] = torch.bincount(labels[eligible], minlength=CLUSTERS).tolist()
            picked = stratified(corner, labels, budget, eligible, [1.] * CLUSTERS)
        elif policy == 'K5':
            if self.task == 'object':
                boundary = eligible & (fraction < 1)  # the mask edge crosses the patch
            else:
                boundary = label_boundary(feature_clusters(features, corner, eligible), grid)
            inner = eligible & ~boundary
            quota = allocate(budget, [int(boundary.sum()), int(inner.sum())], [1., 1.])
            picked = torch.cat((stratified(corner, cell, int(quota[0]), boundary),
                                stratified(corner, cell, int(quota[1]), inner)))
            details['boundary_patches'] = int(boundary.sum())
        elif policy == 'K6':
            first = int(top(corner, 1, eligible)[0])
            picked = farthest_points(lambda i: 1 - features @ features[i], first, budget, eligible)
        elif policy in ('K7', 'K8'):
            rows = eligible.nonzero().flatten()
            current = features[rows.to(features.device)]
            matched_rows, link, best_rows = reciprocal_matches(current, self.previous['features'])
            age_rows = torch.where(matched_rows, self.previous['age'][link] + 1, 0)
            matched = torch.zeros(total, dtype=torch.bool)
            matched[rows] = matched_rows
            best = torch.full((total,), -1.)
            best[rows] = best_rows
            age = torch.zeros(total, dtype=torch.long)
            age[rows] = age_rows
            order = lexicographic(matched.nonzero().flatten(), (age.double(), best.double(), corner))
            persistent = budget if policy == 'K7' else budget - budget // 2
            picked = complete(order[:persistent], persistent)
            details.update(matched=int(matched.sum()), max_track_age=int(age.max()))
            if policy == 'K8':
                history = torch.cat(self.retained)
                novelty = 1 - torch.cat([(chunk @ history.T).max(1).values
                                         for chunk in features.split(256)])
                rest = eligible.clone()
                rest[picked] = False
                picked = torch.cat((picked, top(novelty.double().cpu(), budget // 2, rest)))
            self.previous = dict(features=current, age=age_rows)
        elif policy == 'K9':
            assert self.demand is not None and self.demand.shape == (total,)
            picked = top(self.demand.double().cpu(), budget, eligible)
        else:
            assert policy == 'K10'
            xyz = points[PATCH // 2::PATCH, PATCH // 2::PATCH].detach().double().cpu()
            assert xyz.shape == (*grid, 3)
            pooled = F.avg_pool2d(confidence.detach().float()[None, None], PATCH, PATCH).flatten().cpu()
            finite = torch.isfinite(xyz).all(-1)
            steps = torch.cat(((xyz[1:] - xyz[:-1]).norm(dim=-1)[finite[1:] & finite[:-1]],
                               (xyz[:, 1:] - xyz[:, :-1]).norm(dim=-1)[finite[:, 1:] & finite[:, :-1]]))
            steps = steps[steps > 0]
            spacing = float(steps.median()) if len(steps) else 1.
            xyz = (xyz / spacing).reshape(-1, 3)
            usable = finite.flatten() & eligible
            details.update(nonfinite_points=int((eligible & ~finite.flatten()).sum()),
                           median_spacing=spacing)
            if usable.any():
                first = int(top(pooled.double(), 1, usable)[0])
                count = min(budget, int(usable.sum()))
                picked = complete(farthest_points(lambda i: (xyz - xyz[i]).norm(dim=-1),
                                                  first, count, usable), budget)
            else:
                picked = complete(torch.zeros(0, dtype=torch.long), budget)
        assert not background[picked].any() and eligible[picked].all()
        assert len(picked) == (int(eligible.sum()) if policy == 'object_dense' else budget)
        picked = torch.cat((picked, background.nonzero().flatten()))
        if policy == 'K8':
            self.retained.append(self.tokens[picked.to(self.tokens.device)])
        picked = picked.sort().values
        target = fraction >= .5
        details.update(budget=len(picked), background_kept=int(background.sum()),
                       eligible_selected=int(eligible[picked].sum()),
                       weak_corner_selected=int((corner[picked] <= 1e-6).sum()),
                       target_selected=int(target[picked].sum()), target_patches=int(target.sum()))
        return picked, details

    def cache_bytes(self):
        return sum(t.numel() * t.element_size()
                   for layer in self.model.cache.values() for t in layer.values())

    def state_bytes(self):
        """Selector state kept between admissions, including the kept patch IDs
(full retention needs none)."""
        tensors = ([] if self.policy == 'all' else list(self.selected.values())) + self.retained
        if self.previous is not None:
            tensors += list(self.previous.values())
        return sum(t.numel() * t.element_size() for t in tensors)

    def close(self):
        for hook in self.hooks:
            hook.remove()
