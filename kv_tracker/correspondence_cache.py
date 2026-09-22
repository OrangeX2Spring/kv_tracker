"""Experimental half-patch query cache with spatially distributed track support.

Both variants use reciprocal encoder matches checked in one native rebuild's
coordinate system. The semantic variant additionally constrains matches and
spatial quotas by the supplied target mask. No merging or extra model forward.
"""
import math
import time

import torch
import torch.nn.functional as F

from kv_tracker.combined_cache import CombinedCache


def match_patches(features, points, history_features, history_points, radius,
                  labels=None, history_labels=None):
    """Reciprocal cosine matches, within a geometry radius, to surviving tokens.

Features are normalized CPU float32. Missing/ambiguous matches return -1; they
are expected when views do not overlap. Chunking bounds temporary CPU storage.
"""
    assert features.ndim == history_features.ndim == 2
    assert features.dtype == history_features.dtype == torch.float32
    assert features.device.type == history_features.device.type == 'cpu'
    assert points.shape == (len(features), 3)
    assert history_points.shape == (len(history_features), 3)
    assert (labels is None) == (history_labels is None)
    if labels is not None:
        assert labels.shape == (len(features),) and labels.dtype == torch.bool
        assert history_labels.shape == (len(history_features),)
        assert history_labels.dtype == torch.bool
    best = torch.full((len(features),), -float('inf'))
    links = torch.full((len(features),), -1, dtype=torch.long)
    reverse_score = torch.full((len(history_features),), -float('inf'))
    reverse = torch.full((len(history_features),), -1, dtype=torch.long)
    for start in range(0, len(features), 128):
        stop = min(start + 128, len(features))
        similarity = features[start:stop] @ history_features.T
        valid = (torch.cdist(points[start:stop], history_points) <= radius) & (radius > 0)
        if labels is not None:
            valid &= labels[start:stop, None] == history_labels[None]
        similarity.masked_fill_(~valid, -float('inf'))
        best[start:stop], links[start:stop] = similarity.max(1)
        score, index = similarity.max(0)
        improved = score > reverse_score
        reverse[improved] = index[improved] + start
        reverse_score = torch.maximum(reverse_score, score)
    valid = (best >= 0.9) & (reverse[links] == torch.arange(len(features)))
    links[~valid] = -1
    return links, best.masked_fill(~valid, -1.)


def spatial_pick(grid, count, links, scores, track_ids, labels=None):
    """Exact budget: proportional 4x4 cell quotas, then tracks and uniform fill.

With labels, each cell is split into target/non-target strata. Largest remainder
rounding preserves the total count; there is no extra semantic token allowance.
"""
    height, width = grid
    total = height * width
    assert 0 < count <= total and links.shape == scores.shape == (total,)
    assert track_ids.shape == (total,) and track_ids.dtype == torch.long
    if count == total:
        return torch.arange(total)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    groups = ((y * 4 // height) * 4 + x * 4 // width).flatten()
    if labels is not None:
        assert labels.shape == (total,) and labels.dtype == torch.bool
        groups = 2 * groups + labels.long()
    unique, sizes = torch.unique(groups, sorted=True, return_counts=True)
    quotas = sizes * count // total
    remainder = sizes * count % total
    extra = count - int(quotas.sum())
    quotas[remainder.argsort(descending=True, stable=True)[:extra]] += 1
    picked = []
    used_tracks = set()
    for group, quota in zip(unique.tolist(), quotas.tolist()):
        candidates = (groups == group).nonzero().flatten()
        chosen = []
        order = candidates[scores[candidates].argsort(descending=True, stable=True)]
        for index in order.tolist():
            if len(chosen) == quota:
                break
            track = int(track_ids[index])
            if links[index] >= 0 and track not in used_tracks:
                chosen.append(index)
                used_tracks.add(track)
        remaining = candidates[~torch.isin(candidates, torch.tensor(chosen, dtype=torch.long))]
        missing = quota - len(chosen)
        if missing:
            offsets = torch.linspace(0, len(remaining) - 1, missing).round().long()
            chosen.extend(remaining[offsets].tolist())
        picked.extend(chosen)
    assert len(picked) == len(set(picked)) == count
    return torch.tensor(sorted(picked), dtype=torch.long)


class CorrespondenceCache(CombinedCache):
    supports_object_mode = True

    def __init__(self, policy, budget=32, interval=30):
        assert policy in ('dense', 'uniform', 'correspondence', 'semantic_correspondence')
        super().__init__(budget, interval, 1. if policy == 'dense' else 0.5)
        self.policy = policy

    def begin_query(self, frame_id):
        self.capture_enabled = frame_id % self.interval == 0 and len(self.records) < self.budget

    def select(self, frame_id, cached_ids):
        assert sorted(set(cached_ids)) == list(self.records)
        assert max(cached_ids) < frame_id
        if frame_id % self.interval or len(self.records) >= self.budget:
            return False
        self.keep_frame_ids = list(self.records)
        self.pending = frame_id
        self.capture_enabled = False
        return True

    def after_rebuild(self, frame_ids, confidence, *, points, masks):
        started = time.perf_counter()
        assert confidence.ndim == 5 and confidence.shape[:2] == (1, len(frame_ids))
        assert points.shape == (*confidence.shape[:-1], 3)
        assert masks.shape == tuple(confidence.shape[1:-1]) and frame_ids[0] == 0
        height, width = confidence.shape[2:4]
        assert height % 14 == width % 14 == 0
        grid = height // 14, width // 14
        total = math.prod(grid)
        assert self.tokens.shape == (total, 1024)
        labels = F.avg_pool2d(torch.as_tensor(masks[-1]).float()[None, None], 14, 14).flatten() >= .5
        features = F.normalize(self.tokens, dim=-1)
        links = torch.full((total,), -1, dtype=torch.long)
        scores = torch.full((total,), -1.)
        track_ids = frame_ids[-1] * total + torch.arange(total)
        link_rows = []
        radius = None
        if not self.records:
            assert frame_ids == [0, 0]
            picked = torch.arange(total)
        else:
            assert frame_ids == self.keep_frame_ids + [self.pending]
            if self.policy in ('correspondence', 'semantic_correspondence'):
                # Rebuild updates every pointmap together; never mix gauges from
                # separately predicted frames. Old patch choices remain fixed.
                xyz = points[0, :, 7::14, 7::14].detach().float().cpu()
                current = xyz[-1]
                spacing = torch.cat(((current[1:] - current[:-1]).norm(dim=-1).flatten(),
                                     (current[:, 1:] - current[:, :-1]).norm(dim=-1).flatten()))
                spacing = spacing[spacing > 0]
                radius = 0. if not len(spacing) else 2 * float(spacing.median())
                history_points = torch.cat([xyz[slot].reshape(-1, 3)[self.records[i]['indices']]
                                            for slot, i in enumerate(self.keep_frame_ids)])
                history_features = torch.cat([r['patches'] for r in self.records.values()])
                history_labels = torch.cat([r['labels'] for r in self.records.values()])
                history_tracks = torch.cat([r['track_ids'] for r in self.records.values()])
                semantic = self.policy == 'semantic_correspondence'
                links, scores = match_patches(features, current.reshape(-1, 3),
                    history_features, history_points, radius,
                    labels if semantic else None, history_labels if semantic else None)
                matched = links >= 0
                track_ids[matched] = history_tracks[links[matched]]
                history_ids = [(i, int(p)) for i, r in self.records.items() for p in r['indices']]
                link_rows = [dict(patch=int(p), previous_frame=history_ids[int(links[p])][0],
                                  previous_patch=history_ids[int(links[p])][1],
                                  track=int(track_ids[p])) for p in matched.nonzero().flatten()]
            picked = spatial_pick(grid, math.ceil(total * self.patch_fraction), links, scores,
                                  track_ids, labels if self.policy == 'semantic_correspondence' else None)
        self.records[frame_ids[-1]] = dict(patches=features[picked], indices=picked,
                                          labels=labels[picked], track_ids=track_ids[picked])
        special = self.model.patch_start_idx
        token_count = total + special
        indices = torch.cat([torch.cat((torch.arange(special), self.records[i]['indices'] + special))
                             + slot * token_count for slot, i in enumerate(frame_ids)])
        dense_bytes = self.cache_bytes()
        assert set(self.model.cache) == set(range(1, len(self.model.decoder), 2))
        for layer in self.model.cache.values():
            for name in ('k', 'v'):
                tensor = layer[name]
                assert tensor.ndim == 4 and tensor.shape[2] == len(frame_ids) * token_count
                if len(indices) != tensor.shape[2]:
                    layer[name] = tensor.index_select(2, indices.to(tensor.device))
        self.events.append(dict(frame=frame_ids[-1], retained_frame_ids=list(frame_ids),
            evicted=[], patch_indices={str(i): r['indices'].tolist() for i, r in self.records.items()},
            object_patches=int(labels.sum()), object_patches_kept=int(labels[picked].sum()),
            matched_patches=int((links >= 0).sum()), matched_kept=int((links[picked] >= 0).sum()),
            object_matched_kept=int(((links[picked] >= 0) & labels[picked]).sum()),
            distinct_tracks_kept=len(torch.unique(track_ids[picked])), links=link_rows,
            geometry_radius=radius, dense_cache_bytes=dense_bytes, query_cache_bytes=self.cache_bytes(),
            feature_bytes=self.feature_bytes(), retained_tokens=len(indices),
            selector_host_seconds=time.perf_counter() - started))
        self.pending = None
        self.tokens = None
        self.capture_enabled = False
