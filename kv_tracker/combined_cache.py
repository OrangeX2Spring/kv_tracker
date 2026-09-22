"""Redundancy eviction and confidence/novelty pruning of Pi3 query memory.

The pinned Pi3 stores normalized, RoPE-transformed keys in its odd decoder blocks.
Dense reconstruction remains native; only the persistent cache for later queries
is gathered. Selection uses arrival-time encoder features and rebuild confidence.
"""
import math

import torch
import torch.nn.functional as F


class CombinedCache:
    def __init__(self, budget=20, interval=50, patch_fraction=0.5):
        assert budget >= 2 and interval >= 2 and 0 < patch_fraction <= 1
        self.budget = budget
        self.interval = interval
        self.patch_fraction = patch_fraction
        self.records = {}
        self.tokens = None
        self.pending = None
        self.keep_frame_ids = [0]
        self.events = []
        self.capture_enabled = True

    def attach(self, model):
        assert model.patch_size == 14 and model.patch_start_idx == 5
        self.model = model
        self.hook = model.encoder.register_forward_hook(self.capture)

    def capture(self, module, args, output):
        if not self.capture_enabled:
            return
        tokens = output['x_norm_patchtokens']
        assert tokens.ndim == 3 and tokens.shape[-1] == 1024
        self.tokens = tokens[-1].detach().float().cpu()

    def begin_query(self, frame_id):
        self.capture_enabled = (frame_id + 1) % self.interval == 0

    def select(self, frame_id, cached_ids):
        assert sorted(set(cached_ids)) == list(self.records)
        assert max(cached_ids) < frame_id
        if (frame_id + 1) % self.interval:
            return False
        patches = F.normalize(self.tokens, dim=-1)
        descriptor = F.normalize(self.tokens.mean(0), dim=0)
        best = torch.full((len(patches),), -1.0)
        for record in self.records.values():
            history = record['patches']
            for start in range(0, len(patches), 128):
                similarity = patches[start:start + 128] @ history.T
                best[start:start + 128] = torch.maximum(
                    best[start:start + 128], similarity.max(1).values)
        novelty = (1 - best).clamp(0, 2)
        ids = list(self.records) + [frame_id]
        descriptors = torch.stack([r['descriptor'] for r in self.records.values()]
                                  + [descriptor])
        evicted = []
        if len(ids) > self.budget:
            similarity = descriptors @ descriptors.T
            similarity.fill_diagonal_(-float('inf'))
            redundancy = similarity.max(1).values
            eligible = [i for i, value in enumerate(ids) if value not in (0, frame_id)]
            victim = max(eligible, key=lambda i: float(redundancy[i]))
            evicted = [ids.pop(victim)]
        self.keep_frame_ids = ids[:-1]
        self.pending = dict(frame=frame_id, patches=patches, descriptor=descriptor,
                            novelty=novelty, evicted=evicted)
        self.capture_enabled = False
        return True

    def after_rebuild(self, frame_ids, confidence):
        assert confidence.ndim == 5 and confidence.shape[:2] == (1, len(frame_ids))
        assert confidence.shape[-1] == 1 and frame_ids[0] == 0
        h, w = confidence.shape[2:4]
        assert h % 14 == w % 14 == 0
        patch_count = h // 14 * (w // 14)
        assert self.tokens.shape == (patch_count, 1024)
        if not self.records:
            assert frame_ids == [0, 0]
            self.records[0] = dict(descriptor=F.normalize(self.tokens.mean(0), dim=0),
                                  patches=F.normalize(self.tokens, dim=-1),
                                  indices=torch.arange(patch_count))
            evicted = []
        else:
            pending = self.pending
            assert frame_ids == self.keep_frame_ids + [pending['frame']]
            # As in the streaming policy, score against the pre-eviction history.
            pooled_confidence = F.avg_pool2d(
                confidence[0, -1, ..., 0].detach().float()[None, None], 14, 14).flatten().cpu()
            priority = (pooled_confidence.argsort(stable=True).argsort().float()
                        + pending['novelty'].argsort(stable=True).argsort().float())
            count = math.ceil(patch_count * self.patch_fraction)
            picked = priority.argsort(descending=True, stable=True)[:count].sort().values
            self.records = {i: self.records[i] for i in self.keep_frame_ids}
            self.records[pending['frame']] = dict(descriptor=pending['descriptor'],
                patches=pending['patches'][picked], indices=picked)
            evicted = pending['evicted']
        special = self.model.patch_start_idx
        token_count = special + patch_count
        indices = torch.cat([torch.cat((torch.arange(special),
            self.records[frame]['indices'] + special)) + slot * token_count
            for slot, frame in enumerate(frame_ids)])
        dense_bytes = self.cache_bytes()
        assert set(self.model.cache) == set(range(1, len(self.model.decoder), 2))
        for layer in self.model.cache.values():
            for name in ('k', 'v'):
                tensor = layer[name]
                assert tensor.ndim == 4 and tensor.shape[0] == 1
                assert tensor.shape[2] == len(frame_ids) * token_count
                # Leave the full-retention fidelity path bit-identical.
                if len(indices) != tensor.shape[2]:
                    layer[name] = tensor.index_select(2, indices.to(tensor.device))
        self.events.append(dict(frame=frame_ids[-1], retained_frame_ids=list(frame_ids),
            evicted=evicted, patch_indices={str(i): r['indices'].tolist()
                                          for i, r in self.records.items()},
            dense_cache_bytes=dense_bytes, query_cache_bytes=self.cache_bytes(),
            feature_bytes=self.feature_bytes(), retained_tokens=len(indices)))
        self.pending = None
        self.tokens = None
        self.capture_enabled = False

    def cache_bytes(self):
        return sum(t.numel() * t.element_size()
                   for layer in self.model.cache.values() for t in layer.values())

    def feature_bytes(self):
        return sum(t.numel() * t.element_size()
                   for record in self.records.values() for t in record.values())

    def close(self):
        self.hook.remove()
