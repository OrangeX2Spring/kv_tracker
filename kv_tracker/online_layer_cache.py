"""Causal layer budgets at native rebuilds; no offline teacher or dense shadow."""
from functools import partial

import torch
import torch.nn.functional as F

from kv_tracker.patch_select import allocate


def sensitivity(q, k, v, protected):
    """Sampled local-output change when optional retained history is removed."""
    assert q.ndim == k.ndim == v.ndim == 4 and k.shape == v.shape
    assert protected.shape == (k.shape[2],) and protected.dtype == torch.bool
    with torch.autocast('cuda', enabled=False):
        full = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        floor = F.scaled_dot_product_attention(q.float(), k[:, :, protected].float(),
                                             v[:, :, protected].float())
    return float((full - floor).square().mean() / full.square().mean().clamp_min(1e-12))


def budget_quotas(total, rooms, weights):
    """Charge a small observation reserve before sensitivity-driven allocation."""
    reserve = [min(8, room) for room in rooms]
    assert total >= sum(reserve)
    extra = allocate(total - sum(reserve), [r - n for r, n in zip(rooms, reserve)], weights)
    return (extra + torch.tensor(reserve)).tolist()


class OnlineLayerCache:
    def __init__(self, frame_equivalents, policy='online', probe_interval=16, queries=8):
        assert frame_equivalents >= 3 and policy in ('online', 'uniform')
        self.capacity, self.policy = frame_equivalents, policy
        self.interval, self.queries = probe_interval, queries
        assert self.interval > 0 and self.queries > 0
        self.events, self.probes, self.hooks = [], [], []
        self.frame = 0
        self.rows = {}
        self.scores = {}
        self.qkv = {}
        self.active = False

    def attach(self, model):
        self.model = model
        self.layers = list(range(1, len(model.decoder), 2))
        self.scores = {i: 0. for i in self.layers}
        for i in self.layers:
            block = model.decoder[i]
            self.hooks.append(block.register_forward_pre_hook(partial(self.before, i), with_kwargs=True))
            self.hooks.append(block.attn.qkv.register_forward_hook(partial(self.capture, i)))
            self.hooks.append(block.register_forward_hook(partial(self.after, i), with_kwargs=True))

    def begin_query(self, frame_id):
        self.frame = frame_id

    def end_query(self):
        assert not self.qkv

    def before(self, layer, module, args, kwargs):
        self.active = (self.policy == 'online' and self.frame % self.interval == 0
                       and kwargs.get('kv_cache') is not None and layer in self.rows)

    def capture(self, layer, module, args, output):
        if self.active:
            self.qkv[layer] = output

    def after(self, layer, module, args, kwargs, output):
        if layer not in self.qkv:
            return
        attention = module.attn
        projected = self.qkv.pop(layer)
        batch, count, triple = projected.shape
        q, k, v = projected.reshape(batch, count, 3, attention.num_heads,
                                    triple // (3 * attention.num_heads)).transpose(1, 3).unbind(2)
        q, k = attention.q_norm(q).to(v.dtype), attention.k_norm(k).to(v.dtype)
        pos = kwargs['xpos']
        if attention.rope is not None:
            q, k = attention.rope(q, pos), attention.rope(k, pos)
        slots = torch.linspace(0, count - 1, min(count, self.queries), device=q.device).round().long()
        cache = kwargs['kv_cache']
        keys = torch.cat((cache['k'], k), 2)
        values = torch.cat((cache['v'], v), 2)
        ids = self.rows[layer]
        protected = torch.cat(((ids // self.tokens == 0) | (ids // self.tokens == self.latest),
                               torch.ones(count, device=q.device, dtype=torch.bool)))
        score = sensitivity(q[:, :, slots], keys, values, protected)
        self.scores[layer] = .9 * self.scores[layer] + .1 * score
        self.probes.append(dict(frame=self.frame, layer=layer, score=score,
                                smoothed=self.scores[layer], cached_rows=len(ids), queries=len(slots)))

    def after_rebuild(self, frame_ids, confidence, points, masks, rgb):
        first = self.model.cache[self.layers[0]]['k']
        assert first.ndim == 4 and first.shape[2] % len(frame_ids) == 0
        tokens = first.shape[2] // len(frame_ids)
        if not self.events:
            self.tokens = tokens
            self.row_bytes = first.shape[0] * first.shape[1] * first.shape[3] * first.element_size() * 2
            self.limit = self.capacity * self.tokens * len(self.layers) * (self.row_bytes + 8)
        assert tokens == self.tokens
        self.latest = frame_ids[-1]
        # Positions are already encoded into Pi3 K. IDs identify rows, including
        # both physical bootstrap copies; subsequent rebuilds use one anchor.
        ids = torch.tensor(frame_ids, device=first.device)[:, None] * tokens + torch.arange(tokens, device=first.device)
        ids = ids.flatten()
        protected = (ids // tokens == 0) | (ids // tokens == self.latest)
        floor = int(protected.sum())
        total = min(self.capacity * tokens * len(self.layers), len(ids) * len(self.layers))
        optional = len(ids) - floor
        weights = ([self.scores[i] ** .5 for i in self.layers] if self.policy == 'online'
                   else [1.] * len(self.layers))
        quotas = budget_quotas(total - floor * len(self.layers), [optional] * len(self.layers), weights)
        counts = {}
        for layer, quota in zip(self.layers, quotas):
            free = (~protected).nonzero().flatten()
            slots = torch.linspace(0, len(free) - 1, quota, device=first.device).round().long()
            selected = torch.cat((protected.nonzero().flatten(), free[slots])).sort().values
            for name in ('k', 'v'):
                self.model.cache[layer][name] = self.model.cache[layer][name].index_select(2, selected)
            self.rows[layer] = ids.index_select(0, selected)
            counts[layer] = len(selected)
        actual = sum(t.untyped_storage().nbytes() for c in self.model.cache.values() for t in c.values())
        metadata = sum(t.untyped_storage().nbytes() for t in self.rows.values())
        assert actual == total * self.row_bytes and actual + metadata <= self.limit
        self.events.append(dict(frame=self.frame, frame_ids=list(frame_ids), policy=self.policy,
            frame_equivalents=self.capacity, budget_bytes=self.limit, query_cache_bytes=actual,
            persistent_bytes=actual + metadata,
            metadata_bytes=metadata, tokens_by_layer=counts, scores=dict(self.scores),
            last_probe_frame=max((p['frame'] for p in self.probes), default=None),
            protected_rows_per_layer=floor, allocation='sqrt smoothed local sensitivity; uniform within layer'))

    def close(self):
        for hook in self.hooks:
            hook.remove()
