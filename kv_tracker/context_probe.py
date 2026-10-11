"""How much a tracked frame actually uses the cached bank, read from Pi3's own attention.

During a cached query, each cross-frame decoder block (odd index) lets the query
frame's tokens attend to [cached bank tokens, own tokens]. Pi3's camera head
predicts pose from the patch tokens, so the probe recomputes, for the query's
patch tokens (optionally only object patches), the softmax over all keys exactly as
FlashAttentionRope.forward_w_cache does, and splits the probability mass into
cached patches, cached register tokens, own patches and own register tokens.
Register tokens act as attention sinks; reporting them separately keeps them from
flattening the signal (the earlier uncorrected-mass confound).

Per layer it reports:
- support: cached-patch mass / (cached-patch + own-patch mass), sink-free;
- top1: share of (head, query patch) pairs whose most attended key is a cached patch;
- object_support (objects only): the same as support, restricted to cached patches
  inside the anchor's object mask;
- cache_register / own_register: sink mass.
The probe only reads; it never changes the forward pass.
"""
import time

import torch


class ContextProbe:
    def __init__(self, model, patch_start=5):
        assert len(model.decoder) == 36 and model.patch_start_idx == patch_start
        self.model, self.patch_start = model, patch_start
        self.layers = list(range(1, 36, 2))
        self.hooks, self.rows = [], None
        self.query_mask = self.anchor_mask = None
        self.seconds = 0.

    def attach(self):
        assert not self.hooks
        for index in self.layers:
            self.hooks.append(self.model.decoder[index].register_forward_pre_hook(
                lambda block, args, kwargs, index=index: self.measure(index, block, args, kwargs),
                with_kwargs=True))

    def detach(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_anchor(self, token_mask):
        """Object patches of the anchor frame (bool, patches) or None for scenes."""
        self.anchor_mask = token_mask

    def arm(self, token_mask=None):
        """Measure the next cached query; token_mask selects query patches (objects)."""
        self.rows, self.query_mask, self.seconds = {}, token_mask, 0.

    def result(self):
        rows, self.rows = self.rows, None
        assert rows is not None and sorted(rows) == self.layers, 'probe did not see every cross-frame layer'
        return dict(layers=[rows[i] for i in self.layers], seconds=self.seconds)

    @torch.no_grad()
    def measure(self, index, block, args, kwargs):
        cache = kwargs.get('kv_cache')
        if self.rows is None or cache is None:
            return
        started = time.perf_counter()
        x, xpos, attn = args[0], kwargs['xpos'], block.attn
        y = block.norm1(x)
        batch, length, channels = y.shape
        assert batch == 1 and length > self.patch_start
        heads = attn.num_heads
        qkv = attn.qkv(y).reshape(batch, length, 3, heads, channels // heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = attn.q_norm(q).to(v.dtype), attn.k_norm(k).to(v.dtype)
        if attn.rope is not None:
            q, k = attn.rope(q, xpos), attn.rope(k, xpos)
        cached = cache['k']
        assert cached.shape[2] % length == 0, 'cache must hold whole frames of the query size'
        frames = cached.shape[2] // length
        keys = torch.cat([cached, k], dim=2)[0]  # heads, keys, dim
        patches = torch.arange(self.patch_start, length, device=q.device)
        if self.query_mask is not None:
            assert self.query_mask.shape == (length - self.patch_start,)
            patches = patches[self.query_mask.to(q.device)]
        position = torch.arange(keys.shape[1], device=q.device) % length
        is_cache = torch.arange(keys.shape[1], device=q.device) < cached.shape[2]
        is_patch = position >= self.patch_start
        object_key = None
        if self.anchor_mask is not None:
            assert self.anchor_mask.shape == (length - self.patch_start,)
            in_object = torch.zeros(length, dtype=torch.bool, device=q.device)
            in_object[self.patch_start:] = self.anchor_mask.to(q.device)
            object_key = is_cache & in_object[position]
        totals = dict(cache_patch=0., cache_register=0., own_patch=0., own_register=0.,
                      cache_object=0., top1=0.)
        for head in range(heads):  # one head at a time bounds the score matrix
            score = (q[0, head, patches].float() @ keys[head].float().T) * attn.scale
            probability = score.softmax(dim=-1)
            totals['cache_patch'] += float(probability[:, is_cache & is_patch].sum(1).mean())
            totals['cache_register'] += float(probability[:, is_cache & ~is_patch].sum(1).mean())
            totals['own_patch'] += float(probability[:, ~is_cache & is_patch].sum(1).mean())
            totals['own_register'] += float(probability[:, ~is_cache & ~is_patch].sum(1).mean())
            best = probability.argmax(1)
            totals['top1'] += float((is_cache[best] & is_patch[best]).double().mean())
            if object_key is not None:
                totals['cache_object'] += float(probability[:, object_key].sum(1).mean())
        mean = {name: value / heads for name, value in totals.items()}
        row = dict(support=mean['cache_patch'] / (mean['cache_patch'] + mean['own_patch']),
                   top1=mean['top1'], cache_register=mean['cache_register'],
                   own_register=mean['own_register'], cached_frames=frames,
                   query_patches=int(len(patches)))
        if object_key is not None:
            row['object_support'] = mean['cache_object'] / (mean['cache_object'] + mean['own_patch'])
        self.rows[index] = row
        self.seconds += time.perf_counter() - started


def token_mask(mask, patch=14):
    """Pixel mask (H, W) at model resolution to patch tokens: a patch is in when most pixels are."""
    assert mask.ndim == 2 and mask.dtype == torch.bool
    h, w = mask.shape[0] // patch, mask.shape[1] // patch
    cells = mask[:h * patch, :w * patch].reshape(h, patch, w, patch).float().mean((1, 3))
    return (cells > .5).flatten()
