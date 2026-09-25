"""Append arrival-time K/V from fixed-ID cached queries; never rebuild history.

The pinned Pi3 BlockRope can return concatenated, post-RoPE K/V even on its
cached path. Hooks request that tuple and return only the hidden state to Pi3.
Keep just the new tail until the tracker accepts this scheduled keyframe.
"""
from functools import partial
from time import perf_counter

import torch


class AppendOnlyCache:
    def __init__(self, insertion_indices, verify=False, refresh_frame=None,
                 refresh_gauge='frozen'):
        assert insertion_indices == sorted(set(insertion_indices))
        assert all(type(i) is int and i > 0 for i in insertion_indices)
        assert refresh_frame is None or refresh_frame in insertion_indices
        # 'keyframe0' re-anchors output normalization on the refreshed frame-0
        # prediction, as every native rebuild does; 'frozen' keeps bootstrap's.
        assert refresh_gauge in ('frozen', 'keyframe0')
        assert refresh_gauge == 'frozen' or refresh_frame is not None
        self.refresh_frame = refresh_frame
        self.refresh_gauge = refresh_gauge
        self.refresh_events = []
        self.indices = set(insertion_indices)
        self.verify = verify
        self.frame_ids = [0, 0]  # Preserve native bootstrap, including its duplicate.
        self.pending = {}
        self.frame = None
        self.capture = False
        self.events = []
        self.hooks = []

    def attach(self, model):
        self.model = model
        self.layers = list(range(1, len(model.decoder), 2))
        for i in self.layers:
            block = model.decoder[i]
            self.hooks.append(block.register_forward_pre_hook(self.before_block, with_kwargs=True))
            self.hooks.append(block.register_forward_hook(partial(self.after_block, i), with_kwargs=True))

    def begin_query(self, frame):
        assert not self.pending
        self.frame = frame
        self.capture = frame in self.indices

    def before_block(self, module, args, kwargs):
        if self.capture:
            assert kwargs['kv_cache'] is not None and not kwargs['ret_kv']
            kwargs['ret_kv'] = True
            return args, kwargs

    def after_block(self, layer, module, args, kwargs, output):
        if not self.capture:
            return
        hidden, keys, values = output
        old = kwargs['kv_cache']
        count = args[0].shape[1]
        assert keys.ndim == values.ndim == 4 and keys.shape == values.shape
        assert keys.shape[2] == old['k'].shape[2] + count
        # A view here would retain the whole concatenated history in every layer.
        self.pending[layer] = dict(k=keys[:, :, -count:].clone(),
                                   v=values[:, :, -count:].clone())
        return hidden

    def commit(self, frame):
        assert self.capture and frame == self.frame and frame > self.frame_ids[-1]
        assert set(self.pending) == set(self.model.cache) == set(self.layers)
        if self.pending[self.layers[0]]['k'].is_cuda:
            torch.cuda.synchronize()
        started = perf_counter()
        counts = set()
        for i in self.layers:
            for name in ('k', 'v'):
                old, tail = self.model.cache[i][name], self.pending[i][name]
                assert old.shape[:2] == tail.shape[:2] and old.shape[3] == tail.shape[3]
                assert old.dtype == tail.dtype
                assert old.shape[2] == len(self.frame_ids) * tail.shape[2]
                counts.add(tail.shape[2])
                combined = torch.cat((old, tail), dim=2)
                if self.verify:
                    assert torch.equal(combined[:, :, :old.shape[2]], old)
                self.model.cache[i][name] = combined
        assert len(counts) == 1
        if self.pending[self.layers[0]]['k'].is_cuda:
            torch.cuda.synchronize()
        self.frame_ids.append(frame)
        self.events.append(dict(frame=frame, cache_frame_ids=list(self.frame_ids),
            tokens_per_frame=counts.pop(), old_cache_prefix_verified=self.verify,
            commit_seconds=perf_counter() - started,
            cache_bytes=sum(t.numel() * t.element_size()
                            for layer in self.model.cache.values() for t in layer.values())))
        self.pending.clear()
        self.capture = False

    def close(self):
        for hook in self.hooks:
            hook.remove()
