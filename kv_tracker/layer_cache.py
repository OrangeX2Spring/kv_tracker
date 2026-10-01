"""Physically omit historical KV at selected depths; keep every block active."""
import torch


class LayerCache:
    def __init__(self, mode, omitted_layers, special_tokens=5):
        assert mode in ('native', 'omit', 'uniform', 'uniform_special')
        assert omitted_layers == sorted(set(omitted_layers)) and omitted_layers
        self.mode, self.omitted_layers = mode, omitted_layers
        self.special_tokens = special_tokens
        self.events = []

    def __call__(self, cache, frame_ids):
        layers = sorted(cache)
        assert set(self.omitted_layers) < set(layers)
        shape = cache[layers[0]]['k'].shape
        dtype = cache[layers[0]]['k'].dtype
        assert len(shape) == 4
        for layer in cache.values():
            assert set(layer) == {'k', 'v'}
            for value in layer.values():
                assert value.shape == shape and value.dtype == dtype
        count = shape[2]
        total = (len(layers) if self.mode == 'native' else len(layers) - len(self.omitted_layers)) * count
        base, extra = divmod(total, len(layers))
        dense_bytes = sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                           for layer in cache.values() for t in layer.values()}.values())
        counts = {}
        for slot, index in enumerate(layers):
            keep = (count if self.mode == 'native' else
                    (0 if index in self.omitted_layers else count) if self.mode == 'omit' else base + (slot < extra))
            counts[index] = keep
            if keep == count:
                # Native V can be a view of the full QKV projection. Compact all
                # retained arrays in every arm to make actual-byte budgets equal.
                for name in ('k', 'v'):
                    cache[index][name] = cache[index][name].clone(memory_format=torch.contiguous_format)
                continue
            selected = torch.linspace(0, count - 1, keep, device=cache[index]['k'].device).round().long()
            if self.mode == 'uniform_special':
                assert count % len(frame_ids) == 0
                per_frame = count // len(frame_ids)
                assert 0 < self.special_tokens < per_frame
                positions = torch.arange(count, device=selected.device)
                special = positions[positions % per_frame < self.special_tokens]
                patches = positions[positions % per_frame >= self.special_tokens]
                patch_keep = keep - len(special)
                assert 0 <= patch_keep <= len(patches), 'Budget must retain all special tokens'
                patch_slots = torch.linspace(0, len(patches) - 1, patch_keep,
                                             device=selected.device).round().long()
                selected = torch.cat((special, patches[patch_slots])).sort().values
            for name in ('k', 'v'):
                old = cache[index][name]
                # Empty slices would retain the full original storage.
                cache[index][name] = (old.index_select(2, selected) if keep else
                                     old.new_empty(shape[:2] + (0,) + shape[3:]))
        actual_bytes = sum(t.untyped_storage().nbytes()
                           for layer in cache.values() for t in layer.values())
        expected_bytes = total * shape[0] * shape[1] * shape[3] * 2 * cache[layers[0]]['k'].element_size()
        assert actual_bytes == expected_bytes
        self.events.append(dict(frame_ids=list(frame_ids), mode=self.mode,
            omitted_layers=self.omitted_layers, tokens_by_layer=counts,
            dense_bytes=dense_bytes, persistent_bytes=actual_bytes))
        if self.mode == 'uniform_special':
            self.events[-1].update(special_tokens_per_frame=self.special_tokens,
                preserved_special_tokens_per_layer=len(frame_ids) * self.special_tokens,
                selection='all per-frame special tokens plus evenly spaced flattened patches')
