"""Query-only access to a native growing keyframe bank; never evict or rebuild it."""
import numpy as np
import torch


class ActiveKeyframes:
    def __init__(self, mode, quantile=.5):
        assert mode in ('all', 'one', 'half', 'motion', 'alternate')
        assert 0 < quantile < 1
        self.mode, self.quantile = mode, quantile
        self.count = 1
        self.scores = []
        self.events = []
        self.saved = None

    def bootstrap(self, model, rgb):
        self.model = model
        self.previous = self.thumbnail(rgb)

    @staticmethod
    def thumbnail(rgb):
        assert rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8
        return rgb[::16, ::16].astype(np.float64).mean(-1) / 255.

    def begin(self, frame, cached_ids, rgb):
        assert self.saved is None and max(cached_ids) < frame
        unique = sorted(set(cached_ids))
        assert unique[0] == 0
        image = self.thumbnail(rgb)
        score = float(np.abs(image - self.previous).mean())
        self.previous = image
        threshold = float(np.quantile(self.scores[-64:], self.quantile)) if self.scores else None
        previous_count = self.count
        decision = frame >= 8 and frame % 8 == 0
        if self.mode == 'all':
            self.count = len(unique)
        elif self.mode == 'one':
            self.count = 1
        elif self.mode == 'half':
            self.count = max(1, (len(unique) + 1) // 2)
        elif decision:
            grow = score > threshold if self.mode == 'motion' else (frame // 8) % 2 == 1
            self.count = min(len(unique), max(1, self.count + (1 if grow else -1)))
        assert 1 <= self.count <= len(unique)
        selected = [0] + (unique[-(self.count - 1):] if self.count > 1 else [])
        # Preserve BOTH native copies of frame0 at bootstrap: one unique keyframe.
        slots = [i for i, fid in enumerate(cached_ids) if fid in selected]
        self.saved = self.model.cache
        self.versions = {i: {k: (t, t._version) for k, t in layer.items()}
                         for i, layer in self.saved.items()}
        first = next(iter(self.saved.values()))['k']
        assert first.ndim == 4 and first.shape[2] % len(cached_ids) == 0
        tokens = first.shape[2] // len(cached_ids)
        for layer in self.saved.values():
            assert layer['k'].shape == layer['v'].shape
            assert layer['k'].dtype == layer['v'].dtype
            assert layer['k'].shape[2] == len(cached_ids) * tokens
        if len(slots) != len(cached_ids):
            indices = (torch.tensor(slots, device=first.device)[:, None] * tokens
                       + torch.arange(tokens, device=first.device)).flatten()
            self.model.cache = {i: {k: t.index_select(2, indices) for k, t in layer.items()}
                                for i, layer in self.saved.items()}
        self.events.append(dict(frame=frame, mode=self.mode, quantile=self.quantile,
            score=score, threshold=threshold, threshold_start_frame=max(1, frame - 64),
            threshold_end_frame=frame - 1, decision=decision, previous_count=previous_count,
            available_ids=unique, physical_ids=list(cached_ids), selected_ids=selected,
            selected_physical_ids=[cached_ids[i] for i in slots], active_count=self.count,
            available_count=len(unique), active_tokens=len(slots) * tokens,
            stored_bytes=sum(t.numel() * t.element_size() for l in self.saved.values() for t in l.values()),
            active_bytes=sum(t.numel() * t.element_size() for l in self.model.cache.values() for t in l.values())))
        self.scores.append(score)
        self.scores = self.scores[-64:]

    def end(self):
        assert self.saved is not None
        for i, layer in self.saved.items():
            for k, t in layer.items():
                old, version = self.versions[i][k]
                assert t is old and t._version == version
        self.model.cache = self.saved
        self.saved = None
        self.versions = None
        self.events[-1]['bank_preserved'] = True
