"""Whole-image retention before Pi3 reconstruction; full dense query KV."""
import torch
import torch.nn.functional as F

from .combined_cache import CombinedCache


class BankCache(CombinedCache):
    def __init__(self, budget=8, interval=50, policy='relevance'):
        super().__init__(budget, interval, patch_fraction=1.)
        assert budget >= 3 and policy in ('recent', 'relevance')
        self.policy = policy

    def select(self, frame_id, cached_ids):
        assert sorted(set(cached_ids)) == list(self.records)
        assert max(cached_ids) < frame_id
        if (frame_id + 1) % self.interval:
            return False
        descriptor = F.normalize(self.tokens.mean(0), dim=0)
        ids = list(self.records) + [frame_id]
        protected = sorted({0, ids[-2], frame_id})
        eligible = [i for i in ids if i not in protected]
        # Two local views temper a single-image content transition. No GT,
        # future frames, discarded history, or post-rebuild features enter this.
        local = torch.stack((self.records[ids[-2]]['descriptor'], descriptor))
        scores = {i: float((self.records[i]['descriptor'] @ local.T).max())
                  for i in eligible}
        ranking = sorted(eligible, key=lambda i: (
            scores[i] if self.policy == 'relevance' else i, i), reverse=True)
        retained = sorted(protected + ranking[:self.budget - len(protected)])
        self.keep_frame_ids = retained[:-1]
        self.pending = dict(frame=frame_id, descriptor=descriptor,
                            evicted=[i for i in ids if i not in retained],
                            eligible_scores=scores, protected=protected)
        self.capture_enabled = False
        return True

    def after_rebuild(self, frame_ids, confidence, *, points=None, masks=None):
        assert confidence.ndim == 5 and confidence.shape[:2] == (1, len(frame_ids))
        assert frame_ids[0] == 0
        h, w = confidence.shape[2:4]
        assert h % 14 == w % 14 == 0
        tokens_per_frame = 5 + (h // 14) * (w // 14)
        for layer in self.model.cache.values():
            for tensor in layer.values():
                assert tensor.ndim == 4 and tensor.shape[2] == len(frame_ids) * tokens_per_frame
        if not self.records:
            assert frame_ids == [0, 0]
            self.records[0] = dict(descriptor=F.normalize(self.tokens.mean(0), dim=0))
            decision = dict(evicted=[], eligible_scores={}, protected=[0])
        else:
            decision = self.pending
            assert frame_ids == self.keep_frame_ids + [decision['frame']]
            assert len(frame_ids) <= self.budget
            self.records = {i: self.records[i] for i in self.keep_frame_ids}
            self.records[decision['frame']] = dict(descriptor=decision['descriptor'])
        # Deliberately never gather, mask or replace K/V after reconstruction.
        self.events.append(dict(frame=frame_ids[-1], retained_frame_ids=list(frame_ids),
            evicted=decision['evicted'], protected=decision['protected'],
            eligible_scores=decision['eligible_scores'], policy=self.policy,
            dense_cache_bytes=self.cache_bytes(), query_cache_bytes=self.cache_bytes(),
            feature_bytes=self.feature_bytes(), rebuild_input_frames=len(frame_ids)))
        self.pending = None
        self.tokens = None
        self.capture_enabled = False
