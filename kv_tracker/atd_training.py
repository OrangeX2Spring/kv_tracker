"""Shared scene scorer contracts; diagnostic labels do not grant training eligibility."""
import torch
from torch import nn
import torch.nn.functional as F

from kv_tracker.adaptive_tokens import energy, select


def patch_features(rgb, maps=None):
    """RGB statistics/energy, optionally five causal map channels; unknown is explicit."""
    assert rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.dtype == torch.float32
    _, height, width = rgb.shape
    assert height % 14 == width % 14 == 0
    grid = (height // 14, width // 14)
    patches = rgb.reshape(3, grid[0], 14, grid[1], 14).permute(0, 1, 3, 2, 4)
    features = [patches.mean((-1, -2)), patches.std((-1, -2), unbiased=False),
                energy(rgb).reshape(1, *grid)]
    if maps is not None:
        support = maps['support'] > 0
        assert support.shape == (grid[0] * grid[1],)
        depth = maps['front_depth']
        # Normalize the arbitrary tracker gauge; all-unknown has no depth evidence.
        scale = depth[support].median() if support.any() else depth.new_tensor(1.)
        geometry = torch.stack((support.float(), maps['support'].float().log1p(),
            torch.where(support, (depth / scale).log1p(), 0.),
            torch.where(support, maps['confidence'], 0.),
            torch.where(support, maps['secondary_gap'].log1p(), 0.)))
        features.append(geometry.reshape(5, *grid))
    result = torch.cat(features)
    assert torch.isfinite(result).all()
    return result


class PatchScorer(nn.Module):
    """Identical small architecture for RGB-only A and cache-conditioned B."""
    def __init__(self, candidate):
        super().__init__()
        assert candidate in ('A', 'B')
        self.candidate = candidate
        self.network = nn.Sequential(nn.Conv2d(7 if candidate == 'A' else 12, 32, 3, padding=1),
            nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

    def forward(self, features):
        assert features.ndim == 4 and features.shape[1] == (7 if self.candidate == 'A' else 12)
        return self.network(features).flatten(1)


def ranking_loss(predicted, target):
    """Within-frame pair ranking; ties give no supervision, never cross-frame pairs."""
    assert predicted.shape == target.shape and predicted.ndim == 2
    pairs = torch.triu_indices(predicted.shape[1], predicted.shape[1], 1, device=predicted.device)
    direction = (target[:, pairs[0]] - target[:, pairs[1]]).sign()
    usable = direction != 0
    assert usable.any(), 'Teacher provides no ranking signal'
    difference = predicted[:, pairs[0]] - predicted[:, pairs[1]]
    return F.softplus(-direction[usable] * difference[usable]).mean()


def scorer_keep(score, grid, frame):
    """Same half-budget top-score/spread allocation for teacher and learned scorer."""
    assert score.shape == (grid[0] * grid[1],)
    return select(score, torch.zeros_like(score, dtype=torch.bool), 'scene',
                  'oracle_spread', grid, frame)
