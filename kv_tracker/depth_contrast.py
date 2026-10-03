"""LoopCD-inspired layer-depth contrast for dense cached camera queries."""
import torch

from .token_drop import forward_kept, config


class DepthContrastQueries:
    def __init__(self, depth, reference_depth=None, guidance=0.):
        self.depth, self.reference_depth, self.guidance = depth, reference_depth, guidance
        self.keep = None
        self.queries = 0

    def reset(self):
        self.keep = None

    def forward(self, model, image, device):
        assert image.ndim == 3 and image.shape[2] == 3 and str(image.dtype) == 'uint8'
        assert not torch.is_grad_enabled() and not config['mass'] and config['probe'] is None
        imgs = torch.tensor(image[None, None], device=device, dtype=torch.float32) / 255.
        imgs = imgs.permute(0, 1, 4, 2, 3)
        h, w = image.shape[0] // 14, image.shape[1] // 14
        assert image.shape[0] % 14 == image.shape[1] % 14 == 0
        if self.keep is None:
            self.keep = torch.ones(1, h * w, device=device, dtype=torch.bool)
            counts = {v.shape[2] for layer in model.cache.values() for v in layer.values()}
            assert len(counts) == 1
            count = counts.pop()
            assert count % (model.patch_start_idx + h * w) == 0
            model.kept_cache = dict(labels=torch.ones(count, device=device, dtype=torch.long),
                                    distance=None, bias=None)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = forward_kept(model, imgs, self.keep, cam_only=True, use_cache=True,
                decoder_depth=self.depth, reference_depth=self.reference_depth, guidance=self.guidance)
        self.queries += 1
        return result['camera_poses']

    def summary(self):
        return dict(method='layer_depth_contrast', decoder_depth=self.depth,
                    reference_depth=self.reference_depth, guidance=self.guidance,
                    queries=self.queries, recurrent_model=False, training=False,
                    cache='native dense rebuilds unchanged', selection='all patches')
