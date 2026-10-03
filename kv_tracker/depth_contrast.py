"""LoopCD-inspired layer-depth contrast for dense cached camera queries."""
import torch

from .token_drop import config


class DepthContrastQueries:
    def __init__(self, depth, reference_depth=None, guidance=0.):
        self.depth, self.reference_depth, self.guidance = depth, reference_depth, guidance
        self.queries = 0

    def reset(self):
        # Native queries retain no cache references between calls.
        return

    def forward(self, model, image, device):
        assert image.ndim == 3 and image.shape[2] == 3 and str(image.dtype) == 'uint8'
        assert not torch.is_grad_enabled() and not config['mass'] and config['probe'] is None
        assert image.shape[0] % 14 == image.shape[1] % 14 == 0
        decoder = model.decoder
        assert 2 <= self.depth <= len(decoder) and self.depth % 2 == 0
        assert self.guidance >= 0 and (self.reference_depth is not None) == (self.guidance > 0)
        if self.reference_depth is not None:
            assert 2 <= self.reference_depth < self.depth and self.reference_depth % 2 == 0
        imgs = torch.tensor(image[None, None], device=device, dtype=torch.float32) / 255.
        imgs = imgs.permute(0, 1, 4, 2, 3)
        weak, handles = [], []

        def reference_output(module, args, output):
            assert output.ndim == 3 and output.shape[0] == 1
            weak.append(output)

        def guided_input(module, args):
            strong = args[0]
            early = torch.cat(weak, dim=-1)
            assert len(weak) == 2 and early.shape == strong.shape
            guided = (strong.float() + self.guidance * (strong.float() - early.float())).to(strong.dtype)
            return (guided,) + args[1:]

        try:
            # Native decode chooses the final pair using len(model.decoder).
            # Only the query is shortened; restore all layers before rebuilds.
            if self.depth != len(decoder):
                model.decoder = decoder[:self.depth]
            if self.reference_depth is not None:
                handles = [decoder[i].register_forward_hook(reference_output)
                           for i in (self.reference_depth - 2, self.reference_depth - 1)]
                handles.append(model.camera_decoder.register_forward_pre_hook(guided_input, prepend=True))
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = model(imgs, cam_only=True, use_cache=True)
        finally:
            for handle in handles:
                handle.remove()
            model.decoder = decoder
        self.queries += 1
        return result['camera_poses']

    def summary(self):
        return dict(method='layer_depth_contrast', decoder_depth=self.depth,
                    reference_depth=self.reference_depth, guidance=self.guidance,
                    queries=self.queries, recurrent_model=False, training=False,
                    forward_path='native Pi3',
                    cache='native dense rebuilds unchanged', selection='all patches')
