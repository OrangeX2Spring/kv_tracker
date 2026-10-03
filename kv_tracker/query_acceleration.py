"""Query-only execution experiments over an unchanged dense Pi3 cache."""
import torch

from .pi3_utilts import pi3_inference
from .token_drop import forward_kept, config


class QueryAcceleration:
    def __init__(self, method, build_dir=None, checked=False):
        assert method in ('graph', 'half', 'half_graph')
        self.method, self.checked = method, checked
        self.executor = None
        if method in ('graph', 'half_graph'):
            from .graph_query import GraphQueries
            self.executor = GraphQueries(build_dir, checked)
        self.keep = None
        self.queries = 0
        self.max_native_error = 0.

    def reset(self):
        if self.executor is not None:
            self.executor.reset()
        self.keep = None

    def forward(self, model, image, device):
        assert image.ndim == 3 and image.shape[2] == 3
        assert str(image.dtype) == 'uint8' and not torch.is_grad_enabled()
        assert not config['mass'] and config['probe'] is None
        # Match pi3_inference's native upload/normalization exactly.
        imgs = torch.tensor(image[None, None], device=device, dtype=torch.float32) / 255.
        imgs = imgs.permute(0, 1, 4, 2, 3)
        h, w = image.shape[0] // 14, image.shape[1] // 14
        if self.keep is None:
            assert image.shape[0] % 14 == image.shape[1] % 14 == 0
            if self.method == 'graph':
                self.keep = torch.ones(1, h * w, device=device, dtype=torch.bool)
            else:
                y, x = torch.meshgrid(torch.arange(h, device=device),
                                      torch.arange(w, device=device), indexing='ij')
                self.keep = ((y + x) % 2 == 0).reshape(1, -1)
            counts = {v.shape[2] for layer in model.cache.values() for v in layer.values()}
            assert len(counts) == 1
            count = counts.pop()
            assert count % (model.patch_start_idx + h * w) == 0
            # Native rebuilds are untouched. Labels are bookkeeping required by
            # forward_kept; no probe or mass bias consumes them in this experiment.
            model.kept_cache = dict(labels=torch.ones(count, device=device, dtype=torch.long),
                                    distance=None, bias=None)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            expected = (pi3_inference(model, imgs.permute(0, 1, 3, 4, 2), device,
                                      cam_only=True, use_cache=True)
                        if self.checked and self.method == 'graph' else None)
            output = (self.executor.forward(model, imgs, self.keep) if self.executor is not None
                      else forward_kept(model, imgs, self.keep, cam_only=True, use_cache=True))
        pose = output['camera_poses']
        if expected is not None:
            torch.testing.assert_close(pose, expected, rtol=1e-4, atol=1e-4)
            self.max_native_error = max(self.max_native_error, float((pose - expected).abs().max()))
        self.queries += 1
        return pose

    def summary(self):
        return dict(method=self.method, checked=self.checked, queries=self.queries,
                    max_native_error=self.max_native_error
                        if self.checked and self.method == 'graph' else None,
                    captures=self.executor.rows if self.executor is not None else [],
                    backend_load_seconds=self.executor.backend_load_seconds
                        if self.executor is not None else 0.,
                    cache='native dense rebuilds unchanged',
                    selection='all' if self.method == 'graph' else 'fixed checkerboard half')
