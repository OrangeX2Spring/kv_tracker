"""Opt-in real-scene query replay pilot; rebuilds keep the native RoPE backend."""
import time

import torch

from kv_tracker.graph_rope import GraphRoPE, load_graph_rope, cuRoPE2D
from kv_tracker.token_drop import forward_kept


class GraphQueries:
    def __init__(self, build_dir, checked, native_dense=False):
        started = time.perf_counter()
        self.backend = load_graph_rope(build_dir)
        self.backend_load_seconds = time.perf_counter() - started
        self.checked = checked
        self.native_dense = native_dense
        self.graph = None
        self.rows = []
        self.queries = 0
        self.max_pose_error = 0.

    def reset(self):
        # Release the graph and its cache references before the tracker rebuilds.
        self.graph = None
        self.image = self.index = self.features = None

    def _query(self, model, imgs, keep, query_indices=None, defer_camera_head=False):
        if not self.native_dense:
            return forward_kept(model, imgs, keep, cam_only=True, use_cache=True,
                                query_indices=query_indices,
                                defer_camera_head=defer_camera_head)
        if not defer_camera_head:
            return model(imgs, cam_only=True, use_cache=True)
        # Pi3.forward up to the eager camera head, preserving its encoder and
        # decoder tensor layout. Dense replay must not use the kept-token path.
        normalized = (imgs - model.image_mean) / model.image_std
        B, N, C, H, W = normalized.shape
        hidden = model.encoder(normalized.reshape(B * N, C, H, W), is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden['x_norm_patchtokens']
        hidden, pos, mask = model.decode(hidden, N, H, W, use_cache=True)
        camera = model.camera_decoder(hidden, xpos=pos, attn_mask=mask)
        return dict(camera_features=camera.float()[:, model.patch_start_idx:])

    def forward(self, model, imgs, keep):
        assert imgs.shape[:2] == (1, 1) and not torch.is_grad_enabled()
        index = keep[0].nonzero().flatten()
        if self.native_dense:
            assert keep.all()
        assert index.numel() > 0 and (index[1:] > index[:-1]).all()
        height, width = imgs.shape[-2:]
        pointers = {i: {k: v.data_ptr() for k, v in layer.items()}
                    for i, layer in model.cache.items()}
        if self.graph is None:
            torch.cuda.synchronize()
            started = time.perf_counter()
            expected = self._query(model, imgs, keep)['camera_poses'].clone()
            cache = ({i: {k: v.clone() for k, v in layer.items()}
                      for i, layer in model.cache.items()} if self.checked else {})
            labels = model.kept_cache['labels'].clone()
            original = model.rope
            assert isinstance(original, cuRoPE2D)
            replacement = GraphRoPE(self.backend, freq=original.base, F0=original.F0)
            modules = [m for m in model.modules() if getattr(m, 'rope', None) is original]
            self.image, self.index = imgs.clone(), index.clone()
            self.keep = keep.clone()
            self.pointers = pointers
            for module in modules:
                module.rope = replacement
            try:
                adapted = self._query(model, imgs, keep,
                                       query_indices=index)['camera_poses']
                torch.testing.assert_close(adapted, expected, rtol=0, atol=0)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        features = self._query(model, self.image, self.keep,
                            query_indices=self.index,
                            defer_camera_head=True)['camera_features']
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                with torch.amp.autocast('cuda', enabled=False):
                    pose = model.camera_head(features, height // 14, width // 14).reshape(1, 1, 4, 4)
                torch.testing.assert_close(pose, expected, rtol=1e-4, atol=1e-4)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.features = self._query(model, self.image, self.keep,
                        query_indices=self.index,
                        defer_camera_head=True)['camera_features']
                torch.cuda.synchronize()
            finally:
                for module in modules:
                    module.rope = original
            for i, layer in cache.items():
                for key, value in layer.items():
                    assert torch.equal(model.cache[i][key], value)
            assert torch.equal(model.kept_cache['labels'], labels)
            self.rows.append(dict(query=self.queries, kept=index.numel(),
                cached_tokens=labels.numel(), setup_seconds=time.perf_counter() - started))
        else:
            assert imgs.shape == self.image.shape and index.shape == self.index.shape
            assert pointers == self.pointers, 'Replay across a cache rebuild is forbidden'
            expected = (self._query(model, imgs, keep)['camera_poses'].clone()
                        if self.checked else None)
        self.image.copy_(imgs)
        self.index.copy_(index)
        self.graph.replay()
        with torch.amp.autocast('cuda', enabled=False):
            pose = model.camera_head(self.features, height // 14, width // 14).reshape(1, 1, 4, 4)
        if expected is not None:
            torch.testing.assert_close(pose, expected, rtol=1e-4, atol=1e-4)
            self.max_pose_error = max(self.max_pose_error, float((pose - expected).abs().max()))
        assert pointers == {i: {k: v.data_ptr() for k, v in layer.items()}
                            for i, layer in model.cache.items()}
        self.queries += 1
        return dict(camera_poses=pose)
