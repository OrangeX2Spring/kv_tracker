"""Opt-in inference-only Pi3 RoPE with a current-stream CUDA launch.

The kernel is copied from Pi3 27e96ce with only its launch stream changed.
Compile in the graph pilot's job-local /tmp; do not replace the saved backend.
"""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from pi3.curope.curope2d import cuRoPE2D


def load_graph_rope(build_dir):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / 'curope_stream'
    return load(name='kvt_curope_stream',
                sources=[str(source / 'bindings.cpp'), str(source / 'kernels.cu')],
                build_directory=str(build_dir), extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=True)


class GraphRoPE(cuRoPE2D):
    def __init__(self, backend, freq=100., F0=1.):
        super().__init__(freq, F0)
        self.backend = backend

    def forward(self, tokens, positions):
        assert not torch.is_grad_enabled(), 'GraphRoPE is an inference-only pilot'
        assert tokens.is_cuda and tokens.ndim == 4 and tokens.shape[-1] % 4 == 0
        assert positions.shape == (tokens.shape[0], tokens.shape[2], 2)
        assert positions.dtype == torch.long and positions.device == tokens.device
        tokens = tokens.transpose(1, 2).contiguous()
        self.backend.rope_2d(tokens, positions, self.base, self.F0)
        return tokens.transpose(1, 2).contiguous()
