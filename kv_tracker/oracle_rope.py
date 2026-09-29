"""Pi3 CUDA RoPE backward with private contiguous gradient storage.

Upstream mutates its incoming gradient in place. Attention/transpose backward
can supply a strided or shared tensor; copying here satisfies the CUDA contract
without changing the forward kernels or mutating another consumer's gradient.
"""
import torch

from pi3.curope.curope2d import cuRoPE2D, cuRoPE2D_func


class ContiguousRoPEFunction(cuRoPE2D_func):
    @staticmethod
    def backward(ctx, grad_res):
        return cuRoPE2D_func.backward(ctx, grad_res.clone(memory_format=torch.contiguous_format))


class OracleRoPE(cuRoPE2D):
    def forward(self, tokens, positions):
        if not torch.is_grad_enabled():
            return super().forward(tokens, positions)
        tokens = tokens.transpose(1, 2).contiguous()
        tokens = ContiguousRoPEFunction.apply(tokens, positions, self.base, self.F0)
        return tokens.transpose(1, 2).contiguous()
