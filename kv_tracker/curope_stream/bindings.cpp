/*
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/
#include <torch/extension.h>

void rope_2d_cuda(torch::Tensor tokens, const torch::Tensor positions,
                  const float base, const float fwd);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_2d", &rope_2d_cuda, "Pi3 CUDA RoPE on PyTorch's current stream");
}
