# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA coverage for FP8 tensor storage and dequantization."""

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.export.quant_format import QUANTIZATION_FP8_PB_WO
from modelopt.torch.export.quant_utils import to_quantized_weight
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.extensions import get_cuda_ext_fp8
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor import FP8QTensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("precomputed_amax", [False, True])
@pytest.mark.parametrize("layout", ["tensor", "channel", "block_1d", "block_2d"])
def test_fp8_zero_scales_cuda(dtype, precomputed_amax, layout):
    """Real quantization preserves zero scales and matches native CUDA fake quantization."""
    cuda_ext = get_cuda_ext_fp8(raise_if_failed=True)
    values = torch.tensor(
        [-448, -1.0625, -0.125, 0, 0.125, 0.5625, 1.0625, 448],
        device="cuda",
        dtype=dtype,
    )
    weight = values.repeat(9, 3)[:, :19].contiguous()
    row_scales = torch.tensor(
        [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32], device="cuda", dtype=dtype
    ).unsqueeze(-1)
    weight *= row_scales
    axis, block_sizes = None, None
    if layout == "tensor":
        weight.zero_()
        groups = weight.reshape(1, -1)
    elif layout == "channel":
        axis = 0
        weight[0].zero_()
        groups = weight
    else:
        block_height = 8 if layout == "block_2d" else 1
        block_sizes = {-1: 8}
        if block_height == 8:
            block_sizes[-2] = 8
        weight[:block_height, :8].zero_()
        weight[-1, 16:].zero_()
        padded = F.pad(weight, (0, 5, 0, (-weight.shape[0]) % block_height))
        block_shape = (padded.shape[0] // block_height, 3)
        groups = (
            padded.reshape(block_shape[0], block_height, block_shape[1], 8)
            .permute(0, 2, 1, 3)
            .reshape(-1, block_height * 8)
        )

    group_amax = groups.abs().amax(dim=1)
    if block_sizes:
        amax = group_amax.reshape(block_shape)
    elif axis is not None:
        amax = group_amax.reshape(-1, 1)
    else:
        amax = group_amax.squeeze()
    expected_scale = amax / 448
    assert (expected_scale == 0).any()
    if layout != "tensor":
        assert (expected_scale > 0).any()

    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=(4, 3), axis=axis, block_sizes=block_sizes, fake_quant=False
        )
    ).cuda()
    if precomputed_amax:
        # Calibrated block amax may have the right element count but a different shape.
        quantizer.amax = amax.reshape(-1) if block_sizes else amax
    qtensor = quantizer(weight)
    assert isinstance(qtensor, FP8QTensor)
    payload = qtensor._quantized_data
    assert payload.shape == weight.shape
    assert payload.dtype == torch.float8_e4m3fn
    assert torch.isfinite(payload.float()).all()
    assert torch.count_nonzero(payload.float()[weight == 0]) == 0
    torch.testing.assert_close(quantizer._scale, expected_scale, rtol=0, atol=0)

    reference = cuda_ext.fake_e4m3fy_with_axis(groups.contiguous(), group_amax, 0)
    if block_sizes:
        reference = (
            reference.reshape(*block_shape, block_height, 8)
            .permute(0, 2, 1, 3)
            .reshape(padded.shape)[: weight.shape[0], : weight.shape[1]]
        )
    else:
        reference = reference.reshape(weight.shape)
    restored = quantizer(qtensor)
    assert restored.dtype == dtype
    torch.testing.assert_close(restored, reference, rtol=0, atol=0)

    if layout == "block_2d":
        exported = to_quantized_weight(
            weight, quantizer._scale, QUANTIZATION_FP8_PB_WO, block_size=8
        )
        torch.testing.assert_close(exported.view(torch.uint8), payload.view(torch.uint8))
