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

import pytest
import torch

from modelopt.torch.quantization.extensions import get_cuda_ext_mx
from modelopt.torch.quantization.qtensor import MXFP4QTensor
from modelopt.torch.quantization.tensor_quant import mx_format_map


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mxfp4_packed_rounding_matches_cuda(dtype):
    """Packed E2M1 codes agree with CUDA fake quantization at ties and signed zeros."""
    cuda_ext_mx = get_cuda_ext_mx(raise_if_failed=True)
    # Include every midpoint, exact representable values, and both signs of zero.
    values = torch.tensor(
        [0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, 0.5, 1, 1.5, 2, 3, 4, 0.125],
        dtype=dtype,
        device="cuda",
    )
    values = torch.stack((values, -values), dim=-1).flatten()
    scale_factors = torch.tensor([0.125, 1, 16], dtype=dtype, device="cuda").unsqueeze(-1)
    weight = torch.cat((values * scale_factors, torch.zeros(1, 32, dtype=dtype, device="cuda")))

    qtensor, scale = MXFP4QTensor.quantize(weight, block_size=32)
    dequantized = qtensor.dequantize(dtype=dtype, scale=scale, block_sizes={-1: 32})
    simulated = cuda_ext_mx.fused_amax_convert(
        weight,
        32,
        getattr(cuda_ext_mx.Types, mx_format_map[(2, 1)]),
        getattr(cuda_ext_mx.Types, mx_format_map[(8, 0)]),
        None,
    )

    torch.testing.assert_close(dequantized, simulated, rtol=0, atol=0)
    torch.testing.assert_close(
        scale,
        torch.tensor([[124], [127], [131], [0]], dtype=torch.uint8, device="cuda"),
        rtol=0,
        atol=0,
    )
    magnitudes = torch.tensor(
        [0, 0, 2, 2, 4, 4, 6, 6, 7, 1, 2, 3, 4, 5, 6, 0], dtype=torch.uint8, device="cuda"
    )
    codes = torch.stack((magnitudes, magnitudes + 8), dim=-1).flatten()
    codes[1] = 0
    packed = codes[0::2] | (codes[1::2] << 4)
    expected_packed = torch.cat((packed.expand(3, -1), torch.zeros_like(packed).unsqueeze(0)))
    assert qtensor._quantized_data.dtype == torch.uint8
    torch.testing.assert_close(qtensor._quantized_data, expected_packed, rtol=0, atol=0)
