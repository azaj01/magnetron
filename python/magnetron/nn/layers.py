# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math

from .. import Tensor, dtype, context
from .. import dtype as _dtype
from magnetron.nn.module import Module, Parameter

class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous().reshape(x.shape[0], -1)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: dtype.DType | None = None, init: bool = True) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.is_scaled: bool = dtype == _dtype.float8_e4m3fn
        if init:
            self.weight: Parameter = Parameter(Tensor.normal(out_features, in_features, mean=0.0, std=1.0, dtype=dtype) / math.sqrt(in_features + out_features))
        else:
            self.weight: Parameter = Parameter(Tensor.empty(out_features, in_features, dtype=dtype))
        self.bias: Parameter | None = None
        if bias:
            self.bias: Parameter | None = Parameter(Tensor.zeros(out_features, dtype=dtype))
        if self.is_scaled:
            self.weight_scale: Parameter | None = Parameter(Tensor.ones(1, dtype=_dtype.float32))

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.T
        x = x.scaled_matmul(w, self.weight_scale) if self.is_scaled else x.matmul(w)
        if self.bias is not None:
            x = x + self.bias
        return x


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: dtype.DType | None = None, init: bool = True) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        if init:
            self.weight: Parameter = Parameter(Tensor.normal(num_embeddings, embedding_dim, dtype=dtype) / embedding_dim)
        else:
            self.weight: Parameter = Parameter(Tensor.empty(num_embeddings, embedding_dim, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight[x]


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: dtype.DType | None = None, init: bool = True) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.eps: float = eps
        self.weight: Parameter = Parameter(Tensor.ones(dim, dtype=dtype) if init else Tensor.empty(dim, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        rms = (x.sqr().mean(dim=-1, keepdim=True) + self.eps).sqrt_()
        return (x / rms) * self.weight


class LayerNorm(Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5, dtype: dtype.DType | None = None, init: bool = True) -> None:
        super().__init__()
        if dtype is None:
            dtype = context.get_default_dtype()
        self.weight: Parameter = Parameter(Tensor.ones(ndim, dtype=dtype) if init else Tensor.empty(ndim, dtype=dtype))
        self.bias: Parameter | None = Parameter(Tensor.zeros(ndim, dtype=dtype) if init else Tensor.empty(ndim, dtype=dtype)) if bias else None
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        xm = x - mean
        var = xm.sqr().mean(dim=-1, keepdim=True)
        x_hat = xm * (var + self.eps).rsqrt()
        y = self.weight * x_hat
        if self.bias is not None:
            y = y + self.bias
        return y
