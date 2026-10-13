"""Small timm equivalents used by the pinned encoder, with no extra dependency.

All weights are loaded strictly before frozen inference. DropPath is inactive
in eval mode; torch's initializer is only used before checkpoint loading.
"""

import torch
from torch import nn


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


trunc_normal_ = nn.init.trunc_normal_


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, values):
        if self.drop_prob == 0.0 or not self.training:
            return values
        keep = 1.0 - self.drop_prob
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=values.dtype, device=values.device)).floor()
        return values * mask / keep
