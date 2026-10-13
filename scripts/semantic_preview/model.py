"""Inference-compatible InfraredIR WeTr/SegFormer-B3, without mmcv/timm.

Adapted from csmty/InfraredIR at 060a3c2d6b0407f1af1429efbbb643887fbcce64.
Decoder state-dict names, interpolation, normalization eps and feature order
match that implementation. BatchNorm and SyncBatchNorm are equivalent in eval.
SegFormer components: Copyright (c) 2021 NVIDIA; LICENSE_NVIDIA applies.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ._vendor.mix_transformer import mit_b3


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, feature):
        return self.proj(feature.flatten(2).transpose(1, 2))


class ConvModule(nn.Module):
    """The exact conv + norm + activation subset used by this frozen decoder."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.activate(self.bn(self.conv(feature)))


class SegFormerHead(nn.Module):
    def __init__(self, in_channels=(64, 128, 320, 512), embedding_dim=256, num_classes=15):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feature_strides = [4, 8, 16, 32]
        self.num_classes = num_classes
        for i, channels in enumerate(in_channels, 1):
            setattr(self, f"linear_c{i}", MLP(channels, embedding_dim))
        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = ConvModule(embedding_dim * 4, embedding_dim)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, features):
        destination = features[0].shape[-2:]
        projected = []
        for i in (4, 3, 2, 1):
            feature = features[i - 1]
            value = getattr(self, f"linear_c{i}")(feature).permute(0, 2, 1)
            value = value.reshape(feature.shape[0], -1, *feature.shape[-2:])
            if i != 1:
                value = F.interpolate(value, size=destination, mode="bilinear", align_corners=False)
            projected.append(value)
        fused = self.linear_fuse(torch.cat(projected, dim=1))
        return self.linear_pred(self.dropout(fused))


class InfraredSegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 15
        self.embedding_dim = 256
        self.encoder = mit_b3()
        self.decoder = SegFormerHead()
        # Kept for strict compatibility, although upstream forward ignores CAM.
        self.classifier = nn.Conv2d(512, 15, 1, bias=False)

    def forward(self, image):
        return self.decoder(self.encoder(image))
