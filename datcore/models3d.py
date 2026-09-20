"""Reusable 3D encoder blocks.

GroupNorm throughout, never BatchNorm: a fitted batch statistic can leak across
cases at inference time, and every case in this task must be scored independently.

Lifted unchanged from the Phase67 research code so a new experiment and the
archived phases build the same network from the same numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_count_for(channels):
    for candidate in (8, 4, 2):
        if int(channels) % candidate == 0:
            return candidate
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = group_count_for(channels)
        self.norm_one = nn.GroupNorm(groups, channels)
        self.conv_one = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm_two = nn.GroupNorm(groups, channels)
        self.conv_two = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, value):
        residual = self.conv_one(self.activation(self.norm_one(value)))
        residual = self.conv_two(self.activation(self.norm_two(residual)))
        return value + residual


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(group_count_for(in_channels), in_channels)
        self.activation = nn.SiLU(inplace=False)
        self.conv = nn.Conv3d(
            in_channels, out_channels, 3, stride=2, padding=1, bias=False
        )

    def forward(self, value):
        return self.conv(self.activation(self.norm(value)))


class MultiscaleEncoder3D(nn.Module):
    """Four GroupNorm stages; each stage summarized by concatenated average and
    max pooling, projected to a fixed width. No BatchNorm anywhere, so a fitted
    batch statistic can never leak across cases at inference time."""

    def __init__(self, input_channels, channels, blocks_per_stage, projection):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count_for(channels[0]), channels[0]),
            nn.SiLU(inplace=False),
        )
        stages = []
        projections = []
        current = channels[0]
        for index, width in enumerate(channels):
            layers = []
            if index > 0:
                layers.append(Downsample3D(current, width))
                current = width
            for _ in range(blocks_per_stage[index]):
                layers.append(ResidualBlock3D(current))
            stages.append(nn.Sequential(*layers))
            projections.append(
                nn.Sequential(
                    nn.Linear(2 * current, projection),
                    nn.LayerNorm(projection),
                    nn.SiLU(inplace=False),
                )
            )
        self.stages = nn.ModuleList(stages)
        self.projections = nn.ModuleList(projections)
        self.output_dimension = projection * len(channels)

    def forward(self, value):
        value = self.stem(value)
        summaries = []
        for stage, projection in zip(self.stages, self.projections):
            value = stage(value)
            average = F.adaptive_avg_pool3d(value, 1).flatten(1)
            maximum = F.adaptive_max_pool3d(value, 1).flatten(1)
            summaries.append(projection(torch.cat([average, maximum], dim=1)))
        return torch.cat(summaries, dim=1)
