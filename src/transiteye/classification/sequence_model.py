"""Experimental PyTorch model for candidate-centered light-curve sequences.

Import this module only from the optional sequence-model workflow. The production
classifier and backend do not require PyTorch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch import Tensor, nn


def build_candidate_sequence(
    time_days: NDArray[np.float64],
    delivered_flux: NDArray[np.float64],
    detrended_flux: NDArray[np.float64],
    epoch_days: float,
    period_days: float,
    duration_hours: float,
    bins: int = 256,
    half_window_in_durations: float = 5.0,
) -> NDArray[np.float32]:
    """Fold a candidate-centered event window into a fixed three-channel array.

    The candidate ephemeris is used only to align the observations. It is not
    encoded as an input feature. Flux channels are median-centered, scaled by a
    robust MAD estimate, clipped, then summarized by the median in each bin.
    The final channel records relative bin occupancy, with zero marking gaps.
    """
    time = np.asarray(time_days, dtype=np.float64)
    delivered = np.asarray(delivered_flux, dtype=np.float64)
    detrended = np.asarray(detrended_flux, dtype=np.float64)
    if time.ndim != 1 or delivered.shape != time.shape or detrended.shape != time.shape:
        raise ValueError("Time and both flux arrays must be one-dimensional and aligned.")
    if len(time) == 0 or bins < 16:
        raise ValueError("At least one cadence and 16 sequence bins are required.")
    if not np.isfinite(period_days) or period_days <= 0:
        raise ValueError("period_days must be finite and positive.")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError("duration_hours must be finite and positive.")
    if not np.isfinite(epoch_days) or not np.isfinite(half_window_in_durations):
        raise ValueError("Epoch and window size must be finite.")
    if half_window_in_durations <= 0:
        raise ValueError("half_window_in_durations must be positive.")

    duration_days = duration_hours / 24.0
    relative_days = (time - epoch_days + 0.5 * period_days) % period_days - 0.5 * period_days
    relative_duration = relative_days / duration_days
    valid = (
        np.isfinite(relative_duration)
        & np.isfinite(delivered)
        & np.isfinite(detrended)
        & (np.abs(relative_duration) <= half_window_in_durations)
    )
    if not np.any(valid):
        raise ValueError("No valid cadences fall inside the candidate-centered sequence window.")

    window_positions = (relative_duration[valid] + half_window_in_durations) / (
        2.0 * half_window_in_durations
    )
    bin_indices = np.minimum((window_positions * bins).astype(np.int64), bins - 1)
    output = np.zeros((3, bins), dtype=np.float32)
    counts = np.bincount(bin_indices, minlength=bins)

    for channel, flux in enumerate((delivered[valid], detrended[valid])):
        center = float(np.median(flux))
        scale = float(1.4826 * np.median(np.abs(flux - center)))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            scale = float(np.std(flux))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            scale = 1.0
        scaled = np.clip((flux - center) / scale, -10.0, 10.0)
        for bin_index in np.unique(bin_indices):
            output[channel, bin_index] = np.median(scaled[bin_indices == bin_index])

    max_count = int(counts.max())
    if max_count:
        output[2] = counts.astype(np.float32) / max_count
    return output


class _ResidualBlock(nn.Module):
    """Small one-dimensional residual block with an optional downsampling path."""

    def __init__(self, input_channels: int, output_channels: int, stride: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(8, output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, output_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.main(values) + self.skip(values))


class CandidateSequenceCNN(nn.Module):
    """Classify fixed-width sequences centered on a detected candidate event.

    Inputs have shape ``(batch, 3, sequence_length)`` with channels for delivered
    normalized flux, detrended flux, and bin occupancy. Period, epoch, duration,
    target identity, catalog fields, and dataset split are not model inputs.
    """

    def __init__(self, class_count: int = 4, dropout: float = 0.25) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least two.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_channels = 3
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            _ResidualBlock(32, 32, stride=1, kernel_size=7),
            _ResidualBlock(32, 64, stride=2, kernel_size=7),
            _ResidualBlock(64, 128, stride=2, kernel_size=5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, class_count),
        )

    def forward(self, values: Tensor) -> Tensor:
        """Return uncalibrated class logits for a batch of candidate sequences."""
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("Expected input shaped (batch, 3, sequence_length).")
        encoded = self.encoder(values)
        if encoded.shape[-1] % 8 != 0:
            raise ValueError("Encoded sequence length must be divisible into eight segments.")
        pooled_segments = encoded.reshape(
            encoded.shape[0], encoded.shape[1], 8, encoded.shape[-1] // 8
        ).mean(dim=-1).flatten(start_dim=1)
        return self.classifier(pooled_segments)
