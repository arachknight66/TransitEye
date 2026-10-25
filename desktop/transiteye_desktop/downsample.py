"""High-performance downsampling algorithms for interactive desktop visualization.

Downsamples raw time-series to target display resolutions (e.g. 2,000 to 5,000 points)
for 60fps pan/zoom in PyQtGraph without mutating the underlying numerical arrays.
"""

from __future__ import annotations

import numpy as np


def min_max_downsample(
    x: np.ndarray,
    y: np.ndarray,
    target_points: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample (x, y) preserving extreme minimum and maximum values in each bucket.

    This ensures narrow transit dips and high-frequency outliers remain visible
    even when zooming out to view a full 27-day TESS sector.
    """
    n = len(x)
    if n <= target_points or target_points < 4:
        return x, y

    bucket_size = int(np.ceil(n / (target_points // 2)))
    n_buckets = int(np.ceil(n / bucket_size))

    out_x = []
    out_y = []

    for i in range(n_buckets):
        start = i * bucket_size
        end = min(n, start + bucket_size)
        if start >= end:
            continue

        b_x = x[start:end]
        b_y = y[start:end]

        min_idx = int(np.nanargmin(b_y))
        max_idx = int(np.nanargmax(b_y))

        # Add in chronological order
        if min_idx < max_idx:
            out_x.extend([b_x[min_idx], b_x[max_idx]])
            out_y.extend([b_y[min_idx], b_y[max_idx]])
        elif min_idx > max_idx:
            out_x.extend([b_x[max_idx], b_x[min_idx]])
            out_y.extend([b_y[max_idx], b_y[min_idx]])
        else:
            out_x.append(b_x[min_idx])
            out_y.append(b_y[min_idx])

    return np.asarray(out_x, dtype=x.dtype), np.asarray(out_y, dtype=y.dtype)


def lttb_downsample(
    x: np.ndarray,
    y: np.ndarray,
    target_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest Triangle Three Buckets (LTTB) downsampling algorithm."""
    n = len(x)
    if n <= target_points or target_points < 3:
        return x, y

    # Bucket size
    bucket_size = (n - 2) / (target_points - 2)

    sampled_x = np.empty(target_points, dtype=x.dtype)
    sampled_y = np.empty(target_points, dtype=y.dtype)

    sampled_x[0] = x[0]
    sampled_y[0] = y[0]

    a = 0
    for i in range(target_points - 2):
        # Range of current bucket
        range_start = int(np.floor((i + 0) * bucket_size)) + 1
        range_end = int(np.floor((i + 1) * bucket_size)) + 1
        range_end = min(range_end, n)

        # Range of next bucket for average
        next_start = int(np.floor((i + 1) * bucket_size)) + 1
        next_end = int(np.floor((i + 2) * bucket_size)) + 1
        next_end = min(next_end, n)

        if next_start >= next_end:
            avg_x = x[-1]
            avg_y = y[-1]
        else:
            avg_x = float(np.mean(x[next_start:next_end]))
            avg_y = float(np.mean(y[next_start:next_end]))

        point_a_x = x[a]
        point_a_y = y[a]

        # Find point in current bucket with maximum triangle area
        max_area = -1.0
        max_idx = range_start

        for idx in range(range_start, range_end):
            # Triangle area: 0.5 * |(x_a - x_avg)*(y_idx - y_a) - (x_a - x_idx)*(y_avg - y_a)|
            area = abs(
                (point_a_x - avg_x) * (y[idx] - point_a_y)
                - (point_a_x - x[idx]) * (avg_y - point_a_y)
            )
            if area > max_area:
                max_area = area
                max_idx = idx

        sampled_x[i + 1] = x[max_idx]
        sampled_y[i + 1] = y[max_idx]
        a = max_idx

    sampled_x[-1] = x[-1]
    sampled_y[-1] = y[-1]
    return sampled_x, sampled_y
