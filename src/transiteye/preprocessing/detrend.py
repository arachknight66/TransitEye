"""Robust duration-aware Tukey biweight detrending with CUDA acceleration and two-pass transit masking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wotan
from numpy.typing import NDArray

from transiteye.execution.compute import has_cuda, resolve_device
from transiteye.preprocessing.representation import PreparedLightCurve


@dataclass(frozen=True, slots=True)
class DualLightCurve:
    """Delivered and detrended representations of a light curve."""

    prepared: PreparedLightCurve
    detrended_flux: NDArray[np.float64]
    trend_flux: NDArray[np.float64]
    in_transit_mask: NDArray[np.bool_]
    window_days: float
    pass_number: int

    def __post_init__(self) -> None:
        length = len(self.prepared.time_days)
        if len(self.detrended_flux) != length or len(self.trend_flux) != length:
            raise ValueError("Detrended and trend arrays must match prepared light-curve length.")
        if len(self.in_transit_mask) != length:
            raise ValueError("in_transit_mask must match prepared light-curve length.")
        if self.pass_number not in (1, 2):
            raise ValueError("pass_number must be 1 or 2.")


_CUDA_BIWEIGHT_MODULE = None


def _get_cuda_biweight_kernel():
    global _CUDA_BIWEIGHT_MODULE
    if _CUDA_BIWEIGHT_MODULE is not None:
        return _CUDA_BIWEIGHT_MODULE
    import cupy as cp

    code = r'''
    __device__ void sort_small(double* arr, int count) {
        // Shell sort for window sizes up to 2048
        for (int gap = count / 2; gap > 0; gap /= 2) {
            for (int i = gap; i < count; ++i) {
                double temp = arr[i];
                int j = i;
                while (j >= gap && arr[j - gap] > temp) {
                    arr[j] = arr[j - gap];
                    j -= gap;
                }
                arr[j] = temp;
            }
        }
    }

    extern "C" __global__
    void biweight_filter(
        const double* __restrict__ t,
        const double* __restrict__ f,
        const bool* __restrict__ mask,
        double* __restrict__ trend,
        int n,
        double half_window,
        double c_tuning
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n) return;

        double t_cur = t[i];
        double t_min = t_cur - half_window;
        double t_max = t_cur + half_window;

        int start = i;
        while (start > 0 && t[start - 1] >= t_min) {
            start--;
        }
        int end = i;
        while (end < n - 1 && t[end + 1] <= t_max) {
            end++;
        }

        const int MAX_BUF = 2048;
        double buf[MAX_BUF];
        int span = end - start + 1;
        int stride = (span > MAX_BUF) ? ((span + MAX_BUF - 1) / MAX_BUF) : 1;

        int count = 0;
        for (int k = start; k <= end; k += stride) {
            if (mask != nullptr && mask[k]) continue;
            if (count < MAX_BUF) {
                buf[count++] = f[k];
            }
        }

        if (count == 0) {
            trend[i] = f[i];
            return;
        }

        sort_small(buf, count);
        double median = (count % 2 == 1) ? buf[count / 2] : 0.5 * (buf[count / 2 - 1] + buf[count / 2]);

        for (int p = 0; p < count; ++p) {
            buf[p] = fabs(buf[p] - median);
        }
        sort_small(buf, count);
        double mad = (count % 2 == 1) ? buf[count / 2] : 0.5 * (buf[count / 2 - 1] + buf[count / 2]);

        double denom = c_tuning * mad;
        if (denom <= 1e-12) {
            trend[i] = median;
            return;
        }

        double num_sum = 0.0;
        double weight_sum = 0.0;
        for (int k = start; k <= end; k += stride) {
            if (mask != nullptr && mask[k]) continue;
            double val = f[k];
            double u = (val - median) / denom;
            if (fabs(u) < 1.0) {
                double w = (1.0 - u * u);
                w = w * w;
                num_sum += (val - median) * w;
                weight_sum += w;
            }
        }

        if (weight_sum > 0.0) {
            trend[i] = median + (num_sum / weight_sum);
        } else {
            trend[i] = median;
        }
    }
    '''
    mod = cp.RawModule(code=code)
    _CUDA_BIWEIGHT_MODULE = mod.get_function("biweight_filter")
    return _CUDA_BIWEIGHT_MODULE


def detrend_biweight(
    time_days: NDArray[np.float64],
    flux: NDArray[np.float64],
    segment_slices: tuple[tuple[int, int], ...],
    window_days: float = 0.75,
    mask: NDArray[np.bool_] | None = None,
    device: str = "auto",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Detrend flux using Tukey's biweight time-window filter per segment.

    Returns (detrended_flux, trend_flux).
    """
    dev = resolve_device(device)
    trend = np.ones_like(flux)

    if dev == "cuda" and has_cuda():
        import cupy as cp

        kernel = _get_cuda_biweight_kernel()
        t_gpu = cp.asarray(time_days)
        f_gpu = cp.asarray(flux)
        m_gpu = cp.asarray(mask) if mask is not None else None
        trend_gpu = cp.empty_like(f_gpu)

        for start, end in segment_slices:
            seg_len = end - start
            if seg_len <= 1:
                trend[start:end] = flux[start:end]
                continue
            block = 256
            grid = (seg_len + block - 1) // block
            kernel(
                (grid,),
                (block,),
                (
                    t_gpu[start:end],
                    f_gpu[start:end],
                    m_gpu[start:end] if m_gpu is not None else None,
                    trend_gpu[start:end],
                    seg_len,
                    window_days / 2.0,
                    6.0,
                ),
            )
        cp.cuda.Stream.null.synchronize()
        trend = cp.asnumpy(trend_gpu)
    else:
        # CPU path using wotan per segment
        for start, end in segment_slices:
            seg_len = end - start
            if seg_len <= 3:
                trend[start:end] = flux[start:end]
                continue
            seg_t = time_days[start:end]
            seg_f = flux[start:end]
            seg_m = mask[start:end] if mask is not None else None

            # If all or nearly all points are masked, trend fallback to median
            if seg_m is not None and np.count_nonzero(~seg_m) < 3:
                trend[start:end] = np.median(seg_f)
                continue

            _, seg_trend = wotan.flatten(
                seg_t,
                seg_f,
                method="biweight",
                window_length=window_days,
                mask=seg_m,
                return_trend=True,
                edge_cutoff=0.5 * window_days,
            )
            # Interpolate trend over masked points if necessary
            if np.any(np.isnan(seg_trend)):
                valid_trend = ~np.isnan(seg_trend)
                if np.any(valid_trend):
                    seg_trend = np.interp(seg_t, seg_t[valid_trend], seg_trend[valid_trend])
                else:
                    seg_trend = np.full_like(seg_t, np.median(seg_f))
            trend[start:end] = seg_trend

    # Guard against zero or negative trend values
    trend = np.where(trend <= 0.0, 1.0, trend)
    detrended = flux / trend
    return detrended, trend


def build_in_transit_mask(
    time_days: NDArray[np.float64],
    candidate_events: tuple[tuple[float, float, float], ...],
    buffer_factor: float = 1.5,
) -> NDArray[np.bool_]:
    """Build in-transit boolean mask for candidate events.

    Each event tuple is (epoch_days, period_days, duration_hours).
    """
    mask = np.zeros(len(time_days), dtype=np.bool_)
    for epoch, period, dur_hours in candidate_events:
        if period <= 0 or dur_hours <= 0:
            continue
        dur_days = dur_hours / 24.0
        half_window = 0.5 * dur_days * buffer_factor
        phase = ((time_days - epoch + 0.5 * period) % period) - 0.5 * period
        mask |= np.abs(phase) <= half_window
    return mask


def two_pass_detrend(
    prepared: PreparedLightCurve,
    baseline_window_days: float = 0.75,
    candidate_events: tuple[tuple[float, float, float], ...] | None = None,
    device: str = "auto",
) -> DualLightCurve:
    """Execute two-pass duration-aware robust detrending.

    Pass 1: Initial detrend with baseline window and no transit mask.
    Pass 2: Mask candidate events and recompute trend with duration-scaled window.
    """
    if candidate_events is None or len(candidate_events) == 0:
        # Pass 1
        mask = np.zeros(len(prepared.time_days), dtype=np.bool_)
        detrended, trend = detrend_biweight(
            prepared.time_days,
            prepared.flux_normalized,
            prepared.segment_slices,
            window_days=baseline_window_days,
            mask=None,
            device=device,
        )
        return DualLightCurve(
            prepared=prepared,
            detrended_flux=detrended,
            trend_flux=trend,
            in_transit_mask=mask,
            window_days=baseline_window_days,
            pass_number=1,
        )

    # Pass 2
    mask = build_in_transit_mask(prepared.time_days, candidate_events, buffer_factor=1.5)
    max_duration_hours = max(event[2] for event in candidate_events)
    max_duration_days = max_duration_hours / 24.0
    pass2_window = max(3.0 * max_duration_days, 0.5)

    detrended, trend = detrend_biweight(
        prepared.time_days,
        prepared.flux_normalized,
        prepared.segment_slices,
        window_days=pass2_window,
        mask=mask,
        device=device,
    )
    return DualLightCurve(
        prepared=prepared,
        detrended_flux=detrended,
        trend_flux=trend,
        in_transit_mask=mask,
        window_days=pass2_window,
        pass_number=2,
    )
