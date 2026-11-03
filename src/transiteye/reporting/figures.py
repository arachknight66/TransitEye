"""Publication-grade Matplotlib diagnostic vetting sheets and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def generate_vetting_sheet(
    bundle_path: Path | str,
    output_file: Path | str,
    candidate_index: int = 0,
) -> Path:
    """Generate a multi-panel diagnostic vetting sheet for a target candidate."""
    b_path = Path(bundle_path).expanduser().resolve()
    out_path = Path(output_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load result.json
    with open(b_path / "result.json", encoding="utf-8") as f:
        meta = json.load(f)

    # Load arrays.npz
    with np.load(b_path / "arrays.npz") as npz:
        times = npz["time_days"] if "time_days" in npz else npz["time"]
        raw_flux = (
            npz["flux_raw"]
            if "flux_raw" in npz
            else (npz["raw_flux"] if "raw_flux" in npz else npz["flux_detrended"])
        )
        det_flux = npz["flux_detrended"] if "flux_detrended" in npz else npz["flux"]
        trend = (
            npz["trend_flux"]
            if "trend_flux" in npz
            else (npz["trend"] if "trend" in npz else np.ones_like(times))
        )
        bls_periods = npz["bls_periods"] if "bls_periods" in npz else np.array([])
        bls_powers = npz["bls_powers"] if "bls_powers" in npz else np.array([])
        tpf_stamp = (
            npz[f"cand_{candidate_index:02d}_tpf_stamp"]
            if f"cand_{candidate_index:02d}_tpf_stamp" in npz
            else (npz["tpf_stamp"] if "tpf_stamp" in npz else None)
        )
        tpf_diff = (
            npz[f"cand_{candidate_index:02d}_tpf_diff"]
            if f"cand_{candidate_index:02d}_tpf_diff" in npz
            else (npz["tpf_diff"] if "tpf_diff" in npz else None)
        )
        cand_model = (
            npz[f"cand_{candidate_index:02d}_batman_model"]
            if f"cand_{candidate_index:02d}_batman_model" in npz
            else (
                npz[f"cand_{candidate_index:02d}_trapezoid_model"]
                if f"cand_{candidate_index:02d}_trapezoid_model" in npz
                else None
            )
        )

    # Load candidates
    candidates: list[dict[str, Any]] = []
    parquet_path = b_path / "candidates.parquet"
    if parquet_path.exists():
        candidates = pq.read_table(parquet_path).to_pylist()

    cand = candidates[candidate_index] if candidate_index < len(candidates) else {}

    period = float(cand.get("period_days", 1.0))
    epoch = float(cand.get("epoch_days", times[0] if len(times) else 0.0))
    dur_h = float(cand.get("duration_hours", 2.0))
    depth_ppm = float(
        cand.get("measured_depth_ppm")
        if cand.get("measured_depth_ppm") is not None
        else cand.get("depth_ppm", 1000.0)
    )
    dil_depth_ppm = cand.get("dilution_corrected_depth_ppm")
    snr = float(cand.get("detection_snr", 0.0))
    sig = cand.get("search_significance")
    pred_cls = str(cand.get("predicted_class", "unknown"))
    conf = cand.get("calibrated_confidence")
    sig_text = f"{float(sig):.2f}" if sig is not None and np.isfinite(float(sig)) else "unavailable"
    conf_text = f"{float(conf) * 100.0:.1f}% conf" if conf is not None and np.isfinite(float(conf)) else "unavailable"
    abstained = bool(cand.get("abstained", False))
    vetting = str(cand.get("source_assessment", "not_assessed"))

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    target_id = meta.get("target_id") or (meta.get("provenance") or {}).get("target_id") or "Unknown"

    # 1. Full Light Curve
    ax_lc = fig.add_subplot(gs[0, :])
    flux_column = (meta.get("provenance") or {}).get("metadata", {}).get("selected_flux") or "delivered flux"
    ax_lc.plot(times, raw_flux, ".", color="#9E9E9E", markersize=1.5, alpha=0.5, label=f"Normalized {flux_column}")
    ax_lc.plot(times, trend, "-", color="#E53935", linewidth=1.5, label="Biweight Trend")
    ax_lc.set_title(f"Target TIC {target_id} — Full Sector Light Curve", fontsize=12, fontweight="bold")
    ax_lc.set_xlabel("Time (BJD - 2457000)")
    ax_lc.set_ylabel("Normalized Flux")
    ax_lc.legend(loc="upper right", frameon=True)
    ax_lc.grid(True, alpha=0.3)

    # 2. BLS Periodogram
    ax_bls = fig.add_subplot(gs[1, 0])
    if len(bls_periods) > 0 and len(bls_powers) > 0:
        ax_bls.semilogx(bls_periods, bls_powers, color="#1E88E5", linewidth=1.2)
        ax_bls.axvline(period, color="#D81B60", linestyle="--", linewidth=1.5, label=f"P = {period:.4f} d")
        ax_bls.axvline(period / 2.0, color="#FB8C00", linestyle=":", linewidth=1.0, label="P/2 alias")
        ax_bls.axvline(period * 2.0, color="#FB8C00", linestyle=":", linewidth=1.0, label="2P alias")
    ax_bls.set_title("BLS Search Periodogram", fontsize=11, fontweight="bold")
    ax_bls.set_xlabel("Period (days)")
    ax_bls.set_ylabel("BLS Power (SNR²)")
    ax_bls.legend(loc="upper right", frameon=True, fontsize=9)
    ax_bls.grid(True, alpha=0.3)

    # 3. Phase-Folded Transit Profile
    ax_fold = fig.add_subplot(gs[1, 1])
    if len(times) > 0:
        phase_days = (times - epoch + 0.5 * period) % period - 0.5 * period
        phase_hours = phase_days * 24.0
        window_h = max(2.0, 3.5 * dur_h)
        mask = (phase_hours >= -window_h) & (phase_hours <= window_h)

        if np.any(mask):
            w_phase = phase_hours[mask]
            w_flux = det_flux[mask]
            ax_fold.plot(w_phase, w_flux, ".", color="#90CAF9", markersize=3, alpha=0.6, label="Observed")

            # Binned median
            nbins = 30
            bin_edges = np.linspace(-window_h, window_h, nbins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_vals = []
            valid_c = []
            for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], bin_centers, strict=False):
                m = (w_phase >= lo) & (w_phase < hi)
                if np.any(m):
                    bin_vals.append(float(np.median(w_flux[m])))
                    valid_c.append(c)
            if valid_c:
                ax_fold.plot(valid_c, bin_vals, "s-", color="#0D47A1", markersize=5, linewidth=1.5, label="Binned (Median)")

            # Model curve
            if cand_model is not None and len(cand_model) == len(times):
                w_model = cand_model[mask]
                order = np.argsort(w_phase)
                ax_fold.plot(
                    w_phase[order],
                    w_model[order],
                    "-",
                    color="#E53935",
                    linewidth=2.0,
                    label="Fitted Transit Model",
                )
            else:
                m_x = np.linspace(-window_h, window_h, 200)
                half_dur = dur_h / 2.0
                m_y = np.ones_like(m_x)
                m_y[np.abs(m_x) <= half_dur] = 1.0 - (depth_ppm * 1e-6)
                ax_fold.plot(m_x, m_y, "-", color="#E53935", linewidth=2.0, label="Trapezoid Fit")

    ax_fold.set_title("Phase-Folded Transit Profile", fontsize=11, fontweight="bold")
    ax_fold.set_xlabel("Time from Mid-Transit (hours)")
    ax_fold.set_ylabel("Relative Detrended Flux")
    ax_fold.legend(loc="lower right", frameon=True, fontsize=9)
    ax_fold.grid(True, alpha=0.3)

    # 4. Source Localization / Difference Image
    ax_vet = fig.add_subplot(gs[2, 0])
    if tpf_stamp is not None and tpf_diff is not None and tpf_diff.ndim == 2:
        im = ax_vet.imshow(tpf_diff, origin="lower", cmap="coolwarm")
        fig.colorbar(im, ax=ax_vet, label="Difference (OOT - In)", fraction=0.046, pad=0.04)
        ax_vet.set_title("TPF In/Out-of-Transit Difference Image", fontsize=11, fontweight="bold")
    else:
        # Fallback text card if no TPF
        ax_vet.text(0.5, 0.5, "Tier 1 Light-Curve Centroid Only\n(No Companion TPF Supplied)", ha="center", va="center", fontsize=11, color="#616161")
        ax_vet.set_title("Spatial Source Localization", fontsize=11, fontweight="bold")
        ax_vet.set_xticks([])
        ax_vet.set_yticks([])

    # 5. Candidate Metadata & Classification Summary Card
    ax_meta = fig.add_subplot(gs[2, 1])
    ax_meta.axis("off")

    dil_str = (
        f"{float(dil_depth_ppm):.1f} ppm"
        if dil_depth_ppm is not None and not np.isnan(float(dil_depth_ppm))
        else "N/A"
    )
    shift = cand.get("centroid_shift_sigma")
    crowd = cand.get("crowdsap_value")
    shift_text = f"{float(shift):.2f} σ" if shift is not None and np.isfinite(float(shift)) else "unavailable"
    crowd_text = f"{float(crowd):.3f}" if crowd is not None and np.isfinite(float(crowd)) else "unavailable"
    summary_text = (
        f"CANDIDATE CHARACTERIZATION & VETTING\n"
        f"────────────────────────────────────────────\n"
        f"• Candidate ID:        {cand.get('candidate_id', 'cand_01')}\n"
        f"• Period:              {period:.5f} days\n"
        f"• Epoch (T0):          {epoch:.4f} BJD-2457000\n"
        f"• Measured Depth:      {depth_ppm:.1f} ppm (Dilution-corr: {dil_str})\n"
        f"• Transit Duration:    {dur_h:.2f} hours\n"
        f"• Detection SNR:       {snr:.1f}\n"
        f"• Search Significance: {sig_text}\n"
        f"• Morphology Class:    {pred_cls.upper()} ({conf_text})\n"
        f"• Abstention:          {'YES' if abstained else 'NO'}\n"
        f"• Source Attribution:  {vetting.upper()}\n"
        f"• Centroid Shift:      {shift_text}\n"
        f"• CROWDSAP:            {crowd_text}\n"
    )
    ax_meta.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_meta.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#F5F5F5", "edgecolor": "#CCCCCC"},
    )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
