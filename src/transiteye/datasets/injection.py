"""Controlled synthetic augmentation and recovery dataset generation.

Injects exposure-integrated planetary transits, grazing/blended eclipsing binaries,
and rotational/pulsational sinusoids strictly into TRAIN-partition substrates with
deterministic seeds and full provenance lineage.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import batman
import numpy as np
from numpy.typing import NDArray

from transiteye.characterization.fitting import evaluate_trapezoid
from transiteye.datasets.splits import DatasetSplit
from transiteye.domain.models import LightCurve, MorphologyClass, Provenance


@dataclass(frozen=True, slots=True)
class SyntheticInjection:
    """Lineage and parameter metadata for an injected synthetic signal."""

    synthetic_id: str
    parent_source_group_id: str
    parent_target_id: str
    injection_kind: str
    target_class: MorphologyClass
    parameters: dict[str, float]
    seed: int
    is_synthetic: bool = True


@dataclass(frozen=True, slots=True)
class InjectedLightCurve:
    """Delivered substrate light curve combined with a synthetic astrophysical signal."""

    injection: SyntheticInjection
    light_curve: LightCurve
    synthetic_signal_flux: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_deterministic_seed(
    source_group_id: str,
    variant_idx: int,
    injection_kind: str,
) -> int:
    """Generate a reproducible pseudo-random seed from source group and variant index."""
    token = f"{source_group_id}:{variant_idx}:{injection_kind}".encode()
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def synthesize_transit_signal(
    time_days: NDArray[np.float64],
    period_days: float,
    epoch_days: float,
    duration_hours: float,
    depth_ppm: float,
    u_limb_darkening: tuple[float, float] = (0.3, 0.2),
) -> NDArray[np.float64]:
    """Generate an exposure-integrated quadratic limb-darkened planetary transit signal."""
    # Use batman model
    depth_frac = depth_ppm * 1e-6
    rp_rs = math.sqrt(max(1e-6, depth_frac))

    dur_days = duration_hours / 24.0
    # a/Rs approximation from duration for circular orbit (b=0)
    # T = (P / pi) * (Rs / a) -> a/Rs = P / (pi * T)
    a_rs = max(2.0, period_days / (math.pi * dur_days))

    params = batman.TransitParams()
    params.t0 = epoch_days
    params.per = period_days
    params.rp = rp_rs
    params.a = a_rs
    params.inc = 90.0  # b = 0
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(u_limb_darkening)
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, time_days, supersample_factor=5, exp_time=2.0 / 1440.0)
    signal = m.light_curve(params)
    return np.asarray(signal, dtype=np.float64)


def synthesize_eclipsing_binary_signal(
    time_days: NDArray[np.float64],
    period_days: float,
    epoch_days: float,
    duration_hours: float,
    primary_depth_ppm: float,
    secondary_depth_ratio: float = 0.5,
    secondary_phase: float = 0.5,
    ingress_fraction: float = 0.9,  # V-shaped ingress
    ellipsoidal_amplitude_ppm: float = 0.0,
) -> NDArray[np.float64]:
    """Generate an eclipsing binary signal with primary, secondary, and tidal ellipsoidal variations."""
    dur_days = duration_hours / 24.0
    d_pri = primary_depth_ppm * 1e-6
    d_sec = d_pri * secondary_depth_ratio

    # Primary eclipse
    m_pri = evaluate_trapezoid(
        time_days,
        epoch_days=epoch_days,
        period_days=period_days,
        duration_days=dur_days,
        depth=d_pri,
        ingress_fraction=ingress_fraction,
    )

    # Secondary eclipse at epoch + secondary_phase * period
    m_sec = evaluate_trapezoid(
        time_days,
        epoch_days=epoch_days + secondary_phase * period_days,
        period_days=period_days,
        duration_days=dur_days,
        depth=d_sec,
        ingress_fraction=ingress_fraction,
    )

    # Combine eclipses (relative to 1.0)
    eb_signal = np.ones_like(time_days) - (1.0 - m_pri) - (1.0 - m_sec)

    # Add ellipsoidal cosine variation if requested: -A * cos(4*pi*(t - t0)/P)
    if ellipsoidal_amplitude_ppm > 0:
        phases = ((time_days - epoch_days) % period_days) / period_days
        ellip_frac = ellipsoidal_amplitude_ppm * 1e-6
        eb_signal -= ellip_frac * np.cos(4.0 * np.pi * phases)

    return eb_signal


def synthesize_variability_signal(
    time_days: NDArray[np.float64],
    period_days: float,
    amplitude_ppm: float,
    phase_offset: float = 0.0,
    harmonic_ratio: float = 0.25,
) -> NDArray[np.float64]:
    """Generate stellar rotational or pulsational sinusoid modulation."""
    amp_frac = amplitude_ppm * 1e-6
    omega = 2.0 * np.pi / period_days
    t_phase = time_days + phase_offset * period_days
    # Fundamental + second harmonic
    signal = 1.0 + amp_frac * (
        np.sin(omega * t_phase) + harmonic_ratio * np.sin(2.0 * omega * t_phase)
    )
    return signal


def inject_synthetic_signal(
    substrate: LightCurve,
    parent_source_group_id: str,
    split: DatasetSplit,
    variant_idx: int,
    injection_kind: str,
    period_days: float | None = None,
    epoch_days: float | None = None,
    duration_hours: float | None = None,
    depth_ppm: float | None = None,
) -> InjectedLightCurve:
    """Inject a synthetic signal strictly into a TRAIN substrate.

    Raises ValueError if split is not TRAIN to enforce strict partition isolation.
    """
    if split is not DatasetSplit.TRAIN:
        raise ValueError(
            f"Synthetic injection is strictly prohibited on non-train partitions. "
            f"Attempted injection on partition '{split.value}'."
        )

    seed = compute_deterministic_seed(parent_source_group_id, variant_idx, injection_kind)
    rng = np.random.default_rng(seed)

    time = substrate.time_days
    baseline_days = float(time[-1] - time[0])

    if period_days is None:
        period_days = float(rng.uniform(1.0, min(10.0, 0.4 * baseline_days)))
    if epoch_days is None:
        epoch_days = float(time[0] + rng.uniform(0.1, 0.9) * period_days)
    if duration_hours is None:
        duration_hours = float(rng.uniform(1.5, 5.0))
    if depth_ppm is None:
        depth_ppm = float(rng.uniform(200.0, 10000.0))

    params: dict[str, float] = {
        "period_days": period_days,
        "epoch_days": epoch_days,
        "duration_hours": duration_hours,
        "depth_ppm": depth_ppm,
    }

    if injection_kind == "transit":
        target_class = MorphologyClass.TRANSIT_LIKE
        signal = synthesize_transit_signal(
            time, period_days, epoch_days, duration_hours, depth_ppm
        )
    elif injection_kind == "grazing_eb":
        target_class = MorphologyClass.ECLIPSE_LIKE
        sec_ratio = float(rng.uniform(0.3, 0.9))
        params["secondary_depth_ratio"] = sec_ratio
        signal = synthesize_eclipsing_binary_signal(
            time,
            period_days,
            epoch_days,
            duration_hours,
            depth_ppm,
            secondary_depth_ratio=sec_ratio,
            ingress_fraction=0.9,
        )
    elif injection_kind == "variability":
        target_class = MorphologyClass.STELLAR_VARIABILITY
        signal = synthesize_variability_signal(time, period_days, depth_ppm)
    else:
        raise ValueError(f"Unknown injection kind: {injection_kind}")

    # Combine with substrate multiplicatively (flux normalized baseline)
    injected_flux = substrate.flux * signal

    synthetic_id = f"syn_{parent_source_group_id}_{variant_idx:02d}_{injection_kind}"
    target_id = substrate.provenance.target_id or parent_source_group_id

    injection_meta = SyntheticInjection(
        synthetic_id=synthetic_id,
        parent_source_group_id=parent_source_group_id,
        parent_target_id=target_id,
        injection_kind=injection_kind,
        target_class=target_class,
        parameters=params,
        seed=seed,
        is_synthetic=True,
    )

    new_provenance = Provenance(
        source_uri=f"synthetic://{synthetic_id}",
        source_path=substrate.provenance.source_path,
        checksum_sha256=substrate.provenance.checksum_sha256,
        product_type="synthetic_tess_light_curve",
        target_id=target_id,
        sector=substrate.provenance.sector,
        time_system=substrate.provenance.time_system,
        time_reference=substrate.provenance.time_reference,
        metadata={
            **substrate.provenance.metadata,
            "synthetic_id": synthetic_id,
            "parent_source_group_id": parent_source_group_id,
            "is_synthetic": "true",
        },
    )

    new_lc = LightCurve(
        time_days=substrate.time_days,
        flux=injected_flux,
        flux_error=substrate.flux_error,
        quality=substrate.quality,
        cadence_number=substrate.cadence_number,
        centroid_column=substrate.centroid_column,
        centroid_row=substrate.centroid_row,
        flux_kind=substrate.flux_kind,
        flux_unit=substrate.flux_unit,
        provenance=new_provenance,
        metadata=dict(substrate.metadata),
    )

    return InjectedLightCurve(
        injection=injection_meta,
        light_curve=new_lc,
        synthetic_signal_flux=signal,
    )
