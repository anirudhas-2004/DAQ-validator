# scoring_layer.py

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =========================
# Data models
# =========================

@dataclass
class ModeConfig:
    path: str
    stimulus_type: str  # "dc" or "sine"
    sample_rate_hz: float
    expected_dc_v: Optional[float] = None
    expected_sine_freq_hz: Optional[float] = None
    expected_sine_amplitude_v: Optional[float] = None
    full_scale_v: Optional[float] = None


@dataclass
class RunManifest:
    dc: ModeConfig
    sine: ModeConfig
    clip_fraction_limit: float = 0.01
    flatline_std_limit: float = 1e-9
    sine_harmonics: int = 5
    channel_names: Optional[List[str]] = None
    baseline_json: Optional[str] = None


@dataclass
class ModeResult:
    channel: str
    mean: float
    std: float
    rms: float
    peak_to_peak: float
    clip_fraction: float
    amplitude: Optional[float] = None
    phase_deg: Optional[float] = None
    sinad_db: Optional[float] = None
    thd_db: Optional[float] = None
    sfdr_db: Optional[float] = None
    score: float = 0.0
    reasons: List[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass
class ChannelSummary:
    channel: str
    dc: ModeResult
    sine: ModeResult
    combined_score: float
    verdict: str
    reasons: List[str]


# =========================
# IO
# =========================

def load_manifest(path: str | Path) -> RunManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def parse_mode(key: str) -> ModeConfig:
        m = data[key]
        return ModeConfig(
            path=m["path"],
            stimulus_type=m["stimulus_type"].lower(),
            sample_rate_hz=float(m["sample_rate_hz"]),
            expected_dc_v=None if m.get("expected_dc_v") is None else float(m["expected_dc_v"]),
            expected_sine_freq_hz=None if m.get("expected_sine_freq_hz") is None else float(m["expected_sine_freq_hz"]),
            expected_sine_amplitude_v=None if m.get("expected_sine_amplitude_v") is None else float(m["expected_sine_amplitude_v"]),
            full_scale_v=None if m.get("full_scale_v") is None else float(m["full_scale_v"]),
        )

    return RunManifest(
        dc=parse_mode("dc"),
        sine=parse_mode("sine"),
        clip_fraction_limit=float(data.get("clip_fraction_limit", 0.01)),
        flatline_std_limit=float(data.get("flatline_std_limit", 1e-9)),
        sine_harmonics=int(data.get("sine_harmonics", 5)),
        channel_names=data.get("channel_names"),
        baseline_json=data.get("baseline_json"),
    )


def load_matrix_csv(path: str | Path) -> Tuple[List[str], np.ndarray]:
    """
    Expected CSV format:
      - first row: channel names
      - each subsequent row: one sample across all channels
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader if row]

    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{path} did not parse into a 2D matrix")
    if data.shape[1] != len(header):
        raise ValueError(
            f"{path}: header has {len(header)} channels, data has {data.shape[1]} columns"
        )
    return header, data


def load_baseline(path: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =========================
# Math helpers
# =========================

def robust_z(x: float, median: float, mad: float) -> float:
    mad = max(mad, 1e-12)
    return 0.6745 * (x - median) / mad


def sine_fit(y: np.ndarray, fs: float, f0: float) -> Tuple[float, float, float, np.ndarray]:
    """
    Fit y ~= a*sin(wt) + b*cos(wt) + c.
    Returns amplitude, phase_deg, offset, fitted_waveform.
    """
    n = len(y)
    t = np.arange(n, dtype=np.float64) / fs
    w = 2.0 * np.pi * f0

    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = coeffs
    fitted = A @ coeffs

    amplitude = float(np.hypot(a, b))
    phase_deg = float(np.degrees(np.arctan2(b, a)))
    return amplitude, phase_deg, float(c), fitted


def fft_metrics(
    y: np.ndarray,
    fs: float,
    f0: Optional[float],
    harmonics: int = 5,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Approximate SINAD / THD / SFDR from a windowed FFT.
    Best for a single-tone sine stimulus.
    """
    n = len(y)
    if n < 16:
        return None, None, None

    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)

    window = np.hanning(n)
    yw = y * window
    spec = np.fft.rfft(yw)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if f0 is None or f0 <= 0:
        fund_idx = int(np.argmax(mag[1:]) + 1)
    else:
        fund_idx = int(np.argmin(np.abs(freqs - f0)))

    def band_power(center_idx: int, half_width: int = 1) -> float:
        lo = max(1, center_idx - half_width)
        hi = min(len(mag) - 1, center_idx + half_width)
        return float(np.sum(mag[lo:hi + 1] ** 2))

    fundamental_power = band_power(fund_idx, half_width=1)
    if fundamental_power <= 0:
        return None, None, None

    excluded = set(range(max(0, fund_idx - 1), min(len(mag), fund_idx + 2)))
    harm_powers = []

    for k in range(2, harmonics + 1):
        idx = int(round(fund_idx * k))
        if idx >= len(mag):
            break
        harm_powers.append(band_power(idx, half_width=1))
        excluded.update(range(max(0, idx - 1), min(len(mag), idx + 2)))

    noise_bins = [i for i in range(1, len(mag)) if i not in excluded]
    noise_power = float(np.sum(mag[noise_bins] ** 2)) if noise_bins else 0.0
    thd_power = float(np.sum(harm_powers)) if harm_powers else 0.0

    sinad = 10.0 * math.log10(fundamental_power / max(noise_power + thd_power, 1e-24))
    thd = 10.0 * math.log10(thd_power / max(fundamental_power, 1e-24)) if thd_power > 0 else None

    sfdr = None
    spur_mask = np.ones_like(mag, dtype=bool)
    spur_mask[0] = False
    for idx in excluded:
        if 0 <= idx < len(spur_mask):
            spur_mask[idx] = False
    spur_mag = mag[spur_mask]
    if spur_mag.size > 0:
        spur = float(np.max(spur_mag))
        sfdr = 20.0 * math.log10(np.sqrt(fundamental_power) / max(spur, 1e-24))

    return float(sinad), thd, sfdr


def compute_clip_fraction(y: np.ndarray, full_scale_v: Optional[float]) -> float:
    if full_scale_v is None or full_scale_v <= 0:
        peak = float(np.max(np.abs(y)))
        if peak <= 0:
            return 0.0
        return float(np.mean(np.abs(y) >= 0.98 * peak))
    return float(np.mean(np.abs(y) >= 0.98 * full_scale_v))


# =========================
# Per-mode analysis
# =========================

def analyze_dc_channel(
    name: str,
    y: np.ndarray,
    cfg: ModeConfig,
    flatline_std_limit: float,
    clip_fraction_limit: float,
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
    group_medians: Optional[Dict[str, float]] = None,
    group_mads: Optional[Dict[str, float]] = None,
) -> ModeResult:
    y = np.asarray(y, dtype=np.float64)

    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    rms = float(np.sqrt(np.mean(y ** 2)))
    p2p = float(np.max(y) - np.min(y))
    clip_fraction = compute_clip_fraction(y, cfg.full_scale_v)

    reasons: List[str] = []
    score = 0.0

    if not np.isfinite(y).all():
        reasons.append("NaN/Inf present")
        score += 100.0

    if std < flatline_std_limit:
        reasons.append("flatline / zero variance")
        score += 80.0

    if clip_fraction > clip_fraction_limit:
        reasons.append("clipping / saturation suspected")
        score += min(25.0, 100.0 * clip_fraction)

    if cfg.expected_dc_v is not None:
        offset_err = abs(mean - cfg.expected_dc_v)
        score += min(35.0, 20.0 * offset_err)
        if offset_err > 0:
            reasons.append(f"DC offset error {offset_err:.6g} V")

    # Relative channel-to-channel comparison
    if group_medians and group_mads:
        z_mean = abs(robust_z(mean, group_medians["mean"], group_mads["mean"]))
        z_rms = abs(robust_z(rms, group_medians["rms"], group_mads["rms"]))
        if z_mean > 3:
            reasons.append(f"mean outlier z={z_mean:.2f}")
            score += min(25.0, (z_mean - 3.0) * 8.0)
        if z_rms > 3:
            reasons.append(f"noise outlier z={z_rms:.2f}")
            score += min(25.0, (z_rms - 3.0) * 8.0)

    # Baseline comparison
    if baseline and name in baseline:
        b = baseline[name]
        if "dc_mean" in b:
            score += min(10.0, abs(mean - float(b["dc_mean"])) * 10.0)
        if "dc_rms" in b and float(b["dc_rms"]) > 0:
            score += min(10.0, abs(rms - float(b["dc_rms"])) / float(b["dc_rms"]) * 10.0)

    return ModeResult(
        channel=name,
        mean=mean,
        std=std,
        rms=rms,
        peak_to_peak=p2p,
        clip_fraction=clip_fraction,
        score=float(min(100.0, score)),
        reasons=reasons,
    )


def analyze_sine_channel(
    name: str,
    y: np.ndarray,
    cfg: ModeConfig,
    flatline_std_limit: float,
    clip_fraction_limit: float,
    sine_harmonics: int,
    baseline: Optional[Dict[str, Dict[str, float]]] = None,
    group_medians: Optional[Dict[str, float]] = None,
    group_mads: Optional[Dict[str, float]] = None,
) -> ModeResult:
    y = np.asarray(y, dtype=np.float64)

    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    rms = float(np.sqrt(np.mean(y ** 2)))
    p2p = float(np.max(y) - np.min(y))
    clip_fraction = compute_clip_fraction(y, cfg.full_scale_v)

    reasons: List[str] = []
    score = 0.0

    if not np.isfinite(y).all():
        reasons.append("NaN/Inf present")
        score += 100.0

    if std < flatline_std_limit:
        reasons.append("flatline / zero variance")
        score += 80.0

    if clip_fraction > clip_fraction_limit:
        reasons.append("clipping / saturation suspected")
        score += min(30.0, 120.0 * clip_fraction)

    amplitude = phase_deg = sinad_db = thd_db = sfdr_db = None

    if cfg.expected_sine_freq_hz:
        amplitude, phase_deg, offset, fitted = sine_fit(y, cfg.sample_rate_hz, cfg.expected_sine_freq_hz)
        sinad_db, thd_db, sfdr_db = fft_metrics(
            y,
            cfg.sample_rate_hz,
            cfg.expected_sine_freq_hz,
            harmonics=sine_harmonics,
        )

        residual = y - fitted
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))
        if residual_rms > 0.5 * rms:
            reasons.append("poor sine fit / non-sinusoidal response")
            score += 10.0

        if abs(offset) > 0:
            score += min(8.0, abs(offset) * 4.0)

        if cfg.expected_sine_amplitude_v is not None:
            amp_err = abs(amplitude - cfg.expected_sine_amplitude_v)
            score += min(35.0, 20.0 * amp_err)
            if amp_err > 0:
                reasons.append(f"sine amplitude error {amp_err:.6g} V")

        if sinad_db is not None:
            if sinad_db < 70:
                reasons.append(f"low SINAD {sinad_db:.1f} dB")
                score += min(25.0, (70.0 - sinad_db) * 0.5)

        if thd_db is not None:
            if thd_db > -60:
                reasons.append(f"high THD {thd_db:.1f} dB")
                score += min(20.0, (thd_db + 60.0) * 0.5)

        if sfdr_db is not None:
            if sfdr_db < 80:
                reasons.append(f"low SFDR {sfdr_db:.1f} dB")
                score += min(20.0, (80.0 - sfdr_db) * 0.3)

    # Relative channel-to-channel comparison
    if group_medians and group_mads:
        z_amp = 0.0
        if amplitude is not None:
            z_amp = abs(robust_z(amplitude, group_medians["amplitude"], group_mads["amplitude"]))
            if z_amp > 3:
                reasons.append(f"amplitude outlier z={z_amp:.2f}")
                score += min(25.0, (z_amp - 3.0) * 8.0)

        z_sinad = 0.0
        if sinad_db is not None:
            z_sinad = abs(robust_z(sinad_db, group_medians["sinad_db"], group_mads["sinad_db"]))
            if z_sinad > 3:
                reasons.append(f"SINAD outlier z={z_sinad:.2f}")
                score += min(20.0, (z_sinad - 3.0) * 6.0)

    # Baseline comparison
    if baseline and name in baseline:
        b = baseline[name]
        if "sine_amplitude" in b and amplitude is not None:
            score += min(10.0, abs(amplitude - float(b["sine_amplitude"])) * 10.0)
        if "sine_sinad_db" in b and sinad_db is not None:
            score += min(10.0, max(0.0, float(b["sine_sinad_db"]) - sinad_db))

    return ModeResult(
        channel=name,
        mean=mean,
        std=std,
        rms=rms,
        peak_to_peak=p2p,
        clip_fraction=clip_fraction,
        amplitude=amplitude,
        phase_deg=phase_deg,
        sinad_db=sinad_db,
        thd_db=thd_db,
        sfdr_db=sfdr_db,
        score=float(min(100.0, score)),
        reasons=reasons,
    )


# =========================
# Run-level analysis
# =========================

def group_stats(values: Dict[str, float]) -> Tuple[float, float]:
    arr = np.asarray(list(values.values()), dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def analyze_run(manifest_path: str | Path) -> List[ChannelSummary]:
    manifest = load_manifest(manifest_path)
    baseline = load_baseline(manifest.baseline_json)

    dc_channels, dc_data = load_matrix_csv(manifest.dc.path)
    sine_channels, sine_data = load_matrix_csv(manifest.sine.path)

    if dc_channels != sine_channels:
        raise ValueError("DC and sine files must have the same channel order and names")

    if manifest.channel_names is not None and manifest.channel_names != dc_channels:
        raise ValueError("Manifest channel_names do not match file headers")

    # Compute group stats separately for DC and sine
    dc_means = {dc_channels[i]: float(np.mean(dc_data[:, i])) for i in range(dc_data.shape[1])}
    dc_rms = {dc_channels[i]: float(np.sqrt(np.mean(dc_data[:, i] ** 2))) for i in range(dc_data.shape[1])}
    sine_amp = {}
    sine_sinad = {}

    for i, ch in enumerate(sine_channels):
        y = sine_data[:, i]
        amp, _, _, _ = sine_fit(y, manifest.sine.sample_rate_hz, manifest.sine.expected_sine_freq_hz or 0.0)
        sinad, _, _ = fft_metrics(
            y,
            manifest.sine.sample_rate_hz,
            manifest.sine.expected_sine_freq_hz,
            harmonics=manifest.sine_harmonics,
        )
        sine_amp[ch] = amp
        sine_sinad[ch] = sinad if sinad is not None else float("nan")

    dc_med_mean, dc_mad_mean = group_stats(dc_means)
    dc_med_rms, dc_mad_rms = group_stats(dc_rms)

    sine_amp_clean = {k: v for k, v in sine_amp.items() if np.isfinite(v)}
    sine_sinad_clean = {k: v for k, v in sine_sinad.items() if np.isfinite(v)}
    sine_med_amp, sine_mad_amp = group_stats(sine_amp_clean)
    sine_med_sinad, sine_mad_sinad = group_stats(sine_sinad_clean)

    results: List[ChannelSummary] = []

    for idx, ch in enumerate(dc_channels):
        dc_res = analyze_dc_channel(
            name=ch,
            y=dc_data[:, idx],
            cfg=manifest.dc,
            flatline_std_limit=manifest.flatline_std_limit,
            clip_fraction_limit=manifest.clip_fraction_limit,
            baseline=baseline,
            group_medians={"mean": dc_med_mean, "rms": dc_med_rms},
            group_mads={"mean": dc_mad_mean, "rms": dc_mad_rms},
        )

        sine_res = analyze_sine_channel(
            name=ch,
            y=sine_data[:, idx],
            cfg=manifest.sine,
            flatline_std_limit=manifest.flatline_std_limit,
            clip_fraction_limit=manifest.clip_fraction_limit,
            sine_harmonics=manifest.sine_harmonics,
            baseline=baseline,
            group_medians={"amplitude": sine_med_amp, "sinad_db": sine_med_sinad},
            group_mads={"amplitude": sine_mad_amp, "sinad_db": sine_mad_sinad},
        )

        combined_score = 0.4 * dc_res.score + 0.6 * sine_res.score

        reasons = []
        reasons.extend([f"DC: {r}" for r in dc_res.reasons])
        reasons.extend([f"SINE: {r}" for r in sine_res.reasons])

        verdict = "PASS"
        if dc_res.score >= 70 or sine_res.score >= 70:
            verdict = "FAIL"
        elif combined_score >= 50 or dc_res.score >= 25 or sine_res.score >= 25:
            verdict = "SUSPECT"

        results.append(
            ChannelSummary(
                channel=ch,
                dc=dc_res,
                sine=sine_res,
                combined_score=float(min(100.0, combined_score)),
                verdict=verdict,
                reasons=reasons,
            )
        )

    return sorted(results, key=lambda r: r.combined_score, reverse=True)


# =========================
# Reporting
# =========================

def save_report(results: List[ChannelSummary], out_prefix: str | Path) -> None:
    out_prefix = Path(out_prefix)

    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")

    json_path.write_text(
        json.dumps([{
            "channel": r.channel,
            "combined_score": r.combined_score,
            "verdict": r.verdict,
            "reasons": r.reasons,
            "dc": asdict(r.dc),
            "sine": asdict(r.sine),
        } for r in results], indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "channel",
            "verdict",
            "combined_score",
            "dc_score",
            "sine_score",
            "dc_mean",
            "dc_std",
            "dc_rms",
            "dc_clip_fraction",
            "sine_mean",
            "sine_std",
            "sine_rms",
            "sine_clip_fraction",
            "sine_amplitude",
            "sine_phase_deg",
            "sine_sinad_db",
            "sine_thd_db",
            "sine_sfdr_db",
            "reasons",
        ])

        for r in results:
            writer.writerow([
                r.channel,
                r.verdict,
                f"{r.combined_score:.2f}",
                f"{r.dc.score:.2f}",
                f"{r.sine.score:.2f}",
                f"{r.dc.mean:.6g}",
                f"{r.dc.std:.6g}",
                f"{r.dc.rms:.6g}",
                f"{r.dc.clip_fraction:.6g}",
                f"{r.sine.mean:.6g}",
                f"{r.sine.std:.6g}",
                f"{r.sine.rms:.6g}",
                f"{r.sine.clip_fraction:.6g}",
                "" if r.sine.amplitude is None else f"{r.sine.amplitude:.6g}",
                "" if r.sine.phase_deg is None else f"{r.sine.phase_deg:.3f}",
                "" if r.sine.sinad_db is None else f"{r.sine.sinad_db:.2f}",
                "" if r.sine.thd_db is None else f"{r.sine.thd_db:.2f}",
                "" if r.sine.sfdr_db is None else f"{r.sine.sfdr_db:.2f}",
                "; ".join(r.reasons),
            ])


# =========================
# CLI
# =========================

def main(manifest_path: str, out_prefix: str = "daq_validation") -> None:
    results = analyze_run(manifest_path)
    save_report(results, out_prefix)

    suspects = [r for r in results if r.verdict != "PASS"]
    print(f"Validated {len(results)} channels")
    print(f"Suspects: {len(suspects)}")
    print()

    for r in suspects[:30]:
        print(
            f"{r.channel:>20s}  {r.verdict:7s}  "
            f"combined={r.combined_score:5.1f}  "
            f"DC={r.dc.score:5.1f}  SINE={r.sine.score:5.1f}  "
            f"{'; '.join(r.reasons)}"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Two-mode DAQ validator: DC + sine.")
    p.add_argument("--manifest", required=True, help="Path to manifest JSON")
    p.add_argument("--out-prefix", default="daq_validation", help="Output file prefix")
    args = p.parse_args()

    main(args.manifest, args.out_prefix)