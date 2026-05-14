# generate_synthetic_data_to_test.py

from pathlib import Path
import json
import numpy as np
import csv

root = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

# DAQ settings
fs = 102400.0
n = 131072
t = np.arange(n) / fs

channels = ["ch1", "ch2", "ch3", "ch4"]
f0 = 1000.0

# Assume full-scale peak is 10 V for the synthetic test.
# -1 dBFS => 10 * 10^(-1/20)
full_scale_v = 10.0
amp = full_scale_v * (10 ** (-1.0 / 20.0))

# -----------------------------
# Synthetic DC run
# -----------------------------
dc = np.zeros((n, len(channels)), dtype=np.float64)

dc[:, 0] = 0.0015 + 0.00035 * rng.standard_normal(n)   # healthy-ish
dc[:, 1] = 0.0018 + 0.00040 * rng.standard_normal(n)   # healthy-ish
dc[:, 1] += 0.00015 * np.sin(2 * np.pi * 0.7 * t)      # slight drift
dc[:, 2] = 0.1200 + 0.00045 * rng.standard_normal(n)   # offset fault
dc[:, 3] = np.full(n, 0.0)                             # flatline fault

# -----------------------------
# Synthetic sine run
# -----------------------------
sine = np.zeros((n, len(channels)), dtype=np.float64)

base = amp * np.sin(2 * np.pi * f0 * t)

sine[:, 0] = base + 0.010 * rng.standard_normal(n)  # healthy-ish
sine[:, 1] = (
    1.01 * amp * np.sin(2 * np.pi * f0 * t + np.deg2rad(1.0))
    + 0.010 * rng.standard_normal(n)
)  # healthy-ish

sine[:, 2] = 0.65 * amp * np.sin(2 * np.pi * f0 * t) + 0.030 * rng.standard_normal(n)
sine[:, 2] += 0.08 * np.sin(2 * np.pi * 3 * f0 * t)  # extra harmonic distortion

sine[:, 3] = np.clip(
    1.15 * amp * np.sin(2 * np.pi * f0 * t) + 0.005 * rng.standard_normal(n),
    -8.0,
    8.0,
)  # clipping fault

# -----------------------------
# Helper
# -----------------------------
def write_csv(path: Path, header, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data.tolist())


# Write data files
write_csv(root / "dc_run.csv", channels, dc)
write_csv(root / "sine_run.csv", channels, sine)

# Manifest for the analyzer
manifest = {
    "dc": {
        "path": "dc_run.csv",
        "stimulus_type": "dc",
        "sample_rate_hz": fs,
        "expected_dc_v": 0.0,
        "full_scale_v": full_scale_v,
    },
    "sine": {
        "path": "sine_run.csv",
        "stimulus_type": "sine",
        "sample_rate_hz": fs,
        "expected_sine_freq_hz": f0,
        "expected_sine_amplitude_v": amp,
        "full_scale_v": full_scale_v,
    },
    "clip_fraction_limit": 0.01,
    "flatline_std_limit": 1e-9,
    "sine_harmonics": 5,
    "channel_names": channels,
}

(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

notes = {
    "intentional_faults": {
        "ch3": [
            "DC offset fault",
            "lower sine amplitude",
            "extra harmonic distortion",
        ],
        "ch4": [
            "flatline in DC",
            "clipping in sine",
        ],
    },
    "healthy_reference_like": [
        "ch1",
        "ch2",
    ],
}

(root / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

print(f"Created synthetic test bundle in: {root}")
for p in sorted(root.iterdir()):
    print(" -", p.name)