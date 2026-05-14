# DAQ Validator

A lightweight DAQ health validation framework for multi-channel engine instrumentation systems.

This project was developed for validating the **measurement chain below the sensor** before engine experiments are performed. The goal is to identify suspicious DAQ channels before testing so that any remaining issues during the experiment can be more confidently attributed to the **sensor** or the **engine itself**.

The system currently focuses only on validating the **DAQ acquisition path**.

---

# Features

- Multi-channel DAQ validation
- Designed for large systems (1000+ channels)
- Fast pre-run validation workflow
- DC and sine-based channel characterization
- Automatic fault scoring
- Detection of:
  - Flatlined channels
  - Clipping/saturation
  - Excessive offset
  - Amplitude mismatch
  - Distortion
  - Poor SINAD
  - Poor SFDR
  - Harmonic anomalies
- Simple operator GUI
- Synthetic data generator for testing

---

# Hardware Assumptions

The framework was designed around:

- NI PXI-4472
- Function generator
- LabVIEW-based acquisition workflow

The analysis layer itself is written in Python and is independent of the acquisition software.

---

# Files

| File | Description |
|---|---|
| `scoring_layer.py` | Main DAQ analysis and scoring engine |
| `simple_ui.py` | Operator GUI |
| `manifest.json` | Configuration file describing the acquisition |
| `generate_synthetic_daq_bundle.py` | Synthetic data generator for testing |
| `DAQ_Validator_Operator_Manual.pdf` | Full operator/user manual |

---

# Validation Philosophy

The system uses two separate acquisitions:

## 1. DC Test

A shorted-input or DC acquisition is used to detect:

- Flatlined channels
- Offset errors
- Noise floor anomalies
- Saturation
- Dead channels

## 2. Sine Test

A coherent sine wave acquisition is used to detect:

- Gain mismatch
- Distortion
- Harmonics
- Clipping
- Spectral anomalies
- Dynamic performance degradation

---

# Acquisition Settings

## Sampling Rate

```text
102.4 kHz
```

## Samples Per Channel

```text
131072
```

This allows coherent FFT-based spectral analysis.

## Sine Stimulus

```text
1 kHz sine wave
-1 dBFS
```

---

# Expected Directory Structure

```text
project/
│
├── scoring_layer.py
├── simple_ui.py
├── manifest.json
├── dc_run.csv
├── sine_run.csv
├── generate_synthetic_daq_bundle.py
└── DAQ_Validator_Operator_Manual.pdf
```

---

# CSV Format

Both `dc_run.csv` and `sine_run.csv` must follow:

```text
First row  -> channel names
Remaining rows -> samples
```

Example:

```csv
ch1,ch2,ch3
0.001,0.002,0.001
0.001,0.002,0.001
...
```

---

# Running the Analyzer

## Run the scoring engine directly

```bash
python scoring_layer.py --manifest manifest.json
```

or

```bash
python3 scoring_layer.py --manifest manifest.json
```

## Specify output file prefix

```bash
python scoring_layer.py --manifest manifest.json --out-prefix run1
```

This generates:

```text
run1.csv
run1.json
```

---

# Running the UI

```bash
python simple_ui.py
```

The UI:

- launches the analyzer
- loads the generated CSV
- displays:
  - verdict
  - scores
  - SINAD
  - THD
  - SFDR
  - reason strings

## Verdict Colors

| Verdict | Meaning |
|---|---|
| Green | PASS |
| Yellow | SUSPECT |
| Red | FAIL |

---

# Generating Synthetic Test Data

The repository includes a synthetic DAQ data generator.

Run:

```bash
python generate_synthetic_daq_bundle.py
```

This generates:

- `dc_run.csv`
- `sine_run.csv`
- `manifest.json`

with intentionally injected faults.

---

# Injected Synthetic Faults

| Channel | Fault |
|---|---|
| ch1 | Healthy |
| ch2 | Healthy with small drift |
| ch3 | Offset + low amplitude + distortion |
| ch4 | Flatline + clipping |

---

# Metrics Used

## DC Metrics

- Mean
- RMS
- Standard deviation
- Peak-to-peak
- Clip fraction

## Sine Metrics

- Amplitude
- Phase
- SINAD
- THD
- SFDR

---

# Scoring Logic

Each channel receives:

- DC score
- Sine score
- Combined score

The combined score is:

```text
0.4 × DC score + 0.6 × Sine score
```

The spectral test is weighted more heavily because it captures dynamic DAQ behavior.

## Verdict Thresholds

| Combined Score | Verdict |
|---|---|
| Low | PASS |
| Medium | SUSPECT |
| High | FAIL |

---

# Analysis Flow

```text
Acquire DC run
        ↓
Acquire sine run
        ↓
Run scoring engine
        ↓
Compute metrics
        ↓
Compare against:
    - expected values
    - group statistics
    - baseline behavior
        ↓
Generate scores
        ↓
Display suspect channels
```

---

# Dependencies

Install required Python packages:

```bash
pip install numpy
```

Tkinter is included with most Python installations.

---

# Future Improvements

Possible future additions:

- TDMS support
- Real-time acquisition hooks
- Baseline database
- Trend analysis
- Missing-code testing
- Histogram testing
- Crosstalk analysis
- Automated report generation
- Channel grouping
- Frequency sweep characterization
- Impulse response estimation

---

# References

1. NI PXI-4472 Specifications  
   https://www.ni.com/docs/en-US/bundle/ni-447x-specs/page/specs.html

2. Analog Devices – Understanding SINAD, ENOB, SFDR and THD  
   https://www.analog.com/

3. FFT Fundamentals and Spectral Analysis  
   Oppenheim & Schafer, *Discrete-Time Signal Processing*

4. IEEE Standard for Digitizing Waveform Recorders  
   IEEE Std 1057

5. ADC Dynamic Testing Tutorial  
   Walt Kester, Analog Devices

---

# Disclaimer

This project is intended as a DAQ screening and diagnostic aid.

A PASS verdict does not guarantee absolute channel correctness under all operating conditions. The framework is intended to identify suspicious channels and reduce diagnostic uncertainty before experiments.
