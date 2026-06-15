<div align="center">

# heart_beats

### Remote Photoplethysmography (rPPG) Pipeline

**Comparing Algorithms and ROI Selection Methods in RGB Video**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green)](#license)
[![University](https://img.shields.io/badge/University%20of%20Nottingham-School%20of%20Psychology-informational)](https://www.nottingham.ac.uk/psychology/)

*Masters dissertation project by **Samuel Hardy***

</div>

---

## Overview

Remote photoplethysmography (rPPG) is a **contactless method for measuring heart rate** from facial video by detecting subtle changes in skin colour caused by pulsatile blood flow — no sensors, no contact required.

This repository contains the full Python pipeline developed for my masters dissertation, which systematically compared:

| Component | Options Tested |
|---|---|
| **Signal extraction** | R, G, B, Greyscale (weighted & average), ICA, PCA, ZCA, CHROM, POS |
| **ROI strategies** | Full frame, manual crop, BlazeFace bbox, BlazeFace forehead, MediaPipe polygons |
| **HR estimation** | Sliding-window FFT, Welch PSD, Hilbert transform (beat-to-beat) |

Evaluation was performed against **ECG ground truth** across three datasets spanning controlled and naturalistic recording conditions.

---

## Key Findings

> **Best configuration: Hilbert transform + PCA + Polygonal ROI (forehead & cheeks)**

| Configuration | MAE (bpm) | RMSE (bpm) | Pearson *r* | Within ±10 bpm |
|---|:---:|:---:|:---:|:---:|
| **Hilbert + PCA + Polygonal** | **8.04** | **8.97** | **.705** | **73.4%** |
| Hilbert + ICA + Polygonal | 8.48 | — | — | — |
| FFT + ICA/PCA + Polygonal | ~15.1 | — | — | — |
| Welch + PCA/ICA + Polygonal | ~15.3 | — | — | — |

**Key takeaways:**
- The **Hilbert transform** consistently outperformed FFT and Welch across all datasets by tracking beat-to-beat timing rather than locking onto a single dominant frequency
- **Polygonal landmark ROIs** (forehead + cheeks via MediaPipe) combined with **PCA** gave the best overall accuracy
- **Bland–Altman** analysis showed minimal bias (−0.38 bpm), with best agreement in the 75–85 bpm range (~90% within ±10 bpm)
- Accuracy degraded above ~95 bpm

---

## Datasets

> **None of the datasets are included in this repository** due to size and licensing constraints.

<details>
<summary><b>Dataset 1 — UBFC-Phys (subset)</b></summary>

- **Source:** [UBFC-Phys](https://sites.google.com/view/ubfc-phys) (Sabour et al., 2023)
- 9 videos from 3 participants (subset of 56 total)
- EO 23121C RGB camera · 1024×1024 · 35 fps · Motion JPEG
- Ground truth: BVP via Empatica E4 wristband at 64 Hz
- Controlled indoor lighting, ~1 m distance

</details>

<details>
<summary><b>Dataset 2 — StressID</b></summary>

- **Source:** [StressID](https://project.inria.fr/stressid/) (Chaptoukaev et al., 2023)
- 629 recordings from 53 participants
- Logitech QuickCam Pro 9000 · 720p · **15 fps**
- Ground truth: ECG via BioSignalsPlux at 500 Hz
- Challenging conditions: low frame rate, variable lighting, not originally designed for rPPG

</details>

<details>
<summary><b>Dataset 3 — YouTube (Markiplier / Resident Evil: Village)</b></summary>

- 10 non-overlapping 60-second webcam segments (Fischbach, 2021)
- Manually labelled: `baseline` · `scared` · `environmental_fear`
- Used to evaluate rPPG sensitivity to acute fear/stress responses

</details>

---

## Pipeline

### ROI Strategies

| Name | Description |
|---|---|
| `none` | Full frame baseline |
| `manual` | Pre-defined ROI loaded from CSV |
| `face_track` | BlazeFace bounding box, cropped inward (10% width, 20% height) |
| `face_track_forehead` | BlazeFace box, 14%-height strip below hairline |
| `poly_forehead` | Polygonal forehead via 8 MediaPipe landmarks |
| `poly_full` | Polygonal forehead + left & right cheeks via 27 landmarks |

> **ROI stabilisation:** A Kalman smoother reduces frame-to-frame jitter on bounding box trajectories. Failed detections use the filter's prediction step; unrecoverable frames are NaN-interpolated.
>
> **Skin masking:** YCrCb thresholds (133 ≤ Cr ≤ 173, 77 ≤ Cb ≤ 127) isolate skin pixels, cleaned with morphological opening/closing using a 5×5 elliptical structuring element.

---

### Signal Extraction Methods

| Method | Description |
|---|---|
| `R`, `G`, `B` | Raw single-channel mean intensity over skin pixels |
| `grey_w` | Luminance-weighted greyscale: 0.299R + 0.587G + 0.114B |
| `grey_a` | Simple average greyscale |
| `PCA` | Principal Component Analysis on RGB; best of 3 components by MAE |
| `ZCA` | Zero-phase Component Analysis whitening; best of 3 components |
| `ICA` | Independent Component Analysis; best of 3 components |
| `CHROM` | Chrominance-based method (de Haan & Jeanne, 2013) |
| `POS` | Plane Orthogonal to Skin (Wang et al., 2017) |

All signals are resampled to **128 Hz** via PCHIP interpolation and band-pass filtered (**0.75–3.00 Hz** / 45–180 bpm).

---

### Heart Rate Estimation

<details>
<summary><b>FFT (Sliding Window)</b></summary>

- 20 s windows, 3 s step
- Peak frequency in 0.75–3.00 Hz band converted to bpm

</details>

<details>
<summary><b>Welch PSD</b></summary>

- 20 s windows, 3 s step
- Hann taper · 75% segment length · 50% overlap · median averaging

</details>

<details>
<summary><b>Hilbert Transform (Beat-to-Beat) Best</b></summary>

- Analytic signal phase unwrapped; beats detected at 2π increments
- PP intervals median-filtered (51-beat window, 0.15 s threshold) for artefact rejection
- Instantaneous HR smoothed with a 5-sample moving average

</details>

---

## Dependencies

```
python >= 3.9
opencv-python
numpy
scipy
scikit-learn
mediapipe
neurokit2
pandas
matplotlib
```

**MediaPipe model files** (download separately and place in a `models/` folder):
- [`blaze_face_short_range.tflite`](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite)
- [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/fatalwhiskers/heart_beats.git
cd heart_beats

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe models
mkdir models
# Place blaze_face_short_range.tflite and face_landmarker.task in models/
```

---

## Usage

> **Note:** Update the script names and flags below to match your actual entry points.

**Run the full pipeline on a single video:**
```bash
python src/main.py \
  --video path/to/video.mp4 \
  --ground_truth path/to/gt.csv \
  --roi poly_full \
  --method PCA \
  --estimator hilbert \
  --output results/
```

**Run all ROI × method combinations:**
```bash
python src/run_all.py \
  --dataset stressid \
  --data_dir data/stressid/ \
  --output results/stressid_full.csv
```

**Reproduce the best configuration (Hilbert + PCA + Polygonal):**
```bash
python src/main.py \
  --video path/to/video.mp4 \
  --roi poly_full \
  --method PCA \
  --estimator hilbert
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error (bpm) — primary selection criterion |
| **RMSE** | Root Mean Squared Error (bpm) |
| **Pearson *r*** | Linear correlation with ground truth HR |
| **Bland–Altman** | Bias (mean difference) and 95% limits of agreement |
| **Failure rate** | Proportion of recordings with MAE > 10 bpm |

For methods with multiple components (PCA 1–3, ICA 1–3, ZCA 1–3), only the component with the lowest MAE is retained.

---

## Results Summary

**Dataset 2 · 629 recordings · ECG ground truth**

```
Best config:  Hilbert + PCA + Polygonal (forehead & cheeks)

MAE          =  8.04 bpm  (Mdn = 6.18 bpm)
RMSE         =  8.97 bpm
Pearson r    =  .705  [95% CI: .663, .742]
B-A bias     = −0.38 bpm  |  95% LoA: [−19.26, 18.49] bpm
Within ±10   =  73.4%
Failure rate =  26.6%  (vs ≥60% for all other configs)
```

**Dataset 3 · Fear/stress responses (YouTube)**

```
Baseline  →  M = 81.1 bpm,  SD = 5.9   (stable)
Fear      →  M = 89.1 bpm,  SD = 10.1  (+7.96 bpm vs baseline)
```

---

## Limitations

- Dataset 1 was only a 9-video subset (~40 GB even at this scale)
- Dataset 2 was recorded at **15 fps**, limiting temporal resolution
- StressID is not a standard rPPG benchmark, limiting direct comparison with published work
- Accuracy degrades above ~95 bpm
- Manual ROIs are impractical in real-world settings; CNN-based tracking adds compute cost
- Short recordings (~60 s) limit frequency resolution at low frame rates
- rPPG at 15 fps is not suitable for precise clinical monitoring

---

## Citation

If you use this code or pipeline, please cite the associated dissertation:

```bibtex
@thesis{hardy2025rppg,
  author  = {Hardy, Samuel},
  title   = {Comparing rPPG Accuracy: Evaluating Algorithms and ROI
             Selection Methods in RGB Video},
  school  = {School of Psychology, University of Nottingham},
  year    = {2025},
  type    = {Masters Dissertation}
}
```

<details>
<summary><b>Key references</b></summary>

- Poh et al. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express.*
- de Haan & Jeanne (2013). Robust pulse rate from chrominance-based rPPG. *IEEE TBME.*
- Wang et al. (2017). Algorithmic principles of remote PPG. *IEEE TBME.*
- Sabour et al. (2023). UBFC-Phys. *IEEE Transactions on Affective Computing.*
- Chaptoukaev et al. (2023). StressID. *NeurIPS Datasets.*
- Makowski et al. (2021). NeuroKit2. *Behavior Research Methods.*

</details>

---

## Acknowledgements

- University of Nottingham School of Psychology for access to the UBFC-Phys subset
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit) for ground truth signal processing
- [Google MediaPipe](https://ai.google.dev/edge/mediapipe) for BlazeFace and Face Landmarker models
- StressID dataset authors (Chaptoukaev et al., 2023)

---

## License

This project is released for academic and research purposes. Please contact the author before using this code in commercial applications.

---

<div align="center">

*Samuel Hardy · University of Nottingham · 2025*

</div>
