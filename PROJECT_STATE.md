# Emergency Vision AI — Project State & Verified Experiments

**Last Updated:** 2026-09-05  
**Current Milestone:** Second-Stage Production-Aligned Action Recognition (Person Crops)  
**Primary Checkpoint:** `models/action_recognition/r3d18_urfd_person_crops.pth`  
**Base Baseline Checkpoint:** `models/action_recognition/r3d18_urfd_best.pth` (Whole-Frame)  
**Authoritative Dataset & Weights Storage:** Google Drive (`/content/drive/MyDrive/emergency-vision-ai/`)  
**Synchronized Experiments Directory:** `experiments/`

---

## 1. Executive Summary

| Dimension | Baseline Model (`r3d18_urfd_best.pth`) | Person-Crop Model (`r3d18_urfd_person_crops.pth`) | Delta / Status |
| :--- | :--- | :--- | :--- |
| **Training Representation** | Whole-camera frames ($112 \times 112$) | YOLO11n + ByteTrack 5% padded person crops | Domain gap eliminated |
| **Held-Out Test Accuracy** | 100.0% (whole-frame) | **96.81%** (person-crop tubes) | Sequence-level isolated |
| **Held-Out Test Recall** | 100.0% (whole-frame) | **95.12%** (person-crop tubes) | Sensitivity preserved |
| **Production Video Accuracy** | 50.0% (5/10 videos) | **90.0%** (9/10 videos) | **+40.0% improvement** |
| **Production FALL Recall** | **0.0%** (0/5 falls detected) | **80.0%** (4/5 falls detected) | **+80.0% improvement** |
| **Production NORMAL FPR** | 0.0% (0/5 false alarms) | **0.0%** (0/5 false alarms) | **0% False Positive Rate maintained** |
| **NORMAL Specificity** | 100.0% | **100.0%** | Zero false alarms across ADL |
| **Confirmed Fall Events** | 0 events | **4 events** (`fall-01..04`) | Successful multi-frame confirmation |
| **Pipeline Throughput** | 62.52 FPS (Tesla T4) | **58.74 FPS** (Tesla T4) | Exceeds 30 FPS target ($\approx 2\times$) |

---

## 2. Verified Experiment: 2026-09-05 Person-Crop Training

* **Experiment Identifier:** `2026-09-05_r3d18_urfd_person_crops`
* **Local Artifacts:** [`experiments/2026-09-05_r3d18_urfd_person_crops/`](file:///Users/mukhammadiev/Desktop/emergency-vision-ai/experiments/2026-09-05_r3d18_urfd_person_crops/)
  - `train_person_crops_results.json`: Training history and validation/test metrics
  - `pipeline_comparison.json`: Side-by-side comparative multi-video evaluation
  - `benchmark_gpu_results.json`: Latency breakdown and FPS benchmark on Tesla T4
  - `r3d18_urfd_person_crops_metadata.json`: Architecture, hyperparameters, Git SHA, environment
  - `experiment_manifest.json`: Checksums, timestamps, and Google Drive source references
* **Model Checkpoint Name:** `r3d18_urfd_person_crops.pth`
* **Model Size:** 126.6 MB (132,751,435 bytes)
* **Authoritative Checkpoint Path (Google Drive):**
  `/content/drive/MyDrive/emergency-vision-ai/models/action_recognition/r3d18_urfd_person_crops.pth`
* **Training Hardware:** NVIDIA Tesla T4 (Google Colab CUDA)
* **Training Setup:**
  - ResNet3D-18 fine-tuning (`layer3`, `layer4`, `fc`)
  - Representation: 16-frame rolling tube of person bounding box crops with 5% spatial padding, resized to $112 \times 112$
  - Epochs: 12 | Batch Size: 8 | Learning Rate: $10^{-4}$ | Weight Decay: $10^{-4}$
  - Sequence-level deterministic split (Seed: 42): 49 train (21 fall, 28 ADL), 10 val (4 fall, 6 ADL), 11 test (5 fall, 6 ADL)

---

## 3. Held-Out Test Split Metrics (Sequence-Isolated)

Evaluated on 94 extracted 16-frame person tubes from 11 unseen test sequences (5 FALL, 6 NORMAL):

* **Test Loss:** 0.0918
* **Test Accuracy:** 96.81%
* **FALL Recall (Sensitivity):** 95.12%
* **FALL Precision:** 97.50%
* **FALL F1-Score:** 0.9630
* **NORMAL False Positive Rate:** 1.89%
* **Macro F1-Score:** 0.9675
* **Confusion Matrix:**
  - True Normal: 52 | False Positive: 1
  - False Negative: 2 | True Fall: 39

---

## 4. Production Multi-Video Pipeline Comparative Evaluation

Evaluated across 10 URFD sequences (5 FALL: `fall-01` to `fall-05`, 5 NORMAL: `adl-01` to `adl-05`) using the complete production worker pipeline:
$$\text{Input Frame} \rightarrow \text{YOLO11n} \rightarrow \text{ByteTrack} \rightarrow \text{Person Crop (5\% pad)} \rightarrow \text{16-frame Buffer} \rightarrow \text{R3D-18} \rightarrow \text{Temporal Confirmation (2 windows, } P \ge 0.70\text{)}$$

### Per-Video Comparison Matrix

| Video Filename | Ground Truth | Baseline Max $P$ (`r3d18_urfd_best`) | Person-Crop Max $P$ (`r3d18_urfd_person_crops`) | Events (Crop / Base) | Correct (Crop / Base) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `fall-01-cam0.mp4` | **FALL** | 0.2770 | **0.8421** | 1 / 0 | **YES** / NO |
| `fall-02-cam0.mp4` | **FALL** | 0.2788 | **0.7915** | 1 / 0 | **YES** / NO |
| `fall-03-cam0.mp4` | **FALL** | 0.3445 | **0.8834** | 1 / 0 | **YES** / NO |
| `fall-04-cam0.mp4` | **FALL** | 0.1321 | **0.7512** | 1 / 0 | **YES** / NO |
| `fall-05-cam0.mp4` | **FALL** | 0.3681 | **0.6380** | 0 / 0 | **NO** / NO *(Known FN)* |
| `adl-01-cam0.mp4` | **NORMAL** | 0.0149 | **0.0210** | 0 / 0 | **YES** / YES |
| `adl-02-cam0.mp4` | **NORMAL** | 0.0305 | **0.0385** | 0 / 0 | **YES** / YES |
| `adl-03-cam0.mp4` | **NORMAL** | 0.0195 | **0.0192** | 0 / 0 | **YES** / YES |
| `adl-04-cam0.mp4` | **NORMAL** | 0.0454 | **0.0411** | 0 / 0 | **YES** / YES |
| `adl-05-cam0.mp4` | **NORMAL** | 0.0304 | **0.0320** | 0 / 0 | **YES** / YES |

---

## 5. Known Issues & Diagnostic Findings

### A. Known `fall-05` False Negative
* **Symptom:** In `fall-05-cam0.mp4`, the model reached a peak fall probability of **0.6380**, failing to trigger the conservative production threshold of **0.70** required across 2 consecutive windows.
* **Root Cause Analysis:**
  1. The fall in `fall-05` occurs at a low camera angle with partial furniture occlusion as the subject touches the floor.
  2. Bounding box aspect ratio flattens rapidly ($W/H > 1.8$), causing detector confidence jitter.
  3. While $P(\text{FALL})$ is strongly elevated above normal baseline ($0.6380$ vs ADL max $\le 0.0411$), it falls just short of the $0.70$ confirmation threshold.
* **Resolution Plan:**
  - Do NOT lower the global production threshold from 0.70 (to avoid risking false positives in hospital/ADL deployments).
  - Subsequent iteration will incorporate aspect-ratio velocity priors and adaptive temporal confirmation for low-angle occluded tracks.

### B. Evaluator Comparison-Table Event-Count Inconsistency
* **Symptom:** When running `scripts/evaluate_production_pipeline.py` with `--compare-with`, the printed console table previously displayed `Events (A/B) = 0 / 0` and `Correct (A/B) = NO / NO` even when Model A triggered confirmed events and predicted correctly.
* **Root Cause:**
  - `evaluate_single_video()` returned `"confirmed_events_count"` and `"is_correct"`.
  - `compare_pipeline_models()` attempted to read `ra.get("confirmed_fall_events", 0)` and `ra.get("correct")`.
* **Fix Applied:**
  - Updated `scripts/evaluate_production_pipeline.py` to use backwards-compatible key resolution:
    ```python
    ea = ra.get("confirmed_events_count", ra.get("confirmed_fall_events", 0))
    ca = "YES" if (ra.get("is_correct") or ra.get("correct")) else "NO"
    ```
  - Both standalone evaluation and comparative matrix now report consistent event counts and correctness flags.

---

## 6. Production GPU Benchmark Metrics (Tesla T4)

* **Hardware Accelerator:** NVIDIA Tesla T4 (CUDA)
* **Video Input:** `fall-01-cam0.mp4` (160 frames, $640 \times 480$)
* **End-to-End Pipeline Throughput:** **58.74 FPS** (160 frames in 2.72s)
* **Latency Profile:**
  - **YOLO11n + ByteTrack:** Mean: 11.84 ms | P50: 11.20 ms | P95: 14.52 ms
  - **R3D-18 Action Inference (16-frame crops):** Mean: 21.45 ms | P50: 20.90 ms | P95: 24.80 ms
  - **End-to-End Frame Latency:** Mean: 16.20 ms | P50: 15.80 ms | P95: 21.30 ms
* **Confirmed Emergency Events:** 1 confirmed fall event on Track 1 (Confidence: 84.2%)

---

## 7. Operational Workflow for Experiments

To synchronize new or updated experiment artifacts from Google Drive into the local workspace:

```bash
# Automatic discovery (macOS Google Drive Desktop, env var, or Colab)
python3 scripts/sync_experiment_results.py

# Explicit Google Drive path
python3 scripts/sync_experiment_results.py --drive-dir /path/to/MyDrive/emergency-vision-ai
```
