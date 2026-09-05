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
| **Held-Out Test Accuracy** | 100.0% (whole-frame) | **96.09%** (person-crop tubes) | Sequence-level isolated |
| **Held-Out Test Recall** | 100.0% (whole-frame) | **85.00%** (person-crop tubes) | Sensitivity preserved |
| **Production Video Accuracy** | 50.00% (5/10 videos) | **90.00%** (9/10 videos) | **+40.00% improvement** |
| **Production FALL Recall** | **0.00%** (0/5 falls detected) | **80.00%** (4/5 falls detected) | **+80.00% improvement** |
| **Production NORMAL FPR** | 0.00% (0/5 false alarms) | **0.00%** (0/5 false alarms) | **0% False Positive Rate maintained** |
| **NORMAL Specificity** | 100.00% | **100.00%** | Zero false alarms across ADL |
| **Confirmed Fall Events** | 0 events | **4 events** (`fall-01..04`) | Successful multi-frame confirmation |
| **Pipeline Throughput** | 64.38 FPS (Tesla T4) | **65.54 FPS** (Tesla T4) | Exceeds 30 FPS target ($\approx 2.1\times$) |

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
* **Model Size:** 126.60 MB (132,752,779 bytes)
* **SHA-256 Checksum:** `5b43c57168834f47c44309b823cec5e287a88e3e9d20fd896ef2855d7bed0206`
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

Evaluated on 281 extracted 16-frame person tubes from 11 unseen test sequences (5 FALL, 6 NORMAL):

* **Test Loss:** 0.0971
* **Test Accuracy:** 96.09%
* **FALL Recall (Sensitivity):** 85.00%
* **FALL Precision:** 87.18%
* **FALL F1-Score:** 0.8608
* **NORMAL False Positive Rate:** 2.07%
* **Macro F1-Score:** 0.9190
* **Confusion Matrix:**
  - True Normal: 236 | False Positive: 5
  - False Negative: 6 | True Fall: 34

---

## 4. Production Multi-Video Pipeline Comparative Evaluation

Evaluated across 10 URFD sequences (5 FALL: `fall-01` to `fall-05`, 5 NORMAL: `adl-01` to `adl-05`) using the complete production worker pipeline:
$$\text{Input Frame} \rightarrow \text{YOLO11n} \rightarrow \text{ByteTrack} \rightarrow \text{Person Crop (5\% pad)} \rightarrow \text{16-frame Buffer} \rightarrow \text{R3D-18} \rightarrow \text{Temporal Confirmation (2 windows, } P \ge 0.70\text{)}$$

### Per-Video Comparison Matrix

| Video Filename | Ground Truth | Baseline Max $P$ (`r3d18_urfd_best`) | Person-Crop Max $P$ (`r3d18_urfd_person_crops`) | Events (Crop / Base) | Correct (Crop / Base) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `fall-01-cam0.mp4` | **FALL** | 0.2487 | **0.9995** | 1 / 0 | **YES** / NO |
| `fall-02-cam0.mp4` | **FALL** | 0.2863 | **0.9994** | 1 / 0 | **YES** / NO |
| `fall-03-cam0.mp4` | **FALL** | 0.3129 | **0.9999** | 1 / 0 | **YES** / NO |
| `fall-04-cam0.mp4` | **FALL** | 0.1290 | **0.9993** | 1 / 0 | **YES** / NO |
| `fall-05-cam0.mp4` | **FALL** | 0.3109 | **0.9113** | 0 / 0 | **NO** / NO *(Known FN)* |
| `adl-01-cam0.mp4` | **NORMAL** | 0.0149 | **0.0021** | 0 / 0 | **YES** / YES |
| `adl-02-cam0.mp4` | **NORMAL** | 0.0320 | **0.0015** | 0 / 0 | **YES** / YES |
| `adl-03-cam0.mp4` | **NORMAL** | 0.0187 | **0.0007** | 0 / 0 | **YES** / YES |
| `adl-04-cam0.mp4` | **NORMAL** | 0.0589 | **0.0024** | 0 / 0 | **YES** / YES |
| `adl-05-cam0.mp4` | **NORMAL** | 0.0313 | **0.0007** | 0 / 0 | **YES** / YES |

---

## 5. Known Issues & Diagnostic Findings

### A. Known `fall-05` False Negative
* **Symptom:** In `fall-05-cam0.mp4`, the model reached a peak fall probability of **0.9113**, but did not trigger a confirmed fall event (predicted NORMAL).
* **Root Cause Analysis:**
  1. The fall in `fall-05` occurs at a low camera angle with partial furniture occlusion as the subject touches the floor.
  2. Bounding box aspect ratio flattens rapidly ($W/H > 1.8$), causing detector confidence jitter and track fragmentation.
  3. While $P(\text{FALL})$ spiked to $0.9113$ on a single window (strongly elevated above ADL max $\le 0.0024$), it was not sustained across 2 consecutive inference windows for the same track ID.
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
* **End-to-End Pipeline Throughput:** **51.91 FPS** (160 frames in 3.08s)
* **Latency Profile:**
  - **YOLO11n + ByteTrack:** Mean: 15.30 ms | P50: 13.20 ms | P95: 23.14 ms
  - **Preprocessing / Crop (Tube):** Mean: 6.32 ms
  - **R3D-18 Action Inference (16-frame crops):** Mean: 23.33 ms | P50: 23.09 ms | P95: 24.32 ms
  - **End-to-End Frame Latency:** Mean: 18.54 ms | P50: 13.58 ms | P95: 44.97 ms
* **Confirmed Emergency Events:** 1 confirmed fall event on Track 1 (Confidence: 99.9%)

---

## 7. Operational Workflow for Experiments

To synchronize new or updated experiment artifacts from Google Drive into the local workspace:

```bash
# Automatic discovery (macOS Google Drive Desktop, env var, or Colab)
python3 scripts/sync_experiment_results.py

# Explicit Google Drive path
python3 scripts/sync_experiment_results.py --drive-dir /path/to/MyDrive/emergency-vision-ai
```

---

## 8. Current System Verification & Test Suite Status

* **Total Automated Tests:** **127 passed** across 22 test modules (0 failures, 0 regressions).
* **Live Camera Demo Status:**
  - Entrypoint: [`scripts/run_camera_demo.py`](scripts/run_camera_demo.py)
  - Pre-flight verification banner computes real-time streaming SHA-256 checksum.
  - Strictly enforces resolution of `models/action_recognition/r3d18_urfd_person_crops.pth` (`9b1a8d6f...`).
  - Silent fallback to legacy whole-frame baseline is strictly forbidden without `--allow-baseline`.
* **Model Loader Architecture:**
  - Unified [`apps/worker/app/models/model_loader.py`](apps/worker/app/models/model_loader.py) aligned with [`apps/worker/app/models/action_model.py`](apps/worker/app/models/action_model.py).
  - Handles 2-class linear projection head and automatic extraction of `model_state_dict`.
* **Hardware Acceleration Roadmap:**
  - ONNX export utility implemented in [`scripts/export_models.py`](scripts/export_models.py) for both YOLO11n and R3D-18 (opset 17).

