# Emergency Action Recognition ML Roadmap & Dataset Architecture

## 1. Executive Summary

This roadmap establishes the machine learning strategy for **Emergency Action Recognition** in **Emergency Vision AI**. Following the verified production rollout of YOLOv11 person detection, ByteTrack tracking, line-crossing counters, Redis Streams transport, and WebSocket broadcasting, this document defines the actionable computer vision models, dataset acquisition pipeline, transfer learning protocols, and evaluation metrics required to train a production-grade emergency action classifier.

---

## 2. Current ML State & Baseline Analysis

### 2.1 Research Notebook State (`notebooks/01_video_opencv_basics.ipynb`)
- **Model Architecture**: ResNet3D-18 (`torchvision.models.video.r3d_18`) with Kinetics-400 pre-trained backbone weights.
- **Classifier Head**: Linear layer modified to output 10 logits: `model.fc = nn.Linear(512, 10)` for a 10-class UCF101 subset (`aisuko/ucf101-subset`).
- **Dataset Configuration**: 10 non-emergency human action classes (`ApplyEyeMakeup`, `ApplyLipstick`, `Archery`, `BabyCrawling`, `BalanceBeam`, `BandMarching`, `BaseballPitch`, `Basketball`, `BasketballDunk`, `BenchPress`).
- **Dataset Scale**: 300 train, 30 validation, 75 test videos (total 405 clips).
- **Training Strategy**: Linear probing (frozen backbone, training only `model.fc` with Adam $lr=10^{-3}$, CrossEntropyLoss).
- **Observed Performance**: Restored checkpoint evaluation achieved **14.7% test accuracy** (barely above the 10% random guess baseline), exhibiting high bias and mode collapse.

### 2.2 Critical Verdict on Current Checkpoint
- **Educational Baseline Only**: The UCF101 checkpoint serves solely as an instructional proof-of-concept for PyTorch 3D video decoding, uniform frame tensor slicing (`torch.linspace(0, total_frames - 1, 16)`), and confusion matrix generation.
- **Zero Domain Transfer Value**: The UCF101 classes do not represent emergency kinematics (falls, brawls, panic stampedes).
- **Recommended Backbone Initialization**: All production transfer learning **MUST** initialize directly from official **Torchvision Kinetics-400 pre-trained R3D-18 weights** (`torchvision.models.video.r3d_18(weights=R3D_18_Weights.KINETICS400_V1)`), which provide generalizable spatiotemporal motion features trained across 240,000+ video clips.

---

## 3. Accessible Candidate Datasets: Comprehensive Verification Matrix

To ensure development feasibility, every candidate dataset was researched and verified against official primary sources, live download links, licensing restrictions, and operational hurdles.

| Dataset Name | Official Source / Host | Action Classes | Size & Scale | Format & Modality | Licensing & Terms | Download Availability | Suitability for Emergency Vision AI |
|---|---|---|---|---|---|---|---|
| **UR Fall Detection (URFD)** | University of Rzeszow & AGH ([Official Portal](http://fenix.ur.edu.pl/~mkepski/ds/ufd.html)) | • Fall (30 sequences)<br>• Normal ADL (40 sequences) | 70 sequences<br>(~3.5 GB) | Frontal + Overhead RGB PNG sequences (640x480) + 16-bit Depth PNGs | CC BY-NC-SA 4.0 (Non-commercial academic research) | **Verified Live Direct HTTP Download** from official university server without login | **High (Optimal for Milestone 1)**: Zero download friction, clean ground truth, authentic falls vs. daily living |
| **Real Life Violence Situations (RLVS)** | Soliman et al., ICICIS 2019 ([Kaggle Host](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)) | • Violence / Fight (1,000 clips)<br>• Non-Violence (1,000 clips) | 2,000 clips<br>(~2.5 GB) | Short MP4 video clips (~3–6 seconds) | Data files © Original Authors (Research usage with citation) | **Verified Public Access** via Kaggle API / Direct download | **High (Optimal for Violence Milestone)**: Large sample count, real street/surveillance footage, balanced |
| **Hockey Fight Dataset** | University of Córdoba / Nievas et al., 2011 ([Academic Torrents / Mirrors](https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89)) | • Fight (500 clips)<br>• Non-Fight (500 clips) | 1,000 clips<br>(~250 MB) | Short AVI video clips (720x576 @ 25 fps) | Academic research usage with citation | **Verified Public Mirrors** on Kaggle & Academic Torrents | **Medium-High**: Fast, lightweight sanity-check dataset, though domain-specific (hockey arena) |
| **UMN Crowd Anomaly Dataset** | University of Minnesota ([Official MHA Project](http://mha.cs.umn.edu/proj_events.shtml#crowd)) | • Normal crowd motion<br>• Panic / Rapid evacuation | 11 sequences (7,740 frames, ~150 MB) | 3 scenes (1 indoor foyer, 2 outdoor lawn) in AVI / RGB | Academic research usage with citation | **Verified Live Direct Download** from official UMN CS server | **High (Optimal for Panic / Crowd Anomaly Milestone)**: Standard benchmark for sudden crowd evacuation |
| **RWF-2000** | SMIIP Lab / Cheng et al., ICPR 2020 ([Official GitHub](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)) | • Fight (1,000 clips)<br>• Non-Fight (1,000 clips) | 2,000 clips<br>(~10.7 GB) | 5-second AVI clips @ 30 fps (or .npy optical flow) | Strict non-commercial; redistribution prohibited without explicit approval | **Gated Access**: Official repo requires signing an agreement sheet & emailing authors | **Medium (Deferred)**: Excellent quality, but manual approval requirement prevents automated zero-friction setup |
| **Le2i Fall Detection Dataset** | Le2i Lab / Univ. of Burgundy (Charfi et al., 2013) | • Fall (131 videos)<br>• Normal ADL (60 videos) | 191 videos<br>(~3.8 GB) | AVI (320x240 @ 25 fps) across Home, Coffee Shop, Office, Lecture Room | Academic research (original license unspecified) | **Official Server Offline**: `le2i.cnrs.fr` link is dead; accessible only via community mirrors on Kaggle | **Medium (Secondary)**: Good multi-room coverage, but lack of active official host introduces mirror dependency |

---

## 4. Target Emergency-Action Taxonomy

For full operational coverage across safety, medical, and security domains, Emergency Vision AI establishes a structured 4-class taxonomy:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       EMERGENCY VISION AI ACTION TAXONOMY                       │
├───────────────────┬───────────────────┬───────────────────┬─────────────────────┤
│ 0. Normal         │ 1. Fall           │ 2. Fighting       │ 3. Panic / Running  │
│ (Baseline ADL)    │ (Medical/Safety)  │ (Security Hazard) │ (Evacuation Anomaly)│
├───────────────────┼───────────────────┼───────────────────┼─────────────────────┤
│ • Walking         │ • Slip and fall   │ • Physical brawl  │ • Rapid fleeing     │
│ • Standing still  │ • Sudden collapse │ • Punching/kicking│ • Panic stampede    │
│ • Sitting/bending │ • Loss of balance │ • Violent assault │ • Sudden rush to exit│
│ • Routine motion  │ • Worker on floor │ • Body grappling  │ • Chaotic dispersal │
└───────────────────┴───────────────────┴───────────────────┴─────────────────────┘
```

- **Class 0 (`Normal`)**: Baseline activity of daily living (ADL). **Critical requirement**: At least 40–50% of the training dataset must consist of normal walking, sitting, and standing to prevent high false alarm rates.
- **Class 1 (`Fall`)**: Rapid vertical posture collapse followed by floor inactivity.
- **Class 2 (`Fighting`)**: High-acceleration reciprocal human contact, striking, grappling, or aggressive shoving.
- **Class 3 (`Panic / Running`)**: Sudden abnormal acceleration of multiple individuals away from an epicenter or toward an exit.

---

## 5. Practical Phased Dataset & Training Strategy

Rather than attempting an unverified all-in-one crawl, we adopt a **progressive, phased training strategy** using verified, accessible datasets.

### Phase 1: Real-World Fall vs. Normal ADL Classifier (Immediate Milestone)
- **Primary Dataset**: **UR Fall Detection (URFD)** (70 sequences: 30 falls, 40 normal ADL).
- **Why**: 100% verified, direct public HTTP download from university server (`fenix.ur.edu.pl`), zero API credentials required, clean camera angles, standard RGB image sequences.
- **Classes**: 2-class binary (`Normal` vs. `Fall`).

### Phase 2: Violence & Fighting Classifier Integration
- **Primary Dataset**: **Real Life Violence Situations (RLVS)** (2,000 clips) + **Hockey Fight Dataset** (1,000 clips).
- **Classes**: Binary (`Non-Violence` vs. `Violence`).

### Phase 3: Multi-Class Unified Emergency Action Model (4 Classes)
- Combine URFD (Fall + Normal ADL) + RLVS (Fight + Normal) + UMN (Panic Running + Normal) into the unified 4-class `CMES-4` model.

---

## 6. Proposed Model Training Protocol

```
┌────────────────────────────────────────────────────────┐
│ Backbone: R3D-18 (torchvision Kinetics-400 Pretrained) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Classifier Head Warm-Up                       │
│ • Freeze Conv3D layers (layer1, layer2, layer3, layer4)│
│ • Initialize new head: nn.Linear(512, num_classes)     │
│ • Optimizer: AdamW (lr = 1e-3, weight_decay = 1e-4)    │
│ • Epochs: 5–8 epochs                                   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Differential Deep Layer Fine-Tuning           │
│ • Unfreeze layer3 & layer4                             │
│ • Learning Rates:                                      │
│   - Backbone (layer3 & layer4): lr = 1e-5              │
│   - Classifier Head (model.fc): lr = 1e-4              │
│ • Loss: CrossEntropyLoss (with inverse frequency weights)│
│ • Scheduler: CosineAnnealingLR (T_max = 25, eta_min=1e-6)│
│ • Epochs: 20–25 epochs                                 │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Quantization & Deployment Checkpoint          │
│ • Save best validation checkpoint                      │
│ • Optional FP16 / TorchScript / ONNX export            │
│ • Package weights in models/action_recognition/        │
└────────────────────────────────────────────────────────┘
```

### 6.1 Data Ingestion & Augmentation Pipeline
- **Temporal Volume**: $T = 16$ frames sampled across video sequences with stride $s=2$ (covering ~1.0–1.2s of action dynamics).
- **Spatial Tensor**: Resized to $112 \times 112$ pixels (or $224 \times 224$ for higher spatial fidelity).
- **Augmentations**:
  - Random spatial affine/crop ($[0.85, 1.0]$).
  - Random horizontal flip (for symmetric fall and fight movements).
  - Color jitter (brightness $\pm 0.1$, contrast $\pm 0.1$).
  - Random temporal start offset ($\pm 3$ frames).

### 6.2 Data Splitting & Leakage Prevention
- **Split Ratio**: **70% Train / 15% Validation / 15% Test**.
- **Cross-Subject / Scene Isolation**: Sequences originating from the same video capture session or subject ID are strictly confined to either train or test (never split across both).

---

## 7. Evaluation Plan & Operational Thresholds

To prevent false alarms in live surveillance:

1. **Confusion Matrix Analysis**:
   - Comprehensive test confusion matrix inspecting false positives on the `Normal` class.
2. **Per-Class Metrics**:
   - **Fall Recall**: Target $\ge 90\%$ (minimizing missed falls).
   - **Fall Precision**: Target $\ge 85\%$ (suppressing false fall alarms during sitting/bending).
   - **Macro F1-Score** and **PR-AUC**.
3. **Inference Latency Target**:
   - Per-clip inference latency $< 40$ ms on GPU and $< 120$ ms on modern multi-core CPU.
4. **Temporal Moving-Average Smoothing**:
   - Live stream predictions are smoothed over a rolling 3-window sliding buffer: an alert is only triggered if $P(\text{emergency}) > 0.75$ across at least 2 consecutive windows.

---

## 8. Integration Plan with CV Worker Architecture

```
[Video Stream (RTSP / Video File)]
           │
           ▼
[apps/worker/pipeline/capture.py]
           │
           ▼ (Frame Stream)
[apps/worker/pipeline/action_recognition.py]
  ├── Rolling deque buffer (16 frames)
  ├── Preprocess & Normalize -> (1, 3, 16, 112, 112)
  └── R3D-18 Inference (every 8 frames)
           │
           ▼
[Emergency Action Logic: If Action == "Fall" & Conf >= 0.75]
           │
           ▼
[apps/worker/events/publisher.py (RedisStreamEventPublisher)]
  ├── XADD emergency_vision:events
  └── Event: {stream_id, event_type: "fall_detected", confidence: 0.88, ...}
           │
           ▼
[apps/api/services/redis_consumer.py (XREADGROUP + XACK)]
           │
           ▼
[apps/api/services/event_service.py (Single Source of Truth)]
           │
           ▼
[apps/api/services/websocket_manager.py]
           │
           ▼ (Real-time Broadcast)
[Connected WebSocket Clients / Frontend Monitors (/api/v1/ws/events)]
```

---

## 9. Concrete Recommendation for the First Training Run

> ### **Primary Recommendation: UR Fall Detection Dataset (URFD)**
> 
> **Why URFD for our very first real training run:**
> 1. **Verified Direct Accessibility**: Available directly via HTTP from the official University of Rzeszow server ([`http://fenix.ur.edu.pl/~mkepski/ds/ufd.html`](http://fenix.ur.edu.pl/~mkepski/ds/ufd.html)) without credentials, Kaggle API tokens, or gated email approvals.
> 2. **Clear Ground Truth & Clean Modalities**: 70 well-annotated sequences (30 falls, 40 normal ADL) with synchronized frontal and overhead RGB image streams.
> 3. **Manageable Footprint**: ~3.5 GB total, allowing rapid automated downloading, deterministic train/val/test splitting, and fast iteration loops.
> 4. **Clear Operational Impact**: Solves the critical medical/workplace emergency detection milestone (Fall vs. Normal ADL) before scaling to multi-class violence and panic detection.
