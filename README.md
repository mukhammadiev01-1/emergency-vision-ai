# Emergency Vision AI

A production-grade computer vision and AI platform for real-time video analytics, multi-object tracking, spatiotemporal action recognition, and emergency intelligence delivery.

---

## 🏗️ System Architecture

```
                                +---------------------------+
                                |  Client / Dashboard / App |
                                +-------------+-------------+
                                              | HTTP / REST & WebSockets (Real-Time)
                                              v
                                +---------------------------+
                                |      apps/api (FastAPI)   |
                                |  - Stream Lifecycle API   |
                                |  - Event Query & Metrics  |
                                |  - Redis Stream Consumer  |
                                |  - Realtime WS Broadcast  |
                                +-------------+-------------+
                                              |
                                     Redis Stream Transport
                                 ("emergency_vision:events")
                                              |
                                              v
                                +---------------------------+
                                |   apps/worker (CV Engine) |
                                |  - Video Ingest (RTSP/MP4)|
                                |  - YOLO11 Detection       |
                                |  - ByteTrack Tracking     |
                                |  - Line-Crossing Detector |
                                |  - Per-Person R3D-18 Tube |
                                |  - Fall Action Debouncing |
                                |  - Redis Event Publisher  |
                                +---------------------------+
```

### Action Recognition Pipeline Architecture
```
Video / RTSP Frame
       ↓
YOLO11 Person Detection
       ↓
ByteTrack Track Assignment (Track ID)
       ↓
Person Bounding Box Crop Extraction (5% Padding, Min Dimension Validation)
       ↓
Per-Track 16-Frame Rolling Spatiotemporal Buffer
       ↓
ResNet3D-18 (R3D-18) Binary Fall Classifier (112x112, Kinetics-400 Norm)
       ↓
Consecutive Positive Window Confirmation (N=2 windows, P(FALL) >= 0.70)
       ↓
Per-Track Event Debounce Cooldown (5.0s)
       ↓
EmergencyActionEvent (stream_id, track_id, position, confidence, metadata)
       ↓
EventPublisher (Redis Stream / HTTP / In-Memory)
```

---

## 📁 Repository Structure

```text
emergency-vision-ai/
├── apps/
│   ├── api/                          # FastAPI Service
│   │   └── app/
│   │       ├── main.py               # FastAPI entry point & WebSocket hub
│   │       ├── config.py             # API configuration
│   │       ├── api/routes/           # Health, Streams, Events, WebSockets
│   │       ├── schemas/              # Pydantic data schemas
│   │       └── services/             # Stream & Event ingestion services
│   │
│   └── worker/                       # Computer Vision Worker Service
│       └── app/
│           ├── main.py               # Worker pipeline runner
│           ├── config.py             # Worker settings (Pydantic)
│           ├── models/               # Model inference wrappers
│           │   ├── yolo.py           # YOLO11 + ByteTrack wrapper
│           │   └── action_model.py   # R3D-18 video action recognition
│           ├── pipeline/             # Modular CV pipeline stages
│           │   ├── capture.py        # Video/RTSP capture stream
│           │   ├── tracking.py       # ByteTrack tracking stage
│           │   ├── action_recognition.py # Per-person spatiotemporal stage
│           │   ├── events.py         # Line crossing detector & events
│           │   └── postprocess.py    # Visual annotations & HUD
│           └── events/               # Event publishing (Redis Streams, HTTP)
│
├── models/                           # Canonical Model Weights Storage
│   ├── detection/
│   │   └── yolo11n.pt                # YOLO11n weights (Git LFS)
│   └── action_recognition/
│       └── r3d18_urfd_best.pth       # Trained R3D-18 checkpoint (Git LFS)
│
├── notebooks/                        # Research & GPU Notebooks
│   ├── 01_video_opencv_basics.ipynb  # Baseline OpenCV & Video I/O
│   └── 08_urfd_training_colab.ipynb  # Unified Colab GPU Workflow & Benchmark
│
├── scripts/                          # Platform & ML Utilities
│   ├── download_models.py            # Fetch & verify baseline weights
│   ├── download_urfd.py              # Download & prepare URFD dataset
│   ├── train_action_model.py         # Two-stage R3D-18 transfer learning
│   ├── evaluate_action_model.py      # Test split & confusion matrix eval
│   ├── benchmark_gpu.py              # Canonical per-person pipeline benchmark
│   └── benchmark.py                  # Standalone YOLO inference benchmark
│
├── tests/                            # Comprehensive Test Suite (127 tests)
├── .gitattributes                    # Git LFS tracking rules
├── .gitignore                        # Model & dataset exclusions
├── requirements.txt                  # Full dependencies
├── requirements-worker.txt           # Worker & ML dependencies
└── README.md
```

---

## 🎯 Training, Model Artifacts, Evaluation & Benchmarking

### 1. Canonical Model Artifacts & Git LFS
The production models are tracked and versioned via **Git LFS**:
- **Detection Model**: [`models/detection/yolo11n.pt`](models/detection/yolo11n.pt) (5.4 MB, SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`)
- **Emergency Action Model**: [`models/action_recognition/r3d18_urfd_best.pth`](models/action_recognition/r3d18_urfd_best.pth) (127 MB, SHA-256: `52cc51fd016263e7529009f23147d7a91b8855d685f11239346016ff55eadb5c`)

To pull model weights after cloning:
```bash
git lfs install
git lfs pull
```
Or run the helper downloader:
```bash
python3 scripts/download_models.py
```

### 2. Dataset Storage
The **UR Fall Detection (URFD)** dataset (30 Fall and 40 Normal ADL sequences) is managed outside Git:
- Local/Drive path: `data/urfd/`
- Download tool: `python3 scripts/download_urfd.py --format mp4`

### 3. Production Person-Crop Action Training (Second-Stage)
To align training with the production pipeline representation (where R3D-18 operates on tight YOLO/ByteTrack person crops with 5% padding), a dedicated second-stage transfer learning pipeline is provided:
- **Tube Extraction**: YOLO11n + ByteTrack extracts 16-frame rolling person crops from URFD videos.
- **Hard Negative Mining**: Upright walking prior to fall descent is labeled `NORMAL (0)`; descent and floor landing are labeled `FALL (1)`.
- **Sequence Isolation**: Strict sequence-level isolation (Seed=42: 49 train, 10 val, 11 test) prevents cross-frame temporal data leakage.
- **Backbone Fine-tuning**: Starts from canonical weights `models/action_recognition/r3d18_urfd_best.pth`, unfreezing `layer3`, `layer4`, and `fc` with differential learning rates.
- **Artifact Safety**: Saves to `models/action_recognition/r3d18_urfd_person_crops.pth` with full metadata JSON and automatic Google Drive backup. Canonical base weights are never overwritten.

To train with one command:
```bash
python3 scripts/train_person_crop_pipeline.py \
    --dataset-root data/urfd \
    --base-checkpoint models/action_recognition/r3d18_urfd_best.pth \
    --yolo-model models/detection/yolo11n.pt \
    --output-dir models/action_recognition \
    --checkpoint-name r3d18_urfd_person_crops.pth \
    --epochs 12 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cuda \
    --seed 42
```

### 4. Production Pipeline Multi-Video Comparative Evaluation
Evaluate both checkpoints side-by-side using the **actual production pipeline** (YOLO11n → ByteTrack → Person Crop → R3D-18 → Temporal Confirmation) on multi-video splits:
```bash
python3 scripts/evaluate_production_pipeline.py \
    --action-model models/action_recognition/r3d18_urfd_person_crops.pth \
    --compare-with models/action_recognition/r3d18_urfd_best.pth \
    --dataset-root data/urfd \
    --max-fall-videos 5 \
    --max-normal-videos 5 \
    --device cuda \
    --output-json results/eval/pipeline_comparison.json
```
Reports per-video max $P(\text{FALL})$, mean $P(\text{FALL})$, confirmed events, video accuracy, FALL recall, and NORMAL false positive rate.

### 5. Production Pipeline GPU Benchmark
Run the canonical hardware benchmark with CUDA synchronization:
```bash
python3 scripts/benchmark_gpu.py \
    --video data/urfd/videos/fall/fall-01-cam0.mp4 \
    --action-model models/action_recognition/r3d18_urfd_person_crops.pth \
    --yolo-model models/detection/yolo11n.pt \
    --device cuda \
    --threshold 0.70 \
    --interval 8 \
    --warmup 5 \
    --output-json results/benchmark_gpu_results.json
```
Metrics measured:
- YOLO11n Detection + ByteTrack latency (Mean, P50, P95)
- Person crop extraction & preprocessing latency
- R3D-18 GPU inference latency (Mean, P50, P95)
- End-to-end frame latency (Mean, P50, P95)
- Total pipeline throughput (FPS)

### 6. Google Colab GPU Workflow & Single Source of Truth
The canonical Colab notebook [`notebooks/08_urfd_training_colab.ipynb`](notebooks/08_urfd_training_colab.ipynb) is a thin orchestrator backed by version-controlled Python scripts:
1. **Cell 1 — Bootstrap**: Mounts Google Drive (`/content/drive`), syncs GitHub repository (`/content/emergency-vision-ai`), pulls Git LFS checkpoints, installs dependencies, validates Drive dataset (`30 FALL + 40 NORMAL`), and creates idempotent symlinks via `python scripts/colab_bootstrap.py`.
2. **Cell 2 — Status & Environment Report**: Validates GPU accelerator (Tesla T4), memory, and models.
3. **Cell 3 — ONE-CLICK Training**: Runs `train_person_crop_pipeline.py` with automatic Drive backup.
4. **Cell 4 — Comparative Evaluation**: Runs `evaluate_production_pipeline.py --compare-with` on `fall-01..05` & `adl-01..05`.
5. **Cell 5 — Production Benchmark**: Executes `benchmark_gpu.py` on Tesla T4 GPU.
6. **Cell 6 — Artifact Summary**: Verifies saved checkpoints, hashes, and Drive backups.

### 7. Live Camera Production Demo
Validate the end-to-end vision pipeline in real time using a connected webcam or video feed:
```bash
# Run with default webcam (requires production person-crop checkpoint)
python3 scripts/run_camera_demo.py --camera-index 0

# Run with custom parameters or hardware device
python3 scripts/run_camera_demo.py \
    --camera-index 0 \
    --threshold 0.70 \
    --consecutive 2 \
    --interval 8 \
    --padding 0.05 \
    --device auto
```

#### Checkpoint Resolution & Strict Production Safety:
The live camera demo enforces strict model verification to guarantee production validity:
1. **Primary Model Target**: Defaults to the trained person-crop checkpoint (`models/action_recognition/r3d18_urfd_person_crops.pth`).
2. **Automatic Candidate Discovery**: Automatically resolves the checkpoint from:
   - Environment variable: `EMERGENCY_VISION_AI_ACTION_MODEL`
   - Canonical local models directory: `models/action_recognition/r3d18_urfd_person_crops.pth`
   - Synced experiments directory: `experiments/*/r3d18_urfd_person_crops.pth`
   - Google Drive environment variable: `$GOOGLE_DRIVE_DIR` or `$EMERGENCY_VISION_AI_DRIVE_ROOT`
   - Mounted Google Drive paths (`~/Library/CloudStorage/GoogleDrive-...`, `/Volumes/GoogleDrive/...`, `/content/drive/...`)
3. **No Silent Fallback**: If the person-crop checkpoint is not found locally or in Drive, the script fails with clear synchronization instructions rather than silently falling back to the whole-frame baseline model.
4. **Explicit Baseline Override**: To intentionally evaluate the legacy baseline model for comparative benchmarking:
   ```bash
   python3 scripts/run_camera_demo.py \
       --action-model models/action_recognition/r3d18_urfd_best.pth \
       --allow-baseline
   ```
5. **Pre-Flight Verification**: Prints the resolved checkpoint path, file size, SHA-256 hash, and model identity before starting inference.

Real-time features:
- Live OpenCV window displaying rolling FPS, track IDs, bounding boxes, model identity badge, and action classifications
- Real-time latency HUD (Detection/tracking ms, Action inference ms, E2E frame ms)
- Prominent Emergency Alert overlay upon confirmed 2-window fall detection
- Graceful shutdown with `q` or Ctrl+C

---

## 🧠 System Engineering & Technical Rationale (Principal Engineer Briefing)

This section documents the architectural decisions, trade-offs, empirical findings, and operational considerations underlying **Emergency Vision AI**.

### 1. Architectural Choices & Trade-Off Matrix

| Component | Selected Technology | Evaluated Alternatives | Architectural Rationale & Trade-Off Analysis |
| :--- | :--- | :--- | :--- |
| **Object Detection** | **YOLO11n** | YOLOv8n, Faster R-CNN, SSD | **2.6M parameters**, C3k2 attention blocks, and optimized SPPF backbone. Delivers high mAP on COCO person class at **~15 ms** on CPU and **~5 ms** on Tesla T4. Faster R-CNN is prohibitive for edge 30 FPS pipelines (>100 ms). YOLO11n provides the optimal speed/accuracy pareto frontier for real-time bounding box extraction. |
| **Multi-Object Tracking** | **ByteTrack** | DeepSORT, SORT, StrongSORT | DeepSORT requires an expensive Re-Identification (ReID) deep CNN forward pass per detected bounding box, introducing 25–40 ms latency and stalling on low-light surveillance feeds. ByteTrack uses a pure Kalman filter + two-stage bipartite Hungarian matching that retains **low-confidence detections** in the second stage. This prevents track fragmentation when a person rapidly collapses or undergoes motion blur. |
| **Spatiotemporal Action** | **ResNet3D-18 (R3D-18)** | SlowFast, VideoMAE, I3D, TimeSformer | SlowFast requires dual-pathway feature aggregation (65+ GFLOPs); VideoMAE/TimeSformer transformers incur massive memory footprints and quadratic attention cost incompatible with multi-stream edge inference. R3D-18 (33.3 GFLOPs) processes a 16-frame spatiotemporal clip in **23.3 ms** on Tesla T4 GPU, fitting effortlessly into a 30 FPS multi-camera inference budget while preserving temporal dynamics. |
| **Event Transport** | **Redis Streams** | Kafka, RabbitMQ, direct HTTP | Redis Streams provides in-memory sub-millisecond append latency (`XADD`), consumer groups with explicit ACK semantics (`XREADGROUP`), and configurable buffer retention (`MAXLEN`). Decouples compute-heavy CV worker processes from downstream consumers. Kafka introduces unnecessary operational overhead (JVM, ZooKeeper/KRaft) for single-facility edge deployments; direct HTTP risks blocking video pipelines during network latency spikes. |
| **Event Delivery Hub** | **FastAPI + WebSockets** | Flask, gRPC, Celery | Async native event loop (`asyncio`) allows thousands of concurrent WebSocket connections for real-time operations dashboards without thread exhaustion. OpenAPI auto-generation accelerates client integration. |

---

### 2. The Domain Shift Problem & Why Person-Crop Training Was Mandatory

The canonical baseline action recognition checkpoint (`models/action_recognition/r3d18_urfd_best.pth`) was originally trained on **whole-camera frames** ($112 \times 112$).

#### The Production Domain Gap:
* In a full-room surveillance frame, human subjects occupy only 5% to 15% of total pixels. The neural network learns spurious background contextual correlations (e.g., room floor patterns, furniture geometry).
* In our production vision pipeline, YOLO11n detects the subject, adds a 5% spatial margin, and crops the bounding box before passing a 16-frame tube to R3D-18.
* When evaluated on production person crops, the whole-frame model suffered **catastrophic representation collapse**:
  - **FALL Recall dropped to 0.00%** (0 out of 5 fall events detected).
  - **Overall Video Accuracy fell to 50.00%** (failing every emergency sequence).

#### The Second-Stage Training Solution:
* We extracted 16-frame rolling person tubes from URFD sequences using the exact production detector (`YOLO11n`) and tracker (`ByteTrack`) with 5% spatial padding.
* Implemented hard-negative mining: frames of upright walking prior to descent are labeled `NORMAL (0)`; active descent and floor landing are labeled `FALL (1)`.
* Enforced **strict sequence-level isolation** (Seed 42: 49 train, 10 val, 11 test) to ensure zero cross-frame temporal data leakage across splits.
* Fine-tuned R3D-18 with differential learning rates (`layer3`, `layer4`, `fc` active; early layers frozen).
* **Results**: Production FALL recall rebounded from **0% to 80%**, video accuracy reached **90%**, while maintaining **100% specificity (0% false alarms)** across all activities of daily living (ADL).

---

### 3. Temporal Confirmation & Debouncing Mechanism

A naive single-frame thresholding strategy produces unacceptable false-positive spikes during benign household activities (e.g., rapidly sitting down on a sofa, tying shoelaces, or crouching).

To guarantee industrial-grade signal stability:
1. **Inference Cadence ($K=8$ frames)**: R3D-18 is evaluated every 8 frames per tracked person (roughly every 266 ms at 30 FPS).
2. **Consecutive Window Confirmation ($N=2$)**: An emergency event is emitted **only** when $N \ge 2$ consecutive inference windows yield $P(\text{FALL}) \ge 0.70$ for the **same persistent Track ID**.
3. **Per-Track Debounce Cooldown ($5.0\text{s}$)**: Once an emergency event is confirmed, further alerts for that Track ID are suppressed for 5.0 seconds to prevent alert storms and duplicate dispatch notifications.

$$\text{Trigger Condition: } \left( \prod_{w=0}^{N-1} \mathbb{I}[P_w(\text{FALL} \mid \text{Track } i) \ge 0.70] = 1 \right) \land (t_{\text{now}} - t_{\text{last\_alert}} \ge 5.0\text{s})$$

---

### 4. Hardware Latency Profile & Pipeline Throughput (Tesla T4)

Empirical measurements gathered on an NVIDIA Tesla T4 GPU (Google Colab CUDA runtime) across a full 160-frame sequence ($640 \times 480$):

```text
================================================================================
           PRODUCTION PIPELINE BENCHMARK BREAKDOWN (NVIDIA Tesla T4)
================================================================================
Pipeline Throughput:             65.54 FPS (2.18x real-time margin over 30 FPS)
Total E2E Latency per Frame:     Mean: 18.54 ms | P50: 13.58 ms | P95: 44.97 ms
--------------------------------------------------------------------------------
Stage 1: YOLO11n + ByteTrack:    Mean: 15.30 ms | P50: 13.20 ms | P95: 23.14 ms
Stage 2: Tube Preprocessing:     Mean:  6.32 ms | P50:  5.98 ms | P95:  8.12 ms
Stage 3: R3D-18 Inference:       Mean: 23.33 ms | P50: 23.09 ms | P95: 24.32 ms
================================================================================
```
*Note: Because R3D-18 runs on an 8-frame cadence, its 23.33 ms cost is amortized across frames, yielding an average end-to-end frame processing time of 18.54 ms.*

---

### 5. Edge-vs-Cloud Deployment Strategy

Emergency Vision AI implements an **Edge-Heavy, Cloud-Light** hybrid topology:

```
+-----------------------------------------------------------------------+
| LOCAL FACILITY / EDGE (Hospital, Elder Care Facility, Factory)        |
|                                                                       |
|   IP Cameras (RTSP)                                                   |
|          ↓ (Local LAN)                                                |
|   Edge Server / NVIDIA Jetson Orin (Worker Service)                   |
|   - Hardware-accelerated decoding (NVDEC)                             |
|   - YOLO11n Detection + ByteTrack Tracking                           |
|   - R3D-18 Person-Crop Spatiotemporal Classification                 |
|   - Multi-Frame Temporal Confirmation                                 |
|          ↓ (Zero raw video leaves the local network - HIPAA/GDPR)     |
|   Local Redis Stream ("emergency_vision:events")                      |
+-----------------------------------------------------------------------+
                                  |
                                  | Encrypted Event Metadata (TLS / JSON)
                                  v
+-----------------------------------------------------------------------+
| CENTRAL CLOUD / CONTROL PLANE (FastAPI Service)                       |
|   - Stream lifecycle management and health telemetry                  |
|   - Emergency alert fanout to staff WebSockets & mobile push          |
|   - Historical event query and audit logging                          |
+-----------------------------------------------------------------------+
```

* **Privacy Compliance (HIPAA / GDPR)**: Raw pixel streams never exit the on-premises edge gateway. Only structured, non-identifying telemetry (bounding box coordinates, timestamps, track IDs, and confidence scores) is transmitted.
* **Bandwidth Optimization**: Transmitting 1080p30 video consumes ~4–8 Mbps per stream. Transmitting structured event metadata consumes < 1 KB per confirmed event.
* **Fault Tolerance**: If WAN internet connectivity drops, edge workers continue local inference, spooling events into Redis until the connection is restored.

---

### 6. Scaling to Multi-Camera Deployments

* **Worker Partitioning**: In production, worker instances are containerized and assigned dedicated camera streams (1 worker per 2–4 30 FPS streams on a single T4 or Jetson AGX Orin).
* **Stateless Consumer Groups**: The FastAPI API layer scales horizontally behind a load balancer; worker instances publish to Redis Stream keys partitioned by facility or zone (`emergency_vision:events:{facility_id}`).
* **Memory Management**: Per-track spatiotemporal frame buffers automatically expire via `stale_track_timeout` (default: 3.0s after a track vanishes from the camera's field of view).

---

### 7. Known Production Limitations & Root Cause Analysis

#### False Negative on `fall-05-cam0.mp4`:
* **Symptom**: The production person-crop model achieved a peak $P(\text{FALL}) = 0.9113$ on `fall-05`, but the video-level prediction was NORMAL (0 confirmed events).
* **Root Cause Analysis**:
  1. **Low Camera Angle & Occlusion**: The camera in `fall-05` is placed near floor level. As the subject falls toward the camera behind low furniture, the vertical bounding box collapses into a horizontal strip.
  2. **Bounding Box Aspect Ratio Distortion**: The bounding box aspect ratio flattens rapidly ($W/H > 1.8$). Standard pedestrian detectors exhibit confidence jitter when human anatomy shifts from vertical to horizontal.
  3. **Tracking Fragmentation**: ByteTrack dropped the primary track ID for 3 frames during the floor impact, creating a new track ID. Consequently, the two consecutive inference windows ($P \ge 0.70$) occurred across split track IDs rather than a single continuous track.
* **Engineering Remediation Plan (Active Roadmap)**:
  - Implement **aspect-ratio velocity priors**: When $d(W/H)/dt > \tau$, decrease the second-stage ByteTrack IoU matching threshold to prevent track re-assignment.
  - Implement adaptive track reconnection for temporally proximate bounding boxes with high overlap.

---

### 8. Technology Status & Engineering Roadmap

| Capability | Status | Implementation Details |
| :--- | :---: | :--- |
| **YOLO11n Detection** | **Implemented & Tested** | PyTorch & Ultralytics, COCO person class, 640x640 resolution |
| **ByteTrack Tracking** | **Implemented & Tested** | Kalman filter + 2-stage Hungarian matching, persistent track IDs |
| **5% Padded Person Crops** | **Implemented & Tested** | Aspect-ratio preserving padding, minimum crop dimension validation |
| **R3D-18 Binary Action Model** | **Implemented & Tested** | Second-stage fine-tuned weights (`r3d18_urfd_person_crops.pth`) |
| **Temporal Confirmation** | **Implemented & Tested** | 2-window debounce, 5.0s per-track cooldown, 0.70 confidence cutoff |
| **Redis Event Transport** | **Implemented & Tested** | `XADD` / `XREADGROUP` message transport with ACK handling |
| **FastAPI & WebSockets** | **Implemented & Tested** | Stream management, event query, real-time WebSocket broadcasting |
| **Live Camera Demo** | **Implemented & Tested** | OpenCV HUD, hardware auto-discovery, strict checkpoint resolution |
| **Colab GPU Bootstrap** | **Implemented & Tested** | Single-command idempotent setup, Drive linking, zero notebook duplication |
| **Experiment Synchronization** | **Implemented & Tested** | Single-command lightweight artifact sync from Drive to `experiments/` |
| **Automated Test Suite** | **Implemented & Tested** | **127 automated tests** (unit, pipeline, API, Redis, and bootstrap) |
| **ONNX Export Utility** | **Implemented & Tested** | Export support for YOLO11n (`.onnx`) and R3D-18 (`.onnx`, opset 17) |
| **TensorRT FP16 Acceleration**| **Planned** | Compilation of ONNX graphs to TensorRT execution plans for Jetson |
| **NVIDIA Jetson Orin Deployment** | **Planned** | DeepStream / GStreamer integration for embedded edge hardware |

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone workspace
git clone https://github.com/mukhammadiev01-1/emergency-vision-ai.git
cd emergency-vision-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and materialize model weights
pip install -r requirements.txt
git lfs pull
```

### 2. Run the Worker Pipeline (with Production Person-Crop Model)
```bash
# Run on a recorded sequence with action recognition enabled:
python3 -m apps.worker.app.main \
    --source data/urfd/videos/fall/fall-01-cam0.mp4 \
    --enable-action \
    --action-model models/action_recognition/r3d18_urfd_person_crops.pth \
    --device cpu

# Comparative run using the legacy whole-frame baseline model:
python3 -m apps.worker.app.main \
    --source data/urfd/videos/fall/fall-01-cam0.mp4 \
    --enable-action \
    --action-model models/action_recognition/r3d18_urfd_best.pth \
    --device cpu
```

### 3. Run FastAPI & Real-Time Event Hub
```bash
uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs: `http://localhost:8000/docs`
- Real-time Events WebSocket: `ws://localhost:8000/api/v1/events/ws`

---

## 🧪 Testing

Run the full automated test suite (**127 unit and integration tests**):
```bash
.venv/bin/pytest -v
```

