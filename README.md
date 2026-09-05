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
├── tests/                            # Comprehensive Test Suite (63 tests)
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
# Run with default webcam (camera index 0)
python3 scripts/run_camera_demo.py --camera-index 0

# Run with custom parameters or device
python3 scripts/run_camera_demo.py \
    --camera-index 0 \
    --threshold 0.70 \
    --consecutive 2 \
    --interval 8 \
    --padding 0.05 \
    --device auto
```
Real-time features:
- Live OpenCV window displaying rolling FPS, track IDs, bounding boxes, and action classifications
- Real-time latency HUD (Detection/tracking ms, Action inference ms, E2E frame ms)
- Prominent Emergency Alert overlay upon confirmed 2-window fall detection
- Graceful shutdown with `q` or Ctrl+C

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

# Install dependencies
pip install -r requirements.txt
git lfs pull
```

### 2. Run the Worker Pipeline (with Action Recognition)
```bash
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
- Interactive Docs: `http://localhost:8000/docs`
- Real-time Events WebSocket: `ws://localhost:8000/api/v1/events/ws`

---

## 🧪 Testing

Run the full automated test suite (82 unit & integration tests):
```bash
.venv/bin/pytest -v
```
