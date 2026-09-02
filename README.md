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
- **Emergency Action Model**: [`models/action_recognition/r3d18_urfd_best.pth`](models/action_recognition/r3d18_urfd_best.pth) (127 MB, SHA-256: `2e4f379ca7d89858edc077aa6202d0ea537f4d312ace18accd63bffcd9be9920`)

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

### 3. Model Training
The action recognition model was trained on URFD using two-stage transfer learning:
- **Backbone**: `ResNet3D-18` initialized with official Torchvision Kinetics-400 pre-trained weights (`R3D_18_Weights.DEFAULT`).
- **Stage 1 (Head Warm-up)**: Backbone frozen, custom binary classifier head (`fc: 512 -> 2`) trained with AdamW.
- **Stage 2 (Fine-tuning)**: Differential learning rates with Layer 3 and Layer 4 unfrozen.
- **Data Splitting**: Strict sequence-level isolation (70% train / 15% val / 15% test, Seed=42) preventing temporal data leakage.

To re-train:
```bash
python3 scripts/train_action_model.py \
    --dataset-root data/urfd \
    --output-dir models/action_recognition \
    --checkpoint-name r3d18_urfd_best.pth \
    --stage1-epochs 5 \
    --stage2-epochs 20 \
    --batch-size 4 \
    --device cuda
```

### 4. Model Evaluation
Evaluate the canonical checkpoint against the held-out test split:
```bash
python3 scripts/evaluate_action_model.py \
    --checkpoint models/action_recognition/r3d18_urfd_best.pth \
    --dataset-root data/urfd \
    --device cuda \
    --seed 42
```

### 5. Production Pipeline GPU Benchmark
Run the canonical hardware benchmark with CUDA synchronization:
```bash
python3 scripts/benchmark_gpu.py \
    --video data/urfd/videos/fall/fall-01-cam0.mp4 \
    --action-model models/action_recognition/r3d18_urfd_best.pth \
    --yolo-model models/detection/yolo11n.pt \
    --device cuda \
    --threshold 0.70 \
    --interval 8 \
    --warmup 5
```
Metrics measured:
- YOLO11n Detection + ByteTrack latency (Mean, P50, P95)
- Person crop extraction & preprocessing latency
- R3D-18 GPU inference latency (Mean, P50, P95)
- End-to-end frame latency (Mean, P50, P95)
- Total pipeline throughput (FPS)

### 6. Google Colab Workflow
Open [`notebooks/08_urfd_training_colab.ipynb`](notebooks/08_urfd_training_colab.ipynb) in Google Colab on a Tesla T4 GPU:
- **Cell 1**: Mount Google Drive & check CUDA GPU.
- **Cell 2**: Clone/update repository to `/content/emergency-vision-ai`.
- **Cell 3**: Install worker dependencies.
- **Cell 4**: Verify production module imports.
- **Cell 5**: Verify canonical checkpoint & URFD dataset structure.
- **Cell 6**: Action model forward pass smoke test on CUDA.
- **Cell 7**: Model evaluation on held-out test split.
- **Cell 8**: Production GPU benchmark using `scripts/benchmark_gpu.py`.

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

Run the full automated test suite (63 unit & integration tests):
```bash
.venv/bin/pytest -v
```
