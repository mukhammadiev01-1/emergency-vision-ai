# Emergency Vision AI

A production-oriented computer vision and AI platform for real-time video analytics, edge inference, and emergency response intelligence.

---

## 🏗️ System Architecture

```
                                +---------------------------+
                                |  Client / Dashboard / App |
                                +-------------+-------------+
                                              | HTTP / REST
                                              v
                                +---------------------------+
                                |      apps/api (FastAPI)   |
                                |  - Stream Management      |
                                |  - Event Query & Stats    |
                                |  - Health & Metadata      |
                                +-------------+-------------+
                                              |
                          (Future Message Broker / Direct Delegation)
                                              |
                                              v
                                +---------------------------+
                                |   apps/worker (CV Engine) |
                                |  - Video Ingest (RTSP/File|
                                |  - YOLO Detection (v11)   |
                                |  - ByteTrack Tracker      |
                                |  - Line-Crossing Detector |
                                |  - Action Recog (R3D-18)  |
                                +---------------------------+
```

### Core Architectural Principles
1. **Separation of Concerns**: API routes handle HTTP contracts, authentication, stream registries, and event queries. Heavy GPU / CV processing is strictly encapsulated in dedicated worker pipelines.
2. **Modular Pipeline**: Video processing is organized into isolated, reusable stages: `capture` → `preprocess` → `inference` → `tracking` → `segmentation` → `action_recognition` → `events` → `postprocess`.
3. **Pydantic Type Safety**: All inputs, outputs, configurations, and event payloads use strict Pydantic schemas.
4. **Unified Configuration**: Config is centralized across `apps/api/app/config.py` and `apps/worker/app/config.py` backed by `.env`.
5. **Research Integrity**: Experimental research is preserved in `notebooks/` without polluting production services.

---

## 📁 Repository Structure

```
emergency-vision-ai/
├── apps/
│   ├── api/                          # FastAPI Service
│   │   └── app/
│   │       ├── main.py               # FastAPI entry point
│   │       ├── config.py             # API configuration
│   │       ├── api/
│   │       │   ├── dependencies.py   # Dependency injection
│   │       │   └── routes/
│   │       │       ├── health.py     # Health & readiness checks
│   │       │       ├── inference.py  # Detection endpoints
│   │       │       ├── streams.py    # Stream registration & lifecycle
│   │       │       └── events.py     # Event query & statistics
│   │       ├── schemas/              # Pydantic data contracts
│   │       │   ├── detection.py      # Bounding boxes & detections
│   │       │   ├── stream.py         # Stream requests & responses
│   │       │   └── event.py          # Line crossing & spatial events
│   │       └── services/             # Business logic layer
│   │           ├── inference_service.py
│   │           ├── stream_service.py
│   │           └── event_service.py
│   │
│   └── worker/                       # Computer Vision Worker Service
│       └── app/
│           ├── main.py               # Worker pipeline CLI runner
│           ├── config.py             # Worker & model configuration
│           ├── models/               # Model loaders & wrappers
│           │   ├── model_loader.py   # Unified model manager
│           │   ├── yolo.py           # YOLOv11 & ByteTrack wrapper
│           │   └── action_model.py   # R3D-18 action recognition
│           └── pipeline/             # Modular CV pipeline stages
│               ├── capture.py        # Video & RTSP stream ingest
│               ├── preprocess.py     # Resizing & frame skipping
│               ├── inference.py      # Object detection stage
│               ├── tracking.py       # ByteTrack tracking stage
│               ├── segmentation.py   # Instance segmentation stage
│               ├── action_recognition.py # 3D CNN action stage
│               ├── events.py         # Line-crossing event detector
│               └── postprocess.py    # Visual annotations & HUD
│
├── models/                           # Model weights storage
│   ├── detection/                    # YOLO (.pt, .onnx)
│   └── action_recognition/           # R3D-18 weights
│
├── notebooks/                        # Experimental research history
│   └── 01_video_opencv_basics.ipynb  # Baseline Colab experiments
│
├── scripts/                          # Platform utilities
│   ├── download_models.py            # Fetch baseline weights
│   ├── export_models.py              # Export models to ONNX
│   └── benchmark.py                  # Measure FPS & latency
│
├── tests/                            # Unit & integration tests
│   ├── test_api_health.py
│   ├── test_schemas.py
│   └── test_line_crossing.py
│
├── infra/
│   └── docker/
│       ├── Dockerfile.api
│       └── Dockerfile.worker
│
├── requirements.txt                  # Full project dependencies
├── requirements-api.txt              # API-only lightweight dependencies
├── requirements-worker.txt           # CV Worker dependencies
├── docker-compose.yml                # Multi-service container setup
├── .env.example                      # Configuration template
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and enter workspace
cd emergency-vision-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 2. Download Baseline Models
```bash
python3 scripts/download_models.py --yolo-model yolo11n.pt
```

### 3. Run the API Service
```bash
uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 4. Run the Worker Pipeline
```bash
# Process a video or RTSP stream with line-crossing detection:
python3 -m apps.worker.app.main --source path/to/video.mp4 --device cpu
```

### 5. Benchmark Latency & FPS
```bash
python3 scripts/benchmark.py --model models/detection/yolo11n.pt --device cpu --iterations 50
```

### 6. Run via Docker Compose
```bash
docker-compose up --build
```

---

## 🧪 Testing

Run test suites across API endpoints, schemas, and line-crossing logic:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 🛣️ Roadmap & Next Steps
- [ ] Connect API & Worker via Redis Pub/Sub / Queue
- [ ] Add WebSocket stream for real-time client alerts
- [ ] Add PostgreSQL persistence for historical events and stream registries
- [ ] TensorRT acceleration for edge deployment
- [ ] Multi-camera concurrent pipeline management
