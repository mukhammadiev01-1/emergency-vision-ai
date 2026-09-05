"""Person-Crop Tube Dataset for Production-Aligned Action Recognition.

Extracts spatiotemporal 16-frame person tubes from URFD video sequences using the
exact production representation:
    Video -> YOLO11n + ByteTrack -> Person Crop (5% padding) -> 16-Frame Tube -> R3D-18 Normalization.

Preserves strict sequence-level isolation (Seed=42) across train, val, and test splits.
"""
from dataclasses import dataclass
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from apps.worker.app.datasets.urfd_dataset import (
    create_urfd_splits,
    CLASS_NAMES,
    LABEL_FALL,
    LABEL_NORMAL,
    VIDEO_MEAN,
    VIDEO_STD,
)
from apps.worker.app.models.action_model import preprocess_clip_frames
from apps.worker.app.pipeline.action_recognition import extract_person_crop

logger = logging.getLogger("person_crop_dataset")


@dataclass
class TubeSample:
    """Metadata and tensor for a single 16-frame person crop tube."""
    sequence_id: str
    track_id: int
    start_frame: int
    end_frame: int
    label: int
    class_name: str
    tensor: torch.Tensor  # Shape: (3, 16, 112, 112) normalized


def extract_tubes_from_video(
    video_path: str,
    sequence_id: str,
    sequence_label: int,
    yolo_model,
    stride: int = 4,
    padding_ratio: float = 0.05,
    min_track_len: int = 16,
) -> List[TubeSample]:
    """Process a video with YOLO + ByteTrack and extract 16-frame person crop tubes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open video file: %s", video_path)
        return []

    # Map track_id -> List of (frame_idx, crop_img, width, height, y_center)
    tracks: Dict[int, List[Tuple[int, np.ndarray, float, float, float]]] = {}
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run tracking
            results = yolo_model.track(frame, conf=0.35, iou=0.45, classes=[0], persist=True)[0]
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().tolist()

                for box, tid in zip(boxes, track_ids):
                    crop = extract_person_crop(frame, tuple(map(int, box)), padding_ratio=padding_ratio)
                    if crop is not None:
                        w = float(box[2] - box[0])
                        h = float(box[3] - box[1])
                        yc = float(box[1] + box[3]) / 2.0
                        if tid not in tracks:
                            tracks[tid] = []
                        tracks[tid].append((frame_idx, crop, w, h, yc))

            frame_idx += 1
    finally:
        cap.release()

    tube_samples: List[TubeSample] = []

    for tid, crops in tracks.items():
        if len(crops) < min_track_len:
            continue

        # Slide window of size 16 with specified stride
        for start_idx in range(0, len(crops) - 16 + 1, stride):
            window = crops[start_idx : start_idx + 16]
            crop_imgs = [item[1] for item in window]
            start_f = window[0][0]
            end_f = window[-1][0]

            # Bounding box motion features across window
            dy = window[-1][4] - window[0][4]
            aspect_end = window[-1][2] / max(1.0, window[-1][3])
            aspect_start = window[0][2] / max(1.0, window[0][3])

            # Label assignment:
            # For ADL/NORMAL sequences, every tube is NORMAL (0).
            # For FALL sequences:
            #   If window exhibits fall descent (dy >= 8) or horizontal landing (aspect >= 0.75): FALL (1).
            #   If window is before descent (upright walking): NORMAL (0), providing hard negative mining!
            #   Otherwise, if in a FALL sequence and not clearly upright: FALL (1).
            if sequence_label == LABEL_NORMAL:
                tube_label = LABEL_NORMAL
            else:
                is_upright_walking = (aspect_start < 0.55 and aspect_end < 0.55 and abs(dy) < 8.0)
                is_fall_motion = (dy >= 8.0 or aspect_end >= 0.75 or (aspect_end - aspect_start) > 0.25)

                if is_upright_walking:
                    tube_label = LABEL_NORMAL
                elif is_fall_motion:
                    tube_label = LABEL_FALL
                else:
                    tube_label = LABEL_FALL

            clip_tensor = preprocess_clip_frames(crop_imgs).squeeze(0)  # (3, 16, 112, 112)

            tube_samples.append(
                TubeSample(
                    sequence_id=sequence_id,
                    track_id=tid,
                    start_frame=start_f,
                    end_frame=end_f,
                    label=tube_label,
                    class_name=CLASS_NAMES[tube_label],
                    tensor=clip_tensor,
                )
            )

    return tube_samples


class PersonCropDataset(Dataset):
    """PyTorch Dataset yielding (3, 16, 112, 112) person-crop video tubes."""

    def __init__(
        self,
        samples: List[TubeSample],
        mode: str = "train",
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.mode = mode
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply spatial & temporal augmentations during training."""
        if self.mode != "train":
            return clip

        # 1. Random horizontal flip (50% probability)
        if self._rng.random() > 0.5:
            clip = torch.flip(clip, dims=[3])

        # 2. Slight random temporal reverse (10% probability)
        if self._rng.random() > 0.90:
            clip = torch.flip(clip, dims=[1])

        return clip

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        tensor = self._apply_augmentations(sample.tensor.clone())
        return tensor, sample.label


def build_person_crop_splits(
    dataset_root: str,
    yolo_wrapper,
    cache_dir: Optional[str] = ".cache/person_crops",
    stride: int = 4,
    seed: int = 42,
    force_rebuild: bool = False,
) -> Tuple[PersonCropDataset, PersonCropDataset, PersonCropDataset]:
    """Create train, val, and test PersonCropDatasets with strict sequence isolation."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        train_cache = os.path.join(cache_dir, "train_tubes.pt")
        val_cache = os.path.join(cache_dir, "val_tubes.pt")
        test_cache = os.path.join(cache_dir, "test_tubes.pt")

        if not force_rebuild and os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache):
            logger.info("Loading cached person-crop tube datasets from %s...", cache_dir)
            train_samples = torch.load(train_cache, weights_only=False)
            val_samples = torch.load(val_cache, weights_only=False)
            test_samples = torch.load(test_cache, weights_only=False)
            return (
                PersonCropDataset(train_samples, mode="train", seed=seed),
                PersonCropDataset(val_samples, mode="val", seed=seed),
                PersonCropDataset(test_samples, mode="test", seed=seed),
            )

    # Use existing sequence-level splits (Seed=42)
    urfd_train, urfd_val, urfd_test = create_urfd_splits(dataset_root, seed=seed)
    yolo_model = yolo_wrapper._model if hasattr(yolo_wrapper, "_model") else yolo_wrapper

    def extract_split(urfd_split, name: str) -> List[TubeSample]:
        tubes = []
        logger.info("Extracting person-crop tubes for split: %s (%d sequences)...", name, len(urfd_split.samples))
        for sample in urfd_split.samples:
            s_tubes = extract_tubes_from_video(
                video_path=sample.path,
                sequence_id=sample.sequence_id,
                sequence_label=sample.label,
                yolo_model=yolo_model,
                stride=stride,
            )
            tubes.extend(s_tubes)
        logger.info("Split %s yielded %d tubes.", name, len(tubes))
        return tubes

    train_samples = extract_split(urfd_train, "TRAIN")
    val_samples = extract_split(urfd_val, "VAL")
    test_samples = extract_split(urfd_test, "TEST")

    if cache_dir:
        torch.save(train_samples, train_cache)
        torch.save(val_samples, val_cache)
        torch.save(test_samples, test_cache)
        logger.info("Saved extracted tubes to cache directory: %s", cache_dir)

    return (
        PersonCropDataset(train_samples, mode="train", seed=seed),
        PersonCropDataset(val_samples, mode="val", seed=seed),
        PersonCropDataset(test_samples, mode="test", seed=seed),
    )
