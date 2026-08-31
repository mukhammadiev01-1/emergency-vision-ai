"""UR Fall Detection (URFD) Dataset and Video Clip Loader.

Provides spatiotemporal video clip sampling, deterministic train/val/test splits
with strict sequence isolation, and normalization compatible with torchvision R3D-18.
"""
from dataclasses import dataclass
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("emergency_vision.datasets.urfd")

# Binary Class Definitions
LABEL_NORMAL = 0
LABEL_FALL = 1

CLASS_NAMES: Dict[int, str] = {
    LABEL_NORMAL: "NORMAL",
    LABEL_FALL: "FALL",
}

# Standard Torchvision Video Normalization Statistics (Kinetics-400)
VIDEO_MEAN = [0.43216, 0.394666, 0.37645]
VIDEO_STD = [0.22803, 0.22145, 0.216989]


@dataclass
class URFDSample:
    """Metadata representing a single video sequence sample."""
    path: str
    sequence_id: str
    label: int
    class_name: str
    is_video_file: bool = True


def extract_video_frames(video_path: str, target_size: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    """Decode all RGB frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for decoding video frames.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR -> RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size:
                frame_rgb = cv2.resize(frame_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Video file produced 0 valid frames: {video_path}")
    return frames


def extract_directory_frames(dir_path: str, target_size: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    """Decode RGB frames from a directory of ordered PNG/JPEG image files."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required for reading image frames.")

    valid_exts = (".png", ".jpg", ".jpeg")
    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(valid_exts)])
    if not files:
        raise RuntimeError(f"No image frames found in directory: {dir_path}")

    frames = []
    for f in files:
        img_path = os.path.join(dir_path, f)
        bgr = cv2.imread(img_path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if target_size:
                rgb = cv2.resize(rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)

    if not frames:
        raise RuntimeError(f"Directory produced 0 valid frames: {dir_path}")
    return frames


class URFDDataset(Dataset):
    """Spatiotemporal PyTorch Dataset for UR Fall Detection (URFD)."""

    def __init__(
        self,
        samples: List[URFDSample],
        num_frames: int = 16,
        spatial_size: Tuple[int, int] = (112, 112),
        mode: str = "train",
        seed: Optional[int] = None,
    ) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.mode = mode
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

        # Mean and Std tensors: shape (C, 1, 1, 1) for broadcasting over (C, T, H, W)
        self.mean_tensor = torch.tensor(VIDEO_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self.std_tensor = torch.tensor(VIDEO_STD, dtype=torch.float32).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, total_frames: int) -> List[int]:
        """Select exactly num_frames indices using mode-specific temporal sampling."""
        if total_frames <= self.num_frames:
            # Replicate last frame if video is shorter than window
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(total_frames - 1)
            return indices

        if self.mode == "train":
            # Random uniform or jittered temporal sampling
            max_start = total_frames - self.num_frames
            start_idx = self._rng.randint(0, max_start)
            return list(range(start_idx, start_idx + self.num_frames))
        else:
            # Deterministic linear interpolation across the video duration
            return torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()

    def _apply_augmentations(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply data augmentations to clip of shape (C, T, H, W) in train mode."""
        if self.mode != "train":
            return clip

        # 1. Random horizontal flip (50% probability)
        if self._rng.random() > 0.5:
            clip = torch.flip(clip, dims=[3])

        # 2. Random brightness jitter
        brightness_factor = 1.0 + (self._rng.random() - 0.5) * 0.2
        clip = clip * brightness_factor
        clip = torch.clamp(clip, 0.0, 1.0)

        return clip

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        if sample.is_video_file:
            all_frames = extract_video_frames(sample.path, target_size=self.spatial_size)
        else:
            all_frames = extract_directory_frames(sample.path, target_size=self.spatial_size)

        indices = self._sample_indices(len(all_frames))
        selected_frames = [all_frames[i] for i in indices]

        # Stack into numpy array: (T, H, W, C)
        np_clip = np.stack(selected_frames, axis=0)

        # Convert to float tensor and permute to (C, T, H, W) in [0.0, 1.0]
        clip_tensor = torch.from_numpy(np_clip).permute(3, 0, 1, 2).float() / 255.0

        # Apply spatial/color augmentations (in train mode)
        clip_tensor = self._apply_augmentations(clip_tensor)

        # Apply Kinetics-400 video normalization: (x - mean) / std
        clip_tensor = (clip_tensor - self.mean_tensor) / self.std_tensor

        return clip_tensor, sample.label


class SyntheticURFDDataset(Dataset):
    """Synthetic dataset generating in-memory (C, T, H, W) video tensors for smoke testing."""

    def __init__(
        self,
        num_samples: int = 16,
        num_frames: int = 16,
        spatial_size: Tuple[int, int] = (112, 112),
        mode: str = "train",
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.mode = mode
        self.seed = seed
        self._rng = random.Random(seed)

        self.samples = []
        for i in range(num_samples):
            label = LABEL_FALL if (i % 2 == 1) else LABEL_NORMAL
            self.samples.append(
                URFDSample(
                    path=f"synthetic://sample_{i}",
                    sequence_id=f"syn_{i:03d}",
                    label=label,
                    class_name=CLASS_NAMES[label],
                    is_video_file=False,
                )
            )

        self.mean_tensor = torch.tensor(VIDEO_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self.std_tensor = torch.tensor(VIDEO_STD, dtype=torch.float32).view(3, 1, 1, 1)

    def _sample_indices(self, total_frames: int) -> List[int]:
        """Select exactly num_frames indices using mode-specific temporal sampling."""
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(total_frames - 1)
            return indices

        if self.mode == "train":
            max_start = total_frames - self.num_frames
            start_idx = self._rng.randint(0, max_start)
            return list(range(start_idx, start_idx + self.num_frames))
        else:
            return torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        C, T, H, W = 3, self.num_frames, self.spatial_size[0], self.spatial_size[1]

        # Generate synthetic spatiotemporal tensor with distinct kinematics
        base = torch.zeros(C, T, H, W, dtype=torch.float32)
        if sample.label == LABEL_FALL:
            # Fall: downward vertical motion trajectory
            for t in range(T):
                y_pos = int((t / max(1, T - 1)) * (H - 30))
                base[:, t, y_pos : y_pos + 25, 40:70] = 0.8
        else:
            # Normal: horizontal walking trajectory
            for t in range(T):
                x_pos = int((t / max(1, T - 1)) * (W - 30))
                base[:, t, 40:70, x_pos : x_pos + 25] = 0.5

        # Normalize
        normalized = (base - self.mean_tensor) / self.std_tensor
        return normalized, sample.label


def index_urfd_directory(dataset_root: str) -> Tuple[List[URFDSample], List[URFDSample]]:
    """Scan dataset_root and return lists of (fall_samples, normal_samples)."""
    fall_samples: List[URFDSample] = []
    normal_samples: List[URFDSample] = []

    # Check videos directory or flat layout
    candidates = [
        (os.path.join(dataset_root, "videos", "fall"), os.path.join(dataset_root, "videos", "normal")),
        (os.path.join(dataset_root, "fall"), os.path.join(dataset_root, "normal")),
        (os.path.join(dataset_root, "frames", "fall"), os.path.join(dataset_root, "frames", "normal")),
    ]

    fall_dir, normal_dir = None, None
    for f_dir, n_dir in candidates:
        if os.path.exists(f_dir) and os.path.exists(n_dir):
            fall_dir, normal_dir = f_dir, n_dir
            break

    if not fall_dir or not normal_dir:
        # Check if dataset_root itself contains files directly
        logger.warning("Could not find standard URFD fall/normal subdirectories in %s", dataset_root)
        return [], []

    # Process falls
    for item in sorted(os.listdir(fall_dir)):
        p = os.path.join(fall_dir, item)
        is_vid = os.path.isfile(p) and p.lower().endswith((".mp4", ".avi", ".mov"))
        is_dir = os.path.isdir(p)
        if is_vid or is_dir:
            seq_id = os.path.splitext(item)[0]
            fall_samples.append(
                URFDSample(
                    path=p,
                    sequence_id=seq_id,
                    label=LABEL_FALL,
                    class_name=CLASS_NAMES[LABEL_FALL],
                    is_video_file=is_vid,
                )
            )

    # Process normal ADLs
    for item in sorted(os.listdir(normal_dir)):
        p = os.path.join(normal_dir, item)
        is_vid = os.path.isfile(p) and p.lower().endswith((".mp4", ".avi", ".mov"))
        is_dir = os.path.isdir(p)
        if is_vid or is_dir:
            seq_id = os.path.splitext(item)[0]
            normal_samples.append(
                URFDSample(
                    path=p,
                    sequence_id=seq_id,
                    label=LABEL_NORMAL,
                    class_name=CLASS_NAMES[LABEL_NORMAL],
                    is_video_file=is_vid,
                )
            )

    return fall_samples, normal_samples


def create_urfd_splits(
    dataset_root: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_frames: int = 16,
    spatial_size: Tuple[int, int] = (112, 112),
) -> Tuple[URFDDataset, URFDDataset, URFDDataset]:
    """Create train, validation, and test datasets with strict sequence isolation."""
    fall_samples, normal_samples = index_urfd_directory(dataset_root)

    if not fall_samples and not normal_samples:
        raise FileNotFoundError(
            f"No URFD fall or normal sequences found in {dataset_root}. "
            "Please run scripts/download_urfd.py to acquire the dataset."
        )

    rng = random.Random(seed)
    rng.shuffle(fall_samples)
    rng.shuffle(normal_samples)

    def partition(samples: List[URFDSample]) -> Tuple[List[URFDSample], List[URFDSample], List[URFDSample]]:
        n = len(samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_s = samples[:n_train]
        val_s = samples[n_train : n_train + n_val]
        test_s = samples[n_train + n_val :]
        if not test_s and val_s:
            test_s = [val_s.pop()]
        return train_s, val_s, test_s

    f_train, f_val, f_test = partition(fall_samples)
    n_train, n_val, n_test = partition(normal_samples)

    train_samples = f_train + n_train
    val_samples = f_val + n_val
    test_samples = f_test + n_test

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    train_ds = URFDDataset(train_samples, num_frames=num_frames, spatial_size=spatial_size, mode="train", seed=seed)
    val_ds = URFDDataset(val_samples, num_frames=num_frames, spatial_size=spatial_size, mode="val", seed=seed)
    test_ds = URFDDataset(test_samples, num_frames=num_frames, spatial_size=spatial_size, mode="test", seed=seed)

    logger.info(
        "Created URFD splits (seed=%d): Train=%d (Fall=%d, ADL=%d), Val=%d, Test=%d",
        seed,
        len(train_ds),
        len(f_train),
        len(n_train),
        len(val_ds),
        len(test_ds),
    )
    return train_ds, val_ds, test_ds
