"""R3D-18 Action Recognition Model Wrapper.

Provides production inference and preprocessing for 3D CNN video action recognition (R3D-18)
fine-tuned on binary Emergency Action Recognition (NORMAL vs. FALL).
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("emergency_vision.worker.action_model")

# Kinetics-400 video normalization constants used during training
VIDEO_MEAN = [0.43216, 0.394666, 0.37645]
VIDEO_STD = [0.22803, 0.22145, 0.216989]

LABEL_NORMAL = 0
LABEL_FALL = 1
CLASS_NAMES = {LABEL_NORMAL: "NORMAL", LABEL_FALL: "FALL"}


@dataclass
class ActionPrediction:
    """Structured output for action recognition inference."""

    action: str
    confidence: float
    fall_probability: float
    normal_probability: float
    timestamp: datetime
    raw_logits: Optional[List[float]] = None


def preprocess_clip_frames(
    frames: List[np.ndarray],
    spatial_size: Tuple[int, int] = (112, 112),
) -> torch.Tensor:
    """Preprocess exactly 16 video frames into a normalized (1, 3, 16, 112, 112) tensor.

    Args:
        frames: Sequence of 16 OpenCV BGR or RGB images (numpy ndarrays).
        spatial_size: Target (width, height) spatial resolution (default 112x112).

    Returns:
        torch.Tensor of shape (1, 3, 16, 112, 112) and dtype torch.float32.
    """
    import cv2

    if len(frames) != 16:
        raise ValueError(f"Action recognition requires exactly 16 frames, got {len(frames)}")

    processed_frames = []
    mean_np = np.array(VIDEO_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(VIDEO_STD, dtype=np.float32).reshape(1, 1, 3)

    for frame in frames:
        if frame is None or frame.size == 0:
            raise ValueError("Encountered empty or None frame in clip buffer")

        # Convert BGR to RGB if 3 channels
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif len(frame.shape) == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = frame

        # Resize to (112, 112)
        resized = cv2.resize(rgb, spatial_size, interpolation=cv2.INTER_LINEAR)

        # Scale to [0.0, 1.0] and normalize
        norm = (resized.astype(np.float32) / 255.0 - mean_np) / std_np
        processed_frames.append(norm)

    # Stack into (16, 112, 112, 3)
    clip_np = np.stack(processed_frames, axis=0)
    # Permute to (3, 16, 112, 112)
    clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float()
    return clip_tensor


class ActionRecognitionWrapper:
    """Wrapper for 3D CNN video action recognition (R3D-18)."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        num_classes: int = 2,
        pretrained: bool = False,
    ) -> None:
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = self._resolve_device(device)
        self._model: Optional[nn.Module] = None

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve requested device with safe hardware fallbacks."""
        dev = (requested_device or "cpu").lower()
        if dev == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for action model but not available; falling back to CPU.")
            return "cpu"
        elif dev == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested for action model but not available; falling back to CPU.")
            return "cpu"
        return dev

    def _ensure_loaded(self) -> None:
        """Lazy load and initialize the R3D-18 model architecture and weights."""
        if self._model is None:
            from torchvision.models.video import r3d_18, R3D_18_Weights

            if self.weights_path:
                logger.info("Loading R3D-18 action recognition checkpoint from %s on %s...", self.weights_path, self.device)
                model = r3d_18()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                loaded = torch.load(self.weights_path, map_location=self.device, weights_only=False)
                state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
                model.load_state_dict(state_dict)
            elif self.pretrained:
                logger.info("Initializing R3D-18 with official torchvision Kinetics-400 pretrained weights...")
                try:
                    import ssl
                    try:
                        import certifi
                        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
                    except ImportError:
                        ssl._create_default_https_context = ssl._create_unverified_context
                    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
                except Exception as exc:
                    logger.warning("Could not download default online weights (%s), initializing clean R3D-18.", exc)
                    model = r3d_18()

                if self.num_classes != 400:
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            else:
                logger.info("Initializing uninitialized R3D-18 backbone (num_classes=%d)...", self.num_classes)
                model = r3d_18()
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)

            self._model = model.to(self.device)
            self._model.eval()
            logger.info("R3D-18 action recognition model ready on device: %s", self.device)

    def predict_tensor(self, clip_tensor: torch.Tensor) -> ActionPrediction:
        """Run classification on a (1, 3, 16, 112, 112) tensor representing a video clip.

        Returns:
            ActionPrediction containing action name ("NORMAL" or "FALL"), confidence, and probabilities.
        """
        self._ensure_loaded()
        ts = datetime.now(timezone.utc)

        with torch.no_grad():
            tensor = clip_tensor.to(self.device)
            outputs = self._model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            p_normal = float(probs[LABEL_NORMAL]) if len(probs) > LABEL_NORMAL else 0.0
            p_fall = float(probs[LABEL_FALL]) if len(probs) > LABEL_FALL else 0.0
            action_name = CLASS_NAMES.get(pred_class, f"CLASS_{pred_class}")

        return ActionPrediction(
            action=action_name,
            confidence=confidence,
            fall_probability=p_fall,
            normal_probability=p_normal,
            timestamp=ts,
            raw_logits=outputs.cpu().numpy()[0].tolist(),
        )

    def predict_clip(self, clip: Union[torch.Tensor, List[np.ndarray]]) -> ActionPrediction:
        """Run classification on either a preprocessed Tensor or a list of 16 raw frames."""
        if isinstance(clip, list):
            clip_tensor = preprocess_clip_frames(clip)
        else:
            clip_tensor = clip
        return self.predict_tensor(clip_tensor)
