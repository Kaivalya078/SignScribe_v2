"""
core/inference.py — Model loading and prediction.

Provides the ``SignLanguageModel`` class which wraps either:
  - A pickled sklearn classifier (.pkl)   — Random Forest
  - A PyTorch checkpoint (.pth)           — MLP (ASLNet)

Both expose the same ``predict(feature_vector) → (label, confidence)`` API.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical project paths — single source of truth
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "asl_mlp_model.pth"


# ---------------------------------------------------------------------------
# MLP Architecture (must match models/train_mlp.py)
# ---------------------------------------------------------------------------

class ASLNet(nn.Module):
    """3-layer MLP for ASL landmark classification.

    Architecture:  input → 128 (BN+ReLU+Drop) → 64 (BN+ReLU+Drop) → output
    """

    def __init__(
        self,
        input_size: int = 63,
        hidden_sizes: list = None,
        num_classes: int = 29,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_sizes[1], num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Unified model wrapper
# ---------------------------------------------------------------------------

class SignLanguageModel:
    """Unified inference wrapper — supports both sklearn (.pkl) and PyTorch (.pth).

    The model is loaded **once** at construction time.  No disk I/O
    occurs during ``predict()`` calls.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_cap: float = 0.95,
    ):
        """Load the model from *model_path*.

        Parameters
        ----------
        model_path : str | Path | None
            Path to the model file.  Auto-detected by extension:
            - ``.pkl`` → sklearn (Random Forest)
            - ``.pth`` → PyTorch (ASLNet MLP)
            Defaults to ``<project_root>/checkpoints/asl_mlp_model.pth``.
        confidence_cap : float
            Upper bound clamp for reported confidence.
        """
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._confidence_cap = confidence_cap
        self._backend = "unknown"

        if path.suffix == ".pkl":
            self._load_sklearn(path)
        elif path.suffix == ".pth":
            self._load_pytorch(path)
        else:
            raise ValueError(f"Unsupported model format: {path.suffix}")

        logger.info("Model loaded from %s [backend=%s]", path, self._backend)

    # -- sklearn loader ---------------------------------------------------

    def _load_sklearn(self, path: Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self._backend = "sklearn"

    # -- pytorch loader ---------------------------------------------------

    def _load_pytorch(self, path: Path) -> None:
        self._device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        self._classes = checkpoint["classes"]
        input_size = checkpoint.get("input_size", 63)
        hidden_sizes = checkpoint.get("hidden_sizes", [128, 64])
        num_classes = checkpoint.get("num_classes", len(self._classes))

        self._torch_model = ASLNet(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=num_classes,
        )
        self._torch_model.load_state_dict(checkpoint["model_state_dict"])
        self._torch_model.to(self._device)
        self._torch_model.eval()
        self._backend = "pytorch"

    # -- unified predict --------------------------------------------------

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """Return ``(predicted_label, confidence)`` for a (1, 63) input.

        Works identically regardless of backend.
        """
        if self._backend == "sklearn":
            return self._predict_sklearn(feature_vector)
        elif self._backend == "pytorch":
            return self._predict_pytorch(feature_vector)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def _predict_sklearn(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(feature_vector)[0]
            best_idx = int(np.argmax(proba))
            label = str(self._model.classes_[best_idx])
            confidence = min(float(proba[best_idx]), self._confidence_cap)
        else:
            label = str(self._model.predict(feature_vector)[0])
            confidence = 1.0
        logger.debug("Prediction [sklearn]: %s (%.2f)", label, confidence)
        return label, confidence

    def _predict_pytorch(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        tensor = torch.from_numpy(
            feature_vector.astype(np.float32)
        ).to(self._device)
        # Fix #4: enforce (1, 63) shape
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            logits = self._torch_model(tensor)
            proba = torch.softmax(logits, dim=1)[0]
        best_idx = int(torch.argmax(proba))
        label = str(self._classes[best_idx])
        confidence = min(float(proba[best_idx].item()), self._confidence_cap)
        logger.debug("Prediction [pytorch]: %s (%.2f)", label, confidence)
        return label, float(confidence)
