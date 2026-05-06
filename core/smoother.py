"""
core/smoother.py — Confidence-based prediction stability filter.

Acceptance criteria (ALL must be true simultaneously):
  1. Prediction is not None / "nothing"
  2. Model confidence ≥ confidence_threshold   (default 0.80)
  3. Cooldown period elapsed since last accept   (default 1.5 s)
     — cooldown also blocks streak accumulation
  4. Same letter predicted for ≥ stable_frames   (default 3 consecutive)
  5. Prediction differs from last accepted letter (duplicate guard)
  6. Streak completed within max_streak_age       (debounce)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ConfidenceSmoother:
    """Confidence + consecutive-frame stability filter with full edge-case handling."""

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        stable_frames: int = 3,
        cooldown_seconds: float = 1.5,
        max_streak_age: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.stable_frames = stable_frames
        self.cooldown = cooldown_seconds
        self.max_streak_age = max_streak_age

        # Internal state
        self._streak_label: Optional[str] = None
        self._streak_count: int = 0
        self._streak_start_time: float = 0.0
        self._last_accepted_time: float = 0.0
        self._last_accepted_label: Optional[str] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def in_cooldown(self) -> bool:
        """Monotonic cooldown check."""
        return (time.time() - self._last_accepted_time) < self.cooldown

    def add_prediction(
        self, prediction: Optional[str], confidence: float = 0.0
    ) -> Optional[str]:
        """Feed one frame's prediction + confidence.

        Returns the prediction string when all acceptance criteria are
        satisfied, otherwise ``None``.
        """
        now = time.time()

        # ---- gate 1: nothing / no hand → full streak reset ----
        if prediction is None or prediction == "nothing":
            self._reset_streak()
            return None

        # ---- gate 2: confidence too low → full streak reset ----
        if confidence < self.confidence_threshold:
            self._reset_streak()
            return None

        # ---- gate 3: cooldown blocks BOTH streak AND output ----
        if self.in_cooldown():
            self._reset_streak()
            return None

        # ---- gate 4: build or break the streak ----
        if prediction != self._streak_label:
            self._streak_label = prediction
            self._streak_count = 1
            self._streak_start_time = now
        else:
            self._streak_count += 1

        # ---- gate 5: debounce — streak must complete within max_streak_age ----
        if self._streak_count > 1 and (now - self._streak_start_time) > self.max_streak_age:
            self._reset_streak()
            return None

        # ---- gate 6: enough consecutive frames? ----
        if self._streak_count < self.stable_frames:
            return None

        # ---- gate 7: duplicate guard — same letter as last accepted ----
        if prediction == self._last_accepted_label:
            return None

        # ---- accepted! ----
        self._last_accepted_time = now
        self._last_accepted_label = prediction
        self._reset_streak()
        logger.info("Accepted: %s (conf=%.2f, streak=%d)", prediction, confidence, self.stable_frames)
        return prediction

    def get_cooldown_status(self) -> tuple[bool, float]:
        """Return ``(in_cooldown, remaining_seconds)``."""
        elapsed = time.time() - self._last_accepted_time
        is_cd = elapsed < self.cooldown
        remain = self.cooldown - elapsed if is_cd else 0.0
        return is_cd, remain

    def reset(self) -> None:
        """Clear all internal state — call on camera stop/start."""
        self._streak_label = None
        self._streak_count = 0
        self._streak_start_time = 0.0
        self._last_accepted_time = 0.0
        self._last_accepted_label = None

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _reset_streak(self) -> None:
        self._streak_label = None
        self._streak_count = 0
        self._streak_start_time = 0.0
