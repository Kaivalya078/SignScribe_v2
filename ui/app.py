"""
ui/app.py — Tkinter GUI for SignScribe ASL recognition.

This file contains ONLY UI logic.  All backend functionality is
imported from the ``core`` package:

  - core.vision      — hand detection + normalization
  - core.inference    — model loading + prediction
  - core.smoother     — confidence-based stability
  - core.tts          — offline text-to-speech

Phase 5: Heavy work (camera read, MediaPipe detection, model inference)
runs on a background worker thread.  The main thread only polls results
and updates the Tkinter UI — keeping the interface responsive.

Run from the project root:  python -m ui.app
"""

import logging
import threading
import tkinter as tk
from queue import Queue, Empty, Full
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

from core.vision import HandDetector, normalize_landmarks
from core.inference import SignLanguageModel
from core.smoother import ConfidenceSmoother
from core.tts import speak as tts_speak, shutdown as tts_shutdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data bundle passed from worker → main thread
# -----------------------------------------------------------------------

class FrameResult:
    """Immutable result bundle produced by the worker thread."""
    __slots__ = ("frame", "prediction", "confidence", "has_hand", "worker_fps")

    def __init__(self, frame: np.ndarray, prediction, confidence: float,
                 has_hand: bool, worker_fps: float = 0.0):
        self.frame = frame
        self.prediction = prediction
        self.confidence = confidence
        self.has_hand = has_hand
        self.worker_fps = worker_fps


# -----------------------------------------------------------------------
# Background worker — runs camera + detection + inference
# -----------------------------------------------------------------------

class _CameraWorker(threading.Thread):
    """Daemon thread that reads frames, runs detection + inference,
    and pushes ``FrameResult`` objects into a queue.

    The main thread never touches OpenCV or MediaPipe directly.
    """

    def __init__(
        self,
        camera_index: int,
        model: SignLanguageModel,
        result_queue: Queue,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True, name="CameraWorker")
        self._cam_idx = camera_index
        self._model = model
        self._queue = result_queue
        self._stop = stop_event

    def run(self) -> None:
        cap = cv2.VideoCapture(self._cam_idx)
        if not cap.isOpened():
            logger.error("Cannot open webcam %d", self._cam_idx)
            return

        detector = HandDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        # Worker-side FPS tracking
        prev_time = time.time()
        worker_fps = 0.0

        try:
            while not self._stop.is_set():
                ret, frame = cap.read()
                # Fix #1: handle read failures and None frames
                if not ret or frame is None:
                    time.sleep(0.001)
                    continue

                frame = cv2.flip(frame, 1)

                # --- Detection ---
                results = detector.detect(frame)

                prediction = None
                confidence = 0.0
                has_hand = False

                if results.multi_hand_landmarks:
                    has_hand = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    detector.draw_landmarks(frame, hand_landmarks)

                    # --- Normalize + Infer ---
                    features = normalize_landmarks(hand_landmarks.landmark)
                    prediction, confidence = self._model.predict(features)

                # Fix #9: worker FPS
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    worker_fps = 0.9 * worker_fps + 0.1 * (1.0 / dt)

                # Fix #8: copy frame so main thread has its own buffer
                result = FrameResult(
                    frame.copy(), prediction, confidence, has_hand, worker_fps
                )

                # Fix #2: fully non-blocking put
                try:
                    self._queue.put_nowait(result)
                except Full:
                    try:
                        self._queue.get_nowait()  # drop oldest
                    except Empty:
                        pass
                    self._queue.put_nowait(result)

                # Fix #10: prevent CPU spin on fast cameras
                time.sleep(0.001)

        except Exception as e:
            # Fix #11: log worker exceptions instead of crashing silently
            logger.error("CameraWorker error: %s", e, exc_info=True)
        finally:
            detector.close()
            cap.release()
            logger.info("CameraWorker stopped")


# -----------------------------------------------------------------------
# Main application — UI only
# -----------------------------------------------------------------------

class ASLSentenceBuilder:
    """Main application window — UI only."""

    # Max frames buffered between worker and main thread
    _QUEUE_SIZE = 2

    def __init__(self, root):
        self.root = root
        self.root.title("SignScribe - ASL Alphabet Recognition")
        self.root.geometry("1280x720")

        # --- UI Theme and Styling ---
        self.style = ttk.Style(self.root)
        self.style.theme_use('arc')

        # --- Colors ---
        self.COLOR_BG = '#F0F2F5'
        self.COLOR_FRAME = '#FFFFFF'
        self.COLOR_TEXT = '#2E353B'
        self.COLOR_PRIMARY = '#007ACC'
        self.COLOR_SUCCESS = '#28A745'
        self.COLOR_WARN = '#FFC107'
        self.COLOR_DANGER = '#DC3545'
        self.COLOR_DISABLED = '#B0B0B0'

        # --- Fonts ---
        self.FONT_BOLD = ("Segoe UI", 12, "bold")
        self.FONT_NORMAL = ("Segoe UI", 11)
        self.FONT_LARGE = ("Segoe UI", 48, "bold")
        self.FONT_TITLE = ("Segoe UI", 16, "bold")

        self.root.configure(bg=self.COLOR_BG)

        self.CAMERA_WIDTH, self.CAMERA_HEIGHT = 640, 480
        self.camera_running = False
        self.sentence = ""

        # FPS tracking (EMA with α = 0.9)
        self._fps_prev_time = time.time()
        self._fps_value = 0.0
        self._fps_alpha = 0.9

        # Overlay state
        self._overlay_letter = ""
        self._overlay_confidence = 0.0

        # Predefined overlay drawing constants
        self._ovl_font = cv2.FONT_HERSHEY_SIMPLEX
        self._ovl_line = cv2.LINE_AA

        # --- Threading state ---
        self._worker: _CameraWorker | None = None
        self._stop_event = threading.Event()
        self._result_queue: Queue[FrameResult] = Queue(maxsize=self._QUEUE_SIZE)

        # --- Backend modules ---
        self._load_model()
        self.smoother = ConfidenceSmoother(
            confidence_threshold=0.80,
            stable_frames=3,
            cooldown_seconds=1.5,
        )

        self._create_widgets()
        self._poll_results()

    # ------------------------------------------------------------------
    # Model loading (delegates to core.inference)
    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            self.model = SignLanguageModel()  # uses DEFAULT_MODEL_PATH
            print("✅ Landmark model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            self.root.destroy()

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, style='TFrame', padding=20)
        main_frame.pack(fill='both', expand=True)

        # --- Left Panel: Camera ---
        left_panel = ttk.Frame(main_frame, style='Card.TFrame', padding=20)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))

        ttk.Label(left_panel, text="LIVE FEED", font=self.FONT_TITLE,
                  foreground=self.COLOR_TEXT).pack(pady=(0, 15))

        self.camera_label = tk.Label(left_panel, bg='#000000',
                                     text="Camera Off", font=self.FONT_BOLD, fg='white')
        self.camera_label.pack(fill='both', expand=True, pady=5)

        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(pady=(15, 0), fill='x')

        self.start_btn = ttk.Button(controls_frame, text="▶ Start Camera",
                                     command=self.start_camera, style='Success.TButton')
        self.start_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))

        self.stop_btn = ttk.Button(controls_frame, text="⏹ Stop Camera",
                                    command=self.stop_camera, style='Danger.TButton',
                                    state='disabled')
        self.stop_btn.pack(side='left', expand=True, fill='x', padx=5)

        # --- Right Panel: Controls & Output ---
        right_panel = ttk.Frame(main_frame, style='Card.TFrame', padding=20)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))

        ttk.Label(right_panel, text="RECOGNITION", font=self.FONT_TITLE,
                  foreground=self.COLOR_TEXT).pack(pady=(0, 10))

        self.prediction_label = ttk.Label(right_panel, text="...",
                                           font=self.FONT_LARGE,
                                           foreground=self.COLOR_PRIMARY,
                                           anchor='center')
        self.prediction_label.pack(pady=20, fill='x')

        self.status_label = ttk.Label(right_panel, text="Status: Idle",
                                       font=self.FONT_NORMAL,
                                       foreground=self.COLOR_DISABLED,
                                       anchor='center')
        self.status_label.pack(pady=(0, 20), fill='x')

        ttk.Separator(right_panel, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(right_panel, text="SENTENCE", font=self.FONT_TITLE,
                  foreground=self.COLOR_TEXT).pack(pady=10)

        sentence_frame = ttk.Frame(right_panel, style='Card.TFrame', padding=5)
        sentence_frame.pack(pady=5, fill='both', expand=True)

        self.sentence_text = tk.Text(sentence_frame, font=("Segoe UI", 14),
                                      wrap='word', bd=0, bg=self.COLOR_FRAME,
                                      fg=self.COLOR_TEXT, relief='flat',
                                      padx=10, pady=10)
        self.sentence_text.pack(fill='both', expand=True)

        sentence_controls = ttk.Frame(right_panel)
        sentence_controls.pack(pady=(15, 0), fill='x')

        ttk.Button(sentence_controls, text="Clear",
                   command=self.clear_sentence,
                   style='Warning.TButton').pack(side='left', expand=True,
                                                  fill='x', padx=(0, 5))
        ttk.Button(sentence_controls, text="Backspace",
                   command=self.backspace,
                   style='Secondary.TButton').pack(side='left', expand=True,
                                                    fill='x', padx=5)

    # ------------------------------------------------------------------
    # Camera start / stop
    # ------------------------------------------------------------------

    def start_camera(self):
        # Fix #6: prevent multiple workers
        if self._worker and self._worker.is_alive():
            logger.warning("Worker already running — ignoring start")
            return

        try:
            self._stop_event.clear()
            # Drain any stale results from a previous session
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except Empty:
                    break

            self._worker = _CameraWorker(
                camera_index=0,
                model=self.model,
                result_queue=self._result_queue,
                stop_event=self._stop_event,
            )
            self._worker.start()

            self.camera_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.smoother.reset()
            self._fps_prev_time = time.time()
            logger.info("Camera started (worker thread: %s)", self._worker.name)
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")

    def stop_camera(self):
        self.camera_running = False

        # Signal the worker to stop and wait for it
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
            # Fix #7: warn on zombie thread
            if self._worker.is_alive():
                logger.warning("Worker thread did not terminate cleanly")
        self._worker = None

        self.smoother.reset()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.camera_label.config(image='', text="Camera Off", bg='black')
        self.camera_label.image = None
        self.prediction_label.config(text="...")
        self.status_label.config(text="Status: Idle", foreground=self.COLOR_DISABLED)
        logger.info("Camera stopped")

    # ------------------------------------------------------------------
    # HUD overlay (burned into the webcam frame)
    # ------------------------------------------------------------------

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render predicted letter, confidence bar, and FPS onto *frame* (BGR)."""
        h, w = frame.shape[:2]
        font = self._ovl_font
        line = self._ovl_line

        # FPS counter (top-right)
        fps_text = f"FPS: {self._fps_value:.0f}"
        cv2.putText(frame, fps_text, (w - 160, 35),
                    font, 0.8, (0, 255, 200), 2, line)

        if not self._overlay_letter:
            return frame

        letter = self._overlay_letter.upper()
        conf = self._overlay_confidence

        # Colour by confidence
        if conf >= 0.85:
            colour = (0, 220, 80)       # green
        elif conf >= 0.60:
            colour = (0, 200, 255)      # amber
        else:
            colour = (0, 80, 255)       # red

        # Predicted letter
        cv2.putText(frame, letter, (20, 55), font, 1.8, colour, 4, line)

        # Confidence %
        conf_text = f"{conf * 100:.0f}%"
        cv2.putText(frame, conf_text, (20, 90), font, 0.8, colour, 2, line)

        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = 10, 100, 200, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), 1)
        fill_w = int(bar_w * min(conf, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      colour, -1)

        return frame

    # ------------------------------------------------------------------
    # FPS helper
    # ------------------------------------------------------------------

    def _update_fps(self):
        """Compute an exponentially-smoothed FPS value (α = 0.9)."""
        now = time.time()
        dt = now - self._fps_prev_time
        self._fps_prev_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_value = (self._fps_alpha * self._fps_value
                               + (1.0 - self._fps_alpha) * instant_fps)

    # ------------------------------------------------------------------
    # Main UI poll loop (runs on the Tkinter main thread)
    # ------------------------------------------------------------------

    def _poll_results(self):
        """Consume the latest FrameResult from the worker and update the UI.

        This method runs on the main thread via ``root.after()`` and is
        the ONLY place that touches Tkinter widgets.
        """
        if self.camera_running:
            result: FrameResult | None = None

            # Drain the queue — keep only the most recent result
            while True:
                try:
                    result = self._result_queue.get_nowait()
                except Empty:
                    break

            if result is not None:
                # --- Update overlay state ---
                if result.has_hand and result.prediction:
                    self._overlay_letter = result.prediction
                    self._overlay_confidence = result.confidence
                    self.prediction_label.config(
                        text=result.prediction.upper())
                else:
                    self._overlay_letter = ""
                    self._overlay_confidence = 0.0
                    self.prediction_label.config(text="...")

                # --- Status label ---
                is_cooldown, remaining = self.smoother.get_cooldown_status()
                if is_cooldown:
                    self.status_label.config(
                        text=f"⌛ Cooldown ({remaining:.1f}s)",
                        foreground=self.COLOR_DANGER)
                elif not result.has_hand:
                    self.status_label.config(
                        text="🔍 Searching for hand...",
                        foreground=self.COLOR_WARN)
                elif result.confidence < self.smoother.confidence_threshold:
                    self.status_label.config(
                        text="⚠ Low confidence",
                        foreground=self.COLOR_WARN)
                else:
                    self.status_label.config(
                        text="🔎 Detecting...",
                        foreground=self.COLOR_PRIMARY)

                # --- Smoother (runs on main thread — lightweight) ---
                stable = self.smoother.add_prediction(
                    result.prediction, result.confidence)
                if stable:
                    self._add_to_sentence(stable)
                    self.status_label.config(
                        text=f"✅ Stable: {stable.upper()}",
                        foreground=self.COLOR_SUCCESS)

                # --- Overlay + display ---
                self._update_fps()
                frame = self._draw_overlay(result.frame)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = img_tk
                self.camera_label.config(image=img_tk)

        # Poll at ~60 Hz (16 ms) — display rate, not processing rate
        self.root.after(16, self._poll_results)

    # ------------------------------------------------------------------
    # Sentence management
    # ------------------------------------------------------------------

    def _add_to_sentence(self, p):
        if p == "space":
            self.sentence += " "
            words = self.sentence.strip().split()
            if words:
                tts_speak(words[-1])
        elif p == "del":
            self.backspace()
        else:
            self.sentence += p
        self._update_sentence_display()

    def _update_sentence_display(self):
        self.sentence_text.delete(1.0, 'end')
        self.sentence_text.insert(1.0, self.sentence)
        self.sentence_text.see('end')

    def clear_sentence(self):
        self.sentence = ""
        self._update_sentence_display()

    def backspace(self):
        if self.sentence:
            self.sentence = self.sentence[:-1]
            self._update_sentence_display()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def on_closing(self):
        self.stop_camera()
        tts_shutdown()
        self.root.destroy()


def main():
    root = ThemedTk(theme="arc")
    app = ASLSentenceBuilder(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
