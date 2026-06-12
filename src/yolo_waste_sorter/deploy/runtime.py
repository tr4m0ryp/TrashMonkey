"""Edge inference runtime: round-robin streams -> votes -> consensus.

One loop round-robins the grab-latest camera readers, runs the detector on
each fresh frame at the permissive ``conf_floor``, and accumulates per-stream
object sessions: the runtime assumes a single object in view per stream at a
time, so frame-to-object association needs no tracker -- a session opens on
the first qualified-floor detection and CLOSES ``window_seconds`` after it
started. On close the votes go through the SHARED ``consensus_decision``
(imported from the thresholding package, never reimplemented), and the
decision is emitted as one JSON line on stdout AND through the pluggable
``emit`` callback -- the integration seam for downstream consumers.

The emitted ``decision`` field is a trained class name or the literal string
``"rest"`` -- the REST sentinel's wire form. "rest" is NOT a trained class;
it exists only as the consensus rejection outcome.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from yolo_waste_sorter.deploy.artifacts import load_threshold_params
from yolo_waste_sorter.deploy.streams import CameraReader, Frame, start_readers
from yolo_waste_sorter.models.thresholding import THRESHOLDS_FILENAME
from yolo_waste_sorter.models.thresholds import (
    REST,
    ThresholdParams,
    Vote,
    consensus_decision,
)
from yolo_waste_sorter.utils.config import Config

REST_LABEL = "rest"  # wire form of the REST sentinel in emitted decisions

# Detector seam: frame -> detections as (class_id, score). The model-backed
# predictor and the test fake both satisfy this.
PredictFn = Callable[[Frame], list[Vote]]


class DeployError(Exception):
    """Deployment runtime wiring or artifact error."""


@dataclass(frozen=True)
class DecisionEvent:
    """One per-object decision handed to downstream consumers."""

    camera: str
    decision: str  # trained class name, or REST_LABEL
    confidence: float | None  # max winning-class score; None for rest
    frames: int  # qualified-floor votes in the session
    timestamp: float  # wall-clock emit time (time.time())

    def to_json(self) -> str:
        return json.dumps(
            {
                "camera": self.camera,
                "decision": self.decision,
                "confidence": self.confidence,
                "frames": self.frames,
                "timestamp": self.timestamp,
            }
        )


EmitFn = Callable[[DecisionEvent], None]


@dataclass
class _Session:
    """Votes for the single object currently in one stream's view."""

    started: float  # monotonic open time
    votes: list[Vote] = field(default_factory=list)


def load_engine_predictor(engine: Path, *, conf_floor: float) -> PredictFn:
    """TensorRT-engine predictor via lazy ultralytics import (Jetson only)."""
    if not engine.is_file():
        raise DeployError(f"engine artifact not found: {engine} (export on the Jetson, R5)")
    from ultralytics import YOLO

    model = YOLO(str(engine), task="detect")

    def predict(frame: Frame) -> list[Vote]:
        result = model.predict(frame, conf=conf_floor, verbose=False)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [
            (int(c), float(s))
            for c, s in zip(boxes.cls.tolist(), boxes.conf.tolist(), strict=True)
        ]

    return predict


class Runtime:
    """Round-robin inference loop over grab-latest readers (T8)."""

    def __init__(
        self,
        *,
        classes: tuple[str, ...],
        params: ThresholdParams,
        predict: PredictFn,
        readers: Sequence[CameraReader],
        window_seconds: float,
        emit: EmitFn | None = None,
        out: TextIO | None = None,
        poll_interval_s: float = 0.005,
    ) -> None:
        if not readers:
            raise DeployError("runtime needs at least one camera reader")
        if window_seconds <= 0:
            raise DeployError(f"window_seconds must be > 0, got {window_seconds}")
        self.classes = classes
        self.params = params
        self.predict = predict
        self.readers = list(readers)
        self.window_seconds = window_seconds
        self.emit = emit
        self.out = sys.stdout if out is None else out
        self.poll_interval_s = poll_interval_s
        self._sessions: dict[str, _Session] = {}
        self._last_frame_ts: dict[str, float] = {}

    # -- decision plumbing ---------------------------------------------------

    def _decision_label(self, votes: list[Vote]) -> tuple[str, float | None]:
        decision = consensus_decision(votes, self.params)
        if decision is REST:
            return REST_LABEL, None
        assert isinstance(decision, int)
        if not 0 <= decision < len(self.classes):
            raise DeployError(
                f"consensus produced class id {decision} outside the "
                f"{len(self.classes)}-class map -- engine/config mismatch"
            )
        confidence = max(score for class_id, score in votes if class_id == decision)
        return self.classes[decision], confidence

    def _close_session(self, camera: str, session: _Session) -> DecisionEvent:
        label, confidence = self._decision_label(session.votes)
        event = DecisionEvent(
            camera=camera,
            decision=label,
            confidence=confidence,
            frames=len(session.votes),
            timestamp=time.time(),
        )
        self.out.write(event.to_json() + "\n")
        self.out.flush()
        if self.emit is not None:
            self.emit(event)
        return event

    # -- loop ------------------------------------------------------------------

    def step(self, now: float | None = None) -> list[DecisionEvent]:
        """One round-robin pass; returns the decisions closed in this pass."""
        current = time.monotonic() if now is None else now
        events: list[DecisionEvent] = []
        for reader in self.readers:
            camera = reader.name
            session = self._sessions.get(camera)
            if session is not None and current - session.started >= self.window_seconds:
                events.append(self._close_session(camera, session))
                del self._sessions[camera]
                session = None
            snapshot = reader.latest()
            if snapshot is None or reader.is_stale(current):
                continue
            frame, frame_ts = snapshot
            if self._last_frame_ts.get(camera) == frame_ts:
                continue  # grab-latest gave us the same frame again
            self._last_frame_ts[camera] = frame_ts
            detections = [d for d in self.predict(frame) if d[1] >= self.params.conf_floor]
            if not detections:
                continue
            top = max(detections, key=lambda d: d[1])
            if session is None:
                session = _Session(started=current)
                self._sessions[camera] = session
            session.votes.append(top)
        return events

    def run(self, max_cycles: int | None = None) -> int:
        """Loop until ``max_cycles`` (None = forever); returns decisions emitted."""
        emitted = 0
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            emitted += len(self.step())
            cycles += 1
            if self.poll_interval_s > 0:
                time.sleep(self.poll_interval_s)
        return emitted

    def stop(self) -> None:
        """Stop every reader thread; open sessions are dropped, not force-closed."""
        for reader in self.readers:
            reader.stop()


def build_runtime(
    cfg: Config,
    *,
    thresholds_path: Path | None = None,
    predict: PredictFn | None = None,
    emit: EmitFn | None = None,
) -> Runtime:
    """Wire the runtime from config: thresholds artifact, readers, predictor.

    ``predict`` defaults to the TensorRT engine at ``cfg.deploy.engine`` (a
    Jetson-only path); tests and off-Jetson runs inject a callable instead.
    """
    artifact = (
        cfg.deploy.engine.parent / THRESHOLDS_FILENAME if thresholds_path is None
        else thresholds_path
    )
    params = load_threshold_params(artifact)
    if predict is None:
        predict = load_engine_predictor(cfg.deploy.engine, conf_floor=params.conf_floor)
    readers = start_readers(
        cfg.deploy.cameras,
        stale_after_s=cfg.deploy.stale_after_s,
        reconnect_backoff_s=cfg.deploy.reconnect_backoff_s,
    )
    return Runtime(
        classes=cfg.classes,
        params=params,
        predict=predict,
        readers=readers,
        window_seconds=cfg.deploy.window_seconds,
        emit=emit,
    )
