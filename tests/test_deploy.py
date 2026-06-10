"""Deploy package tests (T8): streams, runtime + REAL T9 consensus, artifacts,
export refusal, check_env probes. No network beyond 127.0.0.1, no ultralytics.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from deploy_helpers import CLASSES, MJPEGServer, StubReader, make_params, solid_jpeg, wait_until

from yolo_waste_sorter.deploy import (
    CameraReader,
    DecisionEvent,
    DeployError,
    ExportError,
    Runtime,
    StreamError,
    ensure_jetson_arch,
    export_engine,
    start_readers,
)
from yolo_waste_sorter.deploy.check_env import check_camera, check_l4t, check_nvpmodel, run_checks

# -- streams ----------------------------------------------------------------


def test_reader_keeps_newest_frame_with_monotonic_timestamps() -> None:
    frames = [solid_jpeg((255, 0, 0)), solid_jpeg((0, 255, 0))]
    with MJPEGServer(frames, fps=30.0) as server:
        reader = CameraReader(server.url, stale_after_s=2.0, reconnect_backoff_s=0.2)
        reader.start()
        try:
            assert wait_until(lambda: reader.latest() is not None, timeout_s=10.0)
            snap = reader.latest()
            assert snap is not None
            first_frame, first_ts = snap
            assert first_frame.dtype == np.uint8 and first_frame.ndim == 3
            # keep-newest: the snapshot is replaced, timestamps strictly advance
            assert wait_until(
                lambda: (s := reader.latest()) is not None and s[1] > first_ts, timeout_s=10.0
            )
        finally:
            reader.stop()
            reader.stop()  # idempotent


def test_reader_staleness_budget() -> None:
    with MJPEGServer([solid_jpeg((255, 0, 0))], fps=30.0) as server:
        reader = CameraReader(server.url, stale_after_s=0.5, reconnect_backoff_s=0.2)
        assert reader.is_stale()  # no frame yet
        reader.start()
        try:
            assert wait_until(lambda: reader.latest() is not None, timeout_s=10.0)
            snap = reader.latest()
            assert snap is not None
            ts = snap[1]
            assert not reader.is_stale(now=ts + 0.4)
            assert reader.is_stale(now=ts + 0.6)
        finally:
            reader.stop()


def test_reader_reconnects_after_stream_death() -> None:
    with MJPEGServer([solid_jpeg((255, 0, 0))], fps=30.0, max_frames_per_conn=5) as server:
        reader = CameraReader(server.url, stale_after_s=2.0, reconnect_backoff_s=0.1)
        reader.start()
        try:
            assert wait_until(lambda: reader.reconnects >= 2, timeout_s=15.0)
            assert server.connections >= 2  # it really came back over a NEW connection
            assert reader.latest() is not None
        finally:
            reader.stop()


def test_reader_and_start_readers_fail_fast() -> None:
    with pytest.raises(StreamError, match="non-empty"):
        CameraReader("", stale_after_s=1.0, reconnect_backoff_s=1.0)
    with pytest.raises(StreamError, match="must be > 0"):
        CameraReader("http://x/stream", stale_after_s=0.0, reconnect_backoff_s=1.0)
    with pytest.raises(StreamError, match="at least one stream"):
        start_readers((), stale_after_s=1.0, reconnect_backoff_s=1.0)
    reader = CameraReader("http://x/stream", stale_after_s=1.0, reconnect_backoff_s=1.0)
    reader.start()
    try:
        with pytest.raises(StreamError, match="already started"):
            reader.start()
    finally:
        reader.stop()


# -- runtime (deterministic, StubReader-driven) ------------------------------


def _runtime(predict, readers, **kwargs):  # type: ignore[no-untyped-def]
    defaults = dict(
        classes=CLASSES, params=make_params(), window_seconds=0.4, poll_interval_s=0.0
    )
    defaults.update(kwargs)
    return Runtime(predict=predict, readers=readers, **defaults)


def test_runtime_sessions_emit_bin_then_rest_via_real_consensus() -> None:
    reader = StubReader("cam0")
    scripted = iter(
        [[(0, 0.9)], [(0, 0.85)], [(0, 0.7)], [(1, 0.30)], [(1, 0.32)], [(1, 0.31)]]
    )
    out = io.StringIO()
    emitted: list[DecisionEvent] = []
    rt = _runtime(lambda frame: next(scripted), [reader], emit=emitted.append, out=out)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    for ts in (1.0, 1.1, 1.2):  # session 1: three qualified plastic votes
        reader.set_frame(frame, ts)
        assert rt.step(now=ts) == []
    assert len(rt.step(now=1.45)) == 1  # window (0.4 s) elapsed -> close

    for ts in (2.0, 2.1, 2.2):  # session 2: floor-passing but unqualified -> REST
        reader.set_frame(frame, ts)
        rt.step(now=ts)
    rt.step(now=2.45)

    assert [(e.camera, e.decision, e.frames) for e in emitted] == [
        ("cam0", "plastic", 3),
        ("cam0", "rest", 3),
    ]
    assert emitted[0].confidence == pytest.approx(0.9)
    assert emitted[1].confidence is None
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line["decision"] for line in lines] == ["plastic", "rest"]
    for line, event in zip(lines, emitted, strict=True):
        assert line == json.loads(event.to_json())
        assert set(line) == {"camera", "decision", "confidence", "frames", "timestamp"}
    rt.stop()
    assert reader.stopped


def test_runtime_rejects_out_of_range_class_id() -> None:
    reader = StubReader("cam0")
    rt = _runtime(lambda frame: [(7, 0.95)], [reader], out=io.StringIO())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for ts in (1.0, 1.1, 1.2):
        reader.set_frame(frame, ts)
        rt.step(now=ts)
    with pytest.raises(DeployError, match="outside the 6-class map"):
        rt.step(now=1.45)


def test_runtime_constructor_fails_fast() -> None:
    with pytest.raises(DeployError, match="at least one camera"):
        _runtime(lambda frame: [], [])
    with pytest.raises(DeployError, match="window_seconds"):
        _runtime(lambda frame: [], [StubReader("cam0")], window_seconds=0.0)


# -- runtime end-to-end: 3 live MJPEG streams + fake model -------------------


def _predict_by_color(frame: np.ndarray) -> list[tuple[int, float]]:
    """Dominant BGR channel -> scripted vote: red=plastic 0.9 (bin), green=paper
    0.35 (below tau -> REST), blue=cardboard 0.5 (below high-water -> REST)."""
    blue, green, red = (float(frame[..., i].mean()) for i in range(3))
    if red >= green and red >= blue:
        return [(0, 0.9)]
    if green >= blue:
        return [(1, 0.35)]
    return [(2, 0.5)]


def test_runtime_end_to_end_three_streams() -> None:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    servers = [MJPEGServer([solid_jpeg(c)], fps=30.0) for c in colors]
    out = io.StringIO()
    events: dict[str, list[DecisionEvent]] = {f"cam{i}": [] for i in range(3)}
    for server in servers:
        server.__enter__()
    readers = start_readers(
        tuple(s.url for s in servers), stale_after_s=2.0, reconnect_backoff_s=0.2
    )
    rt = Runtime(
        classes=CLASSES,
        params=make_params(),
        predict=_predict_by_color,
        readers=readers,
        window_seconds=0.5,
        emit=lambda e: events[e.camera].append(e),
        out=out,
        poll_interval_s=0.0,
    )

    def done() -> bool:
        return (
            any(e.decision == "plastic" for e in events["cam0"])
            and bool(events["cam1"])
            and bool(events["cam2"])
        )

    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not done():
            rt.step()
            time.sleep(0.01)
    finally:
        rt.stop()
        for server in servers:
            server.stop()

    assert done()
    plastic = next(e for e in events["cam0"] if e.decision == "plastic")
    assert plastic.confidence == pytest.approx(0.9) and plastic.frames >= 3
    assert all(e.decision == "rest" and e.confidence is None for e in events["cam1"])
    assert all(e.decision == "rest" for e in events["cam2"])
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert {line["camera"] for line in lines} == {"cam0", "cam1", "cam2"}
    assert len(lines) == sum(len(v) for v in events.values())


# -- export host refusal -------------------------------------------------------


def test_export_refuses_non_aarch64_host(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="refusing to export"):
        ensure_jetson_arch(machine=lambda: "x86_64")
    with pytest.raises(ExportError, match="refusing to export"):
        export_engine(tmp_path / "best.pt", imgsz=640, machine=lambda: "arm64")
    ensure_jetson_arch(machine=lambda: "aarch64")  # Jetson passes
    ensure_jetson_arch(force_host=True, machine=lambda: "x86_64")  # explicit override
    # weights check fires before the lazy ultralytics import (not installed here)
    with pytest.raises(ExportError, match="weights not found"):
        export_engine(tmp_path / "best.pt", imgsz=640, machine=lambda: "aarch64")


# -- check_env probes (read-only) ----------------------------------------------


def test_check_l4t(tmp_path: Path) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text("# R36 (release), REVISION: 5.0\nextra\n")
    ok = check_l4t(release)
    assert ok.ok and ok.detail == "# R36 (release), REVISION: 5.0"
    bad = check_l4t(tmp_path / "absent")
    assert not bad.ok and "JetPack" in bad.remediation


def test_check_nvpmodel_states() -> None:
    assert check_nvpmodel(("echo", "NV Power Mode: MAXN_SUPER")).ok
    low = check_nvpmodel(("echo", "NV Power Mode: 15W"))
    assert not low.ok and "nvpmodel -m 2" in low.remediation
    missing = check_nvpmodel(("nonexistent-nvpmodel-binary",))
    assert not missing.ok and "not found" in missing.detail


def test_check_camera_probe_and_run_checks(tmp_path: Path) -> None:
    with MJPEGServer([solid_jpeg((255, 0, 0))], fps=30.0) as server:
        up = check_camera(server.url, timeout_s=5.0)
        assert up.ok and "200" in up.detail
        results = run_checks(
            (server.url,),
            release_path=tmp_path / "absent",
            nvpmodel_command=("echo", "NV Power Mode: MAXN_SUPER"),
        )
    assert [r.ok for r in results] == [False, True, True]
    down = check_camera("http://127.0.0.1:9/stream", timeout_s=1.0)
    assert not down.ok and "unreachable" in down.detail and down.remediation
