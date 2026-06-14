"""Tests for the progress reporting utilities."""

from __future__ import annotations

import io

from trashmonkey.utils.progress import make_progress_printer, print_stage


def test_progress_printer_shows_percentage_and_finishes_with_newline() -> None:
    out = io.StringIO()
    sink = make_progress_printer(out)
    for i in range(1, 11):
        sink("autobox", i, 10)
    text = out.getvalue()
    assert "autobox:" in text
    assert "100%  (10/10)" in text
    assert text.endswith("\n")  # completion emits a newline, not a bare \r


def test_progress_printer_throttles_to_whole_percent() -> None:
    out = io.StringIO()
    sink = make_progress_printer(out)
    for i in range(1, 201):  # 200 updates over a 200-item job
        sink("autobox", i, 200)
    # At most one redraw per integer percent (0..100) -> <= 101 writes, not 200.
    assert out.getvalue().count("%") <= 101


def test_progress_printer_newline_between_labels() -> None:
    out = io.StringIO()
    sink = make_progress_printer(out)
    sink("download", 1, 2)
    sink("autobox", 1, 4)  # label change finishes the prior line
    assert "\n" in out.getvalue()


def test_progress_printer_tolerates_zero_total() -> None:
    out = io.StringIO()
    sink = make_progress_printer(out)
    sink("empty", 0, 0)  # must not raise on division
    assert "100%" in out.getvalue()


def test_print_stage_format() -> None:
    out = io.StringIO()
    print_stage("autobox", 3, 7, stream=out)
    assert out.getvalue() == "[3/7] autobox\n"
