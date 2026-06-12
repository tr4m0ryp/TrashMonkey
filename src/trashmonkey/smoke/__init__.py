"""End-to-end smoke harness (task 016): ``python -m trashmonkey.smoke``.

Runs the full chain against generated synthetic fixtures in a throwaway
workdir: download -> remap -> autobox (forced centerbox) -> qa (acked) ->
dedup -> balance -> split -> 1-epoch CPU train -> three-tier evaluate ->
wilderness dump -> reduced threshold sweep. ``FAKE_MODEL=1`` injects the
offline ultralytics double; without it the real package is required.
"""

from trashmonkey.smoke.cli import main
from trashmonkey.smoke.fakes import install_fake_ultralytics
from trashmonkey.smoke.fixtures import generate_fixtures
from trashmonkey.smoke.workspace import FIXTURES_DIR, SMOKE_CONFIG, materialize

__all__ = [
    "FIXTURES_DIR",
    "SMOKE_CONFIG",
    "generate_fixtures",
    "install_fake_ultralytics",
    "main",
    "materialize",
]
