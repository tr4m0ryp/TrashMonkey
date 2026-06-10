"""Attribution-table tests: real registry validation, coverage, determinism.

Reads ONLY local files (configs/ + reports/); no network, no downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_waste_sorter.data.attribution import (
    CENSUS_IMAGE_COUNTS,
    EXCLUDED_SOURCES,
    latex_escape,
    render_dataset_licenses,
    render_label_mapping,
)
from yolo_waste_sorter.data.download import DROP, SourceSpec, load_registry
from yolo_waste_sorter.utils.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_PATH = REPO_ROOT / "configs" / "datasets.yaml"

DEDUP_PRIORITY_ORDER = [
    "trashnet",
    "drinking-waste",
    "garbage-detection",
    "alistairking-household",
    "realwaste",
]


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(REPO_ROOT / "configs" / "config.yaml")


@pytest.fixture(scope="module")
def registry(config: Config) -> dict[str, SourceSpec]:
    return load_registry(DATASETS_PATH, config.classes)


class TestRealRegistry:
    def test_loads_and_validates(self, registry: dict[str, SourceSpec]) -> None:
        assert len(registry) == 5

    def test_source_order_is_dedup_priority(self, registry: dict[str, SourceSpec]) -> None:
        assert list(registry) == DEDUP_PRIORITY_ORDER

    def test_every_mapping_target_is_class_or_drop(
        self, registry: dict[str, SourceSpec], config: Config
    ) -> None:
        for spec in registry.values():
            for label, target in spec.mapping.items():
                assert target == DROP or target in config.classes, f"{spec.name}: {label}"

    def test_every_class_covered_organic_twice(
        self, registry: dict[str, SourceSpec], config: Config
    ) -> None:
        sources_per_class = {
            cls: [s.name for s in registry.values() if cls in s.mapping.values()]
            for cls in config.classes
        }
        for cls, names in sources_per_class.items():
            assert names, f"target class '{cls}' is not mapped by any source"
        assert len(sources_per_class["organic"]) >= 2

    def test_alistairking_mapping_total_over_30_folders(
        self, registry: dict[str, SourceSpec]
    ) -> None:
        spec = registry["alistairking-household"]
        assert len(spec.mapping) == 30
        assert spec.dropped_labels() == {"clothing", "shoes"}

    def test_realwaste_is_the_leave_out_source(self, config: Config) -> None:
        assert config.eval.leave_out_source == "realwaste"
        assert config.eval.leave_out_source in DEDUP_PRIORITY_ORDER

    def test_census_counts_cover_exactly_the_registry(
        self, registry: dict[str, SourceSpec]
    ) -> None:
        assert set(CENSUS_IMAGE_COUNTS) == set(registry)


class TestRenderedFragments:
    def test_label_mapping_regeneration_is_deterministic(
        self, registry: dict[str, SourceSpec]
    ) -> None:
        committed = (REPO_ROOT / "reports" / "tab_label_mapping.tex").read_text()
        assert render_label_mapping(registry) == committed

    def test_licenses_regeneration_is_deterministic(
        self, registry: dict[str, SourceSpec], config: Config
    ) -> None:
        committed = (REPO_ROOT / "reports" / "tab_dataset_licenses.tex").read_text()
        assert render_dataset_licenses(registry, config.eval.leave_out_source) == committed

    def test_label_mapping_shows_drops_per_source(
        self, registry: dict[str, SourceSpec]
    ) -> None:
        fragment = render_label_mapping(registry)
        # Row cells only (the caption also mentions \textsc{drop});
        # drinking-waste has no DROP labels, so 4 of the 5 sources have a row.
        assert fragment.count(r"& \textsc{drop} &") == 4
        assert "Miscellaneous Trash" in fragment
        assert "plastic\\_water\\_bottles" in fragment

    def test_license_table_roles(
        self, registry: dict[str, SourceSpec], config: Config
    ) -> None:
        fragment = render_dataset_licenses(registry, config.eval.leave_out_source)
        assert fragment.count("& train &") == 4
        assert fragment.count("& test &") == 1
        assert "realwaste & 4{,}752 & CC BY 4.0 & test &" in fragment
        assert fragment.count("& qa-ref &") == 1
        assert "Polygence" in fragment

    def test_license_table_lists_all_exclusions(
        self, registry: dict[str, SourceSpec], config: Config
    ) -> None:
        fragment = render_dataset_licenses(registry, config.eval.leave_out_source)
        for name, reason in EXCLUDED_SOURCES:
            assert latex_escape(name) in fragment
            assert latex_escape(reason) in fragment
        for expected in ("WaDaBa", "ZeroWaste", "TACO", "CompostNet"):
            assert expected in fragment

    def test_unknown_source_fails_fast(self, config: Config) -> None:
        fake = {
            "mystery": SourceSpec(
                name="mystery",
                fetcher=next(iter(load_registry(DATASETS_PATH, config.classes).values())).fetcher,
                license="MIT",
                attribution="nobody",
                annotation_type="cls",
                background="clean",
                mapping={"x": "plastic"},
            )
        }
        with pytest.raises(ValueError, match="mystery"):
            render_dataset_licenses(fake, None)


class TestLatexEscape:
    def test_specials(self) -> None:
        assert latex_escape("a_b & c% #d") == r"a\_b \& c\% \#d"

    def test_backslash_and_braces(self) -> None:
        assert latex_escape(r"\x{y}") == r"\textbackslash{}x\{y\}"

    def test_tilde_and_caret(self) -> None:
        assert latex_escape("~^") == r"\textasciitilde{}\textasciicircum{}"
