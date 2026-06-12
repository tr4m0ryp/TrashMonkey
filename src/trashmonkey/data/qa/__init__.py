"""Label QA for auto-generated boxes (T3): checks, review queue, IoU cross-check.

Public surface re-exported here; submodules are implementation detail.
"""

from .boxes import Box, iou_cxcywh, parse_label_file
from .checks import (
    ALL_FLAGS,
    AREA_RATIO_MAX,
    AREA_RATIO_MIN,
    CONFIDENCE_MIN,
    EDGE_TOUCH_EPS,
    EDGE_TOUCH_MIN,
    FLAG_AREA_EXTREME,
    FLAG_AREA_ZSCORE,
    FLAG_ASPECT_ZSCORE,
    FLAG_BOX_COUNT,
    FLAG_CENTERBOX,
    FLAG_EDGE_CONTACT,
    FLAG_LOW_CONFIDENCE,
    ZSCORE_MAX,
    run_checks,
)
from .crosscheck import IoUStats, iou_crosscheck
from .report import (
    LOC_FAIL_MAX,
    REVIEW_FAIL_MAX,
    TARGET_MEDIAN_IOU,
    ImageQA,
    ProvenanceRecord,
    QAReport,
    load_provenance,
)
from .review import emit_review_queue, stratified_sample

__all__ = [
    "ALL_FLAGS",
    "AREA_RATIO_MAX",
    "AREA_RATIO_MIN",
    "CONFIDENCE_MIN",
    "EDGE_TOUCH_EPS",
    "EDGE_TOUCH_MIN",
    "FLAG_AREA_EXTREME",
    "FLAG_AREA_ZSCORE",
    "FLAG_ASPECT_ZSCORE",
    "FLAG_BOX_COUNT",
    "FLAG_CENTERBOX",
    "FLAG_EDGE_CONTACT",
    "FLAG_LOW_CONFIDENCE",
    "LOC_FAIL_MAX",
    "REVIEW_FAIL_MAX",
    "TARGET_MEDIAN_IOU",
    "ZSCORE_MAX",
    "Box",
    "ImageQA",
    "IoUStats",
    "ProvenanceRecord",
    "QAReport",
    "emit_review_queue",
    "iou_crosscheck",
    "iou_cxcywh",
    "load_provenance",
    "parse_label_file",
    "run_checks",
    "stratified_sample",
]
