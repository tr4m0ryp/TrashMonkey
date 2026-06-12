"""Three-tier evaluation (task 011): T6 tiers, T7 escalation, T9 dump.

Report-only on TEST-1/TEST-2: nothing here tunes or selects on the test
tiers. The public surface is re-exported by ``trashmonkey.models
.evaluate`` (the stable import path + CLI).
"""

from trashmonkey.models.evaluation.core import (
    DETECTIONS_FILENAME,
    REPORT_FILENAME,
    SWEEP_CONF,
    evaluate,
)
from trashmonkey.models.evaluation.curves import (
    PRECISION_FLOOR,
    CurveSet,
    conf_at_precision,
    extract_curves,
    save_curves,
)
from trashmonkey.models.evaluation.degraded import (
    load_dataset_spec,
    materialize_severity,
    split_images,
)
from trashmonkey.models.evaluation.detections import (
    ManifestIndex,
    dump_detections,
    image_identity,
    load_manifest_index,
)
from trashmonkey.models.evaluation.report import (
    CLEAN_SEVERITY,
    ClassEval,
    EvalError,
    EvalReport,
    SeverityPoint,
    TierReport,
    load_report,
    report_from_dict,
)

__all__ = [
    "CLEAN_SEVERITY",
    "DETECTIONS_FILENAME",
    "PRECISION_FLOOR",
    "REPORT_FILENAME",
    "SWEEP_CONF",
    "ClassEval",
    "CurveSet",
    "EvalError",
    "EvalReport",
    "ManifestIndex",
    "SeverityPoint",
    "TierReport",
    "conf_at_precision",
    "dump_detections",
    "evaluate",
    "extract_curves",
    "image_identity",
    "load_dataset_spec",
    "load_manifest_index",
    "load_report",
    "materialize_severity",
    "report_from_dict",
    "save_curves",
    "split_images",
]
