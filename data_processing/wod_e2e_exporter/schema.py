"""Manifest and summary schema builders (structure per final.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "v3.3-codex"


@dataclass
class SegmentManifest:
    seg_id: str
    split: str
    scenario_cluster: str
    routing_command: str
    duration_sec: float
    image_dir: str
    calibration_file: str
    ego_state_file: str
    ego_state_hz: float
    ego_state_fields: List[str]
    steering_field: str
    ego_future: Dict[str, Any]
    rater_labels: Dict[str, Any]
    time_info: Dict[str, Any] = field(default_factory=dict)
    camera_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentSummary:
    seg_id: str
    status: str
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)


def build_segment_manifest(*, is_critical_5s: bool, **kwargs) -> Dict[str, Any]:
    """
    Build manifest dict for 20s or 5s view.
    Required kwargs (common):
      seg_id, split, scenario_cluster, routing_command,
      image_dir, calibration_file,
      ego_state_file, ego_state_hz, ego_state_fields, steering_field,
      ego_future (dict with available/file/reason),
      rater_labels (dict per spec),
      camera_info (fps_hz,num_cameras,camera_ids),
      time_info (dict of required time fields),
      duration_sec (float).
    Additional time fields differ for 20s vs 5s per final.md.
    """
    seg_id = kwargs["seg_id"]
    split = kwargs["split"]
    scenario_cluster = kwargs["scenario_cluster"]
    routing_command = kwargs["routing_command"]
    routing_command_name = kwargs.get("routing_command_name")
    duration_sec = kwargs["duration_sec"]
    image_dir = kwargs["image_dir"]
    calibration_file = kwargs["calibration_file"]
    ego_state_file = kwargs["ego_state_file"]
    ego_state_hz = kwargs["ego_state_hz"]
    ego_state_fields = kwargs["ego_state_fields"]
    steering_field = kwargs["steering_field"]
    ego_future = kwargs["ego_future"]
    rater_labels = kwargs["rater_labels"]
    camera_info = kwargs["camera_info"]
    time_info = kwargs["time_info"]

    base = {
        "scenario_cluster": scenario_cluster,
        "seg_id": seg_id,
        "split": split,
        "routing_command": routing_command,
        "routing_command_name": routing_command_name,
        "time": {
            "segment_t0_sec": 0.0,
            "duration_sec": float(duration_sec),
        },
        "camera": {
            "fps_hz": camera_info.get("fps_hz", 10.0),
            "num_cameras": camera_info.get("num_cameras", 8),
            "camera_ids": camera_info.get("camera_ids", list(range(1, 9))),
            "image_dir": image_dir,
            "calibration_file": calibration_file,
        },
        "ego_state": {
            "file": ego_state_file,
            "hz": ego_state_hz,
            "fields": ego_state_fields,
            "steering_field": steering_field,
        },
        "ego_future": ego_future,
        "rater_labels": {
            "available": rater_labels.get("available", False),
            "reason": rater_labels.get("reason", ""),
            "missing_score_value": -1.0,
        },
    }

    # rater trajectories (only for val)
    if "trajectories" in rater_labels:
        base["rater_labels"]["trajectories"] = rater_labels["trajectories"]

    cam_positions = camera_info.get("positions_ego_m")
    if cam_positions:
        base["camera"]["positions_ego_m"] = cam_positions
    cam_labels = camera_info.get("orientation_labels")
    if cam_labels:
        base["camera"]["orientation_labels"] = cam_labels

    # time info
    if is_critical_5s:
        base["time"].update({
            "parent_seg_id": time_info["parent_seg_id"],
            "critical_timestamp_in_parent_sec": time_info["critical_timestamp_in_parent_sec"],
            "parent_frame_index_start_10hz": time_info["parent_frame_index_start_10hz"],
            "mapping_note": time_info.get("mapping_note", "nearest"),
        })
    else:
        base["time"].update({
            "critical_timestamp_sec": time_info["critical_timestamp_sec"],
            "critical_frame_index_10hz": time_info["critical_frame_index_10hz"],
        })

    return base


def build_segment_summary(**kwargs) -> Dict[str, Any]:
    """Build summary dict with schema_version and status/errors/warnings."""
    seg_id = kwargs["seg_id"]
    status = kwargs["status"]
    errors = kwargs.get("errors", [])
    warnings = kwargs.get("warnings", [])
    counts = kwargs.get("counts", {})
    timing = kwargs.get("timing", {})
    scenario_cluster = kwargs.get("scenario_cluster", None)

    out = {}
    if scenario_cluster is not None:
        out["scenario_cluster"] = scenario_cluster
    out.update({
        "seg_id": seg_id,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
        "timing": timing,
    })
    return out
