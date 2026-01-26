"""Camera calibration parsing and camera_calib.json writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from .utils import write_json


FRAME_CONVENTION = {
    "ego": "x-forward, y-left, z-up",
    "camera": "x-forward(out_of_lens), y-left, z-up",
}

MATRIX_CONVENTION = {
    "vector_is_column": True,
    "multiply": "left",  # p_out = T * p_in
}


def _flatten_intrinsic(calib) -> Dict[str, Any]:
    intr = list(getattr(calib, "intrinsic", []))
    fx, fy, cx, cy = (float(intr[i]) if i < len(intr) else 0.0 for i in range(4))
    K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
    out = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K}
    # optional fields if present
    if hasattr(calib, "width") and hasattr(calib, "height"):
        out["resolution"] = {"width": int(calib.width), "height": int(calib.height)}
    if hasattr(calib, "distortion"):
        dist = list(getattr(calib, "distortion", []))
        if dist:
            out["distortion"] = dist
    return out


def _flatten_extrinsic(calib) -> Dict[str, Any]:
    extr = list(getattr(calib, "extrinsic", ()).transform)
    T_ego_from_cam = [list(extr[i:i + 4]) for i in range(0, len(extr), 4)] if len(extr) == 16 else None
    out: Dict[str, Any] = {}
    if T_ego_from_cam:
        out["T_ego_from_cam"] = T_ego_from_cam
    return out


def parse_calibration(frame) -> Dict[int, Dict[str, Any]]:
    """
    Extract per-cam intrinsics/extrinsics from a frame.
    Reference parsing details from export_package_fast_overlay_keymoment.py.
    """
    calib_map: Dict[int, Dict[str, Any]] = {}
    ctx = getattr(frame, "context", None)
    cals: Iterable[Any] = getattr(ctx, "camera_calibrations", []) if ctx else []
    for calib in cals:
        cam_id = int(getattr(calib, "name", -1))
        if cam_id <= 0:
            continue
        entry: Dict[str, Any] = {
            "cam_id": cam_id,
            "frame_convention": FRAME_CONVENTION,
            "matrix_convention": MATRIX_CONVENTION,
        }
        entry.update(_flatten_intrinsic(calib))
        entry.update(_flatten_extrinsic(calib))
        calib_map[cam_id] = entry
    return calib_map


def write_camera_calib_json(calib_map: Dict[int, Dict[str, Any]], out_path: Path) -> None:
    """Write camera_calib.json with required conventions and 1-based cam ids."""
    payload = {
        "frame_convention": FRAME_CONVENTION,
        "matrix_convention": MATRIX_CONVENTION,
        "cameras": {str(k): v for k, v in sorted(calib_map.items(), key=lambda kv: kv[0])},
    }
    write_json(out_path, payload)
