"""Exporters for images/trajectory/meta/manifest/summary (20s + critical 5s)."""

from __future__ import annotations

import json
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .utils import ensure_dir, write_json
from . import schema, camera_meta, slicer


def _base_seg_id(seg_id: str) -> str:
    if not seg_id:
        return seg_id
    if "-" not in seg_id:
        return seg_id
    base, tail = seg_id.rsplit("-", 1)
    return base if tail.isdigit() else seg_id


def _write_logs(log_dir: Path, content: str) -> None:
    ensure_dir(log_dir)
    log_path = log_dir / "export.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(content + "\n")


def _list_available_cam_ids(frame) -> List[int]:
    imgs = getattr(frame, "images", []) if frame else []
    return sorted({int(getattr(im, "name", -1)) for im in imgs if hasattr(im, "name")})


def _get_image_bytes_by_cam_id(frame, cam_id: int) -> bytes | None:
    imgs = getattr(frame, "images", []) if frame else []
    for im in imgs:
        if int(getattr(im, "name", -1)) == int(cam_id):
            return bytes(getattr(im, "image", b""))
    return None


def _write_placeholder_csv(path: Path, headers: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow([0.0] + ["" for _ in headers[1:]])  # t_sec = 0.0


def _write_ego_state_placeholder(path: Path, num_rows: int, hz: float, warnings: List[Dict[str, Any]]) -> None:
    headers = ["t_sec", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "yaw_rad"]
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(max(1, num_rows)):
            writer.writerow([f"{i / hz:.6f}"] + [""] * (len(headers) - 1))
    warnings.append({"type": "ego_state_placeholder", "message": f"ego_state placeholder rows={max(1, num_rows)}", "where": str(path)})


def _is_missing_value(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def _fill_rows_from_states(
    rows: List[Dict[str, Any]],
    t_sec: List[float],
    past_states: Any,
    future_states: Any,
    critical_idx: int | None,
    warnings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    n = min(len(rows), len(t_sec))
    if n == 0:
        return rows
    past_rows = _states_to_rows(past_states)
    future_rows = _states_to_rows(future_states)
    if not past_rows and not future_rows:
        return rows

    if critical_idx is None or critical_idx < 0 or critical_idx >= n:
        critical_idx = 0
    anchor_t = float(t_sec[critical_idx]) if t_sec else 0.0

    def _nearest_index(target_t: float, tol: float = 0.075) -> int | None:
        best = None
        best_dt = None
        for i in range(n):
            dt = abs(t_sec[i] - target_t)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = i
        if best_dt is None or best_dt > tol:
            return None
        return best

    def _xy_close(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-2) -> bool:
        ax = a.get("x_m")
        ay = a.get("y_m")
        bx = b.get("x_m")
        by = b.get("y_m")
        if _is_missing_value(ax) or _is_missing_value(ay) or _is_missing_value(bx) or _is_missing_value(by):
            return False
        try:
            return abs(float(ax) - float(bx)) <= tol and abs(float(ay) - float(by)) <= tol
        except Exception:
            return False

    future_offset = 1.0 / 4.0
    if past_rows and future_rows and _xy_close(past_rows[-1], future_rows[0]):
        future_offset = 0.0

    filled_rows = set()

    def _merge_row(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if k == "t_sec":
                continue
            if k not in dst or _is_missing_value(dst.get(k)):
                dst[k] = float(v) if v is not None else float("nan")

    if past_rows:
        for i, r in enumerate(past_rows):
            t_rel = (i - (len(past_rows) - 1)) / 4.0
            idx = _nearest_index(anchor_t + t_rel)
            if idx is None:
                continue
            _merge_row(rows[idx], r)
            filled_rows.add(idx)

    if future_rows:
        for i, r in enumerate(future_rows):
            t_rel = future_offset + (i / 4.0)
            idx = _nearest_index(anchor_t + t_rel)
            if idx is None:
                continue
            _merge_row(rows[idx], r)
            filled_rows.add(idx)

    if filled_rows:
        warnings.append({
            "type": "ego_state_from_states",
            "message": f"filled {len(filled_rows)} ego_state rows from past/future states",
            "where": "exporters.ego_state",
        })

    return rows


def _build_ego_state_rows(
    frames: List[Any],
    t_sec: List[float],
    warnings: List[Dict[str, Any]],
    past_states: Any = None,
    future_states: Any = None,
    critical_idx: int | None = None,
) -> List[Dict[str, Any]]:
    n = len(t_sec) if t_sec else len(frames)
    if n <= 0:
        return []
    times = list(t_sec) if len(t_sec) == n else [i / 10.0 for i in range(n)]

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rows.append({
            "t_sec": float(times[i]),
            "x_m": float("nan"),
            "y_m": float("nan"),
            "z_m": float("nan"),
            "vx_mps": float("nan"),
            "vy_mps": float("nan"),
            "ax_mps2": float("nan"),
            "ay_mps2": float("nan"),
            "yaw_rad": float("nan"),
        })

    rows = _fill_rows_from_states(rows, times, past_states, future_states, critical_idx, warnings)

    # Backfill kinematics if still missing but positions are available.
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        if _is_missing_value(rows[i].get("vx_mps")) and not _is_missing_value(rows[i].get("x_m")) and not _is_missing_value(rows[i - 1].get("x_m")):
            rows[i]["vx_mps"] = (rows[i]["x_m"] - rows[i - 1]["x_m"]) / dt
        if _is_missing_value(rows[i].get("vy_mps")) and not _is_missing_value(rows[i].get("y_m")) and not _is_missing_value(rows[i - 1].get("y_m")):
            rows[i]["vy_mps"] = (rows[i]["y_m"] - rows[i - 1]["y_m"]) / dt
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        if _is_missing_value(rows[i].get("ax_mps2")) and not _is_missing_value(rows[i].get("vx_mps")) and not _is_missing_value(rows[i - 1].get("vx_mps")):
            rows[i]["ax_mps2"] = (rows[i]["vx_mps"] - rows[i - 1]["vx_mps"]) / dt
        if _is_missing_value(rows[i].get("ay_mps2")) and not _is_missing_value(rows[i].get("vy_mps")) and not _is_missing_value(rows[i - 1].get("vy_mps")):
            rows[i]["ay_mps2"] = (rows[i]["vy_mps"] - rows[i - 1]["vy_mps"]) / dt

    return rows


def _write_ego_state_csv(path: Path, rows: List[Dict[str, Any]], warnings: List[Dict[str, Any]]) -> List[str]:
    ensure_dir(path.parent)
    if not rows:
        warnings.append({"type": "ego_state_empty", "message": "no ego_state rows", "where": str(path)})
        _write_placeholder_csv(path, ["t_sec", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "yaw_rad"])
        return []
    fields = ["t_sec", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "yaw_rad"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r.get(k, float("nan")) for k in fields])
    return fields


def _write_rater_trajs(
    traj_dir: Path,
    rater_trajs: List[Any],
    warnings: List[Dict[str, Any]],
    num_trajs: int = 3,
) -> List[Dict[str, Any]]:
    """Write fixed count rater trajectories (val) and return entries relative to critical_5s/."""
    ensure_dir(traj_dir)
    manifest_entries: List[Dict[str, Any]] = []
    by_idx: Dict[int, Any] = {}
    for rt in rater_trajs or []:
        idx = getattr(rt, "index", None)
        if idx is None:
            continue
        if int(idx) not in by_idx:
            by_idx[int(idx)] = rt

    for idx in range(num_trajs):
        rt = by_idx.get(idx)
        df = getattr(rt, "df", None) if rt is not None else None
        score = getattr(rt, "score", None) if rt is not None else None
        score_missing = score is None or (isinstance(score, float) and score != score)
        score_clean = -1.0 if score_missing else float(score)
        fname = traj_dir / f"rater_traj_{idx}_0to5s_4hz.csv"
        ensure_dir(fname.parent)
        if df is None or df.empty:
            with fname.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t_sec", "x_m", "y_m", "z_m"])
                w.writerow(["0.0", "nan", "nan", "nan"])
            warnings.append({"type": "rater_missing", "message": f"rater {idx} missing or empty", "where": str(fname)})
        else:
            df = df.copy()
            if "t_sec" not in df.columns:
                df.insert(0, "t_sec", [f"{i/4.0:.6f}" for i in range(len(df))])
            if "x" in df.columns:
                df.rename(columns={"x": "x_m"}, inplace=True)
            if "y" in df.columns:
                df.rename(columns={"y": "y_m"}, inplace=True)
            if "z" in df.columns:
                df.rename(columns={"z": "z_m"}, inplace=True)
            for col in ["x_m", "y_m", "z_m"]:
                if col not in df.columns:
                    df[col] = "nan"
            ordered_cols = ["t_sec", "x_m", "y_m", "z_m"] + [c for c in df.columns if c not in {"t_sec", "x_m", "y_m", "z_m"}]
            df = df[ordered_cols]
            df.to_csv(fname, index=False)
        if score_missing:
            warnings.append({"type": "rater_score_missing", "message": f"rater {idx} score missing; default -1.0", "where": str(fname)})
        manifest_entries.append({"file": str(fname.relative_to(traj_dir.parent)), "score": score_clean})
    return manifest_entries


def _prefix_rater_entries(entries: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    return [{"file": f"{prefix}{e['file']}", "score": e.get("score", -1.0)} for e in entries]


def _write_images(frames: List[Any], out_root: Path, warnings: List[Dict[str, Any]]) -> int:
    """
    Write camera images for a sequence of frames into out_root/cam_{id}/{idx}.jpg.
    Returns number of images written.
    """
    written = 0
    for frame_idx, frame in enumerate(frames):
        cam_ids = _list_available_cam_ids(frame)
        if not cam_ids:
            warnings.append({"type": "missing_images", "message": f"no images in frame_idx={frame_idx}", "where": "exporters.images"})
            continue
        for cid in cam_ids:
            jpg = _get_image_bytes_by_cam_id(frame, cid)
            if jpg is None:
                warnings.append({"type": "missing_image_cam", "message": f"missing cam {cid} at frame_idx={frame_idx}", "where": "exporters.images"})
                continue
            cam_dir = out_root / f"cam_{cid}"
            ensure_dir(cam_dir)
            out_jpg = cam_dir / f"{frame_idx:06d}.jpg"
            with out_jpg.open("wb") as f:
                f.write(jpg)
            written += 1
    return written


def _states_to_rows(states_msg) -> List[Dict[str, Any]]:
    """Convert past/future states message to list of dicts aligned by index."""
    if states_msg is None:
        return []
    fields = [
        ("pos_x", "x_m"),
        ("pos_y", "y_m"),
        ("pos_z", "z_m"),
        ("vel_x", "vx_mps"),
        ("vel_y", "vy_mps"),
        ("accel_x", "ax_mps2"),
        ("accel_y", "ay_mps2"),
        ("heading", "yaw_rad"),
    ]
    arrays = {}
    lengths = []
    for proto_name, out_name in fields:
        if hasattr(states_msg, proto_name):
            arr = list(getattr(states_msg, proto_name))
            arrays[out_name] = arr
            lengths.append(len(arr))
    if not arrays or not lengths:
        return []
    n = min(l for l in lengths if l > 0)
    rows = []
    for i in range(n):
        row = {}
        for out_name, arr in arrays.items():
            if len(arr) > i:
                row[out_name] = arr[i]
        rows.append(row)
    return rows


def _write_traj_csv(path: Path, rows: List[Dict[str, Any]], hz: float, warnings: List[Dict[str, Any]]) -> None:
    """Write trajectory CSV with t_sec leading column."""
    ensure_dir(path.parent)
    if not rows:
        warnings.append({"type": "missing_traj", "message": f"no data for {path.name}", "where": str(path)})
        _write_placeholder_csv(path, ["t_sec", "x_m", "y_m", "z_m"])
        return
    # collect all keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = [k for k in ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "yaw_rad"] if k in keys] + [k for k in keys if k not in {"x_m", "y_m", "z_m", "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "yaw_rad"}]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_sec"] + keys)
        for i, r in enumerate(rows):
            t = i / hz
            writer.writerow([f"{t:.6f}"] + [r.get(k, "") for k in keys])


def export_segment(segment_record: Any, out_dir: Path, overwrite: bool = False) -> Dict[str, Any]:
    """
    Write full 20s view plus critical_5s subdir.
    Current state: writes directory layout, camera_calib.json, real images, placeholder CSVs/manifests/summaries.
    """
    seg_base = _base_seg_id(segment_record.seg_id)
    cluster_dir_name = str(segment_record.scenario_cluster).replace("/", "_").replace("\\", "_")
    seg_dir = Path(out_dir) / cluster_dir_name / f"{cluster_dir_name}_segment-{seg_base}"
    if seg_dir.exists() and not overwrite:
        return {"status": "skipped_exists", "seg_dir": str(seg_dir)}

    # Prepare directories per spec
    images_20s = seg_dir / "images"
    traj_20s = seg_dir / "trajectory"
    meta_20s = seg_dir / "meta"
    crit_dir = seg_dir / "critical_5s"
    images_5s = crit_dir / "images"
    traj_5s = crit_dir / "trajectory"
    meta_5s = crit_dir / "meta"
    log_dir = seg_dir / "_logs"

    for p in [images_20s, traj_20s, meta_20s, images_5s, traj_5s, meta_5s, log_dir]:
        ensure_dir(p)

    # Compute mapping and write camera calib
    mapping = slicer.slice_segment(segment_record, segment_record.critical_frame_index_10hz)
    calib_map = camera_meta.parse_calibration(segment_record.frame)
    camera_meta.write_camera_calib_json(calib_map, meta_20s / "camera_calib.json")
    camera_meta.write_camera_calib_json(calib_map, meta_5s / "camera_calib.json")

    warnings: List[Dict[str, Any]] = []
    counts: Dict[str, Any] = {}

    # Real image export for available frames
    frames_20s = getattr(segment_record, "frames_20s", [segment_record.frame])
    if len(frames_20s) != 200:
        warnings.append({"type": "frame_count_mismatch", "message": f"frames_20s={len(frames_20s)} (expected 200)", "where": "exporters.images"})
    counts["frames_available"] = len(frames_20s)
    img_written_20s = _write_images(frames_20s, images_20s, warnings)
    counts["images_written_20s"] = img_written_20s

    # 5s images based on slicer indices (clip to available frames)
    img_idx_map = mapping["images_indices"]
    frames_5s = [frames_20s[i] for i in img_idx_map if i < len(frames_20s)]
    if len(frames_5s) != 50:
        warnings.append({"type": "critical_window_short", "message": f"frames_5s={len(frames_5s)} (expected 50)", "where": "exporters.images"})
    img_written_5s = _write_images(frames_5s, images_5s, warnings)
    counts["images_written_5s"] = img_written_5s

    # Trajectory CSVs from proto states (best-effort)
    past_rows = _states_to_rows(getattr(segment_record, "past_states", None))
    future_rows = _states_to_rows(getattr(segment_record, "future_states", None))
    _write_traj_csv(traj_20s / "ego_past_4s_4hz.csv", past_rows, hz=4.0, warnings=warnings)
    _write_traj_csv(traj_20s / "ego_future_5s_4hz.csv", future_rows, hz=4.0, warnings=warnings)
    _write_traj_csv(traj_5s / "ego_future_0to5s_4hz.csv", future_rows, hz=4.0, warnings=warnings)
    # ego_state from past/future states (4Hz) aligned to 10Hz timeline
    frames_20s = getattr(segment_record, "frames_20s", [segment_record.frame])
    times_20s = getattr(segment_record, "cam_times_20s", [i / 10.0 for i in range(len(frames_20s))])
    crit_idx_20s = int(getattr(segment_record, "critical_frame_index_10hz", 0) or 0)
    past_states_list = getattr(segment_record, "past_states_20s", None)
    future_states_list = getattr(segment_record, "future_states_20s", None)
    past_states = past_states_list[crit_idx_20s] if past_states_list and 0 <= crit_idx_20s < len(past_states_list) else getattr(segment_record, "past_states", None)
    future_states = future_states_list[crit_idx_20s] if future_states_list and 0 <= crit_idx_20s < len(future_states_list) else getattr(segment_record, "future_states", None)
    ego_rows_20s = _build_ego_state_rows(frames_20s, times_20s, warnings, past_states=past_states, future_states=future_states, critical_idx=crit_idx_20s)
    fields_20s = _write_ego_state_csv(traj_20s / "ego_state_20s_10hz.csv", ego_rows_20s, warnings)
    # 5s frames based on slicer indices
    indices_5s = mapping["images_indices"]
    frames_5s = [frames_20s[i] for i in indices_5s if i < len(frames_20s)]
    times_5s = [times_20s[i] - times_20s[indices_5s[0]] for i in indices_5s if i < len(times_20s)] if indices_5s and len(times_20s) > 0 else [i / 10.0 for i in range(len(frames_5s))]
    ego_rows_5s = _build_ego_state_rows(frames_5s, times_5s, warnings, past_states=past_states, future_states=future_states, critical_idx=0)
    fields_5s = _write_ego_state_csv(traj_5s / "ego_state_0to5s_10hz.csv", ego_rows_5s, warnings)
    counts["csv_written"] = 5

    # summary
    # Ego state availability (placeholder for now)
    ego_state_fields = fields_20s if fields_20s else ["t_sec", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "ax_mps2", "ay_mps2", "yaw_rad"]
    steering_field = ""
    if not steering_field:
        warnings.append({"type": "missing_steering", "message": "no steering field found", "where": "exporters.ego_state"})

    # Ego future availability
    ego_future_available = len(future_rows) > 0
    ego_future_info_20s = {
        "available": ego_future_available,
        "file": "trajectory/ego_future_5s_4hz.csv" if ego_future_available else "",
        "reason": "" if ego_future_available else "no_future_states",
    }
    ego_future_info_5s = {
        "available": ego_future_available,
        "file": "trajectory/ego_future_0to5s_4hz.csv" if ego_future_available else "",
        "reason": "" if ego_future_available else "no_future_states",
    }

    # rater labels (val only)
    rater_manifest_entries_5s: List[Dict[str, Any]] = []
    if str(segment_record.split).lower() == "val":
        rater_manifest_entries_5s = _write_rater_trajs(traj_5s, getattr(segment_record, "rater_trajs", []), warnings)
        rater_labels_info_5s = {
            "available": True,
            "reason": "",
            "trajectories": rater_manifest_entries_5s,
        }
        rater_labels_info_20s = {
            "available": True,
            "reason": "",
            "trajectories": _prefix_rater_entries(rater_manifest_entries_5s, "critical_5s/"),
        }
    else:
        rater_labels_info_5s = {
            "available": False,
            "reason": "split_no_rater_labels",
        }
        rater_labels_info_20s = dict(rater_labels_info_5s)

    # derive critical timestamp from time array if available
    crit_idx = int(segment_record.critical_frame_index_10hz or 0)
    crit_ts = segment_record.critical_timestamp_sec
    if getattr(segment_record, "cam_times_20s", None) and len(segment_record.cam_times_20s) > crit_idx:
        crit_ts = segment_record.cam_times_20s[crit_idx]

    status = "success" if not warnings else "partial"
    summary = schema.build_segment_summary(
        seg_id=seg_base,
        status=status,
        errors=[],
        warnings=warnings,
        counts=counts,
        timing={},
        scenario_cluster=segment_record.scenario_cluster,
    )

    manifest_20s = schema.build_segment_manifest(
        is_critical_5s=False,
        seg_id=seg_base,
        split=segment_record.split,
        scenario_cluster=segment_record.scenario_cluster,
        routing_command=segment_record.routing_command,
        duration_sec=20.0,
        image_dir="images",
        calibration_file="meta/camera_calib.json",
        ego_state_file="trajectory/ego_state_20s_10hz.csv",
        ego_state_hz=10.0,
        ego_state_fields=ego_state_fields,
        steering_field=steering_field,
        ego_future=ego_future_info_20s,
        rater_labels=rater_labels_info_20s,
        camera_info={"fps_hz": 10.0, "num_cameras": 8, "camera_ids": list(range(1, 9))},
        time_info={
            "critical_timestamp_sec": crit_ts,
            "critical_frame_index_10hz": crit_idx,
        },
    )

    manifest_5s = schema.build_segment_manifest(
        is_critical_5s=True,
        seg_id=seg_base,
        split=segment_record.split,
        scenario_cluster=segment_record.scenario_cluster,
        routing_command=segment_record.routing_command,
        duration_sec=5.0,
        image_dir="images",
        calibration_file="meta/camera_calib.json",
        ego_state_file="trajectory/ego_state_0to5s_10hz.csv",
        ego_state_hz=10.0,
        ego_state_fields=ego_state_fields,
        steering_field=steering_field,
        ego_future=ego_future_info_5s if ego_future_available else {"available": False, "reason": ego_future_info_5s.get("reason", "")},
        rater_labels=rater_labels_info_5s,
        camera_info={"fps_hz": 10.0, "num_cameras": 8, "camera_ids": list(range(1, 9))},
        time_info={
            "parent_seg_id": seg_base,
            "critical_timestamp_in_parent_sec": crit_ts,
            "parent_frame_index_start_10hz": mapping["mapping"].parent_frame_index_start_10hz,
            "mapping_note": mapping["mapping"].mapping_note,
        },
    )

    # Write manifest/summary files
    write_json(seg_dir / "segment_manifest.json", manifest_20s)
    write_json(seg_dir / "segment_summary.json", summary)
    write_json(crit_dir / "segment_manifest.json", manifest_5s)
    write_json(crit_dir / "segment_summary.json", summary)

    _write_logs(log_dir, f"seg_id={segment_record.seg_id} status=partial note=pending_logic")

    return {
        "status": "partial",
        "seg_dir": str(seg_dir),
        "manifest_20s": manifest_20s,
        "manifest_5s": manifest_5s,
        "summary": summary,
    }
