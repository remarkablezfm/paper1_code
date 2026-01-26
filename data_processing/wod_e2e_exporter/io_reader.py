"""Input readers: TFRecord/raw data -> SegmentRecord."""

from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from google.protobuf.descriptor import FieldDescriptor
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset

from .utils import safe_float, win_to_wsl_path
from .time_align import build_camera_timegrid


@dataclass
class RaterTrajectory:
    index: int
    df: pd.DataFrame
    score: Optional[float]


@dataclass
class SegmentRecord:
    """Container for one 20s segment and critical moment annotations."""

    seg_id: str
    split: str
    scenario_cluster: str
    routing_command: str
    critical_frame_index_10hz: int
    critical_timestamp_sec: float
    frame: Any  # open_dataset.Frame
    past_states: Any
    future_states: Any
    rater_trajs: List[RaterTrajectory]
    rater_scores: List[Optional[float]]
    e2e_raw: wod_e2ed_pb2.E2EDFrame
    # new fields for full 20s aggregation
    frames_20s: List[Any]
    cam_times_20s: List[float]
    frame_ids: List[int] = field(default_factory=list)
    frame_ids_raw: List[Optional[int]] = field(default_factory=list)
    past_states_20s: List[Any] = field(default_factory=list)
    future_states_20s: List[Any] = field(default_factory=list)


# --------------------------
# Index helpers
# --------------------------
def split_seg_frame(seg_id: str) -> Tuple[str, Optional[int]]:
    """
    Split seg_id like 'abcd-058' into ('abcd', 58). If no numeric suffix, return (seg_id, None).
    """
    if not seg_id:
        return seg_id, None
    if "-" not in seg_id:
        return seg_id, None
    base, tail = seg_id.rsplit("-", 1)
    if tail.isdigit():
        try:
            return base, int(tail)
        except Exception:
            return base, None
    return seg_id, None


def read_seg_index_csv(index_csv: str) -> Dict[str, List[Tuple[str, int, str]]]:
    """
    Read seg_index CSV (columns: seg_id, tfrecord_file, record_idx) and group by base seg_id (strip '-NNN').
    Returns mapping base_seg_id -> list of (tfrecord_file, record_idx, seg_id_raw).
    """
    grouped: Dict[str, List[Tuple[str, int, str]]] = {}
    with open(index_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg_raw = row.get("seg_id", "")
            base, _ = split_seg_frame(seg_raw)
            tfp = win_to_wsl_path(row.get("tfrecord_file", ""))
            ridx = int(row.get("record_idx", 0))
            grouped.setdefault(base, []).append((tfp, ridx, seg_raw))
    # sort each list by (tfrecord, record_idx) for determinism
    for k in list(grouped.keys()):
        grouped[k] = sorted(grouped[k], key=lambda x: (x[0], int(x[1])))
    return grouped


def read_seg_index_enriched(index_csv: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read enriched seg_index CSV (seg_id, frame_id, tfrecord_file, record_idx, scenario_cluster)
    and group by seg_id (base).
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with open(index_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base = row.get("seg_id", "")
            grouped.setdefault(base, []).append({
                "seg_id": base,
                "frame_id": int(row["frame_id"]) if row.get("frame_id", "").isdigit() else None,
                "tfrecord_file": win_to_wsl_path(row.get("tfrecord_file", "")),
                "record_idx": int(row.get("record_idx", 0)),
                "scenario_cluster": row.get("scenario_cluster", "UNKNOWN"),
            })
    for k in list(grouped.keys()):
        grouped[k] = sorted(
            grouped[k],
            key=lambda r: (r.get("frame_id") if r.get("frame_id") is not None else 1e9, r.get("record_idx", 0)),
        )
    return grouped


def read_val_index_csv(val_index_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Read val index CSV and return mapping:
      seg_id -> {critical_frame_index_10hz, scenario_cluster, routing_command}
    """
    df = pd.read_csv(val_index_csv)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    def _pick(row: Dict[str, Any], names: List[str]) -> Any:
        for name in names:
            if name not in row:
                continue
            val = row[name]
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            if val == "":
                continue
            return val
        return None

    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        rd = r.to_dict()
        seg_raw = _pick(rd, ["seg_id", "segment_id", "sequence_id", "sequence_name"])
        key_idx = _pick(rd, ["key_idx", "key_index", "key_frame", "keyframe_idx"])
        if seg_raw is None or key_idx is None:
            continue
        try:
            key_i = int(float(key_idx))
        except Exception:
            continue
        base_seg, _ = split_seg_frame(str(seg_raw))

        scenario_cluster = _pick(rd, ["class", "class_name", "scenario_cluster"])
        intent = _pick(rd, ["intent", "routing_command", "route_command"])
        routing_command = str(intent) if intent is not None else None

        if base_seg in out:
            continue
        out[base_seg] = {
            "critical_frame_index_10hz": key_i,
            "scenario_cluster": str(scenario_cluster) if scenario_cluster is not None else None,
            "routing_command": routing_command,
        }
    return out


# --------------------------
# Protobuf helpers (from reference script)
# --------------------------
def parse_e2ed(raw_bytes: bytes) -> wod_e2ed_pb2.E2EDFrame:
    msg = wod_e2ed_pb2.E2EDFrame()
    msg.ParseFromString(raw_bytes)
    return msg


def _is_message(x) -> bool:
    return hasattr(x, "ListFields") and callable(getattr(x, "ListFields"))


def extract_xyz_recursive(msg, max_depth: int = 6) -> Optional[np.ndarray]:
    """
    Try to find repeated scalar fields pos_x/pos_y/pos_z or x/y/z somewhere inside msg.
    Return (N,3) float32 in vehicle frame.
    """
    if max_depth < 0 or (not _is_message(msg)):
        return None

    field_map = {fd.name: val for fd, val in msg.ListFields()}

    def _pack_xyz(xv, yv, zv):
        x = np.asarray(list(xv), dtype=np.float32)
        y = np.asarray(list(yv), dtype=np.float32)
        z = np.asarray(list(zv), dtype=np.float32) if zv is not None else np.zeros_like(x)
        if z.size == 0:
            z = np.zeros_like(x)
        n = min(len(x), len(y), len(z))
        if n <= 0:
            return None
        return np.stack([x[:n], y[:n], z[:n]], axis=1)

    if "pos_x" in field_map and "pos_y" in field_map:
        return _pack_xyz(field_map["pos_x"], field_map["pos_y"], field_map.get("pos_z", None))

    if "x" in field_map and "y" in field_map:
        return _pack_xyz(field_map["x"], field_map["y"], field_map.get("z", None))

    # descend
    for fd, val in msg.ListFields():
        if fd.type != FieldDescriptor.TYPE_MESSAGE:
            continue
        if fd.label == FieldDescriptor.LABEL_REPEATED:
            if len(val) == 0:
                continue
            cand = extract_xyz_recursive(val[0], max_depth=max_depth - 1)
            if cand is not None:
                return cand
        else:
            cand = extract_xyz_recursive(val, max_depth=max_depth - 1)
            if cand is not None:
                return cand

    return None


def extract_preference_trajectories(e2e: wod_e2ed_pb2.E2EDFrame) -> List[RaterTrajectory]:
    """
    Extract preference trajectories and scores.
    This mirrors export_package_fast_overlay_keymoment.py but keeps only data frames + scores.
    """
    out: List[RaterTrajectory] = []
    for i, pref in enumerate(getattr(e2e, "preference_trajectories", [])):
        xyz = extract_xyz_recursive(pref, max_depth=6)
        score = safe_float(getattr(pref, "preference_score", None))
        if xyz is None:
            df = pd.DataFrame(columns=["x", "y", "z"])
        else:
            df = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        out.append(RaterTrajectory(index=i, df=df, score=score))
    return out


def _pick_rater_trajs(
    rater_trajs_list: List[List[RaterTrajectory]],
    rater_scores_list: List[List[Optional[float]]],
    prefer_idx: Optional[int],
) -> Tuple[List[RaterTrajectory], List[Optional[float]]]:
    def _scores_valid(scores: List[Optional[float]]) -> bool:
        for s in scores:
            if s is None:
                continue
            if isinstance(s, float) and math.isnan(s):
                continue
            if float(s) != -1.0:
                return True
        return False

    def _trajs_have_points(trajs: List[RaterTrajectory]) -> bool:
        for t in trajs:
            df = getattr(t, "df", None)
            if df is not None and not df.empty:
                return True
        return False

    if not rater_trajs_list:
        return [], []
    if prefer_idx is not None and 0 <= prefer_idx < len(rater_trajs_list):
        scores = rater_scores_list[prefer_idx] if prefer_idx < len(rater_scores_list) else []
        if rater_trajs_list[prefer_idx] and _scores_valid(scores) and _trajs_have_points(rater_trajs_list[prefer_idx]):
            return rater_trajs_list[prefer_idx], scores
    for i, scores in enumerate(rater_scores_list):
        if _scores_valid(scores) and i < len(rater_trajs_list) and _trajs_have_points(rater_trajs_list[i]):
            return rater_trajs_list[i], scores
    for i, trajs in enumerate(rater_trajs_list):
        if trajs:
            scores = rater_scores_list[i] if i < len(rater_scores_list) else []
            return trajs, scores
    scores = rater_scores_list[0] if rater_scores_list else []
    return rater_trajs_list[0], scores


def _first_scored_rater_idx(rater_scores_list: List[List[Optional[float]]]) -> Optional[int]:
    for i, scores in enumerate(rater_scores_list):
        for s in scores:
            if s is None:
                continue
            if isinstance(s, float) and math.isnan(s):
                continue
            if float(s) != -1.0:
                return i
    return None


def _pick_route_command(e2e: wod_e2ed_pb2.E2EDFrame) -> str:
    # routing_command field name may vary; default to GO_STRAIGHT if missing.
    for cand in ["route_command", "routing_command", "intent"]:
        if hasattr(e2e, cand):
            val = getattr(e2e, cand)
            try:
                return str(val)
            except Exception:
                return "GO_STRAIGHT"
    return "GO_STRAIGHT"


def _pick_seg_id(frame: open_dataset.Frame) -> str:
    ctx = getattr(frame, "context", None)
    name = getattr(ctx, "name", None) if ctx else None
    return str(name) if name not in (None, "") else "unknown_seg"


def _pick_critical_idx(e2e: wod_e2ed_pb2.E2EDFrame) -> int:
    # prefer key_frame_index if exists, else 0
    for cand in ["key_frame_index", "key_idx", "critical_frame_index_10hz"]:
        if hasattr(e2e, cand):
            try:
                return int(getattr(e2e, cand))
            except Exception:
                continue
    return 0


def load_segment_from_tfrecord(
    tfrecord_path: str,
    record_idx: int,
    split: str = "val",
    scenario_cluster: str = "UNKNOWN",
    routing_command: Optional[str] = None,
) -> SegmentRecord:
    """
    Parse TFRecord by index, using export_package_fast_overlay_keymoment.py as parsing reference.

    Args:
        tfrecord_path: path to TFRecord file.
        record_idx: 0-based index within TFRecord.
        split: dataset split.
        scenario_cluster: provided externally (metadata).
        routing_command: optional override; if None, derive from proto if available.
    """
    tfrecord_path = win_to_wsl_path(tfrecord_path)
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    target = int(record_idx)

    for i, rec in enumerate(ds):
        if i != target:
            continue
        raw = bytes(rec.numpy())
        e2e = parse_e2ed(raw)
        frame = e2e.frame
        seg_id = _pick_seg_id(frame)
        base_seg, frame_id = split_seg_frame(seg_id)
        route_cmd = routing_command or _pick_route_command(e2e)
        critical_idx = _pick_critical_idx(e2e)
        ts_sec = float(getattr(frame, "timestamp_micros", 0) or 0) / 1e6

        prefs = extract_preference_trajectories(e2e)
        scores = [p.score for p in prefs]

        return SegmentRecord(
            seg_id=seg_id,
            split=str(split),
            scenario_cluster=str(scenario_cluster),
            routing_command=str(route_cmd),
            critical_frame_index_10hz=int(critical_idx),
            critical_timestamp_sec=ts_sec,
            frame=frame,
            past_states=getattr(e2e, "past_states", None),
            future_states=getattr(e2e, "future_states", None),
            rater_trajs=prefs,
            rater_scores=scores,
            e2e_raw=e2e,
            frames_20s=[frame],
            cam_times_20s=[0.0],
            frame_ids=[frame_id] if frame_id is not None else [],
            frame_ids_raw=[frame_id] if frame_id is not None else [],
            past_states_20s=[getattr(e2e, "past_states", None)],
            future_states_20s=[getattr(e2e, "future_states", None)],
        )

    raise IndexError(f"record_idx={record_idx} not found in TFRecord {tfrecord_path}")


def load_segment_from_index(
    seg_id: str,
    seg_rows: List[Tuple[str, str]],
    split: str = "val",
    scenario_cluster: str = "UNKNOWN",
    routing_command: Optional[str] = None,
) -> SegmentRecord:
    """
    Aggregate a segment from multiple (tfrecord_file, record_idx) entries.
    seg_rows: list of (tfrecord_path, record_idx) for this seg_id.
    """
    frames: List[Any] = []
    past_states_list: List[Any] = []
    future_states_list: List[Any] = []
    rater_trajs_list: List[List[RaterTrajectory]] = []
    rater_scores_list: List[List[Optional[float]]] = []
    frame_ids_raw: List[Optional[int]] = []
    first_seg: Optional[SegmentRecord] = None
    # sort by (tfrecord path, record_idx) to keep deterministic, but prefer numeric suffix order if available
    seg_rows_sorted = sorted(seg_rows, key=lambda x: (x[0], int(x[1])))
    for tfp, ridx in seg_rows_sorted:
        sr = load_segment_from_tfrecord(
            tfrecord_path=tfp,
            record_idx=int(ridx),
            split=split,
            scenario_cluster=scenario_cluster,
            routing_command=routing_command,
        )
        if first_seg is None:
            first_seg = sr
        frames.append(sr.frame)
        past_states_list.append(getattr(sr, "past_states", None))
        future_states_list.append(getattr(sr, "future_states", None))
        rater_trajs_list.append(getattr(sr, "rater_trajs", []) or [])
        rater_scores_list.append(getattr(sr, "rater_scores", []) or [])
        if getattr(sr, "frame_ids_raw", None):
            frame_ids_raw.append(getattr(sr, "frame_ids_raw")[0])

    if first_seg is None:
        raise ValueError(f"No rows for seg_id={seg_id}")

    first_seg.frames_20s = frames
    first_seg.cam_times_20s = build_camera_timegrid(len(frames), fps_hz=10.0).t_sec
    first_seg.critical_frame_index_10hz = first_seg.critical_frame_index_10hz or 0
    first_seg.past_states_20s = past_states_list
    first_seg.future_states_20s = future_states_list
    if frame_ids_raw:
        first_seg.frame_ids_raw = frame_ids_raw
    scored_idx = _first_scored_rater_idx(rater_scores_list)
    if scored_idx is not None:
        first_seg.critical_frame_index_10hz = scored_idx
        first_seg.critical_timestamp_sec = float(scored_idx) / 10.0
    selected_idx = scored_idx if scored_idx is not None else first_seg.critical_frame_index_10hz
    picked_trajs, picked_scores = _pick_rater_trajs(
        rater_trajs_list,
        rater_scores_list,
        selected_idx,
    )
    first_seg.rater_trajs = picked_trajs
    first_seg.rater_scores = picked_scores
    return first_seg


def load_segment_from_enriched_rows(
    seg_id: str,
    rows: List[Dict[str, Any]],
    split: str = "val",
    routing_command: Optional[str] = None,
    scenario_cluster: Optional[str] = None,
    critical_frame_index_10hz: Optional[int] = None,
) -> SegmentRecord:
    """
    Aggregate segment using enriched rows (with frame_id and scenario_cluster).
    rows: list of dicts with keys seg_id, frame_id, tfrecord_file, record_idx, scenario_cluster
    """
    if not rows:
        raise ValueError(f"No rows for seg_id={seg_id}")
    # sort by frame_id if available, else record_idx
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("frame_id") if r.get("frame_id") is not None else 1e9, r.get("record_idx", 0)),
    )
    scenario_cluster = scenario_cluster or rows_sorted[0].get("scenario_cluster", "UNKNOWN")
    frames: List[Any] = []
    past_states_list: List[Any] = []
    future_states_list: List[Any] = []
    rater_trajs_list: List[List[RaterTrajectory]] = []
    rater_scores_list: List[List[Optional[float]]] = []
    frame_ids: List[Optional[int]] = []
    frame_ids_raw: List[Optional[int]] = []
    first_seg: Optional[SegmentRecord] = None
    base_frame_id: Optional[int] = None

    for idx, r in enumerate(rows_sorted):
        sr = load_segment_from_tfrecord(
            tfrecord_path=r["tfrecord_file"],
            record_idx=int(r["record_idx"]),
            split=split,
            scenario_cluster=scenario_cluster,
            routing_command=routing_command,
        )
        if first_seg is None:
            first_seg = sr
        frames.append(sr.frame)
        past_states_list.append(getattr(sr, "past_states", None))
        future_states_list.append(getattr(sr, "future_states", None))
        rater_trajs_list.append(getattr(sr, "rater_trajs", []) or [])
        rater_scores_list.append(getattr(sr, "rater_scores", []) or [])
        fid = r.get("frame_id")
        if isinstance(fid, int):
            if base_frame_id is None:
                base_frame_id = fid
            rel_fid = fid - base_frame_id
            frame_ids.append(rel_fid)
            frame_ids_raw.append(fid)
        else:
            frame_ids.append(None)
            frame_ids_raw.append(None)
    if first_seg is None:
        raise ValueError(f"No rows parsed for seg_id={seg_id}")

    first_seg.frames_20s = frames
    first_seg.cam_times_20s = build_camera_timegrid(len(frames), fps_hz=10.0).t_sec
    first_seg.frame_ids = frame_ids
    first_seg.frame_ids_raw = frame_ids_raw
    first_seg.past_states_20s = past_states_list
    first_seg.future_states_20s = future_states_list
    first_seg.scenario_cluster = scenario_cluster
    selected_idx: Optional[int] = None
    if critical_frame_index_10hz is not None:
        crit_idx = int(critical_frame_index_10hz)
        if base_frame_id is not None:
            crit_idx = crit_idx - base_frame_id
        if crit_idx < 0:
            crit_idx = 0
        first_seg.critical_frame_index_10hz = crit_idx
        first_seg.critical_timestamp_sec = float(crit_idx) / 10.0
        selected_idx = crit_idx
    scored_idx = _first_scored_rater_idx(rater_scores_list)
    if scored_idx is not None:
        first_seg.critical_frame_index_10hz = scored_idx
        first_seg.critical_timestamp_sec = float(scored_idx) / 10.0
        selected_idx = scored_idx
    picked_trajs, picked_scores = _pick_rater_trajs(
        rater_trajs_list,
        rater_scores_list,
        selected_idx,
    )
    first_seg.rater_trajs = picked_trajs
    first_seg.rater_scores = picked_scores
    return first_seg


def load_segment_from_cache_dict(
    cache_dict: Dict[str, Any],
    split: str = "val",
    scenario_cluster: Optional[str] = None,
    routing_command: Optional[str] = None,
    critical_frame_index_10hz: Optional[int] = None,
) -> SegmentRecord:
    """
    Build SegmentRecord from val_cache dict (with e2e_bytes).
    Expected keys: seg_id, scenario_cluster, frame_ids, e2e_bytes.
    """
    if not isinstance(cache_dict, dict):
        raise TypeError("cache_dict must be a dict")
    e2e_bytes = cache_dict.get("e2e_bytes", [])
    if not e2e_bytes:
        raise ValueError("cache_dict missing e2e_bytes")

    frames: List[Any] = []
    past_states_list: List[Any] = []
    future_states_list: List[Any] = []
    rater_trajs_list: List[List[RaterTrajectory]] = []
    rater_scores_list: List[List[Optional[float]]] = []

    first_seg: Optional[SegmentRecord] = None

    for raw in e2e_bytes:
        e2e = parse_e2ed(raw)
        frame = e2e.frame
        if first_seg is None:
            seg_id = cache_dict.get("seg_id") or _pick_seg_id(frame)
            route_cmd = routing_command or _pick_route_command(e2e)
            critical_idx = critical_frame_index_10hz
            if critical_idx is None:
                critical_idx = cache_dict.get("critical_frame_index_10hz")
            if critical_idx is None:
                critical_idx = _pick_critical_idx(e2e)
            ts_sec = float(getattr(frame, "timestamp_micros", 0) or 0) / 1e6
            scen = scenario_cluster or cache_dict.get("scenario_cluster") or "UNKNOWN"
            prefs = extract_preference_trajectories(e2e)
            scores = [p.score for p in prefs]
            first_seg = SegmentRecord(
                seg_id=str(seg_id),
                split=str(split),
                scenario_cluster=str(scen),
                routing_command=str(route_cmd),
                critical_frame_index_10hz=int(critical_idx),
                critical_timestamp_sec=ts_sec,
                frame=frame,
                past_states=getattr(e2e, "past_states", None),
                future_states=getattr(e2e, "future_states", None),
                rater_trajs=prefs,
                rater_scores=scores,
                e2e_raw=e2e,
                frames_20s=[frame],
                cam_times_20s=[0.0],
                frame_ids=[],
                frame_ids_raw=[],
                past_states_20s=[getattr(e2e, "past_states", None)],
                future_states_20s=[getattr(e2e, "future_states", None)],
            )

        frames.append(frame)
        past_states_list.append(getattr(e2e, "past_states", None))
        future_states_list.append(getattr(e2e, "future_states", None))
        prefs = extract_preference_trajectories(e2e)
        rater_trajs_list.append(prefs)
        rater_scores_list.append([p.score for p in prefs])

    if first_seg is None:
        raise ValueError("no frames parsed from cache_dict")

    first_seg.frames_20s = frames
    first_seg.cam_times_20s = build_camera_timegrid(len(frames), fps_hz=10.0).t_sec
    first_seg.past_states_20s = past_states_list
    first_seg.future_states_20s = future_states_list

    frame_ids_raw = cache_dict.get("frame_ids")
    if isinstance(frame_ids_raw, list):
        first_seg.frame_ids_raw = frame_ids_raw
        base = None
        frame_ids = []
        for fid in frame_ids_raw:
            if fid is None:
                frame_ids.append(None)
                continue
            if base is None:
                base = fid
            frame_ids.append(fid - base)
        first_seg.frame_ids = frame_ids

    scored_idx = _first_scored_rater_idx(rater_scores_list)
    if scored_idx is not None:
        first_seg.critical_frame_index_10hz = scored_idx
        first_seg.critical_timestamp_sec = float(scored_idx) / 10.0
    selected_idx = scored_idx if scored_idx is not None else first_seg.critical_frame_index_10hz
    picked_trajs, picked_scores = _pick_rater_trajs(
        rater_trajs_list,
        rater_scores_list,
        selected_idx,
    )
    first_seg.rater_trajs = picked_trajs
    first_seg.rater_scores = picked_scores
    return first_seg
