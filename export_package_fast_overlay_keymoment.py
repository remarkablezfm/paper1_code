#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_package_fast.py

Purpose
-------
Export WOD-E2E validation "key moments" packages for ONE class (e.g., "Cut_ins", "Special Vehicles").

This script is designed to be:
- Correct: keep all expected exported files per keyframe.
- Fast(er): do NOT scan all TFRecords. It uses `tfrecord_file` + `record_idx` from `val_index_filled.csv`,
  and processes only TFRecord files that actually contain the target class. Each TFRecord is streamed once and
  only required record indices are parsed. Optional multi-worker parallelization across TFRecord files.

Expected input
--------------
val_index_filled.csv should contain at least:
- class / class_name
- seg_id
- key_idx
- tfrecord_file (absolute path or relative to dataset_root)
- record_idx (0-based within that TFRecord)
Optional:
- intent
- pref_scores (e.g., "[9.0, 6.0, 10.0]")

Output structure (for each segment)
-----------------------------------
<out_root>/
  <ClassName>_<seg_id>/
    segment_manifest.json
    segment_ego_timeseries.csv
    segment_summary.json
    <key_idx>/
      frame.json
      cam_1.jpg ... cam_8.jpg   (whatever exists in record)
      ego_past.csv
      ego_future.csv
      pref_traj_0.csv ...
      overlay_front.jpg
      overlay_rear.jpg
      overlay_meta.json

Plus a class-level export index:
<out_root>/export_index.csv

Dependencies
------------
- python>=3.8
- tensorflow (same env you used for waymo-open-dataset)
- waymo-open-dataset
- numpy, pandas, matplotlib
"""

import argparse
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# headless matplotlib (important for WSL / server)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from google.protobuf.descriptor import FieldDescriptor
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset


# --------------------------
# Small utilities
# --------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def win_to_wsl_path(p: str) -> str:
    """Convert Windows path like D:\Datasets\... to /mnt/d/Datasets/... if running in WSL."""
    if p is None:
        return p
    p = str(p)
    if p.startswith("/mnt/"):
        return p
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def parse_pref_scores(s: str) -> List[float]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    # allow either json-like list or plain string
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [float(x) for x in v]
    except Exception:
        pass
    # fallback: split by comma
    s2 = s.strip("[]() ")
    if not s2:
        return []
    out = []
    for part in s2.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    return out


# --------------------------
# Protobuf parsing
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


def ego_states_xyz(states_msg, fallback_z: float = 0.0) -> Optional[np.ndarray]:
    """
    For EgoTrajectoryStates in WOD-E2E: often repeated scalar fields like pos_x/pos_y/(pos_z).
    Return (N,3).
    """
    if states_msg is None:
        return None
    if not (hasattr(states_msg, "pos_x") and hasattr(states_msg, "pos_y")):
        return None

    x = np.asarray(list(getattr(states_msg, "pos_x", [])), dtype=np.float32)
    y = np.asarray(list(getattr(states_msg, "pos_y", [])), dtype=np.float32)
    z = np.asarray(list(getattr(states_msg, "pos_z", [])), dtype=np.float32)

    if len(x) == 0 or len(y) == 0:
        return None

    if len(z) == 0:
        z = np.full((min(len(x), len(y)),), float(fallback_z), dtype=np.float32)

    n = min(len(x), len(y), len(z))
    return np.stack([x[:n], y[:n], z[:n]], axis=1)


def ego_states_kine_df(states_msg) -> pd.DataFrame:
    """
    Export all common kinematic repeated arrays if present.
    Columns: pos_x,pos_y,pos_z, vel_x,vel_y, accel_x,accel_y
    (not all exist for both past/future)
    """
    cols = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y"]
    data: Dict[str, List[float]] = {}
    lens = []
    for c in cols:
        if hasattr(states_msg, c):
            arr = list(getattr(states_msg, c))
            data[c] = arr
            lens.append(len(arr))
    if not data:
        return pd.DataFrame()

    # align lengths (min non-zero)
    nonzero = [l for l in lens if l > 0]
    n = min(nonzero) if nonzero else 0
    for k in list(data.keys()):
        if len(data[k]) == 0:
            data.pop(k)
        else:
            data[k] = data[k][:n]
    return pd.DataFrame(data)


def extract_preference_trajectories(e2e: wod_e2ed_pb2.E2EDFrame) -> List[Dict[str, Any]]:
    out = []
    for i, pref in enumerate(getattr(e2e, "preference_trajectories", [])):
        xyz = extract_xyz_recursive(pref, max_depth=6)
        score = safe_float(getattr(pref, "preference_score", None))
        if xyz is None:
            df = pd.DataFrame(columns=["x", "y", "z"])
        else:
            df = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        out.append({"index": i, "preference_score": score, "xyz": xyz, "df": df})
    return out


# --------------------------
# Camera utilities
# --------------------------
def get_image_bytes_by_cam_id(frame, cam_id: int) -> Optional[bytes]:
    cam_id = int(cam_id)
    for im in frame.images:
        if int(im.name) == cam_id:
            return bytes(im.image)
    return None


def get_camera_calib_by_cam_id(frame, cam_id: int):
    cam_id = int(cam_id)
    for c in frame.context.camera_calibrations:
        if int(c.name) == cam_id:
            return c
    return None


def list_available_cam_ids(frame) -> List[int]:
    return sorted({int(im.name) for im in frame.images})


# --------------------------
# Manual projection (user-verified)
def calib_to_mats(calib):
    """Return fx, fy, cx, cy, T_vehicle_from_cam(4x4). (extrinsic is camera->vehicle)"""
    intr = list(calib.intrinsic)
    fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
    T_v_from_c = np.array(list(calib.extrinsic.transform), dtype=np.float32).reshape(4, 4)  # camera->vehicle
    return fx, fy, cx, cy, T_v_from_c


def vehicle_xyz_to_cam_xyz(xyz_vehicle: np.ndarray, T_vehicle_from_cam: np.ndarray) -> np.ndarray:
    """
    xyz_vehicle: (N,3) in vehicle FLU (x forward, y left, z up)
    T_vehicle_from_cam: camera->vehicle
    returns xyz_cam: (N,3) in camera frame
    """
    xyz_vehicle = np.asarray(xyz_vehicle, dtype=np.float32)
    if xyz_vehicle.ndim != 2 or xyz_vehicle.shape[1] != 3:
        raise ValueError(f"xyz_vehicle must be (N,3), got {xyz_vehicle.shape}")
    N = xyz_vehicle.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    P_v = np.concatenate([xyz_vehicle, ones], axis=1)  # (N,4)

    # convert vehicle->camera
    T_c_from_v = np.linalg.inv(np.asarray(T_vehicle_from_cam, dtype=np.float32))  # vehicle->camera
    P_c = (T_c_from_v @ P_v.T).T  # (N,4)
    return P_c[:, :3]


def project_cam_xyz_to_uv(
    xyz_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float = 0.5,
):
    """
    Waymo camera frame convention used here:
      x: depth (forward out of lens), y: left, z: up
    Projection:
      u = cx - fx*(y/x)
      v = cy - fy*(z/x)
    Returns:
      uv: (N,2), mask: (N,) depth validity mask (x > min_depth)
    """
    xyz_cam = np.asarray(xyz_cam, dtype=np.float32)
    x = xyz_cam[:, 0]  # depth
    y = xyz_cam[:, 1]
    z = xyz_cam[:, 2]
    mask = x > float(min_depth)

    x_safe = np.clip(x, 1e-6, None)
    u = cx - fx * (y / x_safe)
    v = cy - fy * (z / x_safe)
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv, mask


def project_vehicle_xyz_to_uv(xyz_vehicle: np.ndarray, calib, min_depth: float = 0.5):
    fx, fy, cx, cy, T_v_from_c = calib_to_mats(calib)
    xyz_cam = vehicle_xyz_to_cam_xyz(xyz_vehicle, T_v_from_c)  # vehicle->camera inside
    return project_cam_xyz_to_uv(xyz_cam, fx, fy, cx, cy, min_depth=min_depth)


def in_bounds_mask(uv: np.ndarray, H: int, W: int) -> np.ndarray:
    return (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)


def decode_jpeg_np(jpeg_bytes: bytes) -> np.ndarray:
    return tf.io.decode_jpeg(jpeg_bytes).numpy()


def save_overlay(
    rgb: np.ndarray,
    calib,
    ego_past: Optional[np.ndarray],
    ego_future: Optional[np.ndarray],
    pref_items: List[Dict[str, Any]],
    title: str,
    out_path: Path,
    min_depth: float = 0.05,
) -> Dict[str, Any]:
    """Render overlay image (ego past/future + preference trajectories) using manual projection."""
    H, W = rgb.shape[:2]
    meta: Dict[str, Any] = {
        "image_shape": [int(H), int(W), int(rgb.shape[2]) if rgb.ndim == 3 else 1],
        "min_depth": float(min_depth),
        "visible": {},
        "errors": [],
    }

    def _count_visible(xyz: Optional[np.ndarray]) -> int:
        if xyz is None or len(xyz) < 2:
            return 0
        uv, m = project_vehicle_xyz_to_uv(xyz, calib, min_depth=min_depth)
        m = m & in_bounds_mask(uv, H, W)
        return int(m.sum())

    try:
        meta["visible"]["ego_past"] = _count_visible(ego_past)
        meta["visible"]["ego_future"] = _count_visible(ego_future)
        meta["visible"]["pref"] = [_count_visible(pi.get("xyz", None)) for pi in pref_items]
    except Exception as e:
        meta["errors"].append(f"visibility_error:{type(e).__name__}:{e}")

    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    # keyframe origin at vehicle frame (0,0,0)
    try:
        origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        uv0, m0 = project_vehicle_xyz_to_uv(origin, calib, min_depth=min_depth)
        if bool(m0[0]) and bool(in_bounds_mask(uv0, H, W)[0]):
            ax.scatter([uv0[0, 0]], [uv0[0, 1]], s=90, marker="*", label="keyframe (0,0,0)")
    except Exception as e:
        meta["errors"].append(f"origin_error:{type(e).__name__}:{e}")

    def _draw_poly(xyz: Optional[np.ndarray], marker: str, linewidth: int, label: str):
        if xyz is None or len(xyz) < 2:
            return
        uv, m = project_vehicle_xyz_to_uv(xyz, calib, min_depth=min_depth)
        m = m & in_bounds_mask(uv, H, W)
        if not bool(m.any()):
            return
        ax.plot(uv[m, 0], uv[m, 1], marker=marker, linewidth=linewidth, label=label)

    try:
        _draw_poly(ego_past, marker=".", linewidth=2, label="ego_past")
        _draw_poly(ego_future, marker="x", linewidth=3, label="ego_future")

        for i, pi in enumerate(pref_items):
            xyz = pi.get("xyz", None)
            score = pi.get("preference_score", None)
            if score is None:
                lab = f"pref{i}"
            else:
                try:
                    lab = f"pref{i} score={float(score):.0f}"
                except Exception:
                    lab = f"pref{i} score={score}"
            _draw_poly(xyz, marker="o", linewidth=2, label=lab)
    except Exception as e:
        meta["errors"].append(f"draw_error:{type(e).__name__}:{e}")

    # Avoid warning if nothing was drawn
    handles, labs = ax.get_legend_handles_labels()
    if labs:
        ax.legend()

    plt.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    return meta


def has_meaningful_rater_score(scores: List[Any]) -> bool:
    """Whether this keyframe should render overlays.

    In WOD-E2E val, many frames have rater "no preference" scores (typically -1).
    We only render overlay images for frames that have at least one meaningful
    (non -1) rater score.
    """
    for s in scores or []:
        try:
            v = float(s)
        except Exception:
            continue
        if math.isnan(v):
            continue
        if v != -1.0:
            return True
    return False

# --------------------------
# Pose / kinematics summary
# --------------------------
def pose_transform_to_4x4(pose_msg) -> Optional[np.ndarray]:
    if pose_msg is None or not hasattr(pose_msg, "transform"):
        return None
    arr = np.asarray(list(pose_msg.transform), dtype=np.float32)
    if arr.size != 16:
        return None
    return arr.reshape(4, 4)


def yaw_from_T(T: np.ndarray) -> float:
    # yaw around Z from rotation matrix (assuming vehicle frame x-forward, y-left, z-up)
    R = T[:3, :3]
    return float(math.atan2(R[1, 0], R[0, 0]))


def current_kine_from_past(past_states) -> Dict[str, Any]:
    """
    Approx current (at key moment) using last element of past arrays, if present.
    """
    out: Dict[str, Any] = {}
    for name in ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y"]:
        if hasattr(past_states, name):
            arr = list(getattr(past_states, name))
            if len(arr) > 0:
                out[name] = float(arr[-1])
    vx = out.get("vel_x", None)
    vy = out.get("vel_y", None)
    if vx is not None and vy is not None:
        out["speed"] = float(math.sqrt(vx * vx + vy * vy))
    ax = out.get("accel_x", None)
    ay = out.get("accel_y", None)
    if ax is not None and ay is not None:
        out["accel"] = float(math.sqrt(ax * ax + ay * ay))
    return out


# --------------------------
# Index reading
# --------------------------
@dataclass
class IndexRow:
    class_name: str
    seg_id: str
    key_idx: int
    tfrecord_file: str
    record_idx: int
    intent: Optional[int] = None
    pref_scores: Optional[str] = None


def _pick_col(row: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return None


def read_val_index(val_index_csv: str) -> List[IndexRow]:
    rows: List[IndexRow] = []
    df = pd.read_csv(val_index_csv)

    # normalize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    for _, r in df.iterrows():
        rd = r.to_dict()

        class_name = _pick_col(rd, ["class_name", "class", "Class", "scenario_cluster"])
        seg_id = _pick_col(rd, ["seg_id", "segment_id", "sequence_id", "sequence_name"])
        key_idx = _pick_col(rd, ["key_idx", "key_index", "key_frame", "keyframe_idx"])
        tfrecord_file = _pick_col(rd, ["tfrecord_file", "tfrecord", "file", "path"])
        record_idx = _pick_col(rd, ["record_idx", "record", "record_index", "idx"])

        if class_name is None or seg_id is None or key_idx is None or tfrecord_file is None or record_idx is None:
            continue

        try:
            key_i = int(key_idx)
            rec_i = int(record_idx)
        except Exception:
            continue

        intent = _pick_col(rd, ["intent", "command"])
        intent_i = int(intent) if intent not in (None, "", np.nan) else None

        pref_scores = _pick_col(rd, ["pref_scores", "scores", "rater_scores"])

        rows.append(IndexRow(
            class_name=str(class_name),
            seg_id=str(seg_id),
            key_idx=key_i,
            tfrecord_file=str(tfrecord_file),
            record_idx=rec_i,
            intent=intent_i,
            pref_scores=str(pref_scores) if pref_scores is not None and str(pref_scores) != "nan" else None,
        ))
    return rows


def normalize_class_name_for_folder(class_name: str) -> str:
    # Keep the user's original class string for filtering, but make a safe folder name.
    return class_name.replace("/", "_").replace("\\", "_").replace(" ", "_")


# --------------------------
# Core export per record
# --------------------------
def export_one_keyframe(
    e2e: wod_e2ed_pb2.E2EDFrame,
    row: IndexRow,
    seg_dir: Path,
    front_cam_id: int,
    rear_cam_id: int,
) -> Dict[str, Any]:
    """
    Export all artifacts for this keyframe into seg_dir/<key_idx>/...
    Return small row for class-level export index.
    """
    key_idx = int(row.key_idx)
    key_dir = seg_dir / str(key_idx)
    ensure_dir(key_dir)

    frame = e2e.frame
    ctx_name = getattr(frame.context, "name", None)
    intent_val = int(getattr(e2e, "intent", -1)) if hasattr(e2e, "intent") else -1
    if row.intent is not None:
        intent_val = int(row.intent)

    # ---- save frame.json (light) ----
    frame_json = {
        "class_name": row.class_name,
        "seg_id": row.seg_id,
        "key_idx": key_idx,
        "context_name": ctx_name,
        "intent": intent_val,
        "timestamp_micros": int(getattr(frame, "timestamp_micros", 0) or 0),
        "available_cam_ids": list_available_cam_ids(frame),
        "front_cam_id": int(front_cam_id),
        "rear_cam_id": int(rear_cam_id),
    }
    with open(key_dir / "frame.json", "w", encoding="utf-8") as f:
        json.dump(frame_json, f, ensure_ascii=False, indent=2)

    # ---- save all camera JPEGs (raw bytes, no decode) ----
    for cid in list_available_cam_ids(frame):
        jpeg_bytes = get_image_bytes_by_cam_id(frame, cid)
        if jpeg_bytes is None:
            continue
        out_jpg = key_dir / f"cam_{cid}.jpg"
        with open(out_jpg, "wb") as f:
            f.write(jpeg_bytes)

    # ---- kinematics CSVs ----
    past_df = ego_states_kine_df(getattr(e2e, "past_states", None))
    future_df = ego_states_kine_df(getattr(e2e, "future_states", None))
    past_df.to_csv(key_dir / "ego_past.csv", index=False)
    future_df.to_csv(key_dir / "ego_future.csv", index=False)

    ego_past_xyz_arr = ego_states_xyz(getattr(e2e, "past_states", None), fallback_z=0.0)
    ego_future_xyz_arr = ego_states_xyz(getattr(e2e, "future_states", None), fallback_z=0.0)

    # ---- preference trajectories ----
    pref_items = extract_preference_trajectories(e2e)
    # Scores: prefer index CSV if present (it may be authoritative for some pipelines)
    scores_from_index = parse_pref_scores(row.pref_scores) if row.pref_scores else []
    scores_from_proto = [pi.get("preference_score", None) for pi in pref_items]
    scores = scores_from_index if len(scores_from_index) > 0 else scores_from_proto

    for pi in pref_items:
        i = int(pi["index"])
        pi["df"].to_csv(key_dir / f"pref_traj_{i}.csv", index=False)

    score_vals = [s for s in scores if s is not None and not (isinstance(s, float) and math.isnan(s))]
    score_max = float(np.max(score_vals)) if score_vals else float("nan")
    score_mean = float(np.mean(score_vals)) if score_vals else float("nan")

    # ---- overlays (front & rear) ----
    overlay_meta: Dict[str, Any] = {
        "front_cam_id": str(front_cam_id),
        "rear_cam_id": str(rear_cam_id),
        "camera_ops_available": False,  # we intentionally use manual projection
        "projection": {},
        "drawn_trajectories": [],
        "note": "",
        "scores": [float(s) if s is not None and not (isinstance(s, float) and math.isnan(s)) else None for s in scores],
    }

    # Only render overlay images for keyframes that have meaningful (non -1) rater score(s).
    # NOTE: We still write overlay_meta.json for every keyframe to keep the package structure stable.
    should_render_overlay = has_meaningful_rater_score(scores)

    def _render(cam_id: int, out_name: str,
                ego_past_for_cam: Optional[np.ndarray],
                ego_future_for_cam: Optional[np.ndarray],
                pref_items_for_cam: List[Dict[str, Any]]):
        jpeg = get_image_bytes_by_cam_id(frame, cam_id)
        calib = get_camera_calib_by_cam_id(frame, cam_id)
        if jpeg is None or calib is None:
            overlay_meta["projection"][str(cam_id)] = [["all", "missing_rgb_or_calib"]]
            return
        rgb = decode_jpeg_np(jpeg)
        meta = save_overlay(
            rgb=rgb,
            calib=calib,
            ego_past=ego_past_for_cam,
            ego_future=ego_future_for_cam,
            pref_items=pref_items_for_cam,
            title=f"{row.seg_id} key_idx={key_idx} | cam{cam_id} overlay",
            out_path=key_dir / out_name,
            min_depth=0.05,
        )
        overlay_meta["projection"][str(cam_id)] = meta.get("errors", []) if meta.get("errors") else []
        overlay_meta["drawn_trajectories"].append({
            "cam_id": str(cam_id),
            "visible": meta.get("visible", {}),
        })

    if should_render_overlay:
        # FRONT: draw 4 trajectories (ego past + ego future + 3 preference trajectories)
        try:
            _render(int(front_cam_id), "overlay_front.jpg",
                    ego_past_for_cam=ego_past_xyz_arr,
                    ego_future_for_cam=ego_future_xyz_arr,
                    pref_items_for_cam=pref_items)
        except Exception as e:
            overlay_meta["projection"][str(front_cam_id)] = [f"projection_error:{type(e).__name__}:{e}"]

        # REAR: only draw ego past (per user's VLM-oriented requirement)
        try:
            _render(int(rear_cam_id), "overlay_rear.jpg",
                    ego_past_for_cam=ego_past_xyz_arr,
                    ego_future_for_cam=None,
                    pref_items_for_cam=[])
        except Exception as e:
            overlay_meta["projection"][str(rear_cam_id)] = [f"projection_error:{type(e).__name__}:{e}"]
    else:
        overlay_meta["note"] = "skipped_overlay: no meaningful rater score (all -1)"
        overlay_meta["projection"][str(front_cam_id)] = [["all", "skipped"]]
        overlay_meta["projection"][str(rear_cam_id)] = [["all", "skipped"]]

    with open(key_dir / "overlay_meta.json", "w", encoding="utf-8") as f:
        json.dump(overlay_meta, f, ensure_ascii=False, indent=2)

    # row returned for class index + segment manifest
    return {
        "class_name": row.class_name,
        "seg_id": row.seg_id,
        "key_idx": key_idx,
        "intent": intent_val,
        "score_max": score_max,
        "score_mean": score_mean,
        "key_folder": str(key_dir),
    }


def process_tfrecord_file(
    tfrecord_path: str,
    rows_for_file: List[IndexRow],
    out_root: Path,
    class_name: str,
    front_cam_id: int,
    rear_cam_id: int,
    dataset_root: str,
) -> List[Dict[str, Any]]:
    """
    Process ONE TFRecord file: stream it once, parse only needed record indices.
    Return exported rows for class-level index.
    """
    exported: List[Dict[str, Any]] = []

    # map record_idx -> row(s). In practice there should be 1 row per record_idx.
    idx2rows: Dict[int, List[IndexRow]] = {}
    for r in rows_for_file:
        idx2rows.setdefault(int(r.record_idx), []).append(r)

    needed = set(idx2rows.keys())
    if not needed:
        return exported

    # resolve tfrecord path
    tfp = tfrecord_path
    tfp = win_to_wsl_path(tfp)
    if not os.path.isabs(tfp):
        tfp = os.path.join(win_to_wsl_path(dataset_root), tfp)
    tfp = tfp.replace("\\", "/")

    if not os.path.exists(tfp):
        raise FileNotFoundError(f"TFRecord not found: {tfp}")

    max_needed = max(needed)

    ds = tf.data.TFRecordDataset(tfp, compression_type="")
    found = set()

    for i, rec in enumerate(ds):
        if i > max_needed and len(found) == len(needed):
            break
        if i not in needed:
            continue

        raw = bytes(rec.numpy())
        e2e = parse_e2ed(raw)

        # each record belongs to exactly one segment folder based on seg_id
        for row in idx2rows[i]:
            seg_dir = out_root / f"{normalize_class_name_for_folder(class_name)}_{row.seg_id}"
            ensure_dir(seg_dir)
            exported_row = export_one_keyframe(
                e2e=e2e,
                row=row,
                seg_dir=seg_dir,
                front_cam_id=front_cam_id,
                rear_cam_id=rear_cam_id,
            )
            exported.append(exported_row)

        found.add(i)
        if len(found) == len(needed):
            # still keep going? no need
            if i >= max_needed:
                break

    return exported


def finalize_segment_summaries(out_root: Path, class_name: str) -> None:
    """
    Walk segment folders and create:
    - segment_manifest.json
    - segment_ego_timeseries.csv
    - segment_summary.json
    This is done after keyframes are exported, so we can compute per-segment stats.
    """
    class_prefix = f"{normalize_class_name_for_folder(class_name)}_"
    for seg_dir in sorted(out_root.glob(f"{class_prefix}*")):
        if not seg_dir.is_dir():
            continue

        key_dirs = sorted([p for p in seg_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
        manifest: Dict[str, Any] = {
            "class_name": class_name,
            "seg_id": seg_dir.name[len(class_prefix):],
            "keyframes": [],
        }

        timeseries_rows: List[Dict[str, Any]] = []

        for kd in key_dirs:
            frame_json_path = kd / "frame.json"
            if not frame_json_path.exists():
                continue
            try:
                fr = json.loads(frame_json_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            key_idx = int(fr.get("key_idx", int(kd.name)))
            intent = fr.get("intent", None)

            # pose
            pose_tx = pose_ty = pose_tz = pose_yaw = None
            # pose transform is not stored in frame.json; read directly from TFRecord would be expensive here.
            # We'll approximate kinematics from ego_past.csv (last row), which is sufficient for "ego running" summary.
            past_csv = kd / "ego_past.csv"
            vx = vy = ax = ay = speed = accel = None
            if past_csv.exists():
                try:
                    dfp = pd.read_csv(past_csv)
                    if len(dfp) > 0:
                        last = dfp.iloc[-1].to_dict()
                        vx = safe_float(last.get("vel_x", None))
                        vy = safe_float(last.get("vel_y", None))
                        ax = safe_float(last.get("accel_x", None))
                        ay = safe_float(last.get("accel_y", None))
                        if vx is not None and vy is not None:
                            speed = float(math.sqrt(vx * vx + vy * vy))
                        if ax is not None and ay is not None:
                            accel = float(math.sqrt(ax * ax + ay * ay))
                except Exception:
                    pass

            # score stats
            score_max = score_mean = None
            # overlay_meta contains visible but not scores; use pref_scores in pref_traj? Not available.
            # We'll read from export_index.csv later for class-level, but for per-segment timeseries we can read pref_scores from frame.json? not stored.
            # So leave score fields blank here; downstream can join with export_index.csv if needed.

            ts_row = {
                "key_idx": int(key_idx),
                "intent": int(intent) if intent is not None and str(intent) != "nan" else None,
                "pose_tx": pose_tx,
                "pose_ty": pose_ty,
                "pose_tz": pose_tz,
                "pose_yaw": pose_yaw,
                "vx": vx,
                "vy": vy,
                "ax": ax,
                "ay": ay,
                "speed": speed,
                "accel": accel,
            }
            timeseries_rows.append(ts_row)

            manifest["keyframes"].append({
                "key_idx": key_idx,
                "folder": kd.name,
                "frame_json": str(frame_json_path.name),
                "has_overlay_front": (kd / "overlay_front.jpg").exists(),
                "has_overlay_rear": (kd / "overlay_rear.jpg").exists(),
            })

        ts_df = pd.DataFrame(timeseries_rows).sort_values("key_idx") if timeseries_rows else pd.DataFrame()
        ts_csv = seg_dir / "segment_ego_timeseries.csv"
        ts_df.to_csv(ts_csv, index=False)

        def _stats(col: str) -> Dict[str, Any]:
            if col not in ts_df.columns or len(ts_df) == 0:
                return {}
            s = pd.to_numeric(ts_df[col], errors="coerce").dropna()
            if len(s) == 0:
                return {}
            return {
                "mean": float(s.mean()),
                "max": float(s.max()),
                "min": float(s.min()),
                "p95": float(s.quantile(0.95)),
            }

        summary = {
            "class_name": class_name,
            "seg_id": manifest["seg_id"],
            "n_keyframes": len(manifest["keyframes"]),
            "key_idx_min": int(ts_df["key_idx"].min()) if len(ts_df) else None,
            "key_idx_max": int(ts_df["key_idx"].max()) if len(ts_df) else None,
            "speed_stats": _stats("speed"),
            "accel_stats": _stats("accel"),
        }

        with open(seg_dir / "segment_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        with open(seg_dir / "segment_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# --------------------------
# Main export
# --------------------------
def export_class(
    dataset_root: str,
    val_index_csv: str,
    out_root: str,
    class_name: str,
    front_cam_id: int,
    rear_cam_id: int,
    num_workers: int,
) -> None:
    dataset_root = win_to_wsl_path(dataset_root)
    val_index_csv = win_to_wsl_path(val_index_csv)
    out_root = win_to_wsl_path(out_root)

    out_root_p = Path(out_root)
    ensure_dir(out_root_p)

    rows = read_val_index(val_index_csv)
    seg_rows = [r for r in rows if str(r.class_name) == str(class_name)]
    if len(seg_rows) == 0:
        print(f"[WARN] No rows for class_name='{class_name}'. Check spelling. Available examples: {sorted({r.class_name for r in rows})[:10]}")
        return

    # group by tfrecord_file
    file2rows: Dict[str, List[IndexRow]] = {}
    for r in seg_rows:
        file2rows.setdefault(r.tfrecord_file, []).append(r)

    print(f"[INFO] class='{class_name}'")
    print(f"  -> rows(keyframes): {len(seg_rows)}")
    print(f"  -> tfrecord files : {len(file2rows)}")
    print(f"  -> out_root       : {out_root_p}")

    exported_rows: List[Dict[str, Any]] = []

    # parallel across tfrecord files
    workers = max(1, int(num_workers))
    if workers == 1:
        for tfp, rws in file2rows.items():
            exported_rows.extend(
                process_tfrecord_file(
                    tfrecord_path=tfp,
                    rows_for_file=rws,
                    out_root=out_root_p,
                    class_name=class_name,
                    front_cam_id=front_cam_id,
                    rear_cam_id=rear_cam_id,
                    dataset_root=dataset_root,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for tfp, rws in file2rows.items():
                futures.append(ex.submit(
                    process_tfrecord_file,
                    tfp, rws, out_root_p, class_name, front_cam_id, rear_cam_id, dataset_root
                ))
            for fu in as_completed(futures):
                exported_rows.extend(fu.result())

    # class-level export index
    out_index = out_root_p / "export_index.csv"
    with open(out_index, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "seg_id", "key_idx", "intent", "score_max", "score_mean", "key_folder"])
        for r in sorted(exported_rows, key=lambda x: (x["seg_id"], int(x["key_idx"]))):
            w.writerow([r["class_name"], r["seg_id"], int(r["key_idx"]), r["intent"], r["score_max"], r["score_mean"], r["key_folder"]])

    # segment summaries
    finalize_segment_summaries(out_root_p, class_name)

    print("[DONE]")
    print(f"  -> class index: {out_index}")
    print(f"  -> segments   : {len({r['seg_id'] for r in exported_rows})}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export WOD-E2E keyframe packages for a given class (fast version).")
    p.add_argument("--dataset_root", required=True, help="Dataset root folder, e.g. /mnt/d/Datasets/WOD_E2E_Camera_v1")
    p.add_argument("--val_index_csv", required=True, help="Path to val_index_filled.csv (with tfrecord_file + record_idx).")
    p.add_argument("--out_root", required=True, help="Output root folder for this class, e.g. /mnt/d/.../Cut_ins")
    p.add_argument("--class_name", required=True, help='Class name exactly as in CSV, e.g. "Cut_ins" or "Special Vehicles"')
    p.add_argument("--front_cam_id", type=int, default=1, help="Front camera numeric id (default: 1)")
    p.add_argument("--rear_cam_id", type=int, default=7, help="Rear camera numeric id (default: 7)")
    p.add_argument("--num_workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))), help="Parallel workers across TFRecord files (default: min(8, cpu_count)).")
    return p


def main():
    args = build_argparser().parse_args()
    export_class(
        dataset_root=args.dataset_root,
        val_index_csv=args.val_index_csv,
        out_root=args.out_root,
        class_name=args.class_name,
        front_cam_id=int(args.front_cam_id),
        rear_cam_id=int(args.rear_cam_id),
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
