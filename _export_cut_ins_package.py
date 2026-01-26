#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export WOD-E2E (val) Cut_ins packages for VLM research.

Requirements (pip):
  - numpy
  - pandas
  - tensorflow (2.x)
  - opencv-python
  - waymo-open-dataset (compatible with your TF version)
  - (optional) tqdm

Typical usage (WSL):
  python export_cut_ins_package.py \
    --dataset_root "/mnt/d/Datasets/WOD_E2E_Camera_v1" \
    --val_index_csv "/mnt/d/Datasets/WOD_E2E_Camera_v1/val_index_filled.csv" \
    --out_root "/mnt/d/Datasets/WOD_E2E_Camera_v1/Cut_ins" \
    --class_name "Cut_ins" \
    --front_cam_id "1" \
    --rear_cam_id "7"

Notes:
  - This script does NOT use meta/val_sequence_name_to_scenario_cluster.json.
  - It relies on val_index_filled.csv to locate (tfrecord_file, record_idx) for each (seg_id, key_idx).
  - It saves per-keyframe package + per-segment kinematics summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- hard deps ----------
try:
    import tensorflow as tf
except Exception as e:
    raise ImportError("TensorFlow is required. Please install tensorflow==2.x") from e

try:
    import cv2
except Exception as e:
    raise ImportError("opencv-python is required. Please install opencv-python") from e

# Waymo protos / ops
try:
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
    from waymo_open_dataset import dataset_pb2 as open_dataset
except Exception as e:
    raise ImportError(
        "waymo-open-dataset is required with correct version for your TF.\n"
        "Example: pip install waymo-open-dataset-tf-2-13-0"
    ) from e

# Optional camera ops (for projection overlays)
try:
    from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
    _HAS_CAMERA_OPS = True
except Exception:
    py_camera_model_ops = None
    _HAS_CAMERA_OPS = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


# ---------------------------
# Utils
# ---------------------------

def transform_vehicle_to_world(points_xyz_vehicle: np.ndarray, vehicle_pose_4x4: np.ndarray) -> np.ndarray:
    """points: [N,3] in vehicle frame -> world frame using 4x4 pose."""
    pts = np.asarray(points_xyz_vehicle, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # [N,4]
    w = (vehicle_pose_4x4.astype(np.float32) @ pts_h.T).T  # [N,4]
    return w[:, :3]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def to_wsl_path_if_needed(s: str) -> str:
    """Convert Windows path D:\... to WSL /mnt/d/... if needed."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return s

def safe_json_dump(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_e2ed_frame(raw_bytes: bytes) -> wod_e2ed_pb2.E2EDFrame:
    msg = wod_e2ed_pb2.E2EDFrame()
    msg.ParseFromString(raw_bytes)
    return msg

def enum_name_safely(enum_wrapper, value: Any) -> str:
    """Robust enum name from protobuf EnumTypeWrapper."""
    try:
        v = int(value)
    except Exception:
        return str(value)
    # Most stable: use DESCRIPTOR
    try:
        return enum_wrapper.DESCRIPTOR.values_by_number[v].name
    except Exception:
        return str(value)

def cam_name_from_id(cam_id: Any) -> str:
    if hasattr(open_dataset, "CameraName"):
        return enum_name_safely(open_dataset.CameraName, cam_id)
    return str(int(cam_id))

def intent_name_from_value(v: Any) -> str:
    if hasattr(wod_e2ed_pb2, "Intent"):
        try:
            return wod_e2ed_pb2.Intent.Name(int(v))
        except Exception:
            pass
    return str(int(v))

def jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to BGR image for cv2."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def save_jpeg_bytes(jpeg_bytes: bytes, out_path: Path) -> None:
    out_path.write_bytes(jpeg_bytes)

def extract_camera_calib_map(frame) -> Dict[int, Any]:
    """frame.context.camera_calibrations: map cam_id -> calib proto."""
    m: Dict[int, Any] = {}
    if hasattr(frame, "context") and hasattr(frame.context, "camera_calibrations"):
        for c in frame.context.camera_calibrations:
            try:
                m[int(c.name)] = c
            except Exception:
                pass
    return m

def extract_pose_4x4(frame) -> Optional[np.ndarray]:
    """frame.pose.transform is 16 floats (row-major 4x4)."""
    if hasattr(frame, "pose") and hasattr(frame.pose, "transform"):
        t = list(frame.pose.transform)
        if len(t) == 16:
            return np.array(t, dtype=np.float32).reshape(4, 4)
    return None

def extract_ego_states_scalar_fields(states_msg) -> Dict[str, np.ndarray]:
    """
    EgoTrajectoryStates may be a message with repeated scalar fields:
      pos_x, pos_y, pos_z, vel_x, vel_y, accel_x, accel_y, ...
    Return dict of arrays; missing fields omitted.
    """
    out: Dict[str, np.ndarray] = {}
    if states_msg is None:
        return out

    # Known common fields in your logs
    candidates = [
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "accel_x", "accel_y", "accel_z",
        "yaw", "heading"
    ]
    for k in candidates:
        if hasattr(states_msg, k):
            v = list(getattr(states_msg, k))
            if len(v) > 0:
                out[k] = np.array(v, dtype=np.float32)

    return out

def build_traj_df_from_fields(fields: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Build a trajectory dataframe with schema columns.
    Prefer pos_x,pos_y,pos_z; optionally vel/accel.
    """
    # length guess: choose pos_x if exists else any field length
    n = None
    for k, arr in fields.items():
        if arr is not None and len(arr) > 0:
            n = len(arr)
            break
    if n is None:
        return pd.DataFrame(columns=["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"])

    def col(name, fallback=np.nan):
        if name in fields and len(fields[name]) == n:
            return fields[name]
        return np.full((n,), fallback, dtype=np.float32)

    df = pd.DataFrame({
        "t": np.arange(n, dtype=np.float32),
        "x": col("pos_x"),
        "y": col("pos_y"),
        "z": col("pos_z"),
        "vx": col("vel_x"),
        "vy": col("vel_y"),
        "vz": col("vel_z"),
        "ax": col("accel_x"),
        "ay": col("accel_y"),
        "az": col("accel_z"),
    })
    return df

def parse_pref_scores(pref_scores_str: str) -> List[float]:
    s = str(pref_scores_str).strip()
    if not s:
        return []
    out: List[float] = []
    for p in s.split("|"):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out

def extract_preference_trajectories(e2e: wod_e2ed_pb2.E2EDFrame) -> List[Dict[str, Any]]:
    """
    Extract preference trajectories with best-effort.
    Returns list of dicts: {score, traj_fields, df}
    If we cannot find point sequences, df may be empty.
    """
    out: List[Dict[str, Any]] = []
    if not hasattr(e2e, "preference_trajectories"):
        return out

    for i, tr in enumerate(e2e.preference_trajectories):
        item: Dict[str, Any] = {"index": i}

        # score (sometimes preference_score exists)
        if hasattr(tr, "preference_score"):
            try:
                item["preference_score"] = float(tr.preference_score)
            except Exception:
                item["preference_score"] = None

        # Heuristic extraction:
        # - case A: repeated scalar fields pos_x/pos_y/pos_z
        fields = {}
        for k in ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y"]:
            if hasattr(tr, k):
                v = list(getattr(tr, k))
                if len(v) > 0:
                    fields[k] = np.array(v, dtype=np.float32)

        # - case B: repeated message of points/states
        if not fields:
            # find best repeated message field by name
            rep = []
            if hasattr(tr, "ListFields"):
                for fd, val in tr.ListFields():
                    if fd.label == fd.LABEL_REPEATED and fd.type == fd.TYPE_MESSAGE:
                        rep.append((fd.name.lower(), fd.name, val))
            best = None
            kws = ["state", "states", "trajectory", "point", "points", "sample", "samples"]
            for low, name, val in rep:
                score = sum(k in low for k in kws)
                cand = (score, name, val)
                if best is None or cand[0] > best[0]:
                    best = cand
            if best is not None and len(best[2]) > 0:
                pts = best[2]
                # attempt x/y/z or pos_x/pos_y/pos_z in each point
                xs, ys, zs = [], [], []
                for p in pts:
                    if hasattr(p, "x") and hasattr(p, "y"):
                        xs.append(float(getattr(p, "x")))
                        ys.append(float(getattr(p, "y")))
                        zs.append(float(getattr(p, "z")) if hasattr(p, "z") else 0.0)
                    elif hasattr(p, "pos_x") and hasattr(p, "pos_y"):
                        xs.append(float(getattr(p, "pos_x")))
                        ys.append(float(getattr(p, "pos_y")))
                        zs.append(float(getattr(p, "pos_z")) if hasattr(p, "pos_z") else 0.0)
                    else:
                        # cannot parse this point schema
                        xs = []
                        break
                if xs:
                    fields["pos_x"] = np.array(xs, dtype=np.float32)
                    fields["pos_y"] = np.array(ys, dtype=np.float32)
                    fields["pos_z"] = np.array(zs, dtype=np.float32)

        item["traj_fields"] = list(fields.keys())
        item["df"] = build_traj_df_from_fields(fields)
        out.append(item)

    return out


# ---------------------------
# Projection + overlay
# ---------------------------

def try_project_vehicle_points_to_image(
    points_xyz_vehicle: np.ndarray,
    cam_calib_proto: Any,
    cam_image_metadata: Any,
    vehicle_pose_4x4: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:
    """
    Best-effort projection for your py_camera_model_ops version.
    Needs camera_image_metadata + global_coordinate for world_to_image.
    Prefer vehicle_to_image for vehicle-frame points.
    Returns (uv Nx2) or (None, reason).
    """
    if not _HAS_CAMERA_OPS:
        return None, "camera_ops_not_available"
    if points_xyz_vehicle is None or len(points_xyz_vehicle) == 0:
        return None, "empty_points"
    if cam_calib_proto is None:
        return None, "missing_camera_calibration"
    if cam_image_metadata is None:
        return None, "missing_camera_image_metadata"

    pts = tf.convert_to_tensor(np.asarray(points_xyz_vehicle, dtype=np.float32))
    tf_pose = None
    if vehicle_pose_4x4 is not None:
        tf_pose = tf.convert_to_tensor(vehicle_pose_4x4.astype(np.float32))

    # ---- 1) Prefer vehicle_to_image (vehicle frame points) ----
    fn = getattr(py_camera_model_ops, "vehicle_to_image", None)
    if callable(fn):
        # try a few signatures (different WOD versions)
        candidates = []
        # most likely in your version:
        candidates.append((pts, cam_calib_proto, cam_image_metadata))               # +0
        if tf_pose is not None:
            candidates.append((pts, cam_calib_proto, cam_image_metadata, tf_pose)) # +pose
            candidates.append((pts, cam_calib_proto, tf_pose, cam_image_metadata)) # swapped
        for args in candidates:
            try:
                uv = fn(*args).numpy()
                return uv[:, :2], f"ok_vehicle_to_image:{len(args)}args"
            except TypeError as e:
                last_err = e
            except Exception as e:
                return None, f"vehicle_to_image_error:{type(e).__name__}:{str(e)[:120]}"

    # ---- 2) Fallback: world_to_image ----
    fn2 = getattr(py_camera_model_ops, "world_to_image", None)
    if callable(fn2):
        # if we have pose, transform vehicle points to world points
        if vehicle_pose_4x4 is not None:
            pts_world = transform_vehicle_to_world(points_xyz_vehicle, vehicle_pose_4x4)
            ptsw = tf.convert_to_tensor(np.asarray(pts_world, dtype=np.float32))
            # global_coordinate should be True for world points
            candidates2 = []
            candidates2.append((ptsw, cam_calib_proto, cam_image_metadata, True))
            if tf_pose is not None:
                candidates2.append((ptsw, cam_calib_proto, cam_image_metadata, tf_pose, True))
                candidates2.append((ptsw, cam_calib_proto, tf_pose, cam_image_metadata, True))
            for args in candidates2:
                try:
                    uv = fn2(*args).numpy()
                    return uv[:, :2], f"ok_world_to_image:{len(args)}args"
                except TypeError as e:
                    last_err = e
                except Exception as e:
                    return None, f"world_to_image_error:{type(e).__name__}:{str(e)[:120]}"
        else:
            return None, "world_to_image_needs_pose_for_vehicle_points"

    # ---- None worked ----
    try:
        return None, f"projection_error:TypeError:{str(last_err)[:160]}"
    except Exception:
        return None, "no_known_projection_fn"


def draw_polyline(img_bgr: np.ndarray, uv: np.ndarray, color: Tuple[int,int,int], thickness: int = 2) -> np.ndarray:
    if img_bgr is None:
        return img_bgr
    out = img_bgr.copy()
    if uv is None or len(uv) < 2:
        return out

    h, w = out.shape[:2]
    pts = []
    for u, v in uv:
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x = int(round(float(u)))
        y = int(round(float(v)))
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))
    if len(pts) >= 2:
        cv2.polylines(out, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)
        # start point marker
        cv2.circle(out, pts[0], 4, color, -1)
    return out


# ---------------------------
# Segment-level kinematics summary
# ---------------------------

def ego_current_from_past(past_fields: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Use last element of past arrays as "current" state at key moment.
    """
    def last(name: str) -> Optional[float]:
        if name in past_fields and len(past_fields[name]) > 0:
            return float(past_fields[name][-1])
        return None

    x = last("pos_x")
    y = last("pos_y")
    z = last("pos_z")
    vx = last("vel_x")
    vy = last("vel_y")
    ax = last("accel_x")
    ay = last("accel_y")

    speed = None
    if vx is not None and vy is not None:
        speed = float(math.sqrt(vx*vx + vy*vy))

    accel = None
    if ax is not None and ay is not None:
        accel = float(math.sqrt(ax*ax + ay*ay))

    return {
        "x": x, "y": y, "z": z,
        "vx": vx, "vy": vy,
        "ax": ax, "ay": ay,
        "speed": speed,
        "accel": accel,
    }


# ---------------------------
# Main export pipeline
# ---------------------------

def load_cutins_index(val_index_csv: Path, class_name: str) -> pd.DataFrame:
    df = pd.read_csv(val_index_csv)
    if "class" not in df.columns:
        raise ValueError(f"val_index_filled.csv must contain column 'class'. got columns={list(df.columns)}")

    df["class"] = df["class"].astype(str).str.strip()
    df = df[df["class"] == class_name].copy()

    if len(df) == 0:
        raise ValueError(f"No rows for class={class_name}. Check class names in your file.")

    # required columns
    required = ["seg_id", "key_idx"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {val_index_csv}")

    # locate tfrecord + record index
    # allow alternative names
    file_col = None
    for cand in ["tfrecord_file", "file", "tfrecord_path"]:
        if cand in df.columns:
            file_col = cand
            break
    if file_col is None:
        raise ValueError("Cannot find tfrecord file column. Expected one of: tfrecord_file/file/tfrecord_path")

    rec_col = None
    for cand in ["record_idx", "record", "example_idx"]:
        if cand in df.columns:
            rec_col = cand
            break
    if rec_col is None:
        raise ValueError("Cannot find record index column. Expected one of: record_idx/record/example_idx")

    df["tfrecord_file"] = df[file_col].astype(str).apply(to_wsl_path_if_needed)
    df["record_idx"] = df[rec_col].astype(int)

    # normalize
    df["seg_id"] = df["seg_id"].astype(str).str.strip()
    df["key_idx"] = df["key_idx"].astype(int)

    # rater presence
    if "pref_scores" in df.columns:
        df["pref_scores"] = df["pref_scores"].astype(str)
        df["has_rater"] = df["pref_scores"].str.strip().ne("")
    else:
        df["pref_scores"] = ""
        df["has_rater"] = False

    # prefer row with pref_scores if duplicates exist for same (seg_id,key_idx)
    df["_pref_nonempty"] = df["has_rater"]

    sort_cols = ["seg_id", "key_idx", "_pref_nonempty", "record_idx"]
    df = df.sort_values(sort_cols, ascending=[True, True, False, True]).drop_duplicates(["seg_id", "key_idx"], keep="first")
    df = df.drop(columns=["_pref_nonempty"])

    return df


def read_needed_records(grouped: Dict[str, List[int]]) -> Dict[Tuple[str, int], bytes]:
    """
    Read TFRecord files sequentially and extract needed record indices.
    Returns dict[(file_path, record_idx)] = raw_bytes
    """
    out: Dict[Tuple[str, int], bytes] = {}
    file_items = list(grouped.items())

    it = tqdm(file_items, desc="Scanning TFRecord files") if _HAS_TQDM else file_items
    for file_path, idx_list in it:
        idx_set = set(int(x) for x in idx_list)
        if not idx_set:
            continue
        max_idx = max(idx_set)

        ds = tf.data.TFRecordDataset(file_path, compression_type="")
        for i, raw in enumerate(ds):
            if i > max_idx:
                break
            if i in idx_set:
                out[(file_path, i)] = bytes(raw.numpy())

        # sanity
        missing = [i for i in idx_set if (file_path, i) not in out]
        if missing:
            print(f"[WARN] Missing {len(missing)} records in {file_path}: e.g. {missing[:5]}")

    return out


def export_segment(
    class_root: Path,
    class_name: str,
    seg_id: str,
    seg_rows: pd.DataFrame,
    record_bytes_map: Dict[Tuple[str, int], bytes],
    front_cam_id: str,
    rear_cam_id: str,
    fps_assumed: float = 10.0,
) -> None:
    """
    Export all keyframes for one segment.
    """
    seg_dir = class_root / f"{class_name}_{seg_id}"
    ensure_dir(seg_dir)

    manifest: Dict[str, Any] = {
        "class_name": class_name,
        "seg_id": seg_id,
        "front_cam_id": front_cam_id,
        "rear_cam_id": rear_cam_id,
        "fps_assumed": fps_assumed,
        "keyframes": [],
    }

    timeseries_rows: List[Dict[str, Any]] = []

    # sort by key_idx
    seg_rows = seg_rows.sort_values("key_idx")

    for _, r in seg_rows.iterrows():
        key_idx = int(r["key_idx"])
        tfrecord_file = str(r["tfrecord_file"])
        record_idx = int(r["record_idx"])

        raw = record_bytes_map.get((tfrecord_file, record_idx), None)
        if raw is None:
            print(f"[WARN] No bytes for seg={seg_id} key_idx={key_idx} file={tfrecord_file} rec={record_idx}")
            continue

        e2e = parse_e2ed_frame(raw)
        frame = e2e.frame

        key_dir = seg_dir / str(key_idx)  # per your requirement (no padding)
        ensure_dir(key_dir)

        # ---------- camera JPEGs ----------
        cam_map: Dict[str, str] = {}
        cam_files: Dict[str, str] = {}
        images_by_id: Dict[int, bytes] = {}
        image_meta_by_id: Dict[int, Any] = {}

        if hasattr(frame, "images"):
            for img in frame.images:
                cid = int(img.name)
                cname = cam_name_from_id(cid)
                cam_map[str(cid)] = cname

                jpeg = bytes(img.image)
                images_by_id[cid] = jpeg
                out_img = key_dir / f"cam_{cid}.jpg"
                save_jpeg_bytes(jpeg, out_img)
                cam_files[str(cid)] = out_img.name

                # ✅ 新增：保存 camera_image_metadata（world_to_image / vehicle_to_image 需要）
                if hasattr(img, "camera_image_metadata"):
                    image_meta_by_id[cid] = img.camera_image_metadata
                else:
                    image_meta_by_id[cid] = None

        # ---------- ego trajectories ----------
        past_fields = extract_ego_states_scalar_fields(e2e.past_states) if hasattr(e2e, "past_states") else {}
        future_fields = extract_ego_states_scalar_fields(e2e.future_states) if hasattr(e2e, "future_states") else {}

        past_df = build_traj_df_from_fields(past_fields)
        future_df = build_traj_df_from_fields(future_fields)

        past_csv = key_dir / "ego_past.csv"
        future_csv = key_dir / "ego_future.csv"
        past_df.to_csv(past_csv, index=False)
        future_df.to_csv(future_csv, index=False)

        # ---------- preference trajectories ----------
        pref_items = extract_preference_trajectories(e2e)
        # If index CSV has pref_scores, keep it as authoritative list for "scores"
        scores_from_index = parse_pref_scores(r.get("pref_scores", ""))
        scores_from_proto = [pi.get("preference_score", None) for pi in pref_items]
        # choose score list
        scores = scores_from_index if len(scores_from_index) > 0 else scores_from_proto

        pref_files: List[str] = []
        for pi in pref_items:
            i = int(pi["index"])
            df_tr = pi["df"]
            out_tr = key_dir / f"pref_traj_{i}.csv"
            df_tr.to_csv(out_tr, index=False)
            pref_files.append(out_tr.name)

        # ---------- overlay (front/rear) ----------
        overlay_meta: Dict[str, Any] = {
            "front_cam_id": front_cam_id,
            "rear_cam_id": rear_cam_id,
            "camera_ops_available": _HAS_CAMERA_OPS,
            "projection": {},
            "drawn_trajectories": [],
            "note": "",
        }

        calib_map = extract_camera_calib_map(frame)
        pose_4x4 = extract_pose_4x4(frame)

        # choose trajectories to draw: ego past, ego future, all preference trajectories
        traj_to_draw: List[Tuple[str, pd.DataFrame]] = [("ego_past", past_df), ("ego_future", future_df)]
        for pi in pref_items:
            traj_to_draw.append((f"pref_traj_{pi['index']}", pi["df"]))

        # colors in BGR (do not overthink; deterministic)
        palette = [
            (0, 255, 0),    # green
            (0, 0, 255),    # red
            (255, 0, 0),    # blue
            (0, 255, 255),  # yellow
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan
        ]

        def overlay_on_cam(cam_id_str: str, out_name: str) -> None:
            cam_id = int(cam_id_str)
            jpeg = images_by_id.get(cam_id, None)
            if jpeg is None:
                overlay_meta["projection"][cam_id_str] = "missing_camera_image"
                return
            calib = calib_map.get(cam_id, None)
            if calib is None:
                overlay_meta["projection"][cam_id_str] = "missing_camera_calibration"
                return

            img_bgr = jpeg_bytes_to_bgr(jpeg)
            if img_bgr is None:
                overlay_meta["projection"][cam_id_str] = "jpeg_decode_failed"
                return

            proj_statuses = []
            out_img = img_bgr.copy()

            for ti, (tname, tdf) in enumerate(traj_to_draw):
                if tdf is None or len(tdf) < 2:
                    continue
                # points in vehicle frame (x,y,z)
                pts = tdf[["x", "y", "z"]].to_numpy(dtype=np.float32)
                meta = image_meta_by_id.get(cam_id, None)
                if meta is None:
                    overlay_meta["projection"][cam_id_str] = "missing_camera_image_metadata"
                    return

                uv, status = try_project_vehicle_points_to_image(pts, calib, meta, pose_4x4)
                
                proj_statuses.append((tname, status))

                if uv is not None:
                    color = palette[ti % len(palette)]
                    out_img = draw_polyline(out_img, uv, color=color, thickness=2)
                    overlay_meta["drawn_trajectories"].append({
                        "name": tname,
                        "cam_id": cam_id_str,
                        "status": status,
                        "num_points": int(len(pts)),
                    })

            overlay_meta["projection"][cam_id_str] = proj_statuses

            out_path = key_dir / out_name
            cv2.imwrite(str(out_path), out_img)

        # write overlay images if possible
        try:
            overlay_on_cam(front_cam_id, "overlay_front.jpg")
            overlay_on_cam(rear_cam_id, "overlay_rear.jpg")
        except Exception as e:
            overlay_meta["note"] = f"overlay_failed:{type(e).__name__}:{str(e)[:200]}"

        safe_json_dump(overlay_meta, key_dir / "overlay_meta.json")

        # ---------- frame.json ----------
        intent_raw = int(getattr(e2e, "intent", r.get("intent", -1)))
        frame_json = {
            "class_name": class_name,
            "seg_id": seg_id,
            "key_idx": key_idx,
            "t_est_sec": float(key_idx) / float(fps_assumed),  # no absolute time, only sequence time
            "context_name": getattr(frame.context, "name", "") if hasattr(frame, "context") else "",
            "intent_raw": intent_raw,
            "intent_name": intent_name_from_value(intent_raw),
            "camera_map": cam_map,           # id -> enum name (best-effort)
            "camera_files": cam_files,       # id -> filename in this folder
            "front_cam_id": front_cam_id,
            "rear_cam_id": rear_cam_id,
            "ego": {
                "past_csv": past_csv.name,
                "future_csv": future_csv.name,
                "past_fields": list(past_fields.keys()),
                "future_fields": list(future_fields.keys()),
            },
            "rater": {
                "scores": scores,
                "pref_traj_files": pref_files,
                "n_pref_traj": len(pref_files),
                "scores_source": "val_index_filled" if len(scores_from_index) > 0 else "proto_or_empty",
            },
            "overlay": {
                "front": "overlay_front.jpg",
                "rear": "overlay_rear.jpg",
                "meta": "overlay_meta.json",
            },
            "schema_version": "v1",
        }
        safe_json_dump(frame_json, key_dir / "frame.json")

        # ---------- segment time series row (use current state from past) ----------
        cur = ego_current_from_past(past_fields)
        ts_row = {
            "seg_id": seg_id,
            "class_name": class_name,
            "key_idx": key_idx,
            "t_est_sec": float(key_idx) / float(fps_assumed),
            "intent_raw": intent_raw,
            "intent_name": intent_name_from_value(intent_raw),
            "has_rater": bool(str(r.get("pref_scores", "")).strip() != ""),
            "n_scores": len(scores) if scores is not None else 0,
            "score_max": float(np.nanmax(scores)) if scores else np.nan,
            "score_mean": float(np.nanmean(scores)) if scores else np.nan,
            "x": cur.get("x"),
            "y": cur.get("y"),
            "z": cur.get("z"),
            "vx": cur.get("vx"),
            "vy": cur.get("vy"),
            "ax": cur.get("ax"),
            "ay": cur.get("ay"),
            "speed": cur.get("speed"),
            "accel": cur.get("accel"),
        }
        timeseries_rows.append(ts_row)

        manifest["keyframes"].append({
            "key_idx": key_idx,
            "folder": str(key_idx),
            "record": {
                "tfrecord_file": tfrecord_file,
                "record_idx": record_idx,
            }
        })

    # ---------- per-segment summaries ----------
    ts_df = pd.DataFrame(timeseries_rows).sort_values("key_idx")
    ts_path = seg_dir / "segment_ego_timeseries.csv"
    ts_df.to_csv(ts_path, index=False)

    summary: Dict[str, Any] = {
        "class_name": class_name,
        "seg_id": seg_id,
        "num_keyframes_exported": int(len(ts_df)),
        "key_idx_min": int(ts_df["key_idx"].min()) if len(ts_df) else None,
        "key_idx_max": int(ts_df["key_idx"].max()) if len(ts_df) else None,
        "fps_assumed": fps_assumed,
        "duration_est_sec": float((ts_df["key_idx"].max() - ts_df["key_idx"].min()) / fps_assumed) if len(ts_df) else None,
        "rater_coverage": float(ts_df["has_rater"].mean()) if "has_rater" in ts_df.columns and len(ts_df) else None,
        "speed_stats": {},
        "accel_stats": {},
        "displacement_m": None,
    }

    def add_stats(col: str) -> Dict[str, Any]:
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

    summary["speed_stats"] = add_stats("speed")
    summary["accel_stats"] = add_stats("accel")

    # displacement using (x,y) between first and last valid points
    if "x" in ts_df.columns and "y" in ts_df.columns and len(ts_df) > 1:
        xy = ts_df[["x", "y"]].dropna()
        if len(xy) >= 2:
            x0, y0 = float(xy.iloc[0]["x"]), float(xy.iloc[0]["y"])
            x1, y1 = float(xy.iloc[-1]["x"]), float(xy.iloc[-1]["y"])
            summary["displacement_m"] = float(math.sqrt((x1-x0)**2 + (y1-y0)**2))

    safe_json_dump(summary, seg_dir / "segment_summary.json")
    safe_json_dump(manifest, seg_dir / "segment_manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True, help="Dataset root, e.g. /mnt/d/Datasets/WOD_E2E_Camera_v1")
    ap.add_argument("--val_index_csv", type=str, required=True, help="val_index_filled.csv path")
    ap.add_argument("--out_root", type=str, required=True, help="Output root folder, e.g. /mnt/d/.../Cut_ins")
    ap.add_argument("--class_name", type=str, default="Cut_ins")
    ap.add_argument("--front_cam_id", type=str, default="1")
    ap.add_argument("--rear_cam_id", type=str, default="7")
    ap.add_argument("--fps_assumed", type=float, default=10.0)
    args = ap.parse_args()

    dataset_root = Path(to_wsl_path_if_needed(args.dataset_root))
    val_index_csv = Path(to_wsl_path_if_needed(args.val_index_csv))
    out_root = Path(to_wsl_path_if_needed(args.out_root))

    ensure_dir(out_root)

    print("dataset_root:", dataset_root)
    print("val_index_csv:", val_index_csv)
    print("out_root:", out_root)
    print("class_name:", args.class_name)
    print("camera_ops available:", _HAS_CAMERA_OPS)

    df = load_cutins_index(val_index_csv, args.class_name)
    print(f"[INFO] rows after filter+dedup (keyframes): {len(df)}")
    print(f"[INFO] unique segments: {df['seg_id'].nunique()}")

    # Group needed records by file
    grouped: Dict[str, List[int]] = {}
    for file_path, g in df.groupby("tfrecord_file"):
        grouped[file_path] = g["record_idx"].astype(int).tolist()

    record_bytes_map = read_needed_records(grouped)

    # Export per segment
    seg_ids = sorted(df["seg_id"].unique().tolist())
    seg_iter = tqdm(seg_ids, desc="Exporting segments") if _HAS_TQDM else seg_ids

    for seg_id in seg_iter:
        seg_rows = df[df["seg_id"] == seg_id].copy()
        export_segment(
            class_root=out_root,
            class_name=args.class_name,
            seg_id=str(seg_id),
            seg_rows=seg_rows,
            record_bytes_map=record_bytes_map,
            front_cam_id=str(args.front_cam_id),
            rear_cam_id=str(args.rear_cam_id),
            fps_assumed=float(args.fps_assumed),
        )

    print("\n[DONE] Export completed.")
    print("Output at:", out_root)


if __name__ == "__main__":
    main()
