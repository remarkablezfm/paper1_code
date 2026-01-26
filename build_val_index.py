#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_val_index_v2.py

功能：
- 扫描 WOD-E2E val 目录下所有 TFRecord（每条 record = 一个 key moment 小场景）
- 从 E2EDFrame.frame.context.name 抽取 (seg_id, key_idx)
- 读取分类映射 JSON（sequence_name/seg_id -> scenario_cluster）
- 输出：
  1) val_index.csv：全局索引（1行=1个key moment）
  2) val_index_manifest.json：全局统计与参数记录（便于复现/断点/审计）

依赖：
- python >= 3.9
- tensorflow
- waymo_open_dataset（你当前能 import 的版本）
- pandas
- tqdm（可选；没有也能跑）
"""

import os
import glob
import json
import argparse
import platform
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


# -----------------------------
# path utils (Windows <-> WSL)
# -----------------------------
def normalize_path(p: str) -> str:
    """把 Windows 盘符路径尽量转成 WSL 可用路径（若当前环境是 Linux/WSL）"""
    if not p:
        return p
    p = p.strip().strip('"').strip("'")

    # 如果本来就存在，直接用
    if os.path.exists(p):
        return p

    # 在 Linux/WSL 下，把 D:\xxx -> /mnt/d/xxx
    if platform.system().lower() == "linux":
        if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
            drive = p[0].lower()
            rest = p[2:].replace("\\", "/")
            cand = f"/mnt/{drive}{rest}"
            if os.path.exists(cand):
                return cand

    # 最后：把反斜杠替换成斜杠试一次
    cand2 = p.replace("\\", "/")
    return cand2


# -----------------------------
# parsing helpers
# -----------------------------
def split_context_name(context_name: str) -> Tuple[str, int]:
    """
    context_name: "<seg_id>-<key_idx>"
    seg_id 本身可能包含 '-'，所以从最后一个 '-' 切。
    """
    if not context_name or "-" not in context_name:
        return context_name, -1
    seg_id, key = context_name.rsplit("-", 1)
    try:
        return seg_id, int(key)
    except Exception:
        return seg_id, -1


def safe_len_repeated_scalar(msg, field: str) -> int:
    if msg is None or (not hasattr(msg, field)):
        return 0
    try:
        return len(list(getattr(msg, field)))
    except Exception:
        return 0


def pref_scores_str(e2e: wod_e2ed_pb2.E2EDFrame) -> str:
    if not hasattr(e2e, "preference_trajectories"):
        return ""
    scores = []
    for p in e2e.preference_trajectories:
        if hasattr(p, "preference_score"):
            scores.append(str(float(p.preference_score)))
    return "|".join(scores)


def cam_ids_str(e2e: wod_e2ed_pb2.E2EDFrame) -> str:
    frame = e2e.frame
    if not hasattr(frame, "images"):
        return ""
    ids = [str(int(im.name)) for im in frame.images]
    return "|".join(ids)


def make_output_folder(cls: str, seg_id: str, key_idx: int) -> str:
    # 你希望：类别 + seg_id + key_frame
    # 用稳定分隔符，脚本解析更方便
    return f"{cls}__{seg_id}__k{key_idx}"


# -----------------------------
# category loader (robust)
# -----------------------------
def load_sequence_cluster_map(json_path: str) -> Dict[str, str]:
    """
    读取 val_sequence_name_to_scenario_cluster.json
    返回：{ sequence_name/seg_id -> cluster_label(str) }

    兼容几种常见结构：
    1) 直接 dict: { "seg_id": "cluster" , ... }
    2) 外面包一层 key: {"val_sequence_name_to_scenario_cluster": {...}}
       或 {"sequence_name_to_scenario_cluster": {...}}
    3) list[dict]：每项含 sequence_name/seg_id + scenario_cluster/cluster/label
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # case: wrapped dict
    if isinstance(obj, dict):
        for k in ["val_sequence_name_to_scenario_cluster", "sequence_name_to_scenario_cluster", "mapping", "data"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    mapping: Dict[str, str] = {}

    # dict mapping
    if isinstance(obj, dict):
        for kk, vv in obj.items():
            if isinstance(vv, (str, int, float)):
                mapping[str(kk)] = str(vv)
        if mapping:
            return mapping

    # list of dict
    if isinstance(obj, list):
        for it in obj:
            if not isinstance(it, dict):
                continue
            seq = it.get("sequence_name") or it.get("sequence") or it.get("seg_id") or it.get("segment_id")
            cls = it.get("scenario_cluster") or it.get("cluster") or it.get("label") or it.get("class") or it.get("category")
            if seq is None or cls is None:
                continue
            mapping[str(seq)] = str(cls)

    return mapping


# -----------------------------
# main scan
# -----------------------------
def scan_val_tfrecords(
    val_dir: str,
    category_json: Optional[str],
    out_csv: str,
    out_manifest_json: str,
    fps_assumed: float = 10.0,
    time_scheme: str = "A",
    front_cam_id: int = 1,
    rear_cam_id: int = 7,
    limit_files: Optional[int] = None,
    limit_records_per_file: Optional[int] = None,
) -> pd.DataFrame:
    val_dir = normalize_path(val_dir)
    if category_json:
        category_json = normalize_path(category_json)

    tfrecords = sorted(glob.glob(os.path.join(val_dir, "*.tfrecord*")))
    if not tfrecords:
        raise FileNotFoundError(f"No tfrecord files found in: {val_dir}")

    if limit_files is not None:
        tfrecords = tfrecords[: int(limit_files)]

    cat_map: Dict[str, str] = {}
    cat_loaded = False
    if category_json and os.path.exists(category_json):
        cat_map = load_sequence_cluster_map(category_json)
        cat_loaded = True

    rows: List[Dict[str, Any]] = []

    # tqdm 可选
    try:
        from tqdm import tqdm
        file_iter = tqdm(tfrecords, desc="Scanning TFRecords")
    except Exception:
        file_iter = tfrecords

    total_records = 0
    for tfrec in file_iter:
        ds = tf.data.TFRecordDataset(tfrec, compression_type="")

        for record_idx, raw in enumerate(ds):
            if limit_records_per_file is not None and record_idx >= int(limit_records_per_file):
                break

            total_records += 1
            e2e = wod_e2ed_pb2.E2EDFrame()
            e2e.ParseFromString(raw.numpy())

            ctx = ""
            if hasattr(e2e, "frame") and hasattr(e2e.frame, "context") and hasattr(e2e.frame.context, "name"):
                ctx = e2e.frame.context.name

            seg_id, key_idx = split_context_name(ctx)

            cls = cat_map.get(seg_id, "UNKNOWN") if cat_loaded else "UNKNOWN"

            num_images = len(e2e.frame.images) if hasattr(e2e, "frame") and hasattr(e2e.frame, "images") else 0
            cams = cam_ids_str(e2e)

            past_len = safe_len_repeated_scalar(getattr(e2e, "past_states", None), "pos_x")
            future_len = safe_len_repeated_scalar(getattr(e2e, "future_states", None), "pos_x")

            scores = pref_scores_str(e2e)
            has_pref = 1 if scores != "" else 0
            intent = int(getattr(e2e, "intent", -1))

            out_folder = make_output_folder(cls, seg_id, key_idx)

            rows.append({
                "class": cls,
                "seg_id": seg_id,
                "key_idx": key_idx,
                "context_name": ctx,
                "tfrecord_file": tfrec,
                "record_idx": record_idx,
                "intent": intent,
                "has_pref": has_pref,
                "pref_scores": scores,
                "past_len": int(past_len),
                "future_len": int(future_len),
                "num_images": int(num_images),
                "cam_ids": cams,
                "front_cam_id": int(front_cam_id),
                "rear_cam_id": int(rear_cam_id),
                "fps_assumed": float(fps_assumed),
                "time_scheme": time_scheme,
                "output_folder": out_folder,
            })

    df = pd.DataFrame(rows)

    # 写 CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    # 生成 manifest
    cls_counts = df["class"].value_counts(dropna=False).to_dict()
    unknown_count = int((df["class"] == "UNKNOWN").sum()) if "class" in df.columns else 0
    pref_rate = float(df["has_pref"].mean()) if len(df) > 0 else 0.0

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "build_val_index_v2.py",
        "val_dir": val_dir,
        "category_json": category_json,
        "category_loaded": bool(cat_loaded),
        "category_map_size": int(len(cat_map)) if cat_loaded else 0,
        "front_cam_id": int(front_cam_id),
        "rear_cam_id": int(rear_cam_id),
        "fps_assumed": float(fps_assumed),
        "time_scheme": str(time_scheme),
        "time_convention": {
            "scheme": str(time_scheme),
            "A_past": "t_rel_key_idx = -N..-1",
            "A_future_pref": "t_rel_key_idx = 0..N-1",
        },
        "scan_limits": {
            "limit_files": None if limit_files is None else int(limit_files),
            "limit_records_per_file": None if limit_records_per_file is None else int(limit_records_per_file),
        },
        "summary": {
            "num_files_scanned": int(len(tfrecords)),
            "num_records_scanned": int(total_records),
            "num_rows_index": int(len(df)),
            "unknown_class_rows": int(unknown_count),
            "has_pref_rate": pref_rate,
        },
        "class_distribution": cls_counts,
        "columns": list(df.columns),
    }

    os.makedirs(os.path.dirname(out_manifest_json) or ".", exist_ok=True)
    with open(out_manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return df


def main():
    default_cat = r"D:\Datasets\WOD_E2E_Camera_v1\meta\val_sequence_name_to_scenario_cluster.json"

    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True, help="Path to WOD-E2E val folder containing tfrecords")
    ap.add_argument("--category_json", default=default_cat, help="Path to val sequence_name->cluster json")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (global index)")
    ap.add_argument("--out_manifest_json", required=True, help="Output manifest JSON path")
    ap.add_argument("--fps_assumed", type=float, default=10.0, help="Assumed FPS/Hz for relative timeline")
    ap.add_argument("--time_scheme", default="A", help="Time scheme label (e.g., A)")
    ap.add_argument("--front_cam_id", type=int, default=1, help="Front view camera id")
    ap.add_argument("--rear_cam_id", type=int, default=7, help="Rear view camera id")
    ap.add_argument("--limit_files", type=int, default=0, help="Debug: scan only first N files (0=all)")
    ap.add_argument("--limit_records_per_file", type=int, default=0, help="Debug: scan only first N records per file (0=all)")
    args = ap.parse_args()

    df = scan_val_tfrecords(
        val_dir=args.val_dir,
        category_json=args.category_json if args.category_json else None,
        out_csv=args.out_csv,
        out_manifest_json=args.out_manifest_json,
        fps_assumed=args.fps_assumed,
        time_scheme=args.time_scheme,
        front_cam_id=args.front_cam_id,
        rear_cam_id=args.rear_cam_id,
        limit_files=args.limit_files if args.limit_files > 0 else None,
        limit_records_per_file=args.limit_records_per_file if args.limit_records_per_file > 0 else None,
    )

    print("done. rows:", len(df))
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

