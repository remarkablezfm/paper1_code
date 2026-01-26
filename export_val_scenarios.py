#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_val_scenes.py

从 val_index.csv 导出每个 key moment 小场景到一个文件夹：
  <class>__<seg_id>__k<key_idx>/
    meta.json
    calib.json
    manifest.json
    ego_past.csv
    ego_future.csv
    pref_0_scoreX.csv ...
    images/cam_1.jpg ... cam_8.jpg
    overlays/cam_1_overlay.jpg
    overlays/cam_7_overlay.jpg
    DONE.txt

依赖：
  pip install pandas numpy opencv-python tqdm
  + 你的环境已能 import:
    tensorflow
    waymo_open_dataset.protos.end_to_end_driving_data_pb2

注意：
- 本脚本不依赖绝对时间戳，使用 Scheme A 时间轴（10Hz 默认）
- heading 使用 B：heading_vel = atan2(vy, vx)（低速设 NaN）
"""

import os
import json
import argparse
import platform
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


# -----------------------------
# path utils (Windows <-> WSL)
# -----------------------------
def normalize_path(p: str) -> str:
    if not p:
        return p
    p = p.strip().strip('"').strip("'")
    if os.path.exists(p):
        return p
    if platform.system().lower() == "linux":
        if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
            drive = p[0].lower()
            rest = p[2:].replace("\\", "/")
            cand = f"/mnt/{drive}{rest}"
            if os.path.exists(cand):
                return cand
    return p.replace("\\", "/")


# -----------------------------
# basic helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def split_context_name(context_name: str) -> Tuple[str, int]:
    if not context_name or "-" not in context_name:
        return context_name, -1
    seg_id, key = context_name.rsplit("-", 1)
    try:
        return seg_id, int(key)
    except Exception:
        return seg_id, -1


def rep_arr(msg, field, dtype=np.float32):
    if msg is None or (not hasattr(msg, field)):
        return None
    try:
        a = np.array(list(getattr(msg, field)), dtype=dtype)
        return a if a.size > 0 else None
    except Exception:
        return None


def stack_xyz(x, y, z=None):
    if x is None or y is None:
        return None
    if z is None:
        z = np.zeros_like(x)
    n = min(len(x), len(y), len(z))
    return np.stack([x[:n], y[:n], z[:n]], axis=1)


def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# -----------------------------
# time scheme A
# -----------------------------
def xyz_to_df_scheme_a(xyz, fps=10.0, kind="future_or_pref", prefix="pos_"):
    if xyz is None:
        return None
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    t_idx = np.arange(n, dtype=int)
    if kind == "past":
        t_rel = t_idx - n         # -N..-1
    else:
        t_rel = t_idx             # 0..N-1

    df = pd.DataFrame({
        "t_idx": t_idx,
        "t_sec": t_idx / float(fps),
        "t_rel_key_idx": t_rel,
        "t_rel_key_sec": t_rel / float(fps),
        f"{prefix}x": xyz[:, 0],
        f"{prefix}y": xyz[:, 1],
        f"{prefix}z": xyz[:, 2],
    })
    return df


# -----------------------------
# ego extraction (+ heading B)
# -----------------------------
def extract_ego_kinematics(e2e: wod_e2ed_pb2.E2EDFrame) -> Dict[str, Any]:
    past = getattr(e2e, "past_states", None)
    fut  = getattr(e2e, "future_states", None)

    out = {
        "past": {
            "pos": stack_xyz(rep_arr(past, "pos_x"), rep_arr(past, "pos_y"), rep_arr(past, "pos_z")),
            "vel": stack_xyz(rep_arr(past, "vel_x"), rep_arr(past, "vel_y"), rep_arr(past, "vel_z")),
            "acc": stack_xyz(rep_arr(past, "accel_x"), rep_arr(past, "accel_y"), rep_arr(past, "accel_z")),
        },
        "future": {
            "pos": stack_xyz(rep_arr(fut, "pos_x"), rep_arr(fut, "pos_y"), rep_arr(fut, "pos_z")),
            "vel": stack_xyz(rep_arr(fut, "vel_x"), rep_arr(fut, "vel_y"), rep_arr(fut, "vel_z")),
            "acc": stack_xyz(rep_arr(fut, "accel_x"), rep_arr(fut, "accel_y"), rep_arr(fut, "accel_z")),
        }
    }
    return out


def add_speed_heading_B(df: pd.DataFrame, speed_eps=0.2) -> pd.DataFrame:
    # heading B: atan2(vy, vx) in vehicle frame
    if ("vel_x" not in df.columns) or ("vel_y" not in df.columns):
        return df
    vx = df["vel_x"].to_numpy(dtype=np.float32)
    vy = df["vel_y"].to_numpy(dtype=np.float32)
    speed = np.sqrt(vx*vx + vy*vy)
    heading = wrap_to_pi(np.arctan2(vy, vx)).astype(np.float32)
    heading[speed < float(speed_eps)] = np.nan

    df["speed"] = speed
    df["heading_vel_rad"] = heading
    df["heading_vel_deg"] = heading * 180.0 / np.pi
    return df


# -----------------------------
# preference trajectory extraction (robust-ish)
# -----------------------------
def extract_pref_xyz(pref_msg) -> Optional[np.ndarray]:
    """
    尝试从 preference_trajectory message 内找出一组 (x,y,z) 或 (x,y) repeated scalar float fields。
    兼容不同 proto 版本：优先找 pos_x/pos_y/pos_z，其次找 x/y/z。
    """
    # 先尝试标准命名
    x = rep_arr(pref_msg, "pos_x") or rep_arr(pref_msg, "x")
    y = rep_arr(pref_msg, "pos_y") or rep_arr(pref_msg, "y")
    z = rep_arr(pref_msg, "pos_z") or rep_arr(pref_msg, "z")
    xyz = stack_xyz(x, y, z)
    if xyz is not None:
        return xyz

    # 再用反射：找 repeated scalar float fields 里最像 x/y/z 的
    try:
        fields = pref_msg.ListFields()
    except Exception:
        return None

    rep_float = {}
    for fd, val in fields:
        # repeated scalar
        if fd.label == fd.LABEL_REPEATED and fd.cpp_type in (fd.CPPTYPE_FLOAT, fd.CPPTYPE_DOUBLE):
            arr = np.array(list(val), dtype=np.float32)
            if arr.size > 0:
                rep_float[fd.name] = arr

    if not rep_float:
        return None

    # 选三元组/二元组：名字包含 x/y/z 且长度一致
    names = list(rep_float.keys())
    # 优先 pos_x/pos_y/pos_z
    for px, py, pz in [("pos_x", "pos_y", "pos_z"), ("x", "y", "z")]:
        if px in rep_float and py in rep_float:
            return stack_xyz(rep_float[px], rep_float[py], rep_float.get(pz, None))

    # 最后：模糊匹配
    def pick(cands, must_contain):
        for n in cands:
            if must_contain in n.lower():
                return n
        return None

    xname = pick(names, "x")
    yname = pick(names, "y")
    zname = pick(names, "z")
    if xname and yname:
        return stack_xyz(rep_float[xname], rep_float[yname], rep_float.get(zname, None))
    return None


def extract_preferences(e2e: wod_e2ed_pb2.E2EDFrame) -> List[Dict[str, Any]]:
    prefs = []
    for i, p in enumerate(getattr(e2e, "preference_trajectories", [])):
        score = float(getattr(p, "preference_score", np.nan))
        xyz = extract_pref_xyz(p)
        prefs.append({"i": i, "score": score, "xyz": xyz})
    return prefs


# -----------------------------
# calibration export
# -----------------------------
def calib_to_dict(calib) -> Dict[str, Any]:
    d: Dict[str, Any] = {"camera_id": int(calib.name) if hasattr(calib, "name") else None}
    if hasattr(calib, "width"): d["width"] = int(calib.width)
    if hasattr(calib, "height"): d["height"] = int(calib.height)
    if hasattr(calib, "rolling_shutter_direction"):
        d["rolling_shutter_direction"] = int(calib.rolling_shutter_direction)

    if hasattr(calib, "intrinsic"):
        d["intrinsic"] = [float(x) for x in list(calib.intrinsic)]
        if len(d["intrinsic"]) >= 4:
            d["fx_fy_cx_cy"] = d["intrinsic"][:4]

    if hasattr(calib, "distortion"):
        d["distortion"] = [float(x) for x in list(calib.distortion)]

    if hasattr(calib, "extrinsic") and hasattr(calib.extrinsic, "transform"):
        ex = [float(x) for x in list(calib.extrinsic.transform)]
        d["extrinsic_flat"] = ex
        if len(ex) == 16:
            d["extrinsic_4x4"] = np.array(ex, dtype=np.float32).reshape(4, 4).tolist()

    return d


def export_calib_json(frame, out_path: str):
    cams = []
    if hasattr(frame, "context") and hasattr(frame.context, "camera_calibrations"):
        for c in frame.context.camera_calibrations:
            cams.append(calib_to_dict(c))
    payload = {"camera_calibrations": cams}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# images + overlays
# -----------------------------
def decode_jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    rgb = tf.io.decode_jpeg(jpeg_bytes).numpy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def export_clean_images(frame, out_dir: str) -> Dict[int, str]:
    ensure_dir(out_dir)
    saved = {}
    for im in frame.images:
        cam_id = int(im.name)
        bgr = decode_jpeg_to_bgr(im.image)
        fn = os.path.join(out_dir, f"cam_{cam_id}.jpg")
        cv2.imwrite(fn, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved[cam_id] = fn
    return saved


def get_cam_calib_map(frame) -> Dict[int, Any]:
    m = {}
    if hasattr(frame, "context") and hasattr(frame.context, "camera_calibrations"):
        for c in frame.context.camera_calibrations:
            m[int(c.name)] = c
    return m


def project_vehicle_points_to_image_pyops(points_xyz: np.ndarray, cam_calib) -> Optional[np.ndarray]:
    """
    使用 waymo 的 camera ops（最稳）进行投影。
    如果 ops 不可用，会返回 None（外层可选择跳过 overlay 或改用你的既有投影函数）。
    """
    try:
        from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
    except Exception:
        return None

    # points_xyz: (N,3) vehicle frame
    pts = tf.constant(points_xyz.astype(np.float32))
    # calib: 需要 dict-like tensor/结构；py ops 支持传入 CameraCalibration proto
    try:
        uv, depth = py_camera_model_ops.world_to_image(
            cam_calib,
            pts
        )
        uv = uv.numpy()
        depth = depth.numpy()
        # 只保留 depth>0
        mask = depth > 0
        return uv[mask]
    except Exception:
        return None


def draw_polyline(img_bgr: np.ndarray, uv: np.ndarray, color, thickness=2):
    if uv is None or len(uv) < 2:
        return
    pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def export_overlay(frame, ego, prefs, cam_id: int, out_path: str, fps=10.0):
    # 找图像
    target = None
    for im in frame.images:
        if int(im.name) == int(cam_id):
            target = im
            break
    if target is None:
        return {"ok": False, "reason": f"missing image for cam {cam_id}"}

    img = decode_jpeg_to_bgr(target.image)

    calib_map = get_cam_calib_map(frame)
    if cam_id not in calib_map:
        return {"ok": False, "reason": f"missing calibration for cam {cam_id}"}
    cam_calib = calib_map[cam_id]

    # 准备 4 条轨迹：ego past, ego future, pref0..2
    # 颜色：past(青), future(绿), prefs(红/橙/紫)
    tracks = []
    if ego["past"]["pos"] is not None: tracks.append(("ego_past", ego["past"]["pos"], (255, 255, 0)))
    if ego["future"]["pos"] is not None: tracks.append(("ego_future", ego["future"]["pos"], (0, 255, 0)))
    pref_colors = [(0,0,255), (0,128,255), (255,0,255)]
    for p in prefs:
        if p["xyz"] is None: 
            continue
        tracks.append((f"pref_{p['i']}_score{p['score']}", p["xyz"], pref_colors[p["i"] % len(pref_colors)]))

    used = []
    for name, xyz, color in tracks:
        uv = project_vehicle_points_to_image_pyops(xyz, cam_calib)
        if uv is None:
            # ops 不可用 / 投影失败：不画，但返回原因
            used.append({"name": name, "drawn": False, "reason": "projection_failed_or_ops_missing"})
            continue
        draw_polyline(img, uv, color=color, thickness=2)
        used.append({"name": name, "drawn": True, "n_uv": int(len(uv))})

    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return {"ok": True, "used": used}


# -----------------------------
# write csvs
# -----------------------------
def write_ego_csv(out_path: str, block: Dict[str, Any], kind: str, fps=10.0, speed_eps=0.2):
    pos, vel, acc = block["pos"], block["vel"], block["acc"]
    if pos is None:
        return False

    df = xyz_to_df_scheme_a(pos, fps=fps, kind=kind, prefix="pos_")

    if vel is not None:
        vel = np.asarray(vel, dtype=np.float32)
        n = min(len(df), len(vel))
        df.loc[:n-1, ["vel_x","vel_y","vel_z"]] = vel[:n]
    if acc is not None:
        acc = np.asarray(acc, dtype=np.float32)
        n = min(len(df), len(acc))
        df.loc[:n-1, ["acc_x","acc_y","acc_z"]] = acc[:n]

    df = add_speed_heading_B(df, speed_eps=speed_eps)
    df.to_csv(out_path, index=False)
    return True


def write_pref_csv(out_path: str, xyz: np.ndarray, fps=10.0):
    df = xyz_to_df_scheme_a(xyz, fps=fps, kind="future_or_pref", prefix="pos_")
    df.to_csv(out_path, index=False)
    return True


# -----------------------------
# per-sample export
# -----------------------------
def read_one_record(tfrecord_file: str, record_idx: int) -> wod_e2ed_pb2.E2EDFrame:
    ds = tf.data.TFRecordDataset(tfrecord_file, compression_type="")
    raw = None
    for i, r in enumerate(ds):
        if i == int(record_idx):
            raw = r.numpy()
            break
    if raw is None:
        raise RuntimeError(f"record_idx {record_idx} not found in {tfrecord_file}")
    e2e = wod_e2ed_pb2.E2EDFrame()
    e2e.ParseFromString(raw)
    return e2e


def export_one_sample(row: Dict[str, Any], out_root: str,
                      fps=10.0, speed_eps=0.2,
                      overlay_cam_ids=(1,7),
                      skip_done=True) -> Dict[str, Any]:
    cls = str(row.get("class", "UNKNOWN"))
    seg_id = str(row.get("seg_id", ""))
    key_idx = int(row.get("key_idx", -1))
    context_name = str(row.get("context_name", ""))

    out_folder = str(row.get("output_folder", f"{cls}__{seg_id}__k{key_idx}"))
    out_dir = os.path.join(out_root, out_folder)
    done_flag = os.path.join(out_dir, "DONE.txt")
    if skip_done and os.path.exists(done_flag):
        return {"skipped": True, "out_dir": out_dir}

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "images"))
    ensure_dir(os.path.join(out_dir, "overlays"))

    tfrecord_file = normalize_path(str(row["tfrecord_file"]))
    record_idx = int(row["record_idx"])

    e2e = read_one_record(tfrecord_file, record_idx)
    frame = e2e.frame

    ego = extract_ego_kinematics(e2e)
    prefs = extract_preferences(e2e)
    scores = [p["score"] for p in prefs if not np.isnan(p["score"])]

    # meta.json
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "class": cls,
        "seg_id": seg_id,
        "key_idx": key_idx,
        "context_name": context_name,
        "tfrecord_file": tfrecord_file,
        "record_idx": record_idx,
        "fps_assumed": float(fps),
        "time_scheme": "A",
        "heading_source": "B: heading_vel=atan2(vy,vx)",
        "heading_speed_eps": float(speed_eps),
        "intent_raw": int(getattr(e2e, "intent", -1)),
        "pref_scores": scores,
        "overlay_cam_ids": [int(x) for x in overlay_cam_ids],
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # calib.json
    export_calib_json(frame, os.path.join(out_dir, "calib.json"))

    # images
    img_paths = export_clean_images(frame, os.path.join(out_dir, "images"))

    # ego csvs
    wrote_past = write_ego_csv(os.path.join(out_dir, "ego_past.csv"), ego["past"], kind="past", fps=fps, speed_eps=speed_eps)
    wrote_fut  = write_ego_csv(os.path.join(out_dir, "ego_future.csv"), ego["future"], kind="future_or_pref", fps=fps, speed_eps=speed_eps)

    # pref csvs
    pref_files = []
    for p in prefs:
        if p["xyz"] is None:
            continue
        fn = f"pref_{p['i']}_score{int(round(p['score']))}.csv"
        write_pref_csv(os.path.join(out_dir, fn), p["xyz"], fps=fps)
        pref_files.append(fn)

    # overlays
    overlay_reports = {}
    for cid in overlay_cam_ids:
        out_img = os.path.join(out_dir, "overlays", f"cam_{cid}_overlay.jpg")
        overlay_reports[str(cid)] = export_overlay(frame, ego, prefs, cam_id=int(cid), out_path=out_img, fps=fps)

    # manifest.json
    manifest = {
        "files": {
            "meta": "meta.json",
            "calib": "calib.json",
            "ego_past": "ego_past.csv" if wrote_past else None,
            "ego_future": "ego_future.csv" if wrote_fut else None,
            "pref_csvs": pref_files,
            "images": {str(k): os.path.relpath(v, out_dir).replace("\\","/") for k,v in img_paths.items()},
            "overlays": {str(cid): f"overlays/cam_{cid}_overlay.jpg" for cid in overlay_cam_ids},
        },
        "stats": {
            "num_images": int(len(img_paths)),
            "past_len": None if ego["past"]["pos"] is None else int(len(ego["past"]["pos"])),
            "future_len": None if ego["future"]["pos"] is None else int(len(ego["future"]["pos"])),
            "num_prefs": int(len(prefs)),
        },
        "overlay_reports": overlay_reports
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(done_flag, "w", encoding="utf-8") as f:
        f.write("OK\n")

    return {"skipped": False, "out_dir": out_dir}


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, help="Path to val_index.csv")
    ap.add_argument("--out_root", required=True, help="Output root folder for per-sample folders")
    ap.add_argument("--fps", type=float, default=10.0, help="Assumed fps (relative timeline)")
    ap.add_argument("--speed_eps", type=float, default=0.2, help="Speed threshold below which heading set to NaN")
    ap.add_argument("--overlay_cams", default="1,7", help="Comma-separated cam ids for overlays, default=1,7")
    ap.add_argument("--max_rows", type=int, default=0, help="Debug: process only first N rows (0=all)")
    ap.add_argument("--skip_done", action="store_true", help="Skip folders that already have DONE.txt")
    args = ap.parse_args()

    index_csv = normalize_path(args.index_csv)
    out_root = normalize_path(args.out_root)
    ensure_dir(out_root)

    df = pd.read_csv(index_csv)
    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows))

    overlay_cam_ids = tuple(int(x.strip()) for x in args.overlay_cams.split(",") if x.strip())

    # tqdm optional
    try:
        from tqdm import tqdm
        it = tqdm(df.to_dict(orient="records"), total=len(df), desc="Exporting scenes")
    except Exception:
        it = df.to_dict(orient="records")

    n_ok, n_skip, n_fail = 0, 0, 0
    for row in it:
        try:
            r = export_one_sample(
                row=row,
                out_root=out_root,
                fps=args.fps,
                speed_eps=args.speed_eps,
                overlay_cam_ids=overlay_cam_ids,
                skip_done=args.skip_done,
            )
            if r.get("skipped"):
                n_skip += 1
            else:
                n_ok += 1
        except Exception as e:
            n_fail += 1
            # 失败也记录一下，方便你定位
            err_path = os.path.join(out_root, "_errors.log")
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat(timespec='seconds')}] FAIL row={row.get('context_name','')} err={repr(e)}\n")

    print(f"done. ok={n_ok}, skipped={n_skip}, failed={n_fail}")
    print("out_root:", out_root)
    if n_fail > 0:
        print("see:", os.path.join(out_root, "_errors.log"))


if __name__ == "__main__":
    main()
