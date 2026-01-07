#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export WOD-E2E videos (CAM_ID numeric) from all TFRecord shards under val/.

Key points:
- Groups frames by base segment_id extracted from frame.context.name: "<segment_id>-<frame_idx>"
- Saves JPEG frames to a temp folder (fast, low RAM), then encodes MP4 with OpenCV.
- Optional gap filling by repeating last frame to cover missing frame_idx.
"""

import os
import re
import glob
import argparse
import shutil
from typing import Optional, Tuple, Dict

# -------------------------
# Dependency checks
# -------------------------
def _require_imports():
    try:
        import numpy as np  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: numpy. Install with: pip install numpy") from e

    try:
        import cv2  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: opencv-python (cv2). Install with: pip install opencv-python") from e

    try:
        import tensorflow as tf  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: tensorflow. Install per your environment.") from e

    try:
        from waymo_open_dataset import dataset_pb2 as open_dataset  # noqa: F401
        from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: waymo_open_dataset protos. "
            "You need a Waymo Open Dataset package compatible with your TF version.\n"
            "Hint: you already imported these in notebook; ensure same env for running this script."
        ) from e


_require_imports()

import numpy as np
import cv2
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# -------------------------
# Path helpers (Windows / WSL)
# -------------------------
def to_wsl_path(p: str) -> str:
    # If running under Linux/WSL but given Windows path like D:\..., convert to /mnt/d/...
    if os.name != "nt" and re.match(r"^[A-Za-z]:\\", p):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p

# -------------------------
# Proto parse
# -------------------------
def parse_e2ed(raw_bytes: bytes) -> wod_e2ed_pb2.E2EDFrame:
    m = wod_e2ed_pb2.E2EDFrame()
    m.ParseFromString(raw_bytes)
    return m

def get_context_name(frame) -> str:
    if hasattr(frame, "context") and hasattr(frame.context, "name") and frame.context.name:
        return frame.context.name
    return ""

def split_context_name(ctx_name: str) -> Tuple[str, Optional[int]]:
    """
    Expected: "<segment_hex>-<frame_idx>"
    Returns: (segment_id, frame_idx) ; frame_idx can be None if format not matched.
    """
    if not ctx_name or "-" not in ctx_name:
        return ctx_name, None
    base, suf = ctx_name.rsplit("-", 1)
    try:
        return base, int(suf)
    except ValueError:
        return ctx_name, None

def get_cam_jpeg_bytes(frame, cam_id: int) -> Optional[bytes]:
    for img in frame.images:
        if int(img.name) == int(cam_id):
            return img.image
    return None

# -------------------------
# Video encode helpers
# -------------------------
def imread_rgb(path: str) -> Optional[np.ndarray]:
    # Windows-safe binary read
    b = np.fromfile(path, dtype=np.uint8)
    img_bgr = cv2.imdecode(b, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def export_segment_video(
    seg_id: str,
    seg_dir: str,
    out_mp4: str,
    fps: int,
    fill_gaps: bool,
    min_unique_frames: int,
) -> Tuple[bool, str]:
    jpgs = sorted(glob.glob(os.path.join(seg_dir, "*.jpg")))
    if len(jpgs) < min_unique_frames:
        return False, f"skip: too few frames ({len(jpgs)})"

    idxs = []
    idx2path = {}
    for p in jpgs:
        base = os.path.basename(p)
        try:
            idx = int(os.path.splitext(base)[0])
        except ValueError:
            continue
        idxs.append(idx)
        idx2path[idx] = p

    if not idxs:
        return False, "skip: no valid frame_idx jpgs"

    idxs.sort()
    idx_min, idx_max = idxs[0], idxs[-1]

    first_rgb = imread_rgb(idx2path[idx_min])
    if first_rgb is None:
        return False, "skip: cannot read first frame"

    h, w = first_rgb.shape[:2]
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last_rgb = first_rgb
    written = 0

    if fill_gaps:
        for i in range(idx_min, idx_max + 1):
            if i in idx2path:
                rgb = imread_rgb(idx2path[i])
                if rgb is not None:
                    last_rgb = rgb
            vw.write(cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR))
            written += 1
    else:
        for i in idxs:
            rgb = imread_rgb(idx2path[i])
            if rgb is None:
                continue
            vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            written += 1

    vw.release()
    dur = written / float(fps)
    return True, f"ok: frames_written={written}, duration≈{dur:.2f}s, idx=[{idx_min},{idx_max}]"

# -------------------------
# Main pipeline
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help=r"Dataset root, e.g. D:\Datasets\WOD_E2E_Camera_v1")
    ap.add_argument("--split", type=str, default="val", choices=["val", "training"],
                    help="Which split folder to scan (default: val)")
    ap.add_argument("--cam-id", type=int, default=1, help="Numeric camera id (default: 1=FRONT)")
    ap.add_argument("--fps", type=int, default=10, help="FPS for output video (default: 10)")
    ap.add_argument("--fill-gaps", action="store_true", help="Fill missing frame_idx by repeating last frame")
    ap.add_argument("--min-frames", type=int, default=2, help="Minimum unique frames to export a video")
    ap.add_argument("--cleanup", action="store_true", help="Remove temp jpg frames after exporting videos")
    ap.add_argument("--max-shards", type=int, default=0, help="Limit number of tfrecord shards scanned (0=all)")
    args = ap.parse_args()

    data_root = to_wsl_path(args.data_root)
    split_dir = os.path.join(data_root, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    out_dir = os.path.join(data_root, f"videos_{args.split}_cam{args.cam_id}")
    tmp_dir = os.path.join(out_dir, f"_tmp_frames_cam{args.cam_id}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    tfrecords = sorted(glob.glob(os.path.join(split_dir, "*.tfrecord*")))
    if not tfrecords:
        raise FileNotFoundError(f"No tfrecord files under: {split_dir}")

    if args.max_shards and args.max_shards > 0:
        tfrecords = tfrecords[:args.max_shards]

    print(f"[INFO] split_dir: {split_dir}")
    print(f"[INFO] shards: {len(tfrecords)}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] tmp_dir: {tmp_dir}")
    print(f"[INFO] cam_id={args.cam_id}, fps={args.fps}, fill_gaps={args.fill_gaps}, min_frames={args.min_frames}")

    # Pass 1: dump jpeg frames (avoid RAM blow)
    seg_meta: Dict[str, Dict[str, int]] = {}
    jpeg_written = 0

    for si, fp in enumerate(tfrecords, 1):
        comp = "GZIP" if fp.endswith(".gz") else ""
        ds = tf.data.TFRecordDataset(fp, compression_type=comp)

        for r in ds:
            e2e = parse_e2ed(r.numpy())
            fr = e2e.frame

            ctx = get_context_name(fr)
            seg_id, frame_idx = split_context_name(ctx)
            if not seg_id or frame_idx is None:
                continue

            jpeg = get_cam_jpeg_bytes(fr, args.cam_id)
            if jpeg is None:
                continue

            seg_dir = os.path.join(tmp_dir, seg_id)
            os.makedirs(seg_dir, exist_ok=True)

            out_jpg = os.path.join(seg_dir, f"{frame_idx:06d}.jpg")
            if not os.path.exists(out_jpg):
                with open(out_jpg, "wb") as f:
                    f.write(jpeg)
                jpeg_written += 1

            m = seg_meta.get(seg_id)
            if m is None:
                seg_meta[seg_id] = {"count": 1, "idx_min": frame_idx, "idx_max": frame_idx}
            else:
                m["count"] += 1
                if frame_idx < m["idx_min"]:
                    m["idx_min"] = frame_idx
                if frame_idx > m["idx_max"]:
                    m["idx_max"] = frame_idx

        if si % 10 == 0 or si == len(tfrecords):
            print(f"[PASS1] scanned {si}/{len(tfrecords)} shards | segments_seen={len(seg_meta)} | jpeg_written={jpeg_written}")

    print(f"[PASS1 DONE] segments_seen={len(seg_meta)} | jpeg_written={jpeg_written}")

    # Pass 2: export videos per segment
    seg_dirs = sorted(glob.glob(os.path.join(tmp_dir, "*")))
    exported = 0
    processed = 0

    for seg_dir in seg_dirs:
        seg_id = os.path.basename(seg_dir)
        out_mp4 = os.path.join(out_dir, f"{seg_id}_cam{args.cam_id}.mp4")
        if os.path.exists(out_mp4):
            continue

        ok, msg = export_segment_video(
            seg_id=seg_id,
            seg_dir=seg_dir,
            out_mp4=out_mp4,
            fps=args.fps,
            fill_gaps=args.fill_gaps,
            min_unique_frames=args.min_frames,
        )
        processed += 1
        if ok:
            exported += 1

        if processed % 200 == 0:
            print(f"[PASS2] processed={processed} exported={exported} (latest: {seg_id} -> {msg})")

    print(f"[PASS2 DONE] exported={exported} videos | out_dir={out_dir}")

    if args.cleanup:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[CLEANUP] removed tmp_dir: {tmp_dir}")


if __name__ == "__main__":
    main()
