"""Scan TFRecord files to build seg_id -> (tfrecord_file, record_idx) mapping.

Usage:
  python -m wod_e2e_exporter.scan_segments --dataset_root /path/to/WOD_E2E_Camera_v1 --split val --out seg_index_val.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import tensorflow as tf

from .io_reader import parse_e2ed
from .utils import win_to_wsl_path


def iter_tfrecord_files(dataset_root: str, split: str) -> List[str]:
    """List TFRecord files under dataset_root/split (non-recursive glob for *.tfrecord-*)."""
    split_dir = Path(win_to_wsl_path(dataset_root)) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    files = sorted(str(p) for p in split_dir.glob("*.tfrecord-*"))
    if not files:
        raise FileNotFoundError(f"no tfrecord files found under {split_dir}")
    return files


def scan_file(tfrecord_path: str) -> Iterable[Tuple[str, str, int]]:
    """Yield (seg_id, tfrecord_path, record_idx) for each record in the file."""
    tfp = win_to_wsl_path(tfrecord_path)
    ds = tf.data.TFRecordDataset(tfp, compression_type="")
    for i, rec in enumerate(ds):
        raw = bytes(rec.numpy())
        e2e = parse_e2ed(raw)
        frame = e2e.frame
        seg_id = getattr(frame.context, "name", "unknown_seg")
        yield (seg_id, tfp, i)


def build_index(dataset_root: str, split: str, out_csv: str) -> None:
    files = iter_tfrecord_files(dataset_root, split)
    rows: List[Tuple[str, str, int]] = []
    for tfp in files:
        print(f"[SCAN] {tfp}")
        rows.extend(list(scan_file(tfp)))

    print(f"[WRITE] {out_csv} rows={len(rows)}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)  
        w.writerow(["seg_id", "tfrecord_file", "record_idx"])
        for seg_id, tfp, idx in rows:
            w.writerow([seg_id, tfp, idx])


def main():
    ap = argparse.ArgumentParser(description="Scan WOD-E2E TFRecord files to build segment index.")
    ap.add_argument("--dataset_root", required=True, help="Dataset root (contains val/train/test folders).")
    ap.add_argument("--split", choices=["val", "train", "test"], default="val")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    build_index(dataset_root=args.dataset_root, split=args.split, out_csv=args.out)


if __name__ == "__main__":
    main()
