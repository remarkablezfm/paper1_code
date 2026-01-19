"""CLI entrypoint for WOD-E2E exporter."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .io_reader import read_seg_index_enriched, read_val_index_csv, load_segment_from_enriched_rows
from .exporters import export_segment


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export WOD-E2E segments (20s + critical 5s).")
    p.add_argument("--out_root", required=False, help="Output root folder.")
    p.add_argument("--split", choices=["val", "train", "test"], default="val")
    p.add_argument("--scenario_cluster", default="ALL", help="Scenario cluster filter or ALL.")
    p.add_argument("--max_segments", type=int, default=None, help="Max segments to export.")
    p.add_argument("--num_workers", type=int, default=1, help="Parallel workers.")
    p.add_argument("--overwrite", choices=["true", "false"], default="false")
    p.add_argument("--fail_fast", choices=["true", "false"], default="false")
    p.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p.add_argument("--val_index_csv", help="Val index CSV path (if needed for val).")
    p.add_argument("--seg_index_enriched_csv", help="Enriched seg index CSV (seg_id, frame_id, tfrecord_file, record_idx, scenario_cluster).")
    return p


def _parse_bool(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("wod_e2e_exporter")

    if not args.out_root:
        logger.error("--out_root is required.")
        return 2
    if not args.seg_index_enriched_csv:
        logger.error("--seg_index_enriched_csv is required for segment aggregation.")
        return 2

    out_root = Path(args.out_root)
    overwrite = _parse_bool(args.overwrite)
    fail_fast = _parse_bool(args.fail_fast)

    val_index_map = {}
    if args.val_index_csv:
        val_index_map = read_val_index_csv(args.val_index_csv)
        logger.info("Loaded val index rows=%d", len(val_index_map))

    idx = read_seg_index_enriched(args.seg_index_enriched_csv)
    logger.info("Loaded segments=%d from %s", len(idx), args.seg_index_enriched_csv)

    exported = 0
    for seg_id, rows in idx.items():
        if args.max_segments is not None and exported >= args.max_segments:
            break

        meta = val_index_map.get(seg_id, {})
        scenario_cluster = meta.get("scenario_cluster") or rows[0].get("scenario_cluster", "UNKNOWN")
        if args.scenario_cluster != "ALL" and scenario_cluster != args.scenario_cluster:
            continue

        critical_idx = meta.get("critical_frame_index_10hz", None)
        routing_command = meta.get("routing_command", None)

        try:
            seg = load_segment_from_enriched_rows(
                seg_id,
                rows,
                split=args.split,
                routing_command=routing_command,
                scenario_cluster=scenario_cluster,
                critical_frame_index_10hz=critical_idx,
            )
            out = export_segment(seg, out_root, overwrite=overwrite)
            logger.info("exported seg_id=%s status=%s", seg_id, out.get("status"))
            exported += 1
        except Exception as e:
            logger.exception("failed exporting seg_id=%s error=%s", seg_id, e)
            if fail_fast:
                return 1

    logger.info("exported=%d", exported)
    return 0


if __name__ == "__main__":
    sys.exit(main())
