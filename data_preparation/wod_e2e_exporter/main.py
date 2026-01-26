#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - CLI 入口

职责：
- CLI 参数解析
- 调度所有模块
- 多进程/线程控制
- 全局日志
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .exporters import export_full_segment
from .io_reader import SegmentRecord, read_segment_from_tfrecords
from .utils import ensure_dir, safe_json_dump, setup_logger, to_wsl_path_if_needed

# 尝试导入 tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False


# ============================================================================
# 配置数据结构
# ============================================================================

@dataclass
class ExportConfig:
    """导出配置"""
    dataset_root: Path
    out_root: Path
    split: str = "val"
    scenario_cluster: str = "ALL"
    max_segments: Optional[int] = None
    num_workers: int = 1
    overwrite: bool = False
    fail_fast: bool = False
    log_level: str = "INFO"
    ego_state_hz: float = 10.0


# ============================================================================
# Segment 发现与索引
# ============================================================================

def discover_segments(
    dataset_root: Path,
    split: str = "val",
    scenario_cluster: str = "ALL",
) -> List[Dict[str, Any]]:
    """
    发现数据集中的所有 segments
    
    Args:
        dataset_root: 数据集根目录
        split: 数据集分割 (val/train/test)
        scenario_cluster: 场景类别过滤（ALL 表示所有）
    
    Returns:
        segment 信息列表
    """
    segments = []
    
    # WOD-E2E 的典型目录结构:
    # dataset_root/
    #   {split}/
    #     {scenario_cluster}/
    #       *.tfrecord
    
    split_dir = dataset_root / split
    if not split_dir.exists():
        # 尝试其他可能的结构
        split_dir = dataset_root
    
    # 查找所有 tfrecord 文件
    tfrecord_files = list(split_dir.rglob("*.tfrecord"))
    
    if not tfrecord_files:
        # 尝试查找 tfrecord-* 格式
        tfrecord_files = list(split_dir.rglob("tfrecord-*"))
    
    for tf_path in tfrecord_files:
        # 从路径推断 scenario_cluster
        rel_path = tf_path.relative_to(split_dir) if tf_path.is_relative_to(split_dir) else tf_path
        parts = rel_path.parts
        
        sc = parts[0] if len(parts) > 1 else "unknown"
        
        # 过滤 scenario_cluster
        if scenario_cluster != "ALL" and sc != scenario_cluster:
            continue
        
        # 从文件名推断 segment ID
        seg_id = tf_path.stem
        
        segments.append({
            "seg_id": seg_id,
            "split": split,
            "scenario_cluster": sc,
            "tfrecord_path": str(tf_path),
        })
    
    return segments


def load_segment_index_csv(
    index_csv: Path,
    split: str = "val",
    scenario_cluster: str = "ALL",
) -> List[Dict[str, Any]]:
    """
    从索引 CSV 加载 segment 列表
    
    CSV 格式应包含: seg_id, tfrecord_file, record_idx, class (scenario_cluster)
    """
    import pandas as pd
    
    df = pd.read_csv(index_csv)
    
    # 标准化列名
    col_mapping = {
        "segment_id": "seg_id",
        "tfrecord": "tfrecord_file",
        "class": "scenario_cluster",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # 过滤
    if scenario_cluster != "ALL" and "scenario_cluster" in df.columns:
        df = df[df["scenario_cluster"] == scenario_cluster]
    
    # 按 segment 分组
    segments = []
    if "seg_id" in df.columns:
        for seg_id, group in df.groupby("seg_id"):
            tfrecord_files = group["tfrecord_file"].unique().tolist() if "tfrecord_file" in group.columns else []
            record_indices = group["record_idx"].tolist() if "record_idx" in group.columns else None
            sc = group["scenario_cluster"].iloc[0] if "scenario_cluster" in group.columns else "unknown"
            
            segments.append({
                "seg_id": str(seg_id),
                "split": split,
                "scenario_cluster": sc,
                "tfrecord_paths": [to_wsl_path_if_needed(p) for p in tfrecord_files],
                "record_indices": record_indices,
            })
    
    return segments


# ============================================================================
# 单 Segment 处理
# ============================================================================

def process_single_segment(
    seg_info: Dict[str, Any],
    config: ExportConfig,
    logger: logging.Logger,
) -> Tuple[str, bool, List[str]]:
    """
    处理单个 segment
    
    Returns:
        (seg_id, success, errors)
    """
    seg_id = seg_info["seg_id"]
    split = seg_info.get("split", config.split)
    scenario_cluster = seg_info.get("scenario_cluster", "unknown")
    
    # 构建输出目录
    output_dir = config.out_root / scenario_cluster / f"{scenario_cluster}_segment-{seg_id}"
    
    # 检查是否已存在
    if output_dir.exists() and not config.overwrite:
        logger.info(f"Skipping {seg_id} (already exists)")
        return seg_id, True, []
    
    # 清理已存在的目录（如果 overwrite=True）
    if output_dir.exists() and config.overwrite:
        import shutil
        shutil.rmtree(output_dir)
    
    # 读取数据
    tfrecord_paths = seg_info.get("tfrecord_paths", [])
    if not tfrecord_paths and "tfrecord_path" in seg_info:
        tfrecord_paths = [seg_info["tfrecord_path"]]
    
    record_indices = seg_info.get("record_indices")
    
    try:
        segment = read_segment_from_tfrecords(
            tfrecord_paths=tfrecord_paths,
            seg_id=seg_id,
            split=split,
            scenario_cluster=scenario_cluster,
            record_indices=record_indices,
        )
        
        if not segment.frames:
            return seg_id, False, ["No frames found in segment"]
        
        # 导出
        success, errors = export_full_segment(
            segment=segment,
            output_base=output_dir,
            ego_state_hz=config.ego_state_hz,
        )
        
        return seg_id, success, errors
        
    except Exception as e:
        import traceback
        return seg_id, False, [f"Exception: {str(e)}\n{traceback.format_exc()}"]


# ============================================================================
# 批量处理
# ============================================================================

def process_segments_sequential(
    segments: List[Dict[str, Any]],
    config: ExportConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """顺序处理所有 segments"""
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
        "errors": {},
    }
    
    iterator = tqdm(segments, desc="Exporting") if HAS_TQDM else segments
    
    for seg_info in iterator:
        seg_id, success, errors = process_single_segment(seg_info, config, logger)
        
        if success:
            if not errors:
                results["success"].append(seg_id)
            else:
                results["skipped"].append(seg_id)
        else:
            results["failed"].append(seg_id)
            results["errors"][seg_id] = errors
            
            if config.fail_fast:
                logger.error(f"Fail fast: stopping due to error in {seg_id}")
                break
        
        logger.info(f"Processed {seg_id}: {'SUCCESS' if success else 'FAILED'}")
    
    return results


def process_segments_parallel(
    segments: List[Dict[str, Any]],
    config: ExportConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """并行处理所有 segments"""
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
        "errors": {},
    }
    
    # 使用进程池
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_single_segment, seg_info, config, logger): seg_info
            for seg_info in segments
        }
        
        iterator = as_completed(futures)
        if HAS_TQDM:
            iterator = tqdm(iterator, total=len(futures), desc="Exporting")
        
        for future in iterator:
            seg_info = futures[future]
            seg_id = seg_info["seg_id"]
            
            try:
                seg_id, success, errors = future.result()
                
                if success:
                    if not errors:
                        results["success"].append(seg_id)
                    else:
                        results["skipped"].append(seg_id)
                else:
                    results["failed"].append(seg_id)
                    results["errors"][seg_id] = errors
                    
                    if config.fail_fast:
                        logger.error(f"Fail fast: stopping due to error in {seg_id}")
                        executor.shutdown(wait=False)
                        break
                        
            except Exception as e:
                results["failed"].append(seg_id)
                results["errors"][seg_id] = [str(e)]
    
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="WOD-E2E Data Exporter - Export Waymo End-to-End driving data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export val split
  python -m wod_e2e_exporter.main --dataset_root /data/WOD_E2E --out_root /output

  # Export specific scenario cluster
  python -m wod_e2e_exporter.main --dataset_root /data/WOD_E2E --out_root /output --scenario_cluster Cut_ins

  # Parallel export with 4 workers
  python -m wod_e2e_exporter.main --dataset_root /data/WOD_E2E --out_root /output --num_workers 4
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root directory"
    )
    
    # 可选参数
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "test"],
        help="Dataset split (default: val)"
    )
    parser.add_argument(
        "--scenario_cluster",
        type=str,
        default="ALL",
        help="Scenario cluster filter (default: ALL)"
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Maximum number of segments to process"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Overwrite existing segments (default: false)"
    )
    parser.add_argument(
        "--fail_fast",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Stop on first error (default: false)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--index_csv",
        type=str,
        default=None,
        help="Path to segment index CSV (optional)"
    )
    parser.add_argument(
        "--ego_state_hz",
        type=float,
        default=10.0,
        help="Ego state sampling rate in Hz (default: 10.0)"
    )
    
    return parser.parse_args()


def main() -> int:
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = ExportConfig(
        dataset_root=Path(to_wsl_path_if_needed(args.dataset_root)),
        out_root=Path(to_wsl_path_if_needed(args.out_root)),
        split=args.split,
        scenario_cluster=args.scenario_cluster,
        max_segments=args.max_segments,
        num_workers=args.num_workers,
        overwrite=args.overwrite.lower() == "true",
        fail_fast=args.fail_fast.lower() == "true",
        log_level=args.log_level,
        ego_state_hz=args.ego_state_hz,
    )
    
    # 创建输出目录
    ensure_dir(config.out_root)
    
    # 设置日志
    log_file = config.out_root / "_logs" / "export.log"
    ensure_dir(log_file.parent)
    logger = setup_logger("wod_e2e_exporter", log_file, config.log_level)
    
    logger.info("=" * 60)
    logger.info("WOD-E2E Data Exporter")
    logger.info("=" * 60)
    logger.info(f"Dataset root: {config.dataset_root}")
    logger.info(f"Output root: {config.out_root}")
    logger.info(f"Split: {config.split}")
    logger.info(f"Scenario cluster: {config.scenario_cluster}")
    logger.info(f"Workers: {config.num_workers}")
    logger.info(f"Overwrite: {config.overwrite}")
    logger.info(f"Fail fast: {config.fail_fast}")
    
    # 发现 segments
    if args.index_csv:
        index_csv = Path(to_wsl_path_if_needed(args.index_csv))
        logger.info(f"Loading segments from index CSV: {index_csv}")
        segments = load_segment_index_csv(index_csv, config.split, config.scenario_cluster)
    else:
        logger.info("Discovering segments from dataset...")
        segments = discover_segments(config.dataset_root, config.split, config.scenario_cluster)
    
    if not segments:
        logger.error("No segments found!")
        return 1
    
    logger.info(f"Found {len(segments)} segments")
    
    # 限制数量
    if config.max_segments is not None and config.max_segments < len(segments):
        segments = segments[:config.max_segments]
        logger.info(f"Limited to {config.max_segments} segments")
    
    # 处理
    start_time = time.time()
    
    if config.num_workers > 1:
        results = process_segments_parallel(segments, config, logger)
    else:
        results = process_segments_sequential(segments, config, logger)
    
    elapsed = time.time() - start_time
    
    # 总结
    logger.info("=" * 60)
    logger.info("Export Summary")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Success: {len(results['success'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Skipped: {len(results['skipped'])}")
    
    if results["failed"]:
        logger.warning(f"Failed segments: {results['failed']}")
        for seg_id, errors in results["errors"].items():
            logger.warning(f"  {seg_id}: {errors}")
    
    # 保存结果摘要
    summary_path = config.out_root / "_logs" / "export_summary.json"
    safe_json_dump({
        "config": {
            "dataset_root": str(config.dataset_root),
            "out_root": str(config.out_root),
            "split": config.split,
            "scenario_cluster": config.scenario_cluster,
            "num_workers": config.num_workers,
        },
        "results": {
            "total": len(segments),
            "success": len(results["success"]),
            "failed": len(results["failed"]),
            "skipped": len(results["skipped"]),
            "failed_segments": results["failed"],
        },
        "timing": {
            "total_seconds": elapsed,
            "avg_per_segment": elapsed / len(segments) if segments else 0,
        }
    }, summary_path)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("Done!")
    
    # 返回状态码
    if results["failed"] and config.fail_fast:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
