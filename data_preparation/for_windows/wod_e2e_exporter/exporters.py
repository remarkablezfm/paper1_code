"""
exporters.py - 写文件模块

职责：
- 写 images/trajectory/meta/manifest/summary（20s 与 5s）
- 调用 schema/time_align/utils/camera_meta
- 不负责数据解析或切片（由 io_reader 和 slicer 负责）
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .camera_meta import CameraCalibration, create_dummy_calibration
from .io_reader import (
    CameraImage,
    EgoState,
    FrameData,
    RaterTrajectory,
    SegmentRecord,
    ego_states_to_dataframe,
)
from .schema import (
    CameraInfo,
    CountsInfo,
    EgoFutureInfo,
    EgoStateInfo,
    RaterLabelsInfo,
    RaterTrajectoryEntry,
    SegmentManifest,
    SegmentSummary,
    TimeInfo,
    TimingInfo,
    create_20s_manifest,
    create_5s_manifest,
    create_summary,
    finalize_summary,
)
from .slicer import SlicedSegment, slice_segment_to_5s
from .time_align import (
    CAMERA_HZ,
    CRITICAL_DURATION_SEC,
    CriticalMoment,
    FULL_DURATION_SEC,
    TRAJECTORY_HZ,
    format_hz_string,
    get_ego_future_filename,
    get_ego_past_filename,
    get_ego_state_filename,
    get_rater_traj_filename,
)
from .utils import (
    cam_id_to_folder_name,
    ensure_dir,
    frame_index_to_filename,
    safe_csv_write,
    safe_json_dump,
)


# ============================================================================
# 图像导出
# ============================================================================

def export_images(
    frames: List[FrameData],
    output_dir: Path,
    summary: SegmentSummary,
) -> int:
    """
    导出相机图像
    
    Args:
        frames: 帧数据列表
        output_dir: 输出目录（images/）
        summary: summary 对象（用于记录警告）
    
    Returns:
        写入的图像数量
    """
    ensure_dir(output_dir)
    image_count = 0
    
    # 创建 8 个相机目录
    for cam_id in range(1, 9):
        cam_dir = output_dir / cam_id_to_folder_name(cam_id)
        ensure_dir(cam_dir)
    
    # 导出每帧的图像
    for frame in frames:
        for cam_id, cam_image in frame.images.items():
            # 确保 cam_id 在 1-8 范围内
            if not (1 <= cam_id <= 8):
                summary.add_warning(
                    type_="invalid_cam_id",
                    message=f"Camera ID {cam_id} out of range 1-8",
                    where=f"frame_{frame.frame_idx}"
                )
                continue
            
            # 构建输出路径
            cam_dir = output_dir / cam_id_to_folder_name(cam_id)
            filename = frame_index_to_filename(frame.frame_idx)
            out_path = cam_dir / filename
            
            # 写入 JPEG
            try:
                out_path.write_bytes(cam_image.jpeg_bytes)
                image_count += 1
            except Exception as e:
                summary.add_warning(
                    type_="image_write_error",
                    message=str(e),
                    where=str(out_path)
                )
    
    return image_count


# ============================================================================
# 轨迹 CSV 导出
# ============================================================================

def get_csv_fields_from_df(df: pd.DataFrame) -> List[str]:
    """从 DataFrame 获取有效字段列表"""
    if df.empty:
        return []
    return [col for col in df.columns if not df[col].isna().all()]


def determine_steering_field(df: pd.DataFrame) -> Optional[str]:
    """
    确定使用哪个转向字段
    
    优先级：
    1. steering_angle_rad
    2. steering_wheel_angle_rad
    3. curvature_1pm
    """
    if "steering_angle_rad" in df.columns and not df["steering_angle_rad"].isna().all():
        return "steering_angle_rad"
    if "steering_wheel_angle_rad" in df.columns and not df["steering_wheel_angle_rad"].isna().all():
        return "steering_wheel_angle_rad"
    if "curvature_1pm" in df.columns and not df["curvature_1pm"].isna().all():
        return "curvature_1pm"
    return None


def validate_ego_state_df(df: pd.DataFrame, summary: SegmentSummary, where: str) -> bool:
    """
    验证 ego_state DataFrame 是否满足最低要求
    
    必须列：t_sec, x_m, y_m, z_m, vx_mps, vy_mps, ax_mps2, ay_mps2, yaw_rad
    """
    required = ["t_sec", "x_m", "y_m", "z_m"]
    velocity_required = ["vx_mps", "vy_mps"]
    accel_required = ["ax_mps2", "ay_mps2"]
    
    missing = [col for col in required if col not in df.columns]
    if missing:
        summary.add_error(
            type_="missing_required_field",
            message=f"Missing required fields: {missing}",
            where=where
        )
        return False
    
    # 检查速度字段
    has_vx_vy = all(col in df.columns and not df[col].isna().all() for col in velocity_required)
    has_speed = "speed_mps" in df.columns and not df["speed_mps"].isna().all()
    
    if not has_vx_vy and not has_speed:
        summary.add_error(
            type_="missing_velocity_field",
            message="Missing velocity fields (vx_mps, vy_mps) or speed_mps",
            where=where
        )
        return False
    
    if not has_vx_vy and has_speed:
        summary.add_warning(
            type_="using_speed_mps",
            message="Using speed_mps instead of vx_mps/vy_mps",
            where=where
        )
    
    # 检查加速度字段
    has_accel = all(col in df.columns and not df[col].isna().all() for col in accel_required)
    if not has_accel:
        summary.add_warning(
            type_="missing_acceleration",
            message="Missing acceleration fields (ax_mps2, ay_mps2)",
            where=where
        )
    
    # 检查 yaw
    if "yaw_rad" not in df.columns or df["yaw_rad"].isna().all():
        summary.add_warning(
            type_="missing_yaw",
            message="Missing yaw_rad field",
            where=where
        )
    
    return True


def export_ego_state_csv(
    states: List[EgoState],
    output_path: Path,
    summary: SegmentSummary,
    hz: float,
) -> Tuple[bool, List[str], Optional[str], bool]:
    """
    导出 ego_state CSV
    
    Returns:
        (success, fields, steering_field, uses_speed_mps)
    """
    df = ego_states_to_dataframe(states)
    
    if df.empty:
        summary.add_error(
            type_="empty_ego_state",
            message="No ego state data to export",
            where=str(output_path)
        )
        return False, [], None, False
    
    # 验证
    is_valid = validate_ego_state_df(df, summary, str(output_path))
    
    # 确保 t_sec 从 0.0 开始
    if df["t_sec"].iloc[0] != 0.0:
        df["t_sec"] = df["t_sec"] - df["t_sec"].iloc[0]
    
    # 确定转向字段
    steering_field = determine_steering_field(df)
    
    # 检查是否使用 speed_mps
    uses_speed = "speed_mps" in df.columns and (
        "vx_mps" not in df.columns or df["vx_mps"].isna().all()
    )
    
    # 获取有效字段
    fields = get_csv_fields_from_df(df)
    
    # 写入 CSV
    safe_csv_write(df, output_path)
    
    return is_valid, fields, steering_field, uses_speed


def export_trajectory_csv(
    states: List[EgoState],
    output_path: Path,
    summary: SegmentSummary,
) -> bool:
    """
    导出轨迹 CSV（ego_past / ego_future）
    
    最低要求：t_sec, x_m, y_m, z_m
    """
    df = ego_states_to_dataframe(states)
    
    if df.empty:
        return False
    
    # 确保 t_sec 从 0.0 开始
    if len(df) > 0 and df["t_sec"].iloc[0] != 0.0:
        df["t_sec"] = df["t_sec"] - df["t_sec"].iloc[0]
    
    # 写入 CSV
    safe_csv_write(df, output_path)
    return True


def export_rater_trajectory_csv(
    rater: RaterTrajectory,
    output_path: Path,
    summary: SegmentSummary,
) -> bool:
    """导出 rater 轨迹 CSV"""
    df = rater.to_dataframe()
    
    if df.empty:
        summary.add_warning(
            type_="empty_rater_trajectory",
            message=f"Rater {rater.rater_idx} trajectory is empty",
            where=str(output_path)
        )
        return False
    
    # 确保 t_sec 从 0.0 开始
    if len(df) > 0 and df["t_sec"].iloc[0] != 0.0:
        df["t_sec"] = df["t_sec"] - df["t_sec"].iloc[0]
    
    safe_csv_write(df, output_path)
    return True


# ============================================================================
# Meta 导出
# ============================================================================

def export_camera_calib(
    calibration: Optional[CameraCalibration],
    output_path: Path,
    summary: SegmentSummary,
) -> bool:
    """导出 camera_calib.json"""
    if calibration is None:
        summary.add_warning(
            type_="missing_calibration",
            message="No camera calibration data, using dummy values",
            where=str(output_path)
        )
        calibration = create_dummy_calibration()
    
    # 检查完整性
    missing = calibration.missing_cameras()
    if missing:
        summary.add_warning(
            type_="incomplete_calibration",
            message=f"Missing calibration for cameras: {missing}",
            where=str(output_path)
        )
    
    ensure_dir(output_path.parent)
    safe_json_dump(calibration.to_dict(), output_path)
    return True


# ============================================================================
# Manifest / Summary 导出
# ============================================================================

def export_manifest(manifest: SegmentManifest, output_path: Path, is_5s: bool = False) -> None:
    """导出 segment_manifest.json"""
    ensure_dir(output_path.parent)
    safe_json_dump(manifest.to_dict(is_5s=is_5s), output_path)


def export_summary(summary: SegmentSummary, output_path: Path) -> None:
    """导出 segment_summary.json"""
    ensure_dir(output_path.parent)
    safe_json_dump(summary.to_dict(), output_path)


# ============================================================================
# 完整 Segment 导出
# ============================================================================

def export_20s_segment(
    segment: SegmentRecord,
    output_dir: Path,
    ego_state_hz: float = CAMERA_HZ,
) -> Tuple[SegmentManifest, SegmentSummary]:
    """
    导出 20s 全量 segment
    
    Args:
        segment: SegmentRecord 对象
        output_dir: 输出目录
        ego_state_hz: ego_state 采样率
    
    Returns:
        (manifest, summary)
    """
    start_ts = time.time()
    summary = create_summary(segment.seg_id)
    
    ensure_dir(output_dir)
    
    # 1. 导出图像
    images_dir = output_dir / "images"
    image_count = export_images(segment.frames, images_dir, summary)
    summary.counts.num_images_written = image_count
    summary.counts.num_frames = len(segment.frames)
    
    # 2. 导出 ego_state
    traj_dir = output_dir / "trajectory"
    ensure_dir(traj_dir)
    
    ego_state_filename = get_ego_state_filename("20s", ego_state_hz)
    ego_state_path = traj_dir / ego_state_filename
    
    # 从帧数据构建 ego states
    ego_states = [f.ego_state for f in segment.frames if f.ego_state]
    
    success, fields, steering_field, uses_speed = export_ego_state_csv(
        ego_states, ego_state_path, summary, ego_state_hz
    )
    
    if success:
        summary.counts.num_csv_written += 1
    
    # 3. 导出 ego_past（如果有）
    ego_past_available = False
    if segment.ego_past_states:
        ego_past_filename = get_ego_past_filename("4s")
        ego_past_path = traj_dir / ego_past_filename
        if export_trajectory_csv(segment.ego_past_states, ego_past_path, summary):
            summary.counts.num_csv_written += 1
            ego_past_available = True
    
    # 4. 导出 ego_future（如果有）
    ego_future_available = False
    ego_future_file = None
    ego_future_reason = None
    
    if segment.ego_future_states:
        ego_future_filename = get_ego_future_filename("5s")
        ego_future_path = traj_dir / ego_future_filename
        if export_trajectory_csv(segment.ego_future_states, ego_future_path, summary):
            summary.counts.num_csv_written += 1
            ego_future_available = True
            ego_future_file = f"trajectory/{ego_future_filename}"
    else:
        if segment.split == "test":
            ego_future_reason = "ego_future not available for test split"
        else:
            ego_future_reason = "ego_future data not found"
            summary.add_warning(
                type_="missing_ego_future",
                message=ego_future_reason,
                where="trajectory/"
            )
    
    # 5. 导出 camera_calib.json
    meta_dir = output_dir / "meta"
    calib_path = meta_dir / "camera_calib.json"
    export_camera_calib(segment.camera_calibration, calib_path, summary)
    summary.counts.num_json_written += 1
    
    # 6. 确定 critical moment
    critical_ts = segment.critical_timestamp_sec
    critical_frame = segment.critical_frame_index_10hz
    
    if critical_ts is None:
        # 使用默认：20s 中的第 15s 开始（最后 5s）
        critical_ts = FULL_DURATION_SEC - CRITICAL_DURATION_SEC
        critical_frame = int(critical_ts * CAMERA_HZ)
        summary.add_warning(
            type_="default_critical_moment",
            message=f"Using default critical moment at {critical_ts}s",
            where="segment"
        )
    
    # 7. 创建 manifest
    manifest = create_20s_manifest(
        seg_id=segment.seg_id,
        split=segment.split,
        scenario_cluster=segment.scenario_cluster,
        routing_command=segment.routing_command,
        critical_timestamp_sec=critical_ts,
        critical_frame_index_10hz=critical_frame,
        ego_state_file=f"trajectory/{ego_state_filename}",
        ego_state_hz=ego_state_hz,
        ego_state_fields=fields,
        steering_field=steering_field,
        ego_future_available=ego_future_available,
        ego_future_file=ego_future_file,
        ego_future_reason=ego_future_reason,
        uses_speed_mps=uses_speed,
    )
    
    # 8. 导出 manifest 和 summary
    export_manifest(manifest, output_dir / "segment_manifest.json", is_5s=False)
    summary.counts.num_json_written += 1
    
    # 9. 创建日志目录
    logs_dir = output_dir / "_logs"
    ensure_dir(logs_dir)
    
    # 完成
    summary = finalize_summary(summary, start_ts)
    export_summary(summary, output_dir / "segment_summary.json")
    summary.counts.num_json_written += 1
    
    return manifest, summary


def export_5s_segment(
    sliced: SlicedSegment,
    output_dir: Path,
    ego_state_hz: float = CAMERA_HZ,
) -> Tuple[SegmentManifest, SegmentSummary]:
    """
    导出 critical 5s segment
    
    Args:
        sliced: SlicedSegment 对象
        output_dir: 输出目录（critical_5s/）
        ego_state_hz: ego_state 采样率
    
    Returns:
        (manifest, summary)
    """
    start_ts = time.time()
    summary = create_summary(sliced.seg_id)
    
    ensure_dir(output_dir)
    
    # 1. 导出图像
    images_dir = output_dir / "images"
    image_count = export_images(sliced.frames, images_dir, summary)
    summary.counts.num_images_written = image_count
    summary.counts.num_frames = len(sliced.frames)
    
    # 2. 导出 ego_state
    traj_dir = output_dir / "trajectory"
    ensure_dir(traj_dir)
    
    ego_state_filename = get_ego_state_filename("0to5s", ego_state_hz)
    ego_state_path = traj_dir / ego_state_filename
    
    success, fields, steering_field, uses_speed = export_ego_state_csv(
        sliced.ego_states, ego_state_path, summary, ego_state_hz
    )
    
    if success:
        summary.counts.num_csv_written += 1
    
    # 3. 导出 ego_future
    ego_future_available = False
    ego_future_file = None
    ego_future_reason = None
    
    if sliced.ego_future_states:
        ego_future_filename = get_ego_future_filename("0to5s")
        ego_future_path = traj_dir / ego_future_filename
        if export_trajectory_csv(sliced.ego_future_states, ego_future_path, summary):
            summary.counts.num_csv_written += 1
            ego_future_available = True
            ego_future_file = f"trajectory/{ego_future_filename}"
    else:
        if sliced.split == "test":
            ego_future_reason = "ego_future not available for test split"
        else:
            ego_future_reason = "ego_future data not found"
    
    # 4. 导出 rater 轨迹（仅 val）
    rater_entries: List[RaterTrajectoryEntry] = []
    
    if sliced.split == "val" and sliced.rater_trajectories:
        for rater in sliced.rater_trajectories:
            rater_filename = get_rater_traj_filename(rater.rater_idx, "0to5s")
            rater_path = traj_dir / rater_filename
            
            if export_rater_trajectory_csv(rater, rater_path, summary):
                summary.counts.num_csv_written += 1
            
            # 记录到 manifest
            score = rater.score if rater.score >= 0 else -1.0
            rater_entries.append(RaterTrajectoryEntry(
                file=f"trajectory/{rater_filename}",
                score=score,
            ))
            
            if score < 0:
                summary.add_warning(
                    type_="missing_rater_score",
                    message=f"Rater {rater.rater_idx} score is missing, using -1.0",
                    where=f"trajectory/{rater_filename}"
                )
    
    # 5. 导出 camera_calib.json（独立完整副本）
    meta_dir = output_dir / "meta"
    calib_path = meta_dir / "camera_calib.json"
    export_camera_calib(sliced.camera_calibration, calib_path, summary)
    summary.counts.num_json_written += 1
    
    # 6. 创建 manifest
    manifest = create_5s_manifest(
        seg_id=sliced.seg_id,
        split=sliced.split,
        scenario_cluster=sliced.scenario_cluster,
        routing_command=sliced.routing_command,
        parent_seg_id=sliced.parent_seg_id,
        critical_timestamp_in_parent_sec=sliced.critical_timestamp_in_parent_sec,
        parent_frame_index_start_10hz=sliced.parent_frame_index_start_10hz,
        ego_state_file=f"trajectory/{ego_state_filename}",
        ego_state_hz=ego_state_hz,
        ego_state_fields=fields,
        steering_field=steering_field,
        ego_future_available=ego_future_available,
        ego_future_file=ego_future_file,
        ego_future_reason=ego_future_reason,
        uses_speed_mps=uses_speed,
        rater_trajectories=rater_entries,
    )
    
    # 7. 导出 manifest
    export_manifest(manifest, output_dir / "segment_manifest.json", is_5s=True)
    summary.counts.num_json_written += 1
    
    # 完成
    summary = finalize_summary(summary, start_ts)
    export_summary(summary, output_dir / "segment_summary.json")
    summary.counts.num_json_written += 1
    
    # 添加 slicer 的警告
    for w in sliced.warnings:
        summary.add_warning(
            type_="slicer_warning",
            message=w,
            where="slicer"
        )
    
    return manifest, summary


def export_full_segment(
    segment: SegmentRecord,
    output_base: Path,
    ego_state_hz: float = CAMERA_HZ,
) -> Tuple[bool, List[str]]:
    """
    完整导出一个 segment（20s + critical 5s）
    
    Args:
        segment: SegmentRecord 对象
        output_base: segment 输出基目录
        ego_state_hz: ego_state 采样率
    
    Returns:
        (success, errors)
    """
    errors = []
    
    # 1. 导出 20s
    try:
        manifest_20s, summary_20s = export_20s_segment(
            segment, output_base, ego_state_hz
        )
        
        if summary_20s.status == "failed":
            errors.append("20s export failed")
            return False, errors
            
    except Exception as e:
        errors.append(f"20s export exception: {str(e)}")
        return False, errors
    
    # 2. 切片为 5s
    critical_moment = CriticalMoment(
        timestamp_sec=manifest_20s.time.critical_timestamp_sec,
        frame_index_10hz=manifest_20s.time.critical_frame_index_10hz,
    )
    
    sliced, slice_errors = slice_segment_to_5s(segment, critical_moment)
    
    if sliced is None:
        errors.extend(slice_errors)
        return False, errors
    
    # 3. 导出 5s
    critical_5s_dir = output_base / "critical_5s"
    
    try:
        manifest_5s, summary_5s = export_5s_segment(
            sliced, critical_5s_dir, ego_state_hz
        )
        
        if summary_5s.status == "failed":
            errors.append("5s export failed")
            # 继续，因为 20s 已成功
            
    except Exception as e:
        errors.append(f"5s export exception: {str(e)}")
    
    return len(errors) == 0, errors
