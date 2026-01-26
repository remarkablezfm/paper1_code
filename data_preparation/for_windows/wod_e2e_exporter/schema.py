"""
schema.py - manifest/summary 数据结构 + 生成 + 校验

职责：定义和生成 segment_manifest.json 和 segment_summary.json
Schema version: v3.3-codex
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from . import __schema_version__
from .time_align import (
    CAMERA_HZ,
    CRITICAL_DURATION_SEC,
    FULL_DURATION_SEC,
    TRAJECTORY_HZ,
)


# ============================================================================
# 枚举定义
# ============================================================================

class RoutingCommand(str, Enum):
    """导航命令枚举"""
    GO_STRAIGHT = "GO_STRAIGHT"
    GO_LEFT = "GO_LEFT"
    GO_RIGHT = "GO_RIGHT"
    UNKNOWN = "UNKNOWN"


class ExportStatus(str, Enum):
    """导出状态枚举"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class Split(str, Enum):
    """数据集分割"""
    VAL = "val"
    TRAIN = "train"
    TEST = "test"


# ============================================================================
# Error / Warning 结构
# ============================================================================

@dataclass
class ExportMessage:
    """错误或警告消息"""
    type: str  # 类型标识符
    message: str  # 详细消息
    where: str  # 位置信息（文件/字段等）
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "type": self.type,
            "message": self.message,
            "where": self.where
        }


# ============================================================================
# Manifest 数据结构
# ============================================================================

@dataclass
class TimeInfo:
    """时间信息"""
    segment_t0_sec: float = 0.0
    duration_sec: float = 0.0
    # 仅 20s manifest 需要
    critical_timestamp_sec: Optional[float] = None
    critical_frame_index_10hz: Optional[int] = None
    critical_window_sec: Optional[float] = None
    # 仅 5s manifest 需要
    parent_seg_id: Optional[str] = None
    critical_timestamp_in_parent_sec: Optional[float] = None
    parent_frame_index_start_10hz: Optional[int] = None
    mapping_note: Optional[str] = None
    
    def to_dict(self, is_5s: bool = False) -> Dict[str, Any]:
        d = {
            "segment_t0_sec": self.segment_t0_sec,
            "duration_sec": self.duration_sec,
        }
        if not is_5s:
            # 20s manifest
            if self.critical_timestamp_sec is not None:
                d["critical_timestamp_sec"] = self.critical_timestamp_sec
            if self.critical_frame_index_10hz is not None:
                d["critical_frame_index_10hz"] = self.critical_frame_index_10hz
            if self.critical_window_sec is not None:
                d["critical_window_sec"] = self.critical_window_sec
        else:
            # 5s manifest - parent mapping
            if self.parent_seg_id is not None:
                d["parent_seg_id"] = self.parent_seg_id
            if self.critical_timestamp_in_parent_sec is not None:
                d["critical_timestamp_in_parent_sec"] = self.critical_timestamp_in_parent_sec
            if self.parent_frame_index_start_10hz is not None:
                d["parent_frame_index_start_10hz"] = self.parent_frame_index_start_10hz
            if self.mapping_note is not None:
                d["mapping_note"] = self.mapping_note
        return d


@dataclass
class CameraInfo:
    """相机信息"""
    fps_hz: float = CAMERA_HZ
    num_cameras: int = 8
    camera_ids: List[int] = field(default_factory=lambda: list(range(1, 9)))
    image_dir: str = "images/"
    calibration_file: str = "meta/camera_calib.json"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps_hz": self.fps_hz,
            "num_cameras": self.num_cameras,
            "camera_ids": self.camera_ids,
            "image_dir": self.image_dir,
            "calibration_file": self.calibration_file,
        }


@dataclass
class EgoStateInfo:
    """ego_state 信息"""
    file: str
    hz: float
    fields: List[str]
    steering_field: Optional[str] = None
    uses_speed_mps: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "file": self.file,
            "hz": self.hz,
            "fields": self.fields,
        }
        if self.steering_field:
            d["steering_field"] = self.steering_field
        if self.uses_speed_mps:
            d["uses_speed_mps"] = True
        return d


@dataclass
class EgoFutureInfo:
    """ego_future 信息"""
    available: bool = True
    file: Optional[str] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"available": self.available}
        if self.available and self.file:
            d["file"] = self.file
        if not self.available and self.reason:
            d["reason"] = self.reason
        return d


@dataclass
class RaterTrajectoryEntry:
    """单条 rater 轨迹信息"""
    file: str
    score: float  # -1.0 表示缺失
    
    def to_dict(self) -> Dict[str, Any]:
        return {"file": self.file, "score": self.score}


@dataclass
class RaterLabelsInfo:
    """rater_labels 信息"""
    available: bool = True
    reason: Optional[str] = None
    missing_score_value: float = -1.0
    trajectories: List[RaterTrajectoryEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "available": self.available,
            "missing_score_value": self.missing_score_value,
        }
        if not self.available:
            d["reason"] = self.reason or "not_available_for_this_split"
        if self.available and self.trajectories:
            d["trajectories"] = [t.to_dict() for t in self.trajectories]
        return d


@dataclass
class SegmentManifest:
    """segment_manifest.json 完整结构"""
    schema_version: str = __schema_version__
    seg_id: str = ""
    split: str = "val"
    scenario_cluster: str = ""
    routing_command: str = "UNKNOWN"
    time: TimeInfo = field(default_factory=TimeInfo)
    camera: CameraInfo = field(default_factory=CameraInfo)
    ego_state: Optional[EgoStateInfo] = None
    ego_future: EgoFutureInfo = field(default_factory=EgoFutureInfo)
    rater_labels: RaterLabelsInfo = field(default_factory=RaterLabelsInfo)
    
    def to_dict(self, is_5s: bool = False) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "seg_id": self.seg_id,
            "split": self.split,
            "scenario_cluster": self.scenario_cluster,
            "routing_command": self.routing_command,
            "time": self.time.to_dict(is_5s=is_5s),
            "camera": self.camera.to_dict(),
        }
        if self.ego_state:
            d["ego_state"] = self.ego_state.to_dict()
        d["ego_future"] = self.ego_future.to_dict()
        d["rater_labels"] = self.rater_labels.to_dict()
        return d


# ============================================================================
# Summary 数据结构
# ============================================================================

@dataclass
class CountsInfo:
    """统计计数"""
    num_frames: int = 0
    num_cameras: int = 8
    num_images_written: int = 0
    num_csv_written: int = 0
    num_json_written: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "num_frames": self.num_frames,
            "num_cameras": self.num_cameras,
            "num_images_written": self.num_images_written,
            "num_csv_written": self.num_csv_written,
            "num_json_written": self.num_json_written,
        }


@dataclass
class TimingInfo:
    """导出耗时信息"""
    export_duration_sec: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"export_duration_sec": self.export_duration_sec}
        if self.start_time:
            d["start_time"] = self.start_time
        if self.end_time:
            d["end_time"] = self.end_time
        return d


@dataclass
class SegmentSummary:
    """segment_summary.json 完整结构"""
    schema_version: str = __schema_version__
    seg_id: str = ""
    status: str = "success"
    errors: List[ExportMessage] = field(default_factory=list)
    warnings: List[ExportMessage] = field(default_factory=list)
    counts: CountsInfo = field(default_factory=CountsInfo)
    timing: TimingInfo = field(default_factory=TimingInfo)
    
    def add_error(self, type_: str, message: str, where: str) -> None:
        """添加错误"""
        self.errors.append(ExportMessage(type=type_, message=message, where=where))
        if self.status == "success":
            self.status = "partial"
    
    def add_warning(self, type_: str, message: str, where: str) -> None:
        """添加警告"""
        self.warnings.append(ExportMessage(type=type_, message=message, where=where))
    
    def mark_failed(self) -> None:
        """标记为失败"""
        self.status = "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seg_id": self.seg_id,
            "status": self.status,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "counts": self.counts.to_dict(),
            "timing": self.timing.to_dict(),
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_20s_manifest(
    seg_id: str,
    split: str,
    scenario_cluster: str,
    routing_command: str,
    critical_timestamp_sec: float,
    critical_frame_index_10hz: int,
    ego_state_file: str,
    ego_state_hz: float,
    ego_state_fields: List[str],
    steering_field: Optional[str] = None,
    ego_future_available: bool = True,
    ego_future_file: Optional[str] = None,
    ego_future_reason: Optional[str] = None,
    uses_speed_mps: bool = False,
) -> SegmentManifest:
    """创建 20s manifest"""
    manifest = SegmentManifest(
        seg_id=seg_id,
        split=split,
        scenario_cluster=scenario_cluster,
        routing_command=routing_command,
        time=TimeInfo(
            segment_t0_sec=0.0,
            duration_sec=FULL_DURATION_SEC,
            critical_timestamp_sec=critical_timestamp_sec,
            critical_frame_index_10hz=critical_frame_index_10hz,
            critical_window_sec=CRITICAL_DURATION_SEC,
        ),
        ego_state=EgoStateInfo(
            file=ego_state_file,
            hz=ego_state_hz,
            fields=ego_state_fields,
            steering_field=steering_field,
            uses_speed_mps=uses_speed_mps,
        ),
        ego_future=EgoFutureInfo(
            available=ego_future_available,
            file=ego_future_file,
            reason=ego_future_reason,
        ),
    )
    
    # rater_labels 根据 split 设置
    if split == "val":
        manifest.rater_labels = RaterLabelsInfo(available=True)
    else:
        manifest.rater_labels = RaterLabelsInfo(
            available=False,
            reason=f"rater_labels not available for {split} split"
        )
    
    return manifest


def create_5s_manifest(
    seg_id: str,
    split: str,
    scenario_cluster: str,
    routing_command: str,
    parent_seg_id: str,
    critical_timestamp_in_parent_sec: float,
    parent_frame_index_start_10hz: int,
    ego_state_file: str,
    ego_state_hz: float,
    ego_state_fields: List[str],
    steering_field: Optional[str] = None,
    ego_future_available: bool = True,
    ego_future_file: Optional[str] = None,
    ego_future_reason: Optional[str] = None,
    uses_speed_mps: bool = False,
    rater_trajectories: Optional[List[RaterTrajectoryEntry]] = None,
) -> SegmentManifest:
    """创建 5s manifest"""
    manifest = SegmentManifest(
        seg_id=seg_id,
        split=split,
        scenario_cluster=scenario_cluster,
        routing_command=routing_command,
        time=TimeInfo(
            segment_t0_sec=0.0,
            duration_sec=CRITICAL_DURATION_SEC,
            parent_seg_id=parent_seg_id,
            critical_timestamp_in_parent_sec=critical_timestamp_in_parent_sec,
            parent_frame_index_start_10hz=parent_frame_index_start_10hz,
            mapping_note="nearest",
        ),
        ego_state=EgoStateInfo(
            file=ego_state_file,
            hz=ego_state_hz,
            fields=ego_state_fields,
            steering_field=steering_field,
            uses_speed_mps=uses_speed_mps,
        ),
        ego_future=EgoFutureInfo(
            available=ego_future_available,
            file=ego_future_file,
            reason=ego_future_reason,
        ),
    )
    
    # 5s 的 camera 路径调整
    manifest.camera.image_dir = "images/"
    manifest.camera.calibration_file = "meta/camera_calib.json"
    
    # rater_labels 根据 split 设置
    if split == "val":
        manifest.rater_labels = RaterLabelsInfo(
            available=True,
            trajectories=rater_trajectories or []
        )
    else:
        manifest.rater_labels = RaterLabelsInfo(
            available=False,
            reason=f"rater_labels not available for {split} split"
        )
    
    return manifest


def create_summary(seg_id: str) -> SegmentSummary:
    """创建空白 summary"""
    return SegmentSummary(
        seg_id=seg_id,
        timing=TimingInfo(
            start_time=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    )


def finalize_summary(summary: SegmentSummary, start_ts: float) -> SegmentSummary:
    """完成 summary，填充结束时间和耗时"""
    end_ts = time.time()
    summary.timing.export_duration_sec = round(end_ts - start_ts, 3)
    summary.timing.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return summary


# ============================================================================
# 校验函数
# ============================================================================

def validate_manifest(manifest: SegmentManifest, is_5s: bool = False) -> List[str]:
    """
    校验 manifest 完整性
    
    Returns:
        错误消息列表（空表示通过）
    """
    errors = []
    
    # 必需字段
    if not manifest.seg_id:
        errors.append("seg_id is empty")
    if not manifest.split:
        errors.append("split is empty")
    if not manifest.scenario_cluster:
        errors.append("scenario_cluster is empty")
    
    # 时间信息
    if manifest.time.duration_sec <= 0:
        errors.append("duration_sec must be > 0")
    
    if not is_5s:
        # 20s 必须有 critical 信息
        if manifest.time.critical_timestamp_sec is None:
            errors.append("20s manifest missing critical_timestamp_sec")
        if manifest.time.critical_frame_index_10hz is None:
            errors.append("20s manifest missing critical_frame_index_10hz")
    else:
        # 5s 必须有 parent mapping
        if manifest.time.parent_seg_id is None:
            errors.append("5s manifest missing parent_seg_id")
        if manifest.time.critical_timestamp_in_parent_sec is None:
            errors.append("5s manifest missing critical_timestamp_in_parent_sec")
    
    # ego_state 必须存在
    if manifest.ego_state is None:
        errors.append("ego_state is missing")
    else:
        required_fields = ["t_sec", "x_m", "y_m"]
        for f in required_fields:
            if f not in manifest.ego_state.fields:
                errors.append(f"ego_state missing required field: {f}")
    
    return errors


def validate_summary(summary: SegmentSummary) -> List[str]:
    """
    校验 summary 完整性
    
    Returns:
        错误消息列表
    """
    errors = []
    
    if not summary.seg_id:
        errors.append("seg_id is empty")
    if summary.status not in ["success", "partial", "failed"]:
        errors.append(f"invalid status: {summary.status}")
    
    return errors
