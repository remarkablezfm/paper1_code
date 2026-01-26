"""
slicer.py - 基于 critical moment 生成 5s 子切片

职责：
- 从 20s SegmentRecord 切出 critical 5s 子集
- 生成内存对象，不写文件（由 exporters 负责）
- 时间重映射（5s 窗口从 0.0 开始）
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .io_reader import (
    CameraImage,
    EgoState,
    FrameData,
    RaterTrajectory,
    SegmentRecord,
    TrajectoryPoint,
)
from .time_align import (
    CAMERA_HZ,
    CRITICAL_DURATION_SEC,
    CriticalMoment,
    FRAMES_5S_10HZ,
    TRAJECTORY_HZ,
    frame_idx_to_t_sec,
    generate_t_sec_array,
    map_20s_frame_to_5s_frame,
    t_sec_to_frame_idx,
    validate_critical_moment,
)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SlicedSegment:
    """切片后的 5s segment 数据"""
    # 原始 segment 信息
    parent_seg_id: str
    seg_id: str  # 与 parent 相同
    split: str
    scenario_cluster: str
    routing_command: str
    
    # Critical moment 映射信息
    critical_timestamp_in_parent_sec: float
    parent_frame_index_start_10hz: int
    
    # 5s 窗口内的帧（时间已重映射为 0-5s）
    frames: List[FrameData] = field(default_factory=list)
    
    # 5s 窗口内的 ego 状态序列
    ego_states: List[EgoState] = field(default_factory=list)
    
    # 5s 窗口内的 ego future 轨迹
    ego_future_states: List[EgoState] = field(default_factory=list)
    
    # Rater 轨迹（5s 窗口内，时间重映射）
    rater_trajectories: List[RaterTrajectory] = field(default_factory=list)
    
    # 相机标定（复制自 parent）
    camera_calibration: Optional[any] = None
    
    # 警告信息
    warnings: List[str] = field(default_factory=list)
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def duration_sec(self) -> float:
        return CRITICAL_DURATION_SEC
    
    def get_ego_state_dataframe(self) -> pd.DataFrame:
        """获取 ego state DataFrame"""
        from .io_reader import ego_states_to_dataframe
        return ego_states_to_dataframe(self.ego_states)
    
    def get_ego_future_dataframe(self) -> pd.DataFrame:
        """获取 ego future DataFrame"""
        from .io_reader import ego_states_to_dataframe
        return ego_states_to_dataframe(self.ego_future_states)


# ============================================================================
# 切片函数
# ============================================================================

def slice_frames(
    frames: List[FrameData],
    critical_moment: CriticalMoment
) -> Tuple[List[FrameData], List[str]]:
    """
    切片帧数据到 5s 窗口
    
    Args:
        frames: 20s 全量帧列表
        critical_moment: critical moment 信息
    
    Returns:
        (sliced_frames, warnings)
    """
    warnings = []
    sliced = []
    
    start_frame = critical_moment.frame_start_10hz
    end_frame = critical_moment.frame_end_10hz
    
    # 预期帧数
    expected_count = FRAMES_5S_10HZ
    
    for frame in frames:
        if start_frame <= frame.frame_idx < end_frame:
            # 深拷贝帧数据
            new_frame = copy.deepcopy(frame)
            
            # 重映射帧索引和时间
            new_frame.frame_idx = frame.frame_idx - start_frame
            new_frame.t_sec = frame_idx_to_t_sec(new_frame.frame_idx, CAMERA_HZ)
            
            # 重映射 ego_state 时间
            if new_frame.ego_state:
                new_frame.ego_state.t_sec = new_frame.t_sec
            
            sliced.append(new_frame)
    
    # 检查帧数
    if len(sliced) < expected_count:
        warnings.append(
            f"Expected {expected_count} frames in 5s window, got {len(sliced)}"
        )
    
    return sliced, warnings


def slice_ego_states(
    states: List[EgoState],
    critical_moment: CriticalMoment,
    hz: float
) -> Tuple[List[EgoState], List[str]]:
    """
    切片 ego 状态序列到 5s 窗口
    
    Args:
        states: 原始 ego 状态列表
        critical_moment: critical moment 信息
        hz: 状态采样率
    
    Returns:
        (sliced_states, warnings)
    """
    warnings = []
    sliced = []
    
    if not states:
        return sliced, warnings
    
    # 计算 5s 窗口的时间范围
    t_start = critical_moment.timestamp_sec
    t_end = critical_moment.window_end_sec
    
    for state in states:
        # 判断是否在窗口内
        if t_start - 0.001 <= state.t_sec < t_end + 0.001:
            new_state = copy.deepcopy(state)
            # 重映射时间
            new_state.t_sec = state.t_sec - t_start
            sliced.append(new_state)
    
    return sliced, warnings


def slice_rater_trajectories(
    rater_trajs: List[RaterTrajectory],
    critical_moment: CriticalMoment
) -> Tuple[List[RaterTrajectory], List[str]]:
    """
    切片 rater 轨迹到 5s 窗口
    
    Args:
        rater_trajs: 原始 rater 轨迹列表
        critical_moment: critical moment 信息
    
    Returns:
        (sliced_trajectories, warnings)
    """
    warnings = []
    sliced = []
    
    t_start = critical_moment.timestamp_sec
    t_end = critical_moment.window_end_sec
    
    for rater in rater_trajs:
        new_rater = RaterTrajectory(
            rater_idx=rater.rater_idx,
            score=rater.score,
            points=[]
        )
        
        for point in rater.points:
            if t_start - 0.001 <= point.t_sec < t_end + 0.001:
                new_point = copy.deepcopy(point)
                new_point.t_sec = point.t_sec - t_start
                new_rater.points.append(new_point)
        
        sliced.append(new_rater)
    
    return sliced, warnings


def infer_critical_moment(segment: SegmentRecord) -> Optional[CriticalMoment]:
    """
    从 segment 数据推断 critical moment
    
    优先顺序：
    1. 直接从 segment 的解析结果获取
    2. 如果没有，使用最后 5s 作为默认（但会发出警告）
    
    Returns:
        CriticalMoment 或 None
    """
    # 方法 1：直接使用已解析的值
    if segment.critical_timestamp_sec is not None:
        return CriticalMoment(
            timestamp_sec=segment.critical_timestamp_sec,
            frame_index_10hz=segment.critical_frame_index_10hz or int(
                segment.critical_timestamp_sec * CAMERA_HZ
            ),
        )
    
    # 方法 2：使用 ego_future 的起始时间作为 critical moment
    # （假设 future 轨迹从 critical moment 开始）
    if segment.ego_future_states:
        first_future = segment.ego_future_states[0]
        # 需要知道 future 在 20s 中的绝对时间
        # 这取决于数据的具体格式
    
    # 方法 3：使用 segment 的最后 5s（fallback）
    if segment.frames:
        total_duration = segment.duration_sec
        if total_duration >= CRITICAL_DURATION_SEC:
            # 使用最后 5s
            ts = total_duration - CRITICAL_DURATION_SEC
            return CriticalMoment(
                timestamp_sec=ts,
                frame_index_10hz=t_sec_to_frame_idx(ts, CAMERA_HZ),
            )
    
    return None


# ============================================================================
# 主切片接口
# ============================================================================

def slice_segment_to_5s(
    segment: SegmentRecord,
    critical_moment: Optional[CriticalMoment] = None
) -> Tuple[Optional[SlicedSegment], List[str]]:
    """
    将 20s segment 切片为 critical 5s
    
    Args:
        segment: 20s SegmentRecord
        critical_moment: critical moment（可选，None 则自动推断）
    
    Returns:
        (SlicedSegment, errors) - 如果切片失败返回 (None, errors)
    """
    errors = []
    warnings = []
    
    # 获取或推断 critical moment
    if critical_moment is None:
        critical_moment = infer_critical_moment(segment)
    
    if critical_moment is None:
        errors.append("Cannot determine critical moment for slicing")
        return None, errors
    
    # 验证 critical moment
    is_valid, err_msg = validate_critical_moment(critical_moment, segment.duration_sec)
    if not is_valid:
        errors.append(f"Invalid critical moment: {err_msg}")
        return None, errors
    
    # 创建切片结果
    sliced = SlicedSegment(
        parent_seg_id=segment.seg_id,
        seg_id=segment.seg_id,
        split=segment.split,
        scenario_cluster=segment.scenario_cluster,
        routing_command=segment.routing_command,
        critical_timestamp_in_parent_sec=critical_moment.timestamp_sec,
        parent_frame_index_start_10hz=critical_moment.frame_index_10hz,
        camera_calibration=segment.camera_calibration,
    )
    
    # 切片帧数据
    sliced_frames, frame_warnings = slice_frames(segment.frames, critical_moment)
    sliced.frames = sliced_frames
    warnings.extend(frame_warnings)
    
    # 从切片帧提取 ego states
    for frame in sliced.frames:
        if frame.ego_state:
            sliced.ego_states.append(frame.ego_state)
    
    # 切片 ego future（需要时间对齐）
    if segment.ego_future_states:
        # ego_future 通常是从 critical moment 开始的 5s 未来轨迹
        # 直接使用，但重映射时间从 0 开始
        sliced_future = []
        for i, state in enumerate(segment.ego_future_states):
            new_state = copy.deepcopy(state)
            new_state.t_sec = i / TRAJECTORY_HZ
            sliced_future.append(new_state)
        sliced.ego_future_states = sliced_future
    
    # 切片 rater 轨迹
    if segment.rater_trajectories:
        # rater 轨迹通常是 5s 窗口内的，直接重映射时间
        for rater in segment.rater_trajectories:
            new_rater = RaterTrajectory(
                rater_idx=rater.rater_idx,
                score=rater.score,
                points=[]
            )
            for i, point in enumerate(rater.points):
                new_point = copy.deepcopy(point)
                new_point.t_sec = i / TRAJECTORY_HZ
                new_rater.points.append(new_point)
            sliced.rater_trajectories.append(new_rater)
    
    sliced.warnings = warnings
    
    return sliced, errors


def create_5s_ego_state_from_frames(
    frames: List[FrameData],
    hz: float
) -> List[EgoState]:
    """
    从帧数据创建 ego state 序列
    
    用于当没有独立的 ego_state 序列时，从帧数据重建
    """
    states = []
    
    for frame in frames:
        if frame.ego_state:
            state = copy.deepcopy(frame.ego_state)
            states.append(state)
        else:
            # 创建占位 state
            state = EgoState(
                t_sec=frame.t_sec,
                x_m=0.0,
                y_m=0.0,
                z_m=0.0,
            )
            states.append(state)
    
    return states


def resample_trajectory_to_hz(
    points: List[TrajectoryPoint],
    target_hz: float,
    duration_sec: float
) -> List[TrajectoryPoint]:
    """
    将轨迹重采样到目标采样率
    
    Args:
        points: 原始轨迹点
        target_hz: 目标采样率
        duration_sec: 目标时长
    
    Returns:
        重采样后的轨迹点列表
    """
    if not points:
        return []
    
    # 目标时间点
    n_target = int(duration_sec * target_hz) + 1
    target_times = generate_t_sec_array(n_target, target_hz)
    
    # 原始时间和值
    src_times = np.array([p.t_sec for p in points])
    src_x = np.array([p.x_m for p in points])
    src_y = np.array([p.y_m for p in points])
    src_z = np.array([p.z_m for p in points])
    
    # 线性插值
    new_x = np.interp(target_times, src_times, src_x)
    new_y = np.interp(target_times, src_times, src_y)
    new_z = np.interp(target_times, src_times, src_z)
    
    # 构建新轨迹点
    result = []
    for i in range(len(target_times)):
        point = TrajectoryPoint(
            t_sec=float(target_times[i]),
            x_m=float(new_x[i]),
            y_m=float(new_y[i]),
            z_m=float(new_z[i]),
        )
        result.append(point)
    
    return result
