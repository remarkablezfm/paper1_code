"""
time_align.py - 唯一时间轴规则

职责：统一输出 t_sec，禁止各模块自行算时间
所有时间相关计算必须通过本模块进行

硬约束：
- 每个 segment 的第一帧时间必须定义为 0.0s
- 所有 CSV 第一列必须为 t_sec，且第一行必须为 0.0
- 时间生成规则：
  - 相机（10Hz）：t_sec = frame_idx / 10.0
  - 轨迹（4Hz）：t_sec = i / 4.0
  - ego_state：t_sec = i / hz_state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ============================================================================
# 常量定义
# ============================================================================

# 采样率
CAMERA_HZ: float = 10.0  # 相机帧率
TRAJECTORY_HZ: float = 4.0  # 轨迹采样率（ego_past, ego_future, rater_traj）

# 时间窗口
FULL_DURATION_SEC: float = 20.0  # 20s 全量
CRITICAL_DURATION_SEC: float = 5.0  # critical 5s 窗口

# 帧数计算
FRAMES_20S_10HZ: int = int(FULL_DURATION_SEC * CAMERA_HZ)  # 200 帧
FRAMES_5S_10HZ: int = int(CRITICAL_DURATION_SEC * CAMERA_HZ)  # 50 帧
FRAMES_5S_4HZ: int = int(CRITICAL_DURATION_SEC * TRAJECTORY_HZ)  # 20 点


# ============================================================================
# 时间索引转换
# ============================================================================

def frame_idx_to_t_sec(frame_idx: int, hz: float = CAMERA_HZ) -> float:
    """
    将帧索引转换为时间（秒）
    
    Args:
        frame_idx: 帧索引（0-based）
        hz: 采样率
    
    Returns:
        时间（秒），从 0.0 开始
    """
    return float(frame_idx) / float(hz)


def t_sec_to_frame_idx(t_sec: float, hz: float = CAMERA_HZ, rounding: str = "nearest") -> int:
    """
    将时间（秒）转换为帧索引
    
    Args:
        t_sec: 时间（秒）
        hz: 采样率
        rounding: 取整方式 ("nearest", "floor", "ceil")
    
    Returns:
        帧索引（0-based）
    """
    raw_idx = t_sec * hz
    
    if rounding == "floor":
        return int(np.floor(raw_idx))
    elif rounding == "ceil":
        return int(np.ceil(raw_idx))
    else:  # nearest
        return int(np.round(raw_idx))


def generate_t_sec_array(n_points: int, hz: float) -> np.ndarray:
    """
    生成时间序列数组
    
    Args:
        n_points: 点数
        hz: 采样率
    
    Returns:
        从 0.0 开始的时间数组，shape=(n_points,)
    """
    return np.arange(n_points, dtype=np.float64) / hz


# ============================================================================
# Critical Moment 处理
# ============================================================================

@dataclass
class CriticalMoment:
    """Critical moment 数据结构"""
    timestamp_sec: float  # 相对于 20s 起点的时间（秒）
    frame_index_10hz: int  # 对应的 10Hz 帧索引
    window_sec: float = CRITICAL_DURATION_SEC  # 窗口长度（固定 5.0）
    
    @property
    def window_start_sec(self) -> float:
        """5s 窗口起始时间"""
        return self.timestamp_sec
    
    @property
    def window_end_sec(self) -> float:
        """5s 窗口结束时间"""
        return self.timestamp_sec + self.window_sec
    
    @property
    def frame_start_10hz(self) -> int:
        """5s 窗口起始帧索引（10Hz）"""
        return self.frame_index_10hz
    
    @property
    def frame_end_10hz(self) -> int:
        """5s 窗口结束帧索引（10Hz，不含）"""
        return self.frame_index_10hz + FRAMES_5S_10HZ


def validate_critical_moment(
    cm: CriticalMoment,
    total_duration_sec: float = FULL_DURATION_SEC
) -> Tuple[bool, Optional[str]]:
    """
    验证 critical moment 是否有效
    
    Returns:
        (is_valid, error_message)
    """
    # 检查时间戳是否在有效范围内
    if cm.timestamp_sec < 0:
        return False, f"critical_timestamp_sec < 0: {cm.timestamp_sec}"
    
    if cm.window_end_sec > total_duration_sec + 0.1:  # 允许小误差
        return False, f"critical window exceeds 20s: end={cm.window_end_sec}"
    
    # 检查帧索引
    if cm.frame_index_10hz < 0:
        return False, f"critical_frame_index_10hz < 0: {cm.frame_index_10hz}"
    
    expected_frame = t_sec_to_frame_idx(cm.timestamp_sec, CAMERA_HZ)
    if abs(cm.frame_index_10hz - expected_frame) > 1:
        return False, (
            f"frame_index mismatch: got {cm.frame_index_10hz}, "
            f"expected {expected_frame} from t={cm.timestamp_sec}s"
        )
    
    return True, None


# ============================================================================
# 20s <-> 5s 时间映射
# ============================================================================

def map_20s_to_5s_time(t_20s_sec: float, critical_moment: CriticalMoment) -> float:
    """
    将 20s 时间映射到 5s 本地时间
    
    Args:
        t_20s_sec: 20s 视图中的时间（秒）
        critical_moment: critical moment 信息
    
    Returns:
        5s 视图中的本地时间（从 0.0 开始）
    """
    return t_20s_sec - critical_moment.timestamp_sec


def map_5s_to_20s_time(t_5s_sec: float, critical_moment: CriticalMoment) -> float:
    """
    将 5s 本地时间映射回 20s 时间
    
    Args:
        t_5s_sec: 5s 视图中的本地时间（秒）
        critical_moment: critical moment 信息
    
    Returns:
        20s 视图中的时间
    """
    return t_5s_sec + critical_moment.timestamp_sec


def map_20s_frame_to_5s_frame(
    frame_20s: int,
    critical_moment: CriticalMoment,
    hz: float = CAMERA_HZ
) -> int:
    """
    将 20s 帧索引映射到 5s 本地帧索引
    
    Args:
        frame_20s: 20s 视图中的帧索引
        critical_moment: critical moment 信息
        hz: 采样率
    
    Returns:
        5s 视图中的本地帧索引（从 0 开始）
    """
    offset = t_sec_to_frame_idx(critical_moment.timestamp_sec, hz)
    return frame_20s - offset


def map_5s_frame_to_20s_frame(
    frame_5s: int,
    critical_moment: CriticalMoment,
    hz: float = CAMERA_HZ
) -> int:
    """
    将 5s 本地帧索引映射回 20s 帧索引
    """
    offset = t_sec_to_frame_idx(critical_moment.timestamp_sec, hz)
    return frame_5s + offset


# ============================================================================
# 轨迹时间重采样
# ============================================================================

def get_5s_window_indices(
    critical_moment: CriticalMoment,
    hz: float,
    total_points: int
) -> Tuple[int, int]:
    """
    获取 5s 窗口在原始数据中的索引范围
    
    Args:
        critical_moment: critical moment 信息
        hz: 数据采样率
        total_points: 总数据点数
    
    Returns:
        (start_idx, end_idx) - 半开区间 [start, end)
    """
    start_idx = t_sec_to_frame_idx(critical_moment.timestamp_sec, hz, rounding="floor")
    end_idx = t_sec_to_frame_idx(critical_moment.window_end_sec, hz, rounding="ceil")
    
    # 边界裁剪
    start_idx = max(0, start_idx)
    end_idx = min(total_points, end_idx)
    
    return start_idx, end_idx


def slice_trajectory_to_5s(
    t_sec: np.ndarray,
    data: np.ndarray,
    critical_moment: CriticalMoment
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将轨迹数据切片到 5s 窗口，并重新调整时间到从 0 开始
    
    Args:
        t_sec: 原始时间数组
        data: 原始数据（与 t_sec 对应）
        critical_moment: critical moment 信息
    
    Returns:
        (new_t_sec, new_data) - 5s 窗口内的数据，时间从 0.0 开始
    """
    # 找到 5s 窗口内的点
    mask = (
        (t_sec >= critical_moment.timestamp_sec - 0.001) &
        (t_sec < critical_moment.window_end_sec + 0.001)
    )
    
    new_t = t_sec[mask] - critical_moment.timestamp_sec
    new_data = data[mask]
    
    return new_t, new_data


# ============================================================================
# 时间戳格式化
# ============================================================================

def format_hz_string(hz: float) -> str:
    """
    格式化采样率字符串，用于文件名
    
    Args:
        hz: 采样率
    
    Returns:
        例如 "10hz", "4hz"
    """
    if hz == int(hz):
        return f"{int(hz)}hz"
    return f"{hz:.1f}hz"


def format_duration_string(duration_sec: float) -> str:
    """
    格式化时长字符串
    
    Args:
        duration_sec: 时长（秒）
    
    Returns:
        例如 "20s", "5s"
    """
    if duration_sec == int(duration_sec):
        return f"{int(duration_sec)}s"
    return f"{duration_sec:.1f}s"


# ============================================================================
# 文件名生成
# ============================================================================

def get_ego_state_filename(duration_str: str, hz: float) -> str:
    """
    生成 ego_state CSV 文件名
    
    Examples:
        - ego_state_20s_10hz.csv
        - ego_state_0to5s_10hz.csv
    """
    hz_str = format_hz_string(hz)
    return f"ego_state_{duration_str}_{hz_str}.csv"


def get_ego_future_filename(duration_str: str, hz: float = TRAJECTORY_HZ) -> str:
    """
    生成 ego_future CSV 文件名
    
    Examples:
        - ego_future_5s_4hz.csv
        - ego_future_0to5s_4hz.csv
    """
    hz_str = format_hz_string(hz)
    return f"ego_future_{duration_str}_{hz_str}.csv"


def get_ego_past_filename(duration_str: str, hz: float = TRAJECTORY_HZ) -> str:
    """
    生成 ego_past CSV 文件名
    
    Examples:
        - ego_past_4s_4hz.csv
    """
    hz_str = format_hz_string(hz)
    return f"ego_past_{duration_str}_{hz_str}.csv"


def get_rater_traj_filename(rater_idx: int, duration_str: str, hz: float = TRAJECTORY_HZ) -> str:
    """
    生成 rater 轨迹 CSV 文件名
    
    Examples:
        - rater_traj_0_0to5s_4hz.csv
        - rater_traj_1_0to5s_4hz.csv
    """
    hz_str = format_hz_string(hz)
    return f"rater_traj_{rater_idx}_{duration_str}_{hz_str}.csv"
