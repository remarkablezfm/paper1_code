"""
io_reader.py - 读取 TFRecord/原始数据 -> SegmentRecord

职责：只读不写文件，将原始数据转换为内存中的数据结构
参考：_export_cut_ins_package.py 的解析细节
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# TensorFlow 和 Waymo 依赖
# ============================================================================

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError("TensorFlow is required. Install: pip install tensorflow") from e

try:
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
    from waymo_open_dataset import dataset_pb2 as open_dataset
except ImportError as e:
    raise ImportError(
        "waymo-open-dataset is required. "
        "Install: pip install waymo-open-dataset-tf-2-x-x"
    ) from e

from .camera_meta import CameraCalibration, extract_camera_calibration_from_frame
from .time_align import CAMERA_HZ, TRAJECTORY_HZ, generate_t_sec_array
from .utils import to_wsl_path_if_needed


# ============================================================================
# 枚举名称解析（参考 _export_cut_ins_package.py）
# ============================================================================

def enum_name_safely(enum_wrapper: Any, value: Any) -> str:
    """从 protobuf EnumTypeWrapper 安全获取枚举名称"""
    try:
        v = int(value)
    except Exception:
        return str(value)
    try:
        return enum_wrapper.DESCRIPTOR.values_by_number[v].name
    except Exception:
        return str(value)


def cam_name_from_id(cam_id: Any) -> str:
    """获取相机名称"""
    if hasattr(open_dataset, "CameraName"):
        return enum_name_safely(open_dataset.CameraName, cam_id)
    return str(int(cam_id))


def intent_to_routing_command(intent_value: int) -> str:
    """
    将 intent 值转换为 routing_command
    
    WOD Intent enum (参考 end_to_end_driving_data.proto):
    - 0: INTENT_UNKNOWN
    - 1: GO_STRAIGHT  
    - 2: GO_LEFT (turn left)
    - 3: GO_RIGHT (turn right)
    """
    mapping = {
        0: "UNKNOWN",
        1: "GO_STRAIGHT",
        2: "GO_LEFT",
        3: "GO_RIGHT",
    }
    return mapping.get(intent_value, "UNKNOWN")


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class CameraImage:
    """单帧相机图像"""
    cam_id: int
    cam_name: str
    jpeg_bytes: bytes
    timestamp_usec: Optional[int] = None
    # 相机图像元数据（用于投影，如果有的话）
    camera_image_metadata: Optional[Any] = None


@dataclass
class EgoState:
    """自车状态（单个时间点）"""
    t_sec: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    vz_mps: Optional[float] = None
    ax_mps2: Optional[float] = None
    ay_mps2: Optional[float] = None
    az_mps2: Optional[float] = None
    yaw_rad: Optional[float] = None
    yaw_rate_rps: Optional[float] = None
    speed_mps: Optional[float] = None
    steering_angle_rad: Optional[float] = None
    steering_wheel_angle_rad: Optional[float] = None
    curvature_1pm: Optional[float] = None
    roll_rad: Optional[float] = None
    pitch_rad: Optional[float] = None


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    t_sec: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: Optional[float] = None
    vy_mps: Optional[float] = None
    ax_mps2: Optional[float] = None
    ay_mps2: Optional[float] = None
    yaw_rad: Optional[float] = None
    yaw_rate_rps: Optional[float] = None


@dataclass
class RaterTrajectory:
    """评分者轨迹"""
    rater_idx: int
    score: float  # -1.0 表示缺失
    points: List[TrajectoryPoint] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        if not self.points:
            return pd.DataFrame(columns=[
                "t_sec", "x_m", "y_m", "z_m",
                "vx_mps", "vy_mps", "ax_mps2", "ay_mps2",
                "yaw_rad", "yaw_rate_rps"
            ])
        
        data = []
        for p in self.points:
            data.append({
                "t_sec": p.t_sec,
                "x_m": p.x_m,
                "y_m": p.y_m,
                "z_m": p.z_m,
                "vx_mps": p.vx_mps,
                "vy_mps": p.vy_mps,
                "ax_mps2": p.ax_mps2,
                "ay_mps2": p.ay_mps2,
                "yaw_rad": p.yaw_rad,
                "yaw_rate_rps": p.yaw_rate_rps,
            })
        return pd.DataFrame(data)


@dataclass
class FrameData:
    """单帧完整数据"""
    frame_idx: int
    t_sec: float
    images: Dict[int, CameraImage] = field(default_factory=dict)
    ego_state: Optional[EgoState] = None
    camera_calibration: Optional[CameraCalibration] = None
    vehicle_pose_4x4: Optional[np.ndarray] = None
    context_name: str = ""


@dataclass
class SegmentRecord:
    """完整 segment 的数据记录"""
    seg_id: str
    split: str
    scenario_cluster: str
    routing_command: str
    
    # 帧数据（20s 全量）
    frames: List[FrameData] = field(default_factory=list)
    
    # 轨迹数据（全 segment）
    ego_past_states: List[EgoState] = field(default_factory=list)
    ego_future_states: List[EgoState] = field(default_factory=list)
    
    # Rater 轨迹（仅 val）
    rater_trajectories: List[RaterTrajectory] = field(default_factory=list)
    
    # Critical moment（从标注解析）
    critical_timestamp_sec: Optional[float] = None
    critical_frame_index_10hz: Optional[int] = None
    
    # 相机标定（所有帧共享）
    camera_calibration: Optional[CameraCalibration] = None
    
    # 元数据
    source_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def duration_sec(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].t_sec - self.frames[0].t_sec + (1.0 / CAMERA_HZ)
    
    def get_ego_state_dataframe(self) -> pd.DataFrame:
        """将 ego_past + 当前 + ego_future 合并为 DataFrame"""
        # 这里实现取决于具体数据结构
        # 简化版：从帧数据提取
        states = []
        for f in self.frames:
            if f.ego_state:
                states.append(f.ego_state)
        
        if not states:
            return pd.DataFrame()
        
        return ego_states_to_dataframe(states)
    
    def get_ego_past_dataframe(self) -> pd.DataFrame:
        """获取 ego past 轨迹 DataFrame"""
        return ego_states_to_dataframe(self.ego_past_states)
    
    def get_ego_future_dataframe(self) -> pd.DataFrame:
        """获取 ego future 轨迹 DataFrame"""
        return ego_states_to_dataframe(self.ego_future_states)


# ============================================================================
# DataFrame 转换辅助函数
# ============================================================================

def ego_states_to_dataframe(states: List[EgoState]) -> pd.DataFrame:
    """将 EgoState 列表转换为 DataFrame"""
    if not states:
        return pd.DataFrame(columns=[
            "t_sec", "x_m", "y_m", "z_m",
            "vx_mps", "vy_mps", "ax_mps2", "ay_mps2",
            "yaw_rad"
        ])
    
    data = []
    for s in states:
        row = {
            "t_sec": s.t_sec,
            "x_m": s.x_m,
            "y_m": s.y_m,
            "z_m": s.z_m,
            "vx_mps": s.vx_mps,
            "vy_mps": s.vy_mps,
            "vz_mps": s.vz_mps,
            "ax_mps2": s.ax_mps2,
            "ay_mps2": s.ay_mps2,
            "az_mps2": s.az_mps2,
            "yaw_rad": s.yaw_rad,
            "yaw_rate_rps": s.yaw_rate_rps,
            "speed_mps": s.speed_mps,
        }
        # 转向字段
        if s.steering_angle_rad is not None:
            row["steering_angle_rad"] = s.steering_angle_rad
        if s.steering_wheel_angle_rad is not None:
            row["steering_wheel_angle_rad"] = s.steering_wheel_angle_rad
        if s.curvature_1pm is not None:
            row["curvature_1pm"] = s.curvature_1pm
        
        data.append(row)
    
    return pd.DataFrame(data)


# ============================================================================
# 原始数据解析（参考 _export_cut_ins_package.py）
# ============================================================================

def parse_e2ed_frame(raw_bytes: bytes) -> wod_e2ed_pb2.E2EDFrame:
    """解析 E2EDFrame protobuf"""
    msg = wod_e2ed_pb2.E2EDFrame()
    msg.ParseFromString(raw_bytes)
    return msg


def extract_ego_states_scalar_fields(states_msg: Any) -> Dict[str, np.ndarray]:
    """
    从 EgoTrajectoryStates 消息提取字段
    参考 _export_cut_ins_package.py 的 extract_ego_states_scalar_fields
    """
    out: Dict[str, np.ndarray] = {}
    if states_msg is None:
        return out
    
    # 已知字段
    candidates = [
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "accel_x", "accel_y", "accel_z",
        "yaw", "heading"
    ]
    
    for k in candidates:
        if hasattr(states_msg, k):
            v = list(getattr(states_msg, k))
            if len(v) > 0:
                out[k] = np.array(v, dtype=np.float32)
    
    return out


def build_ego_states_from_fields(
    fields: Dict[str, np.ndarray],
    hz: float = TRAJECTORY_HZ
) -> List[EgoState]:
    """从字段字典构建 EgoState 列表"""
    # 确定长度
    n = 0
    for arr in fields.values():
        if len(arr) > n:
            n = len(arr)
    
    if n == 0:
        return []
    
    # 生成时间
    t_sec = generate_t_sec_array(n, hz)
    
    states = []
    for i in range(n):
        state = EgoState(
            t_sec=float(t_sec[i]),
            x_m=float(fields.get("pos_x", [np.nan] * n)[i]) if "pos_x" in fields else 0.0,
            y_m=float(fields.get("pos_y", [np.nan] * n)[i]) if "pos_y" in fields else 0.0,
            z_m=float(fields.get("pos_z", [np.nan] * n)[i]) if "pos_z" in fields else 0.0,
            vx_mps=float(fields["vel_x"][i]) if "vel_x" in fields and i < len(fields["vel_x"]) else None,
            vy_mps=float(fields["vel_y"][i]) if "vel_y" in fields and i < len(fields["vel_y"]) else None,
            vz_mps=float(fields["vel_z"][i]) if "vel_z" in fields and i < len(fields["vel_z"]) else None,
            ax_mps2=float(fields["accel_x"][i]) if "accel_x" in fields and i < len(fields["accel_x"]) else None,
            ay_mps2=float(fields["accel_y"][i]) if "accel_y" in fields and i < len(fields["accel_y"]) else None,
            az_mps2=float(fields["accel_z"][i]) if "accel_z" in fields and i < len(fields["accel_z"]) else None,
            yaw_rad=float(fields["yaw"][i]) if "yaw" in fields and i < len(fields["yaw"]) else (
                float(fields["heading"][i]) if "heading" in fields and i < len(fields["heading"]) else None
            ),
        )
        states.append(state)
    
    return states


def extract_preference_trajectories(e2e: wod_e2ed_pb2.E2EDFrame) -> List[RaterTrajectory]:
    """
    提取 preference trajectories（rater 轨迹）
    参考 _export_cut_ins_package.py 的 extract_preference_trajectories
    """
    out: List[RaterTrajectory] = []
    
    if not hasattr(e2e, "preference_trajectories"):
        return out
    
    for i, tr in enumerate(e2e.preference_trajectories):
        rater = RaterTrajectory(rater_idx=i, score=-1.0)
        
        # 尝试获取分数
        if hasattr(tr, "preference_score"):
            try:
                rater.score = float(tr.preference_score)
            except Exception:
                pass
        
        # 提取轨迹点
        fields: Dict[str, np.ndarray] = {}
        
        # Case A: 标量字段
        for k in ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "accel_x", "accel_y"]:
            if hasattr(tr, k):
                v = list(getattr(tr, k))
                if len(v) > 0:
                    fields[k] = np.array(v, dtype=np.float32)
        
        # Case B: 嵌套消息
        if not fields:
            rep = []
            if hasattr(tr, "ListFields"):
                for fd, val in tr.ListFields():
                    if fd.label == fd.LABEL_REPEATED and fd.type == fd.TYPE_MESSAGE:
                        rep.append((fd.name.lower(), fd.name, val))
            
            best = None
            kws = ["state", "states", "trajectory", "point", "points", "sample", "samples"]
            for low, name, val in rep:
                score = sum(k in low for k in kws)
                cand = (score, name, val)
                if best is None or cand[0] > best[0]:
                    best = cand
            
            if best is not None and len(best[2]) > 0:
                pts = best[2]
                xs, ys, zs = [], [], []
                for p in pts:
                    if hasattr(p, "x") and hasattr(p, "y"):
                        xs.append(float(getattr(p, "x")))
                        ys.append(float(getattr(p, "y")))
                        zs.append(float(getattr(p, "z")) if hasattr(p, "z") else 0.0)
                    elif hasattr(p, "pos_x") and hasattr(p, "pos_y"):
                        xs.append(float(getattr(p, "pos_x")))
                        ys.append(float(getattr(p, "pos_y")))
                        zs.append(float(getattr(p, "pos_z")) if hasattr(p, "pos_z") else 0.0)
                    else:
                        xs = []
                        break
                
                if xs:
                    fields["pos_x"] = np.array(xs, dtype=np.float32)
                    fields["pos_y"] = np.array(ys, dtype=np.float32)
                    fields["pos_z"] = np.array(zs, dtype=np.float32)
        
        # 构建轨迹点
        if fields:
            n = max(len(arr) for arr in fields.values())
            t_arr = generate_t_sec_array(n, TRAJECTORY_HZ)
            
            for j in range(n):
                point = TrajectoryPoint(
                    t_sec=float(t_arr[j]),
                    x_m=float(fields["pos_x"][j]) if "pos_x" in fields else 0.0,
                    y_m=float(fields["pos_y"][j]) if "pos_y" in fields else 0.0,
                    z_m=float(fields["pos_z"][j]) if "pos_z" in fields else 0.0,
                    vx_mps=float(fields["vel_x"][j]) if "vel_x" in fields and j < len(fields["vel_x"]) else None,
                    vy_mps=float(fields["vel_y"][j]) if "vel_y" in fields and j < len(fields["vel_y"]) else None,
                    ax_mps2=float(fields["accel_x"][j]) if "accel_x" in fields and j < len(fields["accel_x"]) else None,
                    ay_mps2=float(fields["accel_y"][j]) if "accel_y" in fields and j < len(fields["accel_y"]) else None,
                )
                rater.points.append(point)
        
        out.append(rater)
    
    return out


def extract_pose_4x4(frame: Any) -> Optional[np.ndarray]:
    """提取车辆位姿 4x4 矩阵"""
    if hasattr(frame, "pose") and hasattr(frame.pose, "transform"):
        t = list(frame.pose.transform)
        if len(t) == 16:
            return np.array(t, dtype=np.float32).reshape(4, 4)
    return None


def extract_images_from_frame(frame: Any) -> Dict[int, CameraImage]:
    """从 Frame 提取所有相机图像"""
    images: Dict[int, CameraImage] = {}
    
    if not hasattr(frame, "images"):
        return images
    
    for img in frame.images:
        try:
            cam_id = int(img.name)
            cam_name = cam_name_from_id(cam_id)
            jpeg_bytes = bytes(img.image)
            
            # 获取时间戳（如果有）
            timestamp = None
            if hasattr(img, "camera_trigger_time"):
                timestamp = int(img.camera_trigger_time)
            
            # 获取 camera_image_metadata（用于投影）
            meta = None
            if hasattr(img, "camera_image_metadata"):
                meta = img.camera_image_metadata
            
            images[cam_id] = CameraImage(
                cam_id=cam_id,
                cam_name=cam_name,
                jpeg_bytes=jpeg_bytes,
                timestamp_usec=timestamp,
                camera_image_metadata=meta,
            )
        except Exception:
            continue
    
    return images


def extract_critical_moment_from_e2e(e2e: wod_e2ed_pb2.E2EDFrame) -> Tuple[Optional[float], Optional[int]]:
    """
    从 E2EDFrame 提取 critical moment
    
    Returns:
        (critical_timestamp_sec, critical_frame_index_10hz) 或 (None, None)
    """
    # 尝试从各种可能的字段获取 critical moment
    # 这取决于具体的 proto 定义
    
    # 方法1：直接字段
    if hasattr(e2e, "critical_moment_timestamp_sec"):
        ts = float(e2e.critical_moment_timestamp_sec)
        idx = int(ts * CAMERA_HZ)
        return ts, idx
    
    # 方法2：从 key_moment 获取
    if hasattr(e2e, "key_moment"):
        km = e2e.key_moment
        if hasattr(km, "timestamp_sec"):
            ts = float(km.timestamp_sec)
            idx = int(ts * CAMERA_HZ)
            return ts, idx
    
    # 方法3：使用帧索引推断（如果有 key_idx）
    # 这需要外部提供
    
    return None, None


# ============================================================================
# 主读取接口
# ============================================================================

def read_segment_from_tfrecords(
    tfrecord_paths: List[str],
    seg_id: str,
    split: str = "val",
    scenario_cluster: str = "",
    record_indices: Optional[List[int]] = None,
) -> SegmentRecord:
    """
    从 TFRecord 文件读取 segment 数据
    
    Args:
        tfrecord_paths: TFRecord 文件路径列表
        seg_id: segment ID
        split: 数据集分割 (val/train/test)
        scenario_cluster: 场景类别
        record_indices: 要读取的记录索引（可选，None 表示全部）
    
    Returns:
        SegmentRecord 对象
    """
    record = SegmentRecord(
        seg_id=seg_id,
        split=split,
        scenario_cluster=scenario_cluster,
        routing_command="UNKNOWN",
        source_files=tfrecord_paths,
    )
    
    frame_idx = 0
    
    for tfrecord_path in tfrecord_paths:
        tfrecord_path = to_wsl_path_if_needed(tfrecord_path)
        
        if not os.path.exists(tfrecord_path):
            record.warnings.append(f"TFRecord not found: {tfrecord_path}")
            continue
        
        ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
        
        for rec_idx, raw in enumerate(ds):
            # 如果指定了索引，只读取特定记录
            if record_indices is not None and rec_idx not in record_indices:
                continue
            
            try:
                e2e = parse_e2ed_frame(bytes(raw.numpy()))
                frame = e2e.frame
                
                # 提取图像
                images = extract_images_from_frame(frame)
                
                # 提取 ego 状态
                past_fields = extract_ego_states_scalar_fields(
                    e2e.past_states if hasattr(e2e, "past_states") else None
                )
                future_fields = extract_ego_states_scalar_fields(
                    e2e.future_states if hasattr(e2e, "future_states") else None
                )
                
                # 更新 segment 级别的轨迹
                if not record.ego_past_states:
                    record.ego_past_states = build_ego_states_from_fields(past_fields)
                if not record.ego_future_states:
                    record.ego_future_states = build_ego_states_from_fields(future_fields)
                
                # 提取 rater 轨迹（仅第一帧，因为所有帧应该相同）
                if not record.rater_trajectories and split == "val":
                    record.rater_trajectories = extract_preference_trajectories(e2e)
                
                # 提取相机标定（第一帧）
                if record.camera_calibration is None:
                    record.camera_calibration = extract_camera_calibration_from_frame(frame)
                
                # 提取 routing command
                if record.routing_command == "UNKNOWN":
                    if hasattr(e2e, "intent"):
                        record.routing_command = intent_to_routing_command(int(e2e.intent))
                
                # 提取 critical moment
                if record.critical_timestamp_sec is None:
                    ts, idx = extract_critical_moment_from_e2e(e2e)
                    if ts is not None:
                        record.critical_timestamp_sec = ts
                        record.critical_frame_index_10hz = idx
                
                # 构建当前帧的 ego state
                current_state = None
                if past_fields:
                    # 使用 past 的最后一个点作为当前状态
                    past_states = build_ego_states_from_fields(past_fields)
                    if past_states:
                        current_state = past_states[-1]
                        # 调整时间为当前帧时间
                        current_state.t_sec = frame_idx / CAMERA_HZ
                
                # 提取位姿
                pose = extract_pose_4x4(frame)
                
                # 构建 FrameData
                frame_data = FrameData(
                    frame_idx=frame_idx,
                    t_sec=frame_idx / CAMERA_HZ,
                    images=images,
                    ego_state=current_state,
                    camera_calibration=record.camera_calibration,
                    vehicle_pose_4x4=pose,
                    context_name=getattr(frame.context, "name", "") if hasattr(frame, "context") else "",
                )
                
                record.frames.append(frame_data)
                frame_idx += 1
                
            except Exception as e:
                record.warnings.append(f"Error parsing record {rec_idx}: {str(e)}")
                continue
    
    return record


def read_single_e2ed_frame(
    tfrecord_path: str,
    record_idx: int
) -> Optional[wod_e2ed_pb2.E2EDFrame]:
    """
    读取单个 E2EDFrame
    
    Args:
        tfrecord_path: TFRecord 文件路径
        record_idx: 记录索引
    
    Returns:
        E2EDFrame 或 None
    """
    tfrecord_path = to_wsl_path_if_needed(tfrecord_path)
    
    if not os.path.exists(tfrecord_path):
        return None
    
    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    
    for idx, raw in enumerate(ds):
        if idx == record_idx:
            return parse_e2ed_frame(bytes(raw.numpy()))
        if idx > record_idx:
            break
    
    return None


# ============================================================================
# 批量读取（优化性能）
# ============================================================================

def batch_read_records(
    file_to_indices: Dict[str, List[int]]
) -> Dict[Tuple[str, int], bytes]:
    """
    批量读取多个 TFRecord 文件中的指定记录
    
    Args:
        file_to_indices: {tfrecord_path: [record_indices]}
    
    Returns:
        {(file_path, record_idx): raw_bytes}
    """
    result: Dict[Tuple[str, int], bytes] = {}
    
    for file_path, indices in file_to_indices.items():
        file_path = to_wsl_path_if_needed(file_path)
        
        if not os.path.exists(file_path):
            continue
        
        idx_set = set(indices)
        max_idx = max(idx_set) if idx_set else -1
        
        ds = tf.data.TFRecordDataset(file_path, compression_type="")
        
        for i, raw in enumerate(ds):
            if i > max_idx:
                break
            if i in idx_set:
                result[(file_path, i)] = bytes(raw.numpy())
    
    return result
