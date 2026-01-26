"""
camera_meta.py - 解析并生成 camera_calib.json

职责：
- 从 WOD 数据中解析相机内外参
- 生成符合规范的 camera_calib.json
- 不得写 images/trajectory 文件

输出格式要求：
- 内参：K（3x3）或 fx, fy, cx, cy
- 外参：T_ego_from_cam（4x4）或 T_cam_from_ego（4x4）
- 必须写清约定：frame_convention, matrix_convention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ============================================================================
# 相机名称映射 (WOD CameraName enum)
# ============================================================================

# Waymo 相机 ID 到名称的映射
CAMERA_ID_TO_NAME: Dict[int, str] = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
    6: "REAR_LEFT",      # 有些版本是这样
    7: "REAR",           # cam_7 通常是后视
    8: "REAR_RIGHT",
}

# 标准 8 相机 ID 列表
STANDARD_CAMERA_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class IntrinsicParams:
    """相机内参"""
    fx: float
    fy: float
    cx: float
    cy: float
    # 可选：完整 K 矩阵
    K: Optional[List[List[float]]] = None
    # 可选：畸变参数
    distortion: Optional[List[float]] = None
    distortion_model: Optional[str] = None
    # 图像分辨率
    width: Optional[int] = None
    height: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }
        if self.K is not None:
            d["K"] = self.K
        if self.distortion is not None:
            d["distortion"] = self.distortion
        if self.distortion_model is not None:
            d["distortion_model"] = self.distortion_model
        if self.width is not None:
            d["width"] = self.width
        if self.height is not None:
            d["height"] = self.height
        return d
    
    @classmethod
    def from_K_matrix(cls, K: np.ndarray, width: int = None, height: int = None) -> "IntrinsicParams":
        """从 3x3 K 矩阵构建"""
        K = np.asarray(K)
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            K=K.tolist(),
            width=width,
            height=height,
        )


@dataclass
class ExtrinsicParams:
    """相机外参"""
    # 4x4 变换矩阵：从相机坐标到 ego 坐标
    T_ego_from_cam: List[List[float]]
    # 可选：逆变换
    T_cam_from_ego: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "T_ego_from_cam": self.T_ego_from_cam,
        }
        if self.T_cam_from_ego is not None:
            d["T_cam_from_ego"] = self.T_cam_from_ego
        return d
    
    @classmethod
    def from_4x4_matrix(cls, T: np.ndarray, is_cam_from_ego: bool = False) -> "ExtrinsicParams":
        """
        从 4x4 矩阵构建
        
        Args:
            T: 4x4 变换矩阵
            is_cam_from_ego: True 表示 T 是 T_cam_from_ego，需要求逆
        """
        T = np.asarray(T).reshape(4, 4)
        
        if is_cam_from_ego:
            T_cam_from_ego = T
            T_ego_from_cam = np.linalg.inv(T)
        else:
            T_ego_from_cam = T
            T_cam_from_ego = np.linalg.inv(T)
        
        return cls(
            T_ego_from_cam=T_ego_from_cam.tolist(),
            T_cam_from_ego=T_cam_from_ego.tolist(),
        )


@dataclass
class CameraCalibEntry:
    """单个相机的标定信息"""
    cam_id: int
    cam_name: str
    intrinsic: IntrinsicParams
    extrinsic: ExtrinsicParams
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cam_id": self.cam_id,
            "cam_name": self.cam_name,
            "intrinsic": self.intrinsic.to_dict(),
            "extrinsic": self.extrinsic.to_dict(),
        }


@dataclass
class CameraCalibration:
    """完整的相机标定信息（所有相机）"""
    cameras: Dict[int, CameraCalibEntry] = field(default_factory=dict)
    frame_convention: str = "ego: x-forward, y-left, z-up; cam: x-right, y-down, z-forward"
    matrix_convention: Dict[str, Any] = field(default_factory=lambda: {
        "vector_is_column": True,
        "multiply": "left",
        "description": "p_ego = T_ego_from_cam @ p_cam"
    })
    
    def add_camera(self, entry: CameraCalibEntry) -> None:
        """添加相机标定"""
        self.cameras[entry.cam_id] = entry
    
    def get_camera(self, cam_id: int) -> Optional[CameraCalibEntry]:
        """获取相机标定"""
        return self.cameras.get(cam_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_convention": self.frame_convention,
            "matrix_convention": self.matrix_convention,
            "cameras": {
                f"cam_{cid}": entry.to_dict()
                for cid, entry in sorted(self.cameras.items())
            }
        }
    
    def is_complete(self) -> bool:
        """检查是否包含所有 8 个相机"""
        return all(cid in self.cameras for cid in STANDARD_CAMERA_IDS)
    
    def missing_cameras(self) -> List[int]:
        """返回缺失的相机 ID 列表"""
        return [cid for cid in STANDARD_CAMERA_IDS if cid not in self.cameras]


# ============================================================================
# 从 Waymo 数据解析
# ============================================================================

def parse_camera_calibration_from_proto(calib_proto: Any) -> Optional[CameraCalibEntry]:
    """
    从 Waymo camera_calibration proto 解析标定信息
    
    Args:
        calib_proto: frame.context.camera_calibrations 中的单个元素
    
    Returns:
        CameraCalibEntry 或 None
    """
    try:
        cam_id = int(calib_proto.name)
        cam_name = CAMERA_ID_TO_NAME.get(cam_id, f"CAMERA_{cam_id}")
        
        # 解析内参
        # Waymo 格式：intrinsic 是一个 9 元素数组 [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        # 或者某些版本是不同格式
        intrinsic_data = list(calib_proto.intrinsic)
        
        if len(intrinsic_data) >= 4:
            fx, fy, cx, cy = intrinsic_data[:4]
            distortion = intrinsic_data[4:] if len(intrinsic_data) > 4 else None
        else:
            # 尝试其他格式
            return None
        
        # 构建 K 矩阵
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 获取图像尺寸（如果可用）
        width = getattr(calib_proto, 'width', None)
        height = getattr(calib_proto, 'height', None)
        
        intrinsic = IntrinsicParams(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            K=K.tolist(),
            distortion=distortion,
            distortion_model="radtan" if distortion else None,
            width=int(width) if width else None,
            height=int(height) if height else None,
        )
        
        # 解析外参
        # Waymo 格式：extrinsic.transform 是 16 元素数组（行优先 4x4）
        extrinsic_data = list(calib_proto.extrinsic.transform)
        
        if len(extrinsic_data) != 16:
            return None
        
        # Waymo 的 extrinsic 是 T_vehicle_from_camera (vehicle = ego)
        T_ego_from_cam = np.array(extrinsic_data, dtype=np.float64).reshape(4, 4)
        
        extrinsic = ExtrinsicParams.from_4x4_matrix(T_ego_from_cam, is_cam_from_ego=False)
        
        return CameraCalibEntry(
            cam_id=cam_id,
            cam_name=cam_name,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )
        
    except Exception as e:
        # 解析失败，返回 None
        return None


def extract_camera_calibration_from_frame(frame: Any) -> CameraCalibration:
    """
    从 Waymo Frame 提取所有相机标定
    
    Args:
        frame: Waymo Frame proto (from E2EDFrame.frame)
    
    Returns:
        CameraCalibration 对象
    """
    calib = CameraCalibration()
    
    if not hasattr(frame, 'context'):
        return calib
    
    if not hasattr(frame.context, 'camera_calibrations'):
        return calib
    
    for cam_calib in frame.context.camera_calibrations:
        entry = parse_camera_calibration_from_proto(cam_calib)
        if entry is not None:
            calib.add_camera(entry)
    
    return calib


def create_dummy_calibration() -> CameraCalibration:
    """
    创建虚拟标定（用于测试或数据不可用时）
    
    Returns:
        包含 8 个相机的虚拟标定
    """
    calib = CameraCalibration()
    
    # 典型的前视相机内参（1920x1280）
    default_fx = 2000.0
    default_fy = 2000.0
    default_cx = 960.0
    default_cy = 640.0
    
    # 单位矩阵作为默认外参
    identity_T = np.eye(4).tolist()
    
    for cam_id in STANDARD_CAMERA_IDS:
        entry = CameraCalibEntry(
            cam_id=cam_id,
            cam_name=CAMERA_ID_TO_NAME.get(cam_id, f"CAMERA_{cam_id}"),
            intrinsic=IntrinsicParams(
                fx=default_fx,
                fy=default_fy,
                cx=default_cx,
                cy=default_cy,
                K=[
                    [default_fx, 0, default_cx],
                    [0, default_fy, default_cy],
                    [0, 0, 1]
                ],
                width=1920,
                height=1280,
            ),
            extrinsic=ExtrinsicParams(
                T_ego_from_cam=identity_T,
                T_cam_from_ego=identity_T,
            ),
        )
        calib.add_camera(entry)
    
    return calib


# ============================================================================
# 坐标变换工具
# ============================================================================

def transform_points_cam_to_ego(
    points_cam: np.ndarray,
    T_ego_from_cam: np.ndarray
) -> np.ndarray:
    """
    将相机坐标系中的点变换到 ego 坐标系
    
    Args:
        points_cam: 相机坐标系中的点，shape=(N, 3)
        T_ego_from_cam: 4x4 变换矩阵
    
    Returns:
        ego 坐标系中的点，shape=(N, 3)
    """
    points_cam = np.asarray(points_cam)
    T = np.asarray(T_ego_from_cam)
    
    # 齐次坐标
    N = points_cam.shape[0]
    ones = np.ones((N, 1))
    points_h = np.hstack([points_cam, ones])  # (N, 4)
    
    # 变换
    points_ego_h = (T @ points_h.T).T  # (N, 4)
    
    return points_ego_h[:, :3]


def transform_points_ego_to_cam(
    points_ego: np.ndarray,
    T_cam_from_ego: np.ndarray
) -> np.ndarray:
    """
    将 ego 坐标系中的点变换到相机坐标系
    
    Args:
        points_ego: ego 坐标系中的点，shape=(N, 3)
        T_cam_from_ego: 4x4 变换矩阵
    
    Returns:
        相机坐标系中的点，shape=(N, 3)
    """
    points_ego = np.asarray(points_ego)
    T = np.asarray(T_cam_from_ego)
    
    N = points_ego.shape[0]
    ones = np.ones((N, 1))
    points_h = np.hstack([points_ego, ones])
    
    points_cam_h = (T @ points_h.T).T
    
    return points_cam_h[:, :3]


def project_points_to_image(
    points_cam: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    """
    将相机坐标系中的 3D 点投影到图像平面
    
    Args:
        points_cam: 相机坐标系中的点，shape=(N, 3)，z-forward
        K: 3x3 内参矩阵
    
    Returns:
        图像坐标 (u, v)，shape=(N, 2)
    """
    points_cam = np.asarray(points_cam)
    K = np.asarray(K)
    
    # 归一化坐标
    z = points_cam[:, 2:3]  # (N, 1)
    z = np.clip(z, 1e-6, None)  # 避免除零
    
    # 投影到图像平面
    points_norm = points_cam[:, :2] / z  # (N, 2)
    
    # 应用内参
    u = K[0, 0] * points_norm[:, 0] + K[0, 2]
    v = K[1, 1] * points_norm[:, 1] + K[1, 2]
    
    return np.stack([u, v], axis=1)
