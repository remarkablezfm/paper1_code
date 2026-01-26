"""
utils.py - 通用工具函数

职责：路径处理、JSON/CSV读写、安全写入、日志配置
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ============================================================================
# 路径工具
# ============================================================================

def to_wsl_path_if_needed(s: str) -> str:
    """
    路径处理函数 (Windows 版本)
    
    在纯 Windows 环境中，不进行 WSL 路径转换，
    只进行基本的路径规范化。
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    
    # Windows 版本：不转换为 WSL 路径，直接返回规范化的路径
    # 如果是 WSL 路径格式 /mnt/d/...，转换为 Windows 格式 D:\...
    if s.startswith("/mnt/") and len(s) > 5:
        drive = s[5].upper()
        rest = s[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    
    # 规范化路径分隔符
    return s.replace("/", "\\")


def ensure_dir(p: Union[Path, str]) -> Path:
    """确保目录存在，返回 Path 对象"""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_remove_dir(p: Union[Path, str]) -> None:
    """安全删除目录"""
    p = Path(p)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)


# ============================================================================
# JSON 工具
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def safe_json_dump(obj: Any, path: Union[Path, str], indent: int = 2) -> None:
    """安全写入 JSON 文件，支持 numpy 类型"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, cls=NumpyEncoder)


def safe_json_load(path: Union[Path, str]) -> Any:
    """安全读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# CSV 工具
# ============================================================================

def safe_csv_write(df: pd.DataFrame, path: Union[Path, str], index: bool = False) -> None:
    """安全写入 CSV 文件"""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def safe_csv_read(path: Union[Path, str]) -> pd.DataFrame:
    """安全读取 CSV 文件"""
    return pd.read_csv(path)


# ============================================================================
# 日志配置
# ============================================================================

def setup_logger(
    name: str,
    log_file: Optional[Union[Path, str]] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 清除已有 handlers
    logger.handlers = []
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件 handler（如果指定）
    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# 数据验证工具
# ============================================================================

def validate_required_fields(
    data: Dict[str, Any],
    required: List[str],
    context: str = ""
) -> List[str]:
    """
    验证字典中是否包含必需字段
    
    Returns:
        缺失字段列表
    """
    missing = []
    for field in required:
        if field not in data or data[field] is None:
            missing.append(field)
    return missing


def is_valid_number(val: Any) -> bool:
    """检查值是否为有效数值（非 NaN、非 None）"""
    if val is None:
        return False
    try:
        f = float(val)
        return np.isfinite(f)
    except (ValueError, TypeError):
        return False


# ============================================================================
# 图片文件名工具
# ============================================================================

def frame_index_to_filename(frame_idx: int, extension: str = ".jpg") -> str:
    """
    将帧索引转换为6位零填充文件名
    
    Args:
        frame_idx: 帧索引（0-based）
        extension: 文件扩展名
    
    Returns:
        例如 "000000.jpg"
    """
    return f"{frame_idx:06d}{extension}"


def filename_to_frame_index(filename: str) -> int:
    """
    从文件名解析帧索引
    
    Args:
        filename: 例如 "000000.jpg"
    
    Returns:
        帧索引
    """
    stem = Path(filename).stem
    return int(stem)


# ============================================================================
# 相机 ID 工具
# ============================================================================

def cam_id_to_folder_name(cam_id: int) -> str:
    """
    相机 ID 转文件夹名（1-based）
    
    Args:
        cam_id: 相机 ID (1-8)
    
    Returns:
        例如 "cam_1"
    """
    return f"cam_{cam_id}"


def folder_name_to_cam_id(folder_name: str) -> int:
    """
    文件夹名转相机 ID
    
    Args:
        folder_name: 例如 "cam_1"
    
    Returns:
        相机 ID
    """
    return int(folder_name.split("_")[1])


# ============================================================================
# 统计工具
# ============================================================================

def compute_stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
    """
    计算数组的统计量
    
    Returns:
        包含 mean, min, max, std 的字典
    """
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    
    if len(valid) == 0:
        return {"mean": None, "min": None, "max": None, "std": None}
    
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
    }
