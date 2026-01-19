"""Time alignment utilities: single source of truth for t_sec generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TimeGrid:
    hz: float
    t_sec: List[float]


def build_camera_timegrid(num_frames: int, fps_hz: float = 10.0) -> TimeGrid:
    """Generate camera time axis starting at 0.0 with fixed fps (10Hz default)."""
    hz = float(fps_hz)
    t = [i / hz for i in range(max(0, int(num_frames)))]
    return TimeGrid(hz=hz, t_sec=t)


def build_traj_timegrid(num_points: int, hz: float = 4.0) -> TimeGrid:
    """Generate trajectory/ego_state time axis starting at 0.0."""
    hz_f = float(hz)
    t = [i / hz_f for i in range(max(0, int(num_points)))]
    return TimeGrid(hz=hz_f, t_sec=t)


def map_critical_to_20s(critical_frame_idx_10hz: int) -> dict:
    """
    Map critical 5s window into 20s indices/timestamps.
    TODO: define mapping_note (nearest/floor/ceil) once implementation starts.
    """
    idx = int(critical_frame_idx_10hz)
    t_sec = idx / 10.0
    # 5s window anchored at critical moment, start index floored to stay aligned with 10Hz frames
    start_idx = max(0, idx - 0)  # placeholder if later we allow window offset
    return {
        "critical_timestamp_sec": t_sec,
        "critical_frame_index_10hz": idx,
        "critical_window_sec": 5.0,
        "parent_frame_index_start_10hz": start_idx,
        "mapping_note": "nearest",
    }
