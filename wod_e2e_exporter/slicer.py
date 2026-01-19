"""Critical 5s slicer: derive sub-window from 20s segment without writing files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SliceMapping:
    parent_seg_id: str
    critical_timestamp_in_parent_sec: float
    parent_frame_index_start_10hz: int
    mapping_note: str


def slice_segment(segment_record: Any, critical_frame_index_10hz: int) -> Dict[str, Any]:
    """
    Create in-memory 5s view (images/trajectory/meta refs) from 20s segment.
    Returns dict with:
      - images_indices: range of frame indices in 20s (10Hz) belonging to 5s window
      - mapping: SliceMapping
      - critical_idx_10hz: critical frame index inside 5s (usually 0)
    """
    # Align 5s window so that critical frame is the first frame of the 5s slice (t=0 inside 5s).
    crit_idx_parent = max(0, int(critical_frame_index_10hz))
    window_len_frames = int(5.0 * 10)  # 5s at 10Hz -> 50 frames

    frame_ids = getattr(segment_record, "frame_ids", None)
    if frame_ids:
        # Select by frame_id range if available
        start_frame_id = crit_idx_parent
        end_frame_id = start_frame_id + window_len_frames - 1
        images_indices = [i for i, fid in enumerate(frame_ids) if fid is not None and start_frame_id <= fid <= end_frame_id]
        start_idx = start_frame_id
        mapping_note = "by_frame_id"
    else:
        start_idx = crit_idx_parent
        images_indices = list(range(start_idx, start_idx + window_len_frames))
        mapping_note = "nearest"

    mapping = SliceMapping(
        parent_seg_id=segment_record.seg_id,
        critical_timestamp_in_parent_sec=crit_idx_parent / 10.0,
        parent_frame_index_start_10hz=start_idx,
        mapping_note=mapping_note,  # critical maps to first frame in 5s
    )

    return {
        "images_indices": images_indices,
        "mapping": mapping,
        "critical_idx_10hz": 0,
    }
