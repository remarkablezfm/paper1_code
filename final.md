````md
# WOD-E2E 数据导出器需求规范（最终版，给 Codex / Claude Code / Cursor）

> **实现前置要求（必须遵守）**
1) 你必须把我提供的参考代码文件 **`_export_cut_ins_package.py`** 当作“解析细节的权威参考”，尤其用于：数据集命名、字段键名、segment/cam id 映射、以及任何从 TFRecord/原始数据中取值的细节。  
2) 本文用于“框死输出结构与接口”。若本文与 `_export_cut_ins_package.py` 在“解析实现细节”上冲突：  
   - **输出结构/文件名/字段名**以本文为准；  
   - **如何从数据中正确读取/解析到这些内容**以 `_export_cut_ins_package.py` 为准。

---

## 0. 目标与范围

我要实现一个 WOD-E2E 数据处理与导出工具：

- 当前以 **`val` split** 为基础实现（优先跑通）。
- 必须预留 `train/test` 接口（字段会略有差异），要求至少不崩溃并在 manifest/summary 中标注不可用字段。
- 输出用于后续研究：我会在 **critical 5s** 窗口内重规划轨迹（planning/control），并计算新轨迹分数（如 RFS 或我自定义评分）。  
  因此导出必须保留 **足够的运动学信息** 与 **相机->ego 坐标变换参数**（投影基础）。
- **不做 overlay / 可视化**（后续再加）。

---

## 1. 核心设计：20s 全量 + critical 5s 套娃（5s 必须独立可用）

### 1.1 时间规则（硬约束）
- 每个 segment 的第一帧时间 **必须定义为 `0.0s`**。
- 所有 CSV 第一列必须为 `t_sec`（单位秒），且第一行必须为 `0.0`。
- 时间生成规则（统一、可复现）：
  - 相机（10Hz）：`t_sec = frame_idx / 10.0`
  - 轨迹（4Hz）：`t_sec = i / 4.0`
  - ego_state（hz_state）：`t_sec = i / hz_state`

### 1.2 两套视图（必须同时导出）
1) **20s 全量视图**：上下文容器（cam + 运动学 + 可用轨迹）  
2) **critical 5s 视图**：以 `critical moment` 为锚点的 5 秒窗口（信息更丰富）

**关键要求：`critical_5s/` 必须独立可用**  
后续只读 `critical_5s/` 目录，就能完成训练/评测/规划控制，不依赖再读 20s 目录。

### 1.3 critical moment 规则（硬约束）
- `critical moment` 必须从标注解析得到（禁止启发式，如“最后 5 秒”）。
- 必须输出：
  - `critical_timestamp_sec`（相对 20s 起点）
  - `critical_frame_index_10hz`（能映射到 20s images 的帧号）
  - `critical_window_sec = 5.0`
- 5s 与 20s 的映射必须写入 **5s 自己的 manifest**（parent/mapping 字段）。

---

## 2. 输出目录结构（必须一字不差）

```text
{out_root}/{ScenarioCluster}/{ScenarioCluster}_segment-{seg_id}/
├── segment_manifest.json
├── segment_summary.json
├── images/                               # 20s 全量相机序列
│   └── cam_{1..8}/
│       ├── 000000.jpg
│       ├── 000001.jpg
│       └── ...
├── trajectory/                           # 20s 全量（运动学必须足够）
│   ├── ego_state_20s_{hz}hz.csv          # 必须存在：全量运动学状态序列
│   ├── ego_past_4s_4hz.csv               # 若数据提供则写
│   └── ego_future_5s_4hz.csv             # train/val；test 可能不可用（manifest 标注）
├── meta/                                 # 20s 的相机->ego 投影基础（集中保存）
│   └── camera_calib.json                 # 必须存在：K + 外参 + 约定
├── critical_5s/                          # 5s 必须完整、独立可用
│   ├── segment_manifest.json
│   ├── segment_summary.json
│   ├── images/
│   │   └── cam_{1..8}/
│   │       ├── 000000.jpg
│   │       └── ...
│   ├── trajectory/
│   │   ├── ego_state_0to5s_{hz}hz.csv
│   │   ├── ego_future_0to5s_4hz.csv      # train/val；test 可能不可用（manifest 标注）
│   │   ├── rater_traj_0_0to5s_4hz.csv    # 仅 val
│   │   ├── rater_traj_1_0to5s_4hz.csv    # 仅 val
│   │   └── rater_traj_2_0to5s_4hz.csv    # 仅 val
│   └── meta/
│       └── camera_calib.json             # 必须存在：完整投影参数（不依赖上层）
└── _logs/
    └── export.log
````

---

## 3. 相机图片导出规则（硬约束）

* 固定 8 个相机文件夹：`cam_1 ... cam_8`（**必须 1-based**）
* 图片名严格 6 位零填充：`000000.jpg`，`000001.jpg`...
* 20s：帧号从 0 连续编号
* 若源数据缺帧：

  * 导出文件名仍保持连续编号（从 0 连续）
  * 缺失情况必须写入 `segment_summary.json` 的 `warnings`
* 5s：同样使用 `cam_{1..8}/000000.jpg` 规则

  * `000000` 表示关键窗口内部索引（`t_crit=0.0` 对应第一帧）

---

## 4. CSV 规范（必须“信息足够”，为规划控制服务）

### 4.1 全局 CSV 规则（所有 CSV）

* 第一列：`t_sec`，且第一行 `t_sec=0.0`
* 单位统一：m / s / rad / mps / mps2
* 禁止伪造缺失：不能用 0 填补缺失字段

  * 允许 NaN 或缺列
  * 但必须在 summary 记录原因（errors 或 warnings）
* 字段命名必须全项目一致（同一语义只能一种名字）

### 4.2 ego_state CSV（硬要求：规划控制底座）

文件（必须存在）：

* `trajectory/ego_state_20s_{hz}hz.csv`
* `critical_5s/trajectory/ego_state_0to5s_{hz}hz.csv`

必须列（缺任意 => `partial/failed`，并写 error）：

* `t_sec`
* `x_m, y_m, z_m`
* `vx_mps, vy_mps`（至少平面分量）
* `ax_mps2, ay_mps2`
* `yaw_rad`（或 `heading_rad`，只能选一种，全项目一致）

转向字段（必须满足其一，且全项目固定一种命名；优先级）：

1. `steering_angle_rad`
2. `steering_wheel_angle_rad`
3. `curvature_1pm`

可选列（源数据有则保留）：

* `vz_mps, az_mps2, yaw_rate_rps, roll_rad, pitch_rad`

若源数据缺 `vx/vy` 仅有标量速度：

* 允许使用 `speed_mps` 替代，但必须：

  * 在 manifest 里声明使用了 `speed_mps`
  * 并在 summary 写 warning

### 4.3 轨迹 CSV（ego_past / ego_future / rater_traj）

最低必须列：

* `t_sec, x_m, y_m, z_m`

可选列（有则保留）：

* `vx_mps, vy_mps, ax_mps2, ay_mps2, yaw_rad, yaw_rate_rps, ...`

---

## 5. rater labels（val）与分数 score（必须）

* 仅 `val`：导出 3 条 rater 轨迹文件：

  * `rater_traj_0_0to5s_4hz.csv`
  * `rater_traj_1_0to5s_4hz.csv`
  * `rater_traj_2_0to5s_4hz.csv`
* **score 不写进 CSV**，写进 `manifest`，与每条 rater 文件绑定：

  * `rater_labels.trajectories = [{file, score}, ...]`
* **缺失分数的默认值**：`-1.0`

  * 若 rater score 缺失或不可用：写 `-1.0` 并在 summary 写 warning（不要因为 -1.0 阻断导出）。
* 对 `train/test`：

  * `rater_labels.available=false` 且 `reason` 必填

---

## 6. meta/camera_calib.json（必须集中保存 cam->ego 投影基础）

`meta/camera_calib.json` 必须包含每个 cam（cam_id=1..8）的条目，并满足：

* 内参（必须其一）：

  * `K`（3x3）或 `fx, fy, cx, cy`
* 外参（必须其一，且全项目一致）：

  * `T_ego_from_cam`（4x4）或 `T_cam_from_ego`（4x4）
* 必须写清约定（必须）：

  * `frame_convention`：描述 ego/cam 轴定义
  * `matrix_convention`：例如

    * `vector_is_column: true`
    * `multiply: "left"`（`p_out = T * p_in`）
* 可选：畸变参数、分辨率（若源数据提供）

**重要**：

* `critical_5s/meta/camera_calib.json` 必须是完整文件（拷贝或重写均可），不能依赖上层 meta。

---

## 7. manifest / summary（必须）

### 7.1 segment_manifest.json（机器读，字段固定）

20s 与 5s 各自目录都必须有自己的 manifest（结构一致，值不同）。

最少必须字段：

* `schema_version`：固定为 `v3.3-codex`
* `seg_id`, `split`（val/train/test）
* `scenario_cluster`
* `routing_command`：枚举 `GO_STRAIGHT|GO_LEFT|GO_RIGHT`
* `time`：

  * `segment_t0_sec=0.0`
  * `duration_sec`（20s=20.0；5s=5.0）
  * 20s 必须额外包含：

    * `critical_timestamp_sec`
    * `critical_frame_index_10hz`
    * `critical_window_sec=5.0`
  * 5s 必须额外包含 parent/mapping：

    * `parent_seg_id`
    * `critical_timestamp_in_parent_sec`
    * `parent_frame_index_start_10hz`
    * `mapping_note`（例如 rounding 规则：nearest/floor/ceil）
* `camera`：

  * `fps_hz=10`, `num_cameras=8`, `camera_ids=[1..8]`
  * `image_dir`, `calibration_file`
* `ego_state`：

  * `file`, `hz`, `fields`（实际导出的列名列表）
  * `steering_field`（最终选择的转向字段名）
* `ego_future`：

  * `available`（bool）
  * `file`（available=true 时必填）
  * `reason`（available=false 时必填）
* `rater_labels`：

  * `available`（bool）
  * `reason`（available=false 时必填）
  * `missing_score_value`：固定为 `-1.0`
  * `trajectories`（仅 val）：`[{file, score}, ...]`

### 7.2 segment_summary.json（调试/统计/错误）

必须包含：

* `schema_version`, `seg_id`, `status`（`success|partial|failed`）
* `errors: []`, `warnings: []`（每条至少 `type,message,where`）
* `counts`：至少包含帧数、相机数量、写出的文件数量
* `timing`：至少包含导出耗时（秒）

规则：

* 单个 segment 失败不应导致全局崩溃（除非 `--fail_fast true`）。
* 任何缺字段/掉帧/不可用数据都必须记录在 summary（errors 或 warnings）。

---

## 8. CLI（必须提供）

入口：`python -m wod_e2e_exporter.main ...`（推荐），或 `python main.py ...`（二选一，但必须稳定）

必须参数：

* `--dataset_root <path>`
* `--out_root <path>`

可选参数：

* `--split val|train|test`（默认 val）
* `--scenario_cluster <string|ALL>`（默认 ALL）
* `--max_segments <int>`（默认不限制）
* `--num_workers <int>`（默认 1）
* `--overwrite true|false`（默认 false）
* `--fail_fast true|false`（默认 false）
* `--log_level DEBUG|INFO|WARNING|ERROR`（默认 INFO）

行为：

* 若目标 segment 已存在且 `overwrite=false`：跳过该 segment，并在全局 log 记录。
* 退出码：0 成功；1 致命错误（参数/路径/初始化失败）。

---

## 9. 项目 Python 文件结构与职责（必须按职责拆分）

不允许把所有逻辑塞进一个文件。建议目录如下（可改文件名，但职责必须一一对应）：

```text
wod_e2e_exporter/
├── main.py                 # CLI 参数解析、调度、多进程/线程控制、全局日志
├── io_reader.py            # 读取 TFRecord/原始数据 -> SegmentRecord（只读不写文件）
├── schema.py               # manifest/summary 数据结构 + 生成 + 校验
├── time_align.py           # 唯一时间轴规则（统一输出 t_sec），禁止各模块自行算时间
├── camera_meta.py          # 解析并生成 camera_calib.json（含约定字段）
├── slicer.py               # 基于 critical moment 生成 5s 子切片（内存对象，不写文件）
├── exporters.py            # 写文件：images/trajectory/meta/manifest/summary（20s 与 5s）
├── utils.py                # 通用工具：路径、json/csv、安全写入、日志
└── __init__.py
```

强制依赖方向（禁止反向 import）：

* `main.py` 可调度所有模块
* `exporters.py` 可调用 `schema/time_align/utils/camera_meta`
* `slicer.py` 不能写文件（由 exporters 负责）
* `camera_meta.py` 不得写 images/trajectory
* `time_align.py` 是时间规则唯一真源

---

## 10. 验收标准（最小闭环）

1. **val 任意一个 segment**

* 20s 与 5s 两套目录都生成
* 5s 目录包含：8 cam images + ego_state + ego_future + 3 条 rater_traj
* 5s manifest 中 `rater_labels.trajectories` 每条都有 `score`（缺失则为 `-1.0`）
* 20s 与 5s 的 `meta/camera_calib.json` 都存在且完整（含内参/外参/约定）

2. **test 任意一个 segment**

* 仍生成 20s 与 5s（5s 独立可用）
* 对不可用字段（如 ego_future、rater_labels）：

  * `available=false` 且 `reason` 明确
  * 程序不崩溃

3. **一致性**

* 所有 CSV 第一行 `t_sec=0.0`
* images 文件名从 `000000.jpg` 开始，6 位零填充
* cam 文件夹必须是 `cam_1..cam_8`
* `schema_version` 恒为 `v3.3-codex`

```
```
