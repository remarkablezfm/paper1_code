# WOD-E2E Data Exporter

一个用于从 Waymo Open Dataset End-to-End (WOD-E2E) TFRecord 文件中提取和导出驾驶数据的工具。

## 功能特性

- 📷 **图像导出**: 从 8 个相机视角导出 JPEG 图像
- 🚗 **轨迹数据**: 导出 ego vehicle 状态、历史轨迹和未来轨迹
- 👤 **Rater 数据**: 导出人工标注的轨迹数据（仅验证集）
- ⏱️ **时间切片**: 支持导出完整 20s 片段和关键 5s 片段
- 🔧 **相机标定**: 导出相机内外参数
- ⚡ **并行处理**: 支持多进程并行导出
- 📊 **进度条显示**: 实时显示导出进度（需安装 tqdm）

## 项目结构

```
data_preparation/
├── wod_e2e_exporter/
│   ├── __init__.py        # 包初始化
│   ├── __main__.py        # 模块入口
│   ├── main.py            # CLI 入口与调度
│   ├── io_reader.py       # TFRecord 数据读取
│   ├── exporters.py       # 数据导出功能
│   ├── slicer.py          # 时间切片处理
│   ├── time_align.py      # 时间对齐工具
│   ├── schema.py          # 数据结构定义
│   ├── camera_meta.py     # 相机参数处理
│   └── utils.py           # 通用工具函数
├── requirements.txt       # Python 依赖
└── README.md              # 项目说明
```

## 环境要求

- Python >= 3.8
- TensorFlow >= 2.10.0
- Waymo Open Dataset SDK（版本需与 TensorFlow 匹配）

## 安装

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**: Waymo Open Dataset SDK 版本需与 TensorFlow 版本匹配。请根据你的 TensorFlow 版本选择对应的 SDK：
> - TF 2.10: `waymo-open-dataset-tf-2-10-0`
> - TF 2.11: `waymo-open-dataset-tf-2-11-0`
> - TF 2.12: `waymo-open-dataset-tf-2-12-0`
> - TF 2.13: `waymo-open-dataset-tf-2-13-0`

### 3. 进度条支持（推荐）

`tqdm` 已包含在 `requirements.txt` 中。安装后运行时会自动显示进度条：

```
Exporting: 45%|████████████████░░░░░░░░░░░░░░░░| 450/1000 [12:30<15:20, 0.55it/s]
```

如果未安装 `tqdm`，程序仍可正常运行，但不会显示进度条。

## 使用方法

### 基本用法

```bash
python -m wod_e2e_exporter --dataset_root /path/to/WOD_E2E --out_root /path/to/output
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_root` | str | **必需** | WOD-E2E 数据集根目录 |
| `--out_root` | str | **必需** | 输出目录 |
| `--split` | str | `val` | 数据集分割 (`val`, `train`, `test`) |
| `--scenario_cluster` | str | `ALL` | 场景类别过滤 |
| `--max_segments` | int | None | 最大处理 segment 数量 |
| `--num_workers` | int | `1` | 并行处理的工作进程数 |
| `--overwrite` | str | `false` | 是否覆盖已存在的输出 |
| `--fail_fast` | str | `false` | 遇到错误时是否立即停止 |
| `--log_level` | str | `INFO` | 日志级别 (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--index_csv` | str | None | segment 索引 CSV 文件路径（可选） |
| `--ego_state_hz` | float | `10.0` | ego state 采样率 (Hz) |

### 使用示例

```bash
# 导出验证集所有 segments
python -m wod_e2e_exporter --dataset_root /data/WOD_E2E --out_root /output

# 导出特定场景类别
python -m wod_e2e_exporter --dataset_root /data/WOD_E2E --out_root /output --scenario_cluster Cut_ins

# 使用 4 个工作进程并行导出
python -m wod_e2e_exporter --dataset_root /data/WOD_E2E --out_root /output --num_workers 4

# 限制处理数量进行测试
python -m wod_e2e_exporter --dataset_root /data/WOD_E2E --out_root /output --max_segments 10

# 使用索引 CSV 文件
python -m wod_e2e_exporter --dataset_root /data/WOD_E2E --out_root /output --index_csv /path/to/index.csv

# 完整示例：使用索引文件 + 并行处理 + 特定场景
python -m wod_e2e_exporter \
    --dataset_root /mnt/d/Datasets/WOD_E2E_Camera_v1 \
    --out_root /mnt/d/Datasets/WOD_E2E_Camera_v1/output \
    --index_csv /mnt/d/Datasets/WOD_E2E_Camera_v1/val_index_filled.csv \
    --scenario_cluster Cut_ins \
    --num_workers 4
```

### 场景类别 (scenario_cluster)

| 类别名称 | 说明 |
|----------|------|
| `Interections` | 交叉路口 |
| `Cut_ins` | 切入场景 |
| `Pedestrian` | 行人场景 |
| `Cyclist` | 骑行者场景 |
| `Construction` | 施工场景 |
| `Foreign Object Debris` | 异物碎片 |
| `Single-Lane Maneuvers` | 单车道操作 |
| `ALL` | 所有场景（默认） |

## 输出结构

每个 segment 的输出目录结构如下：

```
{scenario_cluster}_segment-{seg_id}/
├── images/
│   ├── cam_front/           # 前置相机图像
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── cam_front_left/      # 前左相机图像
│   ├── cam_front_right/     # 前右相机图像
│   ├── cam_side_left/       # 侧左相机图像
│   ├── cam_side_right/      # 侧右相机图像
│   └── ...                  # 其他相机
├── trajectory/
│   ├── ego_state_20s_10hz.csv      # 完整 20s ego 状态
│   ├── ego_past_4s.csv             # 过去 4s 轨迹
│   └── ego_future_5s.csv           # 未来 5s 轨迹
├── meta/
│   └── camera_calib.json           # 相机标定参数
├── critical_5s/                    # 关键 5s 片段
│   ├── images/
│   ├── trajectory/
│   │   ├── ego_state_0to5s_10hz.csv
│   │   ├── ego_future_0to5s.csv
│   │   └── rater_*.csv             # Rater 轨迹（仅 val）
│   ├── meta/
│   └── segment_manifest.json
├── segment_manifest.json           # 完整 segment 元数据
├── segment_summary.json            # 导出摘要与警告
└── _logs/                          # 日志目录
```

## 输出文件说明

### segment_manifest.json

包含 segment 的完整元数据：
- 基本信息：`seg_id`, `split`, `scenario_cluster`
- 时间信息：`critical_timestamp_sec`, `duration_sec`
- 轨迹文件路径和字段信息
- 相机配置

### ego_state CSV 文件

包含以下字段（部分可能为空）：
- `t_sec`: 时间戳（秒）
- `x_m`, `y_m`, `z_m`: 位置（米）
- `vx_mps`, `vy_mps`: 速度（米/秒）
- `ax_mps2`, `ay_mps2`: 加速度（米/秒²）
- `yaw_rad`: 航向角（弧度）
- `steering_angle_rad` / `steering_wheel_angle_rad`: 转向角度

## 日志与调试

- 导出日志保存在 `{out_root}/_logs/export.log`
- 每次导出完成后会生成 `export_summary.json` 统计信息
- 每个 segment 的 `segment_summary.json` 包含警告和错误信息

## 常见问题

### 1. TensorFlow 与 Waymo SDK 版本不匹配

确保 `waymo-open-dataset-tf-X-X-X` 的版本与安装的 TensorFlow 版本一致。

### 2. 内存不足

- 减少 `--num_workers` 的数量
- 使用 `--max_segments` 分批处理

### 3. 找不到 TFRecord 文件

检查数据集目录结构是否符合预期：
```
dataset_root/
  {split}/
    {scenario_cluster}/
      *.tfrecord
```

## License

本项目用于学术研究目的。使用 Waymo Open Dataset 请遵守其 [使用条款](https://waymo.com/open/terms/)。
