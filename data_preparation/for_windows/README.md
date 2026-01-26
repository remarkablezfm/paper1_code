# WOD-E2E Exporter (Windows 版本)

基于 WSL `waymo_clean` 环境配置的 Windows 版本。

## 重要警告

**Waymo Open Dataset SDK 官方仅支持 Linux**。在 Windows 上可能无法正常工作。

如果 Windows 安装失败，请使用 WSL。

## 环境配置 (与 WSL waymo_clean 一致)

| 包 | 版本 |
|---|---|
| Python | 3.9 |
| TensorFlow | 2.13.0 |
| Waymo SDK | waymo-open-dataset-tf-2-12-0==1.6.7 |
| NumPy | 1.23.5 |
| Pandas | 1.5.3 |
| OpenCV | 4.5.5.64 |
| Protobuf | 3.20.3 |

## 安装步骤

### 方法 1: 使用安装脚本 (推荐)

1. 打开 **Anaconda Prompt** 或 **PowerShell**
2. 进入此目录
3. 运行:
   ```cmd
   install.bat
   ```

### 方法 2: 手动安装

```cmd
conda create -n wod_e2e python=3.9 -y
conda activate wod_e2e

pip install numpy==1.23.5 pandas==1.5.3 opencv-python==4.5.5.64 tqdm==4.67.1
pip install tensorflow==2.13.0
pip install protobuf==3.20.3
pip install waymo-open-dataset-tf-2-12-0==1.6.7
```

## 使用方法

### 1. 修改配置

编辑 `run_test.bat` 或 `run_full.bat` 中的路径:

```bat
set DATASET_ROOT=D:\你的数据集路径
set OUT_ROOT=D:\输出路径
```

### 2. 运行测试

```cmd
run_test.bat
```

### 3. 完整导出

```cmd
run_full.bat
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `install.bat` | 安装 conda 环境 |
| `run_test.bat` | 测试脚本 (1 个 segment) |
| `run_full.bat` | 完整导出脚本 |
| `environment.yml` | Conda 环境配置 |
| `requirements.txt` | Pip 依赖列表 |
| `wod_e2e_exporter/` | Python 源代码 |

## 故障排除

### Waymo SDK 安装失败

这是预期的，因为 Waymo SDK 官方不支持 Windows。

**解决方案**: 使用 WSL

### TensorFlow 安装失败

尝试:
```cmd
pip install tensorflow==2.13.0 --no-cache-dir
```

### protobuf 版本冲突

确保安装 protobuf 3.20.3:
```cmd
pip uninstall protobuf -y
pip install protobuf==3.20.3
```
