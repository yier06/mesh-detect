# mesh_detect

工位1 **环光 Mesh 检测与定位** 视觉项目。  
对 `1ms-环光.bmp` 等图像进行 mesh 数量统计、中心定位、角度计算，输出机械臂抓取坐标。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | **3.10+**（推荐 **3.11**） |
| 操作系统 | Windows / Linux |
| 依赖 | OpenCV、NumPy |

> Python 3.9 及以下不支持 `tuple[...] \| None` 类型注解语法，需升级或使用 `Optional[Tuple[...]]` 改写。

---

## 新建环境

### 方式一：Conda（推荐）

```bash
# 进入项目目录
cd mesh_detect

# 用 environment.yml 一键创建
conda env create -f environment.yml

# 激活环境
conda activate mesh_detect

# 验证
python --version
python -c "import cv2, numpy; print('opencv', cv2.__version__, 'numpy', numpy.__version__)"
```

手动创建：

```bash
conda create -n mesh_detect python=3.11 -y
conda activate mesh_detect
pip install -r requirements.txt
```

### 方式二：venv + pip

```bash
cd mesh_detect

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

---

## 项目结构

```
mesh_detect/
├── config/                  # 配置与数据模型
│   ├── detector_config.py   # DetectorConfig / MeshTarget / DetectionResult
│   └── mesh_detector_config.json
├── core/                    # 流水线核心
│   ├── context.py           # PipelineContext 上下文
│   └── pipeline.py          # ImagePipeline 流水线
├── processors/              # 图像处理步骤
│   ├── preprocess.py        # 灰度、模糊、阈值、形态学、ROI
│   ├── feature_extraction.py# 轮廓检测、mesh 特征
│   └── peak_finder.py       # 投影峰检测
├── parsers/                 # 图像/配置解析（扩展）
├── models/                  # 结果模型（扩展）
├── utils/                   # 可视化工具
├── scripts/                 # 辅助脚本
├── main.py                  # 完整 CLI 入口（单文件版算法）
├── demo.py                  # 流水线模块化 Demo
├── requirements.txt         # pip 依赖
├── environment.yml          # conda 环境
└── README.md
```

---

## 运行方式

**所有命令均需在 `mesh_detect` 目录下执行**，以保证模块导入路径正确。

```bash
cd mesh_detect
conda activate mesh_detect
```

### 1. 完整 CLI（main.py）

支持单张/文件夹、快速模式、过程图分目录保存：

```bash
# 默认处理上级目录中的 1ms*.bmp
python main.py

# 指定图像
python main.py --image ../1ms-环光.bmp --output-dir ../output/mesh_detect

# 批量处理文件夹
python main.py --image .. --config config/mesh_detector_config.json --output-dir ../output/batch

# 快速模式（不保存过程图，~75ms/张）
python main.py --image .. --fast --output-dir ../output/fast
```

### 2. 流水线 Demo（demo.py）

模块化流水线调试，逐步 `imshow` 中间结果：

```bash
python demo.py
```

> 注意：`demo.py` 中图像路径默认为 `images/1ms-环光.bmp`，运行前请确认路径存在，或改为实际路径。  
> Windows 中文路径建议使用 `main.py` 中的 `imdecode` 读图方式。

---

## 配置说明

配置文件：`config/mesh_detector_config.json`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `blur_kernel` | 5 | 高斯模糊核 |
| `threshold_value` | 0 | 0=Otsu 自动阈值 |
| `edge_margin_px` | 280 | 左右晕影屏蔽宽度 |
| `roi_y_ratio` | 0.50 | ROI 起始高度比例 |
| `min_area` | 8000 | 最小轮廓面积 |
| `nominal_mesh_width_px` | 85 | mesh 标准宽度（粘连分裂） |
| `debug` | true | 是否输出过程图 |
| `pixel_size_mm` | 0 | 像素→毫米标定（0=仅输出 px） |

---

## 输出说明

### 检测结果 JSON

```json
{
  "found": true,
  "mesh_count": 18,
  "origin_x_px": 419.08,
  "origin_y_px": 1315.09,
  "meshes": [
    {
      "index": 0,
      "center_x_px": 419.08,
      "center_y_px": 1315.09,
      "angle_deg": 6.25,
      "offset_x_px": 0.0,
      "offset_y_px": 0.0
    }
  ],
  "next_action": "pick_mesh"
}
```

- `found=false` 时 `next_action=move_to_next_station`，表示无 mesh，移下一工位
- 原点：最左侧 mesh 中心 `(0, 0)` 相对坐标

### 过程图（调试模式）

每张图保存在独立子目录：

```
output/mesh_detect/
└── 1ms-环光/
    ├── 01_original.png
    ├── ...
    ├── 11_final_result.png
    └── result.json
```

---

## 性能参考（2448×2048）

| 模式 | 耗时 |
|------|------|
| `--fast` | ~75 ms/张 |
| 调试模式（含 11 张 PNG） | ~600 ms/张 |

---

## 常见问题

### `conda` 命令找不到

使用完整路径：

```powershell
D:\miniconda3\Scripts\conda.exe create -n mesh_detect python=3.11 -y
D:\miniconda3\envs\mesh_detect\python.exe main.py
```

### `TypeError: unsupported operand type(s) for |`

Python 版本低于 3.10，请升级到 3.11 或修改类型注解为 `Optional[Tuple[...]]`。

### 中文路径读图失败

`main.py` 已使用 `imdecode` 兼容中文路径；`demo.py` 中 `cv2.imread` 在 Windows 中文路径下可能失败，建议改用 bytes 读图或切换到 `main.py`。

### Cursor / VS Code 选择解释器

`Ctrl+Shift+P` → **Python: Select Interpreter** → 选择 `mesh_detect` 环境的 `python.exe`。

---

## 相关文档

上级目录 `工位1/` 中还有：

- `算法处理方案书.md` — 完整算法方案
- `项目难点与解决方案.md` — 调试问题记录

---

## 依赖清单

见 [requirements.txt](./requirements.txt)：

```
opencv-python>=4.8.0,<5.0.0
numpy>=1.24.0,<3.0.0
```
