import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any
import time
import numpy as np
import cv2

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}

@dataclass(frozen=True)
class DetectorConfig:
    blur_kernel: int = 5
    morph_kernel: int = 3
    morph_close_iter: int = 1
    threshold_value: int = 0
    roi_y_ratio: float = 0.50
    edge_margin_px: int = 280
    min_area: int = 8000
    max_area_ratio: float = 1.0 / 25.0
    min_aspect: float = 1.5
    split_width_ratio: float = 1.45
    nominal_mesh_width_px: float = 85.0
    min_peak_projection: float = 0.30
    projection_smooth_kernel: int = 21
    min_peak_distance_px: int = 40
    pixel_size_mm: float = 0.0
    output_dir: str = "output/mesh_detect"
    debug: bool = True

@dataclass
class MeshTarget:
    index: int
    center_x_px: float
    center_y_px: float
    width_px: float
    height_px: float
    angle_deg: float
    offset_x_px: float
    offset_y_px: float
    offset_x_mm: float | None
    offset_y_mm: float| None

@dataclass
class DetectResult:
    found: bool
    mesh_count: int
    origin_x_px: float
    origin_y_px: float
    origin_description: str
    meshes: list[MeshTarget]
    next_action: str
    message: str

def _morph_mask(
    mask: np.ndarray,
    kermnel_size: int,
    close_iter: int
)-> np.ndarray:
    kernel_size  = max(1, int(kernel_size))
    if kernel_size <= 1:
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if close_iter > 0:
        opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return opened

def _threshold_dark(
    gray: np.ndarray,
    threshold_value: int
)->tuple[np.ndarray, int]:
    if threshold_value > 0:
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        return mask, threshold_value
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask, int(threshold)

def detect_meshes_ringlight(
    image_bgr:np.ndarray,
    config: DetectorConfig = DetectorConfig(),
)->tuple[DetectResult, dict[str, np.ndarray]]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is empty")
    debug_images: dict[str, np.ndarray] = {}
    height, width = image_bgr.shape[:2]
    if config.debug:
        debug_images["01_original"] = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if config.debug:
        debug_images["02_gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    blurred = cv2.GaussianBlur(gray, config.blur_kernel)

    if config.debug:
        debug_images["03_blurred"] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    dark_mask, used_threshold = _threshold_dark(blurred, config.threshold_value)

    if config.debug:
        threshold_vis = dark_mask.copy()
        cv2.putText(
            threshold_vis,
            f"threshold = {used_threshold}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            255,
            2,
        )
        debug_images["04_dark_mask_raw"] = cv2.cvtColor(threshold_vis, cv2.COLOR_GRAY2BGR)

    pass

def process_one_image(
    image_path:Path, 
    config:dict, 
    output_dir:Path,
    mode: str = "fast"
)->dict[str, Any]:
    run_config = replace(config, debug = False) if mode == "fast" else config
    started = time.perf_counter()



    pass

def is_img_file(path: Path) -> bool:
    file_suffix = path.suffix.lower()
    return file_suffix in IMAGE_EXTENSIONS

def collect_image_paths(path:str) -> list[Path]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.is_file():
        if not is_img_file(path):
            raise ValueError(f"不支持的图像格式：{path.suffix}")
        return [path]
    images = sorted(
        image_path
        for image_path in path.iterdir()
        if is_img_file(image_path) 
    )

    if not images:
        raise FileNotFoundError(f"文件夹中未找到图像：{path}")
    
    return images

parser = argparse.ArgumentParser(description="处理mesh检测与定位")
parser.add_argument("--image_dir", default=None, help="输入图像路径")
parser.add_argument("--config", default=None, help="JSON配置路径")
parser.add_argument("--output_dir", default=None, help="输出目录")
parser.add_argument("--mode", default="fast", help="运行模式（fast/store-img）")
args = parser.parse_args()

for img in collect_image_paths(args.image_dir):
    process_one_image(img, args.config, args.output_dir)