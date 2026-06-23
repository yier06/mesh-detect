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

def _find_projection_peaks(
    roi_mask: np.ndarray,
    roi_y0: int,
    config: DetectorConfig,
    y_center: float | None = None,
    y_half_width: float = 120.0,
)-> list[tuple[float, float]]:
    band_mask = roi_mask.copy()
    if y_center is not None:
        band_mask = np.zeros_like(roi_mask)
        y1 = max(0, int(y_center - y_half_width) - roi_y0)
        y2 = min(roi_mask.shape[0], int(y_center + y_half_width) - roi_y0)
        if y2 > y1:
            band_mask[y1:y2, :] = roi_mask[y1:y2, :]
    
    projection = np.sum(band_mask > 0, axis=0).astype(np.float32)
    kernel = max(3, int(config.projection_smooth_kernel))
    if kernel % 2 == 0:
        kernel += 1
    smooth = np.convolve(projection, np.ones(kernel, dtype=np.float32) / kernel, mode="same")
    threshold = float(smooth.max() * config.min_peak_projection)
    peaks: list[int] = []
    for x in range(config.edge_margin_px, roi_mask.shape[1] - config.edge_margin_px):
        if smooth[x] < threshold:
            continue
        if smooth[x] < smooth[x - 1] or smooth[x] < smooth[x + 1]:
            continue
        if peaks and x - peaks[-1] < config.min_peak_distance_px:
            if smooth[x] > smooth[peaks[-1]]:
                peaks[-1] = x
            continue
        peaks.append(x)
    centers: list[tuple[float, float]] = []
    half_window = 45
    for peak_x in peaks:
        x1 = max(0, peak_x - half_window)
        x2 = min(roi_mask.shape[1], peak_x + half_window)
        patch = roi_mask[:, x1:x2]
        ys, xs = np.where(patch > 0)
        if len(xs) < 30:
            continue
        center_x = x1 + float(np.mean(xs))
        center_y = roi_y0 + float(np.mean(ys))
        centers.append((center_x, center_y))
    return centers

def _rect_angle_deg(
    rect: tuple[tuple[float, float], tuple[float, float], float]
)->float:
    (_, _), (width, height), angle = rect
    angle = float(angle)
    if width < height:
        angle += 90.0
    while angle >45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return angle

def _contour_to_mesh(
    contour: np,ndarray
)-> tuple[float, float, float, float, float] | None:
    area = cv2.contourArea(contour)
    if area <= 0:
        return None
    
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    
    rect = cv2.minAreaRect(contour)
    (_, _), (width, height), _ = rect
    short_side = max(1.0, min(width, height))
    long_side = max(width, height)
    aspect = long_side / short_side

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    mesh_width = short_side
    mesh_height = long_side
    return center_x, center_y, mesh_width, mesh_height, _rect_angle_deg(rect)

def _split_wide_blob(
    contour: np.ndarray,
    nominal_width: float,
    split_ratio: float,
)-> list[np.ndarray]:
    x, y, width, height = cv2.boundingRect(contour)
    if width < nominal_width * split_ratio:
        return [contour]
    
    count = max(2, int(round(width / nominal_width)))
    step = width / count

    points = contour.reshape(-1, 2)
    slices: list[np.ndarray] = []

    for index in range(count):
        x1 = x + int(round(index * step))
        x2 = x + int(round(index + 1) * step)
        selected = points[(points[:, 0] >= x1) & (points[:, 0] < x2)]
        if len(selected) < 8:
            continue
        slices.append(selected.reshape(-1, 1, 2).astype(np.int32))
    
    return slices if slices else [contour]



def _apply_edge_mask(
    mask: np.ndarray,
    margin_px: int
)-> np.ndarray:
    masked = mask.copy()
    h, w = mask.shape
    margin = max(0,int(margin_px))
    if margin <= 0:
        return masked
    masked[:, :margin] = 0
    masked[:, w-margin:] = 0
    masked[:, int(h*0.08), :] = 0
    return masked

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

    morphed = _morph_mask(dark_mask, config.morph_kernel, config.morgh_close_iter)
    if config.debug:
        debug_images["05_dark_mask_morph"] = cv2.cvtColor(morphed,cv2.COLOR_GRAY2BGR)
    
    edge_masked = _apply_edge_mask(morphed, config.edge_argin_px)
    if config.debug:
        debug_images["06-edge_masked"] = cv2.cvtColor(edge_masked, cv2.COLOR_GRAY2BGR)
    
    roi_y0 = int(height * config.roi_y_ration)
    roi_mask = np.zeros_like(edge_masked)
    roi_mask[roi_y0:, :] = edge_masked[roi_y0:, :]
    if config.debug:
        roi_vis = image_bgr.copy()
        cv2.rectangle(roi_vis, (0, roi_y0), (width - 1, height - 1), (0, 255, 255), 2)
        cv2.rectangle(
            roi_vis, 
            (config.edge_margin_px, roi_y0),
            (width - config.edge_margin_px, height - 1),
            (255, 0, 0),
            2,
            )
        debug_images["07_roi_overlay"] = roi_vis
        debug_images["08_roi_mask"] = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)

    max_area = height * width * config.max_area_ratio
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_candidates: list[np.ndarray] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_area or area > max_area:
            continue
        
        _, _, box_width, box_height = cv2.boundingRect(contour)
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect < config.min_aspect:
            continue

        contour_candidates.extend(
            _split_wide_blob(contour, config.nominal_mesh_width_px, config.split_ratio)

        )

        contour_centers: list[tuple[float, float, float, float, float]] = []
        component_vis = image_bgr.copy()
        for contour in contour_candidates:
            mesh = _contour_to_mesh(contour)
            if mesh is None:
                continue

            cx, cy, mesh_width, mesh_height, angle = mesh
            if not (200 <= mesh_height <= 450 and 40 <= mesh_width <= 120):
                continue

            contour_centers.append(mesh)
            cv2.drawContours(component_vis, [contour], -1, (255, 0, 0), 2)
            cv2.circle(component_vis, (int(round(cx)), int(round(cy))), 5, (0, 255, 0), -1)
        if config.debug:
            debug_images["09_contour_components"] = component_vis
        
        projection_centers = _find_projection_peaks(roi_mask, roi_y0, config)
        projection_vis = image_bgr.copy()
        for peak_x, peak_y in projection_centers:
            cv2.circle(projection_vis, (int(round(peak_x)), int(round(peak_y))), 8, (0, 0, 255), 2)
        if config.debug:
            debug_images["10_projection_centers"] = projection_vis

def process_one_image(
    image_path:Path, 
    config:dict, 
    output_dir:Path,
    mode: str = "fast"
)->dict[str, Any]:
    run_config = replace(config, debug = False) if mode == "fast" else config
    started = time.perf_counter()


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