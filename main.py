"""
工位1 - 环光图像 mesh 检测与定位

功能:
1. 统计图像中 mesh 数量
2. 定位每个 mesh 中心与角度 (供机械臂抓取)
3. 以最左侧 mesh 中心为原点建立相对坐标系
4. 无 mesh 时返回空结果, 便于流程跳转到下一工位
5. 保存各处理步骤调试图 (每张图单独子文件夹)

依赖: pip install opencv-python numpy

用法:
    python mesh_ringlight_detector.py
    python mesh_ringlight_detector.py --image 1ms-环光.bmp --output-dir output/mesh_detect
    python mesh_ringlight_detector.py --image ./images --output-dir output/mesh_detect
    python mesh_ringlight_detector.py --image ./images --fast
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DetectorConfig:
  blur_kernel: int = 5
  morph_kernel: int = 3
  morph_close_iter: int = 1
  threshold_value: int = 0  # 0 = Otsu
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
  pixel_size_mm: float = 0.0  # 0 表示仅输出像素坐标
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
  offset_y_mm: float | None


@dataclass
class DetectionResult:
  found: bool
  mesh_count: int
  origin_x_px: float
  origin_y_px: float
  origin_description: str
  meshes: list[MeshTarget]
  next_action: str
  message: str


def load_image(path: str | Path) -> np.ndarray:
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(path)

  data = path.read_bytes()
  image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
  if image is None:
    raise ValueError(f"无法解码图像: {path}")
  return image


def load_config(path: str | Path | None) -> DetectorConfig:
  if path is None:
    return DetectorConfig()

  with open(path, "r", encoding="utf-8") as file:
    raw = json.load(file)
  defaults = DetectorConfig()
  return DetectorConfig(**{k: raw.get(k, getattr(defaults, k)) for k in defaults.__dataclass_fields__})


def _blur(gray: np.ndarray, kernel: int) -> np.ndarray:
  kernel = max(1, int(kernel))
  if kernel % 2 == 0:
    kernel += 1
  if kernel <= 1:
    return gray
  return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def _threshold_dark(gray: np.ndarray, threshold_value: int) -> tuple[np.ndarray, int]:
  if threshold_value > 0:
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return mask, threshold_value

  threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  return mask, int(threshold)


def _morph_mask(mask: np.ndarray, kernel_size: int, close_iter: int) -> np.ndarray:
  kernel_size = max(1, int(kernel_size))
  if kernel_size <= 1:
    return mask

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
  opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
  if close_iter > 0:
    opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
  return opened


def _apply_edge_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
  masked = mask.copy()
  h, w = masked.shape
  margin = max(0, int(margin_px))
  if margin <= 0:
    return masked

  masked[:, :margin] = 0
  masked[:, w - margin :] = 0
  masked[: int(h * 0.08), :] = 0
  return masked


def _rect_angle_deg(rect: tuple[tuple[float, float], tuple[float, float], float]) -> float:
  (_, _), (width, height), angle = rect
  angle = float(angle)
  if width < height:
    angle += 90.0
  while angle > 45.0:
    angle -= 90.0
  while angle < -45.0:
    angle += 90.0
  return angle


def _split_wide_blob(
  contour: np.ndarray,
  nominal_width: float,
  split_ratio: float,
) -> list[np.ndarray]:
  x, y, width, height = cv2.boundingRect(contour)
  if width < nominal_width * split_ratio:
    return [contour]

  count = max(2, int(round(width / nominal_width)))
  step = width / count
  points = contour.reshape(-1, 2)
  slices: list[np.ndarray] = []

  for index in range(count):
    x1 = x + int(round(index * step))
    x2 = x + int(round((index + 1) * step))
    selected = points[(points[:, 0] >= x1) & (points[:, 0] < x2)]
    if len(selected) < 8:
      continue
    slices.append(selected.reshape(-1, 1, 2).astype(np.int32))

  return slices if slices else [contour]


def _contour_to_mesh(contour: np.ndarray) -> tuple[float, float, float, float, float] | None:
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


def _find_projection_peaks(
  roi_mask: np.ndarray,
  roi_y0: int,
  config: DetectorConfig,
  y_center: float | None = None,
  y_half_width: float = 120.0,
) -> list[tuple[float, float]]:
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


def _cluster_main_row(
  centers: list[tuple[float, float, float, float, float]],
  y_tolerance_px: float = 80.0,
) -> list[tuple[float, float, float, float, float]]:
  """保留同一水平排上的 mesh, 去掉投影误检的上下镜像点。"""
  if not centers:
    return []

  ys = np.array([item[1] for item in centers], dtype=np.float32)
  sorted_indices = np.argsort(ys)
  best_cluster: list[int] = []
  cluster_start = 0

  for index in range(1, len(sorted_indices) + 1):
    if index < len(sorted_indices):
      prev_y = ys[sorted_indices[index - 1]]
      curr_y = ys[sorted_indices[index]]
      if curr_y - prev_y <= y_tolerance_px:
        continue

    cluster = sorted_indices[cluster_start:index]
    if len(cluster) > len(best_cluster):
      best_cluster = cluster.tolist()
    cluster_start = index

  row = [centers[index] for index in best_cluster]
  row.sort(key=lambda item: item[0])
  return row


def _deduplicate_by_x(
  centers: list[tuple[float, float, float, float, float]],
  min_distance_px: float = 35.0,
) -> list[tuple[float, float, float, float, float]]:
  if not centers:
    return []

  deduplicated: list[tuple[float, float, float, float, float]] = []
  for center in sorted(centers, key=lambda item: item[0]):
    if not deduplicated:
      deduplicated.append(center)
      continue

    last = deduplicated[-1]
    if center[0] - last[0] < min_distance_px:
      last_area = last[2] * last[3]
      center_area = center[2] * center[3]
      if center_area > last_area:
        deduplicated[-1] = center
      continue

    deduplicated.append(center)

  return deduplicated


def _recover_missing_by_projection(
  row_centers: list[tuple[float, float, float, float, float]],
  projection_centers: list[tuple[float, float]],
  y_tolerance_px: float = 80.0,
  min_distance_px: float = 40.0,
) -> list[tuple[float, float, float, float, float]]:
  if not row_centers:
    return []

  row_y = float(np.median([item[1] for item in row_centers]))
  recovered = list(row_centers)

  for px, py in projection_centers:
    if abs(py - row_y) > y_tolerance_px:
      continue
    if any(abs(px - item[0]) < min_distance_px for item in recovered):
      continue
    recovered.append((px, row_y, 80.0, 370.0, 0.0))

  recovered.sort(key=lambda item: item[0])
  return recovered


def detect_meshes_ringlight(
  image_bgr: np.ndarray,
  config: DetectorConfig = DetectorConfig(),
) -> tuple[DetectionResult, dict[str, np.ndarray]]:
  if image_bgr is None or image_bgr.size == 0:
    raise ValueError("image_bgr is empty")

  debug_images: dict[str, np.ndarray] = {}
  height, width = image_bgr.shape[:2]

  if config.debug:
    debug_images["01_original"] = image_bgr.copy()

  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  if config.debug:
    debug_images["02_gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

  blurred = _blur(gray, config.blur_kernel)
  if config.debug:
    debug_images["03_blurred"] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

  dark_mask, used_threshold = _threshold_dark(blurred, config.threshold_value)
  if config.debug:
    threshold_vis = dark_mask.copy()
    cv2.putText(
      threshold_vis,
      f"threshold={used_threshold}",
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.0,
      255,
      2,
    )
    debug_images["04_dark_mask_raw"] = cv2.cvtColor(threshold_vis, cv2.COLOR_GRAY2BGR)

  morphed = _morph_mask(dark_mask, config.morph_kernel, config.morph_close_iter)
  if config.debug:
    debug_images["05_dark_mask_morph"] = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)

  edge_masked = _apply_edge_mask(morphed, config.edge_margin_px)
  if config.debug:
    debug_images["06_edge_masked"] = cv2.cvtColor(edge_masked, cv2.COLOR_GRAY2BGR)

  roi_y0 = int(height * config.roi_y_ratio)
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
      _split_wide_blob(contour, config.nominal_mesh_width_px, config.split_width_ratio)
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

  main_row = _cluster_main_row(contour_centers)
  deduplicated = _deduplicate_by_x(main_row)

  if deduplicated:
    row_y = float(np.median([item[1] for item in deduplicated]))
    row_projection = _find_projection_peaks(
      roi_mask,
      roi_y0,
      config,
      y_center=row_y,
      y_half_width=100.0,
    )
    merged_centers = _recover_missing_by_projection(deduplicated, row_projection)
  else:
    merged_centers = _recover_missing_by_projection([], projection_centers)
  if not merged_centers:
    result = DetectionResult(
      found=False,
      mesh_count=0,
      origin_x_px=0.0,
      origin_y_px=0.0,
      origin_description="无有效 mesh, 未建立坐标系",
      meshes=[],
      next_action="move_to_next_station",
      message="未检测到 mesh, 请移动到下一工位/下一拍照位",
    )
    if config.debug:
      empty_vis = image_bgr.copy()
      cv2.putText(
        empty_vis,
        "NO MESH FOUND - MOVE NEXT",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
      )
      debug_images["11_final_result"] = empty_vis
    return result, debug_images

  origin_x, origin_y = merged_centers[0][0], merged_centers[0][1]
  meshes: list[MeshTarget] = []
  for index, (cx, cy, mesh_width, mesh_height, angle) in enumerate(merged_centers):
    offset_x = cx - origin_x
    offset_y = cy - origin_y
    offset_x_mm = offset_x * config.pixel_size_mm if config.pixel_size_mm > 0 else None
    offset_y_mm = offset_y * config.pixel_size_mm if config.pixel_size_mm > 0 else None
    meshes.append(
      MeshTarget(
        index=index,
        center_x_px=round(cx, 2),
        center_y_px=round(cy, 2),
        width_px=round(mesh_width, 2),
        height_px=round(mesh_height, 2),
        angle_deg=round(angle, 2),
        offset_x_px=round(offset_x, 2),
        offset_y_px=round(offset_y, 2),
        offset_x_mm=round(offset_x_mm, 3) if offset_x_mm is not None else None,
        offset_y_mm=round(offset_y_mm, 3) if offset_y_mm is not None else None,
      )
    )

  final_vis = image_bgr.copy()
  cv2.drawMarker(
    final_vis,
    (int(round(origin_x)), int(round(origin_y))),
    (255, 0, 255),
    markerType=cv2.MARKER_CROSS,
    markerSize=30,
    thickness=3,
  )
  cv2.putText(
    final_vis,
    "ORIGIN (leftmost mesh)",
    (int(round(origin_x)) + 20, int(round(origin_y)) - 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 255),
    2,
  )

  for mesh in meshes:
    point = (int(round(mesh.center_x_px)), int(round(mesh.center_y_px)))
    cv2.circle(final_vis, point, 8, (0, 255, 0), 2)
    cv2.putText(
      final_vis,
      f"#{mesh.index}",
      (point[0] + 10, point[1] - 10),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (0, 255, 0),
      2,
    )
    cv2.putText(
      final_vis,
      f"({mesh.offset_x_px:.0f},{mesh.offset_y_px:.0f})",
      (point[0] + 10, point[1] + 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 0),
      1,
    )

  cv2.putText(
    final_vis,
    f"COUNT={len(meshes)}",
    (30, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.4,
    (0, 255, 0),
    3,
  )
  if config.debug:
    debug_images["11_final_result"] = final_vis

  result = DetectionResult(
    found=True,
    mesh_count=len(meshes),
    origin_x_px=round(origin_x, 2),
    origin_y_px=round(origin_y, 2),
    origin_description="最左侧 mesh 中心作为相对坐标原点 (0,0)",
    meshes=meshes,
    next_action="pick_mesh",
    message=f"检测到 {len(meshes)} 个 mesh, 可按 index 顺序抓取",
  )
  return result, debug_images


def save_debug_images(debug_images: dict[str, np.ndarray], output_dir: str | Path) -> None:
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  for name, image in debug_images.items():
    file_path = output_path / f"{name}.png"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
      raise RuntimeError(f"无法编码调试图: {file_path}")
    file_path.write_bytes(encoded.tobytes())


def result_to_dict(result: DetectionResult) -> dict[str, Any]:
  payload = asdict(result)
  return payload


def find_default_image(search_dir: Path) -> Path:
  candidates = sorted(search_dir.glob("1ms*.bmp"))
  if not candidates:
    raise FileNotFoundError(f"未找到 1ms 环光图: {search_dir}")
  return candidates[0]


def is_image_file(path: Path) -> bool:
  return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(path: Path) -> list[Path]:
  if not path.exists():
    raise FileNotFoundError(path)

  if path.is_file():
    if not is_image_file(path):
      raise ValueError(f"不支持的图像格式: {path}")
    return [path]

  images = sorted(
    file_path
    for file_path in path.iterdir()
    if is_image_file(file_path)
  )
  if not images:
    raise FileNotFoundError(f"文件夹中未找到图像: {path}")
  return images


def image_output_dir(base_output_dir: Path, image_path: Path) -> Path:
  return base_output_dir / image_path.stem


def process_one_image(
  image_path: Path,
  config: DetectorConfig,
  output_dir: Path,
  fast: bool,
) -> dict[str, Any]:
  run_config = replace(config, debug=False) if fast else config
  started = time.perf_counter()

  image = load_image(image_path)
  result, debug_images = detect_meshes_ringlight(image, run_config)
  elapsed_ms = (time.perf_counter() - started) * 1000.0

  payload = result_to_dict(result)
  payload["image"] = str(image_path)
  payload["elapsed_ms"] = round(elapsed_ms, 1)

  if not fast and debug_images:
    image_dir = image_output_dir(output_dir, image_path)
    save_debug_images(debug_images, image_dir)
    with open(image_dir / "result.json", "w", encoding="utf-8") as file:
      json.dump(payload, file, ensure_ascii=False, indent=2)
    payload["process_images_saved_to"] = str(image_dir)

  return payload


def main() -> int:
  parser = argparse.ArgumentParser(description="工位1 环光 mesh 检测与定位")
  parser.add_argument(
    "--image",
    default=None,
    help="输入图像路径或文件夹路径, 默认自动查找当前目录 1ms*.bmp",
  )
  parser.add_argument("--config", default=None, help="JSON 配置路径")
  parser.add_argument("--output-dir", default=None, help="输出根目录")
  parser.add_argument(
    "--fast",
    action="store_true",
    help="快速模式: 不生成过程图, 仅输出检测结果",
  )
  args = parser.parse_args()

  script_dir = Path(__file__).resolve().parent
  input_path = Path(args.image) if args.image else script_dir
  config = load_config(args.config)
  output_dir = Path(args.output_dir or config.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  image_paths = (
    collect_image_paths(input_path)
    if args.image
    else [find_default_image(script_dir)]
  )

  batch_results: list[dict[str, Any]] = []
  for image_path in image_paths:
    payload = process_one_image(image_path, config, output_dir, args.fast)
    batch_results.append(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload.get("process_images_saved_to"):
      print(f"process_images_saved_to: {payload['process_images_saved_to']}")

  if args.fast:
    if len(batch_results) == 1:
      summary_path = output_dir / "result.json"
      summary_payload = batch_results[0]
    else:
      summary_path = output_dir / "batch_results.json"
      summary_payload = {
        "image_count": len(batch_results),
        "found_count": sum(1 for item in batch_results if item["found"]),
        "fast_mode": True,
        "results": batch_results,
      }
    with open(summary_path, "w", encoding="utf-8") as file:
      json.dump(summary_payload, file, ensure_ascii=False, indent=2)
    print(f"summary_saved_to: {summary_path}")
  elif len(image_paths) > 1:
    summary_path = output_dir / "batch_results.json"
    summary_payload = {
      "image_count": len(batch_results),
      "found_count": sum(1 for item in batch_results if item["found"]),
      "fast_mode": False,
      "results": batch_results,
    }
    with open(summary_path, "w", encoding="utf-8") as file:
      json.dump(summary_payload, file, ensure_ascii=False, indent=2)
    print(f"summary_saved_to: {summary_path}")

  all_found = all(item["found"] for item in batch_results)
  return 0 if all_found else 2


if __name__ == "__main__":
  raise SystemExit(main())
