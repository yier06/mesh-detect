# 特征提取：轮廓、质心、角度
import cv2
import numpy as np
from typing import Dict, Any,Tuple, Optional
from core.context import PipelineContext
import utils.visualization as vis

def _split_wide_blob(contour: np.ndarray, config: Dict[str, Any]) -> list[np.ndarray]:
    nominal_width = config.nominal_mesh_width_px
    split_ratio = config.split_width_ratio
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

# 轮廓检测
def find_contours(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    contours, _ = cv2.findContours(ctx.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctx.metadata["contours"] = contours
    contour_candidates: list[np.ndarray] = []
    height, width = ctx.image.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_area or area > height * width * config.max_area_ratio:
            continue
        _, _, box_width, box_height = cv2.boundingRect(contour)
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect < config.min_aspect:
            continue
        contour_candidates.extend(_split_wide_blob(contour, config))
    ctx.metadata["contour_candidates"] = contour_candidates
    contour_centers: list[tuple[float, float, float, float, float]] = []
    component_vis = ctx.debug_images["00_original"].copy()
    for contour in contour_candidates:
        mesh = _contour_to_mesh(contour)
        if mesh is None:
            continue
        cx, cy, mesh_width, mesh_height, angle = mesh
        if not (200 <= mesh_height <= 450 and 40 <= mesh_width <= 120):
            continue
        contour_centers.append(mesh)
        if config.debug:
            vis.draw_contour(component_vis, contour, color="red", thickness=2)
            vis.draw_circle(component_vis, (int(round(cx)), int(round(cy))), radius=5, color="green", thickness=-1)

    ctx.metadata["contour_centers"] = contour_centers
    if config.debug:
        ctx.debug_images["09_contour_components"] = component_vis
    return ctx

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




