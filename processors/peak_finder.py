# 高级定位：投影寻峰 (_find_projection_peaks)
import cv2
import numpy as np
from typing import Dict, Any
from core.context import PipelineContext

def find_projection_peaks(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    roi_mask = ctx.metadata["roi_mask"]
    roi_y0 = ctx.metadata["roi_y0"]
    projection_centers: list[tuple[float, float]] = []
    band_mask = roi_mask.copy()
    y_center = ctx.metadata["y_center"]
    y_half_width = ctx.metadata["y_half_width"]
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
    ctx.metadata["projection_centers"] = centers
    return ctx