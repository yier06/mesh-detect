# 预处理：形态学、边缘裁剪
import cv2
import numpy as np
from typing import Dict, Any
from core.context import PipelineContext

# 灰度化
def to_grayscale(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    gray = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)
    ctx.image = gray
    if config.debug:
        ctx.debug_images["02_gray"] = gray
    return ctx


# 高斯模糊
def gaussian_blur(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    blurred = cv2.GaussianBlur(ctx.image, (config.blur_kernel, config.blur_kernel), 0)
    ctx.image = blurred
    if config.debug:
        ctx.debug_images["03_blurred"] = blurred
    return ctx

# 二值化
def threshold_dark(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    threshold_value = config.threshold_value
    if threshold_value > 0:
        _, mask = cv2.threshold(ctx.image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        ctx.image = mask
        ctx.metadata["threshold"] = threshold_value
        if config.debug:
            ctx.debug_images["04_dark_mask_raw"] = mask
    else:
        threshold, mask = cv2.threshold(ctx.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ctx.image = mask
        ctx.metadata["threshold"] = int(threshold)
        if config.debug:
            ctx.debug_images["04_dark_mask_raw"] = mask
    return ctx

# 形态学操作
def morph_mask(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    kernel_size = config.morph_kernel
    if kernel_size <= 1:
        return ctx
    close_iter = config.morph_close_iter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(ctx.image, cv2.MORPH_OPEN, kernel, iterations=1)
    if close_iter > 0:
        opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        ctx.image = opened
        if config.debug:
            ctx.debug_images["05_dark_mask_morph"] = opened
    return ctx

# 边缘裁剪
def apply_edge_mask(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    margin_px = config.edge_margin_px
    if margin_px <= 0:
        return ctx
    masked = ctx.image.copy()
    h, w = masked.shape
    margin = max(0, int(margin_px))
    masked[:, :margin] = 0
    masked[:, w-margin:] = 0
    masked[: int(h*0.08), :] = 0
    ctx.image = masked
    if config.debug:
        ctx.debug_images["06_edge_masked"] = masked
    return ctx

# 区域裁剪
def apply_roi_mask(ctx: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
    roi_y_ratio = config.roi_y_ratio
    if roi_y_ratio <= 0 or roi_y_ratio >= 1:
        return ctx
    h, w = ctx.image.shape
    roi_y0 = int(h * roi_y_ratio)
    roi_mask = np.zeros_like(ctx.image)
    roi_mask[roi_y0:, :] = ctx.image[roi_y0:, :]
    ctx.image = roi_mask
    if config.debug:
        ctx.debug_images["07_roi_overlay"] = roi_mask
    return ctx

