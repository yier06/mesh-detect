# main.py
from core.pipeline import ImagePipeline
from processors import *
from utils import debug_utils
from config import *
from pathlib import Path
import cv2
import numpy as np

import os
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}

def is_img_file(path: Path) -> bool:
    file_suffix = path.suffix.lower()
    return file_suffix in IMAGE_EXTENSIONS

def collect_image_paths(path:str) -> list[Path]:
    path = Path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.is_file():
        if not is_img_file(path):
            raise ValueError(f"不支持的图像格式：{path.suffix}")
        return [path]
    images_list = sorted(
        image_path
        for image_path in path.iterdir()
        if is_img_file(image_path) 
    )

    if not images_list:
        raise FileNotFoundError(f"文件夹中未找到图像：{path}")
    

    
    return images_list


def main():
    # 1. 在入口处实例化流水线
    pipeline = ImagePipeline()
    
    # 2. 将具体的算法函数“注入”到流水线中
    pipeline.register(to_grayscale)
    pipeline.register(gaussian_blur)
    pipeline.register(threshold_dark)
    pipeline.register(morph_mask)
    pipeline.register(apply_edge_mask)
    pipeline.register(apply_roi_mask)
    pipeline.register(find_contours)
    # pipeline.register(find_projection_peaks)

    
    # 3. 读取图像并启动流水线
    raw_images_path = collect_image_paths("image/1ms-环光.bmp")
    config = DetectorConfig()
    for raw_image_path in raw_images_path:
        # 中文读取失败
        # raw_image = cv2.imread(raw_image_path)
        raw_image = cv2.imdecode(np.fromfile(raw_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        ctx = pipeline.run(raw_image, config)
        debug_utils.show_debug_images({"00_original": ctx.debug_images["00_original"]})
        debug_utils.show_debug_images({"02_gray": ctx.debug_images["02_gray"]})
        debug_utils.show_debug_images({"03_blurred": ctx.debug_images["03_blurred"]})
        debug_utils.show_debug_images({"04_dark_mask_raw": ctx.debug_images["04_dark_mask_raw"]})
        debug_utils.show_debug_images({"05_dark_mask_morph": ctx.debug_images["05_dark_mask_morph"]})
        debug_utils.show_debug_images({"06_edge_masked": ctx.debug_images["06_edge_masked"]})
        debug_utils.show_debug_images({"07_roi_overlay": ctx.debug_images["07_roi_overlay"]})
        debug_utils.show_debug_images({"09_contour_components": ctx.debug_images["09_contour_components"]})
        


if __name__ == "__main__":
    main()