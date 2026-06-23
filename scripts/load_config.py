import argparse
import json
import os
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}

def process_one_image(
    image_path:Path, 
    config:dict, 
    output_dir:Path
)->dict[str, Any]:
    
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