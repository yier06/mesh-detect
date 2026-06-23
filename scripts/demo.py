import argparse
from pathlib import Path
from dataclasses import replace
from typing import Any
import time
from read_img import collect_image_paths
import numpy as np

def detect_meshes_ringlight(
    image_path:Path,
    config:dict,
)->tuple[dict[str, Any], dict[str, np.ndarray]]:
    pass

def process_one_image(
    image_path:Path, 
    config:dict, 
    output_dir:Path,
    mode: str = "fast"
)->dict[str, Any]:
    run_config = replace(config, debug = False) if mode == "fast" else config
    started = time.perf_counter()
    result = detect_meshes_ringlight(image_path, run_config)

def main():
    parser = argparse.ArgumentParser(description="处理mesh检测与定位")
    parser.add_argument("--image_dir", default=None, help="输入图像路径")
    parser.add_argument("--config", default=None, help="JSON配置路径")
    parser.add_argument("--output_dir", default=None, help="输出目录")
    parser.add_argument("--mode", default="fast", help="运行模式（fast/store-img）")
    args = parser.parse_args()

    for img in collect_image_paths(args.image_dir):
        process_one_image(img, args.config, args.output_dir)