import os
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}

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