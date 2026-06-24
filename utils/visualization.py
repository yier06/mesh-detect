# # 画图、保存结果、调试用

import cv2
import numpy as np
from typing import Tuple, List, Union

# 定义一些常用颜色 (B, G, R)
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
}

def draw_rectangle(
    image: np.ndarray, 
    top_left: Tuple[int, int], 
    bottom_right: Tuple[int, int], 
    color: Union[str, Tuple[int, int, int]] = 'green', 
    thickness: int = 2
) -> np.ndarray:
    """绘制矩形"""
    color_val = COLORS.get(color, color) if isinstance(color, str) else color
    cv2.rectangle(image, top_left, bottom_right, color_val, thickness)
    return image

def draw_circle(
    image: np.ndarray, 
    center: Tuple[int, int], 
    radius: int = 5, 
    color: Union[str, Tuple[int, int, int]] = 'red', 
    thickness: int = -1 # -1 表示填充
) -> np.ndarray:
    """绘制圆形"""
    color_val = COLORS.get(color, color) if isinstance(color, str) else color
    cv2.circle(image, center, radius, color_val, thickness)
    return image

def draw_cross(
    image: np.ndarray, 
    center: Tuple[int, int], 
    size: int = 20, 
    color: Union[str, Tuple[int, int, int]] = 'magenta', 
    thickness: int = 2
) -> np.ndarray:
    """绘制十字标记"""
    color_val = COLORS.get(color, color) if isinstance(color, str) else color
    cv2.drawMarker(image, center, color_val, cv2.MARKER_CROSS, size, thickness)
    return image

def draw_contour(
    image: np.ndarray, 
    contour: np.ndarray, 
    color: Union[str, Tuple[int, int, int]] = 'blue', 
    thickness: int = 2
) -> np.ndarray:
    """绘制单个轮廓"""
    color_val = COLORS.get(color, color) if isinstance(color, str) else color
    cv2.drawContours(image, [contour], -1, color_val, thickness)
    return image

def put_text(
    image: np.ndarray, 
    text: str, 
    position: Tuple[int, int], 
    color: Union[str, Tuple[int, int, int]] = 'white', 
    scale: float = 0.7, 
    thickness: int = 2
) -> np.ndarray:
    """添加文字"""
    color_val = COLORS.get(color, color) if isinstance(color, str) else color
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color_val, thickness)
    return image

def save_image(image: np.ndarray, path: str) -> None:
    """保存图像"""
    cv2.imwrite(path, image)