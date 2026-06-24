import cv2
import numpy as np
from typing import Dict, Optional

def show_debug_images(
    debug_images: Dict[str, np.ndarray], 
    max_width: int = 1000, 
    max_height: int = 800,
    wait_key: bool = True
) -> None:
    """
    批量显示调试图像，并自动处理自适应缩放，防止窗口超出屏幕。
    
    :param debug_images: 包含图像名称和图像的字典 (ctx.debug_images)
    :param max_width: 窗口允许的最大宽度
    :param max_height: 窗口允许的最大高度
    :param wait_key: 是否阻塞等待按键 (True 则按任意键关闭，False 则不阻塞)
    """
    if not debug_images:
        print("[Warning] 没有可显示的调试图像。")
        return

    for name, img in debug_images.items():
        # 1. 防御性检查：确保图像有效
        if img is None or not isinstance(img, np.ndarray):
            print(f"[Warning] 调试图像 '{name}' 为空或格式错误，已跳过。")
            continue
            
        # 2. 自适应缩放逻辑
        h, w = img.shape[:2]
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            # 缩小图像使用 INTER_AREA 算法，画质更好
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # 3. 显示图像
        cv2.imshow(name, img)

    # 4. 等待按键并销毁窗口
    if wait_key:
        print("[Info] 调试图像已显示，按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()