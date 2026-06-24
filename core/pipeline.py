from typing import Callable
import numpy as np
import cv2
# 导入上面定义的 Context
from core.context import PipelineContext 

class ImagePipeline:
    def __init__(self):
        # 注册表：按顺序存放处理函数
        self._steps: list[Callable] = []

    def register(self, step_func: Callable):
        """将处理函数注册到流水线中（支持作为装饰器使用）"""
        self._steps.append(step_func)
        return step_func  

    def run(self, image: np.ndarray, config: dict) -> PipelineContext:
        """启动流水线，自动处理 Debug 图像的保存"""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        # 1. 初始化上下文背包
        ctx = PipelineContext(image=image)
        
        # 2. 自动保存原始图像
        if config.debug:
            ctx.debug_images["00_original"] = image.copy()

        # 3. 按顺序执行所有注册的步骤
        for step in self._steps:
            # 将上下文背包和配置传入处理函数
            # 约定：每个步骤必须接收 ctx，修改 ctx 后返回 ctx
            ctx = step(ctx, config)
            
            # 核心：自动拦截并保存中间图像，无需在算法内部写 if debug
            if config.debug:
                # 如果 metadata 中刚刚被写入了新的 debug_key
                # 或者我们在函数中直接把图像塞进了 debug_images，都可以
                pass  # 这里可以保持原样，或者让算法函数自己往 ctx.debug_images 里塞图

        return ctx