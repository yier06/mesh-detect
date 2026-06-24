from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass
class PipelineContext:
    # 当前正在处理的图像
    image: np.ndarray 
    
    # 存放各种中间变量、额外参数（阈值、轮廓等）
    metadata: Dict[str, Any] = field(default_factory=dict) 
    
    # 存放需要自动保存的 debug 图像
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)