"""core.context

PipelineContext dataclass.

说明：
- 在 dataclasses 字段上使用可变默认值（例如直接写 `metadata: Dict[str, Any] = {}`）会在模块/类定义时创建一个字典对象，
  该对象会被所有实例共享，导致所谓的“可变默认值陷阱”。
- 推荐的做法是使用 `dataclasses.field(default_factory=...)`，例如 `field(default_factory=dict)`。
  dataclasses 会在每次创建实例时调用这个工厂函数（这里是 `dict()`），从而为每个实例生成独立的字典对象，
  避免实例间意外共享可变状态。

简短示例：

正确（使用 default_factory，每个实例有自己的字典）：

```python
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

@dataclass
class WithFactory:
    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

a = WithFactory(image=np.zeros((1, 1)))
b = WithFactory(image=np.zeros((1, 1)))
a.metadata['x'] = 1
assert b.metadata == {}  # 互不影响
```

错误（直接写字面量，会被所有实例共享）：

```python
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class WithLiteral:
    image: np.ndarray
    metadata: Dict[str, Any] = {}  # 不要这样写

c = WithLiteral(image=np.zeros((1, 1)))
d = WithLiteral(image=np.zeros((1, 1)))
c.metadata['y'] = 2
assert d.metadata == {'y': 2}  # 因为共享同一个 dict
```
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class PipelineContext:
    # 当前正在处理的图像
    image: np.ndarray 
    
    # 存放各种中间变量、额外参数（阈值、轮廓等）
    # 使用 field(default_factory=dict) 确保每个实例都有独立的字典，避免可变默认值被共享
    metadata: Dict[str, Any] = field(default_factory=dict) 
    
    # 存放需要自动保存的 debug 图像
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
