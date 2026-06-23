from dataclasses import dataclass

@dataclass(frozen=True)
class DetectorConfig:
  blur_kernel: int = 5
  morph_kernel: int = 3
  morph_close_iter: int = 1
  threshold_value: int = 0  # 0 = Otsu
  roi_y_ratio: float = 0.50
  edge_margin_px: int = 280
  min_area: int = 8000
  max_area_ratio: float = 1.0 / 25.0
  min_aspect: float = 1.5
  split_width_ratio: float = 1.45
  nominal_mesh_width_px: float = 85.0
  min_peak_projection: float = 0.30
  projection_smooth_kernel: int = 21
  min_peak_distance_px: int = 40
  pixel_size_mm: float = 0.0  # 0 表示仅输出像素坐标
  output_dir: str = "output/mesh_detect"
  debug: bool = True


@dataclass
class MeshTarget:
  index: int
  center_x_px: float
  center_y_px: float
  width_px: float
  height_px: float
  angle_deg: float
  offset_x_px: float
  offset_y_px: float
  offset_x_mm: float | None
  offset_y_mm: float | None


@dataclass
class DetectionResult:
  found: bool
  mesh_count: int
  origin_x_px: float
  origin_y_px: float
  origin_description: str
  meshes: list[MeshTarget]
  next_action: str
  message: str