"""Настройка внутреннего размера 3D-кадра на время запуска."""

from dataclasses import dataclass, field


RENDER_RESOLUTIONS = (None, (960, 540), (1280, 720), (1600, 900), (1920, 1080),
                      (2560, 1440), (3840, 2160))


@dataclass
class RenderResolution:
    selected: int = 0
    highlighted: int = 0
    unavailable: set[int] = field(default_factory=set)
    error: str | None = None
    output_size: tuple[int, int] | None = None
