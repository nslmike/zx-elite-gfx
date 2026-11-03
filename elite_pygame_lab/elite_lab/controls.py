"""Современное управление позой просмотрщика; не игровая физика Elite.

Тригонометрия используется только для удобного задания входа. Рендерер и
Z80-эталон получают одинаковые квантованные старшие байты nose/roof.
"""

from math import cos, radians, sin

from .types import Pose


def pose_from_angles(yaw: float, pitch: float, roll: float, depth: int) -> Pose:
    cy, sy = cos(radians(yaw)), sin(radians(yaw))
    cp, sp = cos(radians(pitch)), sin(radians(pitch))
    cr, sr = cos(radians(roll)), sin(radians(roll))

    def rotate(x: float, y: float, z: float) -> tuple[int, int, int]:
        x, z = cy * x + sy * z, -sy * x + cy * z
        y, z = cp * y - sp * z, sp * y + cp * z
        x, y = cr * x - sr * y, sr * x + cr * y
        return tuple(max(-127, min(127, round(v * 127))) for v in (x, y, z))

    return Pose((0, 0, depth), rotate(0, 0, 1), rotate(0, 1, 0))
