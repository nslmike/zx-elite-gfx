"""Данные на границах этапов; внутри арифметики сохраняется знак-модуль.

Адреса в комментариях относятся к закреплённому листингу Rasmus-M,
commit 66ee08efb2aceddbd67d5003c28fed3477b34850.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Vertex:
    x: int
    y: int
    z: int
    distance_limit: int
    faces: tuple[int, int, int, int]


@dataclass(frozen=True)
class Edge:
    vertex_a: int
    vertex_b: int
    faces: tuple[int, int]
    distance_limit: int


@dataclass(frozen=True)
class Face:
    nx: int
    ny: int
    nz: int
    distance_limit: int


@dataclass(frozen=True)
class Model:
    name: str
    vertices: tuple[Vertex, ...]
    edges: tuple[Edge, ...]
    faces: tuple[Face, ...]
    normal_scale: int
    face_vertices: tuple[int, ...] = ()  # По одной опорной вершине на грань.
    vertex_test_default: bool = False
    hull_triangles: tuple[tuple[int, int, int], ...] = ()  # Порядок граней из таблицы, обход наружу.

    @property
    def default_face_test(self) -> str:
        if self.hull_triangles:
            return "hull"
        return "vertex" if self.vertex_test_default else "elite"


@dataclass(frozen=True)
class Pose:
    """Положение и старшие байты nose/roof исходного 39-байтного состояния.

    UI округляет базис до [-127,127]. Эти же байты получает Z80-эталон.
    Это управление учебным просмотрщиком, а не перенос игровой физики.
    """

    position: tuple[int, int, int] = (0, 0, 1024)
    nose: tuple[int, int, int] = (0, 0, 127)
    roof: tuple[int, int, int] = (0, 127, 0)


@dataclass(frozen=True)
class ProjectedVertex:
    """Три байта записи исходной таблицы $A8FA.

    flags: bit0 — отрицательный мировой Y (на экране вниз),
    bit1 — отрицательный X, bit7 — недопустимая вершина.
    Нулевой модуль со знаком намеренно не нормализуется.
    """

    xmag: int
    ymag: int
    flags: int

    @property
    def x(self) -> int:
        return 127 + (-self.xmag if self.flags & 2 else self.xmag)

    @property
    def y(self) -> int:
        return 63 + (self.ymag if self.flags & 1 else -self.ymag)

    @property
    def invalid(self) -> bool:
        return bool(self.flags & 128)

    @classmethod
    def from_screen(cls, x: int, y: int) -> "ProjectedVertex":
        dx, dy = x - 127, y - 63
        return cls(abs(dx), abs(dy), (2 if dx < 0 else 0) | (1 if dy > 0 else 0))


@dataclass(frozen=True)
class EdgeDecision:
    index: int
    draw: bool
    reason: str
    start: ProjectedVertex | None = None
    end: ProjectedVertex | None = None


@dataclass
class GeometryTrace:
    distance: int
    face_flags: list[int]
    projected: list[ProjectedVertex | None]
    edge_decisions: list[EdgeDecision]
    details: dict = field(default_factory=dict)
