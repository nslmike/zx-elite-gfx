"""Каталог каркасов ZX Spectrum Elite из исходной таблицы $6180.

blueprints.bin содержит ровно $6180..$6FA2 закреплённого листинга и кассеты.
Каждая 4-байтная запись указывает на заголовок из 23 байт. Три смещения
от начала заголовка ведут к вершинам, рёбрам и исходным нормалям. Несколько
заголовков могут ссылаться на одну геометрию.
"""

from math import isqrt
from pathlib import Path
from struct import iter_unpack, unpack_from

from .types import Edge, Face, Model, Vertex


BLUEPRINT_BASE = 0x6180
_DATA_DIR = Path(__file__).resolve().parent.parent / "reference_data"
BLUEPRINT_DATA = (_DATA_DIR / "blueprints.bin").read_bytes()

# Индекс соответствует номеру в таблице $6180, в том числе для пар вариантов.
_NAMES = (
    "Coriolis", "Missile", "Escape pod", "Splinter", "Cargo canister",
    "Boulder", "Asteroid", "Rock Hermit", "Viper", "Cobra Mk III",
    "Python (торговый)", "Python (пиратский)", "Cobra Mk III (пиратская)",
    "Asp Mk II", "Fer-de-Lance", "Thargoid", "Sidewinder", "Krait", "Thargon",
)
COBRA_ID = 9
COBRA_IDS = (COBRA_ID, 12)  # Торговая и пиратская версии используют один каркас.
KRAIT_ID = 17
DODO_ID = len(_NAMES)  # Индекс в демо, за пределами кассетной таблицы.

# Крайние точки усов (5, 6) и рисунок на обшивке (7..16) не задают плоскости.
# Эти шесть замкнутых контуров составлены из рёбер корпуса в порядке face ID.
_KRAIT_HULL = ((0, 3, 1), (0, 2, 3), (0, 4, 2), (0, 1, 4), (1, 3, 2), (1, 2, 4))


def _signed(value: int, flags: int, mask: int) -> int:
    return -value if flags & mask else value


def _decode_model(data: bytes, base: int, name: str, *, vertex_test_default=False, hull_triangles=()) -> Model:
    nv, ne, nf = data[base + 3:base + 6]
    vo, eo, fo = unpack_from("<3H", data, base + 12)
    vertices = []
    for x, y, z, flags, ab, cd in iter_unpack("6B", data[base + vo:base + vo + nv * 6]):
        vertices.append(Vertex(_signed(x, flags, 128), _signed(y, flags, 64),
                               _signed(z, flags, 32), flags & 31,
                               (ab >> 4, ab & 15, cd >> 4, cd & 15)))
    edges = []
    for limit, a, b, faces in iter_unpack("4B", data[base + eo:base + eo + ne * 4]):
        edges.append(Edge(a, b, (faces >> 4, faces & 15), limit))
    normals = []
    for x, y, z, flags in iter_unpack("4B", data[base + fo:base + fo + nf * 4]):
        normals.append(Face(_signed(x, flags, 128), _signed(y, flags, 64),
                            _signed(z, flags, 32), flags & 31))
    # Начало первого ребра между разными гранями. Детали внутри грани
    # (например, прямоугольник дока) помечены одним и тем же номером дважды.
    face_vertices = tuple(next(edge.vertex_a for edge in edges
                               if edge.faces[0] != edge.faces[1] and i in edge.faces)
                          for i in range(nf))
    return Model(name, tuple(vertices), tuple(edges), tuple(normals), data[base + 19],
                 face_vertices, vertex_test_default, hull_triangles)


def _load_model(index: int, name: str) -> Model:
    address, = unpack_from("<H", BLUEPRINT_DATA, index * 4 + 1)
    return _decode_model(BLUEPRINT_DATA, address - BLUEPRINT_BASE, name,
                         hull_triangles=_KRAIT_HULL if index == KRAIT_ID else ())


# Геометрия Dodo взята из отдельной спектрумовской версии Elite Legend 1.09.
_dodo = _decode_model((_DATA_DIR / "dodo_elite_legend.bin").read_bytes(), 0,
                      "Dodo · Elite Legend 1.09", vertex_test_default=True)
MODELS = tuple(_load_model(index, name) for index, name in enumerate(_NAMES)) + (_dodo,)
COBRA = MODELS[COBRA_ID]


def load_cobra() -> Model:
    return COBRA


def initial_depth(model_id: int) -> int:
    """Начальная дистанция обзора, на которой виден весь каркас."""
    return 1152 if model_id in (0, 10, 11, DODO_ID) else 768  # Станции и обе версии Python.


def minimum_depth(model_id: int) -> int:
    """Ближняя остановка зума: степень двойки за описанной сферой каркаса."""
    vertices = MODELS[model_id].vertices
    radius_squared = max(v.x * v.x + v.y * v.y + v.z * v.z for v in vertices)
    return 1 << isqrt(radius_squared).bit_length()
