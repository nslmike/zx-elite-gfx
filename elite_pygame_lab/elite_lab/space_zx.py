"""Чёрно-белая сцена 256×128 с табличными рёбрами Spectrum Elite.

Камера и положение кораблей общие с цветным пролётом. Здесь адаптируются
нормализация выставочных моделей, многокорабельная перспектива и отсечение
у камеры; это не побитовая копия игровой камеры. Проверка граней и дальности
использует исходные таблицы, а пиксели рёбер пишет настоящий ZX trace_line.
Невидимая оболочка нужна только для перекрытий *между* объектами и звёздами:
она никогда не добавляет рёбер и не заменяет исходную проверку граней.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .arithmetic import SignMag
from .model import MODELS
from .raster import apply_trace, spectrum_address, trace_line
from .renderer import classify_faces, select_edges
from .space_mesh import _hull_polygons, _triangulate, source_transform
from .types import Pose, ProjectedVertex


ZX_SIZE = (256, 128)
_NEAR = .06
_GRID_Y, _GRID_X = np.mgrid[:128, :256]


@lru_cache(maxsize=4096)
def _pixel_trace(start, end):
    """Неизменяемая настоящая трасса повторно используется между кадрами.

    Дальние рёбра часто сохраняют одни и те же целые концы несколько кадров.
    Маска чужих корпусов применяется заново и в этот кеш не входит.
    """
    return trace_line(start, end)


def fit_viewport(size):
    """Прямоугольник GL: квадратные пиксели, игровое поле ровно 2:1."""
    width, height = (max(1, int(n)) for n in size)
    scale = min(width / ZX_SIZE[0], height / ZX_SIZE[1])
    # Чётная ширина сохраняет 2:1 даже при нечётном размере окна.
    h = max(1, int(ZX_SIZE[1] * scale))
    w = 2 * h
    if w > width:
        w, h = width, max(1, width // 2)
    return (width - w) // 2, (height - h) // 2, w, h


def viewport_projection(projection, size, viewport):
    """Сохранить экранные размеры и центры при M, обрезав только поля.

    Одинаковые множители X/Y после увеличения bitmap сохраняют пропорции.
    Вертикальные поля окна не растягивают настоящий игровой bitmap 256×128.
    """
    result = projection.copy()
    result[0] *= size[0] / viewport[2]
    result[1] *= size[1] / viewport[3]
    return result


@dataclass(frozen=True)
class ZXBlueprint:
    points: np.ndarray
    center: np.ndarray
    scale: float
    triangles: np.ndarray


@lru_cache(maxsize=len(MODELS))
def blueprint(model_id):
    """Те же центр и масштаб, что у классического цветного корпуса.

    space_mesh центрирует также добавленные стволы и фонари. Используем его
    сохранённое преобразование, сами дополнительные детали в ZX не рисуются.
    """
    model = MODELS[model_id]
    raw = np.array([(v.x, v.y, v.z) for v in model.vertices], dtype=float)
    polygons = _hull_polygons(model_id)
    center, scale = source_transform(model_id)
    points = (raw - center) / scale
    triangles = np.array([tuple(loop[i] for i in tri) for _, loop in polygons
                          for tri in _triangulate(raw[list(loop)])], dtype=np.intp)
    points.setflags(write=False)
    center.setflags(write=False)
    triangles.setflags(write=False)
    return ZXBlueprint(points, center, scale, triangles)


def _near_clip(points):
    """Sutherland–Hodgman в camera-space; Z положителен перед камерой."""
    result = []
    previous = points[-1]
    previous_inside = previous[2] >= _NEAR
    for point in points:
        inside = point[2] >= _NEAR
        if inside != previous_inside:
            t = (_NEAR - previous[2]) / (point[2] - previous[2])
            result.append(previous + t * (point - previous))
        if inside:
            result.append(point)
        previous, previous_inside = point, inside
    return result


def _project(points, focal):
    points = np.asarray(points)
    inverse_z = 1 / points[:, 2]
    xy = points[:, :2] * inverse_z[:, None] * (focal, -focal) + (128, 64)
    return np.column_stack((xy, inverse_z))


def _screen_line(a, b):
    """Ограничить расширенный экран байтовым диапазоном ZX-клиппера.

    Сам клиппер $B044 затем отсекает реальное игровое поле. Предварительная
    вещественная обрезка нужна только перспективе новой широкой камеры:
    координата далеко за кадром не должна переполнить исходный один байт.
    Третья компонента — 1/Z, она линейна вдоль экранного отрезка.
    """
    start, end = 0., 1.
    delta = b - a
    for axis, lo, hi in ((0, -127., 381.), (1, -191., 317.)):
        if abs(delta[axis]) < 1e-12:
            if not lo <= a[axis] <= hi:
                return None
            continue
        t0, t1 = sorted(((lo - a[axis]) / delta[axis], (hi - a[axis]) / delta[axis]))
        start, end = max(start, t0), min(end, t1)
        if start > end:
            return None
    return a + delta * start, a + delta * end


def _depth_triangle(depth, owner, triangle, ident):
    """Оболочка не видна: только глубина для чужих линий и звёзд."""
    x0, y0 = np.maximum(np.floor(triangle[:, :2].min(axis=0)), (0, 0)).astype(int)
    x1, y1 = np.minimum(np.ceil(triangle[:, :2].max(axis=0)), (255, 127)).astype(int)
    if x0 > x1 or y0 > y1:
        return
    a, b, c = triangle
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-10:
        return
    y = _GRID_Y[y0:y1 + 1, x0:x1 + 1]
    x = _GRID_X[y0:y1 + 1, x0:x1 + 1]
    # В том же центре пикселя, который пишет trace_line: не MSAA и не фильтр.
    u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    w = 1 - u - v
    inv_z = u * a[2] + v * b[2] + w * c[2]
    patch = depth[y0:y1 + 1, x0:x1 + 1]
    covered = (u >= -1e-7) & (v >= -1e-7) & (w >= -1e-7) & (inv_z > patch)
    patch[covered] = inv_z[covered]
    owner[y0:y1 + 1, x0:x1 + 1][covered] = ident


def _front_triangles(triangles):
    """Отсечь обратную сторону замкнутой оболочки перед depth-проходом.

    В camera-space +Z вперёд и экранном Y вниз наружные контуры space_mesh
    имеют отрицательную экранную площадь. Этот тест относится только к маске
    чужого корпуса; видимые табличные рёбра выбирает classify_faces отдельно.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return triangles[area < -1e-10]


@dataclass(frozen=True)
class ZXFrame:
    pixels: np.ndarray
    bitmap: bytes
    viewport: tuple[int, int, int, int]
    visible_ships: tuple[int, ...]
    edge_count: int


class SpaceZXRenderer:
    """CPU bitmap; GPU требуется только для увеличения готовых пикселей."""

    def __init__(self):
        rng = np.random.default_rng(1984)
        stars = rng.normal(size=(1024, 3))
        self.stars = stars / np.linalg.norm(stars, axis=1)[:, None]
        # $C000..$CFFF → последовательные строки обычного изображения.
        y = np.arange(128)
        self._rows = ((y & 0x40) << 5) | ((y & 7) << 8) | ((y & 0x38) << 2)

    def frame(self, scene, size, projection=None):
        viewport = fit_viewport(size)
        vp = scene.view_projection(size[0] / size[1]) if projection is None else projection
        camera = np.array((vp[0, :3], vp[1, :3], vp[3, :3]))
        camera /= np.linalg.norm(camera, axis=1)[:, None]
        adjusted = viewport_projection(vp, size, viewport)
        # После вписывания физический масштаб обеих осей совпадает.
        focal = 64 * np.linalg.norm(adjusted[1, :3])
        eye = scene.eye
        depth = np.zeros((128, 256), dtype=float)
        owner = np.full((128, 256), -1, dtype=np.int16)
        lines, visible = [], []
        for ident, (ship, obj) in enumerate(zip(scene.ships, scene.objects)):
            if not 0 <= ship.model_id < len(MODELS):
                continue
            data = blueprint(ship.model_id)
            model = MODELS[ship.model_id]
            rotation_scale = camera @ obj.matrix[:3, :3]
            position = camera @ (obj.matrix[:3, 3] - eye)
            points = data.points @ rotation_scale.T + position
            if points[:, 2].max() < _NEAR:
                continue
            # Обрезаем по сфере до небольших массивов треугольников и трасс.
            radius = float(np.linalg.norm(rotation_scale, axis=0).max())
            if position[2] > radius + _NEAR:
                margin = radius * focal * (1 + np.abs(position[:2]) / position[2]) / (position[2] - radius)
                center = _project(position[None, :], focal)[0, :2]
                if np.any(center + margin < (0, 0)) or np.any(center - margin > (255, 127)):
                    continue
            visible.append(ident)
            fully_front = bool(np.all(points[:, 2] >= _NEAR))
            screen_points = _project(points, focal) if fully_front else None
            if fully_front:
                for screen in _front_triangles(screen_points[data.triangles]):
                    _depth_triangle(depth, owner, screen, ident)
            else:
                for indices in data.triangles:
                    polygon = _near_clip(points[list(indices)])
                    if len(polygon) < 3:
                        continue
                    screen = _project(polygon, focal)
                    triangles = np.array([screen[[0, i, i + 1]] for i in range(1, len(screen) - 1)])
                    for triangle in _front_triangles(triangles):
                        _depth_triangle(depth, owner, triangle, ident)

            scale = float(np.linalg.norm(obj.matrix[:3, 0]))
            rotation = rotation_scale / scale
            # Исходная позиция относительно *ненормализованного* центра.
            origin = position * (data.scale / scale) - rotation @ data.center
            pose = Pose(position=tuple(max(-32767, min(32767, round(v))) for v in origin))
            basis = tuple(tuple(SignMag.from_int(max(-127, min(127, round(c * 127))))
                                for c in row) for row in rotation)
            distance = max(0, min(31, max(0, pose.position[2]) >> 7) - 2)
            flags, _ = classify_faces(model, pose, basis, distance, use_face_vertices=False)
            # select_edges сохраняет исходный отбор вершин, граней и рёбер.
            # Камерную плоскость обрезаем ниже: в новой сцене ребро может её
            # пересекать, хотя центр корабля находится сбоку или позади.
            projected = [ProjectedVertex(0, 0, 0)
                         if (v.distance_limit == 31 or distance < v.distance_limit)
                         and any(flags[f] for f in v.faces) else None for v in model.vertices]
            for decision in select_edges(model, distance, flags, projected):
                if not decision.draw:
                    continue
                edge = model.edges[decision.index]
                if fully_front:
                    a, b = screen_points[[edge.vertex_a, edge.vertex_b]]
                else:
                    a, b = points[[edge.vertex_a, edge.vertex_b]].copy()
                    if a[2] < _NEAR and b[2] < _NEAR:
                        continue
                    if a[2] < _NEAR:
                        a += (_NEAR - a[2]) / (b[2] - a[2]) * (b - a)
                    elif b[2] < _NEAR:
                        b += (_NEAR - b[2]) / (a[2] - b[2]) * (a - b)
                    a, b = _project((a, b), focal)
                line = _screen_line(a, b)
                if line is not None:
                    lines.append((ident, *line))

        bitmap = bytearray(4096)
        stars = self.stars @ camera.T
        stars = stars[stars[:, 2] > .01]
        star_pixels = np.rint(_project(stars, focal)[:, :2]).astype(int)
        star_pixels = star_pixels[(star_pixels >= (0, 0)).all(axis=1)
                                  & (star_pixels < (256, 128)).all(axis=1)]
        for x, y in star_pixels:
            if owner[y, x] < 0:
                address, mask = spectrum_address(int(x), int(y))
                bitmap[address - 0xC000] |= mask

        for ident, a, b in lines:
            start = ProjectedVertex.from_screen(*np.rint(a[:2]).astype(int))
            end = ProjectedVertex.from_screen(*np.rint(b[:2]).astype(int))
            trace = _pixel_trace(start, end)
            # Применяем саму ручную запись, включая её исключённый последний
            # пиксель и исходное приближённое отсечение. Чужая оболочка лишь
            # исключает закрытые шаги; свой корпус не исправляет ZX-тест.
            direction = b[:2] - a[:2]
            squared = max(float(direction @ direction), 1e-12)
            steps = tuple(step for step in trace.steps if step.in_bitmap)
            if not steps:
                continue
            coordinates = np.array([(step.x, step.y) for step in steps])
            x, y = coordinates.T
            t = np.clip((coordinates - a[:2]) @ direction / squared, 0., 1.)
            inv_z = a[2] + t * (b[2] - a[2])
            mask = (owner[y, x] == ident) | (inv_z >= depth[y, x] * (1 - 1e-5))
            allowed = tuple(step for step, keep in zip(steps, mask) if keep)
            if len(allowed) != len(trace.steps):
                # LineTrace неизменяем: оставляем полный настоящий шаг для
                # каждого разрешённого пикселя, меняется только маска сцены.
                trace = replace(trace, steps=allowed)
            apply_trace(bitmap, trace)

        packed = np.frombuffer(bitmap, dtype=np.uint8)
        rows = packed[self._rows[:, None] + np.arange(32)]
        pixels = np.unpackbits(rows, axis=1) * np.uint8(255)
        return ZXFrame(pixels, bytes(bitmap), viewport, tuple(visible), len(lines))
