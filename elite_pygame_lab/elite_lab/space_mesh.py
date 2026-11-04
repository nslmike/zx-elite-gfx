"""Заполненные корпуса для отдельного OpenGL-атласа.

Это визуальная реконструкция по координатам и рёбрам blueprint, а не новый
вариант ZX-алгоритма видимости. Номера машинных граней помогают найти рёбра
контура; готовым многоугольником грань становится только после проверки
циклов, общей топологии, геометрии и направления обхода. Нормали Elite при
построении поверхности не используются.

Некоторые исходные четырёх- и пятиугольники слегка неплоские. Их вершины
сохранены; реальные нормали вычисляются для каждого треугольника. Внутренние
диагонали Python также сохранены. Кили ракеты, усы Krait и носовые стволы
получают небольшую толщину. Рисунок обшивки становится стеклом, соплами или
приподнятыми панелями, а не линиями, просвечивающими сквозь корпус.

Строка массива: position.xyz, normal.xyz, uv.xy, material. Материалы:
0 — обшивка, 1 — тёмный металл, 2 — стекло, 3 — сопло,
4 — окантовка, 5 — камень, 6 — светящееся окно.
Все треугольники имеют наружный CCW-обход: back-face culling допустим.
"""

from collections import defaultdict, deque
from functools import lru_cache
from math import cos, pi, sin

import numpy as np

from .model import COBRA_IDS, DODO_ID, KRAIT_ID, MODELS


_EPS = 1e-8


def _unit(vector):
    length = np.linalg.norm(vector)
    if length < _EPS:
        raise ValueError("Вырожденная геометрия корпуса")
    return vector / length


def _normal(points):
    """Нормаль Ньюэлла, пригодная и для слегка неплоских контуров."""
    return _unit(np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0))


def _basis(normal):
    # Одинаковая система UV на параллельных панелях и их окантовке.
    reference = np.array((0., 1., 0.) if abs(normal[1]) < .85 else (0., 0., -1.))
    u = _unit(np.cross(reference, normal))
    return u, np.cross(normal, u)


def _aperture_uv(points, normal):
    """Локальные 0..1 для одной светящейся апертуры, общие всем её треугольникам."""
    u, v = _basis(normal)
    projected = np.column_stack((points @ u, points @ v))
    span = np.ptp(projected, axis=0)
    if np.any(span < _EPS):
        raise ValueError("Вырожденная светящаяся апертура")
    return (projected - projected.min(axis=0)) / span


def _cycles(edges, *, strip_spurs=False):
    """Замкнутые контуры графа; незамкнутые отростки не становятся фасетами."""
    neighbours = defaultdict(set)
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    if strip_spurs:
        queue = deque(a for a, adjacent in neighbours.items() if len(adjacent) < 2)
        while queue:
            a = queue.popleft()
            if a not in neighbours:
                continue
            for b in neighbours.pop(a):
                neighbours[b].remove(a)
                if len(neighbours[b]) < 2:
                    queue.append(b)
    if any(len(adjacent) != 2 for adjacent in neighbours.values()):
        raise ValueError("Рёбра грани не образуют однозначный замкнутый контур")
    remaining = set(neighbours)
    result = []
    while remaining:
        first = min(remaining)
        loop, previous, current = [], None, first
        while True:
            loop.append(current)
            remaining.remove(current)
            following = next(b for b in sorted(neighbours[current]) if b != previous)
            if following == first:
                break
            previous, current = current, following
        if len(loop) < 3:
            raise ValueError("Контур короче трёх вершин")
        result.append(tuple(loop))
    return result


def _signed_volume(points, polygons):
    return sum(np.dot(points[loop[0]], np.cross(points[loop[i]], points[loop[i + 1]]))
               for _, loop in polygons for i in range(1, len(loop) - 1)) / 6


@lru_cache(maxsize=len(MODELS))
def _hull_polygons(model_id):
    """Проверенные замкнутые контуры, (номер машинной грани, индексы)."""
    model = MODELS[model_id]
    points = np.array([(v.x, v.y, v.z) for v in model.vertices], dtype=float)
    if model_id == 3:
        # У обломка нет ни одной нормали и есть лишь четырёхрёберный силуэт.
        # Четыре исходные точки задают очень тонкий, но ненулевой тетраэдр.
        polygons = [(i, loop) for i, loop in enumerate(((0, 1, 2), (0, 2, 3),
                                                       (0, 3, 1), (1, 3, 2)))]
    else:
        polygons = []
        for face_id in range(len(model.faces)):
            edges = [(e.vertex_a, e.vertex_b) for e in model.edges
                     if e.faces[0] != e.faces[1] and face_id in e.faces]
            cycles = _cycles(edges, strip_spurs=True)
            if len(cycles) != 1:
                raise ValueError(f"{model.name}: нет единственного контура грани {face_id}")
            polygons.append((face_id, cycles[0]))

    # Обход согласуется по смежности, а не по неточным исходным нормалям
    # или знаку скалярного произведения с центром невыпуклой Cobra.
    incidence = defaultdict(list)
    for face_index, (_, loop) in enumerate(polygons):
        for a, b in zip(loop, loop[1:] + loop[:1]):
            incidence[tuple(sorted((a, b)))].append((face_index, a < b))
    if any(len(uses) != 2 for uses in incidence.values()):
        raise ValueError(f"{model.name}: корпус не является замкнутой поверхностью")
    orientation = {0: False}
    queue = deque((0,))
    while queue:
        i = queue.popleft()
        loop = polygons[i][1]
        for a, b in zip(loop, loop[1:] + loop[:1]):
            uses = incidence[tuple(sorted((a, b)))]
            j, same_direction = next(use for use in uses if use[0] != i)
            reverse = orientation[i] ^ ((a < b) == same_direction)
            if j in orientation and orientation[j] != reverse:
                raise ValueError(f"{model.name}: несогласованная ориентация корпуса")
            if j not in orientation:
                orientation[j] = reverse
                queue.append(j)
    if len(orientation) != len(polygons):
        raise ValueError(f"{model.name}: несвязный корпус")
    polygons = [(face, loop[::-1] if orientation[i] else loop)
                for i, (face, loop) in enumerate(polygons)]
    if _signed_volume(points, polygons) < 0:
        polygons = [(face, loop[::-1]) for face, loop in polygons]
    if _signed_volume(points, polygons) < _EPS:
        raise ValueError(f"{model.name}: нулевой объём корпуса")
    used = {a for _, loop in polygons for a in loop}
    if len(used) - len(incidence) + len(polygons) != 2:
        raise ValueError(f"{model.name}: неверная эйлерова характеристика корпуса")
    return tuple(polygons)


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _triangulate(points):
    """Ear clipping в плоскости контура; сохраняет невыпуклые силуэты."""
    n = _normal(points)
    u, v = _basis(n)
    projected = np.column_stack((points @ u, points @ v))
    remaining = list(range(len(points)))
    result = []
    while len(remaining) > 3:
        candidates = []
        for i, b in enumerate(remaining):
            a, c = remaining[i - 1], remaining[(i + 1) % len(remaining)]
            pa, pb, pc = projected[[a, b, c]]
            if _cross2(pb - pa, pc - pb) <= _EPS:
                continue
            blocked = False
            for d in remaining:
                if d in (a, b, c):
                    continue
                pd = projected[d]
                # Кольцо с отверстием разворачивается по мосту в простой
                # контур. Два конца моста встречаются в списке дважды;
                # геометрически совпадающая вершина не блокирует ухо.
                if any(np.linalg.norm(pd - p) < _EPS for p in (pa, pb, pc)):
                    continue
                if min(_cross2(pb - pa, pd - pa), _cross2(pc - pb, pd - pb),
                       _cross2(pa - pc, pd - pc)) >= -_EPS:
                    blocked = True
                    break
            if not blocked:
                candidates.append((np.linalg.norm(points[c] - points[a]), i, (a, b, c)))
        if not candidates:
            raise ValueError("Самопересекающийся или вырожденный контур корпуса")
        _, i, triangle = min(candidates)
        result.append(triangle)
        remaining.pop(i)
    result.append(tuple(remaining))
    return result


def _facet_triangles(model, loop, points):
    """Внутренние blueprint-диагонали имеют приоритет перед ear clipping."""
    for edge in model.edges:
        a, b = edge.vertex_a, edge.vertex_b
        if edge.faces[0] != edge.faces[1] or a not in loop or b not in loop:
            continue
        i, j = sorted((loop.index(a), loop.index(b)))
        if j - i in (1, len(loop) - 1):
            continue
        first = loop[i:j + 1]
        second = loop[j:] + loop[:i + 1]
        return (_facet_triangles(model, first, points)
                + _facet_triangles(model, second, points))
    return [tuple(loop[i] for i in tri) for tri in _triangulate(points[list(loop)])]


class _Mesh:
    def __init__(self, source_points):
        self.rows = []
        self.uv_scale = 2 * np.max(np.linalg.norm(source_points - source_points.mean(axis=0), axis=1))

    def triangle(self, points, material=0, basis=None, uv=None):
        points = np.asarray(points, dtype=float)
        normal = _unit(np.cross(points[1] - points[0], points[2] - points[0]))
        u, v = _basis(normal) if basis is None else basis
        if uv is None:
            uv = np.column_stack((points @ u, points @ v)) / self.uv_scale + .5
        self.rows.extend(np.column_stack((points, np.tile(normal, (3, 1)), uv,
                                          np.full(3, material))))

    def polygon(self, points, material=0, basis=None):
        points = np.asarray(points, dtype=float)
        normal = _normal(points)
        basis = _basis(normal) if basis is None else basis
        # Каждый отдельный источник тяги получает самостоятельные UV.
        # У обшивки, стекла, рам и окон сохраняется прежний масштаб корпуса.
        aperture_uv = _aperture_uv(points, normal) if material == 3 else None
        for triangle in _triangulate(points):
            uv = None if aperture_uv is None else aperture_uv[list(triangle)]
            self.triangle(points[list(triangle)], material, basis, uv)

    def oriented(self, points, normal, material=0, basis=None):
        points = np.asarray(points, dtype=float)
        if np.dot(_normal(points), normal) < 0:
            points = points[::-1]
        self.polygon(points, material, basis)

    def ring(self, outer, inner, normal, material=0, basis=None):
        """Триангуляция двух выпуклых вложенных контуров с видимым мостом.

        Мост соединяет существующие вершины: передняя грань станции
        с отверстием остаётся сшита с соседними гранями без T-стыков.
        """
        outer, inner = np.asarray(outer), np.asarray(inner)
        if np.dot(_normal(outer), normal) < 0:
            outer = outer[::-1]
        if np.dot(_normal(inner), normal) < 0:
            inner = inner[::-1]
        u, v = _basis(normal) if basis is None else basis
        outer2 = np.column_stack((outer @ u, outer @ v))
        inner2 = np.column_stack((inner @ u, inner @ v))
        candidates = []
        for i, a in enumerate(outer2):
            for j, b in enumerate(inner2):
                direction = a - b
                incoming = b - inner2[j - 1]
                outgoing = inner2[(j + 1) % len(inner2)] - b
                # Из вершины отверстия мост сразу выходит наружу, а не
                # пересекает внутренность отверстия на пути к обшивке.
                if min(_cross2(incoming, direction), _cross2(outgoing, direction)) < -_EPS:
                    candidates.append((np.linalg.norm(outer[i] - inner[j]), i, j))
        if not candidates:
            raise ValueError("Не найден видимый мост для отверстия")
        _, i, j = min(candidates)
        outer = np.roll(outer, -i, axis=0)
        inner = np.roll(inner, -j, axis=0)
        joined = np.concatenate((outer, outer[:1], inner[:1], inner[:0:-1], inner[:1]))
        for tri in _triangulate(joined):
            self.triangle(joined[list(tri)], material, (u, v))

    def panel(self, points, normal, material, *, rise=.24, rim=True):
        """Закрытая тонкая панель с основанием внутри корпуса.

        Нулевое смещение не используется: видимые поверхности никогда не
        лежат поверх обшивки в той же плоскости. UV наследуют плоскость грани.
        """
        points = np.asarray(points, dtype=float)
        if np.dot(_normal(points), normal) < 0:
            points = points[::-1]
        basis = _basis(normal)
        bottom = points - normal * .5
        top = points + normal * rise
        self.oriented(bottom, -normal, 1, basis)
        if rim:
            inset = top.mean(axis=0) + (top - top.mean(axis=0)) * .86
            self.ring(top, inset, normal, 1, basis)
            self.oriented(inset, normal, material, basis)
        else:
            self.oriented(top, normal, material, basis)
        for i in range(len(points)):
            j = (i + 1) % len(points)
            self.polygon((bottom[i], bottom[j], top[j], top[i]), 1)

    def canopy(self, points, normal, height):
        """Низкий остеклённый фонарь: плечи, два стекла и несущая рама.

        Это собственное оформление атласа. Подошва входит в исходную
        обшивку, а каждая панель крыши сшита с рамой без накладных полигонов.
        """
        points = np.asarray(points, dtype=float)
        if np.dot(_normal(points), normal) < 0:
            points = points[::-1]
        # По середине передней и задней кромок проходит приподнятый
        # продольный переплёт. Эти вершины входят также в боковой контур.
        outline = np.array((points[0], (points[0]+points[1])*.5, points[1],
                            points[2], (points[2]+points[3])*.5, points[3]))
        bottom = outline - normal * .5
        shoulder = outline + normal * height * .18
        roof = _inset(outline, .79) + normal * height
        roof[[1, 4]] += normal * height * .18
        self.oriented(bottom, -normal, 1)
        for i in range(6):
            j = (i+1) % 6
            self.polygon((bottom[i], bottom[j], shoulder[j], shoulder[i]), 0)
            self.polygon((shoulder[i], shoulder[j], roof[j], roof[i]), 1)
        # Две половины имеют разные реальные нормали: стекло ловит блик
        # при облёте, а граница остаётся тонким металлическим переплётом.
        for indices in ((0, 1, 4, 5), (1, 2, 3, 4)):
            pane = roof[list(indices)]
            glass = _inset(pane, .9)
            self.ring(pane, glass, _normal(pane), 1)
            self.polygon(glass, 2)

    def engine(self, points, normal, height=1.6):
        """Короткий профилированный кожух вокруг исходного контура сопла."""
        points = np.asarray(points, dtype=float)
        if np.dot(_normal(points), normal) < 0:
            points = points[::-1]
        bottom = points - normal * .5
        lip = points + normal * height
        mouth = _inset(points, .82) + normal * height * .8
        throat = _inset(points, .70) + normal * .18
        self.oriented(bottom, -normal, 1)
        for i in range(len(points)):
            j = (i+1) % len(points)
            self.polygon((bottom[i], bottom[j], lip[j], lip[i]), 1)
            self.polygon((lip[i], lip[j], mouth[j], mouth[i]), 4)
            self.polygon((mouth[i], mouth[j], throat[j], throat[i]), 1)
        self.oriented(throat, normal, 3)

    def equipment(self, points, normal, height=1.1):
        """Небольшая приборная врезка со скошенным кожухом и тёмной крышкой."""
        bottom = points - normal * .5
        top = _inset(points, .85) + normal * height
        if np.dot(_normal(bottom), normal) < 0:
            bottom, top = bottom[::-1], top[::-1]
        self.oriented(bottom, -normal, 1)
        for i in range(len(points)):
            j = (i+1) % len(points)
            self.polygon((bottom[i], bottom[j], top[j], top[i]), 0)
        inset = _inset(top, .79)
        self.ring(top, inset, normal, 4)
        self.oriented(inset, normal, 1)

    def beam(self, a, b, radius, material=4, tip_radius=None):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        axis = _unit(b - a)
        u, v = _basis(axis)
        tip_radius = radius if tip_radius is None else tip_radius
        circles = [np.array([p + r * (cos(i * pi / 3) * u + sin(i * pi / 3) * v)
                             for i in range(6)]) for p, r in ((a, radius), (b, tip_radius))]
        self.oriented(circles[0], -axis, 1)
        self.oriented(circles[1], axis, material)
        for i in range(6):
            j = (i + 1) % 6
            self.polygon((circles[0][i], circles[0][j], circles[1][j], circles[1][i]), material)

    def finish(self):
        result = np.asarray(self.rows, dtype=np.float32)
        center = (result[:, :3].min(axis=0) + result[:, :3].max(axis=0)) * .5
        result[:, :3] -= center
        result[:, :3] /= np.linalg.norm(result[:, :3], axis=1).max()
        return np.ascontiguousarray(result)


def _project(points, face_points):
    """Рисунок исходной грани с её округлениями укладывается на поверхность."""
    normal = _normal(face_points)
    center = face_points.mean(axis=0)
    points = np.asarray(points, dtype=float)
    return points - np.outer((points - center) @ normal, normal), normal


def _inset(points, amount):
    return points.mean(axis=0) + (points - points.mean(axis=0)) * amount


def _dock(mesh, outer, aperture, depth, hull_material=0):
    """Настоящее отверстие, скошенный обод, внутренние стены и глухой торец."""
    normal = _normal(outer)
    if np.dot(_normal(aperture), normal) < 0:
        aperture = aperture[::-1]
    basis = _basis(normal)
    bevel = _inset(aperture, .86) - normal * depth * .12
    back = bevel - normal * depth * .88
    mesh.ring(outer, aperture, normal, hull_material, basis)
    for i in range(len(aperture)):
        j = (i + 1) % len(aperture)
        # Обход внутрь шахты: эти стены видны через вход снаружи станции.
        mesh.polygon((aperture[i], aperture[j], bevel[j], bevel[i]), 4)
        mesh.polygon((bevel[i], bevel[j], back[j], back[i]), 1)
    mesh.oriented(back, normal, 1, basis)
    # Свет в глубине отверстия. Небольшие панели приподняты над торцом.
    center = back.mean(axis=0)
    for sign in (-1, 1):
        p = center + sign * .35 * (back[1] - back[0])
        u, v = basis
        width = np.linalg.norm(back[1] - back[0]) * .06
        height = np.linalg.norm(back[2] - back[1]) * .1
        patch = np.array((p - width*u - height*v, p + width*u - height*v,
                          p + width*u + height*v, p - width*u + height*v))
        mesh.panel(patch, normal, 6, rise=.12, rim=False)


def _original_panels(mesh, model_id, points, hull):
    model = MODELS[model_id]
    used = {i for loop in hull.values() for i in loop}
    for face, loop in hull.items():
        edges = [(e.vertex_a, e.vertex_b) for e in model.edges if e.faces == (face, face)
                 and not (e.vertex_a in used and e.vertex_b in used)]
        for detail in _cycles(edges, strip_spurs=True):
            if model_id in (0, DODO_ID):
                continue  # Прямоугольник дока уже вырезан из передней грани.
            patch, normal = _project(points[list(detail)], points[list(loop)])
            material = 2 if model_id == 14 and face == 0 else 3
            if model_id == 14 and face == 9:
                material = 4
            if model_id in COBRA_IDS and len(detail) == 3:
                material = 1
            if material == 3:
                mesh.engine(patch, normal, height=2.1 if model_id in COBRA_IDS else 1.25)
            else:
                mesh.panel(patch, normal, material)


def _station_windows(mesh, front, normal, half_width, half_height):
    """Два коротких ряда окон возле входа задают масштаб станции."""
    u, v = _basis(normal)
    for side in (-1, 1):
        for row in range(-2, 3):
            center = front + u * side * half_width * 1.85 + v * row * half_height * .29
            width, height = half_width * .085, half_height * .045
            patch = np.array((center-width*u-height*v, center+width*u-height*v,
                              center+width*u+height*v, center-width*u+height*v))
            mesh.panel(patch, normal, 6, rise=.18)


def _extra_details(mesh, model_id, points, hull):
    def panel_on(face, indices=None, amount=.55, material=2):
        surface = points[list(hull[face])]
        patch = _inset(surface if indices is None else points[list(indices)], amount)
        patch, normal = _project(patch, surface)
        mesh.panel(patch, normal, material)

    def canopy_on(face, patch, height):
        patch, normal = _project(patch, points[list(hull[face])])
        mesh.canopy(patch, normal, height)

    def triangular_canopy(face, left, right, apex, low=.25, high=.72, height=2.5):
        a, b, c = points[[left, right, apex]]
        patch = np.array((a*(1-low)+c*low, b*(1-low)+c*low,
                          b*(1-high)+c*high, a*(1-high)+c*high))
        canopy_on(face, patch, height)

    if model_id == 1:
        for indices in ((5, 9, 15), (6, 10, 16), (8, 12, 13), (7, 11, 14)):
            patch = points[list(indices)]
            normal = _normal(patch)
            # Каждый киль — отдельная закрытая призма, а не двухсторонний
            # нулевой треугольник. Корень слегка заходит внутрь фюзеляжа.
            mesh.panel(patch - normal * .35, normal, 4, rise=.7, rim=False)
        panel_on(8, amount=.65, material=3)
    elif model_id == 2:
        panel_on(1, amount=.37, material=6)
    elif model_id == 8:
        triangular_canopy(0, 8, 7, 1, height=2.4)
    elif model_id in COBRA_IDS:
        triangular_canopy(0, 1, 0, 2, low=.26, high=.72, height=3.7)
        for face, sign in ((5, -1), (6, 1)):
            # По одной приборной кассете на верхних плечах. Контуры малы
            # относительно крыла и не меняют характерный силуэт Cobra.
            patch = np.array([(sign*x, 0., z) for x, z in ((18, -27), (36, -27),
                                                           (36, -12), (18, -12))])
            patch, normal = _project(patch, points[list(hull[face])])
            mesh.equipment(patch, normal)
        mesh.beam(points[20], points[21], .85, 1, tip_radius=.55)
    elif model_id in (10, 11):
        panel_on(0, amount=.27)
        panel_on(1, amount=.27)
        panel_on(12, amount=.58, material=3)
    elif model_id == 13:
        canopy_on(4, _inset(points[list(hull[4])], .45), height=2.7)
        mesh.beam(points[17], points[18], .55, 1, tip_radius=.35)
    elif model_id == 15:
        # Две параллельные полосы на большой плоскости — характерный рисунок
        # исходного Thargoid. Ставим их на поверхность, с толщиной и глубиной.
        for a, b in ((16, 17), (18, 19)):
            normal = _normal(points[list(hull[9])])
            direction = _unit(points[b] - points[a])
            side = _unit(np.cross(normal, direction)) * 1.8
            patch = np.array((points[a]-side, points[b]-side, points[b]+side, points[a]+side))
            mesh.panel(patch, normal, 6, rise=.3)
        panel_on(4, amount=.56, material=1)
    elif model_id == 16:
        triangular_canopy(0, 0, 1, 4, low=.28, high=.75, height=2.1)
    elif model_id == KRAIT_ID:
        for face, indices in ((0, (7, 10, 8)), (3, (7, 8, 9))):
            panel_on(face, indices, amount=.95)
        for a, b in ((3, 5), (4, 6)):
            mesh.beam(points[a] - (points[b]-points[a]) * .012, points[b], 1.05,
                      4, tip_radius=.52)
    elif model_id == 18:
        panel_on(6, amount=.52, material=6)


@lru_cache(maxsize=len(MODELS))
def _build_cached(model_id):
    model = MODELS[model_id]
    points = np.array([(v.x, v.y, v.z) for v in model.vertices], dtype=float)
    polygons = _hull_polygons(model_id)
    hull = dict(polygons)
    mesh = _Mesh(points)
    for face, loop in polygons:
        facet = points[list(loop)]
        if model_id in (0, DODO_ID) and face == 0:
            indices = (12, 13, 14, 15) if model_id == 0 else (20, 21, 23, 22)
            aperture, _ = _project(points[list(indices)], facet)
            _dock(mesh, facet, aperture, 34 if model_id == 0 else 42)
            _station_windows(mesh, aperture.mean(axis=0), _normal(facet),
                             10 if model_id == 0 else 16, 30 if model_id == 0 else 32)
            continue
        if model_id == 7 and face == 0:
            # Убежище использует тот же каркас, что и астероид. Небольшая
            # врезанная шахта делает назначение модели читаемым в 3D-атласе.
            normal = _normal(facet)
            u, v = _basis(normal)
            center = facet.mean(axis=0)
            r = min(np.linalg.norm(np.cross(b-a, center-a)) / np.linalg.norm(b-a)
                    for a, b in zip(facet, np.roll(facet, -1, axis=0)))
            aperture = np.array([center + r * (x*u + y*v)
                                 for x, y in ((-.33, -.45), (.33, -.45), (.33, .45), (-.33, .45))])
            _dock(mesh, facet, aperture, 11, hull_material=5)
            continue
        material = 5 if model_id in (3, 5, 6, 7) else 0
        if model_id == 4 and face in (0, 6):
            material = 1
        basis = _basis(_normal(facet))
        for triangle in _facet_triangles(model, loop, points):
            mesh.triangle(points[list(triangle)], material, basis)
    _original_panels(mesh, model_id, points, hull)
    _extra_details(mesh, model_id, points, hull)
    result = mesh.finish()
    result.flags.writeable = False
    return result


def build_mesh(model_id: int) -> np.ndarray:
    """Модель в сфере радиуса 1, +Y вверх, +Z — исходный нос Elite.

    Возвращает независимый C-contiguous float32-массив (N, 9), по три строки
    на треугольник. Копия позволяет вызывающему коду безопасно менять буфер.
    UV материала 3 локальны для каждой отдельной апертуры (0..1); у других
    материалов они согласованы в плоскостях граней в масштабе корпуса.
    Для 0..19 результат детерминирован; неизвестный ID вызывает ValueError.
    """
    if isinstance(model_id, bool) or not isinstance(model_id, (int, np.integer)) or not 0 <= model_id < len(MODELS):
        raise ValueError(f"Неизвестный номер модели: {model_id!r}")
    return _build_cached(int(model_id)).copy()
