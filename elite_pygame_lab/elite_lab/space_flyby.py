"""Непрерывный постановочный облёт: отдельная сцена, без арифметики ZX.

Каждый экспонат занимает шесть секунд, полный круг — 126 секунд. Камера
и вращения вычисляются из абсолютного времени, поэтому ``seek(t)`` даёт
тот же кадр, что и обновления от начала показа до t. Ручной выбор набора
сбрасывает только таймер следующей смены, а не положение камеры.
"""

from math import cos, floor, isfinite, pi, sin, sqrt, tau

import numpy as np

from .model import COBRA_ID, DODO_ID, MODELS
from .space_scene import SpaceObject, SpaceScene, _unit


def _rotation(axis, angle):
    """Обычная матрица Родрига; локальная ось не зависит от ракурса камеры."""
    x, y, z = _unit(axis)
    c, s, t = cos(angle), sin(angle), 1 - cos(angle)
    return np.array(((t*x*x+c, t*x*y-s*z, t*x*z+s*y),
                     (t*x*y+s*z, t*y*y+c, t*y*z-s*x),
                     (t*x*z-s*y, t*y*z+s*x, t*z*z+c)))


def _station_dock(ship):
    """Центр настоящего входного контура в нормализованных координатах.

    У построенного корпуса материал 4 — обод входа. В импортных станциях
    Oolite последний материал принадлежит отдельной сетке шахты. Берём
    её передний срез, а не среднее вершин и не центр всего корпуса: у Dodo
    Oolite ось входа смещена по Y примерно на 0.005729 радиуса.
    Все эти исходные сетки ориентированы входом в локальный +Z.
    """
    mesh = ship.vertices()
    slot = len(ship.materials) - 1 if ship.meshes else 4
    points = mesh[mesh[:, 8] == slot, :3]
    front = points[np.abs(points[:, 2] - points[:, 2].max()) < 1e-5]
    return (front.min(axis=0) + front.max(axis=0)).astype(float) * .5


class FlybyScene(SpaceScene):
    """Плавный близкий проход по всем корпусам с независимыми вращениями.

    Основные двадцать мест сохраняются при любой смене геометрии. Гостевое
    место Pioneer лежит между двумя станциями и чуть внутри кольца: когда
    гостя нет, камера видит обе станции, не пустой промежуток каталога.
    ``seek`` воспроизводит исходный запуск, до ручных M/A; текстуры сохраняет.
    """

    seconds_per_ship = 6.0
    slot_count = 21
    duration = seconds_per_ship * slot_count
    default_source_interval = 8.0
    _route = (*range(COBRA_ID, len(MODELS)), len(MODELS), *range(COBRA_ID))
    # Последняя точка замыкает круг. Два полушага освобождают гостевое место.
    _angle_knots = np.array((*range(11), 10.5, *range(11, 21)), dtype=float)

    def __init__(self, selected=None, *, model_source="elite", texture_source="author",
                 catalog=None, source_interval=8.0):
        interval = float(source_interval)
        if not isfinite(interval) or interval < 0:
            raise ValueError("Интервал смены источников должен быть конечным и неотрицательным")
        self.elapsed = 0.0
        self.source_interval = interval
        self.auto_sources = interval > 0
        self._initial_source = model_source
        self._initial_interval = interval
        self._source_epoch = 0.0
        self._layouts = {}
        self._start_slot = self._route.index(COBRA_ID if selected is None else int(selected) % self.slot_count)
        super().__init__(COBRA_ID if selected is None else selected, model_source=model_source,
                         texture_source=texture_source, catalog=catalog)
        self.camera_mode = "flyby"
        self._animate()

    def _load_objects(self, ships):
        # Атласу нужны точные облака для inspect.fit. Постановочному пролёту
        # хватает нормализованных сфер: не сортируем hi-res Pioneer np.unique.
        bases, docks, angles = [], [], []
        for index, ship in enumerate(ships):
            angle = (index - COBRA_ID) * tau / len(MODELS)
            if index == len(MODELS):
                angle = (DODO_ID + .5 - COBRA_ID) * tau / len(MODELS)
            radial = self.ring_radius - (4.7 if index >= len(MODELS) else 0.0)
            position = radial * np.array((sin(angle), 0.0, cos(angle)))
            side = .78
            if index in (0, DODO_ID):
                side = .16
            elif index in (1, 10, 11):
                side = 1.0
            elif index in (15, 18):
                side = 1.2
            if index in (COBRA_ID, 10, 12, 14, 16):
                side = pi - .85  # Задние три четверти: сопла читаются в движении.
            elif index == len(MODELS):
                side = 1.18
            rotation = _rotation((0, 1, 0), angle + pi - side)
            if index in (COBRA_ID, 10, 12):
                rotation = rotation @ _rotation((1, 0, 0), -.14)
            if index == 3 and not ship.meshes:
                # У тонкого обломка широкая плоскость обращена к камере.
                source = np.array([(v.x, v.y, v.z) for v in MODELS[index].vertices], dtype=float)
                _, _, axes = np.linalg.svd(source - source.mean(axis=0))
                source_basis = np.column_stack((axes[0], np.cross(axes[2], axes[0]), axes[2]))
                normal = _unit(-position)
                right = _unit(np.cross((0, 1, 0), normal))
                axis = _unit(right * .86 + np.array((0, .5, 0)))
                rotation = np.column_stack((axis, np.cross(normal, axis), normal)) @ source_basis.T
            model = MODELS[index] if index < len(MODELS) else None
            source_radius = sqrt(max(v.x*v.x + v.y*v.y + v.z*v.z for v in model.vertices)) if model else 100.0
            radius = 1.7 * sqrt(source_radius / 100.0)
            if index == len(MODELS):
                radius = 3.0  # Выставочный масштаб гостя, не масштаб исходного мира.
            matrix = np.eye(4)
            matrix[:3, :3] = rotation * radius
            matrix[:3, 3] = position
            position.setflags(write=False)
            matrix.setflags(write=False)
            bases.append(SpaceObject(index, position, radius, matrix))
            docks.append(_station_dock(ship) if index in (0, DODO_ID) else None)
            angles.append((pi - angle, -.28))
        self.ships = ships
        self._base_objects = tuple(bases)
        self._docks = tuple(docks)
        self.objects = self._base_objects
        self._inspection_angles = angles
        # Эти списки не используются в flyby, но сохраняют внутренний контракт.
        self._body_points = self._model_points = []

    def _apply_source(self, source):
        if source == self.model_source:
            return
        if self.model_source not in self._layouts:
            self._layouts[self.model_source] = (self.ships, self._base_objects, self._docks,
                                                 self._inspection_angles)
        if source in self._layouts:
            self.ships, self._base_objects, self._docks, self._inspection_angles = self._layouts[source]
        else:
            self._load_objects(self.catalog.fleet(source))
            self._layouts[source] = (self.ships, self._base_objects, self._docks,
                                     self._inspection_angles)
        self.model_source = source
        self._overview_cache = self._inspect_cache = None

    def set_source(self, source):
        self._apply_source(source)
        self._source_epoch = self.elapsed
        self._animate()

    def toggle_auto_sources(self):
        self.auto_sources = not self.auto_sources
        if self.auto_sources and self.source_interval == 0:
            self.source_interval = self.default_source_interval
        self._source_epoch = self.elapsed
        return self.auto_sources

    def _advance_sources(self):
        if not self.auto_sources or self.source_interval <= 0:
            return
        delta = self.elapsed - self._source_epoch
        if delta <= 0:
            return
        # Конечные время/интервал могут иметь бесконечное float-отношение.
        # Их точные двоичные дроби дают целое число смен без переполнения;
        # большой integer дальше используется только по модулю числа наборов.
        numerator, denominator = delta.as_integer_ratio()
        interval_numerator, interval_denominator = self.source_interval.as_integer_ratio()
        changes, remainder_numerator = divmod(numerator * interval_denominator,
                                              denominator * interval_numerator)
        remainder = remainder_numerator / (denominator * interval_denominator)
        # Оставляем прежний допуск накопления кадров для обычных интервалов,
        # но ограничиваем его миллиардной долей малого интервала.
        tolerance = min(1e-9, self.source_interval * 1e-9)
        if remainder > 0 and self.source_interval - remainder <= tolerance:
            changes += 1
            remainder -= self.source_interval
        if changes <= 0:
            return
        options = self.catalog.sources
        source = options[(options.index(self.model_source) + changes) % len(options)]
        self._apply_source(source)
        self._source_epoch = self.elapsed - remainder

    def seek(self, seconds):
        seconds = float(seconds)
        if not isfinite(seconds) or seconds < 0:
            raise ValueError("Время облёта должно быть конечным и неотрицательным")
        self.elapsed = seconds
        self.source_interval = self._initial_interval
        self.auto_sources = self.source_interval > 0
        self._source_epoch = 0.0
        self._apply_source(self._initial_source)
        self._advance_sources()
        self._animate()

    def update(self, dt, movement=(0.0, 0.0, 0.0)):
        # Сценарий не зависит от частоты кадров и свободных осей WASD.
        dt = float(dt)
        if not isfinite(dt):
            raise ValueError("Шаг времени должен быть конечным")
        self.elapsed += max(0.0, dt)
        self._advance_sources()
        self._animate()

    def _path_angle(self, phase):
        phase %= self.slot_count
        # У корпуса скорость ниже, между корпусами выше; нулевых остановок нет.
        phase -= .42 * sin(tau * phase) / tau
        index = int(floor(phase))
        u = phase - index
        knots = self._angle_knots
        steps = np.diff(knots)
        previous, current, following = steps[(index - 1) % self.slot_count], steps[index], steps[(index + 1) % self.slot_count]
        # Гармонические касательные монотонного Hermite: полушаг к гостю
        # не создаёт ни разворота, ни излома скорости на входе/выходе дуги.
        m0 = 2 * previous * current / (previous + current)
        m1 = 2 * current * following / (current + following)
        value = ((2*u**3 - 3*u*u + 1) * knots[index]
                 + (u**3 - 2*u*u + u) * m0
                 + (-2*u**3 + 3*u*u) * knots[index + 1]
                 + (u**3 - u*u) * m1)
        return value * tau / len(MODELS)

    def _animate(self):
        phase = self._start_slot + self.elapsed / self.seconds_per_ship
        angle = self._path_angle(phase)
        height = 1.55 * cos(angle - .2) + .45 * sin(2 * angle + .4)
        slot = phase % self.slot_count
        station_weight = min(1.0, max(0.0, min(slot - 9, 13 - slot)))
        station_weight = station_weight**3 * (10 - 15*station_weight + 6*station_weight**2)
        # Крупный корпус в центре встречи; между встречами остаётся виден
        # соседний. Перед станциями дуга заранее и плавно отступает внутрь.
        gap = ((5.15 + 1.7 * sin(pi * phase)**2) * (1 - station_weight)
               + 8.3 * station_weight)
        guest_weight = max(0.0, 1 - abs(slot - 11))
        guest_weight = guest_weight**3 * (10 - 15*guest_weight + 6*guest_weight**2)
        gap += 5.2 * guest_weight
        camera_radius = self.ring_radius - gap
        # Внутренняя траектория всегда оставляет до корпуса несколько радиусов.
        camera_angle = angle - .065
        self._camera_eye = np.array((camera_radius * sin(camera_angle), height,
                                     camera_radius * cos(camera_angle)))
        self._camera_target = self.ring_radius * np.array((sin(angle), 0.0, cos(angle)))
        focus = self._route[int(floor(phase + .5)) % self.slot_count]
        if focus >= len(self.ships):
            focus = DODO_ID if phase % self.slot_count <= 11 else 0
        self.selected = focus
        objects = []
        for obj, dock in zip(self._base_objects, self._docks):
            index = obj.model_id
            time = self.elapsed
            if dock is not None:
                rotation = _rotation((0, 0, 1), time * (.20 if index == 0 else -.17))
            elif index in (3, 5, 6, 7):
                rotation = _rotation((.35 + index*.03, .8, .45), time * (.11 + index*.008))
            elif index == 4:
                rotation = _rotation((.2, 1, .3), time * .16)
            else:
                rotation = (_rotation((0, 1, 0), .21 * sin(time * .19 + index*.63))
                            @ _rotation((1, 0, 0), .105 * sin(time * .23 + index*.81))
                            @ _rotation((0, 0, 1), .14 * sin(time * .17 + index*.49)))
            matrix = obj.matrix.copy()
            matrix[:3, :3] = obj.matrix[:3, :3] @ rotation
            if dock is not None:
                # Вращаем вокруг прямой через вход, а не вокруг центра bbox.
                matrix[:3, 3] += obj.matrix[:3, :3] @ (dock - rotation @ dock)
            position = matrix[:3, 3].copy()
            position.setflags(write=False)
            matrix.setflags(write=False)
            objects.append(SpaceObject(index, position, obj.radius, matrix))
        self.objects = tuple(objects)

    @property
    def eye(self):
        return self._camera_eye.copy()

    @property
    def target(self):
        return self._camera_target.copy()

    @property
    def forward(self):
        return _unit(self._camera_target - self._camera_eye)

    @property
    def distance(self):
        return float(np.linalg.norm(self._camera_target - self._camera_eye))
