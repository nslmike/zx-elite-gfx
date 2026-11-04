"""Непрерывный постановочный облёт: отдельная сцена, без арифметики ZX.

Каждый экспонат занимает шесть секунд, полный круг — 126 секунд. Камера
и вращения вычисляются из отдельных часов: на паузе экспонаты вращаются,
но камера и таймер наборов стоят. ``seek(t)`` воспроизводит исходный запуск,
а ``scrub(dt)`` перематывает текущий показ, сохраняя ручные настройки.
"""

from math import cos, floor, isfinite, pi, sin, sqrt, tau
from sys import float_info

import numpy as np

from .model import COBRA_ID, DODO_ID, MODELS
from .space_scene import SpaceObject, SpaceScene, _rotation, _unit


def _smootherstep(value):
    """Плавный переход 0…1 с нулевыми первой и второй производными на краях."""
    value = min(1.0, max(0.0, value))
    return value**3 * (10 - 15*value + 6*value**2)


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

    Двадцать одно равномерное место сохраняется при любой смене геометрии.
    Гостевое место Pioneer лежит между Coriolis и ракетой на общей окружности;
    без гостя оно остаётся пустым и не меняет путь камеры.
    ``seek`` воспроизводит исходный запуск, до ручных M/A; текстуры сохраняет.
    """

    is_flyby = True
    seconds_per_ship = 6.0
    slot_count = 21
    duration = seconds_per_ship * slot_count
    default_source_interval = 8.0
    # Из внешней точки близкий экспонат и дальняя дуга должны помещаться
    # одновременно. Поле зрения атласа SpaceScene остаётся прежним.
    fov = 82.0
    _route = (*range(COBRA_ID, len(MODELS)), 0, len(MODELS), *range(1, COBRA_ID))
    # Спокойные полные обороты, с индивидуальными скоростями и направлениями.
    _roll_rates = tuple((-1 if index % 3 == 1 else 1) * (.12 + .007 * (index % 7))
                        for index in range(slot_count))
    _exhibit_radii = np.array([
        1.7 * sqrt(sqrt(max(v.x*v.x + v.y*v.y + v.z*v.z for v in model.vertices)) / 100.0)
        for model in MODELS
    ] + [3.0])

    @staticmethod
    def _ring_height(angle):
        # Небольшая волна сохраняет узнаваемую окружность в плане, но разводит
        # соседние силуэты дальней дуги по высоте вместо единственной полосы.
        return .9 * sin(3 * angle + .5) + .5 * sin(2 * angle - .6)

    def __init__(self, selected=None, *, model_source="zx", texture_source="author",
                 catalog=None, source_interval=8.0):
        interval = float(source_interval)
        if not isfinite(interval) or interval < 0:
            raise ValueError("Интервал смены источников должен быть конечным и неотрицательным")
        self.elapsed = 0.0
        self.animation_elapsed = 0.0
        self.paused = False
        self.source_interval = interval
        self.auto_sources = interval > 0
        self._initial_source = model_source
        self._initial_interval = interval
        self._source_epoch = 0.0
        self._source_anchor = model_source
        self._layouts = {}
        self._look_yaw = self._look_pitch = 0.0
        self._camera_up = np.array((0.0, 1.0, 0.0))
        self._start_slot = self._route.index(COBRA_ID if selected is None else int(selected) % self.slot_count)
        # ZX и наша цветная обшивка используют один исходный каталог, но
        # являются разными режимами показа и отдельными шагами таймера.
        super().__init__(COBRA_ID if selected is None else selected,
                         model_source="elite" if model_source == "zx" else model_source,
                         texture_source=texture_source, catalog=catalog)
        self.model_source = model_source
        self.camera_mode = "flyby"
        self._animate()

    def _load_objects(self, ships):
        # Атласу нужны точные облака для inspect.fit. Постановочному пролёту
        # хватает нормализованных сфер: не сортируем hi-res Pioneer np.unique.
        bases, docks, angles = [], [], []
        for index, ship in enumerate(ships):
            angle = self._route.index(index) * tau / self.slot_count
            position = self.ring_radius * np.array((sin(angle), 0.0, cos(angle)))
            position[1] = self._ring_height(angle)
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
            # Камера теперь снаружи: входы станций обращены к внешней орбите.
            # Остальные ракурсы сохраняют передние/задние три четверти.
            rotation = _rotation((0, 1, 0), angle - side)
            if index in (COBRA_ID, 10, 12):
                rotation = rotation @ _rotation((1, 0, 0), -.14)
            if index == 3 and not ship.meshes:
                # У тонкого обломка широкая плоскость обращена к камере.
                source = np.array([(v.x, v.y, v.z) for v in MODELS[index].vertices], dtype=float)
                _, _, axes = np.linalg.svd(source - source.mean(axis=0))
                source_basis = np.column_stack((axes[0], np.cross(axes[2], axes[0]), axes[2]))
                normal = _unit((position[0], 0, position[2]))
                right = _unit(np.cross((0, 1, 0), normal))
                axis = _unit(right * .86 + np.array((0, .5, 0)))
                rotation = np.column_stack((axis, np.cross(normal, axis), normal)) @ source_basis.T
            radius = self._exhibit_radii[index]
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
            self._load_objects(self.catalog.classic if source == "zx" else self.catalog.fleet(source))
            self._layouts[source] = (self.ships, self._base_objects, self._docks,
                                     self._inspection_angles)
        self.model_source = source
        self._overview_cache = self._inspect_cache = None

    def set_source(self, source):
        self._apply_source(source)
        self._source_epoch = self.elapsed
        self._source_anchor = source
        self._animate()

    @property
    def source_options(self):
        """Одинаковая последовательность ручной и автоматической смены."""
        return ("zx", *self.catalog.sources)

    def cycle_source(self):
        options = self.source_options
        self.set_source(options[(options.index(self.model_source) + 1) % len(options)])

    def cycle_textures(self):
        if self.model_source != "zx":
            super().cycle_textures()

    def toggle_auto_sources(self):
        self.auto_sources = not self.auto_sources
        if self.auto_sources and self.source_interval == 0:
            self.source_interval = self.default_source_interval
        self._source_epoch = self.elapsed
        self._source_anchor = self.model_source
        return self.auto_sources

    def _advance_sources(self):
        if not self.auto_sources or self.source_interval <= 0:
            return
        delta = self.elapsed - self._source_epoch
        # Конечные время/интервал могут иметь бесконечное float-отношение.
        # Их точные двоичные дроби дают целое число смен без переполнения;
        # divmod также даёт правильный шаг назад до ручного якоря.
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
        options = self.source_options
        source = options[(options.index(self._source_anchor) + changes) % len(options)]
        self._apply_source(source)

    def seek(self, seconds):
        seconds = float(seconds)
        if not isfinite(seconds) or seconds < 0:
            raise ValueError("Время облёта должно быть конечным и неотрицательным")
        self.elapsed = seconds
        self.animation_elapsed = seconds
        self.source_interval = self._initial_interval
        self.auto_sources = self.source_interval > 0
        self._source_epoch = 0.0
        self._source_anchor = self._initial_source
        self._apply_source(self._initial_source)
        self._advance_sources()
        self._animate()

    def toggle_pause(self):
        """Остановить или продолжить маршрут; вращения и осмотр независимы."""
        self.paused = not self.paused
        return self.paused

    @staticmethod
    def _shift_time(seconds, delta, *, lower=0.0):
        # Даже сумма двух конечных float может переполниться. Часы остаются
        # в представимом диапазоне, в том числе при предельной перемотке.
        return min(float_info.max, max(lower, seconds + delta))

    def scrub(self, delta_seconds):
        """Перемотать текущий показ и вернуть пройденное время маршрута.

        Нижний предел — начало показа. Часы вращений сдвигаются ровно на
        пройденный участок, сохраняя фазу, накопленную во время паузы.
        Ручные наборы, текстуры, пауза и относительный взгляд сохраняются.
        """
        delta = float(delta_seconds)
        if not isfinite(delta):
            raise ValueError("Шаг перемотки должен быть конечным")
        if self.camera_mode == "flight":
            return 0.0
        elapsed = self._shift_time(self.elapsed, delta)
        moved = elapsed - self.elapsed
        self.elapsed = elapsed
        # После стыковки с маршрутом вращения могут пройти нулевой угол
        # назад; нижняя граница начала показа остаётся только у маршрута.
        self.animation_elapsed = self._shift_time(self.animation_elapsed, moved, lower=-float_info.max)
        self._advance_sources()
        self._animate()
        return moved

    def update(self, dt, movement=(0.0, 0.0, 0.0), *, advance_route=True):
        # Сценарий не зависит от частоты кадров и свободных осей WASD.
        dt = float(dt)
        if not isfinite(dt):
            raise ValueError("Шаг времени должен быть конечным")
        dt = max(0.0, dt)
        self.animation_elapsed = self._shift_time(self.animation_elapsed, dt, lower=-float_info.max)
        if self.camera_mode == "flight":
            super().update(dt, movement)
            self._animate()
            return
        # Удержание стрелок временно отдаёт маршрут командам scrub, не
        # меняя выбранную пользователем паузу. Вращения идут и при этом.
        if not self.paused and advance_route:
            self.elapsed = self._shift_time(self.elapsed, dt)
        self._advance_sources()
        self._animate()

    def _path_angle(self, phase):
        phase %= self.slot_count
        # У корпуса скорость ниже, между корпусами выше; нулевых остановок нет.
        phase -= .42 * sin(tau * phase) / tau
        return phase * tau / self.slot_count

    @property
    def empty_slot(self):
        """Встреча с зарезервированным местом, для которого в наборе нет гостя."""
        if self.camera_mode == "flight":
            return False
        phase = self._start_slot + self.elapsed / self.seconds_per_ship
        focus = self._route[int(floor(phase + .5)) % self.slot_count]
        return focus >= len(self.ships)

    def _animate(self):
        phase = self._start_slot + self.elapsed / self.seconds_per_ship
        angle = self._path_angle(phase)
        slot = phase % self.slot_count
        station_weight = _smootherstep(min(slot - 9, 13 - slot))
        # На переходе не удаляемся от обоих соседей. У двух станций плавно
        # отступаем, чтобы крупный корпус оставлял вокруг себя дальнюю дугу.
        index = int(floor(slot))
        blend = _smootherstep(slot - index)
        radius = ((1 - blend) * self._exhibit_radii[self._route[index]]
                  + blend * self._exhibit_radii[self._route[(index + 1) % self.slot_count]])
        gap = 1.7 + .85 * radius + 1.45 * station_weight
        # Вторая половина маршрута проходит под грузом и астероидами.
        # Плавно пересекаем плоскость лишь на двух коротких участках; у
        # станций и в начале сохраняем верхний ракурс с читаемым входом.
        descent = _smootherstep(slot - 14.5)
        ascent = _smootherstep(19.5 - slot)
        elevation = 1 - 2 * descent * ascent
        height = self._ring_height(angle) + (.70 * gap + .40) * elevation
        camera_radius = self.ring_radius + gap
        camera_angle = angle + .025
        self._camera_eye = np.array((camera_radius * sin(camera_angle), height,
                                     camera_radius * cos(camera_angle)))
        radial = np.array((sin(angle), 0.0, cos(angle)))
        tangent = np.array((cos(angle), 0.0, -sin(angle)))
        # Взгляд проходит сквозь ближнюю сторону кольца к его середине.
        # Небольшой боковой и вертикальный сдвиг героя открывает за ним
        # широкую глубокую дугу; наведение не зависит от набора геометрии.
        # Полный roll меняет высоту силуэта: оставляем поля перед верхней
        # строкой и нижней подписью также у поперечного ракурса крыльев.
        tilt = .74 - .12 * descent * ascent
        self._camera_target = self._camera_eye + 20 * (
            -radial + .10 * tangent + np.array((0.0, -tilt * elevation, 0.0)))
        focus = self._route[int(floor(phase + .5)) % self.slot_count]
        if focus >= len(self.ships):
            # Технический индекс нужен базовому контракту selected_object.
            # Интерфейс использует empty_slot и не приписывает месту соседа.
            focus = 0 if slot < self._route.index(len(MODELS)) else 1
        self.selected = focus
        objects = []
        for obj, dock in zip(self._base_objects, self._docks):
            index = obj.model_id
            time = self.animation_elapsed
            if dock is not None:
                rotation = _rotation((0, 0, 1), time * (.20 if index == 0 else -.17))
            elif index in (3, 5, 6, 7):
                rotation = _rotation((.35 + index*.03, .8, .45), time * (.11 + index*.008))
            else:
                # Начальная постановочная поза остаётся постоянной, а полный
                # оборот идёт вокруг продольной +Z канонической сетки.
                rotation = (_rotation((0, 1, 0), .21 * sin(index*.63))
                            @ _rotation((1, 0, 0), .105 * sin(index*.81))
                            @ _rotation((0, 0, 1), .14 * sin(index*.49)
                                        + time * self._roll_rates[index]))
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
        if self.camera_mode == "flight":
            self.selected = self._nearest_object()

    def _nearest_object(self):
        """Ближайший настоящий центр; пустое гостевое место сюда не попадает."""
        eye = self.eye
        return min(self.objects, key=lambda obj: np.linalg.norm(obj.position - eye)).model_id

    def _nearest_visit(self, model_id):
        """Ближайшее неотрицательное посещение с учётом начального --model.

        У границы круга рассматриваем также соседний круг. При равных
        расстояниях выбираем будущее; историю не заменяем новым seek(0).
        """
        first = ((self._route.index(model_id) - self._start_slot) % self.slot_count
                 * self.seconds_per_ship)
        lap = max(0, floor((self.elapsed - first) / self.duration + .5))
        return min(float_info.max, first + lap * self.duration)

    def fly(self):
        """Начать свободный полёт с полным текущим видом; повтор не меняет его."""
        if self.camera_mode == "flight":
            return
        self._flight_distance = self.distance
        super().fly()
        self.selected = self._nearest_object()

    def toggle_flight(self):
        """Передать текущий вид свободной камере или продолжить у соседа.

        Маршрут и его таймер стоят в свободном полёте. Возвращение меняет
        только точку маршрута: вращения, набор, текстуры, пауза и ручной
        взгляд автопилота остаются прежними.
        """
        if self.camera_mode != "flight":
            self.fly()
        else:
            nearest = self._nearest_object()
            elapsed = self._nearest_visit(nearest)
            # Переносим якорь вместе с маршрутом, сохраняя пройденную долю
            # интервала и ручной выбор, в том числе гостя Pioneer.
            self._source_epoch += elapsed - self.elapsed
            self.elapsed = elapsed
            self.camera_mode = "flyby"
            self.flight_speed = 0.0
            self._animate()
        return self.camera_mode

    def zoom(self, steps):
        if self.camera_mode == "flight":
            super().zoom(steps)
            self.selected = self._nearest_object()

    def view_projection(self, aspect):
        projection = super().view_projection(aspect)
        # В узком окне расширяем настоящее поле зрения по обеим осям.
        # Силуэт не растягивается, а горизонтальный угол перестаёт сжиматься:
        # герой остаётся целым и в квадратном/портретном окне.
        projection[:2] *= min(1.0, self._aspect / 1.5)
        return projection

    @property
    def eye(self):
        if self.camera_mode == "flight":
            return self._flight_eye.copy()
        return self._camera_eye.copy()

    @property
    def target(self):
        if self.camera_mode == "flight":
            return self.eye + self.forward * self.distance
        if self._look_yaw == self._look_pitch == 0:
            return self._camera_target.copy()
        return self.eye + self.forward * self.distance

    @property
    def forward(self):
        if self.camera_mode == "flight":
            return super().forward
        return self._view_basis()[2]

    @property
    def up(self):
        """Верх постановочной камеры с учётом относительного взгляда."""
        return self._view_basis()[1]

    def _view_basis(self):
        if self.camera_mode == "flight":
            return super()._view_basis()
        forward = _unit(self._camera_target - self._camera_eye)
        right = _unit(np.cross(forward, self._camera_up))
        up = np.cross(right, forward)
        cy, sy = cos(self._look_yaw), sin(self._look_yaw)
        cp, sp = cos(self._look_pitch), sin(self._look_pitch)
        horizontal = forward * cy + right * sy
        right = right * cy - forward * sy
        forward = horizontal * cp + up * sp
        return right, np.cross(right, forward), forward

    def look_drag(self, dx, dy):
        """Поворот относительно курса; маршрут камеры остаётся прежним."""
        dx, dy = float(dx), float(dy)
        if not (isfinite(dx) and isfinite(dy)):
            return
        if self.camera_mode == "flight":
            super().drag(dx, dy)
        else:
            self._look_yaw = (self._look_yaw + dx * .005 + pi) % tau - pi
            self._look_pitch = float(np.clip(self._look_pitch - dy * .005, -pi/2 + .01, pi/2 - .01))

    def reset_look(self):
        """Только явная команда R возвращает исходное наведение."""
        if self.camera_mode != "flight":
            self._look_yaw = self._look_pitch = 0.0

    @property
    def distance(self):
        if self.camera_mode == "flight":
            return self._flight_distance
        return float(np.linalg.norm(self._camera_target - self._camera_eye))
