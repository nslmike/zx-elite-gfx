"""Кольцо подлинных моделей и камера отдельного OpenGL-атласа.

Математические матрицы записаны построчно и умножаются на векторы-столбцы.
Мир использует Y вверх; камера OpenGL смотрит вдоль локальной оси −Z.
Рендереру передаётся геометрия в единичной сфере и её мировая матрица.
Ни один из численных алгоритмов спектрумовского рендерера здесь не нужен.
"""

from dataclasses import dataclass
from math import atan2, cos, exp, isfinite, pi, sin, sqrt, tan
from sys import float_info

import numpy as np

from .model import COBRA_ID, DODO_ID, MODELS
from .space_assets import AssetCatalog, AssetError, TEXTURE_NAMES


@dataclass(frozen=True)
class SpaceObject:
    """Неподвижный экспонат; matrix включает мировой масштаб radius."""

    model_id: int
    position: np.ndarray
    radius: float
    matrix: np.ndarray


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _rotation(axis, angle):
    """Матрица Родрига: положительный угол следует правилу правой руки."""
    x, y, z = _unit(axis)
    c, s, t = cos(angle), sin(angle), 1 - cos(angle)
    return np.array(((t*x*x+c, t*x*y-s*z, t*x*z+s*y),
                     (t*x*y+s*z, t*y*y+c, t*y*z-s*x),
                     (t*x*z-s*y, t*y*z+s*x, t*z*z+c)))


class SpaceScene:
    """Общий план, осмотр одного корпуса и свободный полёт между ними.

    drag принимает перемещение в логических пикселях SDL. zoom принимает
    дробные шаги колеса/трекпада; положительный шаг приближает. update
    принимает (вправо, вверх, вперёд) либо такую же тройку с четвёртым
    множителем скорости. Выбор через pick не меняет состояние камеры.
    """

    is_flyby = False
    ring_radius = 25.5
    fov = 48.0
    speed = 10.0
    flight_max_speed = 30.0
    flight_acceleration = 10.0

    def __init__(self, selected=None, *, model_source="elite", texture_source="author", catalog=None):
        self.catalog = catalog or AssetCatalog()
        if texture_source not in TEXTURE_NAMES:
            raise AssetError(f"Неизвестный источник текстур: {texture_source}")
        self.model_source = model_source
        self.texture_source = texture_source
        self.effects_enabled = True
        self.animation_elapsed = 0.0
        self._load_objects(self.catalog.fleet(model_source))
        if selected is None:
            selected = COBRA_ID
            if model_source in self.catalog.pack_ids and self.ships[selected].source_id != model_source:
                selected = min(self.catalog.pack(model_source))
        self.selected = int(selected) % len(self.objects)
        self.camera_mode = "overview"
        self._aspect = 1.6
        self._yaw = 0.0
        self._pitch = -0.58
        self._zoom = 1.0
        self._flight_eye = np.zeros(3, dtype=np.float64)
        self._flight_basis = None
        self.flight_profile = "modern"
        self.flight_speed = 0.0
        self._overview_cache = None
        self._inspect_cache = None

    def set_source(self, source):
        if source == self.model_source:
            return
        ships = self.catalog.fleet(source)
        previous = self.selected
        previous_yaw = self._inspection_angles[previous][0]
        self._load_objects(ships)
        self.model_source = source
        self._overview_cache = self._inspect_cache = None
        self.selected = previous if previous < len(ships) else COBRA_ID
        if source in self.catalog.pack_ids and self.ships[self.selected].source_id != source:
            supported = self.catalog.pack(source)
            self.selected = COBRA_ID if COBRA_ID in supported else min(supported)
        if self.camera_mode == "inspect":
            if self.selected == previous:
                self._yaw += self._inspection_angles[self.selected][0] - previous_yaw
            else:
                self.select(self.selected)

    def cycle_source(self):
        options = self.catalog.sources
        self.set_source(options[(options.index(self.model_source) + 1) % len(options)])

    def cycle_textures(self):
        self.texture_source = next(k for k in TEXTURE_NAMES if k != self.texture_source)

    def _load_objects(self, ships):
        objects, body_points, model_points, angles = [], [], [], []
        for index, ship in enumerate(ships):
            model = MODELS[index] if index < len(MODELS) else None
            angle = 2 * pi * (index - COBRA_ID) / len(ships)
            position = self.ring_radius * np.array((sin(angle), 0, cos(angle)))
            # Камера смотрит изнутри кольца наружу: другие экспонаты обычно
            # находятся позади неё, а не образуют ряд за изучаемым корпусом.
            camera_yaw = pi - angle
            pitch = -0.28
            side = 0.85
            if index in (0, DODO_ID):
                side, pitch = 0.12, -0.30
                # Небольшой сдвиг освобождает край от второй большой станции.
                camera_yaw += -0.10 if index == 0 else 0.175
                if index == DODO_ID:
                    pitch = -0.25
            elif index in (1, 10, 11):
                side, pitch = 1.15, -0.30
            elif index in (15, 18):
                side, pitch = pi / 2, -0.30
            angles.append((camera_yaw, pitch))
            yaw = angle + pi - side
            rotation = np.array(((cos(yaw), 0, sin(yaw)),
                                 (0, 1, 0),
                                 (-sin(yaw), 0, cos(yaw))), dtype=np.float64)
            if index == 3 and not ship.meshes:
                # Обломок почти плоский. Его широкая сторона, а не ребро,
                # обращена к камере; длинная ось проходит по диагонали кадра.
                source = np.array([(v.x, v.y, v.z) for v in model.vertices], dtype=float)
                _, _, axes = np.linalg.svd(source - source.mean(axis=0))
                normal, axis = axes[2], axes[0]
                source_basis = np.column_stack((axis, np.cross(normal, axis), normal))
                eye_direction = np.array((-sin(camera_yaw) * cos(pitch), -sin(pitch),
                                          cos(camera_yaw) * cos(pitch)))
                right = np.array((cos(camera_yaw), 0, sin(camera_yaw)))
                up = np.cross(right, -eye_direction)
                normal = _unit(eye_direction + right * 0.3)
                axis = _unit(right * 0.86 + up * 0.5)
                axis = _unit(axis - normal * np.dot(axis, normal))
                rotation = np.column_stack((axis, np.cross(normal, axis), normal)) @ source_basis.T
            source_radius = sqrt(max(v.x ** 2 + v.y ** 2 + v.z ** 2 for v in model.vertices)) if model else 100.0
            # Масштаб каталога сжат по корню: груз остаётся малым, но видимым.
            # Это выставочный масштаб, а не утверждение о размерах кораблей.
            radius = 1.7 * sqrt(source_radius / 100.0)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rotation * radius
            matrix[:3, 3] = position
            position.setflags(write=False)
            matrix.setflags(write=False)
            objects.append(SpaceObject(index, position, radius, matrix))
            mesh = ship.vertices()
            points = np.unique(mesh[:, :3], axis=0) @ matrix[:3, :3].T + position
            # Готовый mesh центрирован и нормализован вместе с надстройками.
            # Обшивка/камень задают композицию; стволы и усы — только поля.
            body = points
            if not ship.meshes:
                body = mesh[np.isin(mesh[:, 8], (0, 5)), :3]
                body = np.unique(body, axis=0) @ matrix[:3, :3].T + position
            body_points.append(body if len(body) else points)
            model_points.append(points)
        self.ships, self.objects = ships, tuple(objects)
        self._body_points, self._model_points, self._inspection_angles = body_points, model_points, angles

    @property
    def selected_object(self):
        return self.objects[self.selected]

    @property
    def forward(self):
        """Единичное направление взгляда в мировых координатах."""
        if self.camera_mode == "flight":
            return self._flight_basis[2].copy()
        return np.array((sin(self._yaw) * cos(self._pitch), sin(self._pitch),
                         -cos(self._yaw) * cos(self._pitch)))

    @property
    def target(self):
        if self.camera_mode == "inspect":
            return self._inspect_framing()[0].copy()
        if self.camera_mode == "flight":
            return self._flight_eye + self.forward
        return self._overview_framing()[0].copy()

    def _overview_framing(self):
        """Вписать наклонённое кольцо, а не огромную описанную сферу.

        Каждая сфера должна остаться внутри боковых плоскостей фрустума
        с полями. Смещение точки наведения к передней дуге компенсирует
        перспективу; в узком окне оно естественно уменьшается.
        """
        key = (self._aspect, self._yaw, self._pitch)
        if self._overview_cache is not None and self._overview_cache[0] == key:
            return self._overview_cache[1]
        forward = self.forward
        right = _unit(np.cross(forward, (0, 1, 0)))
        up = np.cross(right, forward)
        towards_eye = _unit((-forward[0], 0, -forward[2]))
        positions = np.array([obj.position for obj in self.objects])
        radii = np.array([obj.radius for obj in self.objects])
        tangent = tan(self.fov * pi / 360)
        horizontal = tangent * self._aspect * 0.78
        vertical = tangent * 0.74  # По 13% высоты остаётся для HUD.
        distance = self.ring_radius * 2.1
        for _ in range(12):
            c = cos(self._pitch)
            offset = 2 * c * self.ring_radius ** 2 / (
                sqrt(distance ** 2 + 4 * c ** 2 * self.ring_radius ** 2) + distance)
            target = towards_eye * offset
            relative = positions - target
            depths = relative @ forward
            # Поправка радиуса учитывает наклон плоскости фрустума.
            fit_x = (np.abs(relative @ right) + radii * sqrt(1 + horizontal ** 2)) / horizontal - depths
            fit_y = (np.abs(relative @ up) + radii * sqrt(1 + vertical ** 2)) / vertical - depths
            distance = float(max(fit_x.max(), fit_y.max()))
        framing = target, distance
        self._overview_cache = key, framing
        return framing

    def _inspect_framing(self):
        """Крупный план по видимому корпусу, с запасом для тонких деталей."""
        key = (self.selected, self._aspect, self._yaw, self._pitch, self._zoom)
        if self._inspect_cache is not None and self._inspect_cache[0] == key:
            return self._inspect_cache[1]
        forward = self.forward
        right = _unit(np.cross(forward, (0, 1, 0)))
        up = np.cross(right, forward)
        body = self._body_points[self.selected]
        complete = self._model_points[self.selected]
        target = (body.min(axis=0) + body.max(axis=0)) / 2
        tangent = tan(self.fov * pi / 360)

        def fit(points, width, height):
            relative = points - target
            depth = relative @ forward
            fit_x = np.abs(relative @ right) / (tangent * self._aspect * width) - depth
            fit_y = np.abs(relative @ up) / (tangent * height) - depth
            return float(max(fit_x.max(), fit_y.max()))

        def center(distance):
            relative = body - target
            depth = distance + relative @ forward
            x = (relative @ right) / depth
            y = (relative @ up) / depth
            return (right * (x.max() + x.min()) + up * (y.max() + y.min())) * distance / 2

        # Перспектива асимметрична: центр исходных координат не совпадает
        # с центром проекции. Несколько итераций совмещают именно силуэт.
        body_width = min(0.74, 0.56 + max(0.0, 1.3 - self._aspect) * 0.2)
        body_height = 0.68 if self.selected in (0, DODO_ID) else 0.64
        for _ in range(16):
            distance = max(fit(body, body_width, body_height), fit(complete, 0.86, 0.74))
            target = target + center(distance)
        distance *= self._zoom
        distance = max(distance, -float(((complete - target) @ forward).min()) + 0.15)
        for _ in range(12):
            target = target + center(distance)
        framing = target, distance
        self._inspect_cache = key, framing
        return framing

    @property
    def distance(self):
        """Радиус облёта; для обзора учитывает последнее соотношение сторон."""
        if self.camera_mode == "inspect":
            return self._inspect_framing()[1]
        return self._overview_framing()[1] * self._zoom

    @property
    def eye(self):
        if self.camera_mode == "flight":
            return self._flight_eye.copy()
        return self.target - self.forward * self.distance

    def overview(self):
        """Вернуть полный наклонённый круг, не сбрасывая выбранный экспонат."""
        self.camera_mode = "overview"
        self.flight_speed = 0.0
        self._yaw, self._pitch, self._zoom = 0.0, -0.58, 1.0

    def select(self, index, *, focus=True):
        self.selected = int(index) % len(self.objects)
        if focus:
            self.camera_mode = "inspect"
            self.flight_speed = 0.0
            self._yaw, self._pitch = self._inspection_angles[self.selected]
            self._zoom = 1.0

    def fly(self):
        """Передать полёту полную позу камеры; повторный вызов её сохраняет."""
        if self.camera_mode == "flight":
            return
        self.set_flight_pose(self.eye, self._view_basis())

    def set_flight_pose(self, eye, basis):
        """Задать исходный вид полёта: точку и базис (вправо, вверх, вперёд).

        Базис уже ортонормирован; копируем его целиком, сохраняя крен.
        Явный перенос обнуляет тягу и сохраняет выбранную схему управления.
        """
        self._flight_eye = np.array(eye, dtype=np.float64)
        self._flight_basis = np.array(basis, dtype=np.float64)
        self._sync_flight_angles()
        self.flight_speed = 0.0
        self.camera_mode = "flight"

    def _sync_flight_angles(self):
        """Углы нужны только мировым осям Modern, не для хранения крена."""
        right, _, forward = self._flight_basis
        horizontal = float(np.linalg.norm(forward[[0, 2]]))
        self._yaw = (atan2(forward[0], -forward[2]) if horizontal > 1e-12
                     else atan2(right[2], right[0]))
        self._pitch = atan2(forward[1], horizontal)

    def cycle_flight_profile(self):
        """Сменить схему без пересборки позы и начать с нулевой тяги."""
        self.flight_profile = "elite" if self.flight_profile == "modern" else "modern"
        self.flight_speed = 0.0
        return self.flight_profile

    def turn_flight(self, yaw=0.0, pitch=0.0, roll=0.0):
        """Радианы: нос вправо, нос вверх, крен вправо по часовой пилота.

        В Modern рысканье идёт вокруг мировой Y, тангаж — относительно
        горизонта. Elite вращается вокруг собственных осей без полюсов
        и ограничения тангажа. Обе схемы сохраняют полноценный крен.
        """
        yaw, pitch, roll = float(yaw), float(pitch), float(roll)
        if (self.camera_mode != "flight" or not all(map(isfinite, (yaw, pitch, roll)))
                or yaw == pitch == roll == 0):
            return

        def rotate(axis, angle):
            if angle:
                self._flight_basis = self._flight_basis @ _rotation(axis, angle).T

        if self.flight_profile == "elite":
            rotate(self._flight_basis[1], -yaw)
            rotate(self._flight_basis[0], pitch)
        else:
            rotate((0, 1, 0), -yaw)
            self._yaw += yaw
            minimum, maximum = min(-1.48, self._pitch), max(1.48, self._pitch)
            pitch = float(np.clip(self._pitch + pitch, minimum, maximum)) - self._pitch
            rotate((cos(self._yaw), 0, sin(self._yaw)), pitch)
        rotate(self._flight_basis[2], roll)
        # Убираем накопление погрешности после длинных серий полных петель.
        right, _, forward = self._flight_basis
        forward = _unit(forward)
        right = _unit(right - forward * np.dot(right, forward))
        self._flight_basis = np.array((right, np.cross(right, forward), forward))
        self._sync_flight_angles()

    def drag(self, dx, dy):
        if self.camera_mode == "flight":
            horizontal, pitch = float(dx) * .005, -float(dy) * .005
            if self.flight_profile == "elite":
                self.turn_flight(0.0, pitch, horizontal)
            else:
                self.turn_flight(horizontal, pitch, 0.0)
            return
        self._yaw = (self._yaw + float(dx) * 0.005) % (2 * pi)
        minimum, maximum = (-1.48, -0.20) if self.camera_mode == "overview" else (-1.48, 1.48)
        self._pitch = float(np.clip(self._pitch - float(dy) * 0.005, minimum, maximum))

    def zoom(self, steps):
        steps = float(steps)
        if self.camera_mode == "flight":
            if self.flight_profile == "elite":
                self.flight_speed = float(np.clip(self.flight_speed + steps * self.speed * .18,
                                                  0, self.flight_max_speed))
            else:
                self._flight_eye += self.forward * steps * self.speed * 0.18
            return
        limits = (0.32, 2.5) if self.camera_mode == "overview" else (0.36, 4.0)
        self._zoom = float(np.clip(self._zoom * exp(np.clip(-steps * 0.12, -20, 20)), *limits))

    def update(self, dt, movement=(0.0, 0.0, 0.0)):
        """Движение камеры; пауза приложения не должна вызывать скачок."""
        if not self.is_flyby and isfinite(dt) and dt > 0.:
            self.animation_elapsed = min(float_info.max, self.animation_elapsed + float(dt))
        if self.camera_mode != "flight":
            return
        movement = tuple(movement)
        axes = np.clip(np.asarray(movement[:3], dtype=np.float64), -1, 1)
        step = float(np.clip(dt, 0, 0.1))
        if self.flight_profile == "elite":
            previous = self.flight_speed
            acceleration = float(axes[2]) * self.flight_acceleration
            self.flight_speed = float(np.clip(previous + acceleration * step,
                                              0, self.flight_max_speed))
            # Интегрируем разгон точно, включая часть кадра после достижения
            # нуля или максимума. Боковые оси и ускоритель Elite не переносят.
            changing = min(step, (self.flight_speed - previous) / acceleration) if acceleration else step
            distance = ((previous + self.flight_speed) * .5 * changing
                        + self.flight_speed * (step - changing))
            self._flight_eye += self.forward * distance
            return
        magnitude = np.linalg.norm(axes)
        if magnitude > 1:
            axes /= magnitude
        boost = float(np.clip(movement[3], 0.05, 20)) if len(movement) > 3 else 1.0
        right = np.array((cos(self._yaw), 0, sin(self._yaw)))
        # Вертикальное перемещение Space/Shift остаётся вдоль мировой Y.
        displacement = right * axes[0] + np.array((0, axes[1], 0)) + self.forward * axes[2]
        self._flight_eye += displacement * self.speed * boost * step

    def _view_basis(self):
        if self.camera_mode == "flight":
            return tuple(axis.copy() for axis in self._flight_basis)
        forward = self.forward
        right = _unit(np.cross(forward, (0, 1, 0)))
        up = np.cross(right, forward)
        return right, up, forward

    def view_projection(self, aspect):
        self._aspect = max(0.05, float(aspect))
        right, up, forward = self._view_basis()
        eye = self.eye
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = (right, up, -forward)
        view[:3, 3] = -view[:3, :3] @ eye
        near = 0.06
        far = max(2000.0, np.linalg.norm(eye) + self.ring_radius * 3)
        focal = 1 / tan(self.fov * pi / 360)
        projection = np.array(((focal / self._aspect, 0, 0, 0),
                               (0, focal, 0, 0),
                               (0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)),
                               (0, 0, -1, 0)), dtype=np.float64)
        return projection @ view

    def project(self, position, width, height):
        """Центр подсказки в логических пикселях SDL либо None вне кадра."""
        if width <= 0 or height <= 0:
            return None
        clip = self.view_projection(width / height) @ np.append(position, 1.0)
        if clip[3] <= 0:
            return None
        ndc = clip[:3] / clip[3]
        if np.any(np.abs(ndc) > 1):
            return None
        return float((ndc[0] + 1) * width / 2), float((1 - ndc[1]) * height / 2)

    def pick(self, x, y, width, height):
        """Ближайшая видимая сфера; маленьким объектам добавлен запас 14 px."""
        if width <= 0 or height <= 0 or not (0 <= x < width and 0 <= y < height):
            return None
        projection = self.view_projection(width / height)
        ndc = np.array((2 * x / width - 1, 1 - 2 * y / height, 1, 1))
        point = np.linalg.inv(projection) @ ndc
        direction = _unit(point[:3] / point[3] - self.eye)
        eye = self.eye
        hits = []
        padded = []
        focal_pixels = height / (2 * tan(self.fov * pi / 360))
        for obj in self.objects:
            center = self.project(obj.position, width, height)
            if center is None:
                continue
            offset = obj.position - eye
            along = float(np.dot(offset, direction))
            discriminant = obj.radius ** 2 - (float(np.dot(offset, offset)) - along ** 2)
            if discriminant >= 0 and along > 0:
                entry = along - sqrt(max(0, discriminant))
                hits.append((max(0, entry), obj.model_id))
            depth = float(np.dot(offset, self.forward))
            if depth > 0:
                pixel_radius = max(14.0, obj.radius * focal_pixels / depth)
                separation = (x - center[0]) ** 2 + (y - center[1]) ** 2
                if separation <= pixel_radius ** 2:
                    padded.append((separation / pixel_radius ** 2, depth, obj.model_id))
        if hits:
            return min(hits)[1]
        return min(padded)[2] if padded else None
