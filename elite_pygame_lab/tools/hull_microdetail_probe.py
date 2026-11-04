#!/usr/bin/env python3
"""Воспроизводимый A/B микрорельефа на собственной классической Cobra №09.

Из elite_pygame_lab:
    python tools/hull_microdetail_probe.py --draft-dir /private/tmp/elite-microdetail
    python tools/hull_microdetail_probe.py --fly

Использует SpaceRenderer/SpaceLook, исходные меш, карты, BRDF, MSAA и CGL/EGL.
Две точечные вставки GLSL существуют только при создании этого экземпляра.
Production-файлы и материалы сторонних моделей не изменяются.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import asin, atan2, cos, isfinite, radians, sin, tan
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import moderngl
import numpy as np
import pygame

from elite_lab import space_look, space_viewer
from elite_lab.space_assets import AssetCatalog
from elite_lab.space_scene import SpaceObject


ROOT = Path(__file__).resolve().parents[1]
FRAGMENT = ROOT / "elite_lab/shaders/space_hull.frag"
DETAIL = Path(__file__).with_name("hull_microdetail.glsl")
SUN = np.array((-0.68, 0.55, -0.48), dtype="f4")
SUN /= np.linalg.norm(SUN)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def probe_cameras(distance=1.0):
    """Общие локальные ракурсы для записи PNG и старта свободного полёта."""
    ship_target = np.array((0.20, 0.0, -0.18))
    close_target = np.array((0.60, 0.087, -0.25))
    return {
        "ship": (ship_target, ship_target + np.array((0.91, 0.91, 1.33)) * distance,
                 38.0, (1620, 1000)),
        "close": (close_target, close_target + np.array((0.11, 0.18, 0.21)) * distance,
                  39.0, (1200, 820)),
    }


def experiment_shader():
    original = FRAGMENT.read_text()
    # Surface должна быть определена до вставки; original остаётся без изменений.
    anchor = "Surface procedural_surface() {"
    call = "    vec3 n = surface.normal;"
    if original.count(anchor) != 1 or original.count(call) != 1:
        raise RuntimeError("Штатный GLSL изменился: проверьте две точки вставки")
    modified = original.replace(anchor, DETAIL.read_text() + "\n" + anchor)
    return modified.replace(call, "    surface = hull_microdetail(surface);\n" + call)


class ShaderOverlay:
    """Локальный источник одного shader; остальные файлы читает штатный код."""
    def __init__(self, root, fragment):
        self.root, self.fragment = root, fragment

    def __truediv__(self, name):
        if name == "space_hull.frag":
            return self
        return self.root / name

    def read_text(self):
        return self.fragment


class ProbeFlightScene(space_viewer.SpaceScene):
    """Штатное кольцо и камера с аккуратным стартом у обшивки Cobra."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.microdetail_enabled = False
        self.reset_probe()

    def reset_probe(self):
        """R возвращает Cobra и close-план с PNG, сохраняя выбор слоя A/B."""
        self.set_source("elite")
        self.selected = 9
        self.texture_source = "procedural"
        obj = self.objects[9]
        target, eye, self.fov, _ = probe_cameras()["close"]
        axes = obj.matrix[:3, :3]
        forward = unit(axes @ (target - eye))
        self._yaw = atan2(forward[0], -forward[2])
        self._pitch = asin(float(np.clip(forward[1], -1.0, 1.0)))
        self._flight_eye = axes @ eye + obj.position
        self.camera_mode = "flight"
        self._update_probe_speed()

    def _update_probe_speed(self):
        if self.camera_mode == "flight":
            gap = min(np.linalg.norm(self.eye - obj.position) - obj.radius for obj in self.objects)
            self.speed = float(np.clip(gap * 0.9, 0.22, 10.0))

    def update(self, dt, movement=(0.0, 0.0, 0.0)):
        self._update_probe_speed()
        super().update(dt, movement)

    def zoom(self, steps):
        self._update_probe_speed()
        super().zoom(steps)


class ProbeFlightLook(space_look.SpaceLook):
    def __init__(self, ctx, scene):
        self.probe_scene = scene
        super().__init__(ctx)

    def configure_model(self, model_id, *, imported=False, authored=False, cross_bitangent=False):
        super().configure_model(model_id, imported=imported, authored=authored,
                                cross_bitangent=cross_bitangent)
        # Декор создан для Cobra09: другие модели и все чужие материалы проходят
        # исходную ветку, даже когда пользователь меняет набор через M или T.
        enabled = self.probe_scene.microdetail_enabled and model_id == 9 and not (imported or authored)
        self.hull["u_microdetail"].value = float(enabled)


class ProbeFlightRenderer(space_viewer.SpaceRenderer):
    def __init__(self, ctx, scene):
        # Кольцо сохраняет штатные трансформации. Поворачиваем свет вместе
        # с локальным ракурсом Cobra, включая HDR-диск и тени.
        sun = unit(scene.objects[9].matrix[:3, :3] @ SUN).astype("f4")
        shaders = ShaderOverlay(space_look._SHADERS, experiment_shader())
        with patch.object(space_look, "_SHADERS", shaders), \
             patch.object(space_look, "SUN_DIRECTION", tuple(sun)), \
             patch.object(space_look, "_SUN", sun), \
             patch.object(space_viewer, "SUN_DIRECTION", tuple(sun)), \
             patch.object(space_viewer, "SpaceLook", lambda context: ProbeFlightLook(context, scene)):
            super().__init__(ctx, scene)


class ProbeFlightControls(space_viewer.SpaceControls):
    def __init__(self, scene, hud, show_hud):
        super().__init__(scene, hud, show_hud)
        self._caption()

    def _caption(self):
        state = "B: пробный слой" if self.scene.microdetail_enabled else "A: штатная обшивка"
        pygame.display.set_caption(f"Elite · Проба обшивки · {state} · B: переключить · R: к борту · O: кольцо")

    def event(self, e, logical_size, drawable_size):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_b:
            self.scene.microdetail_enabled = not self.scene.microdetail_enabled
            self._caption()
            return True
        if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            self.scene.reset_probe()
            self.hud.menu = None
            return True
        return super().event(e, logical_size, drawable_size)


def run_flight():
    """Ввод, resize, рендер, переключение наборов и cleanup — общий run_space()."""
    print("Проба обшивки: старт A — штатная обшивка; B — пробный слой вкл/выкл; "
          "R — к борту Cobra09, сохраняя выбор слоя; мышь с зажатой кнопкой — обзор; "
          "WASD/QE — полёт; Shift — быстрее; колесо — вперёд/назад; O — кольцо; "
          "Enter — осмотр; F — свободный полёт; Esc — кольцо/выход.", flush=True)
    with patch.object(space_viewer, "SpaceScene", ProbeFlightScene), \
         patch.object(space_viewer, "SpaceRenderer", ProbeFlightRenderer), \
         patch.object(space_viewer, "SpaceControls", ProbeFlightControls):
        space_viewer.run_space(None, None, False, True, "elite", "procedural")


@dataclass
class ProbeScene:
    """Один экспонат, тот же контракт сцены штатного SpaceRenderer."""
    ships: tuple
    objects: tuple
    eye: np.ndarray
    target: np.ndarray
    fov: float = 38.0
    model_source: str = "elite"
    texture_source: str = "procedural"
    camera_mode: str = "inspect"
    ring_radius: float = 25.5
    is_flyby: bool = False

    def view_projection(self, aspect):
        forward = unit(self.target - self.eye)
        right = unit(np.cross(forward, (0, 1, 0)))
        up = np.cross(right, forward)
        view = np.eye(4)
        view[:3, :3] = (right, up, -forward)
        view[:3, 3] = -view[:3, :3] @ self.eye
        near, far = 0.025, 2000.0
        focal = 1 / tan(radians(self.fov) / 2)
        projection = np.array(((focal/aspect, 0, 0, 0), (0, focal, 0, 0),
            (0, 0, -(far+near)/(far-near), -2*far*near/(far-near)), (0, 0, -1, 0)))
        return projection @ view


def read_frame(ctx, renderer, scene, size, detail):
    if "u_microdetail" in renderer.look.hull:
        renderer.look.hull["u_microdetail"].value = detail
    elif detail != 0:
        raise ValueError("Исходный shader не содержит микрорельеф")
    target = ctx.simple_framebuffer(size, samples=min(4, ctx.max_samples))
    resolved = ctx.simple_framebuffer(size)
    try:
        renderer.draw(scene, target, size, False)
        ctx.copy_framebuffer(resolved, target)
        return pygame.image.frombytes(resolved.read(components=3), size, "RGB", True)
    finally:
        resolved.release()
        target.release()


def font(size, bold=False):
    # Системный шрифт с кириллицей, без внешних зависимостей.
    return pygame.font.SysFont("Arial", size, bold=bold)


def compose(images, destination):
    width, margin, gap = 2040, 36, 24
    tile = (width - margin * 2 - gap) // 2
    top_h, close_h = 600, 660
    canvas = pygame.Surface((width, 1520))
    canvas.fill((13, 21, 29))

    def text(label, x, y, size, color=(231, 237, 237), bold=False):
        canvas.blit(font(size, bold).render(label, True, color), (x, y))

    text("Cobra Mk III · №09", margin, 26, 32, bold=True)
    text("Проба микрорельефа обшивки", margin, 67, 22, (150, 176, 182))
    for index, label in enumerate(("A · Штатная обшивка", "B · Пробный микрорельеф")):
        x = margin + index * (tile + gap)
        text(label, x, 111, 28, bold=True)
        top = pygame.transform.smoothscale(images[("ship", index)], (tile, top_h))
        canvas.blit(top, (x, 156))
        text("Ближний план · 1 пиксель рендера = 1 пиксель изображения", x, 782, 19,
             (150, 176, 182))
        close = images[("close", index)]
        # Здесь ровно 1:1: без апскейла, sharpen и изменения экспозиции.
        crop = pygame.Rect((close.get_width()-tile)//2, (close.get_height()-close_h)//2,
                           tile, close_h)
        canvas.blit(close, (x, 818), crop)
    text("Камера, свет и геометрия совпадают. Добавлены рельеф и шероховатость.",
         margin, 1490, 18, (150, 176, 182))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(canvas, str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draft-dir", type=Path, help="Явно сохранить исходные PNG и параметры в этот каталог")
    parser.add_argument("--fly", action="store_true", help="Полёт у Cobra09 со штатной обшивкой; B — пробный слой, R — к борту с тем же слоем")
    parser.add_argument("--yaw", type=float, help="Поворот корабля вокруг Y на PNG, градусы")
    parser.add_argument("--distance", type=float, help="Масштаб дистанций обеих камер на PNG")
    args = parser.parse_args()
    if args.fly:
        if any(option is not None for option in (args.output, args.draft_dir, args.yaw, args.distance)):
            parser.error("--output, --draft-dir, --yaw и --distance относятся к PNG и несовместимы с --fly")
        run_flight()
        return
    args.yaw = 0.0 if args.yaw is None else args.yaw
    args.distance = 1.0 if args.distance is None else args.distance
    args.output = args.output or ROOT/"previews/hull_microdetail_comparison.png"
    if not isfinite(args.yaw) or not isfinite(args.distance) or args.distance <= 0:
        parser.error("--yaw должен быть конечным числом; --distance — конечным числом больше нуля")
    if args.draft_dir:
        args.draft_dir.mkdir(parents=True, exist_ok=True)
    pygame.font.init()
    options = {"backend": "egl"} if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else {}
    ctx = moderngl.create_context(standalone=True, require=330, **options)
    print(f"OpenGL {ctx.info['GL_VERSION']} · {ctx.info['GL_RENDERER']}", flush=True)
    yaw = radians(args.yaw)
    model = np.array(((cos(yaw), 0, sin(yaw), 0), (0, 1, 0, 0),
                      (-sin(yaw), 0, cos(yaw), 0), (0, 0, 0, 1)), dtype=float)
    ship = AssetCatalog().classic[9]
    obj = SpaceObject(9, np.zeros(3), 1.0, model)
    # Верх: реальная модель целиком. Низ: дополнительная близкая камера,
    # не увеличенный пиксельный фрагмент дальнего изображения.
    cameras = probe_cameras(args.distance)
    ship_target, ship_eye, fov, _ = cameras["ship"]
    scene = ProbeScene((ship,), (obj,), ship_eye, ship_target, fov)
    renderer = None
    try:
        images = {}
        for state in (0, 1):
            # A — буквально исходный shader. Два renderer создаются
            # последовательно: не дублируем их код и не держим два тяжёлых неба.
            shaders = ShaderOverlay(space_look._SHADERS, experiment_shader()) if state else space_look._SHADERS
            with patch.object(space_look, "_SHADERS", shaders), \
                 patch.object(space_look, "SUN_DIRECTION", tuple(SUN)), \
                 patch.object(space_look, "_SUN", SUN), \
                 patch.object(space_viewer, "SUN_DIRECTION", tuple(SUN)):
                renderer = space_viewer.SpaceRenderer(ctx, scene)
            for kind, (target, eye, fov, size) in cameras.items():
                scene.target, scene.eye, scene.fov = target, eye, fov
                frame = read_frame(ctx, renderer, scene, size, float(state))
                images[(kind, state)] = frame
                if args.draft_dir:
                    path = args.draft_dir / f"{kind}_{'now' if state == 0 else 'microdetail'}.png"
                    pygame.image.save(frame, str(path))
                    print(path, flush=True)
            renderer.release()
            renderer = None
        compose(images, args.output)
        metadata = {"renderer": ctx.info["GL_RENDERER"], "version": ctx.info["GL_VERSION"],
                    "shader_sha256": hashlib.sha256(FRAGMENT.read_bytes()).hexdigest(),
                    "look_sha256": hashlib.sha256(Path(space_look.__file__).read_bytes()).hexdigest(),
                    "detail_sha256": hashlib.sha256(DETAIL.read_bytes()).hexdigest(),
                    "sun": SUN.tolist(), "yaw": args.yaw, "distance": args.distance,
                    "model": 9, "source": "elite", "samples": min(4, ctx.max_samples),
                    "baseline": "unmodified production fragment shader",
                    "base_g_height_preserved": True, "detail_materials": [0],
                    "cameras": {key: {"target": target.tolist(), "eye": eye.tolist(),
                                      "fov": fov, "size": size}
                                for key, (target, eye, fov, size) in cameras.items()}}
        if args.draft_dir:
            (args.draft_dir / "render.json").write_text(json.dumps(metadata, indent=2)+"\n")
        print(args.output, flush=True)
    finally:
        if renderer is not None:
            renderer.release()
        ctx.release()
        pygame.quit()


if __name__ == "__main__":
    main()
