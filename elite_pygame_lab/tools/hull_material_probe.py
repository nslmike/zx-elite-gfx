#!/usr/bin/env python3
"""Снять A/B обшивки Cobra №09 и Python №10 штатным рендерером OpenGL.

Из elite_pygame_lab:
    python tools/hull_material_probe.py --raw-dir /tmp/elite-material-frames

Для видео требуется ffmpeg с libx264. --still-only создаёт только PNG.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from shutil import which
from statistics import median
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import moderngl
import numpy as np
import pygame

from elite_lab import space_mesh
from elite_lab.space_assets import AssetCatalog
from elite_lab.space_flyby import FlybyScene
from elite_lab.space_look import SUN_DIRECTION
from elite_lab.space_scene import SpaceObject, _unit
from elite_lab.space_viewer import SpaceRenderer
from hull_microdetail_probe import ProbeScene


STEM = "cobra_python_materials"
SHIPS = {9: "Cobra Mk III · №09", 10: "Python · №10"}
CLOSE_SIZE = (960, 490)
FLYBY_SIZE = (800, 450)
FPS, CLIP_FRAMES = 30, 135
FLYBY_SPEED = 0.5
FLYBY_STILLS = {9: 0, 10: 48}
ANGLES = {
    9: ((0.24, 0.089, 0.08), (0.86959090, 0.57588363, 0.34862545), 45.0),
    10: ((-0.19, 0.105, 0.06), (-0.45010563, 0.77384305, -0.07376861), 44.0),
}
SOURCE_FILES = (
    "space_assets.py", "space_mesh.py", "space_look.py", "space_reflections.py", "space_scene.py", "space_flyby.py",
    "space_viewer.py", "shaders/space_hull.vert", "shaders/space_hull.frag",
    "shaders/space_sky.vert", "shaders/space_sky.frag",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close_scene(model_id, catalog, camera=None):
    target, eye, fov = ANGLES[model_id] if camera is None else camera
    obj = SpaceObject(model_id, np.zeros(3), 1.0, np.eye(4))
    return ProbeScene((catalog.classic[model_id],), (obj,), np.array(eye), np.array(target), fov)


def flyby_scene(model_id, catalog):
    return FlybyScene(model_id, model_source="elite", texture_source="procedural",
                      catalog=catalog, source_interval=0)


def flyby_time(frame):
    return frame * FLYBY_SPEED / FPS


def pose(scene, size):
    matrices = [obj.matrix for obj in scene.objects]
    return {
        "eye": scene.eye.tolist(), "target": scene.target.tolist(), "fov": scene.fov,
        "vp": scene.view_projection(size[0] / size[1]).tolist(),
        "objects_sha256": digest(np.asarray(matrices, dtype="<f8").tobytes()),
    }


class Capture:
    def __init__(self, ctx, scene, size):
        self.ctx, self.size = ctx, size
        self.previous_target = ctx.fbo
        self.renderer = SpaceRenderer(ctx, scene)
        self.target = ctx.simple_framebuffer(size, samples=min(4, ctx.max_samples))
        self.resolved = ctx.simple_framebuffer(size)

    def draw(self, scene, enabled):
        look = self.renderer.look
        if hasattr(look, "refined_materials_enabled"):
            look.refined_materials_enabled = enabled
        elif enabled:
            raise RuntimeError("Эта ревизия поддерживает только --baseline-only")
        self.renderer.draw(scene, self.target, self.size, False)

    def frame(self, scene, enabled):
        self.draw(scene, enabled)
        self.ctx.copy_framebuffer(self.resolved, self.target)
        return self.resolved.read(components=3, alignment=1)

    def timing(self, scene, modes):
        samples = {mode: [] for mode in modes}
        for step in range(16):
            # Чередуем порядок A/B. Первый прогон прогревает шейдеры и текстуры.
            for mode in modes if step % 2 == 0 else reversed(modes):
                started = perf_counter()
                self.draw(scene, mode)
                self.ctx.finish()
                if step >= 4:
                    samples[mode].append((perf_counter() - started) * 1000)
        return {"baseline_ms": median(samples[False]),
                **({"refined_ms": median(samples[True])} if True in samples else {})}

    def release(self):
        if self.previous_target is not None:
            self.previous_target.use()
        self.resolved.release()
        self.target.release()
        self.renderer.release()


class Comparison:
    """Компоновка PNG/кадра ролика без ресэмплинга областей рендера."""

    def __init__(self, size, title, models, *, columns=None):
        self.size = size
        width, height = size
        margin, gutter, row_label = 30, 14, 42
        row_height = height + row_label + 16
        top = 102
        self.canvas = pygame.Surface((2 * width + gutter + 2 * margin,
                                      top + len(models) * row_height + 22))
        self.canvas.fill((12, 20, 31))
        self.positions = {}
        self.label(title, (margin, 18), 28, bold=True)
        note = "Одинаковые камера, свет звезды и экспозиция · кадры 1:1"
        note_image = self.text(note, 19, (156, 176, 196))
        self.canvas.blit(note_image, (self.canvas.get_width() - margin - note_image.get_width(), 27))
        if columns is None:
            columns = ("ДО · Штатный материал", "ПОСЛЕ · Новые отражения и шероховатость")
        for column, (label, color) in enumerate(zip(columns, ((172, 189, 207), (153, 217, 224)))):
            x = margin + column * (width + gutter)
            self.label(label, (x, 64), 19, color, bold=True)
            pygame.draw.rect(self.canvas, color, (x, 94, width, 2))
            for index, model_id in enumerate(models):
                y = top + index * row_height
                self.positions[(model_id, column)] = (x, y + row_label)
                self.label(SHIPS[model_id], (x, y + 8), 22, bold=True)
                pygame.draw.rect(self.canvas, (36, 54, 72), (x - 1, y + row_label - 1, width + 2, height + 2), 1)

    @staticmethod
    def text(label, size, color, bold=False):
        path = pygame.font.match_font("arial,dejavusans,notosans", bold=bold)
        if not path:
            raise RuntimeError("Нужен системный шрифт с кириллицей: Arial, DejaVu Sans или Noto Sans")
        return pygame.font.Font(path, size).render(label, True, color)

    def label(self, label, position, size, color=(236, 243, 248), bold=False):
        self.canvas.blit(self.text(label, size, color, bold), position)

    def place(self, model_id, frames):
        for column, data in enumerate(frames):
            surface = pygame.image.frombytes(data, self.size, "RGB", True)
            self.canvas.blit(surface, self.positions[(model_id, column)])


def save_raw(directory, name, frames, size):
    if directory is None:
        return
    for mode, data in zip(("before", "after"), frames):
        surface = pygame.image.frombytes(data, size, "RGB", True)
        pygame.image.save(surface, str(directory / f"{name}_{mode}.png"))


def save_comparison(comparison, directory, suffix):
    pygame.image.save(comparison.canvas, str(directory / f"{STEM}_{suffix}.png"))


def encoder(ffmpeg, size, destination):
    return subprocess.Popen([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{size[0]}x{size[1]}", "-framerate", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags",
        "+faststart", str(destination),
    ], stdin=subprocess.PIPE)


def capture_closeups(ctx, catalog, args, modes):
    canvas = Comparison(CLOSE_SIZE, "Обшивка крупным планом", SHIPS)
    records = {}
    for model_id in SHIPS:
        scene = close_scene(model_id, catalog)
        capture = Capture(ctx, scene, CLOSE_SIZE)
        try:
            timings = capture.timing(scene, modes)
            frames = [capture.frame(scene, mode) for mode in modes]
            records[model_id] = {
                **pose(scene, CLOSE_SIZE), "timing_median_warm": timings,
                "mesh_sha256": digest(catalog.classic[model_id].vertices().tobytes()),
                "rgb_sha256": [digest(data) for data in frames],
            }
            save_raw(args.raw_dir, f"closeup_{model_id:02}", frames, CLOSE_SIZE)
            canvas.place(model_id, frames)
        finally:
            capture.release()
    if len(modes) == 2:
        save_comparison(canvas, args.output_dir, "closeup")
    return records


def capture_uv(ctx, catalog, args):
    size = (520, 350)
    target = np.array((0.4673, 0.0412, -0.3567))
    eye = target + 0.13 * _unit((0.43, 0.90, 0.25))
    scene = close_scene(10, catalog, (target, eye, 44.0))
    baseline = np.load(args.uv_baseline, allow_pickle=False)
    current = catalog.classic[10].vertices()
    if baseline.dtype != current.dtype or baseline.shape != current.shape or not np.isfinite(baseline).all():
        raise ValueError("Исходный меш должен иметь тот же формат, что и текущий меш Python №10")
    if not np.array_equal(baseline[:, :6], current[:, :6]) or not np.array_equal(baseline[:, 8], current[:, 8]):
        raise ValueError("Позиции, нормали и номера материалов должны совпадать")
    if np.array_equal(baseline[:, 6:8], current[:, 6:8]):
        raise ValueError("UV исходного и текущего мешей совпадают")
    frames = []
    for mesh in (baseline, current):
        with patch.object(space_mesh, "build_mesh", return_value=mesh):
            capture = Capture(ctx, scene, size)
        try:
            frames.append(capture.frame(scene, False))
        finally:
            capture.release()
    canvas = Comparison(size, "UV обшивки · Стык панелей 1/5", (10,),
                        columns=("ДО · Старые UV", "ПОСЛЕ · Непрерывные UV"))
    canvas.place(10, frames)
    save_comparison(canvas, args.output_dir, "python_uv")
    save_raw(args.raw_dir, "python_uv", frames, size)
    return {**pose(scene, size), "size": size, "refined_materials": False,
            "mesh_sha256": [digest(mesh.tobytes()) for mesh in (baseline, current)],
            "rgb_sha256": [digest(data) for data in frames]}


def capture_flyby(ctx, catalog, args, modes, ffmpeg):
    poster = Comparison(FLYBY_SIZE, "Автопролёт · Один момент", SHIPS)
    title = "Автопролёт · 0,5×"
    scene = flyby_scene(9, catalog)
    capture = Capture(ctx, scene, FLYBY_SIZE)
    process = None
    records = []
    try:
        for model_id in SHIPS:
            scene = flyby_scene(model_id, catalog)
            video = Comparison(FLYBY_SIZE, title, (model_id,))
            if ffmpeg and process is None:
                process = encoder(ffmpeg, video.canvas.get_size(), args.output_dir / f"{STEM}_flyby.mp4")
            still = FLYBY_STILLS[model_id]
            indices = range(CLIP_FRAMES) if process else (still,)
            for index in indices:
                scene.seek(flyby_time(index))
                if scene.selected != model_id or scene.auto_sources or scene.model_source != "elite":
                    raise RuntimeError("Поза AUTO FLYBY вышла за границы кадра выбранного корабля")
                frames = [capture.frame(scene, mode) for mode in modes]
                records.append({"model": model_id, "local_frame": index, "scene_seconds": scene.elapsed,
                                "selected": scene.selected, "model_source": scene.model_source,
                                "auto_sources": scene.auto_sources, **pose(scene, FLYBY_SIZE),
                                "rgb_sha256": [digest(data) for data in frames]})
                if process:
                    video.place(model_id, frames)
                    process.stdin.write(pygame.image.tobytes(video.canvas, "RGB"))
                if index == still:
                    poster.place(model_id, frames)
                    save_raw(args.raw_dir, f"flyby_{model_id:02}", frames, FLYBY_SIZE)
            print(f"Автопролёт · {SHIPS[model_id]}", flush=True)
        if process:
            process.stdin.close()
            if process.wait():
                raise RuntimeError("FFmpeg не записал ролик")
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        capture.release()
    if len(modes) == 2:
        save_comparison(poster, args.output_dir, "flyby")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "previews")
    parser.add_argument("--raw-dir", type=Path, help="Сохранить исходные PNG и манифест сравнения")
    parser.add_argument("--still-only", action="store_true", help="Пропустить запись MP4")
    parser.add_argument("--baseline-only", action="store_true", help="Снять исходную ревизию без режима ПОСЛЕ")
    parser.add_argument("--uv-baseline", type=Path, help="Отдельно сравнить UV Python №10 с исходным мешем N×9 в NPY")
    args = parser.parse_args()
    if args.baseline_only and args.raw_dir is None:
        parser.error("--baseline-only требует --raw-dir")
    if args.baseline_only and args.uv_baseline:
        parser.error("--baseline-only и --uv-baseline несовместимы")
    needs_video = not (args.still_only or args.baseline_only or args.uv_baseline)
    ffmpeg = which("ffmpeg") if needs_video else None
    if needs_video and not ffmpeg:
        parser.error("Для MP4 нужен ffmpeg или укажите --still-only")
    if not args.baseline_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_dir:
        args.raw_dir.mkdir(parents=True, exist_ok=True)

    modes = (False,) if args.baseline_only else (False, True)
    catalog = AssetCatalog()
    pygame.font.init()
    options = {"backend": "egl"} if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else {}
    ctx = moderngl.create_context(standalone=True, require=330, **options)
    # Standalone-контексту нужна живая цель между независимыми захватами.
    parking = ctx.simple_framebuffer((1, 1))
    parking.use()
    try:
        print(f"OpenGL {ctx.info['GL_VERSION']} · {ctx.info['GL_RENDERER']}", flush=True)
        metadata = {
            "gl_version": ctx.info["GL_VERSION"], "renderer": ctx.info["GL_RENDERER"],
            "msaa_samples": min(4, ctx.max_samples), "sun": SUN_DIRECTION,
            "raw_hash_format": "RGB8, bottom row first, unscaled",
            "closeup_size": CLOSE_SIZE, "flyby_size": FLYBY_SIZE, "flyby_fps": FPS,
            "flyby_speed": FLYBY_SPEED, "flyby_stills": FLYBY_STILLS,
            "variant_order": ["after" if mode else "before" for mode in modes],
            "baseline": "Штатная шероховатость и отражения на тех же мешах и UV, что и после.",
            "source_sha256": {name: digest((ROOT / "elite_lab" / name).read_bytes())
                              for name in SOURCE_FILES if (ROOT / "elite_lab" / name).is_file()},
        }
        if args.uv_baseline:
            metadata["python_uv"] = capture_uv(ctx, catalog, args)
        else:
            metadata["closeup"] = capture_closeups(ctx, catalog, args, modes)
            metadata["flyby"] = capture_flyby(ctx, catalog, args, modes, ffmpeg)
        metadata["gl_error"] = ctx.error
        if args.raw_dir:
            (args.raw_dir / "capture.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
        if metadata["gl_error"] != "GL_NO_ERROR":
            raise RuntimeError(f"OpenGL сообщил об ошибке: {metadata['gl_error']}")
    finally:
        parking.release()
        ctx.release()
        pygame.font.quit()


if __name__ == "__main__":
    main()
