#!/usr/bin/env python3
"""Снять A/B огней Cobra №09 и торгового Python №10 в штатном автопролёте."""

import argparse
import json
import os
from pathlib import Path
from shutil import which
from statistics import median
import sys
from time import perf_counter

from hull_material_probe import FPS, ROOT, Comparison, digest, encoder, pose, save_raw
from hull_microdetail_probe import ProbeScene

import moderngl
import numpy as np
import pygame

from elite_lab.space_assets import AssetCatalog
from elite_lab.space_flyby import FlybyScene
from elite_lab.space_scene import SpaceObject
from elite_lab.space_viewer import SpaceRenderer


STEM = "cobra_python_lights_flyby"
BASELINE_REVISION = "e7190d84d54adeb33076bea71910f1b90c622637"
SIZE = (960, 540)
START_MODEL = 8
SHIPS = {9: "Cobra Mk III · №09", 10: "Торговый Python · №10"}
CLIPS = {9: range(3 * FPS, 9 * FPS), 10: range(9 * FPS, 15 * FPS)}
STILLS = {9: 219, 10: 448}
BLOOM_MARGIN = 24
FREEZE_FPS, FREEZE_SECONDS = 12, 6


def check_gl(ctx, where):
    error = ctx.error
    if error != "GL_NO_ERROR":
        raise RuntimeError(f"{where} · {error}")
    return error


class Capture:
    def __init__(self, ctx, parking, scene, baseline_only, *, size=SIZE):
        self.ctx, self.parking, self.size = ctx, parking, size
        # Независимые буферы хранят прогретое состояние обеих веток.
        modes = (False,) if baseline_only else (False, True)
        self.renderers = {mode: SpaceRenderer(ctx, scene) for mode in modes}
        self.target = ctx.simple_framebuffer(size, samples=min(4, ctx.max_samples))
        self.resolved = ctx.simple_framebuffer(size)

    def load_fleet(self, scene):
        # Изолированные сцены меняют состав флота, даже когда источник тот же.
        for renderer in self.renderers.values():
            renderer._load_fleet(scene)

    def draw(self, scene, enabled):
        scene.effects_enabled = enabled
        self.renderers[enabled].draw(scene, self.target, self.size, False)

    def frame(self, scene, enabled, *, read_shadow=False):
        self.draw(scene, enabled)
        self.ctx.copy_framebuffer(self.resolved, self.target)
        raw = self.resolved.read(components=3, alignment=1)
        info = {"rgb_sha256": digest(raw)}
        if scene.model_source != "zx":
            renderer = self.renderers[enabled]
            hull = renderer.look.hull
            info["pose_uniforms_sha256"] = digest(b"".join(
                hull[key].read() for key in ("u_view_proj", "u_model", "u_eye")))
            info["light_uniforms_sha256"] = digest(b"".join(
                hull[key].read() for key in ("u_sun_direction", "u_shadow_vp", "u_shadow_enabled")))
            light_vp, transforms, lods = renderer._shadow_signature
            info["shadow_inputs_sha256"] = digest(light_vp + b"".join(transforms) + bytes(lods))
            if read_shadow:
                info["shadow_depth_sha256"] = digest(renderer.shadow.read(alignment=1))
        info["gl_error"] = check_gl(self.ctx, f"Кадр · {scene.model_source} · effects={enabled}")
        return raw, info

    def release(self):
        self.parking.use()
        self.resolved.release()
        self.target.release()
        for renderer in self.renderers.values():
            renderer.release()


class Layout:
    def __init__(self, models):
        width, height = SIZE
        margin, gap, header, row_label = 24, 14, 88, 36
        row = row_label + height + 14
        self.canvas = pygame.Surface((width * 2 + gap + margin * 2, header + len(models) * row))
        self.canvas.fill((12, 20, 31))
        self.positions = {}
        self.label("Огни кораблей в автопролёте", (margin, 14), 28, bold=True)
        note = "Одна камера и свет · 1×" if len(models) == 1 else "Одна камера и свет · кадры 1:1"
        info = Comparison.text(note, 19, (162, 181, 201))
        self.canvas.blit(info, (self.canvas.get_width() - margin - info.get_width(), 22))
        columns = (("До", (162, 181, 201)), ("После", (144, 222, 226)))
        for column, (heading, color) in enumerate(columns):
            x = margin + column * (width + gap)
            self.label(heading, (x, 54), 21, color, bold=True)
            pygame.draw.rect(self.canvas, color, (x, 82, width, 2))
            for index, model in enumerate(models):
                y = header + index * row
                self.label(SHIPS[model], (x, y + 7), 20)
                pygame.draw.rect(self.canvas, (40, 55, 70), (x - 1, y + row_label - 1, width + 2, height + 2), 1)
                self.positions[model, column] = (x, y + row_label)

    def label(self, label, position, size, color=(236, 243, 248), bold=False):
        self.canvas.blit(Comparison.text(label, size, color, bold), position)

    def place(self, model, frames):
        for column, raw in enumerate(frames):
            surface = pygame.image.frombytes(raw, SIZE, "RGB", True)
            self.canvas.blit(surface, self.positions[model, column])


def pair(capture, scene, baseline_only, *, read_shadow=False):
    before, base = capture.frame(scene, False, read_shadow=read_shadow)
    if baseline_only:
        return (before,), {"before": base}
    after, effects = capture.frame(scene, True, read_shadow=read_shadow)
    if any(effects[key] != value for key, value in base.items() if key != "rgb_sha256"):
        raise RuntimeError("Между до и после изменились поза, свет или карта теней")
    return (before, after), {"before": base, "after": effects}


def rgb_delta(first, second):
    a = np.frombuffer(first, dtype="u1").astype("i2")
    b = np.frombuffer(second, dtype="u1")
    return np.abs(a - b).reshape(-1, 3).max(axis=1)


def ring_difference(scene, frames, *, size=SIZE, model_ids=SHIPS):
    """Сравнить кольцо за границами освещённых корпусов с полем для ореола."""
    width, height = size
    delta = rgb_delta(*frames).reshape(height, width)
    changed_total = int(np.count_nonzero(delta))
    vp = scene.view_projection(width / height)
    for model in model_ids:
        vertices = scene.ships[model].vertices()[:, :3]
        homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
        clip = homogeneous @ (vp @ scene.objects[model].matrix).T
        if np.any(clip[:, 3] <= 0):
            raise RuntimeError("Корабль со свечением пересёк плоскость камеры")
        # В RGB-буфере OpenGL нижняя строка записана первой.
        pixels = (clip[:, :2] / clip[:, 3, None] + 1) * np.asarray(size) * .5
        lo = np.clip(np.floor(pixels.min(axis=0)).astype(int) - BLOOM_MARGIN, 0, size)
        hi = np.clip(np.ceil(pixels.max(axis=0)).astype(int) + BLOOM_MARGIN, 0, size)
        delta[lo[1]:hi[1], lo[0]:hi[0]] = 0
    return {"changed_pixels": changed_total, "outside_target_bounds": int(np.count_nonzero(delta)),
            "outside_max_rgb8_step": int(delta.max())}


def flyby(capture, scene, args, poster, process):
    records = {}
    for model, clip in CLIPS.items():
        video = Layout((model,)) if process else None
        for index in (clip if not args.still_only else (STILLS[model],)):
            scene.seek(index / FPS)
            if scene.selected != model or scene.auto_sources or scene.model_source != "elite":
                raise RuntimeError("Автопролёт вышел за участок выбранного корабля")
            still = index == STILLS[model]
            frames, audit = pair(capture, scene, args.baseline_only, read_shadow=still)
            record = {"model": model, "seconds": scene.elapsed, **audit}
            records[str(index)] = record
            if not args.baseline_only:
                region = ring_difference(scene, frames)
                record["difference"] = region
                if region["outside_max_rgb8_step"] > 1:
                    raise RuntimeError(f"{scene.elapsed:.3f} с · Флот вне ореола изменился больше чем на 1 уровень RGB8")
            if video:
                video.place(model, frames)
                process.stdin.write(pygame.image.tobytes(video.canvas, "RGB"))
            if still:
                record["camera"] = pose(scene, SIZE)
                if poster:
                    poster.place(model, frames)
                save_raw(args.audit_dir, f"flyby_{model:02}", frames, SIZE)
        print(f"Автопролёт · {SHIPS[model]}", flush=True)
    return records


def fixed_pose(capture, scene, baseline_only):
    records = {}
    for model, index in STILLS.items():
        scene.seek(index / FPS)
        original_time = scene.animation_elapsed
        off0, info0 = capture.frame(scene, False, read_shadow=True)
        records[str(model)] = {"camera": pose(scene, SIZE), "before": info0, "samples": []}
        if baseline_only:
            continue
        on0, _ = capture.frame(scene, True)
        first = np.frombuffer(on0, dtype="u1").reshape(-1, 3)
        sample_count = FREEZE_FPS * FREEZE_SECONDS
        for sample in range(sample_count):
            clock = original_time + sample / FREEZE_FPS
            scene.animation_elapsed = clock
            check_depth = sample == sample_count - 1
            _, off = capture.frame(scene, False, read_shadow=check_depth)
            on, after = capture.frame(scene, True, read_shadow=check_depth)
            if any(info0[key] != value for key, value in off.items()) or any(
                    info0[key] != value for key, value in after.items() if key != "rgb_sha256"):
                raise RuntimeError(f"№{model:02} · При изменении времени изменились исходный кадр, поза, свет или тень")
            if on == off0:
                raise RuntimeError(f"№{model:02} · В контрольном кадре эффект не виден")
            changed = np.count_nonzero(np.any(np.frombuffer(on, dtype="u1").reshape(-1, 3) != first, axis=1))
            records[str(model)]["samples"].append({
                "seconds": clock, "rgb_sha256": after["rgb_sha256"], "pixels_changed_from_first": int(changed),
            })
        if len({sample["rgb_sha256"] for sample in records[str(model)]["samples"]}) < 2:
            raise RuntimeError(f"№{model:02} · Эффект не меняется при остановленной геометрии")
        scene.animation_elapsed = original_time
    return records


def mesh_hashes(ships):
    return {f"{ship.source_id}:{ship.model_id}": [digest(ship.vertices(lod).tobytes())
            for lod in range(len(ship.meshes) or 1)] for ship in ships}


def verify_native_buffers(capture, ships, expected):
    for renderer in capture.renderers.values():
        for ship, levels in zip(ships, renderer.meshes, strict=True):
            key = f"{ship.source_id}:{ship.model_id}"
            if [digest(level.buffer.read()) for level in levels] != expected[key]:
                raise RuntimeError(f"{key} · Сетка в видеопамяти отличается от исходных вершин")


def unaffected(capture, catalog, baseline_only):
    cases = []
    for model in SHIPS:
        scene = FlybyScene(START_MODEL, model_source="zx", catalog=catalog, source_interval=0)
        scene.seek(STILLS[model] / FPS)
        cases.append((f"zx:{model}", scene))
    ships = [ship for ship in catalog.classic if ship.model_id not in SHIPS]
    ships.extend(ship for pack in catalog.pack_ids for model, ship in catalog.pack(pack).items() if model in SHIPS)
    for ship in ships:
        obj = SpaceObject(ship.model_id, np.zeros(3), 1., np.eye(4))
        for textures in (("author", "procedural") if ship.meshes else ("procedural",)):
            scene = ProbeScene((ship,), (obj,), np.array((1.4, 1.15, -2.05)), np.zeros(3),
                               model_source=ship.source_id, texture_source=textures)
            scene.animation_elapsed = 1.27
            cases.append((f"{ship.source_id}:{ship.model_id}:{textures}", scene))
    records, targets = {}, []
    for key, scene in cases:
        capture.load_fleet(scene)
        frames, records[key] = pair(capture, scene, baseline_only)
        if baseline_only:
            continue
        if capture.renderers[True]._effect_profiles:
            # Новые собственные цели всё ещё участвуют в сверке старого кадра
            # с выключенными эффектами. Их режим «включено» проверяет весь флот.
            targets.append(key)
        elif frames[0] != frames[1]:
            delta = rgb_delta(*frames)
            if not (scene.model_source in catalog.pack_ids and scene.texture_source == "procedural" and delta.max() <= 2):
                raise RuntimeError(f"{key} · Эффект изменил контрольный кадр")
            records[key]["toggle_rounding"] = {"max_rgb8_step": int(delta.max()), "pixels": int(np.count_nonzero(delta))}
    print(f"Контроль выключенного режима · {len(cases)} кадров, новые цели · {len(targets)}", flush=True)
    return {"frames": records, "mesh_sha256": mesh_hashes(ships), "new_targets": targets}


def benchmark(capture, scene, baseline_only):
    modes = (False,) if baseline_only else (False, True)
    records = {}
    for model, index in STILLS.items():
        samples = {mode: [] for mode in modes}
        for step in range(30):
            # Обе ветки рисуют одну позу, каждый раз отличную от предыдущей.
            scene.seek((index + step % 2) / FPS)
            for mode in modes if step % 2 == 0 else reversed(modes):
                start = perf_counter()
                capture.draw(scene, mode)
                capture.ctx.finish()
                if step >= 6:
                    samples[mode].append((perf_counter() - start) * 1000)
        records[str(model)] = {"after_ms" if mode else "before_ms": median(samples[mode]) for mode in modes}
    check_gl(capture.ctx, "Прогретый замер")
    return records


def validate_baseline(baseline, current):
    for key in ("settings", "native_mesh_sha256"):
        if baseline[key] != current[key]:
            raise RuntimeError(f"Исходный и текущий захваты не совпали · {key}")
    if baseline["unaffected"]["mesh_sha256"] != current["unaffected"]["mesh_sha256"]:
        raise RuntimeError("Меши контрольных моделей отличаются от исходной ревизии")
    sections = ((baseline["flyby"], current["flyby"]), (baseline["fixed_pose"], current["fixed_pose"]),
                (baseline["unaffected"]["frames"], current["unaffected"]["frames"]))
    for old, new in sections:
        for key, info in new.items():
            if any(info[name] != old[key][name] for name in ("before", "camera") if name in info):
                raise RuntimeError(f"Кадр с выключенным эффектом отличается от исходной ревизии · {key}")
    return {"revision": BASELINE_REVISION, "flyby_frames_exact": len(current["flyby"]),
            "fixed_poses_exact": len(current["fixed_pose"]),
            "unaffected_frames_exact": len(current["unaffected"]["frames"]),
            "source_sha256": baseline["source_sha256"]}


def source_hashes():
    sources = [path for path in (ROOT / "elite_lab").rglob("*") if path.suffix in {".py", ".vert", ".frag", ".glsl"}]
    sources.extend((ROOT / "reference_data").glob("*.bin"))
    return {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sorted(sources)}


def run_capture(ctx, parking, catalog, args, ffmpeg):
    scene = FlybyScene(START_MODEL, model_source="elite", texture_source="procedural", catalog=catalog, source_interval=0)
    original = not hasattr(scene, "effects_enabled")
    if args.baseline_only != original:
        raise RuntimeError("Для --baseline-only используйте исходную ревизию, для A/B — ревизию с эффектами")
    capture = Capture(ctx, parking, scene, args.baseline_only)
    if not args.baseline_only:
        # Повторяем историческое сравнение с включёнными огнями №09 и №10.
        renderer = capture.renderers[True]
        renderer._effect_profiles = {index: profile for index, profile in renderer._effect_profiles.items()
                                     if scene.objects[index].model_id in SHIPS}
    poster = None if args.baseline_only else Layout(SHIPS)
    pending_video = args.output_dir / f".{STEM}-{os.getpid()}.mp4"
    process = None
    try:
        if ffmpeg:
            process = encoder(ffmpeg, Layout((9,)).canvas.get_size(), pending_video)
        settings = {"renderer": ctx.info["GL_RENDERER"], "gl_version": ctx.info["GL_VERSION"],
                    "msaa_samples": min(4, ctx.max_samples), "size": list(SIZE), "fps": FPS, "start_model": START_MODEL,
                    "rgb_hash_format": "RGB8 bottom-up, unscaled"}
        native = mesh_hashes(scene.ships)
        verify_native_buffers(capture, scene.ships, native)
        records = {"settings": settings, "source_sha256": source_hashes(),
                   "native_mesh_sha256": native, "native_gpu_mesh_matches": True,
                   "flyby": flyby(capture, scene, args, poster, process)}
        if process:
            process.stdin.close()
            if process.wait():
                raise RuntimeError("FFmpeg не записал ролик")
        records["fixed_pose"] = fixed_pose(capture, scene, args.baseline_only)
        records["benchmark_draw_finish_warm_median"] = benchmark(capture, scene, args.baseline_only)
        records["unaffected"] = unaffected(capture, catalog, args.baseline_only)
        records["gl_error"] = check_gl(ctx, "Захват")
        if args.baseline:
            records["baseline_match"] = validate_baseline(json.loads(args.baseline.read_text(encoding="utf-8")), records)
        if args.audit_dir:
            (args.audit_dir / "capture.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        if poster:
            pygame.image.save(poster.canvas, str(args.output_dir / f"{STEM}.png"))
        if process:
            pending_video.replace(args.output_dir / f"{STEM}.mp4")
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        if process:
            pending_video.unlink(missing_ok=True)
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "previews", help="Каталог PNG и MP4")
    parser.add_argument("--audit-dir", type=Path, help="Сохранить исходные PNG и JSON с результатами проверки")
    parser.add_argument("--baseline-only", action="store_true", help="Снять кадры в исходной ревизии")
    parser.add_argument("--baseline", type=Path, help="Сверить кадры без эффекта с исходным capture.json")
    parser.add_argument("--still-only", action="store_true", help="Снять только ключевые кадры, без MP4")
    args = parser.parse_args()
    if args.baseline_only and (not args.audit_dir or args.baseline):
        parser.error("--baseline-only требует --audit-dir и несовместим с --baseline")
    needs_video = not (args.baseline_only or args.still_only)
    ffmpeg = which("ffmpeg") if needs_video else None
    if needs_video and not ffmpeg:
        parser.error("Для MP4 нужен ffmpeg. Для PNG укажите --still-only")
    if not args.baseline_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
    pygame.font.init()
    options = {"backend": "egl"} if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else {}
    ctx = moderngl.create_context(standalone=True, require=330, **options)
    parking = ctx.simple_framebuffer((1, 1))
    parking.use()
    try:
        print(f"OpenGL {ctx.info['GL_VERSION']} · {ctx.info['GL_RENDERER']}", flush=True)
        run_capture(ctx, parking, AssetCatalog(), args, ffmpeg)
    finally:
        parking.release()
        ctx.release()
        pygame.font.quit()


if __name__ == "__main__":
    main()
