#!/usr/bin/env python3
"""Снять огни собственного флота и сверить их с принятой пробой Cobra/Python."""

import argparse
import gzip
import json
from math import radians, tan
import os
from pathlib import Path
from shutil import which
import sys
from tempfile import TemporaryDirectory

from hull_material_probe import FPS, ROOT, Comparison, digest, encoder, pose, save_raw
from hull_microdetail_probe import ProbeScene, unit
from ship_lights_probe import Capture, check_gl, mesh_hashes, pair, rgb_delta, ring_difference, source_hashes, verify_native_buffers

import moderngl
import numpy as np
import pygame

from elite_lab.space_assets import AssetCatalog
from elite_lab.space_effects import ship_effects_profile
from elite_lab.space_flyby import FlybyScene
from elite_lab.space_scene import SpaceObject


OLD = {9, 10}
ACTIVE = (0, 1, 2, 4, *range(7, 20))
NEW = tuple(model for model in ACTIVE if model not in OLD)
TILE = (400, 238)
VIDEO = (960, 540)
RING_SECONDS = 1.5
PHASES = tuple(step / 4 for step in range(24))
# ID и начало пятисекундного фрагмента по штатным часам FlybyScene(9).
# Viper проходит во втором круге, когда патрульная пара обращена к камере.
CLIPS = ((8, 244), (12, 16), (14, 28), (15, 34), (18, 52), (19, 58))
CLIP_SECONDS = 5
RAW_FRAMES = "reference_rgb8.gz"
RAW_CONTROLS = "controls_rgb8.gz"
MANIFEST = "capture.json"
NAMES = {
    0: "Coriolis", 1: "Ракета", 2: "Спасательная капсула", 3: "Обломок", 4: "Грузовой контейнер",
    5: "Валун", 6: "Астероид", 7: "Rock Hermit",
    8: "Viper", 9: "Cobra Mk III", 10: "Python · торговый", 11: "Python · пиратский",
    12: "Cobra Mk III · пиратская", 13: "Asp Mk II", 14: "Fer-de-Lance", 15: "Thargoid",
    16: "Sidewinder", 17: "Krait", 18: "Thargon", 19: "Dodo",
}
# Канонические направления доков и двигателей, в том числе боковые ±X у инопланетных аппаратов.
VIEWS = {
    0: (.40, .32, 1.65), 1: (-.40, .46, -1.35), 2: (.86, .86, 1.35), 4: (.72, .93, 1.40),
    7: (.30, 1.10, 1.44),
    8: (.75, .65, -1.35), 9: (.77, .75, -1.4), 10: (.73, .76, -1.4), 11: (.80, .69, -1.4),
    12: (-.79, .72, -1.4), 13: (.81, .76, -1.4), 14: (.78, .66, -1.4), 15: (-1.8, .49, .73),
    16: (.81, .80, -1.4), 17: (.60, .70, -1.5), 18: (1.85, .43, -.68), 19: (.39, .30, 1.65),
}


def isolate(ship, texture_source="procedural", direction=None):
    if direction is None:
        direction = VIEWS.get(ship.model_id, (1.0, .75, 1.4))
    axis = unit(direction)
    right = unit(np.cross(-axis, (0, 1, 0)))
    up = np.cross(right, -axis)
    vertices = ship.vertices()[:, :3]
    x, y, z = (vertices @ vector for vector in (right, up, axis))
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    focus = cx * right + cy * up
    fov = 39.
    half_height = .84 * tan(radians(fov / 2))
    half_width = half_height * TILE[0] / TILE[1]
    depth = np.maximum(np.abs(x - cx) / half_width, np.abs(y - cy) / half_height)
    eye = focus + max(1.15, float(np.max(z + depth))) * axis
    obj = SpaceObject(ship.model_id, np.zeros(3), 1., np.eye(4))
    scene = ProbeScene((ship,), (obj,), eye, focus, fov, model_source=ship.source_id, texture_source=texture_source)
    scene.animation_elapsed = 0.
    return scene


def audit_pair(capture, scene, key, previous=None, *, exact_on=False, read_shadow=False,
               previous_off=None, rounding_limit=0):
    frames, audit = pair(capture, scene, False, read_shadow=read_shadow)
    if previous is not None:
        if any(value != previous["before"][name] for name, value in audit["before"].items() if name != "rgb_sha256"):
            raise RuntimeError(f"{key} · Изменились поза, свет или тень")
        if audit["before"]["rgb_sha256"] != previous["before"]["rgb_sha256"]:
            if previous_off is None:
                raise RuntimeError(f"{key} · Изменился кадр без эффектов")
            delta = rgb_delta(previous_off, frames[0])
            audit["off_rounding"] = {"max_rgb8_step": int(delta.max()), "pixels": int(np.count_nonzero(delta))}
            if audit["off_rounding"]["max_rgb8_step"] > rounding_limit:
                raise RuntimeError(f"{key} · Кадр без эффектов отличается: {audit['off_rounding']}")
        if exact_on and audit["after"] != previous["after"]:
            raise RuntimeError(f"{key} · Изменился кадр принятого или пассивного профиля")
    return frames, audit


def isolated(capture, catalog, profiles, previous, audit_dir):
    records, tiles = {}, {}
    for ship in catalog.classic:
        model = ship.model_id
        key = str(model)
        scene = isolate(ship)
        capture.load_fleet(scene)
        camera = pose(scene, TILE)
        old = previous[key] if previous else None
        if old is not None and camera != old["camera"]:
            raise RuntimeError(f"№{model:02d} · Изменился ракурс в контактной таблице")
        samples, best = [], None
        for index, seconds in enumerate(PHASES if model in ACTIVE else (0.,)):
            scene.animation_elapsed = seconds
            frames, audit = audit_pair(capture, scene, f"№{model:02d} · {seconds:.2f} с",
                                       old["samples"][index] if old else None, exact_on=model not in NEW,
                                       read_shadow=index == 0)
            samples.append({"seconds": seconds, **audit})
            if audit["before"]["rgb_sha256"] != samples[0]["before"]["rgb_sha256"]:
                raise RuntimeError(f"№{model:02d} · При смене фазы изменился кадр с выключенными эффектами")
            if model not in profiles and frames[0] != frames[1]:
                raise RuntimeError(f"№{model:02d} · Пассивный объект получил новые эффекты")
            diff = rgb_delta(*frames)
            score = int(diff[diff > 3].sum())
            if best is None or score > best[0]:
                best = score, seconds, frames
        score, seconds, frames = best
        distinct = len({sample["after"]["rgb_sha256"] for sample in samples})
        if model in profiles and (score == 0 or distinct < 3):
            raise RuntimeError(f"№{model:02d} · С этого ракурса свечение и пульсация не проявились")
        records[key] = {"camera": camera, "samples": samples, "distinct_after": distinct,
                        "poster_seconds": seconds, "poster_effect_score": score}
        tiles[model] = frames[1]
        save_raw(audit_dir, f"isolate_{model:02d}", frames, TILE)
        print(f"Контактный кадр · №{model:02d} · {ship.name} · {distinct} фаз", flush=True)
    return records, tiles


def read_reference(stream, size, expected, label):
    raw = stream.read(size[0] * size[1] * 3)
    if digest(raw) != expected:
        raise RuntimeError(f"{label} · Поток исходных кадров повреждён или снят с другими настройками")
    return raw


def unaffected(capture, catalog, previous, stream):
    cases = {}
    for pack in catalog.pack_ids:
        for ship in catalog.pack(pack).values():
            for textures in ("author", "procedural"):
                cases[f"{pack}:{ship.model_id}:{textures}"] = isolate(ship, textures)
    zx = FlybyScene(9, model_source="zx", catalog=catalog, source_interval=0)
    cases["zx:9"] = zx
    records = {}
    for key, scene in cases.items():
        capture.load_fleet(scene)
        if any(renderer._effect_profiles for renderer in capture.renderers.values()):
            raise RuntimeError(f"{key} · Профиль огней назначен контрольной модели")
        old = previous[key] if previous else None
        old_raw = read_reference(stream, TILE, old["before"]["rgb_sha256"], key) if old else None
        # При новой компиляции GLSL отдельные каналы процедурной импортной
        # обшивки округляются иначе. B в пределах каждой версии совпадает точно.
        limit = 2 if key.endswith(":procedural") else 0
        frames, records[key] = audit_pair(capture, scene, key, old, previous_off=old_raw, rounding_limit=limit)
        if frames[0] != frames[1]:
            delta = rgb_delta(*frames)
            rounding = {"max_rgb8_step": int(delta.max()), "pixels": int(np.count_nonzero(delta))}
            if rounding["max_rgb8_step"] > limit:
                raise RuntimeError(f"{key} · Контрольный кадр изменился при переключении B: {rounding}")
            records[key]["toggle_rounding"] = rounding
        if not previous:
            stream.write(frames[0])
    print(f"Импорт и ZX · {len(records)} контрольных кадров", flush=True)
    return records


class Gallery:
    def __init__(self):
        width, height = TILE
        margin, gap, top, label = 24, 14, 80, 35
        columns = 5
        self.canvas = pygame.Surface((2 * margin + columns * width + (columns - 1) * gap, top + 4 * (height + label + gap)))
        self.canvas.fill((12, 20, 31))
        self.label("Огни собственного флота", (margin, 15), 32, bold=True)
        self.label("17 световых профилей · обломок, валун и астероид для сравнения", (margin, 53), 18, color=(151, 183, 203))
        self.positions = {}
        for model in NAMES:
            row, col = divmod(model, columns)
            x = margin + col * (width + gap)
            y = top + row * (height + label + gap)
            pygame.draw.rect(self.canvas, (27, 46, 62), (x - 1, y + label - 1, width + 2, height + 2), 1)
            accent = (113, 219, 218) if model in ACTIVE else (151, 183, 203)
            self.label(f"№{model:02d}", (x, y + 8), 18, color=accent, bold=True)
            self.label(NAMES[model], (x + 51, y + 7), 20)
            self.positions[model] = x, y + label

    def label(self, name, xy, size, *, color=(236, 243, 248), bold=False):
        self.canvas.blit(Comparison.text(name, size, color, bold), xy)

    def place(self, tiles):
        for model, raw in tiles.items():
            self.canvas.blit(pygame.image.frombytes(raw, TILE, "RGB", True), self.positions[model])


class ClipLayout:
    def __init__(self, model):
        width, height = VIDEO
        margin, gap, top = 24, 14, 112
        self.canvas = pygame.Surface((width * 2 + gap + margin * 2, top + height + 14))
        self.canvas.fill((12, 20, 31))
        self.positions = [(margin, top), (margin + width + gap, top)]
        self.canvas.blit(Comparison.text(f"{NAMES[model]} · №{model:02d}", 28, (236, 243, 248), True), (margin, 14))
        note = Comparison.text("Штатный автопролёт · 1×", 20, (151, 183, 203))
        self.canvas.blit(note, (self.canvas.get_width() - margin - note.get_width(), 21))
        titles = (("До · Огни Cobra и торгового Python", (162, 181, 201)),
                  ("После · Огни флота", (144, 222, 226)))
        for (title, color), (x, _) in zip(titles, self.positions, strict=True):
            self.canvas.blit(Comparison.text(title, 21, color, True), (x, 72))
            pygame.draw.rect(self.canvas, color, (x, 103, width, 2))

    def place(self, before, after):
        for raw, xy in zip((before, after), self.positions, strict=True):
            self.canvas.blit(pygame.image.frombytes(raw, VIDEO, "RGB", True), xy)
        return pygame.image.tobytes(self.canvas, "RGB")


def route_frame(capture, scene, index, old, stream, label):
    scene.seek(index / FPS)
    old_off = read_reference(stream, VIDEO, old["before"]["rgb_sha256"], label) if old else None
    frames, record = audit_pair(capture, scene, label, old, previous_off=old_off)
    record["seconds"] = scene.elapsed
    if old is None:
        stream.write(frames[0])
        stream.write(frames[1])
        return record, None, frames[1]
    original = read_reference(stream, VIDEO, old["after"]["rgb_sha256"], label)
    region = ring_difference(scene, (original, frames[1]), size=VIDEO, model_ids=NEW)
    if region["outside_max_rgb8_step"] > 1:
        raise RuntimeError(f"{label} · Разница вышла за область новых огней: {region}")
    record["difference"] = region
    return record, original, frames[1]


def route(capture, scene, old, stream, process, still_only):
    records = {"ring": {}, "flyby": {}}
    step = round(RING_SECONDS * FPS)
    for index in range(0, round(scene.duration * FPS), step):
        key = str(index)
        previous = old["ring"][key] if old else None
        records["ring"][key], _, _ = route_frame(capture, scene, index, previous, stream, f"Кольцо · {index/FPS:.2f} с")
    print(f"Кольцо · {len(records['ring'])} временных среза", flush=True)
    if still_only:
        return records
    for model, start in CLIPS:
        layout = ClipLayout(model) if process else None
        for index in range(start * FPS, (start + CLIP_SECONDS) * FPS):
            key = str(index)
            previous = old["flyby"][key] if old else None
            record, before, after = route_frame(capture, scene, index, previous, stream, f"Автопролёт · {index/FPS:.2f} с")
            if scene.selected != model:
                raise RuntimeError(f"№{model:02d} · Камера вышла за пределы участка выбранного корабля")
            records["flyby"][key] = {"model": model, **record}
            if process:
                process.stdin.write(layout.place(before, after))
        print(f"Автопролёт · {NAMES[model]} · {CLIP_SECONDS} с", flush=True)
    return records


def capture_all(ctx, parking, args, ffmpeg):
    catalog = AssetCatalog()
    profiles = {ship.model_id for ship in catalog.classic if ship_effects_profile(ship)}
    expected = OLD if args.reference_only else set(ACTIVE)
    if profiles != expected:
        raise RuntimeError(f"Ожидались световые профили {sorted(expected)}, обнаружены {sorted(profiles)}")
    reference = None if args.reference_only else json.loads((args.reference_dir / MANIFEST).read_text(encoding="utf-8"))
    if reference and set(reference["profile_ids"]) != OLD:
        raise RuntimeError("Исходная проба должна содержать только Cobra №09 и Python №10")
    settings = {"renderer": ctx.info["GL_RENDERER"], "gl_version": ctx.info["GL_VERSION"],
                "samples": min(4, ctx.max_samples), "tile_size": list(TILE), "video_size": list(VIDEO),
                "fps": FPS, "rgb_hash_format": "RGB8 bottom-up"}
    native_mesh = mesh_hashes(catalog.classic)
    imported_mesh = mesh_hashes(ship for pack in catalog.pack_ids for ship in catalog.pack(pack).values())
    records = {"settings": settings, "source_sha256": source_hashes(), "profile_ids": sorted(profiles),
               "native_mesh_sha256": native_mesh, "imported_mesh_sha256": imported_mesh}
    if reference:
        for key in ("settings", "native_mesh_sha256", "imported_mesh_sha256"):
            if records[key] != reference[key]:
                raise RuntimeError(f"Изменились настройки GPU или исходные меши · {key}")
    stream_dir = args.audit_dir if args.reference_only else args.reference_dir
    mode = "wb" if args.reference_only else "rb"
    capture = Capture(ctx, parking, isolate(catalog.classic[0]), False, size=TILE)
    try:
        records["isolated"], tiles = isolated(capture, catalog, profiles, reference and reference["isolated"], args.audit_dir)
        with gzip.open(stream_dir / RAW_CONTROLS, mode, compresslevel=1) as stream:
            records["unaffected"] = unaffected(capture, catalog, reference and reference["unaffected"], stream)
    finally:
        capture.release()
    scene = FlybyScene(9, model_source="elite", texture_source="procedural", catalog=catalog, source_interval=0)
    capture = Capture(ctx, parking, scene, False, size=VIDEO)
    try:
        verify_native_buffers(capture, scene.ships, native_mesh)
        records["native_gpu_mesh_matches"] = True
        with TemporaryDirectory(dir=args.output_dir if not args.reference_only else None) as staging:
            staging = Path(staging)
            process = encoder(ffmpeg, ClipLayout(CLIPS[0][0]).canvas.get_size(), staging / "fleet_lights_flyby.mp4") if ffmpeg else None
            try:
                with gzip.open(stream_dir / RAW_FRAMES, mode, compresslevel=1) as stream:
                    records.update(route(capture, scene, reference, stream, process, args.still_only))
                if process:
                    process.stdin.close()
                    if process.wait():
                        raise RuntimeError("FFmpeg не записал ролик")
                records["gl_error"] = check_gl(ctx, "Захват флота")
                if reference:
                    records["reference_source_sha256"] = reference["source_sha256"]
                    gallery = Gallery()
                    gallery.place(tiles)
                    pygame.image.save(gallery.canvas, str(staging / "fleet_lights.png"))
                    for path in staging.iterdir():
                        path.replace(args.output_dir / path.name)
                (args.audit_dir / MANIFEST).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            finally:
                if process and process.poll() is None:
                    process.kill()
                    process.wait()
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True, help="Каталог кадров и отчёта с хешами")
    parser.add_argument("--reference-dir", type=Path, help="Каталог захвата предыдущей пробы Cobra/Python")
    parser.add_argument("--reference-only", action="store_true", help="Снять предыдущую пробу в её исходном окружении")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "previews", help="Каталог итоговых PNG и MP4")
    parser.add_argument("--still-only", action="store_true", help="Снять контактную таблицу и проверить временные срезы кольца")
    args = parser.parse_args()
    if args.reference_only == bool(args.reference_dir):
        parser.error("Укажите --reference-only или --reference-dir")
    ffmpeg = None if args.reference_only or args.still_only else which("ffmpeg")
    if not ffmpeg and not (args.reference_only or args.still_only):
        parser.error("Для MP4 нужен ffmpeg или укажите --still-only")
    if args.reference_dir and args.audit_dir.resolve() == args.reference_dir.resolve():
        parser.error("Разнесите исходный и текущий захваты по разным каталогам")
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    if not args.reference_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    pygame.font.init()
    options = {"backend": "egl"} if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else {}
    ctx = moderngl.create_context(standalone=True, require=330, **options)
    parking = ctx.simple_framebuffer((1, 1))
    parking.use()
    try:
        print(f"OpenGL {ctx.info['GL_VERSION']} · {ctx.info['GL_RENDERER']}", flush=True)
        capture_all(ctx, parking, args, ffmpeg)
    finally:
        parking.release()
        ctx.release()
        pygame.font.quit()


if __name__ == "__main__":
    main()
