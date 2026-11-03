"""Дифференциальная проверка Python-переноса по оригинальным инструкциям Z80.

Запуск из каталога проекта:
    python3 tools/verify_reference.py
    python3 tools/verify_reference.py --lines 1000 --poses 128 --output validation.json

Требуется дополнительная зависимость skoolkit==10.1. Pygame не требуется.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from elite_lab.controls import pose_from_angles
from elite_lab.model import COBRA_ID, DODO_ID, MODELS, minimum_depth
from elite_lab.raster import apply_trace, trace_line
from elite_lab.reference import reference_geometry, reference_line
from elite_lab.renderer import render_geometry
from elite_lab.types import Pose, ProjectedVertex


def rotation_pose(yaw: float, pitch: float, roll: float, depth: int) -> Pose:
    """Подготовка входа теста; тот же готовый квантованный базис идёт обоим кодам."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    # R = Rz(roll) * Ry(yaw) * Rx(pitch), берём столбцы nose и roof.
    nose = (cr * sy * cp + sr * sp, sr * sy * cp - cr * sp, cy * cp)
    roof = (cr * sy * sp - sr * cp, sr * sy * sp + cr * cp, cy * sp)
    return Pose((0, 0, depth),
                tuple(round(127 * value) for value in nose),
                tuple(round(127 * value) for value in roof))


def line_cases(random_count: int, rng: random.Random):
    # Цели проверки: вырожденный отрезок, все октанты, половины delta,
    # переходы между байтами/строками, крайние разрешённые модули, signed zero.
    coordinates = [(0, 0), (0, 1), (0, 23), (1, 0), (17, 0), (1, 1), (4, 5), (127, 63)]
    for (x0, y0), (x1, y1), flags0, flags1 in itertools.product(
            coordinates, coordinates, range(4), range(4)):
        yield ProjectedVertex(x0, y0, flags0), ProjectedVertex(x1, y1, flags1)
    for dx, dy in [(4, 5), (5, 4), (1, 2), (2, 1), (1, 1), (0, 0), (0, 20), (20, 0)]:
        for sx, sy in itertools.product((-1, 1), repeat=2):
            yield ProjectedVertex.from_screen(127, 63), ProjectedVertex.from_screen(
                127 + sx * dx, 63 + sy * dy)
    for _ in range(random_count):
        yield (ProjectedVertex(rng.randrange(128), rng.randrange(64), rng.randrange(4)),
               ProjectedVertex(rng.randrange(128), rng.randrange(64), rng.randrange(4)))


def clipping_cases(random_count: int, rng: random.Random):
    """Все пары внешних областей, знаки нуля, равенство наклонов, $FF и перелив адреса."""
    magnitudes = [(0, 0), (127, 63), (128, 63), (0, 64), (127, 64), (128, 64),
                  (129, 63), (129, 65), (255, 64), (128, 255), (255, 255)]
    for a, b, flags_a, flags_b in itertools.product(magnitudes, magnitudes, range(4), range(4)):
        if a[0] >= 128 or b[0] >= 128 or a[1] >= 64 or b[1] >= 64:
            yield ProjectedVertex(*a, flags_a), ProjectedVertex(*b, flags_b)
    for _ in range(random_count):
        # Первый конец точно внешний, второй произвольный. Проверяем оба порядка.
        a = ProjectedVertex(rng.randrange(128, 256), rng.randrange(256), rng.randrange(4))
        b = ProjectedVertex(rng.randrange(256), rng.randrange(256), rng.randrange(4))
        yield a, b
        yield b, a


def close_model_cases(model_id: int, random_count: int, rng: random.Random):
    nearest = minimum_depth(model_id)
    depths = {nearest, nearest + 1, 64, 96, 128, 192, 255, 256, 257, 383, 384, 511, 512, 640, 767}
    for depth in sorted(d for d in depths if d >= nearest):
        yield pose_from_angles(24, 22, 0, depth)
        yield pose_from_angles(0, 90, 0, depth)
    for _ in range(random_count):
        yield rotation_pose(*(rng.uniform(-math.pi, math.pi) for _ in range(3)), rng.randrange(nearest, 768))


def geometry_cases(random_count: int, rng: random.Random):
    # Пороги LOD и обе стороны корабля, включая дальние сокращённые каркасы.
    depths = [768, 769, 895, 896, 1023, 1024, 1151, 1152, 2559, 2560, 3967, 3968, 6000]
    for depth in depths:
        for yaw, pitch, roll in [(0, 0, 0), (math.pi, 0, 0),
                                 (0.7, 0.4, 0.2), (2.1, -0.8, 1.3)]:
            yield rotation_pose(yaw, pitch, roll, depth)
    for _ in range(random_count):
        yield rotation_pose(*(rng.uniform(-math.pi, math.pi) for _ in range(3)),
                            rng.randrange(768, 6001))


def verify_line(start: ProjectedVertex, end: ProjectedVertex) -> int:
    actual = trace_line(start, end)
    expected = reference_line(start, end)
    writes = [(step.address, step.mask) for step in actual.steps]
    errors = [step.error_before for step in actual.steps]
    frame = bytearray(4096)
    apply_trace(frame, actual)
    context = f"start={start}, end={end}"
    if writes != expected.writes:
        raise AssertionError(f"Пиксельные адреса/маски отличаются: {context}")
    if list(actual.pixels) != expected.pixels:
        raise AssertionError(f"Экранные координаты отличаются: {context}")
    if errors != expected.errors:
        raise AssertionError(f"Аккумулятор ошибки отличается: {context}")
    if bytes(frame) != expected.framebuffer:
        raise AssertionError(f"Framebuffer линии отличается: {context}")
    return len(actual.steps)


def verify_geometry(pose: Pose, model_id: int = COBRA_ID) -> int:
    actual = render_geometry(MODELS[model_id], pose, use_face_vertices=False)
    expected = reference_geometry(pose, model_id)
    context = f"{MODELS[model_id].name} {json.dumps(asdict(pose))}"
    if actual.distance != expected.distance:
        raise AssertionError(f"LOD отличается: {context}")
    if actual.face_flags != expected.face_flags:
        raise AssertionError(f"Флаги граней отличаются: {context}")
    if actual.details["vertex_precision"] != expected.vertex_precision:
        raise AssertionError(f"Ветвь проекции или дробные байты X/Y/Z отличаются: {context}")
    projections = [(v.xmag, v.ymag, v.flags) if v is not None else None for v in actual.projected]
    if projections != expected.projected:
        differences = [(i, a, b) for i, (a, b) in enumerate(zip(projections, expected.projected)) if a != b]
        raise AssertionError(f"Проекции отличаются {differences}: {context}")
    decisions = [decision for decision in actual.edge_decisions if decision.draw]
    if [decision.index for decision in decisions] != expected.edge_indices:
        raise AssertionError(f"Список рёбер отличается: {context}")
    frame = bytearray(4096)
    writes = 0
    for decision in decisions:
        trace = trace_line(decision.start, decision.end)
        if list(trace.pixels) != expected.edge_pixels[decision.index]:
            raise AssertionError(f"Пиксели ребра {decision.index} отличаются: {context}")
        apply_trace(frame, trace)
        writes += len(trace.steps)
    if bytes(frame) != expected.framebuffer:
        raise AssertionError(f"Framebuffer корабля отличается: {context}")
    return writes


def verify_dodo_faces() -> None:
    """Ракурс с ошибочно открытой гранью 9. Сравниваем при одной позе."""
    model = MODELS[DODO_ID]
    pose = pose_from_angles(109.26, 46.73, 0, 1024)
    verify_geometry(pose, DODO_ID)
    original = render_geometry(model, pose, use_face_vertices=False)
    corrected = render_geometry(model, pose)
    assert [i for i, (a, b) in enumerate(zip(original.face_flags, corrected.face_flags)) if a != b] == [9]
    assert (original.face_flags[9], corrected.face_flags[9]) == (1, 0)
    before = {e.index for e in original.edge_decisions if e.draw}
    after = {e.index for e in corrected.edge_decisions if e.draw}
    assert len(after) == 22 and after == before - {16, 27, 28} and len(before) == 25
    assert {0, 30, 31, 32, 33} <= after  # Выбранное ребро и четыре кромки дока.


def verify_face_examples() -> None:
    rows = json.loads((ROOT / "face_comparison.json").read_text(encoding="utf-8"))
    assert len(rows) == len(MODELS)
    for model_id, (model, row) in enumerate(zip(MODELS, rows)):
        assert (row["id"], row["name"]) == (model_id, model.name)
        assert len(model.face_vertices) == len(model.faces)
        for index, anchor in enumerate(model.face_vertices):
            assert any(index in e.faces and e.faces[0] != e.faces[1] and e.vertex_a == anchor
                       for e in model.edges)
        example = row["example"]
        if not model.faces:
            assert example is None and row["different_flags"] == row["different_frames"] == 0
            continue
        pose = pose_from_angles(*example["view"])
        elite = reference_geometry(pose, model_id)
        vertex = render_geometry(model, pose, use_face_vertices=True)
        faces = [i for i, (a, b) in enumerate(zip(elite.face_flags, vertex.face_flags)) if a != b]
        assert faces == example["faces"] and faces, model.name
        before = set(elite.edge_indices)
        after = {e.index for e in vertex.edge_decisions if e.draw}
        assert sorted(before - after) == example["elite_only_edges"]
        assert sorted(after - before) == example["vertex_only_edges"]
        assert example["edge"] in before ^ after
        frame = bytearray(4096)
        for e in vertex.edge_decisions:
            if e.draw:
                apply_trace(frame, trace_line(e.start, e.end))
        pixels = sum((a ^ b).bit_count() for a, b in zip(frame, elite.framebuffer))
        assert pixels == example["changed_pixels"] and pixels > 0, model.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=500, help="Количество дополнительных случайных линий")
    parser.add_argument("--poses", type=int, default=64, help="Количество дополнительных случайных поз")
    parser.add_argument("--seed", type=int, default=1985)
    parser.add_argument("--output", type=Path, help="Сохранить краткий отчёт JSON")
    args = parser.parse_args()
    if args.lines < 0 or args.poses < 0:
        parser.error("Количество тестов должно быть неотрицательным")
    verify_dodo_faces()
    verify_face_examples()
    rng = random.Random(args.seed)
    line_count = line_writes = 0
    for start, end in line_cases(args.lines, rng):
        line_writes += verify_line(start, end)
        line_count += 1
    print(f"Линии: {line_count} случаев, {line_writes} пиксельных записей — совпали", flush=True)
    clip_rng = random.Random(args.seed)
    clip_count = clip_writes = rejected = 0
    for start, end in clipping_cases(args.lines, clip_rng):
        writes = verify_line(start, end)
        clip_count += 1
        clip_writes += writes
        rejected += int(writes == 0)
    print(f"Клиппинг: {clip_count} отрезков, {rejected} отсечены, {clip_writes} записей — совпали", flush=True)
    poses = list(geometry_cases(args.poses, rng))
    catalog = []
    for model_id, model in enumerate(MODELS):
        pose_count = ship_writes = 0
        near_poses = list(close_model_cases(model_id, args.poses, clip_rng))
        for pose in poses + near_poses:
            ship_writes += verify_geometry(pose, model_id)
            pose_count += 1
        print(f"{model.name}: {pose_count} поз, {ship_writes} записей — совпали", flush=True)
        catalog.append({"id": model_id, "name": model.name, "poses": pose_count,
                        "pixel_writes": ship_writes, "close_poses": len(near_poses)})
    result = {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "line_cases": line_count,
        "line_pixel_writes": line_writes,
        "clipping_cases": clip_count,
        "clipping_rejected": rejected,
        "clipping_pixel_writes": clip_writes,
        "close_model_cases": sum(row["close_poses"] for row in catalog),
        "ship_poses": sum(row["poses"] for row in catalog),
        "ship_pixel_writes": sum(row["pixel_writes"] for row in catalog),
        "models": catalog,
        "reference": "original Z80 instructions executed by SkoolKit 10.1",
        "compared": ["line write addresses and bitmasks", "line error accumulator",
                     "face flags", "per-vertex projection shift and three fractional bytes",
                     "projected vertex bytes including signed zero",
                     "selected edge indices", "per-edge pixel sequences", "4096 framebuffer bytes"],
        "scope": "19 original Spectrum blueprints plus Dodo from Elite Legend 1.09 rendered by the original Z80 code, centered front view, original viewport clipping; both projection paths, UI depth limits 32/64/128/256..6000 depending on model",
        "limitation": "Finite differential tests are evidence for the supported domain, not a proof over all inputs.",
    }
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
