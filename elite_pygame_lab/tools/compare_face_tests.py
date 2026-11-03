"""Сравнить тесты граней на моделях каталога и сохранить ракурсы для --mode faces."""

import argparse
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elite_lab.controls import pose_from_angles
from elite_lab.model import DODO_ID, MODELS, minimum_depth
from elite_lab.raster import trace_line
from elite_lab.renderer import render_geometry


DODO_VIEW = (109.26, 46.73, 0, 1024)


def cases(model_id):
    rng = Random(1985)
    angles = [(0, 0, 0), (90, 0, 0), (0, 90, 0), (24, 22, 0),
              (25, 40, 0), (35, 30, 0), (15, 35, 0), DODO_VIEW[:3]]
    angles.extend((rng.randrange(360), rng.randrange(-90, 91), rng.randrange(360))
                  for _ in range(48))
    for depth in (32, 64, 128, 256, 384, 512, 640, 767, 768, 896, 1024, 2048, 3968, 6000):
        if depth >= minimum_depth(model_id):
            for rotation in angles:
                yield (*rotation, depth)


def visible_pixels(geometry):
    pixels = {}
    inside = True
    for edge in geometry.edge_decisions:
        if edge.draw:
            line = trace_line(edge.start, edge.end)
            inside &= line.clip_status == "inside"
            pixels[edge.index] = {(s.x, s.y) for s in line.steps if s.in_bitmap}
    return pixels, inside


def compare(model_id):
    model = MODELS[model_id]
    poses = flag_diffs = edge_diffs = frame_diffs = 0
    best_score = None
    example = None
    for view in cases(model_id):
        pose = pose_from_angles(*view)
        elite = render_geometry(model, pose, use_face_vertices=False)
        vertex = render_geometry(model, pose, use_face_vertices=True)
        poses += 1
        if elite.face_flags == vertex.face_flags:
            continue  # Дальнейшие этапы получают одни и те же входные данные.
        flag_diffs += 1
        a, inside_a = visible_pixels(elite)
        b, inside_b = visible_pixels(vertex)
        edge_diffs += a.keys() != b.keys()
        changed = set().union(*a.values()) ^ set().union(*b.values())
        if not changed:
            continue
        frame_diffs += 1
        # Пример с целым корпусом и заметной разницей. Для Dodo сохраняем кадр пользователя.
        dodo_view = model_id == DODO_ID and view == DODO_VIEW
        score = (dodo_view, inside_a and inside_b, len(changed))
        if best_score is not None and score <= best_score:
            continue
        best_score = score
        edge = max(sorted(a.keys() | b.keys()),
                   key=lambda i: len((a.get(i, set()) ^ b.get(i, set())) & changed))
        example = {"view": view, "edge": 16 if dodo_view else edge,
                   "faces": [i for i, (x, y) in enumerate(zip(elite.face_flags, vertex.face_flags)) if x != y],
                   "elite_only_edges": sorted(a.keys() - b.keys()),
                   "vertex_only_edges": sorted(b.keys() - a.keys()), "changed_pixels": len(changed)}
    return {"id": model_id, "name": model.name, "poses": poses, "different_flags": flag_diffs,
            "different_edges": edge_diffs, "different_frames": frame_diffs, "example": example}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Сохранить результаты в JSON")
    args = parser.parse_args()
    rows = []
    for model_id in range(len(MODELS)):
        row = compare(model_id)
        rows.append(row)
        print(f"{model_id:02} {row['name']:<25} {row['poses']:4} поз · различия: "
              f"флаги {row['different_flags']}, рёбра {row['different_edges']}, "
              f"пиксельные кадры {row['different_frames']}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
