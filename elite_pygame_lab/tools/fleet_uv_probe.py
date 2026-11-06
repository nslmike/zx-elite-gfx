#!/usr/bin/env python3
"""Сравнить UV классических кораблей в штатном OpenGL-рендерере."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import radians, tan
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import moderngl
import numpy as np
import pygame

from elite_lab import space_look
from elite_lab.model import MODELS
from elite_lab.space_assets import AssetCatalog
from elite_lab.space_mesh import source_transform
from elite_lab.space_scene import SpaceObject
from elite_lab.space_viewer import SpaceRenderer
from hull_microdetail_probe import ProbeScene, ShaderOverlay, font, unit


CLOSE_SIZE = (768, 480)
FULL_SIZE = (410, 268)
FOV = 42.0
FULL_VIEWS = ("side_a", "side_b")
UPPER_LOWER = (("сверху", (-.56, .74, .37)), ("снизу", (-.58, -.74, -.37)))
STARBOARD_PORT = (("борт +X", (.92, .39, -.38)), ("борт −X", (-.92, -.34, .32)))
MISSILE_VIEWS = (("нос и кили", (-.75, .57, .39)), ("корма и кили", (-.72, -.61, -.41)))
MODEL_VIEWS = {1: MISSILE_VIEWS, 15: STARBOARD_PORT, 18: STARBOARD_PORT}
STATES = ("before", "after")
SOURCES = (
    "space_mesh.py", "space_viewer.py", "space_assets.py", "space_look.py",
    "space_reflections.py", "space_scene.py", "shaders/space_hull.vert",
    "shaders/space_hull.frag", "shaders/space_sky.vert", "shaders/space_sky.frag",
)


@dataclass(frozen=True)
class Case:
    edge: tuple[int, int]
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    before_sha256: str


# Камеры и рёбра из исходного аудита, до округления координат.
CASES = {
    8: Case((1, 8), (-0.43998616768956095, 1.0996724207007944, -0.2154719763296847),
            (-0.17789210379123688, 0.21619820594787598, -0.34651900827884674),
            "6b53a7b6a8cff1ff05ccb9125784d49408dd84159d8a4b3f11d6f38baaead0a7"),
    13: Case((11, 12), (-0.5599261726969511, 1.4401071805098629, -0.41788364480331947),
             (-0.2806350886821747, 0.17621272802352905, -0.35079385340213776),
             "c316f92bc2f21d516480057236d480eeb1ae23f571855dd87acb2c3280eb576b"),
    14: Case((8, 9), (0.4148010161579541, 0.7606936514288216, -0.4851379705406322),
             (0.24260388314723969, 0.17127833515405655, -0.48520778119564056),
             "6e83ab032c2719956539dad087d72024b726759b3f3f36936524dbec39cea939"),
    15: Case((4, 12), (0.6762854989858921, 0.7715198820560081, -0.7707541499480722),
             (0.0001802593469619751, 0.4927407503128052, -0.4927407503128052),
             "c57f4a0c5a4e3e139958caa14b0116c8fa8845c392ae8525c602879536d9a715"),
    18: Case((4, 9), (0.9837270988958255, 1.0959942273128085, -0.04046881654363092),
             (-0.0027138590812683105, 0.5427733212709427, -0.12438555061817169),
             "4896fb8e1fe32c5d8b474e8a0aae929c077cedb7e774d5ea297cb4349c455a02"),
}

SMALL_CASES = {
    1: Case((2, 3), (-0.04545919968329344, 0.6584907147283023, 0.4957410518204416),
            (0.0, 0.13697151839733124, 0.4129691421985626),
            "bb2653f36254a30574f8a5711243a10ce22529ea67c78e0471c8decac59595b8"),
    16: Case((0, 4), (-0.3707075552356752, 1.2863696327766294, 0.22918606334764485),
             (-0.22447679936885834, 0.11223839968442917, 0.008768633008003235),
             "0e99913fae4445de3b66791b69f08a1a60ad46c560618b82e2b5754fb8a2d39e"),
    17: Case((0, 1), (-0.060158905031375616, 1.5888848269586797, 0.30780196184507275),
             (0.0, 0.08160644769668579, 0.0),
             "b73c87b1da64ad9d70f3093c1fdfd330d649ce341b268ae4f46e1c33af2967e9"),
}

GROUPS = {"fleet": (CASES, "fleet_uv"), "small": (SMALL_CASES, "small_ships_uv")}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def edge_positions(model_id, indices, vertices):
    source = MODELS[model_id].vertices
    points = np.array([(source[i].x, source[i].y, source[i].z) for i in indices])
    center, radius = source_transform(model_id)
    distance = np.linalg.norm((points - center)[:, None, :] / radius - vertices[:, :3], axis=2)
    closest = distance.argmin(axis=1)
    if np.any(distance.min(axis=1) > 1e-7):
        raise ValueError(f"№{model_id:02} · вершины ребра отсутствуют в меше")
    return vertices[closest, :3]


def edge_sample(vertices, endpoints, scale):
    triangles = vertices.reshape(-1, 3, 9)
    first, last = [(triangles[:, :, :3] == point).all(axis=2) for point in endpoints]
    sides = first.any(axis=1) & last.any(axis=1)
    if np.count_nonzero(sides) != 2:
        raise ValueError("Для исследуемого ребра ожидаются два треугольника")
    uvs = np.stack((triangles[sides][first[sides]][:, 6:8],
                    triangles[sides][last[sides]][:, 6:8]), axis=1).astype("f8")
    t = np.linspace(0, 1, 129)[:, None]
    delta = (uvs[0] - uvs[1]) * scale
    repeated = (1 - t) * delta[0] + t * delta[1]
    distance = np.linalg.norm(repeated - np.round(repeated), axis=1) * space_look._SURFACE_SIZE
    normals = triangles[sides][:, 0, 3:6]
    angle = float(np.degrees(np.arccos(np.clip(unit(normals[0]) @ unit(normals[1]), -1, 1))))
    return {"angle_deg": angle, "uv": uvs.tolist(), "median_texels": float(np.median(distance)),
            "max_texels": float(np.max(distance))}


def load_cases(assets, baseline_dir, cases=CASES):
    records = {}
    for model_id, case in cases.items():
        before = np.load(baseline_dir / f"before-{model_id:02}.npy", allow_pickle=False)
        after = assets[model_id].vertices()
        if before.dtype != after.dtype or before.shape != after.shape or not np.isfinite(before).all():
            raise ValueError(f"№{model_id:02} · неверный формат исходного меша")
        hashes = {state: digest(mesh.tobytes()) for state, mesh in zip(STATES, (before, after))}
        if hashes["before"] != case.before_sha256:
            raise ValueError(f"№{model_id:02} · исходный меш отличается от аудита")
        if before[:, (0, 1, 2, 3, 4, 5, 8)].tobytes() != after[:, (0, 1, 2, 3, 4, 5, 8)].tobytes():
            raise ValueError(f"№{model_id:02} · в сравнении изменились позиция, нормаль или материал")
        endpoints = edge_positions(model_id, case.edge, before)
        scale = space_look._STYLES[model_id].scale
        samples = {state: edge_sample(mesh, endpoints, scale)
                   for state, mesh in zip(STATES, (before, after))}
        if samples["before"]["median_texels"] < .01 or samples["after"]["max_texels"] > .01:
            raise ValueError(f"№{model_id:02} · на исходном ребре ожидался разрыв, на новом — совпадение")
        records[model_id] = (before, after, endpoints, {
            "name": assets[model_id].name, "edge": case.edge, "edge_xyz": endpoints.tolist(),
            "mesh_sha256": hashes, "edge_uv": samples, "views": {},
        })
    return records


def fit_view(vertices, direction):
    eye_axis = unit(direction)
    right = unit(np.cross(-eye_axis, (0, 1, 0)))
    up = np.cross(right, -eye_axis)
    x, y, z = (vertices[:, :3] @ axis for axis in (right, up, eye_axis))
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    target = cx * right + cy * up
    aspect = FULL_SIZE[0] / FULL_SIZE[1]
    half_height = .83 * tan(radians(FOV / 2))
    depth = np.maximum(np.abs(x - cx) / (aspect * half_height), np.abs(y - cy) / half_height)
    distance = np.max(z + depth)
    return target + distance * eye_axis, target


def render_variant(ctx, assets, records, directory, atlas=False, *, cases=CASES):
    shader_dir = space_look._SHADERS
    if atlas:
        source = (shader_dir / "space_hull.frag").read_text(encoding="utf-8")
        anchor = "    frag_color = vec4(display_color(color), 1.0);"
        if source.count(anchor) != 1:
            raise RuntimeError("Изменилась точка вывода цвета в шейдере")
        source = source.replace("void main() {", "uniform float u_probe_atlas;\n\nvoid main() {")
        shader_dir = ShaderOverlay(shader_dir, source.replace(anchor, """
    vec4 atlas = texture(u_surface, v_uv * u_texture_scale + u_uv_offset);
    float map_view = clamp(0.17 + (dot(atlas, vec4(.56, .21, 0., .23)) - .33) * 1.5, 0., 1.);
    bool uses_atlas = v_material == 0 || v_material == 1 || v_material == 4;
    float diagnostic = uses_atlas ? map_view : .15 + .03 * float(v_material);
    frag_color = vec4(mix(display_color(color), vec3(diagnostic), u_probe_atlas), 1.0);"""))
    parking = ctx.fbo
    renderer = None
    targets = {}
    try:
        for size in (CLOSE_SIZE, FULL_SIZE):
            targets[size] = (ctx.simple_framebuffer(size, samples=min(4, ctx.max_samples)),
                             ctx.simple_framebuffer(size))
        for model_id, (before, after, endpoints, record) in records.items():
            case = cases[model_id]
            obj = SpaceObject(model_id, np.zeros(3), 1.0, np.eye(4))
            scene = ProbeScene((assets[model_id],), (obj,), np.array(case.eye), np.array(case.target), FOV)
            if renderer is None:
                with patch.object(space_look, "_SHADERS", shader_dir):
                    renderer = SpaceRenderer(ctx, scene)
                if atlas:
                    renderer.look.hull["u_probe_atlas"].value = 1.0
            else:
                # Штатная загрузка сбрасывает кеш теней при смене корабля.
                renderer._load_fleet(scene)
            gpu_buffer = renderer.meshes[0][0].buffer
            hull = renderer.look.hull
            shadow_hashes = {}
            cameras = {"edge": (scene.eye, scene.target)}
            views = MODEL_VIEWS.get(model_id, UPPER_LOWER)
            cameras.update({view: fit_view(after, direction) for view, (_, direction) in zip(FULL_VIEWS, views)})
            for view, (eye, look_at) in cameras.items():
                size = CLOSE_SIZE if view == "edge" else FULL_SIZE
                target, resolved = targets[size]
                scene.eye = eye
                scene.target = look_at
                vp = scene.view_projection(size[0] / size[1])
                info = record["views"].setdefault(view, {
                    "eye": eye.tolist(), "target": scene.target.tolist(), "fov": scene.fov,
                    "size": size, "view_projection": vp.tolist(), "model_matrix": obj.matrix.tolist(),
                })
                if view == "edge":
                    clip = np.column_stack((endpoints, np.ones(2))) @ vp.T
                    info["edge_pixels"] = ((clip[:, :2] / clip[:, 3, None] * (.5, -.5) + .5) * size).tolist()
                frames = {}
                for state, mesh in zip(STATES, (before, after)):
                    gpu_buffer.write(mesh.tobytes())
                    renderer.draw(scene, target, size, False)
                    ctx.copy_framebuffer(resolved, target)
                    raw = resolved.read(components=3, alignment=1)
                    if view == "edge":
                        shadow_hashes[state] = digest(renderer.shadow.read(alignment=1))
                    invariants = {
                        "shadow_sha256": shadow_hashes[state],
                        "light_sha256": digest(b"".join(hull[key].read() for key in
                                                       ("u_sun_direction", "u_shadow_vp", "u_shadow_enabled"))),
                        "pose_sha256": digest(b"".join(hull[key].read() for key in
                                                      ("u_view_proj", "u_model", "u_eye"))),
                    }
                    error = ctx.error
                    if error != "GL_NO_ERROR":
                        raise RuntimeError(f"№{model_id:02} · {view} · {state} · {error}")
                    stem = f"{model_id:02}-{view}-{state}" + ("-atlas" if atlas else "")
                    image = pygame.image.frombytes(raw, size, "RGB", True)
                    pygame.image.save(image, str(directory / f"{stem}.png"))
                    frames[state] = {"png": f"{stem}.png", "rgb8_sha256_bottom_up": digest(raw),
                                     "gl_error": error, **invariants}
                if any(frames["before"][key] != frames["after"][key] for key in invariants):
                    raise RuntimeError(f"№{model_id:02} · {view} · в паре изменились свет, тень или поза")
                info["atlas" if atlas else "default"] = frames
            print(f"№{model_id:02} · {record['name']} · {'карта' if atlas else 'штатный кадр'}", flush=True)
    finally:
        parking.use()
        for target, resolved in targets.values():
            resolved.release()
            target.release()
        if renderer is not None:
            renderer.release()


BG, INK, MUTED, ACCENT = (12, 20, 30), (235, 242, 250), (162, 181, 201), (127, 220, 227)
MARGIN, GAP = 24, 12


def label(canvas, text, position, size, color=INK, bold=False):
    canvas.blit(font(size, bold).render(text, True, color), position)


def blit_frame(canvas, directory, frame, xy, edge=None):
    tile = pygame.image.load(str(directory / frame["png"]))
    if edge is not None:
        a, b = np.asarray(edge)
        dx, dy = unit(b - a)
        normal = np.array((dy, -dx))
        for t in (.26, .74):
            spot = (1 - t) * a + t * b
            for side in (-1, 1):
                p = spot + side * 13 * normal
                q = spot + side * 25 * normal
                pygame.draw.line(tile, (8, 23, 33), p, q, 5)
                pygame.draw.line(tile, ACCENT, p, q, 2)
    canvas.blit(tile, xy)
    pygame.draw.rect(canvas, (43, 64, 83), (*xy, *tile.get_size()), 1)


def make_previews(records, raw, output, prefix="fleet_uv"):
    w, h = CLOSE_SIZE
    row = h + 56
    canvas = pygame.Surface((2 * w + 2 * MARGIN + GAP, 116 + len(records) * row))
    canvas.fill(BG)
    label(canvas, "UV обшивки · Стыки панелей", (MARGIN, 17), 32, bold=True)
    label(canvas, "Голубые штрихи указывают стык · кадры 1:1",
          (MARGIN, 59), 20, MUTED)
    for column, title in enumerate(("ДО · Разрыв рисунка", "ПОСЛЕ · Общие UV на ребре")):
        x = MARGIN + column * (w + GAP)
        label(canvas, title, (x, 92), 21, (MUTED, ACCENT)[column], bold=True)
    for i, (model_id, (_, _, _, record)) in enumerate(records.items()):
        y = 116 + i * row
        edge = "–".join(map(str, record["edge"]))
        name = record["name"]
        angle = f"{record['edge_uv']['after']['angle_deg']:.1f}".replace(".", ",")
        label(canvas, f"№{model_id:02} · {name} · ребро {edge} · {angle}°",
              (MARGIN, y + 8), 23, bold=True)
        view = record["views"]["edge"]
        for column, state in enumerate(STATES):
            x = MARGIN + column * (w + GAP)
            blit_frame(canvas, raw, view["default"][state], (x, y + 41), view["edge_pixels"])
    pygame.image.save(canvas, str(output / f"{prefix}_closeup.png"))

    w, h = FULL_SIZE
    row = h + 51
    canvas = pygame.Surface((4 * w + 2 * MARGIN + 4 * GAP, 116 + len(records) * row))
    canvas.fill(BG)
    label(canvas, "UV обшивки · Полные корпуса", (MARGIN, 17), 32, bold=True)
    label(canvas, "В каждой паре совпадают ракурс, поза, материал и освещение",
          (MARGIN, 59), 20, MUTED)
    for column, title in enumerate(("Вид A · До", "Вид A · После", "Вид B · До", "Вид B · После")):
        x = MARGIN + column * (w + GAP) + (column // 2) * GAP
        label(canvas, title, (x, 92), 20, (MUTED, ACCENT)[column % 2], bold=True)
    for i, (model_id, (_, _, _, record)) in enumerate(records.items()):
        y = 116 + i * row
        sides, _ = zip(*MODEL_VIEWS.get(model_id, UPPER_LOWER))
        label(canvas, f"№{model_id:02} · {record['name']} · {sides[0]}", (MARGIN, y + 6), 22, bold=True)
        label(canvas, sides[1], (MARGIN + 2 * (w + GAP) + GAP, y + 7), 21, MUTED)
        for j, view in enumerate(FULL_VIEWS):
            for k, state in enumerate(STATES):
                x = MARGIN + (2 * j + k) * (w + GAP) + j * GAP
                blit_frame(canvas, raw, record["views"][view]["default"][state], (x, y + 36))
    pygame.image.save(canvas, str(output / f"{prefix}_overview.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Каталог с before-NN.npy из аудита")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Каталог для исходных кадров и capture.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "previews")
    parser.add_argument("--group", choices=GROUPS, default="fleet",
                        help="fleet — исходные пять кораблей, small — Missile, Sidewinder, Krait")
    parser.add_argument("--atlas", action="store_true", help="Дополнительно снять карту без освещения в raw-dir")
    args = parser.parse_args()
    cases, prefix = GROUPS[args.group]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = AssetCatalog().classic
    source_hashes = {name: digest((ROOT / "elite_lab" / name).read_bytes()) for name in SOURCES}
    records = load_cases(assets, args.baseline_dir, cases)
    pygame.font.init()
    options = {"backend": "egl"} if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else {}
    ctx = moderngl.create_context(standalone=True, require=330, **options)
    parking = ctx.simple_framebuffer((1, 1))
    parking.use()
    try:
        render_variant(ctx, assets, records, args.raw_dir, cases=cases)
        if args.atlas:
            render_variant(ctx, assets, records, args.raw_dir, atlas=True, cases=cases)
        status = ctx.error
        if status != "GL_NO_ERROR":
            raise RuntimeError(status)
        if source_hashes != {name: digest((ROOT / "elite_lab" / name).read_bytes()) for name in SOURCES}:
            raise RuntimeError("Исходники изменились во время захвата")
        metadata = {
            "renderer": ctx.info["GL_RENDERER"], "gl_version": ctx.info["GL_VERSION"],
            "msaa_samples": min(4, ctx.max_samples), "gl_error": status,
            "sun_direction": space_look.SUN_DIRECTION, "source_sha256": source_hashes,
            "models": {model_id: record for model_id, (_, _, _, record) in records.items()},
        }
        (args.raw_dir / "capture.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
        make_previews(records, args.raw_dir, args.output_dir, prefix)
        print(f"{metadata['renderer']} · OpenGL {metadata['gl_version']} · {status}", flush=True)
    finally:
        parking.release()
        ctx.release()
        pygame.font.quit()


if __name__ == "__main__":
    main()
