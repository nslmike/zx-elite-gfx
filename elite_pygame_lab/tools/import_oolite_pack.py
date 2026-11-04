#!/usr/bin/env python3
"""Воспроизводимый импорт штатных моделей Oolite (требуются numpy и Pillow).

python tools/import_oolite_pack.py --source /path/to/oolite
Исходник: https://github.com/OoliteProject/oolite/tree/28ceaaba0d79d0b6ea1c044c6235e3616b170098
Скачивание и распаковка исходника выполняются отдельно; импорт не использует сеть.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image

REVISION = "28ceaaba0d79d0b6ea1c044c6235e3616b170098"
REPO = f"https://github.com/OoliteProject/oolite/blob/{REVISION}"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
DEFAULT_AUTHOR = "Griff, CaptSolo, another_commander и участники Oolite"
CATALOG = [
    (0, "Coriolis", "coriolis-station"),
    (1, "Missile", "missile"),
    (2, "Escape pod", "escape-capsule"),
    (3, "Splinter", "splinter"),
    (4, "Cargo canister", "barrel"),
    (5, "Boulder", "boulder"),
    (6, "Asteroid", "asteroid"),
    (7, "Rock Hermit", "rock-hermit"),
    (8, "Viper", "viper"),
    (9, "Cobra Mk III (торговая)", "cobra3"),
    (10, "Python (торговый)", "python-trader"),
    (11, "Python (пиратский)", "python"),
    (12, "Cobra Mk III (пиратская)", "cobra3-pirate"),
    (13, "Asp Mk II", "asp"),
    (14, "Fer-de-Lance", "ferdelance"),
    (15, "Thargoid", "thargoid"),
    (16, "Sidewinder", "sidewinder"),
    (17, "Krait", "krait"),
    (18, "Thargon", "tharglet"),
    (19, "Dodo · Oolite Dodecahedron Station", "dodecahedron-station"),
]


def strip_comments(text):
    """Убирает комментарии OpenStep/DAT, сохраняя URL внутри строк."""
    return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*|\#[^\n]*',
                  lambda m: m[0] if m[0].startswith('"') else " ", text,
                  flags=re.S)


def read_openstep(path):
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[{}()=;,]|[^\s{}()=;,]+',
                        strip_comments(path.read_text()))
    index = 0

    def value():
        nonlocal index
        token = tokens[index]
        index += 1
        if token == "{":
            result = {}
            while tokens[index] != "}":
                key = value()
                assert tokens[index] == "="
                index += 1
                result[key] = value()
                assert tokens[index] == ";"
                index += 1
            index += 1
            return result
        if token == "(":
            result = []
            while tokens[index] != ")":
                result.append(value())
                if tokens[index] == ",":
                    index += 1
            index += 1
            return result
        return json.loads(token) if token.startswith('"') else token

    result = value()
    assert index == len(tokens)
    return result


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-20)


def read_dat(path, smooth=False):
    lines = [line.strip().replace(",", " ")
             for line in strip_comments(path.read_text()).splitlines() if line.strip()]
    nv, nf = int(lines[0].split()[1]), int(lines[1].split()[1])
    vi, fi, ti = [lines.index(s) + 1 for s in ("VERTEX", "FACES", "TEXTURES")]
    positions = np.array([[float(v) for v in l.split()] for l in lines[vi:vi + nv]])
    faces = [l.split() for l in lines[fi:fi + nf]]
    assert all(int(l[6]) == 3 for l in faces), path
    indices = np.array([[int(v) for v in l[7:10]] for l in faces])
    face_normals = np.array([[float(v) for v in l[3:6]] for l in faces])
    smooth_groups = np.array([int(l[0]) for l in faces])
    textures = [l.split() for l in lines[ti:ti + nf]]
    names = {}
    for i, line in enumerate(lines):
        if line.startswith("NAMES "):
            names = {str(k): lines[i + 1 + k] for k in range(int(line.split()[1]))}
    materials = list(dict.fromkeys(names.get(l[0], l[0]) for l in textures))
    slots = [materials.index(names.get(l[0], l[0])) for l in textures]
    uv = np.array([np.array(l[3:], float).reshape(3, 2) / np.array(l[1:3], float)
                   for l in textures])
    # Oolite загружает строки PNG сверху вниз и использует DAT.t без инверсии.
    # У атласа стандартные GL UV; загрузчик переворачивает PNG один раз по Y.
    uv[:, :, 1] = 1 - uv[:, :, 1]
    cross = np.cross(positions[indices[:, 1]] - positions[indices[:, 0]],
                     positions[indices[:, 2]] - positions[indices[:, 0]])
    if "NORMALS" in lines:
        ni = lines.index("NORMALS") + 1
        normals = unit(np.array([[float(v) for v in l.split()]
                                  for l in lines[ni:ni + nv]]))
        corner_normals = normals[indices]
        expected = corner_normals.mean(axis=1)
    else:
        # В legacy DAT нормаль грани задаёт наружную сторону. Нулевые нормали
        # Oolite восстанавливает из исходного clockwise-порядка.
        zero = np.linalg.norm(face_normals, axis=1) < 1e-15
        face_normals[zero] = -cross[zero]
        face_normals = unit(face_normals)
        if smooth:
            # Площадь и smoothGroup — те же критерии, что OOMesh.m.
            groups = {}
            area = np.linalg.norm(cross, axis=1)
            for f, face in enumerate(indices):
                for vertex in face:
                    key = (int(vertex), int(smooth_groups[f]))
                    groups[key] = groups.get(key, np.zeros(3)) + face_normals[f] * area[f]
            corner_normals = np.array([[unit(groups[(int(v), int(smooth_groups[f]))])
                                       for v in face] for f, face in enumerate(indices)])
        else:
            corner_normals = np.repeat(face_normals[:, None, :], 3, axis=1)
        expected = face_normals
    reverse = (cross * expected).sum(axis=1) < 0
    indices[reverse] = indices[reverse][:, [0, 2, 1]]
    uv[reverse] = uv[reverse][:, [0, 2, 1]]
    corner_normals[reverse] = corner_normals[reverse][:, [0, 2, 1]]
    expanded = np.concatenate((positions[indices], corner_normals, uv,
                               np.repeat(np.array(slots)[:, None, None], 3, axis=1)), axis=2)
    return expanded.reshape(-1, 9), materials


def numbers(value):
    return np.array(value.split() if isinstance(value, str) else value, float)


def rotation(quaternion):
    w, x, y, z = numbers(quaternion)
    q = np.array([w, x, y, z]); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class Importer:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.shipdata = read_openstep(source / "Resources/Config/shipdata.plist")
        self.inputs, self.image_cache = {}, {}
        self.mesh_cache = {}
        (output / "textures").mkdir(parents=True, exist_ok=True)
        (output / "attribution").mkdir(parents=True, exist_ok=True)

    def record(self, path):
        name = path.relative_to(self.source).as_posix()
        self.inputs[name] = {"source": f"{REPO}/{name}",
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def resolved(self, key):
        entry = self.shipdata[key]
        result = self.resolved(entry["like_ship"]) if "like_ship" in entry else {}
        return result | entry

    def image(self, name, kind, limit, color=None):
        key = (name, kind, limit, tuple(color) if color is not None else None)
        if key in self.image_cache:
            return self.image_cache[key]
        source = self.source / "Resources/Textures" / name
        self.record(source)
        image = Image.open(source).convert("RGBA")
        image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        a = np.asarray(image).astype(float) / 255
        if kind == "diffuse":
            result = image.convert("RGB")
        elif kind == "normal":
            result = Image.fromarray(np.rint((unit(a[:, :, :3] * 2 - 1) + 1) * 127.5).astype("uint8"))
        elif kind == "specular":
            # Oolite gamma-corrects RGB and only uses specular_modulate_color
            # with a map; specular_color is its no-map fallback, not a multiplier.
            rgb = np.clip(a[:, :, :3] ** 2.2 * color, 0, 1)
            result = Image.fromarray(np.rint(rgb * 255).astype("uint8"))
        elif kind == "gloss":
            result = image.getchannel("A")
        elif kind == "emission":
            # Адаптация alpha emission + RGB-тон: запекаем цвет в sRGB PNG.
            rgb = np.clip(a[:, :, 3:4] ** 2.2 * color, 0, 1) ** (1/2.2)
            result = Image.fromarray(np.rint(rgb * 255).astype("uint8"))
            if not np.any(np.asarray(result)):
                self.image_cache[key] = None
                return None
        else:
            raise ValueError(kind)
        # Diffuse — JPEG высокого качества; карты данных — без потерь.
        suffix = ".jpg" if kind == "diffuse" else ".png"
        digest = hashlib.sha256(result.tobytes() + kind.encode()).hexdigest()[:12]
        filename = f"textures/{kind}_{digest}{suffix}"
        target = self.output / filename
        if not target.exists():
            if suffix == ".jpg":
                result.save(target, quality=92, subsampling=0, optimize=True)
            else:
                result.save(target, optimize=True)
        self.image_cache[key] = filename
        return filename

    def material(self, name, spec, limit):
        diffuse = spec.get("diffuse_map", name)
        if isinstance(diffuse, dict):
            diffuse = diffuse["name"]
        material = {"name": name, "diffuse": self.image(diffuse, "diffuse", limit),
                    "gloss": float(spec.get("gloss", .5)),
                    "shininess": float(spec.get("shininess", 32)),
                    "source_diffuse": f"{REPO}/Resources/Textures/{diffuse}"}
        if "normal_map" in spec:
            material["normal"] = self.image(spec["normal_map"], "normal", limit)
        if "specular_map" in spec:
            modulate = numbers(spec.get("specular_modulate_color", (1, 1, 1)))[:3]
            material["specular"] = self.image(spec["specular_map"], "specular", min(limit, 512), modulate)
            material["gloss_map"] = self.image(spec["specular_map"], "gloss", min(limit, 512))
            material["specular_color"] = [1, 1, 1]
        else:
            material["specular_color"] = numbers(spec.get("specular_color", (.2, .2, .2)))[:3].tolist()
        if "emission_map" in spec:
            emission = spec["emission_map"]
            assert isinstance(emission, dict) and emission.get("extract_channel") == "a", emission
            color = spec.get("emission_modulate_color", (1, 1, 1))
            if isinstance(color, dict):
                color = np.array([float(color[c]) / 255 for c in ("red", "green", "blue")])
            else:
                color = numbers(color)[:3]
            strength = max(1, float(max(color)))
            image = self.image(emission["name"], "emission", limit, color / strength)
            if image:
                material["emission"] = image
                material["emission_strength"] = strength
        return material

    def assemble(self, key, limit, world_rotation=None, world_position=None):
        world_rotation = np.eye(3) if world_rotation is None else world_rotation
        world_position = np.zeros(3) if world_position is None else world_position
        entry = self.resolved(key)
        source = self.source / "Resources/Models" / entry["model"]
        self.record(source)
        vertices, names = read_dat(source, str(entry.get("smooth", "no")) in ("yes", "1", "true"))
        vertices[:, :3] = (vertices[:, :3] * float(entry.get("model_scale_factor", 1))) @ world_rotation.T + world_position
        vertices[:, 3:6] = vertices[:, 3:6] @ world_rotation.T
        materials = [self.material(name, entry.get("materials", {}).get(name, {}), limit) for name in names]
        sources = [source.relative_to(self.source).as_posix()]
        parts = [vertices]
        for sub in entry.get("subentities", []):
            if "subentity_key" not in sub:
                continue  # Игровые flashers не являются геометрией/текстурой.
            r = world_rotation @ rotation(sub.get("orientation", (1, 0, 0, 0)))
            p = world_position + world_rotation @ numbers(sub.get("position", (0, 0, 0)))
            child, child_materials, child_sources = self.assemble(sub["subentity_key"], limit, r, p)
            child[:, 8] += len(materials)
            parts.append(child)
            materials.extend(child_materials)
            sources.extend(child_sources)
        return np.concatenate(parts), materials, sources

    def run(self):
        models = {}
        for model_id, name, key in CATALOG:
            limit = 1024 if model_id in (9, 10, 11, 12, 17) else 512
            vertices, materials, sources = self.assemble("oolite_template_" + key, limit)
            for material in materials:
                label = material["name"].lower()
                kind = 5 if model_id in (3, 5, 6, 7) else 0
                if model_id == 7 and "hermit_rock_" not in label:
                    kind = 1
                if label in ("gun", "engine", "engines") or "dock" in label:
                    kind = 1
                if "glass" in label:
                    kind = 2
                material["procedural_type"] = kind
            center = (vertices[:, :3].min(0) + vertices[:, :3].max(0)) * .5
            vertices[:, :3] -= center
            radius = float(np.linalg.norm(vertices[:, :3], axis=1).max())
            vertices[:, :3] /= radius
            vertices = vertices.astype("float32")
            # Пары Cobra используют общую сетку; материал задаётся отдельно.
            digest = hashlib.sha256(vertices.tobytes()).hexdigest()
            mesh = self.mesh_cache.get(digest)
            if mesh is None:
                mesh = f"{key}.npz"
                np.savez_compressed(self.output / mesh, vertices=vertices)
                self.mesh_cache[digest] = mesh
            author = DEFAULT_AUTHOR
            if model_id == 7:
                author = "Griff, CaptSolo, Killer Wolf и участники Oolite"
            elif model_id in (0, 19):
                author = "Griff, CaptSolo, PAGroove и участники Oolite"
            models[str(model_id)] = {"name": name, "mesh": mesh, "materials": materials,
                                     "author": author, "source": f"{REPO}/{sources[0]}",
                                     "shipdata_key": "oolite_template_" + key,
                                     "source_meshes": [f"{REPO}/{s}" for s in sources],
                                     "triangles": len(vertices) // 3}
            print(f"{model_id:02} {name}: {len(vertices)//3} triangles, {len(materials)} materials")
        for path in ("LICENSE.md", "Doc/contributors.txt", "Resources/Config/shipdata.plist"):
            self.record(self.source / path)
        shutil.copyfile(self.source / "LICENSE.md", self.output / "attribution/UPSTREAM_LICENSE.md")
        shutil.copyfile(self.source / "Doc/contributors.txt", self.output / "attribution/UPSTREAM_CONTRIBUTORS.txt")
        pack = {"id": "oolite", "title": "Oolite", "author": DEFAULT_AUTHOR,
                "license": "CC BY-NC-SA 3.0", "license_spdx": "CC-BY-NC-SA-3.0",
                "license_url": LICENSE_URL,
                "source": f"https://github.com/OoliteProject/oolite/tree/{REVISION}",
                "revision": REVISION, "attribution": "attribution/README.md",
                "normal_basis": "cross(N,T)", "tangent_axis": "+dP/dU",
                "models": models}
        (self.output / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n")
        (self.output / "attribution/source_manifest.json").write_text(json.dumps(self.inputs, ensure_ascii=False, indent=2) + "\n")
        # Удаляем только устаревшие файлы этого импортёра после смены разрешения.
        used = {path for path in self.image_cache.values() if path}
        for path in (self.output / "textures").iterdir():
            if re.fullmatch(r"(diffuse|normal|specular|gloss|emission)_[0-9a-f]{12}\.(png|jpg)", path.name):
                if path.relative_to(self.output).as_posix() not in used:
                    path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Распакованная закреплённая версия Oolite")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "asset_packs/oolite")
    args = parser.parse_args()
    Importer(args.source.resolve(), args.output.resolve()).run()


if __name__ == "__main__":
    main()
