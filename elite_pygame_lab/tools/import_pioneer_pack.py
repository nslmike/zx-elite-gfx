#!/usr/bin/env python3
"""Воспроизводимый импорт Sinonatrix из закреплённой версии Pioneer.

Зависимости только для повторного импорта: numpy и Pillow (DDS/DXT5).
    python3 tools/import_pioneer_pack.py --cache /tmp/elite-pioneer-source

Исходные DAE/DDS загружаются в cache, а не включаются в пакет. SHA-256
проверяется и для уже существующего cache. Импорт не использует Blender/Assimp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


REVISION = "7506ef474e060c129fe27349f279895b2a5d8ed9"
REPOSITORY = "https://github.com/pioneerspacesim/pioneer"
SHIP_PATH = "data/models/ships/sinonatrix/"
RAW = f"https://raw.githubusercontent.com/pioneerspacesim/pioneer/{REVISION}/"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
SLOTS = {"Hull": 0, "Canopy": 1, "Engine": 2}
# Proper rotation: Blender Z_UP / nose +Y -> atlas Y_UP / nose +Z.
# The determinant is +1, so this axis conversion does not reverse winding.
AXES = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

HASHES = {
    "AUTHORS.txt": "bb9f1c443f4660e0f641b4938c854a9b742f9a106717f46f554d2f2695499e11",
    "licenses/CC-BY-SA-3.0.txt": "3f941b3b89cf7b8370ceb83cc76d2120d471b58735d8ca60238a751a48d7f72f",
    SHIP_PATH + "sinonatrix.model": "03a5c09895525680df38267fa21878d59c681e02141bfec0dfd0da895d57569f",
    SHIP_PATH + "sinonatrix_cockpit.model": "150c2e36cb25014a9b357031c0643403823dba6c8cf6c470bf4ff82f0c31a72d",
    SHIP_PATH + "sinonatrix_hi.dae": "3537e997b746747a9beaba4fb314b56a0894c76e8951e2967aa9562d36e9da27",
    SHIP_PATH + "sinonatrix_med.dae": "c7e323ed3d72d4d410ef0565316340d919ae3ed8fbe213a05078ea04f7db3457",
    SHIP_PATH + "sinonatrix_low.dae": "73146a5ad1ed1f5c573f45aec2fbfd4d199906248044a7ae49af07015249daba",
    SHIP_PATH + "engine_diff.dds": "b2af67206d7f919ce511301fbf7bba4b3ae56837a923e00435c4f49d3fc047b3",
    SHIP_PATH + "engine_glow.dds": "b830fb322c80b41f19b902ed71887e3da9a9924a774287d1beb321d1d537d91c",
    SHIP_PATH + "engine_norm.png": "25393a33d5725b1b38060aeeb48b36d839be905ab61b0c3d283ba76f19194614",
    SHIP_PATH + "engine_spec.dds": "fb5e04c90e09e6679b45f143c0aa1e6b9c09faa71bf52111cb2e84c85833e078",
    SHIP_PATH + "fuselage_diff.dds": "88e6f4ea05c9823e8772a72dc94c3bda83db0316095a2f1aec106bd1f60b4d78",
    SHIP_PATH + "fuselage_glow.dds": "701473acf5e67764989b160fb3421fb8044f330d1c152e2b95c113862e636a30",
    SHIP_PATH + "fuselage_norm.png": "90d291944df0401f4dc1f1a87a30e57774489c3f766c7df17d0d4232589bdebf",
    SHIP_PATH + "fuselage_spec.dds": "48e1f219f398850129c52c363e2e858cedc7a53b00f4e18f55dede9fd51d045f",
}


def source_files(cache: Path) -> tuple[dict[str, Path], list[dict]]:
    cache.mkdir(parents=True, exist_ok=True)
    files, manifest = {}, []
    for source, expected in HASHES.items():
        local = cache / Path(source).name
        if not local.exists():
            with urllib.request.urlopen(RAW + source, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"SHA-256 не совпадает: {source}")
            local.write_bytes(data)
        if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SHA-256 не совпадает в cache: {local}")
        files[local.name] = local
        manifest.append({"path": source, "sha256": expected,
                         "bytes": local.stat().st_size, "url": RAW + source})
    return files, manifest


def float_source(element: ET.Element) -> np.ndarray:
    accessor = element.find("c:technique_common/c:accessor", NS)
    stride = int(accessor.get("stride", "1"))
    offset = int(accessor.get("offset", "0"))
    count = int(accessor.get("count"))
    raw = np.fromstring(element.find("c:float_array", NS).text, sep=" ")
    return raw[offset:offset + count * stride].reshape(count, stride)


def cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def triangulate(positions: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping в проекции Newell-нормали; UV/нормали не сливаются."""
    count = len(positions)
    if count == 3:
        return [(0, 1, 2)]
    newell = np.cross(positions, np.roll(positions, -1, axis=0)).sum(axis=0)
    axis = int(np.argmax(np.abs(newell)))
    points = np.delete(positions, axis, axis=1)
    area = sum(cross2(a, b) for a, b in zip(points, np.roll(points, -1, axis=0)))
    sign = 1 if area >= 0 else -1
    epsilon = max(float(np.ptp(points, axis=0).max()) ** 2 * 1e-12, 1e-18)
    corners = [cross2(points[i] - points[i - 1],
                      points[(i + 1) % count] - points[i]) * sign
               for i in range(count)]
    if min(corners) >= -epsilon:
        return [(0, i, i + 1) for i in range(1, count - 1)]
    remaining = list(range(count))
    triangles = []
    while len(remaining) > 3:
        for k, current in enumerate(remaining):
            previous, following = remaining[k - 1], remaining[(k + 1) % len(remaining)]
            a, b, c = points[[previous, current, following]]
            if cross2(b - a, c - b) * sign <= epsilon:
                continue
            inside = False
            for other in remaining:
                if other in (previous, current, following):
                    continue
                p = points[other]
                if (cross2(b - a, p - a) * sign > epsilon and
                        cross2(c - b, p - b) * sign > epsilon and
                        cross2(a - c, p - c) * sign > epsilon):
                    inside = True
                    break
            if inside:
                continue
            triangles.append((previous, current, following))
            remaining.pop(k)
            break
        else:
            # A collinear corner can block an ear. Removing it loses no area.
            for k, current in enumerate(remaining):
                a, b, c = points[[remaining[k - 1], current,
                                  remaining[(k + 1) % len(remaining)]]]
                if abs(cross2(b - a, c - b)) <= epsilon:
                    remaining.pop(k)
                    break
            else:
                raise ValueError("Нельзя корректно триангулировать исходный полигон")
    triangles.append(tuple(remaining))
    return triangles


def read_mesh(path: Path) -> tuple[np.ndarray, dict]:
    root = ET.parse(path).getroot()
    assert root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    elements = {element.get("id"): element for element in root.iter()
                if element.get("id")}
    sources = {key: float_source(element) for key, element in elements.items()
               if element.tag.endswith("}source") and element.find("c:float_array", NS) is not None}
    material_names = {element.get("id"): element.get("name")
                      for element in root.findall("c:library_materials/c:material", NS)}
    # Exported default matrices are at frame 250 (gear down). Pick frame 1
    # explicitly, the beginning of .model's gear_down animation, in every LOD.
    animated = {}
    for channel in root.findall(".//c:channel", NS):
        node_id, sid = channel.get("target").split("/")
        if sid != "transform":
            raise ValueError(f"Неизвестная анимация: {channel.attrib}")
        sampler = elements[channel.get("source")[1:]]
        streams = {e.get("semantic"): e.get("source")[1:] for e in sampler}
        animated[node_id] = sources[streams["OUTPUT"]][0].reshape(4, 4)
    stats = {"instances": 0, "animated_nodes_frame_1": len(animated),
             "ignored_line_primitives": 0, "flipped_mirrored_triangles": 0,
             "removed_degenerate_triangles": 0, "repaired_unbound_primitives": 0}
    chunks = []

    def visit(node, parent):
        transform = animated.get(node.get("id"))
        if transform is None:
            matrix = node.find("c:matrix", NS)
            transform = (np.fromstring(matrix.text, sep=" ").reshape(4, 4)
                         if matrix is not None else np.eye(4))
        world = parent @ transform
        linear = AXES @ world[:3, :3]
        normal_matrix = np.linalg.inv(linear).T
        for instance in node.findall("c:instance_geometry", NS):
            stats["instances"] += 1
            bindings = {entry.get("symbol"): material_names[entry.get("target")[1:]]
                        for entry in instance.findall("c:bind_material/c:technique_common/c:instance_material", NS)}
            mesh = elements[instance.get("url")[1:]].find("c:mesh", NS)
            vertices = {entry.get("id"): {e.get("semantic"): e.get("source")[1:] for e in entry}
                        for entry in mesh.findall("c:vertices", NS)}
            for primitive in mesh:
                kind = primitive.tag.split("}")[1]
                if kind in ("source", "vertices"):
                    continue
                if kind == "lines":
                    stats["ignored_line_primitives"] += 1
                    continue
                if kind not in ("polylist", "triangles"):
                    raise ValueError(f"Неизвестная геометрия: {kind}")
                inputs = list(primitive.findall("c:input", NS))
                stride = max(int(e.get("offset", "0")) for e in inputs) + 1
                binding = {}
                for entry in inputs:
                    semantic = entry.get("semantic")
                    source = entry.get("source")[1:]
                    offset = int(entry.get("offset", "0"))
                    if semantic == "VERTEX":
                        binding["POSITION"] = (sources[vertices[source]["POSITION"]], offset)
                    elif semantic != "TEXCOORD" or entry.get("set", "0") == "0":
                        binding[semantic] = (sources[source], offset)
                if not {"POSITION", "NORMAL", "TEXCOORD"} <= binding.keys():
                    raise ValueError(f"Нет координат/нормалей/UV: {instance.attrib}")
                indices = np.fromstring(primitive.find("c:p", NS).text, sep=" ", dtype=np.int64).reshape(-1, stride)
                polygon_sizes = ([3] * int(primitive.get("count")) if kind == "triangles" else
                                 [int(v) for v in primitive.find("c:vcount", NS).text.split()])
                assert sum(polygon_sizes) == len(indices)
                positions = binding["POSITION"][0][indices[:, binding["POSITION"][1]], :3]
                normals = binding["NORMAL"][0][indices[:, binding["NORMAL"][1]], :3]
                uv = binding["TEXCOORD"][0][indices[:, binding["TEXCOORD"][1]], :2]
                triangles, start = [], 0
                for size in polygon_sizes:
                    triangles.extend(tuple(start + i for i in tri)
                                     for tri in triangulate(positions[start:start + size]))
                    start += size
                triangle_indices = np.array(triangles, dtype=np.int64)
                positions = (positions @ world[:3, :3].T + world[:3, 3]) @ AXES.T
                normals = normals @ normal_matrix.T
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                material = primitive.get("material")
                if material is None and len(bindings) == 1:
                    # Upstream med/low wheel primitives omit their symbol, but
                    # the instance binds exactly one material, Hull; the UVs
                    # still refer to that same fuselage atlas as the high LOD.
                    material = next(iter(bindings))
                    stats["repaired_unbound_primitives"] += 1
                slot = SLOTS[bindings[material]]
                data = np.column_stack((positions, normals, uv,
                                        np.full(len(positions), float(slot))))[triangle_indices]
                if np.linalg.det(linear) < 0:
                    data = data[:, [0, 2, 1], :]
                    stats["flipped_mirrored_triangles"] += len(data)
                cross = np.cross(data[:, 1, :3] - data[:, 0, :3],
                                 data[:, 2, :3] - data[:, 0, :3])
                valid = np.linalg.norm(cross, axis=1) > 1e-12
                stats["removed_degenerate_triangles"] += int((~valid).sum())
                chunks.append(data[valid].reshape(-1, 9))
        for child in node.findall("c:node", NS):
            visit(child, world)

    scene_url = root.find("c:scene/c:instance_visual_scene", NS).get("url")[1:]
    for node in elements[scene_url].findall("c:node", NS):
        visit(node, np.eye(4))
    data = np.concatenate(chunks)
    triangles = data.reshape(-1, 3, 9)
    cross = np.cross(triangles[:, 1, :3] - triangles[:, 0, :3],
                     triangles[:, 2, :3] - triangles[:, 0, :3])
    agreement = np.einsum("ij,ij->i", cross, triangles[:, :, 3:6].sum(axis=1))
    backwards = agreement < 0
    # A handful of authored slivers / non-planar polygons disagree with the
    # source's outward vertex normals after triangulation. Keep those normals
    # and make winding agree explicitly, rather than using two-sided rendering.
    triangles[backwards] = triangles[backwards][:, [0, 2, 1], :]
    stats["repaired_normal_winding_triangles"] = int(backwards.sum())
    stats["triangles"] = len(triangles)
    stats["triangles_per_material"] = {name: int((triangles[:, 0, 8] == slot).sum())
                                       for name, slot in SLOTS.items()}
    return data, stats


def texture(source: Path, output: Path, size: int, specular_gain=1.) -> dict:
    image = Image.open(source).convert("RGB")
    source_size = list(image.size)
    if max(image.size) > size:
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
    pixels = np.array(image)
    if source.stem.endswith("_norm") and list(image.size) != source_size:
        normal = pixels.astype(np.float64) / 127.5 - 1.
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
        pixels = np.clip(np.rint((normal + 1.) * 127.5), 0, 255).astype(np.uint8)
    if specular_gain != 1.:
        pixels = np.rint(pixels.astype(np.float64) * specular_gain).astype(np.uint8)
    Image.fromarray(pixels).save(output, optimize=True)
    return {"source": source.name, "file": output.name,
            "source_size": source_size, "size": list(image.size),
            "specular_gain": specular_gain}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/elite-pioneer-source"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "asset_packs/pioneer")
    parser.add_argument("--texture-size", type=int, default=2048)
    args = parser.parse_args()
    files, source_manifest = source_files(args.cache)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    mesh_data = [read_mesh(files[f"sinonatrix_{level}.dae"]) for level in ("hi", "med", "low")]
    # A single bbox and radius across all LOD avoids size/position jumps.
    bounds = np.array([data[:, :3].min(axis=0) for data, _ in mesh_data]).min(axis=0), np.array([
        data[:, :3].max(axis=0) for data, _ in mesh_data]).max(axis=0)
    center = (bounds[0] + bounds[1]) * .5
    radius = max(float(np.linalg.norm(data[:, :3] - center, axis=1).max())
                 for data, _ in mesh_data)
    mesh_stats = {}
    for level, (data, stats) in zip(("hi", "med", "low"), mesh_data):
        data[:, :3] = (data[:, :3] - center) / radius
        vertices = data.astype(np.float32)
        assert np.isfinite(vertices).all()
        np.savez_compressed(output / f"sinonatrix_{level}.npz", vertices=vertices)
        stats["bounds"] = [vertices[:, :3].min(axis=0).tolist(), vertices[:, :3].max(axis=0).tolist()]
        stats["radius"] = float(np.linalg.norm(vertices[:, :3], axis=1).max())
        mesh_stats[level] = stats
    texture_stats = []
    for part, gain in (("fuselage", .8), ("engine", .4)):
        for role, suffix in (("diffuse", "diff.dds"), ("normal", "norm.png"),
                             ("specular", "spec.dds"), ("emission", "glow.dds")):
            texture_stats.append(texture(files[f"{part}_{suffix}"], output / f"{part}_{role}.png",
                                         args.texture_size, gain if role == "specular" else 1.))
    materials = []
    for name, part, shininess, opacity in (("Hull", "fuselage", 100, 1.),
                                          ("Canopy", "fuselage", 120, .9),
                                          ("Engine", "engine", 80, 1.)):
        materials.append({"name": name, **{role: f"{part}_{role}.png" for role in (
            "diffuse", "normal", "specular", "emission")},
                          "shininess_source": shininess, "opacity_source": opacity,
                          "gloss": round((.85 - (2. / (shininess + 2.)) ** .25) / .66, 6),
                          "procedural_type": {"Hull": 0, "Canopy": 2, "Engine": 1}[name]})
    pack = {
        "id": "pioneer", "title": "Pioneer", "author": "Bálint Szilárd (bszlrd), Pioneer Developers",
        "license": "CC BY-SA 3.0", "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "source": f"{REPOSITORY}/tree/{REVISION}/{SHIP_PATH.rstrip('/')}", "revision": REVISION,
        "models": {"20": {"name": "Sinonatrix Courier", "mesh": "sinonatrix_hi.npz",
                           "lods": ["sinonatrix_med.npz", "sinonatrix_low.npz"],
                           "materials": materials}},
    }
    (output / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n")
    metadata = output / "source_metadata"
    metadata.mkdir(exist_ok=True)
    for name in ("AUTHORS.txt", "sinonatrix.model", "sinonatrix_cockpit.model"):
        shutil.copyfile(files[name], metadata / name)
    shutil.copyfile(files["CC-BY-SA-3.0.txt"], output / "LICENSE-CC-BY-SA-3.0.txt")
    manifest = {"repository": REPOSITORY, "revision": REVISION, "sources": source_manifest,
                "normalization": {"axis_rotation": AXES.tolist(), "center_before_scale": center.tolist(),
                                  "radius_before_scale": radius, "same_for_all_lods": True},
                "meshes": mesh_stats, "textures": texture_stats}
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for level, stats in mesh_stats.items():
        print(level, json.dumps(stats, ensure_ascii=False))
    print(f"Пакет: {output}; {sum(p.stat().st_size for p in output.rglob('*') if p.is_file()):,} байт")


if __name__ == "__main__":
    main()
