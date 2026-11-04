#!/usr/bin/env python3
"""Воспроизводимый импорт реальных авторских Griff SpecGloss, без Blender.

Нужны numpy и Pillow. Например:
  python3 tools/import_griff_pack.py --cache /tmp --download

Публичные ZIP взяты из папки, опубликованной самим Griff. Их не следует
путать с более старыми моделями базовой Oolite. Исходные архивы остаются
в cache; в репозиторий попадают только выбранные сетки, карты и notices.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
from zipfile import ZipFile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FORUM = "https://bb.oolite.space/viewtopic.php?t=21170"
FOLDER = "https://drive.google.com/drive/folders/1_8EtoVvwNyXXvqe5GCGy1HuFRRw3NxB7"
SOURCES = {
    "cobra3-oolite": ("1r3cDqy5qMIo2suZVioNx8HwxxLejrlW_", "45b0063ecd43d2c55c9c9a7845ce1de8eaade612667c38f84f1132409feddae7"),
    "viper-oolite": ("1yXCDGWhK4yDeL0PQpLEI7LLI2ogLMwFW", "19329c1171a78430a40fe4298cc969e9a24d2feb5e3d9aae3adbc0315d30aeab"),
    "sidewinder-oolite": ("1tacwmQ4WcDr6kM0O4jRWzAJCjyewtvFS", "295c0604a773de4ca77e83d432ae8a5e373916681443c075bd77a1ef46413c07"),
}
NOTICES = {
    "sidewinder-folder-license": "13iha6rYEWwpO-cqmBfVf2nRTXcsLyJ0j",
    "root-folder-license": "1J0J7AEgD1cl1znDLOiklBDKSgl5eFnZz",
}
NOTICE_SHA256 = "cde3207f0204f9ca55c6a419340ce8ad278711e38f505cf0d68ca88dcc8d2b6c"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_archive(cache: Path, key: str, download: bool) -> tuple[ZipFile, dict]:
    file_id, sha256 = SOURCES[key]
    path = cache / f"griff-{key}.zip"
    if not path.exists() and download:
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            "--output", str(path),
        ], check=True)
    data = path.read_bytes()
    if digest(data) != sha256:
        raise ValueError(f"{path}: другой SHA-256; исходник обновился или загрузился HTML")
    archive = ZipFile(io.BytesIO(data))
    # Архивы не извлекаются на диск. Всё равно проверяем имена до чтения.
    for entry in archive.namelist():
        p = PurePosixPath(entry.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Небезопасный путь ZIP: {entry}")
    return archive, {
        "file": path.name, "sha256": sha256, "bytes": len(data),
        "url": f"https://drive.google.com/file/d/{file_id}/view", "members": {},
    }


def member(archive: ZipFile, provenance: dict, suffix: str) -> bytes:
    found = [n for n in archive.namelist() if n.endswith(suffix)]
    if len(found) != 1:
        raise ValueError(f"Требуется один {suffix}, найдено {found}")
    data = archive.read(found[0])
    provenance["members"][found[0]] = {"sha256": digest(data), "bytes": len(data)}
    return data


def mesh_from_dat(data: bytes) -> tuple[np.ndarray, dict]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in data.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        key = line.split()[0]
        if key in {"VERTEX", "FACES", "TEXTURES", "NORMALS", "TANGENTS", "NAMES", "END"}:
            current = key
            sections[current] = []
        elif current:
            sections[current].append(line)
    positions = np.array([list(map(float, v.split())) for v in sections["VERTEX"]], dtype=np.float64)
    normals = np.array([list(map(float, n.split())) for n in sections["NORMALS"]], dtype=np.float64)
    faces = np.array([list(map(int, f.split()[7:])) for f in sections["FACES"]], dtype=np.int32)
    texture_rows = [row.split() for row in sections["TEXTURES"]]
    if any(len(row) != 9 for row in texture_rows) or faces.shape[1] != 3:
        raise ValueError("Ожидались текстурированные треугольники DAT")
    slots = np.array([int(row[0]) for row in texture_rows])
    uv = np.array([list(map(float, row[3:])) for row in texture_rows]).reshape(-1, 3, 2)
    scales = np.array([list(map(float, row[1:3])) for row in texture_rows])
    uv /= scales[:, None, :]
    if len(faces) != len(uv) or normals.shape != positions.shape:
        raise ValueError("Неполные normals или UV")

    # Обратное преобразование официального Obj2DatTexNorm.py:
    # Oolite DAT: леворукий X, UV с V вниз; GL: праворукий X, V вверх.
    reflection = np.array([-1., 1., 1.])
    src_points = positions[faces]
    src_normal = normals[faces].mean(axis=1)
    src_normal /= np.linalg.norm(src_normal, axis=1, keepdims=True)
    duv = uv[:, 1:] - uv[:, :1]
    det = duv[:, 0, 0] * duv[:, 1, 1] - duv[:, 0, 1] * duv[:, 1, 0]
    valid = np.abs(det) > 1e-12

    def tangents(points, coordinates, normal):
        d = coordinates[:, 1:] - coordinates[:, :1]
        e = points[:, 1:] - points[:, :1]
        t = (e[:, 0] * d[:, 1, 1, None] - e[:, 1] * d[:, 0, 1, None])
        t *= np.sign(d[:, 0, 0] * d[:, 1, 1] - d[:, 0, 1] * d[:, 1, 0])[:, None]
        t -= normal * np.sum(normal * t, axis=1, keepdims=True)
        return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-30)

    src_tangent = tangents(src_points[valid], uv[valid], src_normal[valid])
    positions *= reflection
    normals *= reflection
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    uv[:, :, 1] = 1. - uv[:, :, 1]
    gl_normal = src_normal[valid] * reflection
    gl_tangent = tangents(positions[faces][valid], uv[valid], gl_normal)
    t_dot = np.sum(gl_tangent * (src_tangent * reflection), axis=1)
    b_dot = np.sum(np.cross(gl_normal, gl_tangent) * (np.cross(src_normal[valid], src_tangent) * reflection), axis=1)
    if np.min(t_dot) < .999999 or np.max(b_dot) > -.999999:
        raise ValueError("Не подтверждён знак преобразования tangent basis")
    # Сохраняем нормали автора, а не пересчитываем их усреднением.
    centre = (positions.min(axis=0) + positions.max(axis=0)) * .5
    positions -= centre
    radius = float(np.linalg.norm(positions, axis=1).max())
    positions /= radius
    points = positions[faces]
    face_cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    outward = np.sum(face_cross * normals[faces].mean(axis=1), axis=1)
    if np.any(outward <= 0):
        raise ValueError("Не все грани CCW и согласованы с нормалями автора")
    mesh = np.concatenate((points, normals[faces], uv, np.repeat(slots[:, None, None], 3, axis=1)), axis=2)
    return mesh.astype(np.float32).reshape(-1, 9), {
        "triangles": len(faces), "material_names": sections["NAMES"],
        "source_bbox_centre": centre.tolist(), "source_radius": radius,
        "uv_degenerate_triangles": int((~valid).sum()),
        "basis_check": {
            "nondegenerate_triangles": int(valid.sum()),
            "min_dot_T_gl_M_T_source": float(t_dot.min()),
            "max_dot_cross_NT_gl_M_cross_NT_source": float(b_dot.max()),
        },
    }


def image_data(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def save_image(image: Image.Image, target: Path, limit: int):
    image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    image.save(target, optimize=True)


def make_material(archive, provenance, prefix: str, maps: dict, output: Path, limit: int) -> dict:
    # With specular_map Oolite uses specular_modulate_color (white default),
    # rather than the specular_color=(.2,.2,.2) fallback in shipdata.
    result = {"gloss": 1., "specular_color": [1., 1., 1.]}
    for kind, filename in maps.items():
        texture_limit = limit
        image = image_data(member(archive, provenance, "/Textures/" + filename))
        target = output / f"{prefix}_{kind}.png"
        if kind == "normal":
            # DAT→GL отражает X. При B=cross(N,T) это меняет знак B;
            # инвертируем G и после уменьшения нормализуем единичный вектор.
            image = image.convert("RGB")
            image.thumbnail((texture_limit, texture_limit), Image.Resampling.LANCZOS)
            normal = np.asarray(image).astype(np.float64) / 127.5 - 1.
            normal[:, :, 1] *= -1.
            normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-20)
            image = Image.fromarray(np.rint(np.clip((normal + 1.) * 127.5, 0, 255)).astype(np.uint8))
        elif kind == "specular":
            # Oolite treats RGB as colour (gamma 2.2), alpha as linear gloss.
            # Preserve the author's coloured specular response as linear RGB.
            alpha = image.convert("RGBA").getchannel("A")
            gloss_target = output / f"{prefix}_gloss.png"
            save_image(alpha, gloss_target, texture_limit)
            result["gloss_map"] = gloss_target.name
            rgb = np.asarray(image.convert("RGB")).astype(np.float64) / 255.
            image = Image.fromarray(np.rint(np.power(rgb, 2.2) * 255.).astype(np.uint8))
        else:
            image = image.convert("RGB")
        save_image(image, target, texture_limit)
        result[kind] = target.name
    return result


def build(args):
    output = args.output.resolve()
    if not 1 <= args.max_texture <= 4096:
        raise ValueError("max-texture должен быть от 1 до 4096")
    output.mkdir(parents=True, exist_ok=True)
    credits = ROOT / "asset_packs/griff/CREDITS.md"
    if output != credits.parent.resolve() and credits.is_file():
        (output / "CREDITS.md").write_bytes(credits.read_bytes())
    archives, provenance = {}, {}
    for key in SOURCES:
        archives[key], provenance[key] = read_archive(args.cache, key, args.download)
    info = {}
    for name, filename in (("cobra3", "griff_HP_cobramk3.dat"), ("viper", "griff_specgloss_viper.dat"), ("sidewinder", "griff_HP_Sidewinder.dat")):
        a, p = archives[name + "-oolite"], provenance[name + "-oolite"]
        mesh, info[name] = mesh_from_dat(member(a, p, "/Models/" + filename))
        np.savez_compressed(output / f"{name}.npz", vertices=mesh)

    cobra = {
        "diffuse": "griff_HP_Cobramk3_diffuse_blue.png", "normal": "griff_HP_Cobramk3_normal.png",
        "specular": "griff_HP_Cobramk3_specgloss_blue.png", "emission": "griff_HP_Cobramk3_glowmap.png",
    }
    materials = {}
    materials["cobra3"] = make_material(archives["cobra3-oolite"], provenance["cobra3-oolite"], "cobra3", cobra, output, args.max_texture)
    pirate = dict(cobra, diffuse="griff_HP_Cobramk3_diffuse_greyred.png", specular="griff_HP_Cobramk3_specgloss_greyred.png")
    # Общие normal/emission не дублируем для пиратской окраски.
    materials["cobra3_pirate"] = dict(materials["cobra3"])
    materials["cobra3_pirate"].update(make_material(
        archives["cobra3-oolite"], provenance["cobra3-oolite"], "cobra3_pirate",
        {k: v for k, v in pirate.items() if k in {"diffuse", "specular"}}, output, args.max_texture))
    materials["viper"] = []
    for part in ("Hull", "LightStrip"):
        maps = {"diffuse": f"griff_SpecGloss_Viper_{part}_Diffuse.png", "normal": f"griff_SpecGloss_Viper_{part}_Normal.png", "specular": f"griff_SpecGloss_Viper_{part}_SpecGloss.png"}
        if part == "Hull":
            maps["emission"] = "griff_SpecGloss_Viper_Hull_Emissive.png"
        materials["viper"].append(make_material(archives["viper-oolite"], provenance["viper-oolite"], "viper_" + part.lower(), maps, output, args.max_texture))
    materials["sidewinder"] = make_material(archives["sidewinder-oolite"], provenance["sidewinder-oolite"], "sidewinder", {
        "diffuse": "griff_HP_Sidewinder_albedo.png", "normal": "griff_HP_Sidewinder_normals.png",
        "specular": "griff_HP_Sidewinder_Specgloss.png", "emission": "griff_HP_Sidewinder_engineglow.png",
    }, output, args.max_texture)
    pack = {
        "id": "griff", "title": "Griff SpecGloss", "author": "Griff",
        "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "source": FORUM, "normal_basis": "cross(N,T)", "models": {
            "9": {"name": "Cobra Mk III (blue Trader)", "mesh": "cobra3.npz", "materials": [materials["cobra3"]],
                  "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"},
            "12": {"name": "Cobra Mk III (grey-red Pirate)", "mesh": "cobra3.npz", "materials": [materials["cobra3_pirate"]],
                   "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"},
            "8": {"name": "GalCop Viper", "mesh": "viper.npz", "materials": materials["viper"],
                  "license": "CC BY-NC-SA 4.0 (манифест), CC BY-NC 4.0 (файл лицензии)",
                  "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                  "notice_license_url": "https://creativecommons.org/licenses/by-nc/4.0/"},
            "16": {"name": "Sidewinder", "mesh": "sidewinder.npz", "materials": [materials["sidewinder"]],
                   "license": "CC BY-NC-SA 3.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/3.0/"},
        },
    }
    for name, filename, suffix in (
        ("cobra3", "LICENSE-cobra3-upstream.txt", ".oxp/licence.txt"),
        ("viper", "LICENSE-viper-upstream.txt", "/CC License.txt"),
        ("viper", "MANIFEST-viper-upstream.plist", ".oxp/manifest.plist"),
    ):
        data = member(archives[name + "-oolite"], provenance[name + "-oolite"], suffix)
        (output / filename).write_bytes(data)
    for name in ("cobra3", "viper", "sidewinder"):
        # shipdata is the authority for selected skins and material channels.
        data = member(archives[name + "-oolite"], provenance[name + "-oolite"], "/Config/shipdata.plist")
        (output / f"SHIPDATA-{name}-upstream.plist").write_bytes(data)
    notices = {}
    for key, file_id in NOTICES.items():
        cache_file = args.cache / f"griff-{key}.txt"
        if not cache_file.exists() and args.download:
            subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                "--output", str(cache_file),
            ], check=True)
        data = cache_file.read_bytes()
        if digest(data) != NOTICE_SHA256:
            raise ValueError(f"{cache_file}: изменился текст авторского уведомления")
        (output / f"LICENSE-{key}-upstream.txt").write_bytes(data)
        notices[key] = {"url": f"https://drive.google.com/file/d/{file_id}/view", "sha256": digest(data)}
    (output / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({
        "author": "Griff", "publication": FORUM, "folder": FOLDER,
        "archives": provenance, "notices": notices, "meshes": info, "max_texture": args.max_texture,
        "normal_basis": "cross(N,T)", "normal_conversion": "invert G, resize, renormalize",
        "specular_conversion": "sRGB RGB -> pow(rgb/255, 2.2) linear RGB; alpha -> linear gloss_map",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Griff SpecGloss: {sum(v['triangles'] for v in info.values())} треугольников, "
          f"{sum(p.stat().st_size for p in output.iterdir() if p.is_file()) / 1024**2:.2f} МиБ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, default=ROOT / "asset_packs/griff")
    parser.add_argument("--max-texture", type=int, default=1024)
    parser.add_argument("--download", action="store_true", help="скачать отсутствующие публичные авторские архивы")
    build(parser.parse_args())
