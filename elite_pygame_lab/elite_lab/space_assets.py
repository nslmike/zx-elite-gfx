"""Локальные источники геометрии и авторских карт космического атласа."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from .model import DODO_ID, MODELS


PACKS = Path(__file__).resolve().parents[1] / "asset_packs"
SOURCE_NAMES = {"elite": "Elite", "oolite": "Oolite", "griff": "Griff SpecGloss",
                "pioneer": "Pioneer", "mixed": "Сборный флот"}
FLYBY_SOURCE_NAMES = {"zx": "ZX · пиксельный каркас", "elite": "Elite · наша обшивка",
                      **{key: value for key, value in SOURCE_NAMES.items() if key != "elite"}}
TEXTURE_NAMES = {"author": "Из набора", "procedural": "Процедурные"}


@dataclass(frozen=True)
class Material:
    diffuse: Path
    normal: Path | None = None
    specular: Path | None = None
    emission: Path | None = None
    gloss_map: Path | None = None
    gloss: float = .5
    emission_strength: float = 1.
    specular_color: tuple[float, float, float] = (1., 1., 1.)
    procedural_type: int = 0


@dataclass(frozen=True)
class ShipAsset:
    model_id: int
    name: str
    source_id: str
    source_title: str
    author: str
    license: str
    meshes: tuple[Path, ...] = ()
    materials: tuple[Material, ...] = ()
    cross_bitangent: bool = False

    def vertices(self, lod=0):
        if self.meshes:
            return _read_mesh(self.meshes[lod], len(self.materials))
        from .space_mesh import build_mesh
        return build_mesh(self.model_id)


class AssetError(ValueError):
    """Неполный набор или неподходящие данные модели."""


@lru_cache(maxsize=64)
def _read_mesh(path, material_count):
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as archive:
            mesh = np.ascontiguousarray(archive["vertices"], dtype=np.float32)
        if mesh.ndim != 2 or mesh.shape[1] != 9 or not len(mesh) or len(mesh) % 3 or not np.isfinite(mesh).all():
            raise ValueError("ожидаются конечные треугольники из 9 компонент")
        slots = mesh[:, 8].reshape(-1, 3)
        if (np.any(slots < 0) or np.any(slots >= material_count) or np.any(slots != np.floor(slots))
                or np.any(slots != slots[:, :1])):
            raise ValueError("у треугольника неверный индекс материала")
    except (OSError, KeyError, ValueError) as exc:
        raise AssetError(f"Ошибка сетки {path.name}: {exc}") from exc
    mesh.setflags(write=False)
    return mesh


def _load_pack(directory):
    def asset_file(relative):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"отсутствует файл набора {relative}")
        return path

    try:
        data = json.loads((directory / "pack.json").read_text(encoding="utf-8"))
        if data["id"] != directory.name:
            raise ValueError("имя каталога не совпадает с id")
        assets = {}
        for key, model in data["models"].items():
            index = int(key)
            materials = []
            for m in model["materials"]:
                maps = {k: asset_file(m[k]) for k in ("normal", "specular", "emission", "gloss_map") if m.get(k)}
                default_specular = (1., 1., 1.) if "specular" in maps else (.18, .18, .18)
                materials.append(Material(
                    asset_file(m["diffuse"]), **maps,
                    gloss=float(m.get("gloss", 1. if "gloss_map" in maps else .5)),
                    emission_strength=float(m.get("emission_strength", 1.)),
                    specular_color=tuple(m.get("specular_color", default_specular)),
                    procedural_type=int(m.get("procedural_type", 5 if index in (3, 5, 6, 7) else 0)),
                ))
            if not materials:
                raise ValueError(f"у модели {index} отсутствуют материалы")
            meshes = tuple(asset_file(f) for f in (model["mesh"], *model.get("lods", ())))
            name = MODELS[index].name if 0 <= index < len(MODELS) else model["name"]
            assets[index] = ShipAsset(index, name, data["id"], data["title"], model.get("author", data["author"]),
                                      model.get("license", data["license"]), meshes, tuple(materials),
                                      data.get("normal_basis") == "cross(N,T)")
        if not assets:
            raise ValueError("набор пуст")
        return assets
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise AssetError(f"Набор {directory.name}: {exc}") from exc


class AssetCatalog:
    def __init__(self, root=PACKS):
        self.root = Path(root).resolve()
        self.pack_ids = tuple(key for key in ("oolite", "griff", "pioneer") if (self.root / key / "pack.json").is_file())
        self.sources = ("elite", *self.pack_ids) + (("mixed",) if self.pack_ids else ())
        self._packs = {}
        self.classic = tuple(ShipAsset(i, model.name, "elite", "Elite Legend 1.09" if i == DODO_ID else "ZX Spectrum Elite",
                                       "PLM (версия Elite Legend)" if i == DODO_ID else "Ian Bell, David Braben (Elite)",
                                       "Отдельная лицензия на каркас не указана")
                             for i, model in enumerate(MODELS))

    def pack(self, source):
        if source not in self._packs:
            self._packs[source] = _load_pack(self.root / source)
        return self._packs[source]

    def fleet(self, source):
        if source not in self.sources:
            raise AssetError(f"Источник «{source}» недоступен. Установлены: {', '.join(self.sources)}")
        ships = dict(enumerate(self.classic))
        if source == "mixed":
            # Для совпадающего ID SpecGloss имеет приоритет перед базовой Oolite.
            for pack_id in ("oolite", "pioneer", "griff"):
                if pack_id in self.pack_ids:
                    ships.update(self.pack(pack_id))
        elif source != "elite":
            ships.update(self.pack(source))
        if set(ships) != set(range(len(ships))):
            raise AssetError("Номера дополнительных объектов должны продолжать каталог Elite")
        return tuple(ships[i] for i in range(len(ships)))
