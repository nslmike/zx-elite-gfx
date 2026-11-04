"""Регрессии выбора корпуса, его UV-карт и источника в текущей камере."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from elite_lab.space_assets import AssetCatalog, AssetError

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


class SpaceSourcesTest(unittest.TestCase):
    def test_sources_cover_their_actual_ships_and_preserve_catalog_ids(self):
        catalog = AssetCatalog()
        self.assertEqual(set(catalog.pack("oolite")), set(range(20)))
        self.assertEqual(set(catalog.pack("griff")), {8, 9, 12, 16})
        self.assertEqual(set(catalog.pack("pioneer")), {20})
        self.assertEqual(catalog.fleet("griff")[17].source_id, "elite")
        combined = catalog.fleet("mixed")
        self.assertEqual([s.model_id for s in combined], list(range(21)))
        self.assertEqual(combined[9].source_id, "griff")
        self.assertEqual(combined[17].source_id, "oolite")
        self.assertEqual(combined[20].name, "Sinonatrix Courier")
        self.assertEqual(combined[9].meshes, combined[12].meshes)
        self.assertNotEqual(combined[9].materials[0].diffuse, combined[12].materials[0].diffuse)
        for asset in combined:
            self.assertTrue(asset.author and asset.license and asset.materials)
            self.assertTrue(all(m.diffuse.is_file() for m in asset.materials))

    def test_optional_pack_cannot_break_the_builtin_catalog(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp) / "griff"
            directory.mkdir()
            (directory / "pack.json").write_text("{}", encoding="utf-8")
            catalog = AssetCatalog(temp)
            self.assertEqual(len(catalog.fleet("elite")), 20)
            with self.assertRaises(AssetError):
                catalog.fleet("griff")
            with self.assertRaises(AssetError):
                catalog.fleet("pioneer")

    @unittest.skipIf(np is None, "Для проверки геометрии нужен numpy из requirements-space.txt")
    def test_native_meshes_material_slots_and_lods(self):
        known = AssetCatalog().fleet("mixed")
        self.assertEqual(len(known[9].vertices()) // 3, 11506)
        courier = known[20]
        sizes = []
        for index in range(len(courier.meshes)):
            mesh = courier.vertices(index)
            sizes.append(len(mesh) // 3)
            self.assertEqual(mesh.shape[1], 9)
            self.assertLessEqual(np.linalg.norm(mesh[:, :3], axis=1).max(), 1.001)
            self.assertLess(mesh[:, 8].max(), len(courier.materials))
        self.assertEqual(sizes, [49804, 13264, 6742])

    @unittest.skipIf(np is None, "Для проверки камеры нужен numpy из requirements-space.txt")
    def test_switching_sources_and_textures_in_an_inspection(self):
        from elite_lab.space_scene import SpaceScene
        scene = SpaceScene()
        scene.select(9)
        scene.drag(34, -21)
        view = scene.forward.copy()
        scene.set_source("griff")
        self.assertEqual((scene.camera_mode, scene.selected), ("inspect", 9))
        np.testing.assert_allclose(scene.forward, view)
        pose = scene.view_projection(1.6).copy()
        scene.cycle_textures()
        self.assertEqual(scene.texture_source, "procedural")
        np.testing.assert_array_equal(scene.view_projection(1.6), pose)
        scene.set_source("pioneer")
        self.assertEqual((scene.camera_mode, scene.selected, len(scene.objects)), ("inspect", 20, 21))
        scene.set_source("elite")
        self.assertEqual((scene.selected, len(scene.objects)), (9, 20))
