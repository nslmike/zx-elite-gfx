"""Ручной показ выбранного флота не зависит от посторонних наборов."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_flyby import FlybyScene
    from elite_lab.space_viewer import SpaceRenderer
except ModuleNotFoundError:
    SpaceRenderer = None


@unittest.skipIf(SpaceRenderer is None, "Нужны зависимости requirements-space.txt")
class SpacePreparationTest(unittest.TestCase):
    def test_manual_elite_launch_does_not_read_a_broken_optional_pack(self):
        with TemporaryDirectory() as directory:
            pack = Path(directory) / "griff"
            pack.mkdir()
            (pack / "pack.json").write_text("{broken optional pack", encoding="utf-8")
            catalog = AssetCatalog(directory)
            self.assertIn("griff", catalog.pack_ids)
            scene = FlybyScene(model_source="elite", source_interval=0, catalog=catalog)
            renderer = SpaceRenderer.__new__(SpaceRenderer)
            renderer.ctx = SimpleNamespace(finish=Mock())
            renderer.look = SimpleNamespace(use_material=Mock())
            renderer._mesh_buffer = Mock()
            with patch.object(catalog, "pack", wraps=catalog.pack) as read_pack:
                renderer.warm_sources(scene)
                read_pack.assert_not_called()
            renderer._mesh_buffer.assert_not_called()
            renderer.look.use_material.assert_not_called()
            renderer.ctx.finish.assert_not_called()
            self.assertEqual(scene.model_source, "elite")
            self.assertEqual(scene.camera_mode, "flyby")
            self.assertFalse(scene.auto_sources)


if __name__ == "__main__":
    unittest.main()
