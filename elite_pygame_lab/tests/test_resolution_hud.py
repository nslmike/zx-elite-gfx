"""Меню разрешения и его области щелчков в атласе и автопролёте."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import pygame

try:
    from elite_lab.space_hud import SpaceHUD
    from elite_lab.space_resolution import RENDER_RESOLUTIONS
except ModuleNotFoundError:
    SpaceHUD = None


@unittest.skipIf(SpaceHUD is None, "Нужны зависимости requirements-space.txt")
class ResolutionHUDTest(unittest.TestCase):
    def setUp(self):
        pygame.font.init()
        self.addCleanup(pygame.font.quit)

    @staticmethod
    def scene(flyby, mode, source="elite"):
        ship = SimpleNamespace(name="Cobra Mk III", materials=(), meshes=(), source_id="elite",
                               source_title="ZX Spectrum Elite")
        return SimpleNamespace(is_flyby=flyby, camera_mode=mode, selected=0, ships=(ship,),
                               model_source=source, texture_source="author", auto_sources=False,
                               elapsed=0, paused=False)

    def test_only_resolution_controls_respond_while_open_in_both_modes(self):
        expected = [f"resolution_{i}" for i in range(len(RENDER_RESOLUTIONS))] + ["close_menu"]
        for flyby, mode in ((False, "inspect"), (False, "flight"), (True, "flyby"), (True, "flight")):
            for logical in ((320, 240), (640, 360), (1360, 820)):
                with self.subTest(flyby=flyby, mode=mode, logical=logical):
                    hud = SpaceHUD(MagicMock())
                    scene = self.scene(flyby, mode)
                    centers = {}
                    for ratio in (1, 2):
                        drawable = tuple(value * ratio for value in logical)
                        hud._resize(drawable, ratio)
                        hud.menu = None
                        surface = hud._paint(scene, None, None)
                        button = hud.options["menu_resolution"]
                        self.assertTrue(surface.get_rect().contains(button))
                        self.assertEqual(hud.tab_at(button.center), "menu_resolution")
                        for action, rect in {**hud.options, **hud.tabs}.items():
                            if action != "menu_resolution":
                                self.assertFalse(rect.colliderect(button), action)
                        hud.menu = "resolution"
                        self.assertEqual(hud.tab_at(button.center), "close_menu")
                        hud._paint(scene, None, None)
                        self.assertEqual(list(hud.options), expected)
                        self.assertEqual(hud.tab_at((0, 0)), "close_menu")
                        for action, rect in hud.options.items():
                            self.assertTrue(surface.get_rect().contains(rect))
                            self.assertEqual(hud.tab_at(rect.center), action)
                            if ratio == 1:
                                centers[action] = rect.center
                            else:
                                pixel = tuple(value * ratio for value in centers[action])
                                self.assertEqual(hud.tab_at(pixel), action)
                        hud.menu = None
                        self.assertIsNone(hud.tab_at(hud.options["resolution_3"].center))
                    hud.release()

    def test_labels_show_native_pixel_size_current_preset_limit_and_zx_note(self):
        hud = SpaceHUD(MagicMock())
        hud._resize((640, 480), 2)
        hud.menu = "resolution"
        hud.resolution.output_size = (640, 480)
        hud.resolution.selected = 4
        hud.resolution.highlighted = 5
        hud.resolution.unavailable.add(6)
        hud.text = Mock(wraps=hud.text)
        scene = self.scene(True, "flyby", source="zx")
        hud._paint(scene, None, None)
        rendered = [call.args[0] for call in hud.text.render.call_args_list]
        options = [label for label in rendered if "p · " in label or label.startswith("Нативное · ")]
        self.assertEqual(options, ["Нативное · 640 × 480", "540p · 960 × 540", "720p · 1280 × 720",
                                   "900p · 1600 × 900", "1080p · 1920 × 1080", "1440p · 2560 × 1440",
                                   "2160p · 3840 × 2160"])
        self.assertEqual(rendered[rendered.index(options[4])+1], "Текущее")
        self.assertEqual(rendered[rendered.index(options[6])+1], "Недоступно")
        self.assertIn("Каркас ZX · 256 × 128. Выбор для цветной сцены.", rendered)
        hud.release()

    def test_long_error_wraps_and_preserves_option_positions_in_small_window(self):
        hud = SpaceHUD(MagicMock())
        hud._resize((320, 240))
        hud.menu = "resolution"
        hud.text = Mock(wraps=hud.text)
        scene = self.scene(False, "inspect")
        hud._paint(scene, None, None)
        before = hud.options.copy()
        error = "Не удалось включить 3840 × 2160. Выберите другой размер."
        hud.resolution.error = error
        hud.text.render.reset_mock()
        hud._paint(scene, None, None)
        warnings = [call.args[0] for call in hud.text.render.call_args_list
                    if call.args[0] in error]
        self.assertEqual(len(warnings), 2)
        self.assertEqual(" ".join(warnings), error)
        self.assertEqual(hud.options, before)
        hud.release()

    def test_cached_layer_tracks_open_menu_state_and_drawable_size(self):
        hud = SpaceHUD(MagicMock())
        scene = self.scene(False, "inspect")
        state = hud.resolution
        with patch.object(hud, "_paint", wraps=hud._paint) as paint:
            hud.draw(scene, (640, 480), None, None)
            state.selected = 1
            hud.draw(scene, (640, 480), None, None)
            self.assertEqual(paint.call_count, 1)
            hud.menu = "resolution"
            hud.draw(scene, (640, 480), None, None)
            self.assertEqual(paint.call_count, 2)
            changes = {"selected": 2, "highlighted": 3, "unavailable": {6},
                       "error": "Размер недоступен", "output_size": (1280, 960)}
            for count, (name, value) in enumerate(changes.items(), start=3):
                setattr(state, name, value)
                hud.draw(scene, (640, 480), None, None)
                hud.draw(scene, (640, 480), None, None)
                self.assertEqual(paint.call_count, count)
            hud.draw(scene, (1280, 960), None, None, pixel_ratio=2)
            self.assertEqual(paint.call_count, 8)
            hud.menu = None
            hud.draw(scene, (1280, 960), None, None, pixel_ratio=2)
            self.assertEqual(paint.call_count, 9)
        hud.release()


if __name__ == "__main__":
    unittest.main()
