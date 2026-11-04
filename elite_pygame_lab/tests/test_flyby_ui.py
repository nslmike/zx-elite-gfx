"""Границы CLI, команды автопролёта и доступность его компактного HUD."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import pygame

from elite_lab.viewer import main

try:
    from elite_lab.space_hud import SpaceHUD
    from elite_lab.space_viewer import FlybyControls
except ModuleNotFoundError:
    SpaceHUD = FlybyControls = None


class FlybyCLITest(unittest.TestCase):
    def test_invalid_times_and_options_fail_before_opening_a_window(self):
        cases = [
            ["--mode", "flyby", "--headless", flag, value]
            for flag in ("--at", "--source-interval")
            for value in ("nan", "inf", "-inf", "-1", "нет")
        ]
        cases += [["--mode", "space", "--source-interval", "0"],
                  ["--mode", "space", "--headless", "--at", "0"],
                  ["--mode", "flyby", "--at", "0"]]
        for args in cases:
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)

    def test_reproducible_flyby_and_existing_space_share_the_entrypoint(self):
        module = ModuleType("elite_lab.space_viewer")
        module.run_space = Mock()
        with patch.dict(sys.modules, {module.__name__: module}):
            main(["--mode", "flyby", "--model", "9", "--model-source", "griff",
                  "--texture-source", "procedural", "--source-interval", "0",
                  "--at", "24.5", "--screenshot", "/tmp/elite-flyby-test.png", "--no-hud"])
            module.run_space.assert_called_once_with(
                9, Path("/tmp/elite-flyby-test.png"), False, False, "griff", "procedural",
                mode="flyby", source_interval=0.0, at=24.5)
            module.run_space.reset_mock()
            main(["--mode", "space", "--headless"])
            module.run_space.assert_called_once_with(
                None, None, True, True, "elite", "author", mode="space", source_interval=8.0, at=None)


@unittest.skipIf(FlybyControls is None, "Нужны зависимости requirements-space.txt")
class FlybyControlsTest(unittest.TestCase):
    def setUp(self):
        self.scene = Mock()
        self.hud = SimpleNamespace(menu=None, tab_at=Mock(return_value=None))
        self.controls = FlybyControls(self.scene, self.hud, True)

    def event(self, kind, **attrs):
        return self.controls.event(pygame.event.Event(kind, **attrs), (500, 300), (1000, 600))

    def test_shortcuts_and_escape_affect_only_the_presentation(self):
        for key, method in ((pygame.K_m, "cycle_source"), (pygame.K_t, "cycle_textures"),
                            (pygame.K_a, "toggle_auto_sources")):
            self.assertTrue(self.event(pygame.KEYDOWN, key=key))
            getattr(self.scene, method).assert_called_once_with()
        self.event(pygame.KEYDOWN, key=pygame.K_h)
        self.assertFalse(self.controls.show_hud)
        self.event(pygame.KEYDOWN, key=pygame.K_i)
        self.assertTrue(self.controls.show_hud)
        self.assertEqual(self.hud.menu, "credits")
        self.assertTrue(self.event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.assertIsNone(self.hud.menu)
        self.assertFalse(self.event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

    def test_camera_input_is_ignored_and_clicks_work_in_drawable_coordinates(self):
        self.event(pygame.MOUSEWHEEL, y=2)
        self.event(pygame.KEYDOWN, key=pygame.K_f)
        self.event(pygame.MOUSEBUTTONDOWN, button=1, pos=(10, 10))
        self.event(pygame.MOUSEMOTION, pos=(50, 60), rel=(40, 50))
        self.event(pygame.MOUSEBUTTONUP, button=1, pos=(50, 60))
        self.controls.update(.25)
        self.assertEqual(self.scene.mock_calls, [unittest.mock.call.update(.25)])
        self.hud.tab_at.return_value = "cycle_source"
        self.event(pygame.MOUSEBUTTONDOWN, button=1, pos=(20, 15))
        self.event(pygame.MOUSEBUTTONUP, button=1, pos=(20, 15))
        self.hud.tab_at.assert_called_with((40, 30))
        self.scene.cycle_source.assert_called_once_with()


@unittest.skipIf(SpaceHUD is None, "Нужны зависимости requirements-space.txt")
class FlybyHUDTest(unittest.TestCase):
    def test_all_four_actions_remain_inside_narrow_and_normal_frames(self):
        pygame.font.init()
        try:
            hud = SpaceHUD.__new__(SpaceHUD)
            hud.texture = None
            hud.ctx = SimpleNamespace(texture=lambda *_: SimpleNamespace(filter=None, release=lambda: None))
            hud.tabs, hud.markers, hud.options, hud.menu = {}, {}, {}, None
            ship = SimpleNamespace(name="Cobra Mk III (пиратская)", materials=(), source_id="elite",
                                   source_title="ZX Spectrum Elite", author="Ian Bell, David Braben",
                                   license="Отдельная лицензия на каркас не указана")
            scene = SimpleNamespace(ships=(ship,), selected=0, model_source="griff", texture_source="author",
                                    auto_sources=True)
            for size in ((320, 600), (600, 900), (1360, 820)):
                with self.subTest(size=size):
                    hud._resize(size)
                    surface = hud._paint_flyby(scene)
                    bounds = surface.get_rect()
                    self.assertEqual(set(hud.options), {"cycle_source", "cycle_textures", "toggle_auto", "menu_credits"})
                    self.assertFalse(hud.tabs or hud.markers)
                    for action, rect in hud.options.items():
                        self.assertTrue(bounds.contains(rect))
                        self.assertEqual(hud.tab_at(rect.center), action)
                    hud.menu = "credits"
                    hud._paint_flyby(scene)
                    hud.menu = None
        finally:
            pygame.font.quit()


if __name__ == "__main__":
    unittest.main()
