"""Границы CLI, команды автопролёта и доступность его компактного HUD."""

from contextlib import redirect_stderr, redirect_stdout
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
                  ["--mode", "space", "--headless", "--model-source", "zx"],
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

    def test_default_and_explicit_zx_are_flyby_only_and_at_keeps_the_initial_source(self):
        module = ModuleType("elite_lab.space_viewer")
        module.run_space = Mock()
        with patch.dict(sys.modules, {module.__name__: module}):
            main(["--mode", "flyby", "--headless", "--at", "16"])
            module.run_space.assert_called_once_with(
                None, None, True, True, "zx", "author", mode="flyby", source_interval=8.0, at=16.0)
            module.run_space.reset_mock()
            main(["--mode", "flyby", "--headless", "--model-source", "zx", "--model", "20",
                  "--source-interval", "0", "--at", "72"])
            module.run_space.assert_called_once_with(
                20, None, True, True, "zx", "author", mode="flyby", source_interval=0.0, at=72.0)

    def test_source_listing_explains_the_six_step_order_and_only_one_griff(self):
        out = StringIO()
        with redirect_stdout(out):
            main(["--list-sources"])
        result = out.getvalue()
        self.assertIn("zx — только flyby", result)
        self.assertIn("Elite · наша обшивка", result)
        self.assertIn("zx → elite → oolite → griff → pioneer → mixed → zx", result)
        self.assertEqual(result.count("griff — Griff SpecGloss"), 1)
        self.assertIn("место 20 остаётся пустым", result)

    @unittest.skipIf(FlybyControls is None, "Нужны зависимости requirements-space.txt")
    def test_empty_guest_id_is_valid_only_for_flyby_before_context_creation(self):
        from elite_lab.space_assets import AssetError
        from elite_lab.space_viewer import run_space
        with patch("elite_lab.space_viewer._window_context", side_effect=RuntimeError("контекст-проверка")) as context:
            with self.assertRaisesRegex(AssetError, "0–19"):
                run_space(20, None, False, True, "elite", mode="space")
            context.assert_not_called()
            with self.assertRaisesRegex(AssetError, "0–20"):
                run_space(21, None, False, True, "zx", mode="flyby")
            context.assert_not_called()
            with self.assertRaisesRegex(SystemExit, "контекст-проверка"):
                run_space(20, None, False, True, "zx", mode="flyby")
            context.assert_called_once_with()


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

    @patch("pygame.mouse.get_rel")
    @patch("pygame.mouse.set_visible")
    @patch("pygame.mouse.get_visible", return_value=True)
    @patch("pygame.event.set_grab")
    @patch("pygame.event.get_grab", return_value=False)
    def test_right_drag_uses_logical_deltas_and_releases_without_resetting_look(
            self, get_grab, set_grab, get_visible, set_visible, get_rel):
        self.event(pygame.MOUSEBUTTONDOWN, button=3, pos=(498, 180))
        set_grab.assert_called_once_with(True)
        set_visible.assert_called_once_with(False)
        self.event(pygame.MOUSEMOTION, pos=(499, 180), rel=(40, -15))
        self.scene.look_drag.assert_called_once_with(40, -15)
        self.event(pygame.MOUSEBUTTONUP, button=3, pos=(499, 180))
        self.assertFalse(self.controls.looking)
        self.assertEqual(set_grab.call_args_list, [unittest.mock.call(True), unittest.mock.call(False)])
        self.assertEqual(set_visible.call_args_list, [unittest.mock.call(False), unittest.mock.call(True)])
        self.scene.reset_look.assert_not_called()
        self.event(pygame.MOUSEMOTION, pos=(100, 100), rel=(40, 20))
        self.scene.look_drag.assert_called_once()
        self.event(pygame.KEYDOWN, key=pygame.K_r)
        self.scene.reset_look.assert_called_once_with()

    @patch("pygame.mouse.get_rel")
    @patch("pygame.mouse.set_visible")
    @patch("pygame.mouse.get_visible", return_value=True)
    @patch("pygame.event.set_grab")
    @patch("pygame.event.get_grab", return_value=False)
    def test_focus_loss_and_exit_release_relative_mouse_capture(self, *_):
        self.event(pygame.MOUSEBUTTONDOWN, button=3, pos=(200, 180))
        self.event(pygame.WINDOWFOCUSLOST)
        self.assertFalse(self.controls.looking)
        self.assertIsNone(self.controls._mouse_state)
        self.event(pygame.MOUSEMOTION, pos=(100, 100), rel=(40, 20))
        self.scene.look_drag.assert_not_called()
        self.event(pygame.MOUSEBUTTONDOWN, button=3, pos=(200, 180))
        self.assertFalse(self.event(pygame.QUIT))
        self.assertFalse(self.controls.looking)


@unittest.skipIf(SpaceHUD is None, "Нужны зависимости requirements-space.txt")
class FlybyHUDTest(unittest.TestCase):
    def test_all_seven_actions_remain_inside_narrow_and_normal_frames(self):
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
                                    auto_sources=True, empty_slot=False, paused=False, elapsed=0, duration=126)
            for size in ((320, 240), (320, 600), (640, 360), (600, 900), (1360, 820)):
                for source in ("zx", "griff"):
                    for paused in (False, True):
                        with self.subTest(size=size, source=source, paused=paused):
                            hud._resize(size)
                            scene.model_source, scene.paused = source, paused
                            surface = hud._paint_flyby(scene)
                            bounds = surface.get_rect()
                            self.assertEqual(set(hud.options), {"toggle_pause", "scrub_back", "scrub_forward",
                                                               "cycle_source", "cycle_textures", "toggle_auto", "menu_credits"})
                            self.assertFalse(hud.tabs or hud.markers)
                            for action, rect in hud.options.items():
                                self.assertTrue(bounds.contains(rect))
                                self.assertEqual(hud.tab_at(rect.center), action)
                                self.assertFalse(any(rect.colliderect(other) for key, other in hud.options.items()
                                                     if key != action))
                            if size == (1360, 820):
                                self.assertEqual(len({rect.top for rect in hud.options.values()}), 1)
                            elif size[0] == 320:
                                self.assertLess(hud.options["toggle_pause"].top, hud.options["cycle_source"].top)
                            hud.menu = "credits"
                            hud._paint_flyby(scene)
                            hud.menu = None
        finally:
            pygame.font.quit()

    def test_absolute_time_and_pause_labels_stay_explicit_after_a_complete_loop(self):
        pygame.font.init()
        try:
            hud = SpaceHUD.__new__(SpaceHUD)
            hud.texture = None
            hud.ctx = SimpleNamespace(texture=lambda *_: SimpleNamespace(filter=None, release=lambda: None))
            hud.tabs, hud.markers, hud.options, hud.menu = {}, {}, {}, None
            hud._resize((320, 600))
            hud.mono = Mock(wraps=hud.mono)
            hud.text = Mock(wraps=hud.text)
            scene = SimpleNamespace(ships=(), selected=0, model_source="zx", texture_source="author",
                                    auto_sources=False, empty_slot=True, paused=False, elapsed=125.9, duration=126)
            for elapsed, expected in ((125.9, "02:05"), (126, "02:06"), (127, "02:07"), (3601, "1:00:01")):
                with self.subTest(elapsed=elapsed):
                    hud.mono.render.reset_mock()
                    scene.elapsed = elapsed
                    hud._paint_flyby(scene)
                    labels = [call.args[0] for call in hud.mono.render.call_args_list]
                    self.assertIn(f"{expected} · ПРОЛЁТ · держать ← / →  4×", labels)
                    self.assertIn("H интерфейс / Esc выход", labels)
            hud.text.render.reset_mock()
            scene.paused = True
            hud._paint_flyby(scene)
            self.assertIn("1:00:01 · ПАУЗА · держать ← / →  4×",
                          [call.args[0] for call in hud.mono.render.call_args_list])
            self.assertIn("Пробел  Продолжить", [call.args[0] for call in hud.text.render.call_args_list])
        finally:
            pygame.font.quit()

    def test_hud_refreshes_on_time_and_pause_changes_without_redrawing_each_frame(self):
        pygame.font.init()
        try:
            hud = SpaceHUD.__new__(SpaceHUD)
            hud.texture = Mock()
            hud.size = (640, 360)
            hud.key, hud.menu = None, None
            hud.ctx = SimpleNamespace(enable_only=Mock())
            hud.quad = Mock()
            hud._paint = Mock(return_value=pygame.Surface(hud.size, pygame.SRCALPHA))
            scene = SimpleNamespace(camera_mode="flyby", selected=0, model_source="zx", texture_source="author",
                                    auto_sources=False, empty_slot=True, paused=False, elapsed=125.1)
            for elapsed, paused, paints in ((125.1, False, 1), (125.9, False, 1), (126, False, 2),
                                             (126, True, 3), (126.9, True, 3), (127, True, 4)):
                with self.subTest(elapsed=elapsed, paused=paused):
                    scene.elapsed, scene.paused = elapsed, paused
                    hud.draw(scene, hud.size, None, None)
                    self.assertEqual(hud._paint.call_count, paints)
            # У обычного атласа нет часов автопролёта и они ему не требуются.
            scene.camera_mode = "inspect"
            del scene.elapsed, scene.paused
            hud.draw(scene, hud.size, None, None)
        finally:
            pygame.font.quit()

    def test_zx_and_empty_slot_labels_do_not_promise_textures_or_other_authors(self):
        pygame.font.init()
        try:
            hud = SpaceHUD.__new__(SpaceHUD)
            hud.texture = None
            hud.ctx = SimpleNamespace(texture=lambda *_: SimpleNamespace(filter=None, release=lambda: None))
            hud.tabs, hud.markers, hud.options, hud.menu = {}, {}, {}, "credits"
            hud._resize((600, 900))
            scene = SimpleNamespace(ships=(), selected=0, model_source="zx", texture_source="author",
                                    auto_sources=False, empty_slot=True, paused=False, elapsed=0, duration=126)
            # Пустое место не обращается к ships[selected] и не выдаёт его авторство.
            with patch.object(hud, "_flyby_credits", wraps=hud._flyby_credits) as credits:
                hud._paint_flyby(scene)
                self.assertIsNone(credits.call_args.args[2])
            labels = []
            hud._flyby_credits(pygame.Surface(hud.size), scene, None,
                              lambda text, *_args, **_kwargs: labels.append(text), 20, 600)
            self.assertIn("Пустое место гостя", labels)
            self.assertFalse(any("Автор —" in text for text in labels))
        finally:
            pygame.font.quit()


if __name__ == "__main__":
    unittest.main()
