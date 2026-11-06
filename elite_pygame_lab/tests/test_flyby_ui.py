"""Границы CLI, команды автопролёта и доступность его компактного HUD."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

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
                  "--at", "24.5", "--screenshot", "/tmp/elite-flyby-test.png", "--no-hud", "--no-effects"])
            module.run_space.assert_called_once_with(
                9, Path("/tmp/elite-flyby-test.png"), False, False, "griff", "procedural",
                mode="flyby", source_interval=0.0, at=24.5, effects_enabled=False)
            module.run_space.reset_mock()
            main(["--mode", "space", "--headless"])
            module.run_space.assert_called_once_with(
                None, None, True, True, "elite", "author", mode="space", source_interval=8.0, at=None,
                effects_enabled=True)

    def test_default_and_explicit_zx_are_flyby_only_and_at_keeps_the_initial_source(self):
        module = ModuleType("elite_lab.space_viewer")
        module.run_space = Mock()
        with patch.dict(sys.modules, {module.__name__: module}):
            main(["--mode", "flyby", "--headless", "--at", "16"])
            module.run_space.assert_called_once_with(
                None, None, True, True, "zx", "author", mode="flyby", source_interval=8.0, at=16.0,
                effects_enabled=True)
            module.run_space.reset_mock()
            main(["--mode", "flyby", "--headless", "--model-source", "zx", "--model", "20",
                  "--source-interval", "0", "--at", "72"])
            module.run_space.assert_called_once_with(
                20, None, True, True, "zx", "author", mode="flyby", source_interval=0.0, at=72.0,
                effects_enabled=True)

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
        self.scene = Mock(camera_mode="flyby", flight_profile="modern")
        self.scene.toggle_flight.side_effect = self.toggle_flight
        self.hud = SimpleNamespace(menu=None, tab_at=Mock(return_value=None))
        self.controls = FlybyControls(self.scene, self.hud, True)
        for target, value in (("pygame.display.get_init", True),
                              ("pygame.event.get_grab", False),
                              ("pygame.mouse.get_visible", True),
                              ("pygame.event.set_grab", None),
                              ("pygame.mouse.set_visible", None),
                              ("pygame.mouse.get_rel", (0, 0))):
            self.enterContext(patch(target, return_value=value))

    def toggle_flight(self):
        self.scene.camera_mode = "flight" if self.scene.camera_mode == "flyby" else "flyby"

    def event(self, kind, **attrs):
        return self.controls.event(pygame.event.Event(kind, **attrs), (500, 300), (1000, 600))

    def key(self, key):
        result = self.event(pygame.KEYDOWN, key=key)
        self.event(pygame.KEYUP, key=key)
        return result

    def click(self, action):
        if self.scene.camera_mode == "flight" and self.controls.looking:
            self.key(pygame.K_ESCAPE)
        self.hud.tab_at.return_value = action
        for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            self.event(kind, button=1, pos=(20, 15))

    def test_shortcuts_and_escape_affect_only_the_presentation(self):
        for key, method in ((pygame.K_m, "cycle_source"), (pygame.K_t, "cycle_textures"),
                            (pygame.K_a, "toggle_auto_sources")):
            self.assertTrue(self.key(key))
            getattr(self.scene, method).assert_called_once_with()
        self.key(pygame.K_h)
        self.assertFalse(self.controls.show_hud)
        self.key(pygame.K_i)
        self.assertTrue(self.controls.show_hud)
        self.assertEqual(self.hud.menu, "credits")
        self.assertTrue(self.key(pygame.K_ESCAPE))
        self.assertIsNone(self.hud.menu)
        self.assertFalse(self.key(pygame.K_ESCAPE))

    def test_auto_wheel_and_left_drag_are_ignored_and_hidpi_clicks_work(self):
        self.event(pygame.MOUSEWHEEL, y=2)
        self.event(pygame.MOUSEBUTTONDOWN, button=1, pos=(10, 10))
        self.event(pygame.MOUSEMOTION, pos=(50, 60), rel=(40, 50))
        self.event(pygame.MOUSEBUTTONUP, button=1, pos=(50, 60))
        self.controls.update(.25)
        self.assertEqual(self.scene.mock_calls, [unittest.mock.call.update(.25)])
        self.click("cycle_source")
        self.hud.tab_at.assert_called_with((40, 30))
        self.scene.cycle_source.assert_called_once_with()

    def test_f_and_hud_toggle_the_same_mode_once(self):
        self.event(pygame.KEYDOWN, key=pygame.K_f)
        self.event(pygame.KEYDOWN, key=pygame.K_f, repeat=True)
        self.event(pygame.KEYDOWN, key=pygame.K_f)
        self.scene.toggle_flight.assert_called_once_with()
        self.assertEqual(self.scene.camera_mode, "flight")
        self.event(pygame.KEYUP, key=pygame.K_f)
        self.click("toggle_flight")
        self.assertEqual(self.scene.toggle_flight.call_count, 2)
        self.assertEqual(self.scene.camera_mode, "flyby")

    def test_free_flight_uses_shared_translation_axes_without_shift_boost(self):
        self.key(pygame.K_f)
        for keys, movement in (((pygame.K_d, pygame.K_SPACE, pygame.K_w), (1, 1, 1, 1)),
                               ((pygame.K_a, pygame.K_s, pygame.K_LSHIFT), (-1, -1, -1, 1))):
            with self.subTest(keys=keys):
                for key in keys:
                    self.event(pygame.KEYDOWN, key=key)
                self.controls.update(.25)
                self.scene.update.assert_called_with(.25, movement)
                for key in keys:
                    self.event(pygame.KEYUP, key=key)
        self.controls.update(.1)
        self.scene.update.assert_called_with(.1, (0, 0, 0, 1))

    def test_free_flight_accepts_precise_wheel_but_ignores_stale_transport_clicks(self):
        self.key(pygame.K_f)
        for action in ("toggle_pause", "scrub_back", "scrub_forward", "toggle_auto"):
            self.click(action)
        self.scene.toggle_pause.assert_not_called()
        self.scene.scrub.assert_not_called()
        self.scene.toggle_auto_sources.assert_not_called()
        self.event(pygame.MOUSEWHEEL, y=1, precise_y=.125)
        self.scene.zoom.assert_called_once_with(.125)
        for key in (pygame.K_EQUALS, pygame.K_MINUS):
            self.key(key)
        self.assertEqual(self.scene.zoom.call_args_list,
                         [unittest.mock.call(.125), unittest.mock.call(1), unittest.mock.call(-1)])

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
        self.key(pygame.K_r)
        self.scene.reset_look.assert_called_once_with()

    @patch("pygame.mouse.get_rel")
    @patch("pygame.mouse.set_visible")
    @patch("pygame.mouse.get_visible", return_value=True)
    @patch("pygame.event.set_grab")
    @patch("pygame.event.get_grab", return_value=False)
    def test_mode_transitions_release_look_transport_and_movement(self, *_):
        self.event(pygame.KEYDOWN, key=pygame.K_RIGHT)
        self.event(pygame.MOUSEBUTTONDOWN, button=3, pos=(498, 180))
        self.key(pygame.K_f)
        self.assertTrue(self.controls.looking)
        self.assertIsNotNone(self.controls._mouse_state)
        self.assertFalse(self.controls._held_keys)
        self.event(pygame.MOUSEMOTION, pos=(499, 180), rel=(40, -15))
        self.scene.look_drag.assert_called_once_with(40, -15)
        self.scene.look_drag.reset_mock()
        self.event(pygame.KEYDOWN, key=pygame.K_w)
        self.event(pygame.MOUSEBUTTONDOWN, button=3, pos=(498, 180))
        self.event(pygame.MOUSEMOTION, pos=(499, 180), rel=(40, -15))
        self.scene.look_drag.assert_called_once_with(40, -15)
        self.click("toggle_flight")
        self.assertFalse(self.controls.looking)
        self.assertFalse(self.controls._flight_keys)
        self.event(pygame.KEYDOWN, key=pygame.K_RIGHT, repeat=True)
        self.assertFalse(self.controls._held_keys)
        self.key(pygame.K_f)
        self.controls.update(.1)
        self.scene.update.assert_called_with(.1, (0, 0, 0, 1))

    @patch("pygame.mouse.get_rel")
    @patch("pygame.mouse.set_visible")
    @patch("pygame.mouse.get_visible", return_value=True)
    @patch("pygame.event.set_grab")
    @patch("pygame.event.get_grab", return_value=False)
    def test_focus_loss_and_exit_release_relative_mouse_capture(self, *_):
        self.key(pygame.K_f)
        self.event(pygame.KEYDOWN, key=pygame.K_w)
        self.event(pygame.MOUSEBUTTONDOWN, button=3, pos=(200, 180))
        self.event(pygame.WINDOWFOCUSLOST)
        self.assertFalse(self.controls.looking)
        self.assertIsNone(self.controls._mouse_state)
        self.event(pygame.MOUSEMOTION, pos=(100, 100), rel=(40, 20))
        self.scene.look_drag.assert_not_called()
        self.event(pygame.KEYDOWN, key=pygame.K_w, repeat=True)
        self.event(pygame.KEYDOWN, key=pygame.K_f, repeat=True)
        self.controls.update(.1)
        self.scene.update.assert_called_with(.1, (0, 0, 0, 1))
        self.assertEqual(self.scene.camera_mode, "flight")
        self.event(pygame.MOUSEBUTTONDOWN, button=3, pos=(200, 180))
        self.assertFalse(self.event(pygame.QUIT))
        self.assertFalse(self.controls.looking)


@unittest.skipIf(SpaceHUD is None, "Нужны зависимости requirements-space.txt")
class FlybyHUDTest(unittest.TestCase):
    def setUp(self):
        pygame.font.init()
        self.addCleanup(pygame.font.quit)

    def hud(self, size=(1360, 820), *, watch_text=False):
        hud = SpaceHUD(MagicMock())
        hud._resize(size)
        if watch_text:
            hud.mono, hud.text = Mock(wraps=hud.mono), Mock(wraps=hud.text)
        return hud

    @staticmethod
    def labels(hud):
        return [call.args[0] for font in (hud.mono, hud.text) for call in font.render.call_args_list]

    def test_all_actions_remain_inside_narrow_and_normal_frames_in_both_modes(self):
        hud = self.hud()
        ship = SimpleNamespace(name="Cobra Mk III (пиратская)", materials=(), source_id="elite",
                               source_title="ZX Spectrum Elite", author="Ian Bell, David Braben",
                               license="Отдельная лицензия на каркас не указана")
        scene = SimpleNamespace(is_flyby=True, camera_mode="flyby", ships=(ship,), selected=0,
                                model_source="griff", texture_source="author", auto_sources=True,
                                empty_slot=False, paused=False, elapsed=0, duration=126)
        for size in ((320, 240), (320, 600), (640, 360), (600, 900), (1360, 820)):
            for source in ("zx", "griff"):
                for mode in ("flyby", "flight"):
                    for paused in (False, True):
                        with self.subTest(size=size, source=source, mode=mode, paused=paused):
                            hud._resize(size)
                            scene.model_source, scene.camera_mode, scene.paused = source, mode, paused
                            bounds = hud._paint(scene, None, None).get_rect()
                            actions = {"toggle_flight", "cycle_source", "cycle_textures",
                                       "menu_credits", "menu_resolution"}
                            if mode == "flyby":
                                actions |= {"toggle_pause", "scrub_back", "scrub_forward", "toggle_auto"}
                            else:
                                actions.add("cycle_flight_profile")
                            self.assertEqual(set(hud.options), actions)
                            self.assertFalse(hud.tabs or hud.markers)
                            for action, rect in hud.options.items():
                                self.assertTrue(bounds.contains(rect))
                                self.assertEqual(hud.tab_at(rect.center), action)
                                self.assertFalse(any(rect.colliderect(other) for key, other in hud.options.items()
                                                     if key != action))
                            if size == (1360, 820):
                                self.assertEqual(len({rect.top for action, rect in hud.options.items()
                                                      if action != "menu_resolution"}), 1)
                            elif size[0] == 320:
                                self.assertLess(hud.options["toggle_flight"].top, hud.options["cycle_source"].top)
                            hud.menu = "credits"
                            hud._paint(scene, None, None)
                            hud.menu = None

    def test_absolute_time_and_pause_labels_stay_explicit_after_a_complete_loop(self):
        hud = self.hud((320, 600), watch_text=True)
        scene = SimpleNamespace(is_flyby=True, camera_mode="flyby", ships=(), selected=0,
                                model_source="zx", texture_source="author", auto_sources=False,
                                empty_slot=True, paused=False, elapsed=125.9, duration=126)
        for elapsed, expected in ((125.9, "02:05"), (126, "02:06"), (127, "02:07"), (3601, "1:00:01")):
            with self.subTest(elapsed=elapsed):
                hud.mono.render.reset_mock()
                scene.elapsed = elapsed
                hud._paint_flyby(scene)
                self.assertIn(f"{expected} · ПРОЛЁТ · держать ← / →  4×", self.labels(hud))
                self.assertIn("H интерфейс / Esc выход", self.labels(hud))
        scene.paused = True
        hud._paint_flyby(scene)
        self.assertIn("1:00:01 · ПАУЗА · держать ← / →  4×", self.labels(hud))
        self.assertIn("Пробел  Продолжить", self.labels(hud))

    def test_zx_and_empty_slot_labels_do_not_promise_textures_or_other_authors(self):
        hud = self.hud((600, 900))
        hud.menu = "credits"
        scene = SimpleNamespace(is_flyby=True, camera_mode="flyby", ships=(), selected=0,
                                model_source="zx", texture_source="author", auto_sources=False,
                                empty_slot=True, paused=False, elapsed=0, duration=126)
        # Пустое место не обращается к ships[selected] и не выдаёт его авторство.
        with patch.object(hud, "_flyby_credits", wraps=hud._flyby_credits) as credits:
            hud._paint_flyby(scene)
            self.assertIsNone(credits.call_args.args[2])
        labels = []
        hud._flyby_credits(pygame.Surface(hud.size), scene, None,
                          lambda text, *_args, **_kwargs: labels.append(text), 20, 600)
        self.assertIn("Пустое место гостя", labels)
        self.assertFalse(any("Автор —" in text for text in labels))

    def test_presentation_hud_cache_tracks_route_mode_object_and_source(self):
        from elite_lab.space_flyby import FlybyScene

        hud = self.hud(watch_text=True)
        scene = FlybyScene(source_interval=0)
        with patch.object(hud, "_paint", wraps=hud._paint) as paint:
            for elapsed, paused, paints in ((125.1, False, 1), (125.9, False, 1), (126, False, 2),
                                           (126, True, 3), (126.9, True, 3), (127, True, 4)):
                with self.subTest(elapsed=elapsed, paused=paused):
                    scene.seek(elapsed)
                    scene.paused = paused
                    hud.draw(scene, hud.size, None, None)
                    self.assertEqual(paint.call_count, paints)
            scene.seek(72)  # У пустого гостевого места ручной режим выбирает реальный объект.
            scene.toggle_flight()
            self.assertTrue(scene.is_flyby)
            self.assertEqual(scene.camera_mode, "flight")
            self.assertFalse(scene.empty_slot)
            for source in ("zx", "griff"):
                scene.set_source(source)
                hud.mono.render.reset_mock()
                hud.text.render.reset_mock()
                before = paint.call_count
                hud.draw(scene, hud.size, None, None)
                scene.update(.1)
                hud.draw(scene, hud.size, None, None)
                self.assertEqual(paint.call_count, before + 1)
                labels = self.labels(hud)
                for expected in ("РУЧНОЙ ПОЛЁТ · маршрут на паузе · 01:12", "F  К автопролёту",
                                 "C  Схема · 3D", "W / S  вперёд / назад", "A / D  стрейф",
                                 "Пробел / Shift  вверх / вниз", "Q / E  крен",
                                 "Стрелки / мышь  взгляд", "Esc  курсор · щелчок  захват"):
                    self.assertIn(expected, labels)
                self.assertTrue(any(label.startswith("Ближайший объект · ") for label in labels))
                self.assertFalse(any(label.startswith(("A  ", "Пробел  ")) or "держать ← / →" in label
                                     for label in labels))
                scene.selected = (scene.selected + 1) % len(scene.ships)
                hud.draw(scene, hud.size, None, None)
                self.assertEqual(paint.call_count, before + 2)
            scene.toggle_flight()
            hud.draw(scene, hud.size, None, None)
            self.assertTrue({"toggle_pause", "toggle_auto", "toggle_flight"} <= hud.options.keys())

    def test_regular_atlas_keeps_its_flight_tabs_and_source_menu(self):
        from elite_lab.space_scene import SpaceScene

        hud = self.hud()
        scene = SpaceScene()
        scene.fly()
        self.assertFalse(scene.is_flyby)
        # Атлас рисуется без часов/паузы показа и сохраняет собственные вкладки.
        with patch.object(hud, "_paint_flyby", side_effect=AssertionError("HUD показа в атласе")):
            hud.draw(scene, hud.size, None, None)
        self.assertEqual(set(hud.tabs), {"overview", "inspect", "flight"})
        self.assertEqual(set(hud.options), {"menu_models", "menu_textures", "menu_credits",
                                            "menu_resolution", "cycle_flight_profile"})


if __name__ == "__main__":
    unittest.main()
