"""Единая клавиатура полёта и безопасный захват мыши в space и flyby."""

from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import pygame

try:
    from elite_lab.space_viewer import FlybyControls, SpaceControls
except ModuleNotFoundError:
    SpaceControls = FlybyControls = None


@unittest.skipIf(SpaceControls is None, "Нужны зависимости requirements-space.txt")
class FlightProfileControlsTest(unittest.TestCase):
    def setUp(self):
        self.mouse = {}
        for target, value in (("pygame.display.get_init", True),
                              ("pygame.event.get_grab", False),
                              ("pygame.mouse.get_visible", True),
                              ("pygame.event.set_grab", None),
                              ("pygame.mouse.set_visible", None),
                              ("pygame.mouse.get_rel", (0, 0))):
            self.mouse[target] = self.enterContext(patch(target, return_value=value))

    def controls(self, kind=SpaceControls, *, profile="modern", flight=True):
        outer = "flyby" if kind is FlybyControls else "overview"
        scene = Mock(camera_mode=outer, flight_profile=profile, selected=0, flight_speed=3.0)
        scene.fly.side_effect = lambda: setattr(scene, "camera_mode", "flight")
        scene.overview.side_effect = lambda: setattr(scene, "camera_mode", "overview")
        scene.toggle_flight.side_effect = lambda: setattr(
            scene, "camera_mode", outer if scene.camera_mode == "flight" else "flight")
        scene.cycle_flight_profile.side_effect = lambda: setattr(
            scene, "flight_profile", "elite" if scene.flight_profile == "modern" else "modern")
        hud = SimpleNamespace(menu=None, tab_at=Mock(return_value=None))
        controls = kind(scene, hud, True)
        if flight:
            self.tap(controls, pygame.K_f)
            scene.reset_mock()
        return controls

    @staticmethod
    def event(controls, kind, **attrs):
        return controls.event(pygame.event.Event(kind, **attrs), (500, 300), (1000, 600))

    def down(self, controls, *keys):
        for key in keys:
            self.event(controls, pygame.KEYDOWN, key=key)

    def up(self, controls, *keys):
        for key in keys:
            self.event(controls, pygame.KEYUP, key=key)

    def tap(self, controls, key):
        result = self.event(controls, pygame.KEYDOWN, key=key)
        self.up(controls, key)
        return result

    def click(self, controls, action=None):
        controls.hud.tab_at.return_value = action
        for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            self.event(controls, kind, button=1, pos=(20, 15))

    def test_modern_translation_roll_and_look_have_identical_axes_in_both_entries(self):
        cases = (
            ((pygame.K_w,), (0, 0, 1, 1), None),
            ((pygame.K_s,), (0, 0, -1, 1), None),
            ((pygame.K_a, pygame.K_SPACE), (-1, 1, 0, 1), None),
            ((pygame.K_d, pygame.K_LSHIFT), (1, -1, 0, 1), None),
            ((pygame.K_RSHIFT,), (0, -1, 0, 1), None),
            ((pygame.K_q,), (0, 0, 0, 1), (0, 0, -.16)),
            ((pygame.K_e,), (0, 0, 0, 1), (0, 0, .16)),
            ((pygame.K_LEFT, pygame.K_UP), (0, 0, 0, 1), (-.16, .16, 0)),
            ((pygame.K_RIGHT, pygame.K_DOWN), (0, 0, 0, 1), (.16, -.16, 0)),
        )
        for kind, (keys, movement, turn) in product((SpaceControls, FlybyControls), cases):
            with self.subTest(kind=kind.__name__, keys=keys):
                controls = self.controls(kind)
                self.down(controls, *keys)
                controls.update(.25)
                controls.scene.update.assert_called_once_with(.25, movement)
                if turn is None:
                    controls.scene.turn_flight.assert_not_called()
                else:
                    self.assertEqual(controls.scene.turn_flight.call_count, 1)
                    for actual, expected in zip(controls.scene.turn_flight.call_args.args, turn):
                        self.assertAlmostEqual(actual, expected)
                controls.scene.select.assert_not_called()
                controls.scene.toggle_pause.assert_not_called()
                controls.scene.scrub.assert_not_called()
                controls.scene.toggle_auto_sources.assert_not_called()

    def test_elite_only_adjusts_retained_speed_pitch_and_roll(self):
        for kind in (SpaceControls, FlybyControls):
            with self.subTest(kind=kind.__name__):
                controls = self.controls(kind, profile="elite")
                ignored = (pygame.K_a, pygame.K_d, pygame.K_q, pygame.K_e,
                           pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT)
                self.down(controls, *ignored, pygame.K_w, pygame.K_RIGHT, pygame.K_UP)
                controls.update(.5)
                controls.scene.update.assert_called_with(.5, (0, 0, 1, 1))
                self.assertEqual(controls.scene.turn_flight.call_count, 1)
                yaw, pitch, roll = controls.scene.turn_flight.call_args.args
                self.assertEqual(yaw, 0)
                self.assertAlmostEqual(pitch, .16)
                self.assertAlmostEqual(roll, .16)
                self.up(controls, pygame.K_w, pygame.K_RIGHT, pygame.K_UP)
                self.down(controls, pygame.K_s, pygame.K_LEFT, pygame.K_DOWN)
                controls.update(.5)
                controls.scene.update.assert_called_with(.5, (0, 0, -1, 1))
                yaw, pitch, roll = controls.scene.turn_flight.call_args.args
                self.assertEqual(yaw, 0)
                self.assertAlmostEqual(pitch, -.16)
                self.assertAlmostEqual(roll, -.16)
                self.up(controls, pygame.K_s, pygame.K_LEFT, pygame.K_DOWN)
                controls.update(.5)
                controls.scene.update.assert_called_with(.5, (0, 0, 0, 1))
                self.assertEqual(controls.scene.flight_speed, 3.0)

    def test_opposite_keys_cancel_and_angular_motion_is_scaled_by_frame_time(self):
        for kind, profile in product((SpaceControls, FlybyControls), ("modern", "elite")):
            with self.subTest(kind=kind.__name__, profile=profile):
                controls = self.controls(kind, profile=profile)
                self.down(controls, *controls._movement_keys, pygame.K_LSHIFT)
                controls.update(.1)
                controls.scene.update.assert_called_once_with(.1, (0, 0, 0, 1))
                controls.scene.turn_flight.assert_not_called()
                self.up(controls, *controls._movement_keys, pygame.K_LSHIFT)
                self.down(controls, pygame.K_UP)
                for dt in (.05, .2):
                    controls.update(dt)
                self.assertEqual(controls.scene.turn_flight.call_count, 2)
                for actual, expected in zip(controls.scene.turn_flight.call_args_list, (.08, .16)):
                    yaw, pitch, roll = actual.args
                    self.assertEqual((yaw, roll), (0, 0))
                    self.assertAlmostEqual(pitch, expected)

    def test_profile_key_and_hud_cycle_once_and_held_keys_need_a_new_press(self):
        for kind in (SpaceControls, FlybyControls):
            with self.subTest(kind=kind.__name__):
                controls = self.controls(kind)
                captures = self.mouse["pygame.event.set_grab"].call_count
                self.down(controls, pygame.K_w, pygame.K_LEFT, pygame.K_c)
                self.down(controls, pygame.K_c)
                self.event(controls, pygame.KEYDOWN, key=pygame.K_c, repeat=True)
                controls.scene.cycle_flight_profile.assert_called_once_with()
                self.assertEqual(controls.scene.flight_profile, "elite")
                self.assertTrue(controls.looking)
                self.assertEqual(self.mouse["pygame.event.set_grab"].call_count, captures)
                controls.update(.1)
                controls.scene.update.assert_called_with(.1, (0, 0, 0, 1))
                controls.scene.turn_flight.assert_not_called()
                self.down(controls, pygame.K_w)
                controls.update(.1)
                controls.scene.update.assert_called_with(.1, (0, 0, 0, 1))
                self.up(controls, pygame.K_w, pygame.K_c)
                self.down(controls, pygame.K_w)
                controls.update(.1)
                controls.scene.update.assert_called_with(.1, (0, 0, 1, 1))
                self.tap(controls, pygame.K_ESCAPE)
                self.click(controls, "cycle_flight_profile")
                self.assertEqual(controls.scene.cycle_flight_profile.call_count, 2)
                self.assertEqual(controls.scene.flight_profile, "modern")
                self.assertFalse(controls.looking)
                controls.update(.1)
                controls.scene.update.assert_called_with(.1, (0, 0, 0, 1))

    def test_auto_capture_escape_hud_and_blank_scene_click_do_not_conflict(self):
        for kind in (SpaceControls, FlybyControls):
            with self.subTest(kind=kind.__name__):
                controls = self.controls(kind)
                self.assertTrue(controls.looking)
                self.event(controls, pygame.MOUSEMOTION, rel=(12, -7), pos=(250, 150))
                drag = controls.scene.look_drag if kind is FlybyControls else controls.scene.drag
                drag.assert_called_once_with(12, -7)
                self.click(controls, "cycle_flight_profile")
                controls.scene.cycle_flight_profile.assert_not_called()
                self.down(controls, pygame.K_ESCAPE)
                self.assertFalse(controls.looking)
                self.assertTrue(self.event(controls, pygame.KEYDOWN, key=pygame.K_ESCAPE, repeat=True))
                self.assertEqual(controls.scene.camera_mode, "flight")
                self.up(controls, pygame.K_ESCAPE)
                self.click(controls, "cycle_flight_profile")
                controls.scene.cycle_flight_profile.assert_called_once_with()
                self.assertFalse(controls.looking)
                self.click(controls)
                self.assertTrue(controls.looking)
                self.event(controls, pygame.MOUSEBUTTONUP, button=3, pos=(20, 15))
                self.assertTrue(controls.looking)
                controls.scene.pick.assert_not_called()

    def test_capture_uses_logical_mouse_delta_and_restores_prior_sdl_state(self):
        controls = self.controls()
        self.mouse["pygame.event.set_grab"].assert_called_once_with(True)
        self.mouse["pygame.mouse.set_visible"].assert_called_once_with(False)
        self.tap(controls, pygame.K_ESCAPE)
        self.assertEqual(self.mouse["pygame.event.set_grab"].call_args_list, [call(True), call(False)])
        self.assertEqual(self.mouse["pygame.mouse.set_visible"].call_args_list, [call(False), call(True)])

    def test_focus_loss_clears_motion_stops_elite_and_does_not_recapture_on_focus_gain(self):
        for kind in (SpaceControls, FlybyControls):
            with self.subTest(kind=kind.__name__):
                controls = self.controls(kind, profile="elite")
                self.down(controls, pygame.K_w, pygame.K_UP)
                self.event(controls, pygame.WINDOWFOCUSLOST)
                self.assertFalse(controls.looking)
                self.assertEqual(controls.scene.flight_speed, 0.0)
                self.assertFalse(controls._flight_keys or controls._view_keys or controls._held_keys)
                self.event(controls, pygame.WINDOWFOCUSGAINED)
                self.assertFalse(controls.looking)
                for key in (pygame.K_w, pygame.K_UP, pygame.K_f, pygame.K_c):
                    self.event(controls, pygame.KEYDOWN, key=key, repeat=True)
                controls.update(.1)
                controls.scene.update.assert_called_once_with(.1, (0, 0, 0, 1))
                controls.scene.turn_flight.assert_not_called()
                controls.scene.cycle_flight_profile.assert_not_called()
                self.assertEqual(controls.scene.camera_mode, "flight")
                self.assertFalse(self.event(controls, pygame.QUIT))
                self.assertFalse(controls.looking)

    def test_escape_keyup_and_source_changes_do_not_reset_elite_throttle(self):
        for kind in (SpaceControls, FlybyControls):
            with self.subTest(kind=kind.__name__):
                controls = self.controls(kind, profile="elite")
                for key in (pygame.K_w, pygame.K_ESCAPE, pygame.K_m, pygame.K_t):
                    self.tap(controls, key)
                self.assertEqual(controls.scene.flight_speed, 3.0)

    def test_shift_arrows_look_without_scrubbing_or_selecting_in_nonflight_modes(self):
        for kind in (SpaceControls, FlybyControls):
            with self.subTest(kind=kind.__name__):
                controls = self.controls(kind, flight=False)
                self.down(controls, pygame.K_LSHIFT, pygame.K_RIGHT, pygame.K_UP)
                controls.update(.25)
                drag = controls.scene.look_drag if kind is FlybyControls else controls.scene.drag
                self.assertEqual(drag.call_count, 1)
                dx, dy = drag.call_args.args
                self.assertAlmostEqual(dx, 32)
                self.assertAlmostEqual(dy, -32)
                controls.scene.select.assert_not_called()
                controls.scene.scrub.assert_not_called()
                self.up(controls, pygame.K_LSHIFT)
                controls.update(.1)
                drag.assert_called_once()
                controls.scene.scrub.assert_not_called()
                controls.scene.select.assert_not_called()

    def test_leaving_flight_releases_mouse_and_held_flight_keys(self):
        for kind in (SpaceControls, FlybyControls):
            with self.subTest(kind=kind.__name__):
                controls = self.controls(kind)
                self.down(controls, pygame.K_a, pygame.K_SPACE, pygame.K_RIGHT)
                self.tap(controls, pygame.K_f if kind is FlybyControls else pygame.K_o)
                self.assertFalse(controls.looking)
                self.assertFalse(controls._flight_keys)
                for key in (pygame.K_a, pygame.K_SPACE, pygame.K_RIGHT):
                    self.event(controls, pygame.KEYDOWN, key=key, repeat=True)
                controls.scene.toggle_auto_sources.assert_not_called()
                controls.scene.toggle_pause.assert_not_called()
                controls.scene.scrub.assert_not_called()
                self.tap(controls, pygame.K_f)
                controls.update(.1)
                controls.scene.update.assert_called_with(.1, (0, 0, 0, 1))


if __name__ == "__main__":
    unittest.main()
