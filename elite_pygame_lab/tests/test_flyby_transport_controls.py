"""Перемотка и переходы в свободный полёт на настоящей динамической сцене."""

from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pygame

try:
    import numpy as np
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_flyby import FlybyScene
    from elite_lab.space_viewer import FlybyControls, SpaceRenderer
except ModuleNotFoundError:
    FlybyControls = None


@unittest.skipIf(FlybyControls is None, "Нужны зависимости requirements-space.txt")
class FlybyTransportControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = AssetCatalog()

    def setUp(self):
        self.restart()

    def restart(self, at=20, paused=False):
        self.scene = FlybyScene(catalog=self.catalog)
        self.scene.seek(at)
        self.scene.paused = paused
        self.hud = SimpleNamespace(menu=None, tab_at=Mock(return_value=None))
        self.controls = FlybyControls(self.scene, self.hud, True)

    def event(self, kind, **attrs):
        return self.controls.event(pygame.event.Event(kind, **attrs), (500, 300), (1000, 600))

    def key(self, key, down=True, **attrs):
        return self.event(pygame.KEYDOWN if down else pygame.KEYUP, key=key, **attrs)

    def test_taps_pause_and_repeat_keep_camera_and_sources_stopped_but_rotate_ships(self):
        self.key(pygame.K_SPACE)
        self.key(pygame.K_SPACE, repeat=True)
        self.key(pygame.K_SPACE)
        self.assertTrue(self.scene.paused)
        for key, expected in ((pygame.K_LEFT, 19), (pygame.K_RIGHT, 20)):
            self.key(key)
            self.key(key, repeat=True)
            self.key(key)
            self.assertEqual(self.scene.elapsed, expected)
            self.key(key, down=False)
        eye, source = self.scene.eye.copy(), self.scene.model_source
        matrix = self.scene.objects[0].matrix.copy()
        self.controls.update(9)
        self.assertEqual((self.scene.elapsed, self.scene.model_source), (20, source))
        np.testing.assert_array_equal(self.scene.eye, eye)
        self.assertFalse(np.allclose(self.scene.objects[0].matrix, matrix))
        self.key(pygame.K_SPACE, down=False)
        self.key(pygame.K_SPACE)
        self.controls.update(.5)
        self.assertFalse(self.scene.paused)
        self.assertEqual(self.scene.elapsed, 20.5)

    def test_hold_speed_and_restored_mode_are_independent_of_frame_rate(self):
        for paused, key, chunks in product((False, True), (pygame.K_LEFT, pygame.K_RIGHT),
                                           ((.75,), (.05, .15, .05, .5))):
            with self.subTest(paused=paused, key=key, chunks=chunks):
                self.restart(paused=paused)
                sign = 1 if key == pygame.K_RIGHT else -1
                self.key(key)
                for seconds in chunks:
                    self.controls.update(seconds)
                self.assertAlmostEqual(self.scene.elapsed, 20 + sign * 3)
                self.assertAlmostEqual(self.scene.animation_elapsed, 20 + sign * 3 + .25)
                self.key(key, down=False)
                self.controls.update(.5)
                self.assertEqual(self.scene.paused, paused)
                self.assertAlmostEqual(self.scene.elapsed, 20 + sign * 3 + (0 if paused else .5))

    def test_round_trip_and_lower_bound_do_not_add_extra_rotation(self):
        for key in (pygame.K_RIGHT, pygame.K_LEFT):
            self.key(key)
            self.controls.update(.75)
            self.key(key, down=False)
        self.assertEqual((self.scene.elapsed, self.scene.animation_elapsed), (20, 20.5))
        for start, rotation in ((.5, .75), (2, .5)):
            with self.subTest(start=start):
                self.restart(at=start)
                self.key(pygame.K_LEFT)
                self.controls.update(.75)
                self.assertEqual(self.scene.elapsed, 0)
                self.assertAlmostEqual(self.scene.animation_elapsed, rotation)

    def test_two_arrows_cancel_and_focus_loss_releases_hold_without_changing_pause(self):
        self.key(pygame.K_LEFT)
        self.controls.update(.1)
        self.key(pygame.K_RIGHT)
        self.controls.update(.5)
        self.assertEqual(self.scene.elapsed, 20)
        self.key(pygame.K_LEFT, down=False)
        self.controls.update(.1)
        self.assertAlmostEqual(self.scene.elapsed, 20.4)
        self.key(pygame.K_SPACE)
        self.event(pygame.WINDOWFOCUSLOST)
        self.key(pygame.K_RIGHT, repeat=True)
        self.key(pygame.K_SPACE, repeat=True)
        self.controls.update(1)
        self.assertTrue(self.scene.paused)
        self.assertAlmostEqual(self.scene.elapsed, 20.4)
        self.key(pygame.K_SPACE)
        self.controls.update(.5)
        self.assertFalse(self.scene.paused)
        self.assertAlmostEqual(self.scene.elapsed, 20.9)

    def test_hidpi_clicks_match_keyboard_actions_without_starting_a_hold(self):
        for key, action in ((pygame.K_SPACE, "toggle_pause"), (pygame.K_LEFT, "scrub_back"),
                            (pygame.K_RIGHT, "scrub_forward")):
            with self.subTest(action=action):
                self.restart()
                self.key(key)
                expected = (self.scene.elapsed, self.scene.paused)
                self.restart()
                self.hud.tab_at.return_value = action
                for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                    self.event(kind, button=1, pos=(20, 15))
                self.assertEqual((self.scene.elapsed, self.scene.paused), expected)
                self.hud.tab_at.assert_called_with((40, 30))
                self.controls.update(.5)
                self.assertEqual(self.scene.elapsed, expected[0] + (0 if expected[1] else .5))

    def test_right_mouse_release_keeps_scrub_and_space_during_hold_sets_resumed_mode(self):
        self.key(pygame.K_RIGHT)
        self.controls.update(.5)
        self.key(pygame.K_SPACE)
        self.event(pygame.MOUSEBUTTONUP, button=3, pos=(200, 100))
        self.controls.update(.5)
        self.assertEqual(self.scene.elapsed, 24)
        self.key(pygame.K_RIGHT, down=False)
        self.controls.update(1)
        self.assertTrue(self.scene.paused)
        self.assertEqual(self.scene.elapsed, 24)

    def test_free_flight_keys_move_camera_without_transport_or_source_clock(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                self.restart(paused=paused)
                self.key(pygame.K_RIGHT)
                self.key(pygame.K_f)
                self.key(pygame.K_f, down=False)
                elapsed, source = self.scene.elapsed, self.scene.model_source
                eye = self.scene.eye.copy()
                matrix = self.scene.objects[0].matrix.copy()
                for key in (pygame.K_SPACE, pygame.K_LEFT, pygame.K_r):
                    self.key(key)
                self.controls.update(9)
                self.assertEqual((self.scene.elapsed, self.scene.model_source, self.scene.paused),
                                 (elapsed, source, paused))
                self.assertFalse(np.allclose(self.scene.eye, eye))
                self.assertFalse(np.allclose(self.scene.objects[0].matrix, matrix))
                self.key(pygame.K_f)
                resumed = self.scene.elapsed
                self.controls.update(.5)
                self.assertAlmostEqual(self.scene.elapsed, resumed + (0 if paused else .5))

    def test_held_a_is_released_on_entry_and_cannot_toggle_sources_after_return(self):
        self.key(pygame.K_a)
        automatic = self.scene.auto_sources
        self.key(pygame.K_f)
        self.key(pygame.K_f, down=False)
        eye = self.scene.eye.copy()
        self.key(pygame.K_a, repeat=True)
        self.controls.update(.1)
        np.testing.assert_array_equal(self.scene.eye, eye)
        self.key(pygame.K_a, down=False)
        self.key(pygame.K_a)
        self.controls.update(.1)
        self.assertFalse(np.allclose(self.scene.eye, eye))
        self.key(pygame.K_f)
        self.key(pygame.K_a, repeat=True)
        self.key(pygame.K_a)
        self.assertEqual(self.scene.auto_sources, automatic)
        self.key(pygame.K_a, down=False)
        self.key(pygame.K_a)
        self.assertEqual(self.scene.auto_sources, not automatic)


@unittest.skipIf(FlybyControls is None, "Нужны зависимости requirements-space.txt")
class FlybyDynamicShadowTest(unittest.TestCase):
    def test_free_camera_keeps_dynamic_shadow_signature_and_static_cache(self):
        renderer = SpaceRenderer.__new__(SpaceRenderer)
        renderer.ctx = SimpleNamespace(enable_only=Mock())
        renderer.shadow = SimpleNamespace(size=(128, 128))
        renderer._shadow_target = SimpleNamespace(use=Mock(), clear=Mock())
        renderer._depth_program = {"light_mvp": Mock()}
        renderer.look = SimpleNamespace(hull={"u_shadow_vp": Mock(), "u_shadow_enabled": Mock()})
        renderer._light_axes = np.eye(3)
        depth = Mock()
        renderer.meshes = [[SimpleNamespace(depth_vao=depth)]]
        objects = (SimpleNamespace(position=np.zeros(3), radius=1, matrix=np.eye(4)),)
        scene = SimpleNamespace(is_flyby=True, camera_mode="flight", objects=objects)
        renderer._shadow_objects, renderer._shadow_signature = objects, None
        renderer._update_shadow(scene, [])
        objects[0].matrix[0, 0] = .9
        renderer._update_shadow(scene, [])
        self.assertEqual(depth.render.call_count, 2)
        renderer._update_shadow(scene, [])
        self.assertEqual(depth.render.call_count, 2)
        scene.is_flyby = False
        objects[0].matrix[0, 0] = .8
        renderer._update_shadow(scene, [])
        self.assertEqual(depth.render.call_count, 2)


if __name__ == "__main__":
    unittest.main()
