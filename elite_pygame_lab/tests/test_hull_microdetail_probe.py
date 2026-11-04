"""PNG-проба и её свободная камера с общим контроллером без окна и GPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import moderngl
    import numpy as np
    import pygame
except ModuleNotFoundError:
    moderngl = None

if moderngl is not None:
    from elite_lab.space_viewer import SpaceRenderer
    from tools.hull_microdetail_probe import ProbeFlightControls, ProbeFlightScene, ProbeScene, probe_cameras


@unittest.skipIf(moderngl is None, "Для пробы нужны requirements-space.txt")
class HullMicrodetailProbeTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(pygame.display, "get_init", return_value=False))
        self.enterContext(patch.object(pygame.display, "set_caption"))

    def controls(self, scene=None):
        scene = scene or ProbeFlightScene()
        hud = SimpleNamespace(menu=None, tab_at=lambda point: None)
        controls = ProbeFlightControls(scene, hud, True)
        self.addCleanup(controls.release)
        return controls

    def event(self, controls, event_type, **values):
        return controls.event(pygame.event.Event(event_type, **values), (1360, 820), (2720, 1640))

    def key(self, controls, key, *, up=False, repeat=False):
        return self.event(controls, pygame.KEYUP if up else pygame.KEYDOWN,
                          key=key, mod=0, repeat=repeat)

    def test_flight_starts_with_the_real_close_camera_basis_and_no_first_frame_jump(self):
        scene = ProbeFlightScene()
        target, eye, fov, _ = probe_cameras()["close"]
        obj = scene.objects[9]
        world_eye = obj.matrix[:3, :3] @ eye + obj.position
        world_target = obj.matrix[:3, :3] @ target + obj.position
        forward = world_target - world_eye
        forward /= np.linalg.norm(forward)
        np.testing.assert_allclose(scene.eye, world_eye, atol=1e-12)
        np.testing.assert_allclose(scene.forward, forward, atol=1e-12)
        self.assertEqual(scene.fov, fov)
        basis = np.array(scene._view_basis())
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        self.assertGreater(basis[1, 1], 0)
        np.testing.assert_allclose(scene.project(world_target, 1360, 820), (680, 410), atol=1e-10)
        projection = scene.view_projection(1.6)
        self.assertTrue(np.isfinite(projection).all())
        controls = self.controls(scene)
        self.assertTrue(controls.looking)
        controls.update(.1)
        np.testing.assert_array_equal(scene.eye, world_eye)
        np.testing.assert_array_equal(scene.view_projection(1.6), projection)

    def test_flight_starts_with_baseline_and_b_toggles_only_trial_layer(self):
        scene = ProbeFlightScene()
        initial_eye = scene.eye.copy()
        self.assertFalse(scene.microdetail_enabled)
        controls = self.controls(scene)
        controls.hud.menu = "textures"
        self.assertTrue(self.key(controls, pygame.K_b))
        self.assertTrue(scene.microdetail_enabled)
        np.testing.assert_array_equal(scene.eye, initial_eye)
        # SDL repeat и второй KEYDOWN без KEYUP не мигают слоем A/B.
        self.key(controls, pygame.K_b, repeat=True)
        self.key(controls, pygame.K_b)
        self.assertTrue(scene.microdetail_enabled)
        scene._flight_eye += (1, 2, 3)
        self.assertTrue(self.key(controls, pygame.K_r))
        self.assertTrue(scene.microdetail_enabled)
        np.testing.assert_array_equal(scene.eye, initial_eye)
        self.assertIsNone(controls.hud.menu)
        self.key(controls, pygame.K_b, up=True)
        self.assertTrue(self.key(controls, pygame.K_b))
        self.assertFalse(scene.microdetail_enabled)
        np.testing.assert_array_equal(scene.eye, initial_eye)

    def test_reset_restores_full_pose_fov_and_stops_old_holds_in_both_profiles(self):
        for profile in ("modern", "elite"):
            with self.subTest(profile=profile):
                controls = self.controls()
                scene = controls.scene
                eye, basis = scene.eye, np.array(scene._view_basis())
                projection = scene.view_projection(1.6)
                if profile == "elite":
                    self.key(controls, pygame.K_c)
                self.key(controls, pygame.K_b)
                self.key(controls, pygame.K_w)
                self.key(controls, pygame.K_RIGHT)
                controls.update(.1)
                self.assertFalse(np.allclose(scene.eye, eye))
                self.assertFalse(np.allclose(scene._view_basis(), basis))
                scene.turn_flight(0, 0, 2.1)
                scene.zoom(1)
                scene.fov = 67
                self.key(controls, pygame.K_ESCAPE)
                self.assertFalse(controls.looking)
                self.key(controls, pygame.K_r)
                self.assertTrue(controls.looking)
                self.assertEqual(scene.flight_profile, profile)
                self.assertEqual(scene.flight_speed, 0)
                self.assertTrue(scene.microdetail_enabled)
                np.testing.assert_array_equal(scene.eye, eye)
                np.testing.assert_array_equal(scene._view_basis(), basis)
                np.testing.assert_array_equal(scene.view_projection(1.6), projection)
                # Клавиши ещё физически удерживаются, но сброс не должен
                # сдвинуть или повернуть исходную позу на следующем кадре.
                controls.update(.1)
                np.testing.assert_array_equal(scene.eye, eye)
                np.testing.assert_array_equal(scene._view_basis(), basis)
                self.key(controls, pygame.K_w, repeat=True)
                controls.update(.1)
                np.testing.assert_array_equal(scene.eye, eye)

    def test_reset_from_overview_reenters_flight_without_changing_a_b_or_profile(self):
        controls = self.controls()
        scene = controls.scene
        eye, basis = scene.eye, np.array(scene._view_basis())
        self.key(controls, pygame.K_c)
        self.key(controls, pygame.K_b)
        self.key(controls, pygame.K_o)
        self.assertEqual(scene.camera_mode, "overview")
        self.assertFalse(controls.looking)
        self.key(controls, pygame.K_r)
        self.assertEqual(scene.camera_mode, "flight")
        self.assertTrue(controls.looking)
        self.assertEqual(scene.flight_profile, "elite")
        self.assertEqual(scene.flight_speed, 0)
        self.assertTrue(scene.microdetail_enabled)
        np.testing.assert_array_equal(scene.eye, eye)
        np.testing.assert_array_equal(scene._view_basis(), basis)

    def test_prestarted_flight_uses_real_events_and_releases_capture_on_focus_loss_and_quit(self):
        with patch.object(pygame.display, "get_init", return_value=True), \
             patch.object(pygame.event, "get_grab", return_value=False), \
             patch.object(pygame.event, "set_grab") as grab, \
             patch.object(pygame.mouse, "get_visible", return_value=True), \
             patch.object(pygame.mouse, "set_visible") as visible, \
             patch.object(pygame.mouse, "get_rel", return_value=(0, 0)) as relative:
            controls = self.controls()
            scene = controls.scene
            self.assertTrue(controls.looking)
            grab.assert_called_with(True)
            visible.assert_called_with(False)
            relative.assert_called_once_with()
            eye, forward = scene.eye, scene.forward
            self.key(controls, pygame.K_w)
            controls.update(.1)
            self.assertFalse(np.allclose(scene.eye, eye))
            self.event(controls, pygame.MOUSEMOTION, rel=(40, -20))
            self.assertFalse(np.allclose(scene.forward, forward))
            self.key(controls, pygame.K_c)
            self.key(controls, pygame.K_w, up=True)
            self.key(controls, pygame.K_w)
            controls.update(.1)
            self.assertGreater(scene.flight_speed, 0)
            self.event(controls, pygame.WINDOWFOCUSLOST)
            self.assertFalse(controls.looking)
            self.assertEqual(scene.flight_speed, 0)
            self.assertFalse(controls._flight_keys)
            grab.assert_called_with(False)
            visible.assert_called_with(True)
            self.key(controls, pygame.K_w, repeat=True)
            eye = scene.eye
            controls.update(.1)
            np.testing.assert_array_equal(scene.eye, eye)
            self.key(controls, pygame.K_r)
            self.assertTrue(controls.looking)
            grab.assert_called_with(True)
            self.assertFalse(self.event(controls, pygame.QUIT))
            self.assertFalse(controls.looking)
            self.assertIsNone(controls._mouse_state)
            grab.assert_called_with(False)
            visible.assert_called_with(True)

    def test_png_scene_reuses_static_shadow_contract(self):
        scene = ProbeScene((), (), np.zeros(3), np.array((0., 0., -1.)))
        renderer = SpaceRenderer.__new__(SpaceRenderer)
        renderer._shadow_objects = scene.objects
        # Общий renderer читает is_flyby даже при готовой карте теней.
        # Для статичного экспоната он должен выйти до обращения к GPU.
        renderer._update_shadow(scene, [])


if __name__ == "__main__":
    unittest.main()
