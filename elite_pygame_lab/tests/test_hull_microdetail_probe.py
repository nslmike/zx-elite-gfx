"""Контракт одиночной PNG-пробы с общим рендерером без окна и GPU."""

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
    from tools.hull_microdetail_probe import ProbeFlightControls, ProbeFlightScene, ProbeScene


@unittest.skipIf(moderngl is None, "Для пробы нужны requirements-space.txt")
class HullMicrodetailProbeTest(unittest.TestCase):
    def test_flight_starts_with_baseline_and_b_toggles_only_trial_layer(self):
        scene = ProbeFlightScene()
        initial_eye = scene.eye.copy()
        self.assertFalse(scene.microdetail_enabled)
        hud = SimpleNamespace(menu="textures")
        with patch.object(pygame.display, "set_caption"):
            controls = ProbeFlightControls(scene, hud, True)
            toggle = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b)
            reset = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)
            self.assertTrue(controls.event(toggle, (1360, 820), (2720, 1640)))
            self.assertTrue(scene.microdetail_enabled)
            np.testing.assert_array_equal(scene.eye, initial_eye)
            scene._flight_eye += (1, 2, 3)
            self.assertTrue(controls.event(reset, (1360, 820), (2720, 1640)))
            self.assertTrue(scene.microdetail_enabled)
            np.testing.assert_array_equal(scene.eye, initial_eye)
            self.assertIsNone(hud.menu)
            self.assertTrue(controls.event(toggle, (1360, 820), (2720, 1640)))
            self.assertFalse(scene.microdetail_enabled)
            np.testing.assert_array_equal(scene.eye, initial_eye)

    def test_png_scene_reuses_static_shadow_contract(self):
        scene = ProbeScene((), (), np.zeros(3), np.array((0., 0., -1.)))
        renderer = SpaceRenderer.__new__(SpaceRenderer)
        renderer._shadow_objects = scene.objects
        # Общий renderer читает is_flyby даже при готовой карте теней.
        # Для статичного экспоната он должен выйти до обращения к GPU.
        renderer._update_shadow(scene, [])


if __name__ == "__main__":
    unittest.main()
