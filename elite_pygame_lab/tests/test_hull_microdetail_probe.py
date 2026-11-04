"""Контракт одиночной PNG-пробы с общим рендерером без окна и GPU."""

import unittest

try:
    import moderngl
    import numpy as np
    import pygame
except ModuleNotFoundError:
    moderngl = None

if moderngl is not None:
    from elite_lab.space_viewer import SpaceRenderer
    from tools.hull_microdetail_probe import ProbeScene


@unittest.skipIf(moderngl is None, "Для пробы нужны requirements-space.txt")
class HullMicrodetailProbeTest(unittest.TestCase):
    def test_png_scene_reuses_static_shadow_contract(self):
        scene = ProbeScene((), (), np.zeros(3), np.array((0., 0., -1.)))
        renderer = SpaceRenderer.__new__(SpaceRenderer)
        renderer._shadow_objects = scene.objects
        # Общий renderer читает is_flyby даже при готовой карте теней.
        # Для статичного экспоната он должен выйти до обращения к GPU.
        renderer._update_shadow(scene, [])


if __name__ == "__main__":
    unittest.main()
