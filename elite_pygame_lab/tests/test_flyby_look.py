"""Последовательность источников и относительный взгляд без изменения маршрута."""

import unittest

try:
    import numpy as np
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_flyby import FlybyScene
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "Для сцены OpenGL нужен numpy из requirements-space.txt")
class FlybySourceAndLookTest(unittest.TestCase):
    def test_default_classic_and_six_sources_have_identical_manual_and_timer_order(self):
        scene = FlybyScene()
        self.assertEqual(scene.model_source, "zx")
        self.assertEqual(scene.ships, scene.catalog.classic)
        order = ("zx", "elite", "oolite", "griff", "pioneer", "mixed")
        self.assertEqual(scene.source_options, order)
        self.assertEqual(scene.catalog.sources, order[1:])
        timed = FlybyScene()
        for turn in range(13):
            with self.subTest(turn=turn):
                timed.seek(turn * 8)
                self.assertEqual(scene.model_source, order[turn % 6])
                self.assertEqual(timed.model_source, scene.model_source)
                scene.cycle_source()

    def test_manual_mode_m_changes_immediately_and_t_keeps_zx_without_texture(self):
        scene = FlybyScene(source_interval=0)
        scene.cycle_textures()
        self.assertEqual((scene.model_source, scene.texture_source), ("zx", "author"))
        scene.update(20)
        self.assertEqual(scene.model_source, "zx")
        scene.cycle_source()
        self.assertEqual(scene.model_source, "elite")
        scene.update(500)
        self.assertEqual(scene.model_source, "elite")
        scene.toggle_auto_sources()
        scene.update(7.5)
        scene.cycle_source()
        self.assertEqual(scene.model_source, "oolite")
        scene.update(7.9)
        self.assertEqual(scene.model_source, "oolite")
        scene.update(.1)
        self.assertEqual(scene.model_source, "griff")

    def test_explicit_source_at_and_empty_guest_start_are_reproducible(self):
        scene = FlybyScene(model_source="griff")
        scene.seek(16)
        self.assertEqual(scene.model_source, "mixed")
        scene.seek(0)
        self.assertEqual(scene.model_source, "griff")
        scene = FlybyScene(20, source_interval=0)
        self.assertTrue(scene.empty_slot)
        self.assertEqual(scene.model_source, "zx")
        self.assertLess(scene.selected, len(scene.ships))
        scene.set_source("pioneer")
        self.assertFalse(scene.empty_slot)
        self.assertEqual(scene.selected, 20)

    def test_look_can_face_backwards_and_pitch_limits_keep_orthonormal_up(self):
        scene = FlybyScene(source_interval=0)
        original = scene.forward.copy()
        eye = scene.eye.copy()
        scene.look_drag(np.pi / .005, 0)
        np.testing.assert_allclose(scene.forward, -original, atol=1e-12)
        for dy in (-1e6, 1e6):
            scene.look_drag(0, dy)
            right, up, forward = scene._view_basis()
            basis = np.array((right, up, -forward))
            np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
            self.assertGreater(np.linalg.det(basis), .999999)
            # За вертикальным пределом камера не переворачивается.
            before = scene.forward.copy()
            scene.look_drag(0, dy)
            np.testing.assert_allclose(scene.forward, before, atol=1e-12)
            self.assertTrue(np.isfinite(scene.view_projection(.6)).all())
        np.testing.assert_array_equal(scene.eye, eye)
        scene.reset_look()
        np.testing.assert_allclose(scene.forward, original, atol=1e-12)

    def test_user_look_persists_across_source_timer_seek_textures_and_resize(self):
        scene = FlybyScene()
        route = FlybyScene()
        scene.look_drag(127, -46)
        offset = (scene._look_yaw, scene._look_pitch)
        for seconds in (7.5, 8, 24, 72, 121, 126):
            scene.seek(seconds)
            route.seek(seconds)
            np.testing.assert_array_equal(scene.eye, route.eye)
            self.assertEqual((scene._look_yaw, scene._look_pitch), offset)
            self.assertLess(np.dot(scene.forward, route.forward), .9)
            scene.cycle_source()
            scene.cycle_textures()
            scene.toggle_auto_sources()
            scene.view_projection(.7)
            scene.view_projection(2)
            self.assertEqual((scene._look_yaw, scene._look_pitch), offset)
            scene.reset_look()
            np.testing.assert_allclose(scene.view_projection(1.6), route.view_projection(1.6), atol=1e-12)
            scene.look_drag(127, -46)

    def test_home_view_uses_the_staged_camera_up_for_relative_rotations(self):
        scene = FlybyScene(source_interval=0)
        scene._camera_up = scene._view_basis()[0]
        staged_up = scene.up.copy()
        home = scene.view_projection(1.6)
        scene.look_drag(30, -40)
        self.assertGreater(np.dot(scene.up, staged_up), .9)
        scene.reset_look()
        np.testing.assert_allclose(scene.view_projection(1.6), home, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
