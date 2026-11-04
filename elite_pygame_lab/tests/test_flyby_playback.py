"""Два времени облёта, обратимая перемотка и ручной якорь источников."""

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from elite_lab.model import COBRA_ID, DODO_ID
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_flyby import FlybyScene, _station_dock


@unittest.skipIf(np is None, "Для сцены OpenGL нужен numpy из requirements-space.txt")
class FlybyPlaybackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = AssetCatalog()

    def scene(self, **kwargs):
        return FlybyScene(catalog=self.catalog, **kwargs)

    def assert_same_poses(self, actual, expected):
        np.testing.assert_allclose([obj.matrix for obj in actual.objects],
                                   [obj.matrix for obj in expected.objects], atol=1e-12)

    def test_pause_rotates_ships_stations_and_rocks_and_resumes_without_phase_jump(self):
        scene = self.scene(model_source="oolite")
        scene.seek(7.5)
        eye, target, selected = scene.eye, scene.target, scene.selected
        before = [obj.matrix.copy() for obj in scene.objects]
        dock_axes = {}
        for station in (0, DODO_ID):
            dock = _station_dock(scene.ships[station])
            dock_axes[station] = np.column_stack((np.r_[dock, 1], np.r_[dock + (0, 0, -.5), 1]))
        self.assertTrue(scene.toggle_pause())
        scene.update(12.25)
        self.assertEqual((scene.elapsed, scene.animation_elapsed), (7.5, 19.75))
        self.assertEqual((scene.selected, scene.model_source), (selected, "oolite"))
        np.testing.assert_array_equal(scene.eye, eye)
        np.testing.assert_array_equal(scene.target, target)
        for model_id in (COBRA_ID, 0, DODO_ID, 3, 5, 6, 7):
            self.assertFalse(np.allclose(scene.objects[model_id].matrix, before[model_id]))
        for station, axis in dock_axes.items():
            np.testing.assert_allclose(scene.objects[station].matrix @ axis,
                                       before[station] @ axis, atol=1e-12)
        expected = self.scene(model_source="oolite", source_interval=0)
        expected.seek(19.75)
        self.assert_same_poses(scene, expected)
        self.assertFalse(scene.toggle_pause())
        scene.update(0)
        self.assert_same_poses(scene, expected)
        scene.update(.5)
        self.assertEqual((scene.elapsed, scene.animation_elapsed, scene.model_source),
                         (8, 20.25, "griff"))
        expected = self.scene(model_source="griff", source_interval=0)
        expected.seek(20.25)
        self.assert_same_poses(scene, expected)

    def test_paused_scrub_reverses_across_laps_and_retains_the_extra_spin(self):
        scene, route = self.scene(), self.scene()
        scene.seek(130.25)
        scene.toggle_pause()
        scene.update(2)
        original = [obj.matrix.copy() for obj in scene.objects]
        for seconds in (268.5, 125.75, 126, 0, 130.25):
            with self.subTest(seconds=seconds):
                delta = seconds - scene.elapsed
                self.assertEqual(scene.scrub(delta), delta)
                self.assertEqual((scene.elapsed, scene.animation_elapsed, scene.paused),
                                 (seconds, seconds + 2, True))
                route.seek(seconds)
                self.assertEqual((scene.selected, scene.empty_slot, scene.model_source),
                                 (route.selected, route.empty_slot, route.model_source))
                np.testing.assert_array_equal(scene.eye, route.eye)
                np.testing.assert_array_equal(scene.target, route.target)
        np.testing.assert_array_equal([obj.matrix for obj in scene.objects], original)

    def test_rewind_clamps_at_zero_and_only_rewinds_the_traversed_rotation_phase(self):
        scene = self.scene(source_interval=0)
        scene.seek(3)
        scene.toggle_pause()
        scene.update(7)
        self.assertEqual(scene.scrub(-100), -3)
        self.assertEqual((scene.elapsed, scene.animation_elapsed), (0, 7))
        at_start = [obj.matrix.copy() for obj in scene.objects]
        self.assertEqual(scene.scrub(-1), 0)
        np.testing.assert_array_equal([obj.matrix for obj in scene.objects], at_start)
        scene.scrub(2)
        expected = self.scene(source_interval=0)
        expected.seek(9)
        self.assert_same_poses(scene, expected)
        self.assertEqual(scene.elapsed, 2)
        scene.seek(4)
        self.assertEqual((scene.elapsed, scene.animation_elapsed), (4, 4))

    def test_manual_source_anchor_replays_in_both_directions_with_short_intervals(self):
        for interval in (8, .000125):
            with self.subTest(interval=interval):
                scene = self.scene(source_interval=interval)
                scene.seek(interval * 1.25)
                scene.toggle_pause()
                scene.update(3)
                scene.cycle_source()
                scene.cycle_textures()
                anchor = scene.elapsed
                self.assertEqual(scene.model_source, "oolite")
                for step, source in ((1, "griff"), (3, "mixed"), (0, "oolite"),
                                     (-1, "elite"), (-1.25, "zx"), (-.25, "elite"), (0, "oolite")):
                    seconds = anchor + step * interval
                    scene.scrub(seconds - scene.elapsed)
                    self.assertEqual(scene.model_source, source)
                    self.assertEqual(scene.texture_source, "procedural")
                    self.assertAlmostEqual(scene.animation_elapsed, seconds + 3)
                    self.assertTrue(scene.paused)

    def test_auto_off_fixes_selection_and_reenable_anchors_the_current_manual_choice(self):
        scene = self.scene()
        scene.seek(26)
        self.assertEqual(scene.model_source, "griff")
        self.assertFalse(scene.toggle_auto_sources())
        scene.cycle_textures()
        for seconds in (0, 260, 2):
            scene.scrub(seconds - scene.elapsed)
            self.assertEqual(scene.model_source, "griff")
        scene.cycle_source()
        scene.scrub(100)
        self.assertEqual(scene.model_source, "pioneer")
        self.assertTrue(scene.toggle_auto_sources())
        scene.scrub(8)
        self.assertEqual(scene.model_source, "mixed")
        scene.scrub(-16)
        self.assertEqual(scene.model_source, "griff")
        self.assertEqual(scene.texture_source, "procedural")


if __name__ == "__main__":
    unittest.main()
