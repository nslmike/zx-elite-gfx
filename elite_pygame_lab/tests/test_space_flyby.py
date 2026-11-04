"""Поведение постановочного пролёта, независимое от OpenGL-контекста."""

import unittest
from fractions import Fraction

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_flyby import FlybyScene, _station_dock


@unittest.skipIf(np is None, "Для сцены OpenGL нужен numpy из requirements-space.txt")
class SpaceFlybyTest(unittest.TestCase):
    def test_finite_extreme_time_and_interval_do_not_overflow_or_switch_at_zero(self):
        for interval, seconds in ((.001, 1e308), (1e-310, 1.0), (5e-324, 1.0)):
            with self.subTest(interval=interval, seconds=seconds):
                scene = FlybyScene(source_interval=interval)
                scene.seek(0)
                scene.update(0)
                self.assertEqual(scene.model_source, "elite")
                scene.seek(interval * .5)
                self.assertEqual(scene.model_source, "elite")
                scene.seek(seconds)
                changes = Fraction(seconds) // Fraction(interval)
                self.assertEqual(scene.model_source, scene.catalog.sources[changes % len(scene.catalog.sources)])
                self.assertTrue(np.isfinite(scene.eye).all())
                self.assertTrue(all(np.isfinite(obj.matrix).all() for obj in scene.objects))

    def test_seek_matches_frame_updates_and_large_steps_including_source_changes(self):
        direct = FlybyScene()
        incremental = FlybyScene()
        for _ in range(66 * 15):
            incremental.update(1 / 15)
        direct.seek(66)
        self.assertEqual((direct.camera_mode, direct.selected, direct.model_source),
                         ("flyby", 20, "pioneer"))
        self.assertEqual((incremental.selected, incremental.model_source),
                         (direct.selected, direct.model_source))
        np.testing.assert_allclose(incremental.eye, direct.eye, atol=1e-9)
        np.testing.assert_allclose(incremental.view_projection(1.6), direct.view_projection(1.6), atol=1e-9)
        np.testing.assert_allclose([obj.matrix for obj in incremental.objects],
                                   [obj.matrix for obj in direct.objects], atol=1e-9)
        large = FlybyScene()
        large.update(289.25)
        direct.seek(289.25)
        self.assertEqual(large.model_source, direct.model_source)
        np.testing.assert_allclose(large.eye, direct.eye)

    def test_manual_source_restarts_just_source_timer_and_textures_stay_manual(self):
        scene = FlybyScene()
        scene.update(7.5)
        view = scene.view_projection(1.6).copy()
        transforms = {obj.model_id: obj.matrix.copy() for obj in scene.objects}
        scene.cycle_source()
        self.assertEqual(scene.model_source, "oolite")
        np.testing.assert_array_equal(scene.view_projection(1.6), view)
        # Корпуса не меняют фазу вращения при смене их геометрии.
        for obj in scene.objects:
            if obj.model_id not in (0, 19):
                np.testing.assert_array_equal(obj.position, transforms[obj.model_id][:3, 3])
            if obj.model_id not in (0, 3, 19):
                np.testing.assert_array_equal(obj.matrix, transforms[obj.model_id])
        scene.cycle_textures()
        self.assertEqual(scene.texture_source, "procedural")
        scene.update(7.9)
        self.assertEqual(scene.model_source, "oolite")
        scene.update(.1)
        self.assertEqual(scene.model_source, "griff")
        self.assertEqual(scene.texture_source, "procedural")
        scene.cycle_textures()
        self.assertEqual(scene.texture_source, "author")

    def test_manual_mode_can_enable_default_auto_and_pause_it(self):
        scene = FlybyScene(source_interval=0)
        scene.update(100)
        self.assertFalse(scene.auto_sources)
        self.assertEqual(scene.model_source, "elite")
        eye = scene.eye.copy()
        self.assertTrue(scene.toggle_auto_sources())
        np.testing.assert_array_equal(scene.eye, eye)
        scene.update(8)
        self.assertEqual(scene.model_source, "oolite")
        self.assertFalse(scene.toggle_auto_sources())
        scene.update(40)
        self.assertEqual(scene.model_source, "oolite")

    def test_real_imported_heroes_show_up_in_first_default_lap(self):
        scene = FlybyScene()
        scene.seek(18)
        self.assertEqual((scene.selected, scene.ships[scene.selected].source_id), (12, "griff"))
        scene.seek(66)
        self.assertEqual((scene.selected, scene.ships[scene.selected].source_id), (20, "pioneer"))
        courier_depth = np.dot(scene.selected_object.position - scene.eye, scene.forward)
        # Станции не перекрывают курьера, как было при внешнем гостевом месте.
        for station in (0, 19):
            depth = np.dot(scene.objects[station].position - scene.eye, scene.forward)
            self.assertLess(courier_depth + scene.selected_object.radius,
                            depth + scene.objects[station].radius)
        ship = scene.ships[20]
        points = np.column_stack((ship.vertices()[:, :3], np.ones(len(ship.vertices()))))
        clip = (scene.view_projection(1360 / 820) @ scene.selected_object.matrix @ points.T).T
        self.assertTrue(np.all(clip[:, 3] > 0))
        ndc = clip[:, :2] / clip[:, 3:]
        self.assertTrue(.38 < np.ptp(ndc[:, 0]) / 2 < .60)
        self.assertTrue(np.all(np.abs(ndc) < .85))

    def test_all_slots_camera_clearance_and_no_empty_frames_without_guest(self):
        scene = FlybyScene(model_source="mixed", source_interval=0)
        visited = []
        for slot in range(scene.slot_count):
            scene.seek(slot * scene.seconds_per_ship)
            visited.append(scene.selected)
        self.assertEqual(set(visited), set(range(21)))
        for source in ("elite", "mixed"):
            scene = FlybyScene(model_source=source, source_interval=0)
            heights = []
            for seconds in np.linspace(0, scene.duration, 253):
                scene.seek(seconds)
                heights.append(scene.eye[1])
                centers = [scene.project(obj.position, 1360, 820) for obj in scene.objects]
                self.assertTrue(any(center is not None for center in centers), (source, seconds))
                for obj in scene.objects:
                    self.assertGreater(np.linalg.norm(scene.eye - obj.position) - obj.radius, .8,
                                       (source, seconds, obj.model_id))
            self.assertGreater(max(heights), 1)
            self.assertLess(min(heights), -1)

    def test_camera_and_velocity_close_the_lap_and_cross_station_knots_smoothly(self):
        scene = FlybyScene(source_interval=0)
        initial = scene.view_projection(1.6).copy()
        scene.seek(scene.duration)
        np.testing.assert_allclose(scene.view_projection(1.6), initial, atol=1e-12)
        epsilon = .001
        for seconds in (6, 60, 66, 72, scene.duration):
            scene.seek(seconds - epsilon)
            before = scene.eye
            scene.seek(seconds)
            center = scene.eye
            scene.seek(seconds + epsilon)
            after = scene.eye
            np.testing.assert_allclose((after - center) / epsilon, (center - before) / epsilon, atol=.002)

    def test_stations_rotate_about_actual_dock_in_classic_and_imported_meshes(self):
        catalog = AssetCatalog()
        for source in ("elite", "oolite"):
            scene = FlybyScene(model_source=source, catalog=catalog, source_interval=0)
            original = [obj.matrix.copy() for obj in scene.objects]
            anchors = {}
            for station in (0, 19):
                dock = _station_dock(scene.ships[station])
                # Две точки закрепляют ось, не только плоскость входа.
                anchors[station] = np.column_stack((np.r_[dock, 1], np.r_[dock + (0, 0, -.5), 1]))
            if source == "oolite":
                self.assertAlmostEqual(_station_dock(scene.ships[19])[1], .00572943, places=6)
            scene.update(17.25)
            for station, anchors_on_axis in anchors.items():
                np.testing.assert_allclose(scene.objects[station].matrix @ anchors_on_axis,
                                           original[station] @ anchors_on_axis, atol=1e-12)
            for obj, first in zip(scene.objects, original):
                self.assertFalse(np.allclose(obj.matrix[:3, :3], first[:3, :3]), obj.model_id)
                np.testing.assert_allclose(np.linalg.svd(obj.matrix[:3, :3])[1], obj.radius)
                self.assertGreater(np.linalg.det(obj.matrix[:3, :3]), 0)

    def test_twenty_and_twenty_one_objects_use_same_camera_and_common_poses(self):
        scene = FlybyScene(source_interval=0)
        scene.seek(66)
        projection = scene.view_projection(1.6).copy()
        matrices = {obj.model_id: obj.matrix.copy() for obj in scene.objects}
        scene.set_source("pioneer")
        self.assertEqual(len(scene.objects), 21)
        np.testing.assert_array_equal(scene.view_projection(1.6), projection)
        for obj in scene.objects[:20]:
            np.testing.assert_array_equal(obj.matrix, matrices[obj.model_id])
        scene.set_source("elite")
        self.assertEqual(len(scene.objects), 20)
        np.testing.assert_array_equal(scene.view_projection(1.6), projection)


if __name__ == "__main__":
    unittest.main()
