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
    @staticmethod
    def _project_centers(scene, width=1360, height=820):
        positions = np.array([obj.position for obj in scene.objects])
        points = np.column_stack((positions, np.ones(len(positions))))
        clip = (scene.view_projection(width / height) @ points.T).T
        ndc = clip[:, :3] / clip[:, 3:]
        in_frame = (clip[:, 3] > 0) & np.all(np.abs(ndc) <= 1, axis=1)
        depths = (positions - scene.eye) @ scene.forward
        return positions, ndc, in_frame, depths

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
        # Курьер остаётся ближним планом перед обеими станциями.
        for station in (0, 19):
            depth = np.dot(scene.objects[station].position - scene.eye, scene.forward)
            self.assertLess(courier_depth + scene.selected_object.radius,
                            depth + scene.objects[station].radius)

    def test_all_slots_external_inward_camera_clearance_and_no_empty_frames(self):
        scene = FlybyScene(model_source="mixed", source_interval=0)
        visited = []
        for slot in range(scene.slot_count):
            scene.seek(slot * scene.seconds_per_ship)
            visited.append(scene.selected)
        self.assertEqual(set(visited), set(range(21)))
        for source in ("elite", "mixed"):
            scene = FlybyScene(model_source=source, source_interval=0)
            for seconds in np.linspace(0, scene.duration, 253):
                scene.seek(seconds)
                radial = scene.eye[[0, 2]]
                self.assertGreater(np.linalg.norm(radial), scene.ring_radius,
                                   (source, seconds))
                inward = -np.dot(scene.forward[[0, 2]], radial) / np.linalg.norm(radial)
                self.assertGreater(inward, .6, (source, seconds))
                centers = [scene.project(obj.position, 1360, 820) for obj in scene.objects]
                self.assertTrue(any(center is not None for center in centers), (source, seconds))
                for obj in scene.objects:
                    self.assertGreater(np.linalg.norm(scene.eye - obj.position) - obj.radius, .8,
                                       (source, seconds, obj.model_id))
                if len(scene.objects) > 20:
                    guest = scene.objects[20]
                    for station in (scene.objects[0], scene.objects[19]):
                        self.assertGreater(np.linalg.norm(guest.position - station.position),
                                           guest.radius + station.radius,
                                           (source, seconds, station.model_id))
            # Облёт показывает верх и низ корпуса на разных участках круга.
            # Мировой Y не годится: сами экспонаты стоят на разной высоте.
            relative_heights = []
            for slot in range(scene.slot_count):
                scene.seek(slot * scene.seconds_per_ship)
                relative_heights.append(scene.eye[1] - scene.selected_object.position[1])
            self.assertGreater(max(relative_heights), 1)
            self.assertLess(min(relative_heights), -1)

    def test_ring_is_visible_behind_hero_throughout_classic_and_guest_laps(self):
        width, height = 1360, 820
        for source in ("elite", "mixed"):
            scene = FlybyScene(model_source=source, source_interval=0)
            # Встречи и три фазы перехода у каждого места, включая гостя.
            for seconds in np.arange(scene.slot_count * 4) * scene.seconds_per_ship / 4:
                with self.subTest(source=source, seconds=seconds):
                    scene.seek(seconds)
                    positions, ndc, in_frame, depths = self._project_centers(scene)
                    hero = scene.selected
                    self.assertTrue(in_frame[hero])
                    # Центр должен быть дальше всей сферы героя, а не просто
                    # перед камерой: иначе сосед сбоку выдаётся за задний план.
                    behind = depths > depths[hero] + scene.selected_object.radius
                    background = in_frame & behind
                    self.assertGreaterEqual(np.count_nonzero(background), len(scene.objects) // 2)
                    pixels_from_hero = ((ndc[:, :2] - ndc[hero, :2])
                                        * np.array((width / 2, height / 2)))
                    separate = np.linalg.norm(pixels_from_hero, axis=1) > width * .15
                    self.assertGreaterEqual(np.count_nonzero(background & separate), 4)
                    # Противоположная камере полудуга должна быть видна почти
                    # целиком; сами сферы и их окклюзию здесь не считаем.
                    far_arc = positions[:, [0, 2]] @ scene.eye[[0, 2]] < 0
                    self.assertGreaterEqual(np.count_nonzero(far_arc), len(scene.objects) // 2 - 1)
                    self.assertGreaterEqual(5 * np.count_nonzero(in_frame & far_arc),
                                            4 * np.count_nonzero(far_arc))

    def test_heroes_are_close_and_framed_at_each_encounter(self):
        for source in ("elite", "mixed"):
            scene = FlybyScene(model_source=source, source_interval=0)
            for slot in range(scene.slot_count):
                with self.subTest(source=source, slot=slot):
                    scene.seek(slot * scene.seconds_per_ship)
                    obj = scene.selected_object
                    points = scene.ships[scene.selected].vertices()[:, :3]
                    points = np.column_stack((points, np.ones(len(points))))
                    clip = (scene.view_projection(1360 / 820) @ obj.matrix @ points.T).T
                    self.assertTrue(np.all(clip[:, 3] > 0))
                    ndc = clip[:, :2] / clip[:, 3:]
                    self.assertTrue(np.all(np.abs(ndc) < 1))
                    # Даже маленький груз различим: длинная сторона не меньше
                    # 12% высоты кадра. Близость измеряем в радиусах корпуса,
                    # чтобы большой силуэт не достигался одним масштабом окна.
                    pixels = np.ptp(ndc, axis=0) * np.array((1360 / 2, 820 / 2))
                    self.assertGreater(pixels.max(), 820 * .12)
                    self.assertLess(np.linalg.norm(obj.position - scene.eye) / obj.radius, 7)

    def test_hero_center_stays_in_frame_across_transitions_and_window_shapes(self):
        for source in ("elite", "mixed"):
            scene = FlybyScene(model_source=source, source_interval=0)
            for seconds in np.linspace(0, scene.duration, scene.slot_count * 4 + 1):
                scene.seek(seconds)
                for width, height in ((1360, 820), (1280, 720), (1024, 768), (820, 820), (820, 1024)):
                    with self.subTest(source=source, seconds=seconds, size=(width, height)):
                        _, ndc, in_frame, _ = self._project_centers(scene, width, height)
                        self.assertTrue(in_frame[scene.selected])
                        # Центр героя не прижат к границе окна.
                        self.assertLess(np.abs(ndc[scene.selected, :2]).max(), .9)

    def test_representative_heroes_are_large_and_fit_standard_and_square_windows(self):
        # Классика, импортный Griff, обе станции и крупный гостевой Pioneer.
        for source, seconds in (("elite", 0), ("elite", 60), ("elite", 66),
                                ("mixed", 18), ("mixed", 66), ("mixed", 72)):
            with self.subTest(source=source, seconds=seconds):
                scene = FlybyScene(model_source=source, source_interval=0)
                scene.seek(seconds)
                vertices = scene.ships[scene.selected].vertices()[:, :3]
                points = np.column_stack((vertices, np.ones(len(vertices))))
                for width, height in ((1360, 820), (820, 820)):
                    clip = (scene.view_projection(width / height)
                            @ scene.selected_object.matrix @ points.T).T
                    self.assertTrue(np.all(clip[:, 3] > 0))
                    ndc = clip[:, :2] / clip[:, 3:]
                    self.assertTrue(np.all(np.abs(ndc) < .98))
                    if width > height:
                        pixels = np.ptp(ndc, axis=0) * np.array((width / 2, height / 2))
                        self.assertGreater(pixels.max(), height * .25)

    def test_camera_and_velocity_close_the_lap_and_cross_station_knots_smoothly(self):
        scene = FlybyScene(source_interval=0)
        initial = scene.view_projection(1.6).copy()
        scene.seek(scene.duration)
        np.testing.assert_allclose(scene.view_projection(1.6), initial, atol=1e-12)
        epsilon = .001
        # Узлы и промежутки всего круга: смена верхнего и нижнего ракурса
        # должна сохранять скорость и плавное направление взгляда.
        for seconds in np.arange(1, scene.slot_count * 4 + 1) * scene.seconds_per_ship / 4:
            scene.seek(seconds - epsilon)
            before = scene.eye
            before_forward = scene.forward
            scene.seek(seconds)
            center = scene.eye
            center_forward = scene.forward
            scene.seek(seconds + epsilon)
            after = scene.eye
            after_forward = scene.forward
            np.testing.assert_allclose((after - center) / epsilon, (center - before) / epsilon, atol=.002)
            np.testing.assert_allclose((after_forward - center_forward) / epsilon,
                                       (center_forward - before_forward) / epsilon, atol=.001)

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
