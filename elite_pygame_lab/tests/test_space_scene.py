import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from elite_lab.model import DODO_ID, MODELS
if np is not None:
    from elite_lab.space_mesh import build_mesh
    from elite_lab.space_scene import SpaceScene


@unittest.skipIf(np is None, "Для сцены OpenGL нужен numpy из requirements-space.txt")
class SpaceSceneTest(unittest.TestCase):
    def test_complete_catalog_is_uniformly_spaced_and_scaled(self):
        scene = SpaceScene()
        self.assertEqual([obj.model_id for obj in scene.objects], list(range(len(MODELS))))
        self.assertEqual(len(scene.objects), 20)
        positions = np.array([obj.position for obj in scene.objects])
        np.testing.assert_allclose(np.linalg.norm(positions, axis=1), scene.ring_radius)
        np.testing.assert_allclose(positions[:, 1], 0)
        distances = np.linalg.norm(positions - np.roll(positions, 1, axis=0), axis=1)
        np.testing.assert_allclose(distances, distances[0])
        self.assertGreater(scene.objects[DODO_ID].radius, scene.objects[4].radius * 2)
        self.assertGreater(distances.min(), 2 * max(obj.radius for obj in scene.objects))
        for obj in scene.objects:
            np.testing.assert_allclose(obj.matrix[:3, 3], obj.position)
            np.testing.assert_allclose(np.linalg.svd(obj.matrix[:3, :3])[1], obj.radius)

    def test_overview_fits_hud_and_every_object_can_be_picked_after_resize(self):
        scene = SpaceScene()
        for width, height in ((1360, 820), (1600, 900), (900, 900), (600, 1100), (1700, 500)):
            centers = []
            for obj in scene.objects:
                center = scene.project(obj.position, width, height)
                self.assertIsNotNone(center)
                centers.append(center)
                x, y = center
                self.assertTrue(width * 0.12 < x < width * 0.88)
                self.assertTrue(height * 0.12 < y < height * 0.88)
                self.assertEqual(scene.pick(x, y, width, height), obj.model_id)
            centers = np.array(centers)
            if width / height < 2:
                self.assertTrue(0.65 < np.ptp(centers[:, 0]) / width < 0.76)
            # Передняя дуга крупнее дальней; наведение сохраняет центр кольца.
            bounds_center = (centers.min(axis=0) + centers.max(axis=0)) / 2
            np.testing.assert_allclose(bounds_center, (width / 2, height / 2), atol=1)

    def test_inspect_centers_and_tightly_frames_the_rendered_hull(self):
        scene = SpaceScene()
        for index in (0, 1, 3, 4, 9, 10, 13, 15, 17, DODO_ID):
            for width, height in ((1360, 820), (1600, 900), (600, 1000)):
                scene.select(index)
                obj = scene.selected_object
                mesh = build_mesh(index)
                # Проверяется именно геометрия, которую получает OpenGL,
                # вместе с её центрированием, надстройками и стволами.
                homogeneous = np.column_stack((mesh[:, :3], np.ones(len(mesh))))
                clip = (scene.view_projection(width / height) @ obj.matrix @ homogeneous.T).T
                ndc = clip[:, :2] / clip[:, 3:]
                self.assertTrue(np.all(clip[:, 3] > 0))
                self.assertTrue(np.all(np.abs(ndc[:, 0]) < 0.87))
                self.assertTrue(np.all(np.abs(ndc[:, 1]) < 0.76))
                hull = ndc[np.isin(mesh[:, 8], (0, 5))]
                bounds_center = (hull.min(axis=0) + hull.max(axis=0)) / 2
                np.testing.assert_allclose(bounds_center, (0, 0), atol=0.001)
                fraction = np.ptp(hull, axis=0) / 2
                if width > height and index in (1, 9, 10, 13):
                    self.assertTrue(0.54 < fraction[0] < 0.58)
                if width > height and index in (0, DODO_ID):
                    self.assertTrue(0.66 < fraction[1] < 0.70)
                self.assertEqual(scene.pick(width / 2, height / 2, width, height), index)
                previous_distance = scene.distance
                scene.zoom(1.25)
                self.assertLess(scene.distance, previous_distance)

    def test_initial_inspect_has_no_other_hulls_in_the_frame(self):
        scene = SpaceScene()
        positions = []
        for obj in scene.objects:
            vertices = np.unique(build_mesh(obj.model_id)[:, :3], axis=0)
            homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
            positions.append(obj.matrix @ homogeneous.T)
        for aspect in (1360 / 820, 16 / 9, 0.6):
            for index in range(len(MODELS)):
                scene.select(index)
                vp = scene.view_projection(aspect)
                for other, points in enumerate(positions):
                    if other == index:
                        continue
                    clip = (vp @ points).T
                    outside_one_plane = any(np.all(sign * clip[:, axis] > clip[:, 3])
                                            for axis in range(3) for sign in (-1, 1))
                    self.assertTrue(outside_one_plane, (index, other, aspect))

    def test_flight_preserves_camera_and_only_camera_moves(self):
        scene = SpaceScene()
        positions = np.array([obj.position.copy() for obj in scene.objects])
        scene.select(9)
        start = scene.eye.copy()
        direction = scene.forward.copy()
        scene.update(0.1, (0, 0, 1))
        np.testing.assert_allclose(scene.eye, start)
        scene.fly()
        np.testing.assert_allclose(scene.eye, start)
        np.testing.assert_allclose(scene.forward, direction)
        scene.update(0.1, (0, 0, 1, 3))
        np.testing.assert_allclose(scene.eye, start + direction * scene.speed * 0.3)
        scene.drag(30, -20)
        self.assertFalse(np.allclose(scene.forward, direction))
        scene.zoom(0.25)
        np.testing.assert_allclose(np.array([obj.position for obj in scene.objects]), positions)

    def test_selection_wraps_and_overview_can_be_restored(self):
        scene = SpaceScene()
        scene.select(-1)
        self.assertEqual(scene.selected, len(MODELS) - 1)
        scene.select(len(MODELS) + 2, focus=False)
        self.assertEqual(scene.selected, 2)
        scene.drag(10000, 10000)
        scene.overview()
        self.assertEqual(scene.camera_mode, "overview")
        self.assertEqual(scene.selected, 2)
        self.assertIsNone(scene.pick(-1, 20, 1200, 750))
        self.assertIsNone(scene.project(scene.eye - scene.forward, 1200, 750))
        self.assertIsNone(scene.project((0, 0, 0), 0, 0))


if __name__ == "__main__":
    unittest.main()
