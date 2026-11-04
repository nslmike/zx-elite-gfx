"""Игровой ZX bitmap и его связь с общей многокорабельной камерой."""

from types import SimpleNamespace
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Для космического режима нужен numpy из requirements-space.txt")

from elite_lab.model import DODO_ID, MODELS
from elite_lab.space_mesh import build_mesh, source_transform
from elite_lab.space_zx import SpaceZXRenderer, blueprint, fit_viewport, viewport_projection


def _projection(aspect):
    return np.array(((1 / aspect, 0, 0, 0), (0, 1, 0, 0),
                     (0, 0, -1, -.12), (0, 0, -1, 0)), dtype=float)


def _object(model_id, position, radius=1., rotation=None):
    matrix = np.eye(4)
    matrix[:3, :3] = (np.eye(3) if rotation is None else rotation) * radius
    matrix[:3, 3] = position
    return SimpleNamespace(model_id=model_id, position=np.array(position, dtype=float),
                           radius=radius, matrix=matrix)


def _scene(*objects):
    return SimpleNamespace(ships=tuple(SimpleNamespace(model_id=obj.model_id) for obj in objects),
                           objects=objects, eye=np.zeros(3), view_projection=_projection)


class SpaceZXTest(unittest.TestCase):
    def renderer(self, stars=()):
        renderer = SpaceZXRenderer()
        renderer.stars = np.asarray(stars, dtype=float).reshape(-1, 3)
        return renderer

    def test_square_pixels_and_same_window_projection_after_source_switch(self):
        point = np.array((.27, -.19, -4., 1.))
        for size in ((1360, 820), (1280, 720), (820, 820), (820, 1024), (1361, 821)):
            with self.subTest(size=size):
                viewport = fit_viewport(size)
                self.assertEqual(viewport[2], 2 * viewport[3])
                self.assertGreaterEqual(viewport[0], 0)
                self.assertGreaterEqual(viewport[1], 0)
                projection = _projection(size[0] / size[1])
                adjusted = viewport_projection(projection, size, viewport)
                old, new = projection @ point, adjusted @ point
                old = old[:2] / old[3] * (np.array(size) / 2) + np.array(size) / 2
                new = new[:2] / new[3] * (np.array(viewport[2:]) / 2)
                new += np.array(viewport[:2]) + np.array(viewport[2:]) / 2
                # Нечётные размеры могут оставлять лишний пиксель у края.
                np.testing.assert_allclose(new, old, atol=.5)
                self.assertAlmostEqual(np.linalg.norm(adjusted[0, :3]) * 128,
                                       np.linalg.norm(adjusted[1, :3]) * 64, places=10)

    def test_original_points_use_exact_mesh_normalization_including_dodo(self):
        for model_id, model in enumerate(MODELS):
            with self.subTest(model_id=model_id):
                data = blueprint(model_id)
                center, radius = source_transform(model_id)
                raw = np.array([(v.x, v.y, v.z) for v in model.vertices])
                np.testing.assert_array_equal(data.points, (raw - center) / radius)
                self.assertEqual(len(data.points), len(model.vertices))
                self.assertFalse(data.points.flags.writeable)
                center[:] = 12345
                self.assertFalse(np.array_equal(center, source_transform(model_id)[0]))
                # Нормализация не меняет оси носа/дока и высоты.
                reconstructed = data.points * radius + data.center
                np.testing.assert_allclose(reconstructed, raw, atol=1e-12)
        self.assertIn("Legend", MODELS[DODO_ID].name)
        # Кеш метаданных не меняет контракт независимой копии build_mesh.
        original = build_mesh(9)
        modified = build_mesh(9)
        modified[:] = 0
        np.testing.assert_array_equal(build_mesh(9), original)

    def test_pixels_are_monochrome_native_and_reproducible_for_all_models(self):
        renderer = self.renderer()
        for model_id in range(len(MODELS)):
            with self.subTest(model_id=model_id):
                scene = _scene(_object(model_id, (.4, .2, -4.)))
                first, second = renderer.frame(scene, (1360, 820)), renderer.frame(scene, (1360, 820))
                self.assertEqual(first.pixels.shape, (128, 256))
                self.assertEqual(first.pixels.dtype, np.uint8)
                self.assertEqual(len(first.bitmap), 4096)
                self.assertTrue(np.isin(first.pixels, (0, 255)).all())
                self.assertGreater(np.count_nonzero(first.pixels), 0)
                self.assertEqual(first.bitmap, second.bitmap)
                self.assertEqual(first.visible_ships, (0,))
                self.assertLessEqual(first.edge_count, len(MODELS[model_id].edges))

    def test_star_directions_preserve_x_and_y_without_mirroring(self):
        renderer = self.renderer(((1, 0, -2), (0, 1, -2), (0, 0, 1)))
        frame = renderer.frame(_scene(), (1024, 512))
        self.assertEqual(set(map(tuple, np.argwhere(frame.pixels))), {(64, 160), (32, 128)})

    def test_distance_removes_blueprint_details_before_the_ship_disappears(self):
        renderer = self.renderer()
        near = renderer.frame(_scene(_object(9, (0, 0, -3))), (1024, 512))
        far = renderer.frame(_scene(_object(9, (0, 0, -40))), (1024, 512))
        # Дальностные поля таблицы убирают детали Cobra, оставляя силуэт.
        self.assertGreater(near.edge_count, far.edge_count)
        self.assertGreater(far.edge_count, 0)
        self.assertTrue(far.pixels.any())

    def test_behind_and_offscreen_objects_do_not_wrap_into_bitmap(self):
        renderer = self.renderer()
        scene = _scene(_object(9, (0, 0, 4)), _object(0, (1000, 0, -4)))
        frame = renderer.frame(scene, (1024, 512))
        self.assertEqual(frame.visible_ships, ())
        self.assertEqual(frame.edge_count, 0)
        self.assertFalse(frame.pixels.any())

    def test_near_plane_and_extreme_off_axis_edges_stay_in_byte_range(self):
        renderer = self.renderer()
        # Корпус пересекает камеру и границы кадра: нет деления на ноль,
        # зеркального корпуса позади и переполнения входного байта $B044.
        scene = _scene(_object(9, (.35, .15, -.25)))
        frame = renderer.frame(scene, (1024, 512))
        self.assertEqual(frame.pixels.shape, (128, 256))
        self.assertTrue(np.isin(frame.pixels, (0, 255)).all())

    def test_near_ship_hides_background_edges_and_stars_but_keeps_own_edges(self):
        renderer = self.renderer()
        near = _object(0, (0, 0, -4), radius=2)
        distant = _object(9, (0, 0, -8), radius=.7)
        baseline = renderer.frame(_scene(near), (1024, 512))
        combined = renderer.frame(_scene(near, distant), (1024, 512))
        self.assertEqual(combined.bitmap, baseline.bitmap)
        self.assertGreater(baseline.edge_count, 0)
        renderer.stars = np.array(((0, 0, -1),))
        star = renderer.frame(_scene(near), (1024, 512))
        self.assertEqual(star.bitmap, baseline.bitmap)

    def test_rotated_up_axis_rolls_the_entire_scene_including_stars(self):
        renderer = self.renderer(((1, 0, -2),))
        scene = _scene()
        normal = renderer.frame(scene, (1024, 512))
        projection = _projection(2)
        projection[[0, 1]] = projection[[1, 0]] * np.array(((-.5,), (2.,)))
        scene.view_projection = lambda aspect: projection
        rolled = renderer.frame(scene, (1024, 512))
        self.assertNotEqual(normal.bitmap, rolled.bitmap)
        self.assertEqual(set(map(tuple, np.argwhere(rolled.pixels))), {(32, 128)})


if __name__ == "__main__":
    unittest.main()
