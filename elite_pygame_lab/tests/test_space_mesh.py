"""Проверки поверхности OpenGL-атласа; numpy — отдельная зависимость атласа."""

from collections import Counter
from itertools import combinations
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Для 3D-атласа нужен numpy из requirements-space.txt")

from elite_lab.model import COBRA_IDS, DODO_ID, KRAIT_ID, MODELS
from elite_lab.space_mesh import (
    _Mesh, _cycles, _facet_triangles, _hull_polygons,
    _signed_volume, build_mesh,
)


def _points(model_id):
    return np.array([(v.x, v.y, v.z) for v in MODELS[model_id].vertices], dtype=float)


def _edges(triangles):
    counts, directions = Counter(), Counter()
    for triangle in triangles:
        points = [tuple(round(float(x), 6) for x in p) for p in triangle]
        for a, b in zip(points, points[1:] + points[:1]):
            key = tuple(sorted((a, b)))
            counts[key] += 1
            directions[key] += 1 if a < b else -1
    return counts, directions


def _ray_hits(mesh, origin, direction):
    """Пересечения луча с треугольниками (Möller–Trumbore)."""
    hits = []
    for triangle in mesh[:, :3].reshape(-1, 3, 3):
        a, b, c = triangle
        e1, e2 = b - a, c - a
        h = np.cross(direction, e2)
        det = e1 @ h
        if abs(det) < 1e-9:
            continue
        s = origin - a
        u = s @ h / det
        q = np.cross(s, e1)
        v = direction @ q / det
        t = e2 @ q / det
        if u >= -1e-7 and v >= -1e-7 and u + v <= 1 + 1e-7 and t >= 0:
            hits.append(origin + t * direction)
    return hits


def _segment_crosses_triangle(p, q, triangle):
    a, b, c = triangle
    e1, e2, direction = b - a, c - a, q - p
    h = np.cross(direction, e2)
    det = e1 @ h
    if abs(det) < 1e-8:
        return False
    s = p - a
    u = s @ h / det
    cross = np.cross(s, e1)
    v = direction @ cross / det
    t = e2 @ cross / det
    return u > 1e-7 and v > 1e-7 and u + v < 1 - 1e-7 and 1e-7 < t < 1 - 1e-7


class SpaceMeshTest(unittest.TestCase):
    def test_all_twenty_models_match_vertex_contract(self):
        self.assertEqual(len(MODELS), 20)
        for model_id in range(len(MODELS)):
            with self.subTest(model_id=model_id):
                data = build_mesh(model_id)
                self.assertEqual(data.dtype, np.float32)
                self.assertEqual(data.shape[1], 9)
                self.assertEqual(len(data) % 3, 0)
                self.assertGreaterEqual(len(data), 12)
                self.assertTrue(data.flags.c_contiguous)
                self.assertTrue(np.isfinite(data).all())
                np.testing.assert_allclose(np.linalg.norm(data[:, 3:6], axis=1), 1, atol=1e-6)
                self.assertAlmostEqual(np.linalg.norm(data[:, :3], axis=1).max(), 1, places=6)
                np.testing.assert_allclose(data[:, :3].min(axis=0) + data[:, :3].max(axis=0),
                                           0, atol=2e-7)
                self.assertTrue(np.isin(data[:, 8], np.arange(7)).all())
                self.assertGreater(data[:, 6:8].min(), -.1)
                self.assertLess(data[:, 6:8].max(), 1.1)

    def test_complete_surfaces_are_closed_and_consistently_wound(self):
        # Проверяются также доки, панели, кили и тонкие стволы: у каждой
        # геометрической кромки ровно два противоположных употребления.
        for model_id in range(len(MODELS)):
            with self.subTest(model_id=model_id):
                data = build_mesh(model_id)
                triangles = data[:, :3].reshape(-1, 3, 3)
                counts, directions = _edges(triangles)
                self.assertEqual(set(counts.values()), {2})
                self.assertEqual(set(directions.values()), {0})
                cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                area2 = np.linalg.norm(cross, axis=1)
                self.assertGreater(area2.min(), 1e-7)
                np.testing.assert_allclose(cross / area2[:, None], data[::3, 3:6], atol=3e-5)
                volume = np.einsum("ij,ij->i", triangles[:, 0],
                                   np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
                self.assertGreater(volume, 0)

    def test_hulls_are_closed_spheres_with_original_vertices(self):
        for model_id in range(len(MODELS)):
            with self.subTest(model_id=model_id):
                points = _points(model_id)
                polygons = _hull_polygons(model_id)
                counts = Counter(tuple(sorted((a, b))) for _, loop in polygons
                                 for a, b in zip(loop, loop[1:] + loop[:1]))
                vertices = {v for _, loop in polygons for v in loop}
                self.assertEqual(set(counts.values()), {2})
                self.assertEqual(len(vertices) - len(counts) + len(polygons), 2)
                self.assertGreater(_signed_volume(points, polygons), 0)
                for _, loop in polygons:
                    triangles = _facet_triangles(MODELS[model_id], loop, points)
                    self.assertEqual(len(triangles), len(loop) - 2)
                    for tri in triangles:
                        self.assertEqual(len(set(tri)), 3)
                        self.assertGreater(np.linalg.norm(np.cross(points[tri[1]] - points[tri[0]],
                                                                   points[tri[2]] - points[tri[0]])), 1e-6)

    def test_antennae_guns_and_fins_do_not_become_hull_facets(self):
        for model_id, expected in ((1, set(range(9))), (9, set(range(12))),
                                   (12, set(range(12))), (13, set(range(13))),
                                   (KRAIT_ID, set(range(5)))):
            with self.subTest(model_id=model_id):
                actual = {v for _, loop in _hull_polygons(model_id) for v in loop}
                self.assertEqual(actual, expected)
        krait = build_mesh(KRAIT_ID)
        antenna = krait[krait[:, 8] == 4, :3]
        self.assertGreater(len(antenna), 60)
        self.assertLess(antenna[:, 0].min(), -.6)
        self.assertGreater(antenna[:, 0].max(), .6)

    def test_main_hulls_have_no_crossing_triangles(self):
        # В том числе невыпуклый сгиб крыла Cobra: замкнутость графа сама
        # по себе ещё не исключает прохождение одной фасеты сквозь другую.
        for model_id, model in enumerate(MODELS):
            with self.subTest(model_id=model_id):
                points = _points(model_id)
                triangles = [tri for _, loop in _hull_polygons(model_id)
                             for tri in _facet_triangles(model, loop, points)]
                for a, b in combinations(triangles, 2):
                    if len(set(a) & set(b)) >= 2:
                        continue
                    first, second = points[list(a)], points[list(b)]
                    for one, other in ((first, second), (second, first)):
                        for p, q in zip(one, np.roll(one, -1, axis=0)):
                            self.assertFalse(_segment_crosses_triangle(p, q, other), (a, b))

    def test_python_uses_blueprint_diagonals(self):
        points = _points(10)
        hull = dict(_hull_polygons(10))
        for face, diagonal in ((8, (3, 5)), (9, (2, 5)), (10, (2, 10)), (11, (3, 10))):
            with self.subTest(face=face):
                triangles = _facet_triangles(MODELS[10], hull[face], points)
                self.assertEqual(len(triangles), 2)
                self.assertTrue(all(set(diagonal) <= set(tri) for tri in triangles))

    def test_station_doors_are_real_recesses(self):
        for model_id, front, depth, half_width in ((0, 160, 34, 10), (DODO_ID, 196, 42, 16)):
            with self.subTest(model_id=model_id):
                data = build_mesh(model_id)
                source = _points(model_id)
                # Окна немного выступают над исходной передней гранью;
                # итоговый bbox и нормировка учитывают их толщину.
                scale = data[:, 0].max() / source[:, 0].max()
                z_offset = data[:, 2].min() - source[:, 2].min() * scale
                hits = _ray_hits(data, np.array((0., 0., 2.)), np.array((0., 0., -1.)))
                self.assertTrue(hits)
                self.assertAlmostEqual(max(p[2] for p in hits), (front - depth) * scale + z_offset, places=6)
                hits = _ray_hits(data, np.array((half_width * 2 * scale, 0., 2.)),
                                 np.array((0., 0., -1.)))
                self.assertAlmostEqual(max(p[2] for p in hits), front * scale + z_offset, places=6)

    def test_glazed_canopies_have_profile_and_stations_have_window_rows(self):
        for model_id in (*COBRA_IDS, 8, 13, 16):
            with self.subTest(model_id=model_id):
                data = build_mesh(model_id)
                normals = data[data[:, 8] == 2, 3:6]
                self.assertGreaterEqual(len(normals), 12)
                # Два настоящих стекла со скатом по обе стороны переплёта.
                self.assertGreater(np.linalg.norm(normals - normals[0], axis=1).max(), .04)
        for model_id in (0, DODO_ID):
            data = build_mesh(model_id)
            self.assertGreaterEqual(np.count_nonzero(data[:, 8] == 6), 72)

    def test_original_details_have_separate_materials(self):
        required = {0: {1, 4, 6}, 1: {3, 4}, 8: {2, 3}, 9: {2, 3}, 13: {2, 3},
                    14: {2, 4}, 15: {6}, 16: {2, 3}, 17: {2, 3, 4}, 19: {1, 4, 6}}
        for model_id, expected in required.items():
            with self.subTest(model_id=model_id):
                self.assertTrue(expected <= set(build_mesh(model_id)[:, 8]))
        for model_id in (3, 5, 6):
            self.assertEqual(set(build_mesh(model_id)[:, 8]), {5})
        self.assertIn(5, set(build_mesh(7)[:, 8]))

    def test_cobra_upper_and_lower_hull_uv_join_at_uniform_scale(self):
        # Решётка, пересекающая машинную грань, должна продолжаться без
        # скачка карты. Проверяем обе стороны всех стыков, а не один ракурс.
        for model_id in COBRA_IDS:
            data = build_mesh(model_id)
            face_rows, offset = {}, 0
            for face, loop in _hull_polygons(model_id):
                count = 3 * (len(loop) - 2)
                face_rows[face] = data[offset:offset + count]
                offset += count
            for faces, direction in ((range(9), 1), (range(10, 13), -1)):
                with self.subTest(model_id=model_id, upper=direction > 0):
                    rows = np.concatenate([face_rows[face] for face in faces])
                    uv_by_position = {}
                    for row in rows:
                        position = tuple(row[:3])
                        if position in uv_by_position:
                            np.testing.assert_array_equal(row[6:8], uv_by_position[position])
                        uv_by_position[position] = row[6:8]
                    # Вдоль X и Z одинаковый масштаб; верх и низ сохраняют
                    # исходную ориентацию, без растяжения отдельных фасетов.
                    projected = np.column_stack((rows[:, 0], rows[:, 2], np.ones(len(rows))))
                    chart = np.linalg.lstsq(projected, rows[:, 6:8], rcond=None)[0]
                    np.testing.assert_allclose(chart[[0, 1], [1, 0]], 0, atol=1e-6)
                    self.assertGreater(chart[0, 0] * direction, 0)
                    self.assertLess(chart[1, 1], 0)
                    self.assertAlmostEqual(abs(chart[0, 0]), abs(chart[1, 1]), places=6)
                    np.testing.assert_allclose(projected @ chart, rows[:, 6:8], atol=1e-6)

            # Кормовая стенка имеет собственную развёртку XY: общая X/Z
            # проекция обшивки не должна схлопнуть её в одномерный отрезок.
            stern = face_rows[9]
            self.assertEqual(np.linalg.matrix_rank(stern[:, 6:8] - stern[0, 6:8]), 2)

            # Исправление сохраняет фазу знакомой решётки: передняя пятая
            # часть правого стыка 0–6 остаётся на старой карте грани 2.
            loop = dict(_hull_polygons(model_id))[2]
            reference = _Mesh(_points(model_id))
            reference.triangle(_points(model_id)[list(loop)])
            old_uv = np.asarray(reference.rows)[:, 6:8]
            i, j = loop.index(0), loop.index(6)
            old_anchor = (4 * old_uv[i] + old_uv[j]) / 5
            new_anchor = (4 * face_rows[2][i, 6:8] + face_rows[2][j, 6:8]) / 5
            np.testing.assert_allclose(new_anchor, old_anchor, atol=1e-7)

    def test_each_engine_aperture_uses_its_own_unit_uv_rectangle(self):
        for model_id in (1, 8, 9, 10, 12, 13, 16, 17):
            with self.subTest(model_id=model_id):
                data = build_mesh(model_id)
                triangles = data[data[:, 8] == 3].reshape(-1, 3, 9)
                remaining = set(range(len(triangles)))
                apertures = []
                while remaining:
                    component = {remaining.pop()}
                    positions = {tuple(p) for p in triangles[next(iter(component)), :, :3]}
                    while True:
                        touching = {i for i in remaining if any(tuple(p) in positions
                                                              for p in triangles[i, :, :3])}
                        if not touching:
                            break
                        component.update(touching)
                        remaining.difference_update(touching)
                        positions.update(tuple(p) for i in touching for p in triangles[i, :, :3])
                    apertures.append(np.concatenate([triangles[i] for i in component]))
                self.assertTrue(apertures)
                for aperture in apertures:
                    np.testing.assert_allclose(aperture[:, 6:8].min(axis=0), 0, atol=1e-6)
                    np.testing.assert_allclose(aperture[:, 6:8].max(axis=0), 1, atol=1e-6)
                    uv_by_position = {}
                    for row in aperture:
                        position = tuple(row[:3])
                        if position in uv_by_position:
                            np.testing.assert_array_equal(row[6:8], uv_by_position[position])
                        uv_by_position[position] = row[6:8]

    def test_ring_keeps_hole_area_and_uv_plane(self):
        for sides in (3, 4, 5, 8):
            with self.subTest(sides=sides):
                angle = np.arange(sides) * 2 * np.pi / sides + .17
                outer = np.column_stack((4 * np.cos(angle), 4 * np.sin(angle), np.zeros(sides)))
                inner = np.array(((-.3, -.8, 0), (.3, -.8, 0), (.3, .8, 0), (-.3, .8, 0)))
                mesh = _Mesh(outer)
                mesh.ring(outer, inner, np.array((0., 0., 1.)))
                data = np.array(mesh.rows)
                triangles = data[:, :3].reshape(-1, 3, 3)
                area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                               triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
                expected = sides * 16 * np.sin(2 * np.pi / sides) / 2 - .6 * 1.6
                self.assertAlmostEqual(area, expected, places=7)
                uv_by_position = {}
                for row in data:
                    key = tuple(row[:3])
                    if key in uv_by_position:
                        np.testing.assert_array_equal(row[6:8], uv_by_position[key])
                    uv_by_position[key] = row[6:8]

    def test_rejects_ambiguous_contours_and_bad_model_ids(self):
        with self.assertRaises(ValueError):
            _cycles(((0, 1), (1, 2), (2, 0), (0, 3), (3, 2)))
        for value in (-1, 20, 1.5, "9", True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_mesh(value)

    def test_cached_buffers_are_independent_and_variants_share_geometry(self):
        for a, b in ((10, 11), COBRA_IDS):
            np.testing.assert_array_equal(build_mesh(a), build_mesh(b))
        original = build_mesh(0)
        changed = build_mesh(0)
        changed[:] = 0
        np.testing.assert_array_equal(build_mesh(0), original)


if __name__ == "__main__":
    unittest.main()
