"""Проверки GGX-окружения без окна и OpenGL."""

import unittest

try:
    import numpy as np
    from elite_lab import space_reflections as reflections
except ImportError:
    np = reflections = None


@unittest.skipIf(reflections is None, "Нужны зависимости requirements-space.txt")
class SpaceReflectionsTests(unittest.TestCase):
    def test_cubemap_projection_preserves_face_order(self):
        faces = np.arange(18, dtype="f4").reshape(6, 3) / 17
        images = [np.broadcast_to(color, (4, 4, 3)) for color in faces]
        directions = np.vstack((np.eye(3), -np.eye(3))).astype("f4")
        sampled = reflections._sky_to_latlong(images, directions)
        np.testing.assert_allclose(sampled, faces[[0, 2, 4, 1, 3, 5]])

    def test_prefilter_preserves_constant_radiance(self):
        radiance = np.array((.031, .36, 2.5), dtype="f4")
        source = np.broadcast_to(radiance, (32, 64, 3))
        for roughness in (.2, .6, 1.0):
            filtered = reflections._prefilter(source, 16, 8, roughness)
            np.testing.assert_allclose(filtered, np.broadcast_to(radiance, filtered.shape),
                                       rtol=1e-6)

    def test_roughness_spreads_a_directional_reflection(self):
        directions, _ = reflections._grid(128, 64)
        spot = np.exp((directions[..., 0:1] - 1) * 95)
        source = np.repeat(spot, 3, axis=-1)
        smooth = reflections._prefilter(source, 64, 32, .2)
        rough = reflections._prefilter(source, 64, 32, .8)
        self.assertLess(float(rough.max()), float(smooth.max()) * .2)
        # Широкий блик виден и там, где гладкий материал не ловит источник.
        self.assertGreater(float(rough[16, 37, 0]), float(smooth[16, 37, 0]))

    def test_poles_have_no_longitudinal_seam_at_any_level(self):
        faces = (np.zeros((4, 4, 3), dtype="f2").tobytes(),) * 6
        levels = reflections.reflection_levels(4, faces)
        width, height = reflections.REFLECTION_SIZE
        for level, data in enumerate(levels):
            rgba = np.frombuffer(data, dtype="f2").reshape(height >> level, width >> level, 4)
            rows = rgba[[0, -1]]
            np.testing.assert_array_equal(rows, np.broadcast_to(rows[:, :1], rows.shape))

    def test_dfg_bounds_and_fresnel_response(self):
        size = reflections.DFG_SIZE
        data = np.frombuffer(reflections.dfg_lut(), dtype="f2").astype("f4")
        dfg = data.reshape(size, size, 2)
        self.assertTrue(np.isfinite(dfg).all())
        self.assertGreaterEqual(float(dfg.min()), 0)
        self.assertLessEqual(float(dfg.sum(axis=-1).max()), 1.001)
        smooth, rough = int(.30 * size), int(.80 * size)
        self.assertGreater(dfg[smooth, 0, 1], dfg[smooth, -1, 1])
        self.assertGreater(dfg[smooth, -1, 0], dfg[rough, -1, 0])


if __name__ == "__main__":
    unittest.main()
