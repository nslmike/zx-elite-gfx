"""Наклон карты проверяется без окна и GPU, по известным полям высоты."""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    from elite_lab import space_look
except ImportError:
    np = space_look = None


@unittest.skipIf(space_look is None, "Нужны зависимости requirements-space.txt")
class SurfaceSlopeTests(unittest.TestCase):
    def slope(self, height):
        size = len(height)
        pixels = np.zeros((size, size, 4), dtype="u1")
        pixels[:, :, 1] = height
        with patch.object(space_look, "_SURFACE_SIZE", size):
            data = space_look._surface_slope_bytes(pixels.tobytes())
        return np.frombuffer(data, dtype="f2").reshape(size, size, 2)

    def test_constant_height_has_zero_slope(self):
        slope = self.slope(np.full((4, 4), 128, dtype="u1"))
        np.testing.assert_array_equal(slope, 0)

    def test_signed_uv_slope_and_periodic_boundary(self):
        # Рост на 64/255 за 1/4 UV. На повторяющейся границе направление
        # меняется: край должен видеть соседний край, не clamp к самому себе.
        height = np.tile(np.array((0, 64, 128, 192), dtype="u1"), (4, 1))
        slope = self.slope(height)
        expected = np.tile(np.array((-256, 256, 256, -256)) / 255.0, (4, 1))
        np.testing.assert_allclose(slope[:, :, 0], expected, rtol=0.001)
        np.testing.assert_array_equal(slope[:, :, 1], 0)
        np.testing.assert_array_equal(slope[:, :, 0].sum(axis=1), 0)

    def test_quarter_turn_rotates_signed_gradient_axes(self):
        u, v = np.meshgrid(np.arange(4), np.arange(4))
        height = (u * 16 + v * 32).astype("u1")
        slope = self.slope(height)
        turned = self.slope(np.rot90(height))
        np.testing.assert_array_equal(turned[:, :, 0], np.rot90(slope[:, :, 1]))
        np.testing.assert_array_equal(turned[:, :, 1], -np.rot90(slope[:, :, 0]))

    def test_classic_bevel_is_wide_and_dent_gradients_are_periodic(self):
        # Два исходных texel шва становятся широкой настоящей фаской,
        # а не только усиливают старую однопиксельную производную.
        pixels = np.full((1024, 1024, 4), 160, dtype="u1")
        pixels[:, 512:514, 1] = 76
        original = pixels.copy()
        with patch.object(space_look, "_SURFACE_SIZE", 1024), patch.object(space_look, "_FINISH_SIZE", 512):
            bevel, dents = space_look._finish_heightfields(pixels.tobytes())
            data = space_look._finish_slope_bytes(pixels.tobytes())
        slope = np.frombuffer(data, dtype="f2").reshape(512, 512, 4).astype("f4")
        self.assertGreaterEqual(np.count_nonzero(bevel[100] < -.01), 13)
        self.assertLess(float(dents.min()), -.2)
        self.assertGreater(float(dents.max()), .02)  # Приподнятый валик удара.
        for offset, height in ((0, bevel), (2, dents)):
            expected_u = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 256
            expected_v = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 256
            np.testing.assert_allclose(slope[:, :, offset], expected_u, atol=.016, rtol=.001)
            np.testing.assert_allclose(slope[:, :, offset + 1], expected_v, atol=.016, rtol=.001)
        np.testing.assert_array_equal(pixels, original)

    def test_classic_profiles_never_replace_imported_or_authored_styles(self):
        def uniforms(model_id, **kwargs):
            look = space_look.SpaceLook.__new__(space_look.SpaceLook)
            look.hull = defaultdict(SimpleNamespace)
            texture = SimpleNamespace(use=lambda **_: None)
            for name in ("_surface", "_surface_slope", "_finish_slope", "_sky", "_stencils"):
                setattr(look, name, texture)
            look.configure_model(model_id, **kwargs)
            return {key: value.value for key, value in look.hull.items()}

        for model_id, style in enumerate(space_look._STYLES):
            native = uniforms(model_id)
            self.assertTrue(native["u_classic_finishes"])
            self.assertEqual(native["u_cobra_finishes"], model_id == 9)
            for source in ({"imported": True}, {"authored": True}, {"imported": True, "authored": True}):
                with self.subTest(model_id=model_id, source=source):
                    foreign = uniforms(model_id, **source)
                    self.assertFalse(foreign["u_classic_finishes"])
                    self.assertFalse(foreign["u_cobra_finishes"])
                    self.assertEqual(foreign["u_roughness"], style.roughness)
                    self.assertEqual(foreign["u_wear"], style.wear)
                    self.assertEqual(foreign["u_metal_color"], space_look._linear(style.metal))

        # Грузовые манёвры обтирают вдоль корпуса; боевые повреждения пиратов
        # сильнее дробят края и мнут листы, при ином силовом наборе.
        transport, pirate = (uniforms(model_id) for model_id in (10, 11))
        self.assertNotEqual(transport["u_finish_layout"], pirate["u_finish_layout"])
        self.assertGreater(transport["u_damage_profile"][1], pirate["u_damage_profile"][1])
        self.assertGreater(pirate["u_damage_profile"][0], transport["u_damage_profile"][0])
        self.assertGreater(pirate["u_damage_profile"][2], transport["u_damage_profile"][2])
        self.assertLess(uniforms(8)["u_damage_profile"][2], transport["u_damage_profile"][2] * .1)


if __name__ == "__main__":
    unittest.main()
