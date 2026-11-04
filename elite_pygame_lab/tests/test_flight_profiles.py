"""Обе схемы полёта используют одну полную позу камеры."""

import unittest
from math import pi

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_flyby import FlybyScene
    from elite_lab.space_scene import SpaceScene


@unittest.skipIf(np is None, "Для сцены OpenGL нужен numpy из requirements-space.txt")
class FlightProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = AssetCatalog()

    def scene(self, scene_type=SpaceScene, *, level=False):
        options = {"source_interval": 0} if scene_type is FlybyScene else {}
        scene = scene_type(catalog=self.catalog, **options)
        if level:
            if scene.is_flyby:
                scene._camera_target = scene.eye + (0, 0, -1)
                scene._camera_up = np.array((0, 1, 0))
            else:
                scene._yaw = scene._pitch = 0.0
        scene.fly()
        return scene

    def assert_basis(self, scene):
        basis = np.array(scene._view_basis())
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(basis[0], basis[2]), basis[1], atol=1e-12)

    def test_default_profile_and_modern_motion_keep_world_axes_and_boost(self):
        for scene_type in (SpaceScene, FlybyScene):
            with self.subTest(scene=scene_type.__name__):
                scene = self.scene(scene_type, level=True)
                self.assertEqual(scene.flight_profile, "modern")
                scene.turn_flight(.4, .3, 1.1)
                eye, forward = scene.eye, scene.forward
                horizontal_right = np.cross(forward, (0, 1, 0))
                horizontal_right /= np.linalg.norm(horizontal_right)
                scene.update(.1, (1, 0, 0, 3))
                np.testing.assert_allclose(scene.eye, eye + horizontal_right * scene.speed * .3)
                eye = scene.eye
                scene.update(.1, (0, 1, 0))
                np.testing.assert_allclose(scene.eye, eye + (0, scene.speed * .1, 0))
                eye = scene.eye
                scene.update(.1, (0, 0, 1))
                np.testing.assert_allclose(scene.eye, eye + forward * scene.speed * .1)
                eye = scene.eye
                scene.update(.1)
                np.testing.assert_array_equal(scene.eye, eye)

    def test_positive_roll_banks_right_and_rotates_the_image_about_the_nose(self):
        for profile in ("modern", "elite"):
            with self.subTest(profile=profile):
                scene = self.scene(level=True)
                if profile == "elite":
                    scene.cycle_flight_profile()
                eye, forward = scene.eye, scene.forward
                scene.turn_flight(0, 0, pi / 2)
                right, up, _ = scene._view_basis()
                np.testing.assert_allclose(right, (0, -1, 0), atol=1e-12)
                np.testing.assert_allclose(up, (1, 0, 0), atol=1e-12)
                np.testing.assert_array_equal(scene.eye, eye)
                np.testing.assert_allclose(scene.forward, forward, atol=1e-12)
                # Неподвижная точка справа от носа после правого крена
                # оказывается над центром кадра: повернулась вся камера.
                x, y = scene.project(eye + forward * 10 + (1, 0, 0), 800, 600)
                self.assertAlmostEqual(x, 400)
                self.assertLess(y, 300)
                self.assert_basis(scene)

    def test_modern_world_yaw_and_pitch_keep_the_existing_roll(self):
        scene = self.scene(level=True)
        scene.turn_flight(0, .3, .9)
        before = np.array(scene._view_basis())
        scene.turn_flight(pi, 0, 0)
        np.testing.assert_allclose(scene._view_basis(), before * (-1, 1, -1), atol=1e-12)
        previous_up = scene._view_basis()[1]
        scene.turn_flight(0, .1, 0)
        self.assertGreater(scene.forward[1], before[2, 1])
        self.assertGreater(np.dot(scene._view_basis()[1], previous_up), .99)
        self.assert_basis(scene)

    def test_elite_local_pitch_passes_both_poles_and_completes_a_loop(self):
        for scene_type in (SpaceScene, FlybyScene):
            with self.subTest(scene=scene_type.__name__):
                scene = self.scene(scene_type, level=True)
                scene.cycle_flight_profile()
                eye, original = scene.eye, np.array(scene._view_basis())
                scene.turn_flight(0, pi / 2, 0)
                np.testing.assert_allclose(scene.forward, (0, 1, 0), atol=1e-12)
                scene.turn_flight(0, pi / 2, 0)
                np.testing.assert_allclose(scene.forward, (0, 0, 1), atol=1e-12)
                np.testing.assert_allclose(scene._view_basis()[1], (0, -1, 0), atol=1e-12)
                scene.turn_flight(0, pi / 2, 0)
                np.testing.assert_allclose(scene.forward, (0, -1, 0), atol=1e-12)
                scene.turn_flight(0, pi / 2, 0)
                np.testing.assert_allclose(scene._view_basis(), original, atol=1e-12)
                for _ in range(720):
                    scene.turn_flight(0, pi / 360, 0)
                np.testing.assert_allclose(scene._view_basis(), original, atol=1e-12)
                np.testing.assert_array_equal(scene.eye, eye)
                self.assert_basis(scene)

    def test_elite_roll_then_pitch_can_turn_the_nose_without_yaw(self):
        scene = self.scene(level=True)
        scene.cycle_flight_profile()
        scene.turn_flight(0, 0, pi / 2)
        scene.turn_flight(0, pi / 2, 0)
        np.testing.assert_allclose(scene.forward, (1, 0, 0), atol=1e-12)
        self.assert_basis(scene)

    def test_mouse_uses_modern_yaw_and_elite_roll_with_local_pitch(self):
        for scene_type in (SpaceScene, FlybyScene):
            for profile in ("modern", "elite"):
                with self.subTest(scene=scene_type.__name__, profile=profile):
                    scene = self.scene(scene_type, level=True)
                    if profile == "elite":
                        scene.cycle_flight_profile()
                    drag = scene.look_drag if scene.is_flyby else scene.drag
                    drag(100, 0)
                    if profile == "modern":
                        self.assertGreater(scene.forward[0], 0)
                    else:
                        np.testing.assert_allclose(scene.forward, (0, 0, -1), atol=1e-12)
                        self.assertLess(scene._view_basis()[0][1], 0)
                    drag(0, -100)
                    self.assertGreater(scene.forward[1], 0)
                    self.assert_basis(scene)

    def test_elite_throttle_coasts_brakes_and_never_reverses_or_strafes(self):
        for scene_type in (SpaceScene, FlybyScene):
            with self.subTest(scene=scene_type.__name__):
                scene = self.scene(scene_type, level=True)
                scene.cycle_flight_profile()
                eye, forward = scene.eye, scene.forward
                scene.update(.1, (1, 1, 0, 3))
                self.assertEqual(scene.flight_speed, 0)
                np.testing.assert_array_equal(scene.eye, eye)
                for _ in range(10):
                    scene.update(.1, (1, 1, 1, 3))
                self.assertAlmostEqual(scene.flight_speed, scene.flight_acceleration)
                np.testing.assert_allclose(scene.eye, eye + forward * scene.flight_acceleration / 2)
                eye, speed = scene.eye, scene.flight_speed
                scene.update(.1, (1, 1, 0))
                self.assertEqual(scene.flight_speed, speed)
                np.testing.assert_allclose(scene.eye, eye + forward * speed * .1)
                for _ in range(11):
                    scene.update(.1, (0, 0, -1))
                self.assertEqual(scene.flight_speed, 0)
                eye = scene.eye
                scene.update(.1, (0, 0, -1))
                np.testing.assert_array_equal(scene.eye, eye)

    def test_elite_speed_caps_and_integrates_partial_acceleration_frame(self):
        scene = self.scene(level=True)
        scene.cycle_flight_profile()
        for _ in range(40):
            scene.update(.1, (0, 0, 1))
        self.assertEqual(scene.flight_speed, scene.flight_max_speed)
        eye = scene.eye
        scene.update(.1)
        np.testing.assert_allclose(scene.eye, eye + scene.forward * scene.flight_max_speed * .1)
        # Скорость достигает ограничения в середине кадра, после чего
        # оставшаяся половина идёт равномерно, без зависимости от FPS.
        scene.flight_speed = scene.flight_max_speed - .5
        eye = scene.eye
        scene.update(.1, (0, 0, 1))
        distance = (scene.flight_max_speed - .25) * .05 + scene.flight_max_speed * .05
        np.testing.assert_allclose(scene.eye, eye + scene.forward * distance)

    def test_wheel_adjusts_elite_speed_without_dolly_or_reverse(self):
        scene = self.scene()
        scene.cycle_flight_profile()
        eye = scene.eye
        scene.zoom(.5)
        self.assertGreater(scene.flight_speed, 0)
        np.testing.assert_array_equal(scene.eye, eye)
        scene.zoom(1e6)
        self.assertEqual(scene.flight_speed, scene.flight_max_speed)
        scene.zoom(-1e6)
        self.assertEqual(scene.flight_speed, 0)
        np.testing.assert_array_equal(scene.eye, eye)
        scene.cycle_flight_profile()
        scene.zoom(.5)
        np.testing.assert_allclose(scene.eye, eye + scene.forward * scene.speed * .09)

    def test_profile_switches_preserve_complete_pose_including_inversion_and_poles(self):
        for scene_type in (SpaceScene, FlybyScene):
            for pitch in (0, pi / 2, -pi / 2, pi, .73):
                with self.subTest(scene=scene_type.__name__, pitch=pitch):
                    scene = self.scene(scene_type, level=True)
                    scene.cycle_flight_profile()
                    scene.turn_flight(.67, pitch, 1.2)
                    scene.update(.1, (0, 0, 1))
                    eye, target = scene.eye, scene.target
                    basis = np.array(scene._view_basis())
                    projection = scene.view_projection(1.6)
                    for expected in ("modern", "elite", "modern", "elite"):
                        self.assertEqual(scene.cycle_flight_profile(), expected)
                        self.assertEqual(scene.flight_speed, 0)
                        np.testing.assert_array_equal(scene.eye, eye)
                        np.testing.assert_array_equal(scene.target, target)
                        np.testing.assert_array_equal(scene._view_basis(), basis)
                        np.testing.assert_array_equal(scene.view_projection(1.6), projection)
                    scene.cycle_flight_profile()
                    scene.turn_flight(0, -.0001 if pitch == pi / 2 else .0001, 0)
                    self.assertLess(np.linalg.norm(np.array(scene._view_basis()) - basis), .001)
                    self.assert_basis(scene)

    def test_repeated_fly_preserves_pose_speed_and_profile_until_explicit_exit(self):
        for scene_type in (SpaceScene, FlybyScene):
            with self.subTest(scene=scene_type.__name__):
                scene = self.scene(scene_type, level=True)
                scene.cycle_flight_profile()
                scene.turn_flight(.4, 2.1, .8)
                scene.update(.1, (0, 0, 1))
                eye, basis, speed = scene.eye, np.array(scene._view_basis()), scene.flight_speed
                scene.fly()
                np.testing.assert_array_equal(scene.eye, eye)
                np.testing.assert_array_equal(scene._view_basis(), basis)
                self.assertEqual(scene.flight_speed, speed)
                scene.cycle_source()
                scene.cycle_textures()
                self.assertEqual(scene.flight_profile, "elite")
                self.assertEqual(scene.flight_speed, speed)
                if scene.is_flyby:
                    nearest = 17
                    scene._flight_eye = scene.objects[nearest].position.copy()
                    scene.toggle_flight()
                    self.assertEqual(scene.selected, nearest)
                else:
                    scene.overview()
                self.assertEqual(scene.flight_speed, 0)
                self.assertEqual(scene.flight_profile, "elite")
                scene.fly()
                self.assertEqual(scene.flight_speed, 0)
                self.assertEqual(scene.flight_profile, "elite")
                eye = scene.eye
                scene.update(.1)
                np.testing.assert_array_equal(scene.eye, eye)


if __name__ == "__main__":
    unittest.main()
