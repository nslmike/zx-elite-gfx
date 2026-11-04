"""Передача камеры пролёта свободному управлению без замены мира."""

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from elite_lab.model import COBRA_ID
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_flyby import FlybyScene
    from elite_lab.space_scene import SpaceScene


@unittest.skipIf(np is None, "Для сцены OpenGL нужен numpy из requirements-space.txt")
class FlybyFreeFlightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = AssetCatalog()

    def scene(self, *args, **kwargs):
        return FlybyScene(*args, catalog=self.catalog, **kwargs)

    def test_flight_entry_preserves_eye_projection_and_full_look_basis(self):
        for entry, turned in (("fly", False), ("fly", True), ("toggle_flight", False), ("toggle_flight", True)):
            with self.subTest(entry=entry, turned=turned):
                scene = self.scene(source_interval=0)
                scene.seek(107.5)
                if turned:
                    # Наклонённый базис и относительный ПКМ вместе дают крен,
                    # который нельзя сохранить одной парой yaw/pitch.
                    scene._camera_up = scene._view_basis()[0]
                    scene.look_drag(137, -63)
                eye, target = scene.eye, scene.target
                basis = np.array(scene._view_basis())
                projections = [scene.view_projection(aspect) for aspect in (.6, 1.6, 2.4)]
                ships, objects = scene.ships, scene.objects
                clocks = scene.elapsed, scene.animation_elapsed
                getattr(scene, entry)()
                self.assertEqual(scene.camera_mode, "flight")
                self.assertTrue(scene.is_flyby)
                self.assertFalse(SpaceScene.is_flyby)
                self.assertIs(scene.ships, ships)
                self.assertIs(scene.objects, objects)
                self.assertEqual((scene.elapsed, scene.animation_elapsed), clocks)
                np.testing.assert_array_equal(scene.eye, eye)
                np.testing.assert_allclose(scene.target, target, atol=1e-12)
                np.testing.assert_allclose(scene._view_basis(), basis, atol=1e-12)
                for aspect, expected in zip((.6, 1.6, 2.4), projections):
                    np.testing.assert_allclose(scene.view_projection(aspect), expected, atol=1e-12)
                scene.update(.1, (1, 1, 1))
                scene.look_drag(53, -29)
                eye, projection = scene.eye, scene.view_projection(1.6)
                scene.fly()
                np.testing.assert_array_equal(scene.eye, eye)
                np.testing.assert_array_equal(scene.view_projection(1.6), projection)

    def test_flight_reuses_atlas_steps_world_vertical_boost_drag_and_wheel(self):
        scene = self.scene(source_interval=0)
        scene.look_drag(53, -29)
        autopilot_look = scene._look_yaw, scene._look_pitch
        scene.toggle_flight()
        eye, forward = scene.eye, scene.forward
        right = np.cross(forward, (0, 1, 0))
        right /= np.linalg.norm(right)
        scene.update(.1, (1, 0, 0, 3))
        np.testing.assert_allclose(scene.eye, eye + right * scene.speed * .3)
        eye = scene.eye
        scene.update(.1, (0, 1, 0))
        np.testing.assert_allclose(scene.eye, eye + (0, scene.speed * .1, 0))
        eye = scene.eye
        scene.update(.1, (0, 0, 1))
        np.testing.assert_allclose(scene.eye, eye + forward * scene.speed * .1)
        scene.look_drag(93, -57)
        self.assertFalse(np.allclose(scene.forward, forward))
        np.testing.assert_allclose(scene._view_basis() @ np.array(scene._view_basis()).T,
                                   np.eye(3), atol=1e-12)
        self.assertEqual((scene._look_yaw, scene._look_pitch), autopilot_look)
        eye, forward = scene.eye, scene.forward
        scene.zoom(.5)
        np.testing.assert_allclose(scene.eye, eye + forward * scene.speed * .09)

    def test_flight_yaw_rotates_around_world_y_and_preserves_pitch(self):
        for rolled in (False, True):
            with self.subTest(rolled=rolled):
                scene = self.scene(source_interval=0)
                if rolled:
                    scene._camera_up = scene._view_basis()[0]
                    scene.look_drag(72, -16)
                scene.toggle_flight()
                eye, forward = scene.eye, scene.forward
                scene.look_drag(np.pi / .005, 0)
                np.testing.assert_array_equal(scene.eye, eye)
                np.testing.assert_allclose(scene.forward, forward * (-1, 1, -1), atol=1e-12)
                scene.look_drag(0, 10)
                self.assertLess(scene.forward[1], forward[1])

    def test_flight_entry_at_poles_keeps_view_and_exits_without_singularity(self):
        for pole in (-1, 1):
            with self.subTest(pole=pole):
                scene = self.scene(source_interval=0)
                scene._camera_target = scene.eye + (0, pole, 0)
                scene._camera_up = np.array((0, 0, 1))
                before = scene.view_projection(1.6)
                scene.toggle_flight()
                np.testing.assert_allclose(scene.view_projection(1.6), before, atol=1e-12)
                scene.look_drag(200, 0)
                self.assertAlmostEqual(scene.forward[1], pole)
                scene.look_drag(0, 20 * pole)
                self.assertLess(abs(scene.forward[1]), 1)
                self.assertTrue(np.isfinite(scene.view_projection(1.6)).all())
                scene.update(.1, (1, 1, 1))
                self.assertTrue(np.isfinite(scene.eye).all())

    def test_route_and_auto_sources_freeze_while_world_rotates(self):
        scene = self.scene()
        scene.seek(7.5)
        scene.toggle_flight()
        before = np.array([obj.matrix for obj in scene.objects])
        eye = scene.eye
        scene.update(31)
        self.assertEqual((scene.elapsed, scene.animation_elapsed, scene.model_source),
                         (7.5, 38.5, "zx"))
        np.testing.assert_array_equal(scene.eye, eye)
        self.assertFalse(np.allclose([obj.matrix for obj in scene.objects], before))
        self.assertEqual(scene.scrub(20), 0)
        self.assertEqual((scene.elapsed, scene.animation_elapsed), (7.5, 38.5))
        scene.cycle_source()
        scene.cycle_textures()
        self.assertEqual((scene.model_source, scene.texture_source), ("elite", "procedural"))
        scene.update(100)
        self.assertEqual(scene.model_source, "elite")

    def test_selection_uses_real_centers_and_guest_only_when_present(self):
        scene = self.scene(20, source_interval=0)
        self.assertTrue(scene.empty_slot)
        scene.toggle_flight()
        self.assertFalse(scene.empty_slot)
        self.assertLess(scene.selected, 20)
        guest_slot = scene._route.index(20)
        angle = guest_slot * 2 * np.pi / scene.slot_count
        scene._flight_eye = scene.ring_radius * np.array((np.sin(angle), 0, np.cos(angle)))
        scene.update(0)
        expected = min(scene.objects, key=lambda obj: np.linalg.norm(obj.position - scene.eye))
        self.assertEqual(scene.selected, expected.model_id)
        self.assertNotEqual(scene.selected, 20)
        scene.set_source("pioneer")
        self.assertEqual(scene.selected, 20)
        self.assertFalse(scene.empty_slot)
        scene.set_source("elite")
        self.assertLess(scene.selected, 20)
        self.assertFalse(scene.empty_slot)

    def test_return_uses_nearest_visit_with_start_model_and_adjacent_laps(self):
        # Времена посещения зависят от --model, а не от начала списка Cobra.
        for start, elapsed, nearest, expected in (
                (None, 125, COBRA_ID, 126),
                (None, 127, 8, 120),
                (None, 50, 0, 66),
                (17, 0, COBRA_ID, 78),
                (17, 250, 17, 252)):
            with self.subTest(start=start, elapsed=elapsed, nearest=nearest):
                scene = self.scene(start, source_interval=0)
                scene.seek(elapsed)
                scene.toggle_flight()
                scene._flight_eye = scene.objects[nearest].position + (0, .01, 0)
                scene.update(0)
                self.assertEqual(scene.selected, nearest)
                animation_elapsed = scene.animation_elapsed
                self.assertEqual(scene.toggle_flight(), "flyby")
                self.assertEqual((scene.elapsed, scene.selected), (expected, nearest))
                self.assertEqual(scene.animation_elapsed, animation_elapsed)

    def test_guest_return_preserves_poses_pause_look_source_and_interval_remainder(self):
        for manual in (False, True):
            with self.subTest(manual=manual):
                scene = self.scene(model_source="zx" if manual else "pioneer")
                scene.seek(5.5)
                scene.look_drag(104, -36)
                look = scene._look_yaw, scene._look_pitch
                scene.toggle_pause()
                scene.toggle_flight()
                if manual:
                    scene.set_source("pioneer")
                scene.look_drag(-217, 81)
                scene._flight_eye = scene.objects[20].position + (0, .1, 0)
                scene.update(20)
                scene.cycle_textures()
                poses = np.array([obj.matrix for obj in scene.objects])
                animation_elapsed = scene.animation_elapsed
                self.assertEqual(scene.toggle_flight(), "flyby")
                self.assertEqual((scene.elapsed, scene.selected, scene.model_source), (72, 20, "pioneer"))
                self.assertEqual(scene.texture_source, "procedural")
                self.assertTrue(scene.paused)
                self.assertEqual(scene.animation_elapsed, animation_elapsed)
                self.assertEqual((scene._look_yaw, scene._look_pitch), look)
                np.testing.assert_array_equal([obj.matrix for obj in scene.objects], poses)
                scene.update(0)
                self.assertEqual(scene.model_source, "pioneer")
                np.testing.assert_array_equal([obj.matrix for obj in scene.objects], poses)
                scene.toggle_pause()
                scene.update((8 if manual else 2.5) - .01)
                self.assertEqual(scene.model_source, "pioneer")
                scene.update(.01)
                self.assertEqual(scene.model_source, "mixed")

    def test_scrub_after_route_jump_reverses_through_negative_rotation_time(self):
        for paused_dt in (0, .25):
            with self.subTest(paused_dt=paused_dt):
                scene = self.scene(model_source="pioneer", source_interval=0)
                scene.seek(1)
                scene.toggle_flight()
                scene._flight_eye = scene.objects[20].position + (0, .1, 0)
                scene.toggle_flight()
                scene.toggle_pause()
                self.assertEqual((scene.elapsed, scene.animation_elapsed), (72, 1))
                scene.scrub(-10)
                self.assertEqual((scene.elapsed, scene.animation_elapsed), (62, -9))
                negative_pose = scene.objects[COBRA_ID].matrix.copy()
                scene.update(paused_dt)
                self.assertEqual((scene.elapsed, scene.animation_elapsed), (62, -9 + paused_dt))
                angle = paused_dt * scene._roll_rates[COBRA_ID]
                c, s = np.cos(angle), np.sin(angle)
                step = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
                np.testing.assert_allclose(scene.objects[COBRA_ID].matrix[:3, :3],
                                           negative_pose[:3, :3] @ step, atol=1e-12)
                scene.scrub(10)
                self.assertEqual((scene.elapsed, scene.animation_elapsed), (72, 1 + paused_dt))
                expected = self.scene(model_source="pioneer", source_interval=0)
                expected.seek(1 + paused_dt)
                np.testing.assert_allclose([obj.matrix for obj in scene.objects],
                                           [obj.matrix for obj in expected.objects], atol=1e-12)
                # У начала маршрута зажимается только пройденный маршрут;
                # накопленная разница часов остаётся обратимой.
                scene.scrub(-1000)
                self.assertEqual((scene.elapsed, scene.animation_elapsed), (0, -71 + paused_dt))
                scene.scrub(72)
                np.testing.assert_allclose([obj.matrix for obj in scene.objects],
                                           [obj.matrix for obj in expected.objects], atol=1e-12)

    def test_flight_toggle_at_finite_extreme_time_remains_finite(self):
        scene = self.scene()
        scene.seek(1e308)
        scene.toggle_flight()
        scene.toggle_flight()
        scene.update(0)
        self.assertTrue(np.isfinite((scene.elapsed, scene.animation_elapsed)).all())
        self.assertTrue(np.isfinite(scene.view_projection(1.6)).all())


if __name__ == "__main__":
    unittest.main()
