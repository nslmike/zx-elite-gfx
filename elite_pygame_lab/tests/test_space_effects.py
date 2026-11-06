"""Область действия огней и детерминированные часы сцены."""

from dataclasses import replace
from pathlib import Path
from sys import float_info
from types import SimpleNamespace
import unittest

import pygame

try:
    import numpy as np
    from elite_lab.space_assets import AssetCatalog
    from elite_lab.space_effects import legacy_light_levels, ship_effects_profile
    from elite_lab.space_fleet_lights import MAX_LENSES, lens_mounts
    from elite_lab.space_flyby import FlybyScene
    from elite_lab.space_mesh import build_mesh
    from elite_lab.space_scene import SpaceScene
    from elite_lab.space_viewer import FlybyControls
except ModuleNotFoundError:
    FlybyControls = None


@unittest.skipIf(FlybyControls is None, "Нужны зависимости requirements-space.txt")
class ShipEffectsTest(unittest.TestCase):
    def test_only_native_artificial_objects_have_effects(self):
        catalog = AssetCatalog()
        active = {ship.model_id for ship in catalog.classic if ship_effects_profile(ship)}
        self.assertEqual(active, set(range(len(catalog.classic))) - {3, 5, 6})
        for model_id in active:
            ship = catalog.classic[model_id]
            self.assertFalse(ship_effects_profile(replace(ship, source_id="griff")))
            self.assertFalse(ship_effects_profile(replace(ship, meshes=(Path("mesh.npz"),))))
        for source in catalog.sources[1:]:
            for ship in catalog.fleet(source):
                if ship.source_id != "elite":
                    self.assertFalse(ship_effects_profile(ship))

    def test_pulses_follow_scene_clock_in_pause_flight_and_seek(self):
        atlas = SpaceScene(9)
        atlas.select(9)
        atlas.update(.75)
        self.assertEqual(atlas.animation_elapsed, .75)

        scene = FlybyScene(9, model_source="elite", source_interval=0)
        scene.seek(6.)
        profile = ship_effects_profile(scene.ships[9]).legacy
        initial = legacy_light_levels(profile, scene.animation_elapsed)
        scene.toggle_pause()
        scene.update(.75)
        self.assertEqual(scene.elapsed, 6.)
        self.assertEqual(scene.animation_elapsed, 6.75)
        self.assertNotEqual(legacy_light_levels(profile, scene.animation_elapsed), initial)
        scene.toggle_flight()
        scene.update(.25)
        self.assertEqual(scene.animation_elapsed, 7.)
        scene.seek(6.)
        self.assertEqual(legacy_light_levels(profile, scene.animation_elapsed), initial)
        # Два сопла Cobra меняют яркость в разной фазе. Python имеет отдельную
        # апертуру и более медленный маяк.
        left, right = initial[0]
        self.assertNotEqual(left, right)
        python = ship_effects_profile(scene.ships[10]).legacy
        self.assertNotEqual(legacy_light_levels(python, 6.)[1], legacy_light_levels(python, 6.75)[1])
        for phase in (-float_info.max, float_info.max):
            for ship_profile in (profile, python):
                engine, marker = legacy_light_levels(ship_profile, phase)
                self.assertTrue(all(.5 < gain < 1.5 for gain in engine))
                self.assertTrue(0 < marker <= 1.)

    def test_new_profiles_have_distinct_animated_signals(self):
        traces = set()
        times = (0., .42, .93, 1.65, 2.4, 3.92, 5.34, 7.12)
        for ship in AssetCatalog().classic:
            profile = ship_effects_profile(ship)
            if profile is None or profile.legacy:
                continue
            with self.subTest(model=ship.model_id):
                beats = (*profile.drive, *(lens.beat for lens in profile.lenses if lens.beat))
                if profile.panels is not None:
                    beats += (profile.panels.first.beat, profile.panels.second.beat)
                self.assertTrue(beats)
                signature = tuple(tuple(beat.level(t) for t in times) for beat in beats)
                self.assertNotIn(signature, traces)
                traces.add(signature)
                for beat, levels in zip(beats, signature):
                    self.assertGreater(max(levels) - min(levels), .12)
                    for phase in (-float_info.max, float_info.max):
                        self.assertTrue(0. < beat.level(phase) < 2.)

    def test_fixture_footprints_fit_their_rendered_hull_faces(self):
        for ship in AssetCatalog().classic:
            profile = ship_effects_profile(ship)
            if profile is None or not profile.lenses:
                continue
            triangles = build_mesh(ship.model_id).reshape(-1, 3, 9)
            hull = triangles[triangles[:, 0, 8] == 0]
            self.assertLessEqual(len(profile.lenses), MAX_LENSES)
            for lens, (point, normal, axis) in zip(profile.lenses, lens_mounts(ship.model_id)):
                with self.subTest(model=ship.model_id, mount=lens.vertices):
                    vertices = hull[:, :, :3]
                    matches = ((hull[:, 0, 3:6] @ normal > .999) &
                               (np.abs((vertices[:, 0] - point) @ normal) < .0005))
                    surface = vertices[matches]
                    self.assertGreater(len(surface), 0)
                    cross_axis = np.cross(normal, axis)
                    direction = np.roll(surface, -1, axis=1) - surface
                    for u in (-1, 0, 1):
                        for v in (-1, 0, 1):
                            sample = (point + u * (lens.size[0] + .005) * np.asarray(axis) +
                                      v * (lens.size[1] + .005) * cross_axis)
                            side = np.cross(direction, sample - surface) @ normal
                            self.assertTrue(np.any(np.all(side >= -1e-6, axis=1)), f"Край линзы {sample}")

    def test_station_profiles_leave_deep_windows_steady(self):
        catalog = AssetCatalog()
        for model_id in (0, 19):
            with self.subTest(model=model_id):
                ship = catalog.classic[model_id]
                panels = ship_effects_profile(ship).panels
                mesh = ship.vertices().reshape(-1, 3, 9)
                windows = mesh[mesh[:, 0, 8] == 6, :, :3]
                exterior = np.all(windows[:, :, 2] >= panels.min_z, axis=1)
                interior = np.all(windows[:, :, 2] < panels.min_z, axis=1)
                self.assertTrue(exterior.any() and interior.any())
                self.assertTrue(np.all(exterior | interior))
                self.assertNotEqual(panels.first.beat, panels.second.beat)

    def test_b_toggles_color_effects_and_leaves_zx_alone(self):
        scene = FlybyScene(9, model_source="elite", source_interval=0)
        controls = FlybyControls(scene, SimpleNamespace(menu=None), False)
        size = (800, 450)

        def tap():
            for kind in (pygame.KEYDOWN, pygame.KEYUP):
                controls.event(pygame.event.Event(kind, key=pygame.K_b), size, size)

        self.assertTrue(scene.effects_enabled)
        tap()
        self.assertFalse(scene.effects_enabled)
        scene.set_source("zx")
        tap()
        self.assertFalse(scene.effects_enabled)
        scene.set_source("elite")
        tap()
        self.assertTrue(scene.effects_enabled)
