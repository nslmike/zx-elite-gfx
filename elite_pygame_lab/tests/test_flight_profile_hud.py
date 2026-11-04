"""Реальные подписи обеих схем, доступность C и обновление скорости в HUD."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pygame

try:
    import numpy as np
    from elite_lab.space_hud import SpaceHUD
except ModuleNotFoundError:
    SpaceHUD = None


@unittest.skipIf(SpaceHUD is None, "Нужны зависимости requirements-space.txt")
class FlightProfileHUDTest(unittest.TestCase):
    def setUp(self):
        pygame.font.init()
        self.addCleanup(pygame.font.quit)

    def hud(self, size=(1360, 820), ratio=1):
        hud = SpaceHUD.__new__(SpaceHUD)
        hud.texture, hud.size, hud.key = None, None, None
        hud.ctx = SimpleNamespace(texture=lambda *_: Mock(), enable_only=Mock())
        hud.quad = Mock()
        hud.tabs, hud.markers, hud.options, hud.menu = {}, {}, {}, None
        hud._resize(size, ratio)
        return hud

    @staticmethod
    def scene(*, flyby, profile="modern"):
        ship = SimpleNamespace(name="Cobra Mk III", materials=(), meshes=(), source_id="elite",
                               source_title="ZX Spectrum Elite", author="Ian Bell, David Braben",
                               license="Источник каркаса")
        return SimpleNamespace(is_flyby=flyby, camera_mode="flight", selected=0, ships=(ship,),
                               model_source="elite", texture_source="author", empty_slot=False,
                               auto_sources=False, elapsed=0, paused=False, flight_profile=profile,
                               flight_speed=0, catalog=SimpleNamespace(sources=("elite",)))

    @staticmethod
    def paint_labels(hud, scene):
        """Запоминаем текст и реальные прямоугольники растрированных шрифтов."""
        labels, images = [], {}
        surface_type = pygame.Surface

        class TrackingSurface(surface_type):
            def blit(self, source, dest, *args, **kwargs):
                if id(source) in images:
                    labels.append((images[id(source)], source.get_rect(topleft=dest)))
                return super().blit(source, dest, *args, **kwargs)

        class TrackingFont:
            def __init__(self, font):
                self.font = font

            def __getattr__(self, name):
                return getattr(self.font, name)

            def render(self, text, *args, **kwargs):
                image = self.font.render(text, *args, **kwargs)
                images[id(image)] = text
                return image

        fonts = {name: getattr(hud, name) for name in ("mono", "text", "symbols", "title", "brand")}
        for name, font in fonts.items():
            setattr(hud, name, TrackingFont(font))
        try:
            with patch("elite_lab.space_hud.pygame.Surface", TrackingSurface):
                surface = SpaceHUD._paint(hud, scene, None, np.eye(4))
        finally:
            for name, font in fonts.items():
                setattr(hud, name, font)
        return surface, labels

    def test_flight_actions_and_complete_hints_fit_standard_and_retina_windows(self):
        for size, ratio in (((640, 360), 1), ((1360, 820), 1),
                            ((2560, 1600), 2), ((3024, 1964), 2)):
            for flyby in (False, True):
                for profile in ("modern", "elite"):
                    with self.subTest(size=size, ratio=ratio, flyby=flyby, profile=profile):
                        hud = self.hud(size, ratio)
                        scene = self.scene(flyby=flyby, profile=profile)
                        surface, labels = self.paint_labels(hud, scene)
                        bounds = surface.get_rect()
                        button = hud.options["cycle_flight_profile"]
                        self.assertEqual(hud.tab_at(button.center), "cycle_flight_profile")
                        self.assertTrue(bounds.contains(button))
                        self.assertIn("C  Схема · " + ("Elite" if profile == "elite" else "3D"),
                                      [text for text, _ in labels])
                        hints = set(hud._flight_hints(scene))
                        rendered = [(text, rect) for text, rect in labels if text in hints]
                        self.assertEqual({text for text, _ in rendered}, hints)
                        for text, rect in rendered:
                            self.assertTrue(bounds.contains(rect), text)
                            self.assertFalse(any(rect.colliderect(other) for _, other in rendered
                                                 if other is not rect), text)
                        if profile == "elite":
                            self.assertTrue(any("0 ед/с" in text for text, _ in labels))
                            self.assertFalse(any("стрейф" in text or "Shift  вверх" in text
                                                 for text, _ in labels))
                        else:
                            self.assertFalse(any("ед/с" in text for text, _ in labels))

    def test_profile_and_displayed_speed_invalidate_cached_layer(self):
        for flyby in (False, True):
            with self.subTest(flyby=flyby):
                hud = self.hud()
                scene = self.scene(flyby=flyby)
                with patch.object(hud, "_paint", wraps=hud._paint) as paint:
                    hud.draw(scene, hud.size, None, np.eye(4))
                    hud.draw(scene, hud.size, None, np.eye(4))
                    self.assertEqual(paint.call_count, 1)
                    scene.flight_profile = "elite"
                    hud.draw(scene, hud.size, None, np.eye(4))
                    self.assertEqual(paint.call_count, 2)
                    scene.flight_speed = 120.6
                    hud.draw(scene, hud.size, None, np.eye(4))
                    self.assertEqual(paint.call_count, 3)
                    _, labels = self.paint_labels(hud, scene)
                    self.assertTrue(any("121 ед/с" in text for text, _ in labels))
                    scene.flight_speed = 120.7
                    hud.draw(scene, hud.size, None, np.eye(4))
                    self.assertEqual(paint.call_count, 3)
                    scene.flight_speed = 0
                    hud.draw(scene, hud.size, None, np.eye(4))
                    self.assertEqual(paint.call_count, 4)

    def test_small_nonzero_speed_stays_visible_and_cache_matches_displayed_text(self):
        cases = ((0, "0 ед/с", 1), (.01, "<0,1 ед/с", 2), (.06, "<0,1 ед/с", 2),
                 (.1, "0,1 ед/с", 3), (.16, "0,2 ед/с", 4), (.24, "0,2 ед/с", 4),
                 (.32, "0,3 ед/с", 5), (4.91, "4,9 ед/с", 6), (4.92, "4,9 ед/с", 6),
                 (5, "5 ед/с", 7), (5.04, "5 ед/с", 7), (0, "0 ед/с", 8))
        for flyby in (False, True):
            hud = self.hud()
            scene = self.scene(flyby=flyby, profile="elite")
            with patch.object(hud, "_paint", wraps=hud._paint) as paint:
                for speed, expected, paints in cases:
                    with self.subTest(flyby=flyby, speed=speed):
                        scene.flight_speed = speed
                        hud.draw(scene, hud.size, None, np.eye(4))
                        self.assertEqual(paint.call_count, paints)
                        _, labels = self.paint_labels(hud, scene)
                        self.assertTrue(any(text.endswith(" · " + expected) for text, _ in labels))
                scene.flight_profile = "modern"
                hud.draw(scene, hud.size, None, np.eye(4))
                before = paint.call_count
                scene.flight_speed = .16
                hud.draw(scene, hud.size, None, np.eye(4))
                self.assertEqual(paint.call_count, before)

    def test_atlas_has_profile_button_only_in_flight_and_keyboard_look_outside(self):
        hud = self.hud()
        scene = self.scene(flyby=False)
        scene.camera_mode = "inspect"
        _, labels = self.paint_labels(hud, scene)
        self.assertNotIn("cycle_flight_profile", hud.options)
        self.assertIn("Shift + стрелки  взгляд", [text for text, _ in labels])
        scene.camera_mode = "flight"
        hud._paint(scene, None, np.eye(4))
        self.assertIn("cycle_flight_profile", hud.options)


if __name__ == "__main__":
    unittest.main()
