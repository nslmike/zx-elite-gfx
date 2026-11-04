"""Размер оконного drawable меняется независимо от логического размера SDL."""

from types import SimpleNamespace
from io import StringIO
import unittest
from unittest.mock import MagicMock, patch

try:
    import moderngl
    import numpy as np
    import pygame
except ModuleNotFoundError:
    moderngl = None

if moderngl is not None:
    from elite_lab.space_flyby import FlybyScene
    from elite_lab.space_hud import SpaceHUD
    from elite_lab.space_scene import SpaceScene
    from elite_lab.space_viewer import (FlybyControls, SpaceControls, SpaceRenderer,
                                      _window_sizes, _window_drawable_query, run_space)


@unittest.skipIf(moderngl is None, "Для окна OpenGL нужны requirements-space.txt")
class SpaceWindowTest(unittest.TestCase):
    def test_optional_cocoa_failure_keeps_the_sdl_window_available(self):
        provider = lambda logical: logical
        with patch("elite_lab.space_viewer.sys.platform", "darwin"), \
             patch("pygame.display.get_driver", return_value="cocoa"), \
             patch("elite_lab.space_viewer._CocoaDrawable", side_effect=OSError("нет AppKit")), \
             patch("elite_lab.space_viewer._sdl_drawable_query", return_value=provider), \
             patch("sys.stdout", new_callable=StringIO) as output:
            self.assertIs(_window_drawable_query(), provider)
        self.assertIn("обычный буфер SDL", output.getvalue())

    def test_other_platforms_do_not_load_cocoa(self):
        provider = lambda logical: logical
        with patch("elite_lab.space_viewer.sys.platform", "linux"), \
             patch("elite_lab.space_viewer._CocoaDrawable") as cocoa, \
             patch("elite_lab.space_viewer._sdl_drawable_query", return_value=provider):
            self.assertIs(_window_drawable_query(), provider)
        cocoa.assert_not_called()

    def test_resize_and_display_density_are_read_again_for_each_frame(self):
        drawable = ((1360, 820), (1600, 1000), (3000, 1800), (1500, 900))
        pixels = MagicMock(side_effect=drawable)
        ctx = SimpleNamespace(_window_drawable=pixels)
        logical = ((1360, 820), (800, 500), (1500, 900), (1500, 900))
        with patch("pygame.display.get_window_size", side_effect=logical), \
             patch("pygame.display.get_active", return_value=True):
            sizes = [_window_sizes(ctx) for _ in logical]
        self.assertEqual(sizes, list(zip(logical, drawable)))
        self.assertEqual([call.args[0] for call in pixels.call_args_list], list(logical))

    def test_temporarily_absent_drawable_is_not_rendered_at_an_invented_size(self):
        ctx = SimpleNamespace(_window_drawable=lambda logical: (0, 0))
        with patch("pygame.display.get_window_size", return_value=(800, 500)), \
             patch("pygame.display.get_active", return_value=True):
            self.assertEqual(_window_sizes(ctx), ((800, 500), None))

    def test_minimized_window_can_report_old_positive_size_without_a_drawable(self):
        pixels = MagicMock(return_value=(1600, 1000))
        ctx = SimpleNamespace(_window_drawable=pixels)
        with patch("pygame.display.get_window_size", return_value=(800, 500)), \
             patch("pygame.display.get_active", return_value=False):
            self.assertEqual(_window_sizes(ctx), ((800, 500), None))
        pixels.assert_not_called()

    def test_window_screenshot_without_drawable_reports_failure(self):
        ctx = MagicMock()
        with patch("elite_lab.space_viewer._window_context", return_value=ctx), \
             patch("elite_lab.space_viewer._window_sizes", return_value=((1360, 820), None)), \
             patch("elite_lab.space_viewer.SpaceRenderer"), \
             patch("pygame.display.set_caption"), patch("sys.stdout", new_callable=StringIO):
            with self.assertRaisesRegex(SystemExit, "буфер недоступен.*--headless"):
                run_space(None, MagicMock(), False, True)

    def test_hud_hitboxes_keep_logical_size_at_both_densities_and_clamps(self):
        pygame.font.init()
        scene = SpaceScene()
        scene.select(9)
        for logical in ((320, 240), (1360, 820), (2400, 1600)):
            hud = SpaceHUD(MagicMock())
            layouts = []
            for ratio in (1, 2):
                drawable = tuple(x * ratio for x in logical)
                hud._resize(drawable, ratio)
                hud._paint(scene, None, scene.view_projection(logical[0] / logical[1]))
                layouts.append((hud.s / ratio, {key: tuple(value) for key, value in hud.tabs.items()}))
                for action, rect in hud.tabs.items():
                    point = tuple(value / ratio for value in rect.center)
                    pixel = SpaceControls._pixel(point, logical, drawable)
                    self.assertEqual(hud.tab_at(pixel), action)
            self.assertAlmostEqual(layouts[0][0], layouts[1][0])
            for key, rect in layouts[0][1].items():
                # Один и тот же логический щелчок должен попадать в кнопку
                # после перехода на Retina. Ширина текста слегка зависит от
                # font hinting, поэтому не требуем побитового удвоения Rect.
                point = pygame.Rect(rect).center
                self.assertEqual(hud.tab_at(SpaceControls._pixel(point, logical,
                                                                 tuple(x * 2 for x in logical))), key)
                np.testing.assert_allclose(np.array(layouts[1][1][key])[[1, 3]] / 2,
                                           np.array(rect)[[1, 3]], atol=1)
            hud.release()

    def test_zx_transport_and_free_flight_hud_keep_logical_hitboxes_on_retina(self):
        pygame.font.init()
        for logical in ((320, 600), (1360, 820)):
            hud = SpaceHUD(MagicMock())
            self.addCleanup(hud.release)
            renderer = SpaceRenderer.__new__(SpaceRenderer)
            renderer.ctx, renderer.hud = hud.ctx, hud
            renderer.source, renderer.zx = "zx", MagicMock()
            layouts = {}
            for ratio in (1, 2):
                with self.subTest(logical=logical, ratio=ratio):
                    scene = FlybyScene(model_source="zx", source_interval=0)
                    controls = FlybyControls(scene, hud, True)
                    drawable = tuple(value * ratio for value in logical)
                    renderer.pixel_ratio = ratio

                    def draw():
                        renderer.draw(scene, MagicMock(), drawable, True)
                        self.assertEqual(hud.pixel_ratio, ratio)
                        for action, rect in hud.options.items():
                            point = tuple(value / ratio for value in rect.center)
                            self.assertEqual(hud.tab_at(controls._pixel(point, logical, drawable)), action)
                        if ratio == 1:
                            layouts[scene.camera_mode] = {action: rect.center for action, rect in hud.options.items()}
                        else:
                            # Тот же логический щелчок попадает в реальную
                            # кнопку и после удвоения оконного backing.
                            for action, point in layouts[scene.camera_mode].items():
                                self.assertEqual(hud.tab_at(controls._pixel(point, logical, drawable)), action)

                    def click(action):
                        point = tuple(value / ratio for value in hud.options[action].center)
                        for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                            controls.event(pygame.event.Event(kind, button=1, pos=point), logical, drawable)

                    draw()
                    click("toggle_pause")
                    self.assertTrue(scene.paused)
                    click("scrub_forward")
                    self.assertEqual(scene.elapsed, 1)
                    click("toggle_flight")
                    self.assertEqual(scene.camera_mode, "flight")
                    draw()
                    self.assertNotIn("toggle_pause", hud.options)
                    self.assertNotIn("toggle_auto", hud.options)
                    click("toggle_flight")
                    self.assertEqual(scene.camera_mode, "flyby")
