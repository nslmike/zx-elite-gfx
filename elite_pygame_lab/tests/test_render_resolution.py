"""Буфер 3D, drawable окна и модальный ввод используют разные координаты."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import pygame

try:
    import moderngl
    from elite_lab.space_flyby import FlybyScene
    from elite_lab.space_resolution import RENDER_RESOLUTIONS, RenderResolution
    from elite_lab.space_viewer import FlybyControls, SpaceControls, SpaceRenderer
except ModuleNotFoundError:
    SpaceRenderer = None


@unittest.skipIf(SpaceRenderer is None, "Нужны зависимости requirements-space.txt")
class RenderResolutionTest(unittest.TestCase):
    def renderer(self, samples=4):
        renderer = SpaceRenderer.__new__(SpaceRenderer)
        renderer.ctx = MagicMock(max_samples=samples, error="GL_NO_ERROR")
        for method in ("texture", "depth_renderbuffer", "renderbuffer", "framebuffer"):
            getattr(renderer.ctx, method).side_effect = lambda *args, **kwargs: Mock()
        renderer._world_target = None
        renderer._draw_world = Mock()
        renderer._present = Mock()
        renderer.hud = SimpleNamespace(resolution=RenderResolution(), draw=Mock())
        renderer.source, renderer.pixel_ratio = "elite", 2.0
        return renderer

    def test_world_resolves_before_presentation_and_native_hud(self):
        renderer = self.renderer()
        ctx = renderer.ctx
        target = Mock()
        scene = Mock(model_source="elite")
        size = (1600, 1000)
        internal = RENDER_RESOLUTIONS[1]
        self.assertIsNone(renderer.prepare_resolution(internal))
        world = renderer._world_target
        self.assertEqual(world.size, internal)
        ctx.texture.assert_called_once_with(internal, 4)
        ctx.depth_renderbuffer.assert_called_once_with(internal, samples=4)
        ctx.renderbuffer.assert_called_once_with(internal, 4, samples=4)

        order = Mock()
        order.attach_mock(renderer._draw_world, "world")
        order.attach_mock(ctx.copy_framebuffer, "resolve")
        order.attach_mock(renderer._present.render, "present")
        order.attach_mock(renderer.hud.draw, "hud")
        renderer.draw(scene, target, size, True, 9)
        vp = scene.view_projection.return_value
        scene.view_projection.assert_called_once_with(size[0] / size[1])
        renderer._draw_world.assert_called_once_with(scene, world.draw_frame, internal, vp, True)
        ctx.copy_framebuffer.assert_called_once_with(world.resolve_frame, world.draw_frame)
        self.assertEqual(ctx.viewport, (0, 0, *size))
        self.assertEqual(renderer.hud.resolution.output_size, size)
        renderer.hud.draw.assert_called_once_with(scene, size, 9, vp, pixel_ratio=2.0)
        self.assertEqual([entry[0] for entry in order.mock_calls], ["world", "resolve", "present", "hud"])

        scene.model_source = renderer.source = "zx"
        renderer.zx = Mock()
        order.reset_mock()
        renderer.draw(scene, target, size, True)
        renderer.zx.draw.assert_called_once_with(scene, size, vp)
        self.assertEqual([entry[0] for entry in order.mock_calls], ["hud"])
        self.assertIs(renderer._world_target, world)

    def test_partial_allocation_failure_keeps_the_previous_target(self):
        renderer = self.renderer(samples=0)
        ctx = renderer.ctx
        internal = RENDER_RESOLUTIONS[2]
        self.assertIsNone(renderer.prepare_resolution(internal))
        old = renderer._world_target
        self.assertIsNone(old.resolve_frame)
        self.assertIsNone(renderer.prepare_resolution(internal))
        ctx.texture.assert_called_once()
        new_texture, new_depth = Mock(), Mock()
        ctx.texture.side_effect = lambda *args: new_texture
        ctx.depth_renderbuffer.side_effect = lambda *args, **kwargs: new_depth
        ctx.framebuffer.side_effect = moderngl.Error("не хватает памяти")
        with patch("sys.stderr"):
            self.assertIn("3840 × 2160", renderer.prepare_resolution(RENDER_RESOLUTIONS[-1]))
        self.assertIs(renderer._world_target, old)
        for resource in old.resources:
            resource.release.assert_not_called()
        for resource in (new_texture, new_depth):
            resource.release.assert_called_once_with()
        self.assertIsNone(renderer.prepare_resolution(None))
        self.assertIsNone(renderer._world_target)
        for resource in old.resources:
            resource.release.assert_called_once_with()


@unittest.skipIf(SpaceRenderer is None, "Нужны зависимости requirements-space.txt")
class ResolutionControlsTest(unittest.TestCase):
    logical = (500, 300)
    drawable = (1000, 600)

    def event(self, kind, **attrs):
        return self.controls.event(pygame.event.Event(kind, **attrs), self.logical, self.drawable)

    def tap(self, key):
        self.event(pygame.KEYDOWN, key=key)
        self.event(pygame.KEYUP, key=key)

    def setUp(self):
        self.hud = SimpleNamespace(menu=None, tab_at=Mock(return_value=None), resolution=RenderResolution())
        self.enterContext(patch("pygame.display.get_init", return_value=False))

    def test_modal_keys_preserve_route_and_restore_capture_only_before_focus_loss(self):
        scene = FlybyScene(source_interval=0)
        scene.paused = True
        self.controls = FlybyControls(scene, self.hud, False)
        self.event(pygame.KEYDOWN, key=pygame.K_RIGHT)
        self.assertEqual(scene.elapsed, 1)
        self.tap(pygame.K_g)
        self.assertTrue(self.controls.show_hud)
        self.assertEqual(self.hud.menu, "resolution")
        self.hud.resolution.unavailable.add(len(RENDER_RESOLUTIONS) - 1)
        self.tap(pygame.K_UP)
        self.assertEqual(self.hud.resolution.highlighted, len(RENDER_RESOLUTIONS) - 2)
        self.controls.update(1)
        self.assertEqual(scene.elapsed, 1)
        self.tap(pygame.K_RETURN)
        self.assertIsNone(self.hud.menu)
        self.event(pygame.KEYDOWN, key=pygame.K_RIGHT, repeat=True)
        self.controls.update(1)
        self.assertEqual(scene.elapsed, 1)
        self.tap(pygame.K_f)
        self.assertTrue(self.controls.looking)
        self.tap(pygame.K_g)
        self.assertFalse(self.controls.looking)
        self.tap(pygame.K_ESCAPE)
        self.assertTrue(self.controls.looking)
        self.tap(pygame.K_g)
        self.event(pygame.WINDOWFOCUSLOST)
        self.event(pygame.WINDOWFOCUSGAINED)
        self.tap(pygame.K_ESCAPE)
        self.assertFalse(self.controls.looking)
        self.assertEqual(scene.camera_mode, "flight")

    def test_retina_mouse_menu_keeps_current_preset_on_failed_selection(self):
        scene = Mock(camera_mode="overview")
        self.controls = SpaceControls(scene, self.hud, True)
        self.controls.prepare_resolution = Mock(side_effect=("Не удалось создать буфер.", None))

        def click(action):
            self.hud.tab_at.return_value = action
            for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                self.event(kind, button=1, pos=(20, 15))
            self.hud.tab_at.assert_called_with((40, 30))

        click("menu_resolution")
        self.assertEqual(self.hud.menu, "resolution")
        click("resolution_1")
        self.assertEqual(self.hud.resolution.selected, 0)
        self.assertIsNotNone(self.hud.resolution.error)
        self.assertEqual(self.hud.menu, "resolution")
        self.tap(pygame.K_KP_ENTER)
        self.assertEqual(self.hud.resolution.selected, 1)
        self.assertIsNone(self.hud.menu)
        scene.pick.assert_not_called()

    def test_scene_picking_matches_output_aspect_at_fractional_backing_size(self):
        scene = Mock(camera_mode="overview")
        controls = SpaceControls(scene, self.hud, False)
        drawable = (1001, 600)
        controls.pick((250, 150), self.logical, drawable)
        scene.pick.assert_called_once_with(250.25, 150, 500.5, 300)


if __name__ == "__main__":
    unittest.main()
