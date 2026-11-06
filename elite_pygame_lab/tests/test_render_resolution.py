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
        renderer.ctx.info = {"GL_MAX_COLOR_TEXTURE_SAMPLES": samples}
        for method in ("texture", "depth_renderbuffer", "renderbuffer"):
            getattr(renderer.ctx, method).side_effect = lambda *args, **kwargs: Mock()
        renderer.ctx.framebuffer.side_effect = lambda color, *args: Mock(color_attachments=(color,))
        renderer._world_target = None
        renderer._internal_size = None
        renderer._effects_active = False
        renderer._effect_profiles = {}
        renderer._failed_target = None
        renderer._bloom = Mock()
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
        renderer._draw_world.assert_called_once_with(scene, world.draw_frame, internal, vp, True, linear=False)
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

    def test_hdr_uses_drawable_samples_and_presents_a_fixed_world_before_hud(self):
        renderer = self.renderer()
        ctx = renderer.ctx
        target = Mock(samples=0)
        renderer._effect_profiles = {9: 1}
        scene = Mock(model_source="elite", effects_enabled=True)
        size = (1001, 603)
        renderer.draw(scene, target, size, False)
        auto = renderer._world_target
        self.assertTrue(auto.linear)
        self.assertIsNone(auto.resolve_frame)
        ctx.depth_renderbuffer.assert_called_once_with(size, samples=0)
        for call in ctx.texture.call_args_list[1:]:
            self.assertEqual(call.args[0], (251, 151))
        renderer._bloom.draw.assert_called_once_with(auto, target, size)

        internal = RENDER_RESOLUTIONS[1]
        self.assertIsNone(renderer.prepare_resolution(internal))
        fixed = renderer._world_target
        self.assertEqual(fixed.samples, 4)
        self.assertIsNotNone(fixed.presentation)
        for resource in auto.resources:
            resource.release.assert_called_once_with()
        order = Mock()
        order.attach_mock(ctx.copy_framebuffer, "resolve")
        order.attach_mock(renderer._bloom.draw, "bloom")
        order.attach_mock(renderer._present.render, "present")
        order.attach_mock(renderer.hud.draw, "hud")
        renderer.draw(scene, target, size, True)
        self.assertEqual([entry[0] for entry in order.mock_calls], ["resolve", "bloom", "present", "hud"])
        renderer._bloom.draw.assert_called_with(fixed, fixed.presentation, internal)
        self.assertEqual(ctx.viewport, (0, 0, *size))

        scene.effects_enabled = False
        renderer.draw(scene, target, size, False)
        self.assertFalse(renderer._world_target.linear)
        for resource in fixed.resources:
            resource.release.assert_called_once_with()

    def test_failed_native_hdr_resize_uses_the_whole_output_and_can_recover(self):
        renderer = self.renderer()
        renderer._effect_profiles = {9: 1}
        scene = Mock(model_source="elite", effects_enabled=True)
        target = Mock(samples=4)
        renderer.draw(scene, target, (960, 540), False)
        old = renderer._world_target
        resize = (1401, 801)
        renderer._bloom.draw.reset_mock()
        with patch.object(renderer.ctx, "framebuffer", side_effect=moderngl.Error("нет буфера")) as create:
            with patch("sys.stderr"):
                renderer.draw(scene, target, resize, False)
                renderer.draw(scene, target, resize, False)
            create.assert_called_once()
        self.assertIs(renderer._world_target, old)
        renderer._draw_world.assert_called_with(scene, target, resize, scene.view_projection.return_value, False)
        renderer._bloom.draw.assert_not_called()

        recovered = (1201, 700)
        renderer.draw(scene, target, recovered, False)
        self.assertEqual(renderer._world_target.size, recovered)
        renderer._bloom.draw.assert_called_once_with(renderer._world_target, target, recovered)
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
