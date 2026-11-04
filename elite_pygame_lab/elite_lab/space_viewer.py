"""Окно и внеэкранный рендер космического атласа на OpenGL Core."""

import os
import sys
from math import hypot, tau

import moderngl
import numpy as np
import pygame

from .space_hud import SpaceHUD
from .space_look import SUN_DIRECTION, SpaceLook
from .space_assets import AssetCatalog, AssetError
from .space_scene import SpaceScene


_SIZE = (1360, 820)
_ORBIT_VERTEX = """#version 330
uniform mat4 vp;
in vec3 position;
void main() { gl_Position = vp * vec4(position, 1.0); }
"""
_ORBIT_FRAGMENT = """#version 330
out vec4 color;
void main() { color = vec4(0.27, 0.38, 0.42, 0.48); }
"""
_DEPTH_VERTEX = """#version 330
uniform mat4 light_mvp;
in vec3 position;
void main() { gl_Position = light_mvp * vec4(position, 1.0); }
"""


def _matrix(matrix):
    return np.ascontiguousarray(matrix.T, dtype="f4").tobytes()


class MeshBuffer:
    """Один VBO, диапазон треугольников для каждого авторского материала."""

    def __init__(self, ctx, program, vertices, imported):
        self.parts = []
        if imported:
            triangles = vertices.reshape(-1, 3, 9)
            slots = triangles[:, 0, 8].astype(int)
            order = np.argsort(slots, kind="stable")
            vertices = triangles[order]
            ids, starts, counts = np.unique(slots[order], return_index=True, return_counts=True)
            self.parts = [(int(i), int(s * 3), int(n * 3)) for i, s, n in zip(ids, starts, counts)]
        self.buffer = ctx.buffer(vertices)
        self.vao = ctx.vertex_array(program, [(self.buffer, "3f 3f 2f 1f",
                                    "in_position", "in_normal", "in_uv", "in_material")])

    def release(self):
        self.vao.release()
        self.buffer.release()


class SpaceRenderer:
    def __init__(self, ctx, scene):
        self.ctx = ctx
        self.look = SpaceLook(ctx)
        self.hud = SpaceHUD(ctx)
        self.buffers = {}
        self.shadow = None
        self._load_fleet(scene)
        # Разметка обзора лежит в мировой плоскости расположения экспонатов.
        angles = (np.arange(220)[:, None] + np.array((0, .55))) * tau / 220
        circle = np.stack((np.sin(angles), np.zeros_like(angles), np.cos(angles)), axis=-1)
        self.orbit_buf = ctx.buffer((circle * scene.ring_radius).astype("f4"))
        self.orbit_program = ctx.program(vertex_shader=_ORBIT_VERTEX, fragment_shader=_ORBIT_FRAGMENT)
        self.orbit = ctx.simple_vertex_array(self.orbit_program, self.orbit_buf, "position")

    def _load_fleet(self, scene):
        buffers = {}
        meshes = []
        try:
            for ship, obj in zip(scene.ships, scene.objects):
                levels = []
                for i, key in enumerate(ship.meshes or (ship.model_id,)):
                    if key not in buffers:
                        buffers[key] = MeshBuffer(self.ctx, self.look.hull, ship.vertices(i), bool(ship.meshes))
                    levels.append(buffers[key])
                meshes.append((levels, _matrix(obj.matrix)))
        except Exception:
            for buf in buffers.values():
                buf.release()
            raise
        for buf in self.buffers.values():
            buf.release()
        self.look.release_imported_textures()
        self.buffers, self.meshes = buffers, meshes
        if self.shadow:
            self.shadow.release()
        self.shadow = self._bake_shadow(scene)
        self.source = scene.model_source

    def _bake_shadow(self, scene):
        # И корабли, и звезда неподвижны. Одной карты глубины хватает на весь облёт.
        ctx = self.ctx
        sun = np.array(SUN_DIRECTION)
        right = np.cross((0, 1, 0), sun)
        right /= np.linalg.norm(right)
        axes = np.array((right, np.cross(sun, right), -sun))
        centers = np.array([obj.position for obj in scene.objects]) @ axes.T
        radii = np.array([obj.radius for obj in scene.objects])[:, None] + .1
        lo, hi = (centers - radii).min(axis=0), (centers + radii).max(axis=0)
        light_vp = np.eye(4)
        light_vp[:3, :3] = axes * (2 / (hi - lo))[:, None]
        light_vp[:3, 3] = -(hi + lo) / (hi - lo)

        resolution = min(3072, ctx.info["GL_MAX_TEXTURE_SIZE"])
        depth = ctx.depth_texture((resolution, resolution))
        depth.filter = (moderngl.LINEAR, moderngl.LINEAR)
        depth.repeat_x = depth.repeat_y = False
        depth.compare_func = "<="
        target = ctx.framebuffer(depth_attachment=depth)
        prog = ctx.program(vertex_shader=_DEPTH_VERTEX, fragment_shader="#version 330\nvoid main() {}")
        target.use()
        target.clear(depth=1.0)
        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        # Корпуса и накладные детали замкнуты. Записываем дальнюю оболочку
        # со стороны звезды, чтобы фильтр тени не затенял сам освещённый фасет.
        ctx.cull_face = "front"
        for obj, (levels, _) in zip(scene.objects, self.meshes):
            vao = ctx.vertex_array(prog, [(levels[0].buffer, "3f 24x", "position")])
            prog["light_mvp"].write(_matrix(light_vp @ obj.matrix))
            vao.render()
            vao.release()
        ctx.cull_face = "back"
        prog.release()
        target.release()
        self.look.hull["u_shadow_vp"].write(_matrix(light_vp))
        self.look.hull["u_shadow"].value = 2
        self.look.hull["u_shadow_enabled"].value = True
        return depth

    def draw(self, scene, target, size, show_hud, hover=None):
        ctx = self.ctx
        if self.source != scene.model_source:
            self._load_fleet(scene)
        target.use()
        ctx.viewport = (0, 0, *size)
        target.clear(0.01, 0.02, 0.03, 1.0, depth=1.0, viewport=ctx.viewport)
        vp = scene.view_projection(size[0] / size[1])
        eye = scene.eye
        self.look.draw_sky(np.linalg.inv(vp), eye)
        if show_hud and scene.camera_mode == "overview":
            ctx.enable_only(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.orbit_program["vp"].write(_matrix(vp))
            self.orbit.render(mode=moderngl.LINES)
        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.front_face = "ccw"
        ctx.cull_face = "back"
        hull = self.look.hull
        self.shadow.use(location=2)
        hull["u_view_proj"].write(_matrix(vp))
        hull["u_eye"].value = tuple(eye)
        planes = np.vstack((vp[3] + vp[:3], vp[3] - vp[:3]))
        planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
        focal = size[1] * np.linalg.norm(vp[1, :3])
        for ship, obj, (levels, transform) in zip(scene.ships, scene.objects, self.meshes):
            if np.any(planes[:, :3] @ obj.position + planes[:, 3] < -obj.radius):
                continue
            depth = vp[3, :3] @ obj.position + vp[3, 3]
            diameter_px = focal * obj.radius / max(.001, depth)
            lod = min(len(levels) - 1, 2 if diameter_px < 80 else 1 if diameter_px < 220 else 0)
            mesh = levels[lod]
            authored = bool(ship.materials) and scene.texture_source == "author"
            hull["u_model"].write(transform)
            self.look.configure_model(obj.model_id, imported=bool(ship.meshes), authored=authored,
                                      cross_bitangent=ship.cross_bitangent)
            if mesh.parts:
                for slot, first, count in mesh.parts:
                    material = ship.materials[slot]
                    if authored:
                        self.look.use_material(material)
                    else:
                        hull["u_part_type"].value = material.procedural_type
                    mesh.vao.render(first=first, vertices=count)
            else:
                mesh.vao.render()
        if show_hud:
            self.hud.draw(scene, size, hover, vp)

    def release(self):
        self.orbit.release()
        self.orbit_buf.release()
        self.orbit_program.release()
        self.shadow.release()
        for buf in self.buffers.values():
            buf.release()
        self.hud.release()
        self.look.release()


def _window_context():
    pygame.display.init()
    pygame.display.set_caption("Elite · Подготовка звёздного атласа…")
    desktop = pygame.display.get_desktop_sizes()[0]
    size = (min(_SIZE[0], int(desktop[0] * .95)), min(_SIZE[1], int(desktop[1] * .90)))
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    for samples in (4, 0):
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, int(samples > 0))
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, samples)
        try:
            pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=1)
            break
        except pygame.error:
            if not samples:
                raise
    return moderngl.create_context(require=330)


class SpaceControls:
    def __init__(self, scene, hud, show_hud):
        self.scene = scene
        self.hud = hud
        self.show_hud = show_hud
        self.press = None
        self.dragging = False

    def _mode(self, mode):
        if mode.startswith("menu_"):
            menu = mode.removeprefix("menu_")
            self.hud.menu = None if self.hud.menu == menu else menu
            return
        self.hud.menu = None
        if mode.startswith("source_"):
            self.scene.set_source(mode.removeprefix("source_"))
        elif mode.startswith("texture_"):
            self.scene.texture_source = mode.removeprefix("texture_")
        elif mode == "overview":
            self.scene.overview()
        elif mode == "inspect":
            self.scene.select(self.scene.selected)
        elif mode == "flight":
            self.scene.fly()

    def pick(self, point, logical_size, drawable_size):
        if self.show_hud:
            pixel = self._pixel(point, logical_size, drawable_size)
            if self.hud.tab_at(pixel):
                return None
            if self.scene.camera_mode == "overview":
                marker = self.hud.marker_at(pixel)
                if marker is not None:
                    return marker
        return self.scene.pick(*point, *logical_size)

    @staticmethod
    def _pixel(point, logical_size, drawable_size):
        return tuple(round(coord * draw / max(logical, 1))
                     for coord, draw, logical in zip(point, drawable_size, logical_size))

    def event(self, e, logical_size, drawable_size):
        scene = self.scene
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.WINDOWFOCUSLOST:
            self.press = None
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                if self.hud.menu:
                    self.hud.menu = None
                elif scene.camera_mode == "overview":
                    return False
                else:
                    scene.overview()
            elif e.key == pygame.K_m:
                self.hud.menu = None
                scene.cycle_source()
            elif e.key == pygame.K_t:
                self.hud.menu = None
                scene.cycle_textures()
            elif e.key == pygame.K_i:
                self.show_hud = True
                self._mode("menu_credits")
            elif e.key in (pygame.K_LEFT, pygame.K_RIGHT):
                scene.select(scene.selected + (1 if e.key == pygame.K_RIGHT else -1),
                             focus=scene.camera_mode == "inspect")
            elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._mode("inspect")
            elif e.key in (pygame.K_o, pygame.K_HOME):
                self._mode("overview")
            elif e.key == pygame.K_TAB:
                self._mode("inspect" if scene.camera_mode == "overview" else "overview")
            elif e.key == pygame.K_f:
                self._mode("flight")
            elif e.key == pygame.K_h:
                self.show_hud = not self.show_hud
                self.hud.menu = None
            elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                scene.zoom(1)
            elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                scene.zoom(-1)
        elif e.type == pygame.MOUSEWHEEL:
            scene.zoom(getattr(e, "precise_y", e.y))
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
            pixel = self._pixel(e.pos, logical_size, drawable_size)
            action = self.hud.tab_at(pixel) if self.show_hud else None
            self.press = (e.button, e.pos, action)
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.press:
            if self.press[2]:
                return True
            # Небольшое дрожание курсора на трекпаде сохраняет обычный щелчок.
            if self.dragging:
                scene.drag(*e.rel)
            else:
                dx, dy = e.pos[0] - self.press[1][0], e.pos[1] - self.press[1][1]
                if hypot(dx, dy) >= 4:
                    self.dragging = True
                    scene.drag(dx, dy)
        elif e.type == pygame.MOUSEBUTTONUP and self.press and e.button == self.press[0]:
            start_action = self.press[2]
            self.press = None
            if not self.dragging and e.button == 1:
                pixel = self._pixel(e.pos, logical_size, drawable_size)
                tab = self.hud.tab_at(pixel) if self.show_hud else None
                if tab and tab == start_action:
                    self._mode(tab)
                elif not (tab or start_action):
                    ident = self.pick(e.pos, logical_size, drawable_size)
                    if ident is not None:
                        scene.select(ident)
            self.dragging = False
        return True


def _save(frame, size, path):
    rgb = frame.read(viewport=(0, 0, *size), components=3, alignment=1)
    surface = pygame.image.frombytes(rgb, size, "RGB", True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def run_space(model_id, screenshot, headless, show_hud, model_source="elite", texture_source="author"):
    catalog = AssetCatalog()
    ships = catalog.fleet(model_source)
    if model_id is not None and model_id not in range(len(ships)):
        raise AssetError(f"Для {model_source} доступен диапазон 0–{len(ships)-1}. Список наборов: --list-sources")
    pygame.font.init()
    ctx = renderer = offscreen = resolved = None
    try:
        try:
            if headless:
                # На Linux без X-сервера Mesa предоставляет автономный EGL-контекст.
                options = {"backend": "egl"} if sys.platform.startswith("linux") and not os.getenv("DISPLAY") else {}
                ctx = moderngl.create_context(standalone=True, require=330, **options)
            else:
                ctx = _window_context()
        except Exception as exc:
            raise SystemExit(f"Космическому атласу нужен OpenGL 3.3 Core. Не удалось открыть контекст: {exc}") from exc
        print(f"OpenGL {ctx.info['GL_VERSION']} · {ctx.info['GL_RENDERER']}", flush=True)
        scene = SpaceScene(model_source=model_source, texture_source=texture_source, catalog=catalog)
        if model_id is not None:
            scene.select(model_id)
        renderer = SpaceRenderer(ctx, scene)
        if not headless:
            pygame.display.set_caption("Elite · Орбитальный каталог")
        if headless or screenshot:
            # Для чтения на Mac сводим выборки MSAA в отдельный буфер с одним
            # цветом на пиксель. Скриншот окна использует тот же контекст и кадр.
            size = _SIZE if headless else ctx.screen.size
            if not headless:
                renderer.draw(scene, ctx.screen, size, show_hud)
                pygame.display.flip()
            samples = min(4, ctx.max_samples)
            offscreen = ctx.simple_framebuffer(size, samples=samples)
            renderer.draw(scene, offscreen, size, show_hud)
            if screenshot:
                resolved = ctx.simple_framebuffer(size)
                ctx.copy_framebuffer(resolved, offscreen)
                _save(resolved, size, screenshot)
            else:
                ctx.finish()
            return

        controls = SpaceControls(scene, renderer.hud, show_hud)
        initial_size = pygame.display.get_window_size()
        density = tuple(px / pt for px, pt in zip(ctx.screen.size, initial_size))
        clock = pygame.time.Clock()
        while True:
            seconds = clock.tick(60) / 1000
            logical_size = pygame.display.get_window_size()
            size = tuple(max(1, round(pt * ratio)) for pt, ratio in zip(logical_size, density))
            for e in pygame.event.get():
                if not controls.event(e, logical_size, size):
                    return
            keys = pygame.key.get_pressed()
            scene.update(seconds, (keys[pygame.K_d] - keys[pygame.K_a], keys[pygame.K_e] - keys[pygame.K_q],
                                   keys[pygame.K_w] - keys[pygame.K_s], 4 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1))
            hover = None
            if (controls.show_hud and scene.camera_mode == "overview" and controls.press is None
                    and pygame.mouse.get_focused()):
                hover = controls.pick(pygame.mouse.get_pos(), logical_size, size)
            renderer.draw(scene, ctx.screen, size, controls.show_hud, hover)
            pygame.display.flip()
    finally:
        for resource in (resolved, offscreen, renderer, ctx):
            if resource is not None:
                resource.release()
        pygame.quit()
