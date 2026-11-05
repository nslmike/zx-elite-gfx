"""Окно и внеэкранный рендер космического атласа на OpenGL Core."""

import os
import sys
from collections import OrderedDict
from math import hypot, isfinite, tau

import moderngl
import numpy as np
import pygame

from .space_hud import SpaceHUD
from .space_look import SUN_DIRECTION, SpaceLook
from .space_assets import AssetCatalog, AssetError
from .space_resolution import RENDER_RESOLUTIONS
from .space_scene import SpaceScene
from .space_zx import SpaceZXRenderer, ZX_SIZE


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
_SCREEN_VERTEX = """#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(2.0 * p - 1.0, 0.0, 1.0);
}
"""
_ZX_FRAGMENT = """#version 330
uniform sampler2D bitmap;
in vec2 uv;
out vec4 color;
void main() { color = vec4(vec3(texture(bitmap, uv).r), 1.0); }
"""
_PRESENT_FRAGMENT = """#version 330
uniform sampler2D frame;
in vec2 uv;
out vec4 color;
void main() { color = texture(frame, uv); }
"""


def _matrix(matrix):
    return np.ascontiguousarray(matrix.T, dtype="f4").tobytes()


class MeshBuffer:
    """Один VBO, диапазон треугольников для каждого авторского материала."""

    def __init__(self, ctx, program, depth_program, vertices, imported):
        self.parts = []
        if imported:
            triangles = vertices.reshape(-1, 3, 9)
            slots = triangles[:, 0, 8].astype(int)
            order = np.argsort(slots, kind="stable")
            vertices = triangles[order]
            ids, starts, counts = np.unique(slots[order], return_index=True, return_counts=True)
            self.parts = [(int(i), int(s * 3), int(n * 3)) for i, s, n in zip(ids, starts, counts)]
        self.nbytes = vertices.nbytes
        self.buffer = ctx.buffer(vertices)
        self.vao = ctx.vertex_array(program, [(self.buffer, "3f 3f 2f 1f",
                                    "in_position", "in_normal", "in_uv", "in_material")])
        self.depth_vao = ctx.vertex_array(depth_program, [(self.buffer, "3f 24x", "position")])

    def release(self):
        self.depth_vao.release()
        self.vao.release()
        self.buffer.release()


class ZXScreen:
    """Увеличение настоящего CPU bitmap ближайшим соседом, без MSAA линий."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.renderer = SpaceZXRenderer()
        self.program = ctx.program(vertex_shader=_SCREEN_VERTEX, fragment_shader=_ZX_FRAGMENT)
        self.program["bitmap"].value = 0
        self.quad = ctx.vertex_array(self.program, [])
        self.texture = ctx.texture(ZX_SIZE, 1, dtype="f1")
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = self.texture.repeat_y = False

    def draw(self, scene, size, vp):
        frame = self.renderer.frame(scene, size, vp)
        self.texture.write(frame.pixels[::-1].tobytes(), alignment=1)
        self.ctx.viewport = frame.viewport
        self.ctx.enable_only(0)
        self.texture.use(0)
        self.quad.render(vertices=3)

    def release(self):
        self.texture.release()
        self.quad.release()
        self.program.release()


class _RenderTarget:
    """Буфер мира и текстура того же размера для разрешения MSAA."""

    def __init__(self, ctx, size):
        self.size = size
        self.resources = []
        samples = min(4, ctx.max_samples)
        try:
            self.texture = ctx.texture(size, 4)
            self.resources.append(self.texture)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.texture.repeat_x = self.texture.repeat_y = False
            depth = ctx.depth_renderbuffer(size, samples=samples)
            self.resources.append(depth)
            if samples:
                color = ctx.renderbuffer(size, 4, samples=samples)
                self.resources.append(color)
                self.resolve_frame = ctx.framebuffer(self.texture)
                self.resources.append(self.resolve_frame)
            else:
                color = self.texture
                self.resolve_frame = None
            self.draw_frame = ctx.framebuffer(color, depth)
            self.resources.append(self.draw_frame)
            error = ctx.error
            if error != "GL_NO_ERROR":
                raise moderngl.Error(error)
        except Exception:
            self.release()
            raise

    def release(self):
        for resource in reversed(self.resources):
            resource.release()


class SpaceRenderer:
    def __init__(self, ctx, scene):
        self.ctx = ctx
        # Внеэкранные кадры используют пиксели 1:1. Оконный цикл выставляет
        # плотность текущего drawable, чтобы размер HUD в пунктах не менялся.
        self.pixel_ratio = 1.0
        self.look = SpaceLook(ctx)
        self.hud = SpaceHUD(ctx)
        self.zx = None
        self._world_target = None
        self._present_program = ctx.program(vertex_shader=_SCREEN_VERTEX, fragment_shader=_PRESENT_FRAGMENT)
        self._present_program["frame"].value = 0
        self._present = ctx.vertex_array(self._present_program, [])
        info = ctx.info
        limit = min(info["GL_MAX_TEXTURE_SIZE"], info["GL_MAX_RENDERBUFFER_SIZE"])
        max_width, max_height = (min(limit, axis) for axis in info["GL_MAX_VIEWPORT_DIMS"])
        self.hud.resolution.unavailable = {index for index, (w, h) in enumerate(RENDER_RESOLUTIONS[1:], 1)
                                           if w > max_width or h > max_height}
        # Кеш разделяет только геометрию: ливрея и авторские материалы всегда
        # выбираются из текущего ShipAsset, в том числе у торговца и пирата.
        self.buffers = OrderedDict()
        self._mesh_budget = 64 * 1024 * 1024
        self._depth_program = ctx.program(vertex_shader=_DEPTH_VERTEX,
                                          fragment_shader="#version 330\nvoid main() {}")
        resolution = min(3072, ctx.info["GL_MAX_TEXTURE_SIZE"])
        self.shadow = ctx.depth_texture((resolution, resolution))
        self.shadow.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow.repeat_x = self.shadow.repeat_y = False
        self.shadow.compare_func = "<="
        self._shadow_target = ctx.framebuffer(depth_attachment=self.shadow)
        sun = np.array(SUN_DIRECTION)
        right = np.cross((0, 1, 0), sun)
        right /= np.linalg.norm(right)
        self._light_axes = np.array((right, np.cross(sun, right), -sun))
        self._load_fleet(scene)
        # Разметка обзора лежит в мировой плоскости расположения экспонатов.
        angles = (np.arange(220)[:, None] + np.array((0, .55))) * tau / 220
        circle = np.stack((np.sin(angles), np.zeros_like(angles), np.cos(angles)), axis=-1)
        self.orbit_buf = ctx.buffer((circle * scene.ring_radius).astype("f4"))
        self.orbit_program = ctx.program(vertex_shader=_ORBIT_VERTEX, fragment_shader=_ORBIT_FRAGMENT)
        self.orbit = ctx.simple_vertex_array(self.orbit_program, self.orbit_buf, "position")

    def prepare_resolution(self, size):
        """Сменить буфер только после успешного создания нового."""
        old = self._world_target
        if size == getattr(old, "size", None):
            return None
        try:
            if size is not None:
                self.ctx.clear_errors()
            target = _RenderTarget(self.ctx, size) if size is not None else None
        except (moderngl.Error, MemoryError) as exc:
            self.ctx.clear_errors()
            print(f"OpenGL · буфер {size[0]}×{size[1]} · {exc}", file=sys.stderr)
            return f"Не удалось включить {size[0]} × {size[1]}. Выберите другой размер."
        self._world_target = target
        if old is not None:
            old.release()
        return None

    def _mesh_buffer(self, ship, lod):
        key = ("mesh", ship.meshes[lod]) if ship.meshes else ("elite", ship.model_id)
        if key not in self.buffers:
            self.buffers[key] = MeshBuffer(self.ctx, self.look.hull, self._depth_program,
                                           ship.vertices(lod), bool(ship.meshes))
        self.buffers.move_to_end(key)
        return key, self.buffers[key]

    def _trim_meshes(self, protected):
        total = sum(buf.nbytes for buf in self.buffers.values())
        for key in tuple(self.buffers):
            if total <= self._mesh_budget:
                break
            if key not in protected:
                buf = self.buffers.pop(key)
                total -= buf.nbytes
                buf.release()

    def _load_fleet(self, scene):
        if scene.model_source == "zx":
            # Каркас берётся из исходных таблиц; ни один треугольный VBO
            # не нужен для его рёбер. Цветные ресурсы остаются в общем кеше M.
            self.meshes = []
            self._mesh_keys = set()
            self._shadow_objects = self._shadow_signature = None
            self.source = scene.model_source
            return
        meshes, used = [], set()
        previous = set(self.buffers)
        try:
            for ship in scene.ships:
                levels = []
                for lod in range(len(ship.meshes) or 1):
                    key, buf = self._mesh_buffer(ship, lod)
                    used.add(key)
                    levels.append(buf)
                meshes.append(levels)
        except Exception:
            for key in set(self.buffers) - previous:
                self.buffers.pop(key).release()
            raise
        # Активный флот не вытесняется; старые наборы живут только в пределах
        # бюджета. Смена M не пересоздаёт одинаковые VBO/VAO и карты теней.
        self._mesh_keys = used
        self._trim_meshes(used)
        self.meshes = meshes
        self._shadow_objects = self._shadow_signature = None
        self.source = scene.model_source

    def warm_sources(self, scene):
        """Подготовить локальные наборы до запуска часов оконного облёта.

        Сцена и активный флот не меняются. При ручной смене наборов сохраняем
        ленивую загрузку: чужой необязательный pack не нужен для запуска.
        """
        if not getattr(scene, "auto_sources", False):
            return
        # Только собственные ассеты пакетов: mixed и подстановки классических
        # кораблей не добавляют ресурсов и не требуют повторного обхода.
        fleets = [scene.catalog.classic]
        fleets.extend(scene.catalog.pack(source).values() for source in scene.catalog.pack_ids)
        for ships in fleets:
            for ship in ships:
                for lod in range(len(ship.meshes) or 1):
                    self._mesh_buffer(ship, lod)
                    self._trim_meshes(self._mesh_keys)
                if scene.texture_source == "author":
                    for material in ship.materials:
                        self.look.use_material(material)
        self.ctx.finish()

    @staticmethod
    def _lod(levels, diameter_px):
        return min(len(levels) - 1, 2 if diameter_px < 80 else 1 if diameter_px < 220 else 0)

    def _update_shadow(self, scene, visible):
        dynamic = scene.is_flyby
        # Экспонаты обычного атласа неизменяемы; движение камеры не требует
        # ни прохода глубины, ни пересчёта световой матрицы.
        if not dynamic and self._shadow_objects is scene.objects:
            return
        axes = self._light_axes
        centers = np.array([obj.position for obj in scene.objects]) @ axes.T
        radii = np.array([obj.radius for obj in scene.objects])[:, None] + .1
        lo, hi = (centers - radii).min(axis=0), (centers + radii).max(axis=0)
        extent = np.maximum(hi - lo, .001)
        light_vp = np.eye(4)
        light_vp[:3, :3] = axes * (2 / extent)[:, None]
        light_vp[:3, 3] = -(hi + lo) / extent
        lods = [0] * len(self.meshes)
        if dynamic:
            # Тень мелкого/дальнего объекта не нуждается в деталях, меньших
            # текселя атласа. Видимый корпус никогда не получает более грубую
            # самотень, чем его текущая геометрия. Вне кадра остаются кастеры:
            # они всё ещё способны затенять соседний видимый корабль.
            scale = self.shadow.size[0] / min(extent[:2])
            lods = [self._lod(levels, 2 * obj.radius * scale)
                    for obj, levels in zip(scene.objects, self.meshes)]
            for index, lod in visible:
                lods[index] = min(lods[index], lod)
        transforms = tuple(_matrix(obj.matrix) for obj in scene.objects)
        signature = (_matrix(light_vp), transforms, tuple(lods))
        if signature != self._shadow_signature:
            ctx = self.ctx
            self._shadow_target.use()
            ctx.viewport = (0, 0, *self.shadow.size)
            self._shadow_target.clear(depth=1.0)
            ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            ctx.front_face = "ccw"
            # Дальняя оболочка не затеняет собственный освещённый фасет.
            ctx.cull_face = "front"
            for obj, levels, lod in zip(scene.objects, self.meshes, lods):
                self._depth_program["light_mvp"].write(_matrix(light_vp @ obj.matrix))
                levels[lod].depth_vao.render()
            ctx.cull_face = "back"
            self.look.hull["u_shadow_vp"].write(_matrix(light_vp))
            self.look.hull["u_shadow_enabled"].value = True
            self._shadow_signature = signature
        self._shadow_objects = scene.objects

    def draw(self, scene, target, size, show_hud, hover=None):
        ctx = self.ctx
        self.hud.resolution.output_size = size
        if self.source != scene.model_source:
            self._load_fleet(scene)
        # Размер окна определяет камеру и видимую область. Другие пропорции
        # внутреннего буфера меняют плотность выборок, а не форму изображения.
        vp = scene.view_projection(size[0] / size[1])
        if scene.model_source == "zx":
            target.use()
            ctx.viewport = (0, 0, *size)
            target.clear(0., 0., 0., 1., depth=1., viewport=ctx.viewport)
            if self.zx is None:
                self.zx = ZXScreen(ctx)
            self.zx.draw(scene, size, vp)
            ctx.viewport = (0, 0, *size)
        else:
            world = self._world_target
            if world is None:
                self._draw_world(scene, target, size, vp, show_hud)
            else:
                self._draw_world(scene, world.draw_frame, world.size, vp, show_hud)
                if world.resolve_frame is not None:
                    ctx.copy_framebuffer(world.resolve_frame, world.draw_frame)
                target.use()
                ctx.viewport = (0, 0, *size)
                ctx.enable_only(0)
                world.texture.use(0)
                self._present.render(vertices=3)
        if show_hud:
            self.hud.draw(scene, size, hover, vp, pixel_ratio=self.pixel_ratio)

    def _draw_world(self, scene, target, size, vp, show_hud):
        ctx = self.ctx
        planes = np.vstack((vp[3] + vp[:3], vp[3] - vp[:3]))
        planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
        focal = size[1] * np.linalg.norm(vp[1, :3])
        visible = []
        for index, (obj, levels) in enumerate(zip(scene.objects, self.meshes)):
            if np.any(planes[:, :3] @ obj.position + planes[:, 3] < -obj.radius):
                continue
            depth = vp[3, :3] @ obj.position + vp[3, 3]
            diameter_px = focal * obj.radius / max(.001, depth)
            visible.append((index, self._lod(levels, diameter_px)))
        self._update_shadow(scene, visible)
        target.use()
        ctx.viewport = (0, 0, *size)
        target.clear(0.01, 0.02, 0.03, 1.0, depth=1.0, viewport=ctx.viewport)
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
        for index, lod in visible:
            ship, obj = scene.ships[index], scene.objects[index]
            mesh = self.meshes[index][lod]
            authored = bool(ship.materials) and scene.texture_source == "author"
            hull["u_model"].write(_matrix(obj.matrix))
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

    def release(self):
        if self.zx is not None:
            self.zx.release()
        if self._world_target is not None:
            self._world_target.release()
        self._present.release()
        self._present_program.release()
        self.orbit.release()
        self.orbit_buf.release()
        self.orbit_program.release()
        self._shadow_target.release()
        self.shadow.release()
        for buf in self.buffers.values():
            buf.release()
        self._depth_program.release()
        self.hud.release()
        self.look.release()


class _CocoaDrawable:
    """Retina для существующего окна pygame, без замены SDL и его контекста.

    pygame 2.6 display.set_mode не запрашивает SDL_WINDOW_ALLOW_HIGHDPI.
    AppKit разрешает включить настоящий backing у уже созданного GL view.
    После этого SDL всё ещё сообщает размер в пунктах: конвертируем через
    сам view, как требует Apple, заново после resize/смены монитора.
    """

    def __init__(self):
        import ctypes as ct

        class Size(ct.Structure):
            _fields_ = (("width", ct.c_double), ("height", ct.c_double))

        self._size_type = Size
        objc = ct.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = (ct.c_char_p,)
        objc.objc_getClass.restype = ct.c_void_p
        objc.sel_registerName.argtypes = (ct.c_char_p,)
        objc.sel_registerName.restype = ct.c_void_p
        selector = objc.sel_registerName
        pointer = ct.CFUNCTYPE(ct.c_void_p, ct.c_void_p, ct.c_void_p)(("objc_msgSend", objc))
        self._call = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_void_p)(("objc_msgSend", objc))
        set_bool = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_void_p, ct.c_bool)(("objc_msgSend", objc))
        # NSSize содержит два CGFloat: возвращается в регистрах как на Intel,
        # так и на ARM64. NSRect потребовал бы отдельного ABI stret на Intel.
        self._convert = ct.CFUNCTYPE(Size, ct.c_void_p, ct.c_void_p, Size)(("objc_msgSend", objc))
        self._update = selector(b"update")
        self._convert_size = selector(b"convertSizeToBacking:")
        self._context = pointer(objc.objc_getClass(b"NSOpenGLContext"), selector(b"currentContext"))
        self._view = pointer(self._context, selector(b"view"))
        if not self._view:
            raise RuntimeError("Не найден Cocoa view оконного OpenGL-контекста")
        set_bool(self._view, selector(b"setWantsBestResolutionOpenGLSurface:"), True)
        self._call(self._context, self._update)
        self._previous = None

    def __call__(self, logical_size):
        pixels = self._convert(self._view, self._convert_size,
                               self._size_type(*logical_size))
        size = (round(pixels.width), round(pixels.height))
        if size != self._previous:
            self._call(self._context, self._update)
            self._previous = size
        return size


def _sdl_drawable_query():
    """Используем именно уже загруженный SDL, а не другую системную копию."""
    import ctypes as ct

    try:
        sdl = ct.CDLL(pygame.base.__file__)
        current = sdl.SDL_GL_GetCurrentWindow
        current.argtypes = ()
        current.restype = ct.c_void_p
        drawable = sdl.SDL_GL_GetDrawableSize
        drawable.argtypes = (ct.c_void_p, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int))
        drawable.restype = None
    except (OSError, AttributeError):
        # На платформах, где зависимые символы .pyd недоступны через dlsym,
        # обычное окно pygame без ALLOW_HIGHDPI имеет один пиксель на пункт.
        return lambda logical: logical

    def size(logical):
        width, height = ct.c_int(), ct.c_int()
        drawable(current(), ct.byref(width), ct.byref(height))
        return width.value, height.value

    return size


def _window_drawable_query():
    if sys.platform == "darwin" and pygame.display.get_driver() == "cocoa":
        try:
            return _CocoaDrawable()
        except (OSError, AttributeError, RuntimeError) as exc:
            print(f"Retina недоступна: {exc}. Используется обычный буфер SDL.", flush=True)
    return _sdl_drawable_query()


def _window_sizes(ctx):
    logical = pygame.display.get_window_size()
    # SDL может сообщать старый положительный размер и после minimize.
    # На Cocoa у такого окна уже нет доступной поверхности для GL draw/flip.
    # get_active проверяет SHOWN/MINIMIZED, а не наличие фокуса клавиатуры.
    if not pygame.display.get_active():
        return logical, None
    drawable = ctx._window_drawable(logical)
    # Свёрнутое окно на некоторых драйверах временно не имеет drawable.
    # Ввод и выход из приложения продолжают работать, рисование ждёт restore.
    return logical, drawable if min(*logical, *drawable) > 0 else None


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
    ctx = moderngl.create_context(require=330)
    ctx._window_drawable = _window_drawable_query()
    _, size = _window_sizes(ctx)
    if size is not None:
        ctx.viewport = (0, 0, *size)
    return ctx


class SpaceControls:
    """Общие удержания клавиш и управление камерой в обоих входах."""

    _angular_speed = 1.6
    _movement_keys = (pygame.K_a, pygame.K_d, pygame.K_q, pygame.K_e, pygame.K_w,
                      pygame.K_s, pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT,
                      pygame.K_UP, pygame.K_DOWN)
    _shift_keys = (pygame.K_LSHIFT, pygame.K_RSHIFT)
    _arrow_keys = (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)
    _presentation = False

    def __init__(self, scene, hud, show_hud):
        self.scene = scene
        self.hud = hud
        self.show_hud = show_hud
        self.press = None
        self.dragging = False
        self.looking = False
        self._mouse_state = None
        self._held_keys = {}
        self._down_keys = set()
        self._flight_keys = set()
        self._view_keys = set()
        self.prepare_resolution = None
        self._resume_resolution_look = False

    def _start_look(self):
        if self.looking:
            return
        self.looking = True
        self.press = None
        self.dragging = False
        if not pygame.display.get_init():
            # Контроллер можно проверять с настоящей сценой без SDL-окна.
            return
        self._mouse_state = (pygame.event.get_grab(), pygame.mouse.get_visible())
        # Относительные перемещения остаются в логических px, включая Retina.
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()

    def _stop_look(self):
        self.looking = False
        if self._mouse_state is not None:
            grabbed, visible = self._mouse_state
            self._mouse_state = None
            pygame.event.set_grab(grabbed)
            pygame.mouse.set_visible(visible)

    def _clear_holds(self):
        self.press = None
        self.dragging = False
        self._held_keys.clear()
        self._flight_keys.clear()
        self._view_keys.clear()

    def _release_motion(self):
        self._stop_look()
        self._clear_holds()

    def release(self):
        """Снять захват и удержания при потере фокуса и выходе из цикла."""
        self._release_motion()
        self._down_keys.clear()
        self._resume_resolution_look = False

    def _drag_look(self, dx, dy):
        if self._presentation:
            self.scene.look_drag(dx, dy)
        else:
            self.scene.drag(dx, dy)

    def _mode(self, mode):
        if mode.startswith("menu_"):
            menu = mode.removeprefix("menu_")
            self.hud.menu = None if self.hud.menu == menu else menu
            return
        self.hud.menu = None
        if mode == "cycle_flight_profile":
            if self.scene.camera_mode == "flight":
                self._clear_holds()
                self.scene.cycle_flight_profile()
        elif mode.startswith("source_"):
            self.scene.set_source(mode.removeprefix("source_"))
        elif mode.startswith("texture_"):
            self.scene.texture_source = mode.removeprefix("texture_")
        elif mode == "overview":
            self._release_motion()
            self.scene.overview()
        elif mode == "inspect":
            self._release_motion()
            self.scene.select(self.scene.selected)
        elif mode == "flight":
            self._release_motion()
            self.scene.fly()
            self._start_look()

    def _toggle_resolution(self):
        if self.hud.menu == "resolution":
            self.hud.menu = None
            if self._resume_resolution_look and self.scene.camera_mode == "flight":
                self._start_look()
            self._resume_resolution_look = False
        else:
            state = self.hud.resolution
            state.highlighted = state.selected
            state.error = None
            self._resume_resolution_look = self.looking and self.scene.camera_mode == "flight"
            self._release_motion()
            self.show_hud = True
            self.hud.menu = "resolution"

    def _select_resolution(self, index):
        state = self.hud.resolution
        size = RENDER_RESOLUTIONS[index]
        if index in state.unavailable:
            state.error = f"{size[0]} × {size[1]} недоступно на этой видеокарте."
            return
        state.highlighted = index
        state.error = self.prepare_resolution(size) if self.prepare_resolution is not None else None
        if state.error is None:
            state.selected = index
            self._toggle_resolution()

    def _resolution_key(self, key):
        state = self.hud.resolution
        if key == pygame.K_ESCAPE:
            self._toggle_resolution()
        elif key in (pygame.K_UP, pygame.K_DOWN):
            options = [index for index in range(len(RENDER_RESOLUTIONS)) if index not in state.unavailable]
            delta = 1 if key == pygame.K_DOWN else -1
            state.highlighted = options[(options.index(state.highlighted) + delta) % len(options)]
            state.error = None
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._select_resolution(state.highlighted)

    def _tab_action(self, point, logical_size, drawable_size):
        if not self.show_hud:
            return None
        action = self.hud.tab_at(self._pixel(point, logical_size, drawable_size))
        if self.hud.menu == "resolution" and not (
                action and (action == "menu_resolution" or action.startswith("resolution_"))):
            return "close_menu"
        return action

    def _activate_tab(self, action):
        if action == "menu_resolution":
            self._toggle_resolution()
        elif self.hud.menu == "resolution":
            if action == "close_menu":
                self._toggle_resolution()
            else:
                self._select_resolution(int(action.removeprefix("resolution_")))
        else:
            self._mode(action)

    def pick(self, point, logical_size, drawable_size):
        if min(*logical_size, *drawable_size) <= 0:
            return None
        if self.show_hud:
            pixel = self._pixel(point, logical_size, drawable_size)
            if self.hud.tab_at(pixel):
                return None
            if self.scene.camera_mode == "overview":
                marker = self.hud.marker_at(pixel)
                if marker is not None:
                    return marker
        # Камера берёт пропорции drawable. Масштабируем логическую ось X
        # на случай округлений Retina, сохраняя логический допуск для выбора.
        width, height = logical_size
        pick_width = height * drawable_size[0] / drawable_size[1]
        return self.scene.pick(point[0] * pick_width / width, point[1], pick_width, height)

    @staticmethod
    def _pixel(point, logical_size, drawable_size):
        return tuple(round(coord * draw / max(logical, 1))
                     for coord, draw, logical in zip(point, drawable_size, logical_size))

    def _key_down(self, e):
        scene = self.scene
        flight = scene.camera_mode == "flight"
        if e.key == pygame.K_f:
            self._mode("flight")
        elif flight and e.key in (*self._movement_keys, *self._shift_keys):
            self._flight_keys.add(e.key)
        elif flight and e.key == pygame.K_c:
            self._mode("cycle_flight_profile")
        elif not flight and e.key in self._shift_keys:
            # Shift, нажатый после стрелки, прекращает перемотку. После его
            # отпускания удержанная стрелка не становится командой маршрута.
            for key in self._arrow_keys:
                if key in self._down_keys:
                    self._held_keys.pop(key, None)
                    self._view_keys.add(key)
        elif (not flight and e.key in self._arrow_keys
              and (self._down_keys.intersection(self._shift_keys)
                   or getattr(e, "mod", 0) & pygame.KMOD_SHIFT)):
            self._view_keys.add(e.key)
        elif e.key == pygame.K_ESCAPE:
            if self.looking:
                self._release_motion()
            elif self.hud.menu:
                self.hud.menu = None
            elif self._presentation or scene.camera_mode == "overview":
                self.release()
                return False
            else:
                self._mode("overview")
        elif e.key == pygame.K_m:
            self.hud.menu = None
            scene.cycle_source()
        elif e.key == pygame.K_t:
            self.hud.menu = None
            scene.cycle_textures()
        elif e.key == pygame.K_i:
            self.show_hud = True
            self._mode("menu_credits")
        elif e.key == pygame.K_h:
            self.show_hud = not self.show_hud
            self.hud.menu = None
        elif not self._presentation and not flight and e.key in (pygame.K_LEFT, pygame.K_RIGHT):
            scene.select(scene.selected + (1 if e.key == pygame.K_RIGHT else -1),
                         focus=scene.camera_mode == "inspect")
        elif not self._presentation and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._mode("inspect")
        elif not self._presentation and e.key in (pygame.K_o, pygame.K_HOME):
            self._mode("overview")
        elif not self._presentation and e.key == pygame.K_TAB:
            self._mode("inspect" if scene.camera_mode == "overview" else "overview")
        elif (flight or not self._presentation) and e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            scene.zoom(1)
        elif (flight or not self._presentation) and e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            scene.zoom(-1)
        return True

    def event(self, e, logical_size, drawable_size):
        scene = self.scene
        resolution_menu = self.hud.menu == "resolution"
        if e.type == pygame.QUIT:
            self.release()
            return False
        if e.type == pygame.WINDOWFOCUSLOST:
            if scene.camera_mode == "flight":
                scene.flight_speed = 0.0
            self.release()
        elif e.type == pygame.KEYDOWN:
            # Команды выполняются один раз до KEYUP; запоздалый SDL repeat
            # после смены режима или потери фокуса не возобновляет движение.
            if e.key in self._down_keys or getattr(e, "repeat", False):
                return True
            self._down_keys.add(e.key)
            if e.key == pygame.K_g:
                self._toggle_resolution()
            elif resolution_menu:
                self._resolution_key(e.key)
            else:
                return self._key_down(e)
        elif e.type == pygame.KEYUP:
            self._held_keys.pop(e.key, None)
            self._down_keys.discard(e.key)
            self._flight_keys.discard(e.key)
            self._view_keys.discard(e.key)
            if e.key in self._shift_keys and not self._down_keys.intersection(self._shift_keys):
                self._view_keys.clear()
        elif resolution_menu and (e.type == pygame.MOUSEWHEEL or
                                  (e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and e.button != 1)):
            return True
        elif e.type == pygame.MOUSEWHEEL:
            if scene.camera_mode == "flight" or not self._presentation:
                scene.zoom(getattr(e, "precise_y", e.y))
        elif (e.type == pygame.MOUSEBUTTONDOWN and scene.camera_mode == "flight" and e.button in (1, 3)
              and not resolution_menu):
            if self.looking:
                return True
            action = self._tab_action(e.pos, logical_size, drawable_size)
            if action is None:
                self._start_look()
            elif e.button == 1:
                self.press = (e.button, e.pos, action)
                self.dragging = False
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3 and self._presentation:
            pixel = self._pixel(e.pos, logical_size, drawable_size)
            if not (self.show_hud and self.hud.tab_at(pixel)):
                self._start_look()
        elif (e.type == pygame.MOUSEBUTTONUP and e.button == 3 and self.looking
              and scene.camera_mode != "flight"):
            self._stop_look()
        elif e.type == pygame.MOUSEMOTION and self.looking:
            self._drag_look(*e.rel)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
            action = self._tab_action(e.pos, logical_size, drawable_size)
            self.press = (e.button, e.pos, action)
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.press:
            if self.press[2]:
                self.dragging |= hypot(e.pos[0] - self.press[1][0], e.pos[1] - self.press[1][1]) >= 4
                return True
            if self._presentation:
                self.dragging |= hypot(e.pos[0] - self.press[1][0], e.pos[1] - self.press[1][1]) >= 4
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
                tab = self._tab_action(e.pos, logical_size, drawable_size)
                if tab and tab == start_action:
                    self._activate_tab(tab)
                elif not self._presentation and not (tab or start_action):
                    ident = self.pick(e.pos, logical_size, drawable_size)
                    if ident is not None:
                        self._release_motion()
                        scene.select(ident)
            self.dragging = False
        return True

    def _update_view(self, seconds):
        if not self._view_keys or not isfinite(seconds) or seconds <= 0:
            return
        keys = self._view_keys
        scale = self._angular_speed * min(seconds, .1) / .005
        dx = (int(pygame.K_RIGHT in keys) - int(pygame.K_LEFT in keys)) * scale
        dy = (int(pygame.K_DOWN in keys) - int(pygame.K_UP in keys)) * scale
        if dx or dy:
            self._drag_look(dx, dy)

    def update(self, seconds):
        seconds = float(seconds)
        if self.scene.camera_mode != "flight":
            self._update_view(seconds)
            self.scene.update(seconds)
            return
        keys = self._flight_keys
        axis = lambda positive, negative: int(positive in keys) - int(negative in keys)
        pitch = axis(pygame.K_UP, pygame.K_DOWN)
        if self.scene.flight_profile == "elite":
            movement = (0, 0, axis(pygame.K_w, pygame.K_s), 1)
            yaw, roll = 0, axis(pygame.K_RIGHT, pygame.K_LEFT)
        else:
            down = bool(keys.intersection(self._shift_keys))
            movement = (axis(pygame.K_d, pygame.K_a), int(pygame.K_SPACE in keys) - down,
                        axis(pygame.K_w, pygame.K_s), 1)
            yaw, roll = axis(pygame.K_RIGHT, pygame.K_LEFT), axis(pygame.K_e, pygame.K_q)
        if (yaw or pitch or roll) and isfinite(seconds) and seconds > 0:
            scale = self._angular_speed * min(seconds, .1)
            self.scene.turn_flight(yaw * scale, pitch * scale, roll * scale)
        self.scene.update(seconds, movement)


class FlybyControls(SpaceControls):
    """Автопролёт с перемоткой и свободный полёт из текущей точки камеры."""

    _scrub_delay = .25
    _scrub_speed = 4.0
    _presentation = True

    def _mode(self, action):
        if action == "menu_credits":
            self.hud.menu = None if self.hud.menu == "credits" else "credits"
            return
        self.hud.menu = None
        if action in ("toggle_flight", "flight"):
            # Клавиши остаются в _down_keys до KEYUP: удержанный F/A или
            # запоздалый repeat не запускает команду уже в другом режиме.
            self._release_motion()
            self.scene.toggle_flight()
            if self.scene.camera_mode == "flight":
                self._start_look()
        elif action == "cycle_flight_profile":
            super()._mode(action)
        elif action == "cycle_source":
            self.scene.cycle_source()
        elif action == "cycle_textures":
            self.scene.cycle_textures()
        elif action == "toggle_auto" and self.scene.camera_mode == "flyby":
            self.scene.toggle_auto_sources()
        elif action == "toggle_pause" and self.scene.camera_mode == "flyby":
            self.scene.toggle_pause()
        elif action == "scrub_back" and self.scene.camera_mode == "flyby":
            self.scene.scrub(-1.0)
        elif action == "scrub_forward" and self.scene.camera_mode == "flyby":
            self.scene.scrub(1.0)

    def _key_down(self, e):
        flight = self.scene.camera_mode == "flight"
        shifted = (self._down_keys.intersection(self._shift_keys)
                   or getattr(e, "mod", 0) & pygame.KMOD_SHIFT)
        if not flight and e.key in (pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT) and not (
                e.key in self._arrow_keys and shifted):
            self._held_keys[e.key] = 0.0
            self._mode({pygame.K_SPACE: "toggle_pause", pygame.K_LEFT: "scrub_back",
                        pygame.K_RIGHT: "scrub_forward"}[e.key])
        elif not flight and e.key == pygame.K_a:
            self._mode("toggle_auto")
        elif not flight and e.key == pygame.K_r:
            self.scene.reset_look()
        else:
            return super()._key_down(e)
        return True

    def update(self, seconds):
        if self.scene.camera_mode == "flight":
            super().update(seconds)
            return
        self._update_view(float(seconds))
        left = self._held_keys.get(pygame.K_LEFT)
        right = self._held_keys.get(pygame.K_RIGHT)
        if left is None and right is None:
            self.scene.update(seconds)
            return
        seconds = float(seconds)
        if not 0 <= seconds < float("inf"):
            # Сцена сохраняет общую проверку конечного dt и обработку dt < 0.
            self.scene.update(seconds, advance_route=False)
            return
        if left is not None:
            self._held_keys[pygame.K_LEFT] = left + seconds
        if right is not None:
            self._held_keys[pygame.K_RIGHT] = right + seconds
        if left is not None and right is not None:
            # Обе стрелки останавливают маршрут, не меняя явную паузу Space.
            self.scene.update(seconds, advance_route=False)
            return
        key, held = (pygame.K_LEFT, left) if left is not None else (pygame.K_RIGHT, right)
        stopped = min(seconds, max(0.0, self._scrub_delay - held))
        active = seconds - stopped
        if active > 0:
            direction = 1 if key == pygame.K_RIGHT else -1
            moved = self.scene.scrub(direction * self._scrub_speed * active)
            # На упоре в начало оставшееся реальное время идёт во вращения.
            stopped += max(0.0, active - abs(moved) / self._scrub_speed)
        if stopped > 0:
            # Сам scrub уже сдвинул фазы на пройденное время: не прибавляем
            # к активной перемотке ещё и обычный ход вращений этого кадра.
            self.scene.update(stopped, advance_route=False)


def _save(frame, size, path):
    rgb = frame.read(viewport=(0, 0, *size), components=3, alignment=1)
    surface = pygame.image.frombytes(rgb, size, "RGB", True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def run_space(model_id, screenshot, headless, show_hud, model_source="elite", texture_source="author", *,
              mode="space", source_interval=8.0, at=None):
    catalog = AssetCatalog()
    ships = catalog.classic if mode == "flyby" and model_source == "zx" else catalog.fleet(model_source)
    model_count = len(ships)
    if mode == "flyby":
        from .space_flyby import FlybyScene
        model_count = FlybyScene.slot_count
    if model_id is not None and model_id not in range(model_count):
        raise AssetError(f"Для {model_source} доступен диапазон 0–{model_count-1}. Список наборов: --list-sources")
    pygame.font.init()
    ctx = renderer = offscreen = resolved = controls = None
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
        if not headless:
            logical, pixels = _window_sizes(ctx)
            width, height = pixels or (0, 0)
            print(f"Окно {logical[0]}×{logical[1]} пт · буфер {width}×{height} пикс.", flush=True)
        if mode == "flyby":
            scene = FlybyScene(model_id, model_source=model_source, texture_source=texture_source,
                               catalog=catalog, source_interval=source_interval)
            if at is not None:
                scene.seek(at)
        else:
            scene = SpaceScene(model_source=model_source, texture_source=texture_source, catalog=catalog)
            if model_id is not None:
                scene.select(model_id)
        renderer = SpaceRenderer(ctx, scene)
        if mode == "flyby" and not headless and not screenshot:
            renderer.warm_sources(scene)
        if not headless:
            pygame.display.set_caption("Elite · Автопролёт" if mode == "flyby" else "Elite · Орбитальный каталог")
        if headless or screenshot:
            # Для чтения на Mac сводим выборки MSAA в отдельный буфер с одним
            # цветом на пиксель. Скриншот окна использует тот же контекст и кадр.
            size = _SIZE if headless else _window_sizes(ctx)[1]
            if size is None:
                raise SystemExit("Оконный буфер недоступен. Восстановите окно или используйте --headless для снимка.")
            if not headless:
                logical_size = pygame.display.get_window_size()
                renderer.pixel_ratio = min(px / pt for px, pt in zip(size, logical_size))
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

        controls = (FlybyControls if mode == "flyby" else SpaceControls)(scene, renderer.hud, show_hud)
        controls.prepare_resolution = renderer.prepare_resolution
        clock = pygame.time.Clock()
        while True:
            seconds = clock.tick(60) / 1000
            events = pygame.event.get()
            logical_size, size = _window_sizes(ctx)
            for e in events:
                if not controls.event(e, tuple(max(1, x) for x in logical_size), size or (1, 1)):
                    return
            controls.update(seconds)
            if size is None:
                continue
            renderer.pixel_ratio = min(px / pt for px, pt in zip(size, logical_size))
            hover = None
            if (controls.show_hud and scene.camera_mode == "overview" and controls.press is None
                    and pygame.mouse.get_focused()):
                hover = controls.pick(pygame.mouse.get_pos(), logical_size, size)
            renderer.draw(scene, ctx.screen, size, controls.show_hud, hover)
            pygame.display.flip()
    finally:
        if controls is not None:
            controls.release()
        for resource in (resolved, offscreen, renderer, ctx):
            if resource is not None:
                resource.release()
        pygame.quit()
