"""Окно и внеэкранный рендер космического атласа на OpenGL Core."""

import os
import sys
from collections import OrderedDict
from math import hypot, tau

import moderngl
import numpy as np
import pygame

from .space_hud import SpaceHUD
from .space_look import SUN_DIRECTION, SpaceLook
from .space_assets import AssetCatalog, AssetError
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
_ZX_VERTEX = """#version 330
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
        self.program = ctx.program(vertex_shader=_ZX_VERTEX, fragment_shader=_ZX_FRAGMENT)
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


class SpaceRenderer:
    def __init__(self, ctx, scene):
        self.ctx = ctx
        # Внеэкранные кадры используют пиксели 1:1. Оконный цикл выставляет
        # плотность текущего drawable, чтобы размер HUD в пунктах не менялся.
        self.pixel_ratio = 1.0
        self.look = SpaceLook(ctx)
        self.hud = SpaceHUD(ctx)
        self.zx = None
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
        if self.source != scene.model_source:
            self._load_fleet(scene)
        vp = scene.view_projection(size[0] / size[1])
        if scene.model_source == "zx":
            target.use()
            ctx.viewport = (0, 0, *size)
            target.clear(0., 0., 0., 1., depth=1., viewport=ctx.viewport)
            if self.zx is None:
                self.zx = ZXScreen(ctx)
            self.zx.draw(scene, size, vp)
            # HUD всегда относится к полному окну: размер текстуры и hitboxes
            # не зависят от вписанной игровой области с её квадратными пикселями.
            ctx.viewport = (0, 0, *size)
            if show_hud:
                self.hud.draw(scene, size, hover, vp, pixel_ratio=self.pixel_ratio)
            return
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
        if show_hud:
            self.hud.draw(scene, size, hover, vp, pixel_ratio=self.pixel_ratio)

    def release(self):
        if self.zx is not None:
            self.zx.release()
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

    def _movement_state(self):
        return pygame.key.get_pressed(), pygame.key.get_mods()

    def update(self, seconds):
        keys, modifiers = self._movement_state()
        self.scene.update(seconds, (keys[pygame.K_d] - keys[pygame.K_a], keys[pygame.K_e] - keys[pygame.K_q],
                                    keys[pygame.K_w] - keys[pygame.K_s],
                                    4 if modifiers & pygame.KMOD_SHIFT else 1))


class FlybyControls(SpaceControls):
    """Автопролёт с перемоткой и свободный полёт из текущей точки камеры."""

    _scrub_delay = .25
    _scrub_speed = 4.0
    _movement_keys = (pygame.K_a, pygame.K_d, pygame.K_q, pygame.K_e, pygame.K_w, pygame.K_s)
    _shift_keys = (pygame.K_LSHIFT, pygame.K_RSHIFT)

    def __init__(self, scene, hud, show_hud):
        super().__init__(scene, hud, show_hud)
        self.looking = False
        self._mouse_state = None
        self._held_keys = {}
        self._down_keys = set()
        self._flight_keys = set()

    def _start_look(self):
        if self.looking:
            return
        self.looking = True
        self.press = None
        self.dragging = False
        self._mouse_state = (pygame.event.get_grab(), pygame.mouse.get_visible())
        # SDL даёт относительные перемещения за краями окна при сочетании
        # grab и скрытого курсора. Поворот измеряем в логических px, не retina.
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

    def _release_motion(self):
        self._stop_look()
        self.press = None
        self.dragging = False
        self._held_keys.clear()
        self._flight_keys.clear()

    def release(self):
        """Снять захват и удержания при потере фокуса и выходе из цикла."""
        self._release_motion()
        self._down_keys.clear()

    def _movement_state(self):
        # После смены режима или фокуса движение начинается только с нового
        # нажатия. Сам расчёт осей и Shift остаётся общим с обычным атласом.
        keys = {key: key in self._flight_keys for key in self._movement_keys}
        modifiers = pygame.KMOD_SHIFT if self._flight_keys.intersection(self._shift_keys) else 0
        return keys, modifiers

    def _mode(self, action):
        if action == "menu_credits":
            self.hud.menu = None if self.hud.menu == "credits" else "credits"
            return
        self.hud.menu = None
        if action == "toggle_flight":
            # Клавиши остаются в _down_keys до KEYUP: удержанный F/A или
            # запоздалый repeat не запускает команду уже в другом режиме.
            self._release_motion()
            self.scene.toggle_flight()
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

    def event(self, e, logical_size, drawable_size):
        if e.type == pygame.QUIT:
            self.release()
            return False
        if e.type == pygame.WINDOWFOCUSLOST:
            self.release()
        elif e.type == pygame.KEYDOWN:
            # Удержания считаем по dt, команды выполняем один раз до KEYUP.
            # После потери фокуса запоздалый SDL repeat не начинает их снова.
            if e.key in self._down_keys or getattr(e, "repeat", False):
                return True
            self._down_keys.add(e.key)
            flight = self.scene.camera_mode == "flight"
            if e.key == pygame.K_f:
                self._mode("toggle_flight")
            elif flight and e.key in (*self._movement_keys, *self._shift_keys):
                self._flight_keys.add(e.key)
            elif not flight and e.key in (pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT):
                self._held_keys[e.key] = 0.0
                self._mode({pygame.K_SPACE: "toggle_pause", pygame.K_LEFT: "scrub_back",
                            pygame.K_RIGHT: "scrub_forward"}[e.key])
            elif e.key == pygame.K_ESCAPE:
                if self.hud.menu:
                    self.hud.menu = None
                else:
                    self.release()
                    return False
            elif e.key == pygame.K_h:
                self.show_hud = not self.show_hud
                self.hud.menu = None
            elif e.key == pygame.K_i:
                self.show_hud = True
                self._mode("menu_credits")
            elif e.key == pygame.K_r and not flight:
                self.scene.reset_look()
            elif flight and e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.scene.zoom(1)
            elif flight and e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.scene.zoom(-1)
            else:
                action = {pygame.K_m: "cycle_source", pygame.K_t: "cycle_textures",
                          pygame.K_a: "toggle_auto"}.get(e.key)
                if action:
                    self._mode(action)
        elif e.type == pygame.KEYUP:
            self._held_keys.pop(e.key, None)
            self._down_keys.discard(e.key)
            self._flight_keys.discard(e.key)
        elif e.type == pygame.MOUSEWHEEL and self.scene.camera_mode == "flight":
            self.scene.zoom(getattr(e, "precise_y", e.y))
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
            self._start_look()
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 3:
            self._stop_look()
        elif e.type == pygame.MOUSEMOTION and self.looking:
            self.scene.look_drag(*e.rel)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            pixel = self._pixel(e.pos, logical_size, drawable_size)
            self.press = (e.button, e.pos, self.hud.tab_at(pixel) if self.show_hud else None)
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.press:
            self.dragging |= hypot(e.pos[0] - self.press[1][0], e.pos[1] - self.press[1][1]) >= 4
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.press:
            start_action = self.press[2]
            self.press = None
            pixel = self._pixel(e.pos, logical_size, drawable_size)
            action = self.hud.tab_at(pixel) if self.show_hud else None
            if not self.dragging and action and action == start_action:
                self._mode(action)
            self.dragging = False
        return True

    def update(self, seconds):
        if self.scene.camera_mode == "flight":
            super().update(seconds)
            return
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
        if isinstance(controls, FlybyControls):
            controls.release()
        for resource in (resolved, offscreen, renderer, ctx):
            if resource is not None:
                resource.release()
        pygame.quit()
