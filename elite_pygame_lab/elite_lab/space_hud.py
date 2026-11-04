"""Экранные подписи космического атласа и выбор режима мышью."""

from math import hypot

import moderngl
import numpy as np
import pygame

from .space_assets import SOURCE_NAMES, TEXTURE_NAMES


_INK = (231, 236, 235)
_MUTED = (159, 181, 189)
_GOLD = (244, 183, 104)
_TABS = (("overview", "O", "Карта"), ("inspect", "Enter", "Осмотр"), ("flight", "F", "Полёт"))
_CLASSES = ("ОРБИТАЛЬНАЯ СТАНЦИЯ", "РАКЕТА", "СПАСАТЕЛЬНАЯ КАПСУЛА", "ОБЛОМОК",
            "ГРУЗОВОЙ КОНТЕЙНЕР", "КАМЕННАЯ ГЛЫБА", "АСТЕРОИД", "СТАНЦИЯ В АСТЕРОИДЕ",
            "ПЕРЕХВАТЧИК", "МНОГОЦЕЛЕВОЙ КОРАБЛЬ", "ГРУЗОВОЙ КОРАБЛЬ", "ПИРАТСКИЙ КОРАБЛЬ",
            "ПИРАТСКИЙ КОРАБЛЬ", "КОСМИЧЕСКИЙ КОРАБЛЬ", "КОСМИЧЕСКИЙ КОРАБЛЬ",
            "ТАРГОИДСКИЙ КОРАБЛЬ", "КОСМИЧЕСКИЙ КОРАБЛЬ", "ИСТРЕБИТЕЛЬ",
            "ТАРГОИДСКИЙ ДРОН", "ОРБИТАЛЬНАЯ СТАНЦИЯ", "КОСМИЧЕСКИЙ КУРЬЕР")

_VERTEX = """#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(2.0 * p - 1.0, 0.0, 1.0);
}"""
_FRAGMENT = """#version 330
uniform sampler2D layer;
in vec2 uv;
out vec4 color;
void main() { color = texture(layer, uv); }
"""


class SpaceHUD:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.quad = ctx.vertex_array(self.prog, [])
        self.prog["layer"].value = 0
        self.texture = None
        self.size = None
        self.key = None
        self.tabs = {}
        self.markers = {}
        self.menu = None
        self.options = {}

    def _resize(self, size):
        if self.texture is not None:
            self.texture.release()
        self.size = size
        self.s = max(.7, min(size[0] / 1240, size[1] / 780, 2.5))
        sans = pygame.font.match_font("helveticaneue,dejavusans,arial")
        mono = pygame.font.match_font("menlo,dejavusansmono,consolas")
        self.mono = pygame.font.Font(mono, round(12 * self.s))
        self.text = pygame.font.Font(sans, round(15 * self.s))
        self.title = pygame.font.Font(sans, round(37 * self.s))
        self.brand = pygame.font.Font(sans, round(23 * self.s))
        self.brand.set_bold(True)
        self.texture = self.ctx.texture(size, 4)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def tab_at(self, point):
        for action, rect in self.options.items():
            if rect.collidepoint(point):
                return action
        if self.menu:
            return "close_menu"
        return next((mode for mode, rect in self.tabs.items() if rect.collidepoint(point)), None)

    def marker_at(self, point):
        return next((index for index, rect in self.markers.items() if rect.collidepoint(point)), None)

    def _paint_flyby(self, scene):
        """Только подпись текущего корпуса и четыре команды у края кадра."""
        w, h = self.size
        s = self.s
        pad = round(28 * s)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        self.tabs.clear()
        self.markers.clear()
        self.options.clear()

        def fit(text, font, width):
            width = round(width)
            if font.size(text)[0] <= width:
                return text
            while text and font.size(text + "…")[0] > width:
                text = text[:-1]
            return text.rstrip() + "…"

        def label(text, x, y, font=None, color=_INK, width=None):
            font = font or self.text
            if width is not None:
                text = fit(text, font, width)
            surf.blit(font.render(text, True, color), (round(x), round(y)))

        # Нижняя полоса не занимает композиционный центр, верхняя — едва видна.
        for end, depth, opacity in ((False, 58, 110), (True, 155, 200)):
            rows = round(depth * s)
            for i in range(min(h, rows)):
                y = h - i - 1 if end else i
                pygame.draw.line(surf, (4, 9, 16, round(opacity * (1 - i / rows) ** 1.6)),
                                 (0, y), (w, y))
        label("E L I T E", pad, 18*s, self.mono, _GOLD)
        if w > 580*s:
            caption = "АВТОПРОЛЁТ     H  интерфейс  /  Esc  выход"
            label(caption, w - pad - self.mono.size(caption)[0], 18*s, self.mono, _MUTED)

        ship = scene.ships[scene.selected]
        authored = bool(ship.materials) and scene.texture_source == "author"
        source = SOURCE_NAMES[scene.model_source]
        textures = "Родные" if authored else "Обшивка"
        auto = "вкл" if scene.auto_sources else "выкл"
        wide = w > 760*s
        buttons = (("cycle_source", "M", f"Модели · {source}" if wide else source),
                   ("cycle_textures", "T", f"Текстуры · {textures}" if wide else textures),
                   ("toggle_auto", "A", f"Авто · {auto}"),
                   ("menu_credits", "I", "Авторы"))
        rows, row = [], []
        x = pad
        gap, bh = round(6*s), round(28*s)
        for action, key, text in buttons:
            caption = f"{key}  {text}"
            width = min(w - 2*pad, self.text.size(caption)[0] + round(20*s))
            if row and x + width > w - pad:
                rows.append(row)
                row, x = [], pad
            row.append((action, caption, x, width))
            x += width + gap
        rows.append(row)
        controls_y = h - round(22*s) - len(rows)*bh - (len(rows)-1)*gap
        for index, row in enumerate(rows):
            y = controls_y + index*(bh+gap)
            for action, caption, x, width in row:
                rect = pygame.Rect(x, y, width, bh)
                self.options[action] = rect
                active = action == "menu_credits" and self.menu == "credits"
                pygame.draw.rect(surf, (12, 23, 31, 188), rect, border_radius=max(1, round(3*s)))
                label(caption, x + 10*s, y + 4*s, color=_GOLD if active else _MUTED,
                      width=width-round(20*s))

        title_y = controls_y - round(38*s)
        label(ship.name, pad, title_y, self.brand, width=w-2*pad)
        source_title = ship.source_title
        if scene.model_source not in (ship.source_id, "mixed"):
            source_title += f" · замена для {source}"
        label(source_title, pad, title_y - 19*s, self.mono, _MUTED, width=w-2*pad)
        if self.menu == "credits":
            self._flyby_credits(surf, scene, ship, label, pad, title_y - 30*s)
        return surf

    def _flyby_credits(self, surf, scene, ship, label, pad, bottom):
        s = self.s
        width = min(self.size[0] - 2*pad, round(520*s))
        texture = ship.source_title if ship.materials and scene.texture_source == "author" else "Процедурная обшивка"
        lines = []
        for text in (ship.name, f"Геометрия — {ship.source_title}", f"Автор — {ship.author}",
                     f"Лицензия — {ship.license}", f"Текстуры — {texture}"):
            current = ""
            for word in text.split():
                if current and self.text.size(current + " " + word)[0] > width - 24*s:
                    lines.append(current)
                    current = ""
                current = (current + " " + word).strip()
            lines.append(current)
        height = round((len(lines)*22 + 39)*s)
        top = max(round(55*s), round(bottom-height))
        rect = pygame.Rect(pad, top, width, height)
        pygame.draw.rect(surf, (12, 23, 31, 246), rect, border_radius=max(1, round(4*s)))
        pygame.draw.line(surf, _GOLD, rect.topleft, rect.topright)
        for index, line in enumerate(lines):
            label(line, pad+12*s, top+(9+index*22)*s, color=_GOLD if index == 0 else _INK)
        label("I / Esc  закрыть", pad+12*s, rect.bottom-23*s, self.mono, _MUTED)

    def _paint(self, scene, hover, vp):
        if scene.camera_mode == "flyby":
            return self._paint_flyby(scene)
        w, h = self.size
        s = self.s
        pad = round(34 * s)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # Тонкие градиенты за подписями оставляют середину кадра прозрачной.
        for end, depth, opacity in ((False, 84, 152), (True, 152, 200)):
            rows = round(depth * s)
            for i in range(min(h, rows)):
                y = h - i - 1 if end else i
                pygame.draw.line(surf, (4, 9, 16, round(opacity * (1 - i / rows) ** 1.5)),
                                 (0, y), (w, y))

        def label(text, x, y, font=None, color=_INK, right=False):
            img = (font or self.text).render(text, True, color)
            surf.blit(img, (int(x - (img.get_width() if right else 0)), int(y)))
            return img.get_width()

        mode = scene.camera_mode
        ident = scene.selected if hover is None or mode != "overview" else hover
        ship = scene.ships[ident]
        name = ship.name
        top = round(20 * s)
        # Знак в форме крыла Cobra по соседству с названием атласа.
        points = [(pad + int(x * s), top + int(y * s)) for x, y in
                  ((0, 18), (12, 4), (29, 14), (13, 20), (0, 18))]
        pygame.draw.lines(surf, _GOLD, False, points, max(1, round(1.4 * s)))
        pygame.draw.line(surf, _GOLD, points[1], points[3], max(1, round(s)))
        x = pad + 41 * s
        x += label("E L I T E", x, top - s, self.brand)
        if w > 710 * s:
            pygame.draw.line(surf, (94, 111, 116), (x + 18*s, top + 4*s), (x + 18*s, top + 22*s))
            label("АТЛАС ФЛОТА", x + 35*s, top + 6*s, self.mono, _MUTED)

        self.tabs.clear()
        x = w - pad
        for tab, key, title in reversed(_TABS):
            caption = title if w < 650*s else key + "  " + title
            tw = self.text.size(caption)[0]
            left = x - tw - 22*s
            r = pygame.Rect(round(left), round(top - 2*s), round(x - left), round(32*s))
            self.tabs[tab] = r
            if mode == tab:
                pygame.draw.rect(surf, (18, 31, 39, 222), r, border_radius=round(3*s))
                pygame.draw.line(surf, _GOLD, (r.left + 8*s, r.bottom), (r.right - 8*s, r.bottom),
                                 max(1, round(s)))
            label(caption, left + 11*s, top + 3*s, color=_INK if mode == tab else _MUTED)
            x = left - 4*s

        self.markers.clear()
        if mode == "overview":
            center = scene.project((0, 0, 0), w, h) or (w / 2, h / 2)
            focal = h * .5 * np.linalg.norm(vp[1, :3])
            for obj in scene.objects:
                pos = scene.project(obj.position, w, h)
                if pos is None:
                    continue
                ox, oy = pos
                depth = vp[3, :3] @ obj.position + vp[3, 3]
                radius = max(6*s, focal * obj.radius / max(depth, .001))
                dx, dy = ox - center[0], oy - center[1]
                dist = max(hypot(dx, dy), 1)
                dx, dy = dx / dist, dy / dist
                is_target = obj.model_id == ident
                color = _GOLD if is_target else (160, 180, 190)
                r0, r1 = radius + 2*s, radius + 9*s
                p0 = (ox + dx*r0, oy + dy*r0)
                p1 = (ox + dx*r1, oy + dy*r1)
                pygame.draw.line(surf, color, p0, p1)
                txt = f"{obj.model_id:02d}"
                lx = p1[0] + 3*s if dx >= 0 else p1[0] - self.mono.size(txt)[0] - 3*s
                ly = p1[1] - self.mono.get_height() / 2
                label(txt, lx, ly, self.mono, color)
                self.markers[obj.model_id] = pygame.Rect((lx, ly), self.mono.size(txt)).inflate(5*s, 5*s)
                if is_target:
                    # Четыре коротких угла оставляют вид на сам корпус свободным.
                    for a, b in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                        cx, cy = ox + a*r0, oy + b*r0
                        pygame.draw.lines(surf, _GOLD, False, ((cx - a*5*s, cy), (cx, cy), (cx, cy - b*5*s)))

        by = h - round(114*s)
        if mode == "overview":
            eyebrow = f"КАРТА  01     /     ОБЪЕКТОВ  {len(scene.objects)}"
            headline = "Орбитальный каталог"
        elif mode == "inspect":
            eyebrow = f"ОБЪЕКТ  {ident:02d}     /     {_CLASSES[ident]}"
            headline = name.split(" · ")[0]
        else:
            eyebrow = "КАРТА  01     /     РУЧНОЕ УПРАВЛЕНИЕ"
            headline = "Свободный полёт"
        label(eyebrow, pad, by, self.mono, _GOLD)
        label(headline, pad - s, by + 18*s, self.title)

        if w > 790*s:
            tag = {"overview": f"ВЫБОР  {ident:02d}", "inspect": "ИСТОЧНИК ГЕОМЕТРИИ",
                   "flight": "КУРСОР КАТАЛОГА"}[mode]
            detail = ship.source_title if mode == "inspect" else name
            if mode == "inspect" and ship.meshes:
                tag = f"{ship.author}  ·  {ship.license}"
                if self.mono.size(tag)[0] > w * .47:
                    tag = "АВТОР И ЛИЦЕНЗИЯ   [I]"
            label(tag, w - pad, by + 7*s, self.mono, _MUTED, right=True)
            label(detail, w - pad, by + 29*s, color=_INK, right=True)

        hints = {
            "overview": ("Щелчок  осмотр", "Стрелки  выбор", "Тянуть  поворот", "Колесо  масштаб", "H  интерфейс"),
            "inspect": ("Стрелки  объекты", "Тянуть  облёт", "Колесо  масштаб", "O  вся карта", "H  интерфейс"),
            "flight": ("WASD  движение", "Q / E  вниз / вверх", "Тянуть  взгляд", "Shift  ускорение", "O  карта", "H  интерфейс"),
        }[mode]
        y = h - round(37*s)
        pygame.draw.line(surf, (91, 113, 122, 125), (pad, y - 7*s), (w - pad, y - 7*s))
        x = pad
        for hint in hints:
            if x + self.text.size(hint)[0] > w - pad:
                break
            x += label(hint, x, y, color=_MUTED)
            x += 23*s
        self._source_menu(surf, scene, ship, label, pad, int(61*s))
        return surf

    def _source_menu(self, surf, scene, ship, label, pad, y):
        """Селекторы остаются на карте и в осмотре. Карточка I показывает полные условия."""
        s = self.s
        w = self.size[0]
        x = pad
        self.options.clear()
        buttons = (("models", "M", "Модели", SOURCE_NAMES[scene.model_source]),
                   ("textures", "T", "Текстуры", TEXTURE_NAMES[scene.texture_source]),
                   ("credits", "I", "", "Авторы"))
        for key, hotkey, title, value in buttons:
            caption = f"{hotkey}  {title} · {value}" if title and w >= 740*s else f"{hotkey}  {value}"
            width = self.text.size(caption)[0] + round((19 if key == "credits" else 32)*s)
            rect = pygame.Rect(x, y, width, round(29*s))
            self.options["menu_" + key] = rect
            pygame.draw.rect(surf, (12, 23, 31, 225), rect, border_radius=round(3*s))
            label(caption, x+9*s, y+3*s, color=_GOLD if self.menu == key else _INK)
            if key != "credits":
                tip = rect.right - 11*s
                pygame.draw.lines(surf, _MUTED, False, ((tip-3*s, y+12*s), (tip, y+15*s), (tip+3*s, y+12*s)))
            x = rect.right + round(5*s)
        if not self.menu:
            return
        anchor = self.options["menu_" + self.menu]
        if self.menu == "credits":
            width = min(w - pad*2, round(475*s))
            left = min(anchor.x, w - pad - width)
            texture = ship.source_title if ship.materials and scene.texture_source == "author" else "Процедурные материалы атласа"
            lines = []
            for line in (f"{ship.name} · {ship.source_title}", f"Автор — {ship.author}",
                         f"Лицензия — {ship.license}", f"Текстуры — {texture}"):
                current = ""
                for word in line.split():
                    if current and self.text.size(current + " " + word)[0] > width - 24*s:
                        lines.append(current)
                        current = ""
                    current = (current + " " + word).strip()
                lines.append(current)
            rect = pygame.Rect(left, anchor.bottom+4*s, width, (len(lines)*22+15)*s)
            pygame.draw.rect(surf, (14, 26, 35, 250), rect, border_radius=round(4*s))
            for i, line in enumerate(lines):
                label(line, left+12*s, rect.top+(7+i*22)*s, color=_GOLD if i == 0 else _INK)
            return
        models = self.menu == "models"
        choices = scene.catalog.sources if models else tuple(TEXTURE_NAMES)
        names = SOURCE_NAMES if models else TEXTURE_NAMES
        current = scene.model_source if models else scene.texture_source
        width = max(anchor.width, round(177*s))
        left = min(anchor.x, w - pad - width)
        y = anchor.bottom + round(4*s)
        for key in choices:
            row = pygame.Rect(left, y, width, round(30*s))
            self.options[("source_" if models else "texture_") + key] = row
            pygame.draw.rect(surf, (19, 35, 44, 252) if key == current else (12, 23, 31, 248), row)
            if key == current:
                pygame.draw.line(surf, _GOLD, row.topleft, row.bottomleft, max(1, round(2*s)))
            label(names[key], left+12*s, y+4*s, color=_GOLD if key == current else _INK)
            y = row.bottom

    def draw(self, scene, size, hover, vp):
        if self.size != size:
            self._resize(size)
        # Во время полёта меняется только 3D. Карту перерисовываем при смене проекции.
        key = (size, scene.camera_mode, scene.selected, hover, scene.model_source, scene.texture_source,
               self.menu, getattr(scene, "auto_sources", None),
               vp.tobytes() if scene.camera_mode == "overview" else None)
        if self.key != key:
            surface = self._paint(scene, hover, vp)
            self.texture.write(pygame.image.tobytes(surface, "RGBA", True))
            self.key = key
        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.texture.use(0)
        self.quad.render(vertices=3)

    def release(self):
        if self.texture is not None:
            self.texture.release()
        self.quad.release()
        self.prog.release()
