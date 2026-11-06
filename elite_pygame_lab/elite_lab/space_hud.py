"""Экранные подписи космического атласа и выбор режима мышью."""

from math import hypot

import moderngl
import numpy as np
import pygame

from .space_assets import FLYBY_SOURCE_NAMES, SOURCE_NAMES, TEXTURE_NAMES
from .space_resolution import RENDER_RESOLUTIONS, RenderResolution


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
        self.pixel_ratio = 1.0
        self.key = None
        self.tabs = {}
        self.markers = {}
        self.menu = None
        self.options = {}
        self.resolution = RenderResolution()

    def _resize(self, size, pixel_ratio=1.0):
        if self.texture is not None:
            self.texture.release()
        self.size = size
        self.pixel_ratio = pixel_ratio
        self.s = max(.7 * pixel_ratio, min(size[0] / 1240, size[1] / 780, 2.5 * pixel_ratio))
        sans = pygame.font.match_font("helveticaneue,dejavusans,arial")
        mono = pygame.font.match_font("menlo,dejavusansmono,consolas")
        self.mono = pygame.font.Font(mono, round(12 * self.s))
        self.text = pygame.font.Font(sans, round(15 * self.s))
        # Helvetica не содержит стрелок; подписи клавиш должны показывать их явно.
        self.symbols = pygame.font.Font(mono, round(15 * self.s))
        self.title = pygame.font.Font(sans, round(37 * self.s))
        self.brand = pygame.font.Font(sans, round(23 * self.s))
        self.brand.set_bold(True)
        self.texture = self.ctx.texture(size, 4)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def tab_at(self, point):
        for action, rect in self.options.items():
            resolution_action = action.startswith("resolution_") or action == "close_menu"
            if resolution_action == (self.menu == "resolution") and rect.collidepoint(point):
                return action
        if self.menu:
            return "close_menu"
        return next((mode for mode, rect in self.tabs.items() if rect.collidepoint(point)), None)

    def marker_at(self, point):
        return next((index for index, rect in self.markers.items() if rect.collidepoint(point)), None)

    @staticmethod
    def _flyby_clock(elapsed):
        """Абсолютное время: после полного круга счётчик не возвращается к нулю."""
        minutes, seconds = divmod(max(0, int(elapsed)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _flight_profile_name(scene):
        return "Elite" if getattr(scene, "flight_profile", "modern") == "elite" else "3D"

    @staticmethod
    def _flight_speed_text(scene):
        """Малая тяга видимо отличается от остановки, большие значения округлены."""
        speed = getattr(scene, "flight_speed", 0)
        if speed == 0:
            value = "0"
        elif speed < .1:
            value = "<0,1"
        elif speed < 5:
            value = f"{speed:.1f}".replace(".", ",")
        else:
            value = f"{speed:.0f}"
        return value + " ед/с"

    @staticmethod
    def _flight_hints(scene):
        """Обе схемы доступны без мыши; единицы скорости относятся к сцене."""
        if getattr(scene, "flight_profile", "modern") == "elite":
            controls = ("← / →  крен", "↑ / ↓  нос вверх / вниз", "W / S  скорость + / −",
                        "Отпустите W/S — скорость сохранится", "Мышь  крен / тангаж")
        else:
            controls = ("W / S  вперёд / назад", "A / D  стрейф", "Пробел / Shift  вверх / вниз",
                        "Q / E  крен", "Стрелки / мышь  взгляд")
        return controls + ("Esc  курсор · щелчок  захват",)

    @staticmethod
    def _effects_hint(scene):
        state = "вкл" if getattr(scene, "effects_enabled", False) else "выкл"
        return f"B  огни флота · {state}"

    def _hint_rows(self, hints, width, font=None):
        """Переносим команды целиком, а слишком длинную подсказку — по словам."""
        font = font or self.mono
        gap = round(18*self.s)
        rows, row, used = [], [], 0
        for hint in hints:
            fragments, fragment = [], ""
            for word in hint.split():
                candidate = (fragment + " " + word).strip()
                if fragment and font.size(candidate)[0] > width:
                    fragments.append(fragment)
                    fragment = word
                else:
                    fragment = candidate
            fragments.append(fragment)
            # Сохраняем двойной пробел между клавишей и назначением, если перенос не нужен.
            if len(fragments) == 1 and font.size(hint)[0] <= width:
                fragments = [hint]
            for fragment in fragments:
                tw = font.size(fragment)[0]
                if row and used + gap + tw > width:
                    rows.append(row)
                    row, used = [], 0
                x = used + (gap if row else 0)
                row.append((fragment, x))
                used = x + tw
        if row:
            rows.append(row)
        return rows

    def _paint_flyby(self, scene):
        """Время, подпись корпуса и компактные команды у края кадра."""
        w, h = self.size
        s = self.s
        pad = round(28 * s)
        flying = scene.camera_mode == "flight"
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
        for end, depth, opacity in ((False, 80, 110), (True, 185, 200)):
            rows = round(depth * s)
            for i in range(min(h, rows)):
                y = h - i - 1 if end else i
                pygame.draw.line(surf, (4, 9, 16, round(opacity * (1 - i / rows) ** 1.6)),
                                 (0, y), (w, y))
        zx = scene.model_source == "zx"
        accent = (235, 245, 255) if zx else _GOLD
        logo = "E L I T E"
        label(logo, pad, 18*s, self.mono, accent)
        mode_title = "СВОБОДНЫЙ ПОЛЁТ" if flying else "АВТОПРОЛЁТ"
        escape = "курсор" if flying else "выход"
        caption = (f"{mode_title}     H  интерфейс  /  Esc  {escape}" if w > 580*s else
                   f"H интерфейс / Esc {escape}")
        if not zx:
            with_effects = self._effects_hint(scene) + "  /  " + caption
            if self.mono.size(with_effects)[0] <= w - 2 * pad - self.mono.size(logo)[0] - 12 * s:
                caption = with_effects
        caption_x = w - pad - self.mono.size(caption)[0]
        label(caption, caption_x, 18*s, self.mono, _MUTED)
        resolution_button = ("menu_resolution", "G", "Разрешение")
        resolution_caption = "  ".join(resolution_button[1:])
        resolution_rect = pygame.Rect(pad+self.mono.size(logo)[0]+round(12*s), round(11*s),
                                     self.text.size(resolution_caption)[0]+round(20*s), round(25*s))
        resolution_on_top = resolution_rect.right + round(6*s) <= caption_x
        if resolution_on_top:
            self.options["menu_resolution"] = resolution_rect
            pygame.draw.rect(surf, (12, 23, 31, 188), resolution_rect,
                             border_radius=max(1, round(3*s)))
            label(resolution_caption, resolution_rect.left+10*s, resolution_rect.top+4*s,
                  color=accent if self.menu == "resolution" else _MUTED)
        paused = scene.paused
        if flying:
            status = f"РУЧНОЙ ПОЛЁТ · маршрут на паузе · {self._flyby_clock(scene.elapsed)}"
            if getattr(scene, "flight_profile", "modern") == "elite":
                status += " · " + self._flight_speed_text(scene)
            label(status,
                  pad, 39*s, self.mono, accent, width=w-2*pad)
            for index, row in enumerate(self._hint_rows(self._flight_hints(scene), w-2*pad)):
                for hint, x in row:
                    label(hint, pad+x, (59+index*19)*s, self.mono, _MUTED)
        else:
            label("ПКМ + тянуть / Shift + стрелки  взгляд · R  по курсу",
                  pad, 39*s, self.mono, _MUTED, width=w-2*pad)
            state = "ПАУЗА" if paused else "ПРОЛЁТ"
            label(f"{self._flyby_clock(scene.elapsed)} · {state} · держать ← / →  4×",
                  pad, 59*s, self.mono, accent if paused else _MUTED, width=w-2*pad)

        empty = bool(getattr(scene, "empty_slot", False))
        ship = None if empty else scene.ships[scene.selected]
        authored = ship is not None and bool(ship.materials) and scene.texture_source == "author"
        source = FLYBY_SOURCE_NAMES[scene.model_source]
        textures = "Родные" if authored else "Наша обшивка"
        auto = "вкл" if scene.auto_sources else "выкл"
        wide = w > 760*s
        flight_button = ("toggle_flight", "F", "К автопролёту" if flying else "Свободный полёт")
        profile = self._flight_profile_name(scene)
        profile_button = ("cycle_flight_profile", "C", f"Схема · {profile}" if wide else profile)
        transport = (flight_button, profile_button) if flying else (
            flight_button, ("toggle_pause", "Пробел", "Продолжить" if paused else "Пауза"),
            ("scrub_back", "←", "−1 с"), ("scrub_forward", "→", "+1 с"))
        models = (("cycle_source", "M", f"Модели · {source}" if wide else source),
                  ("cycle_textures", "T", "Каркас · без текстур" if zx else
                   (f"Текстуры · {textures}" if wide else textures)))
        if not flying:
            models += (("toggle_auto", "A", f"Авто · {auto}"),)
        models += (("menu_credits", "I", "Авторы"),)
        if not resolution_on_top:
            models += (resolution_button,)
        groups = (transport, models)
        gap, bh = round(6*s), round(28*s)
        group_gap = round(22*s)
        prepared = []
        for buttons in groups:
            group = []
            for action, key, text in buttons:
                caption = f"{key}  {text}"
                font = self.symbols if action in ("scrub_back", "scrub_forward") else self.text
                width = min(w - 2*pad, font.size(caption)[0] + round(20*s))
                group.append((action, caption, width, font))
            prepared.append(group)
        joined_width = sum(sum(item[2] for item in group) + (len(group)-1)*gap for group in prepared) + group_gap
        single_row = joined_width <= w - 2*pad
        rows, row, x = [], [], pad
        for group_index, buttons in enumerate(prepared):
            if group_index:
                if single_row:
                    x += group_gap - gap
                else:
                    rows.append(row)
                    row, x = [], pad
            for action, caption, width, font in buttons:
                if row and x + width > w - pad:
                    rows.append(row)
                    row, x = [], pad
                row.append((action, caption, x, width, font))
                x += width + gap
        if row:
            rows.append(row)
        controls_y = h - round(22*s) - len(rows)*bh - (len(rows)-1)*gap
        for index, row in enumerate(rows):
            y = controls_y + index*(bh+gap)
            for action, caption, x, width, font in row:
                rect = pygame.Rect(x, y, width, bh)
                self.options[action] = rect
                active = ((action == "menu_credits" and self.menu == "credits") or
                          (action == "menu_resolution" and self.menu == "resolution") or
                          (action == "toggle_pause" and paused))
                pygame.draw.rect(surf, (12, 23, 31, 188), rect, border_radius=max(1, round(3*s)))
                if action in ("cycle_source", "toggle_flight", "cycle_flight_profile"):
                    pygame.draw.line(surf, accent, rect.topleft, rect.bottomleft, max(1, round(2*s)))
                label(caption, x + 10*s, y + 4*s, font,
                      color=accent if active or action in ("cycle_source", "toggle_flight", "cycle_flight_profile") else _MUTED,
                      width=width-round(20*s))

        title_y = controls_y - round(38*s)
        label("Пустое место" if empty else ship.name, pad, title_y, self.brand,
              color=accent, width=w-2*pad)
        source_title = "Гость Pioneer отсутствует в этом наборе" if empty else ship.source_title
        if not empty and scene.model_source in ("zx", "elite"):
            source_title = FLYBY_SOURCE_NAMES[scene.model_source]
        elif not empty and scene.model_source not in (ship.source_id, "mixed"):
            source_title += f" · замена для {source}"
        if flying:
            source_title = "Ближайший объект · " + source_title
        label(source_title, pad, title_y - 19*s, self.mono, _MUTED, width=w-2*pad)
        if self.menu == "credits":
            self._flyby_credits(surf, scene, ship, label, pad, title_y - 30*s)
        elif self.menu == "resolution":
            self._paint_resolution(surf, zx=zx)
        return surf

    def _flyby_credits(self, surf, scene, ship, label, pad, bottom):
        s = self.s
        width = min(self.size[0] - 2*pad, round(520*s))
        if ship is None:
            details = ("Пустое место гостя", "В этом наборе гостевой модели нет.",
                       "Sinonatrix Courier доступен в Pioneer и сборном флоте.")
        else:
            texture = ("Без текстур · пиксельный каркас" if scene.model_source == "zx" else
                       ship.source_title if ship.materials and scene.texture_source == "author" else
                       "Наша процедурная обшивка")
            details = (ship.name, f"Геометрия — {ship.source_title}", f"Автор — {ship.author}",
                       f"Лицензия — {ship.license}", f"Текстуры — {texture}")
        lines = []
        for text in details:
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
        if getattr(scene, "is_flyby", False):
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

        hints = {
            "overview": ("Щелчок  осмотр", "Стрелки  выбор", "Тянуть  поворот", "Колесо  масштаб",
                         "Shift + стрелки  взгляд", "O  исходный вид", "H  интерфейс"),
            "inspect": ("Стрелки  объекты", "Тянуть  облёт", "Колесо  масштаб", "Shift + стрелки  взгляд",
                        "O  вся карта", "H  интерфейс"),
            "flight": self._flight_hints(scene) + ("O  карта", "H  интерфейс"),
        }[mode]
        hints += (self._effects_hint(scene),)
        hint_font = self.mono if mode == "flight" else self.text
        hint_rows = self._hint_rows(hints, w-2*pad, hint_font)
        hints_y = h - round((37 + (len(hint_rows)-1)*20)*s)
        by = hints_y - round(77*s)
        if mode == "overview":
            eyebrow = f"КАРТА  01     /     ОБЪЕКТОВ  {len(scene.objects)}"
            headline = "Орбитальный каталог"
        elif mode == "inspect":
            eyebrow = f"ОБЪЕКТ  {ident:02d}     /     {_CLASSES[ident]}"
            headline = name.split(" · ")[0]
        else:
            eyebrow = "РУЧНОЕ УПРАВЛЕНИЕ · " + self._flight_profile_name(scene)
            if getattr(scene, "flight_profile", "modern") == "elite":
                eyebrow += " · " + self._flight_speed_text(scene)
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

        pygame.draw.line(surf, (91, 113, 122, 125), (pad, hints_y - 7*s), (w - pad, hints_y - 7*s))
        for index, row in enumerate(hint_rows):
            for hint, x in row:
                label(hint, pad+x, hints_y+index*20*s, hint_font, _MUTED)
        self._source_menu(surf, scene, ship, label, pad, int(61*s))
        return surf

    def _paint_resolution(self, surf, *, zx=False):
        w, h = self.size
        s = self.s
        state = self.resolution
        self.options.clear()
        margin = round(11*s)
        pad, gap = round(16*s), round(5*s)
        width = min(w-2*margin, round(400*s))
        help_text = ("Каркас ZX · 256 × 128. Выбор для цветной сцены." if zx else
                     "Больше пикселей — чётче сцена.")
        help_rows = self._hint_rows((state.error or help_text,), width-2*pad, self.text)
        text_height = self.text.get_height()
        header = self.brand.get_height() + max(2, len(help_rows))*text_height + 3*gap
        footer = self.symbols.get_height() + 2*gap
        count = len(RENDER_RESOLUTIONS)
        row_height = max(1, min(round(27*s), (h-2*(margin+pad)-header-footer) // count))
        panel = pygame.Rect(0, 0, width, 2*pad + header + count*row_height + footer)
        panel.center = surf.get_rect().center
        radius = max(1, round(4*s))
        pygame.draw.rect(surf, (12, 23, 31, 250), panel, border_radius=radius)
        pygame.draw.line(surf, _GOLD, panel.topleft, panel.topright)

        left, right = panel.left + pad, panel.right - pad
        clip = surf.get_clip()
        surf.set_clip(panel)

        def label(text, x, y, font, color=_INK, *, right=False):
            img = font.render(text, True, color)
            surf.blit(img, (x-img.get_width() if right else x, y))

        y = panel.top + pad
        label("Разрешение рендера", left, y, self.brand)
        help_y = y + self.brand.get_height() + gap
        for index, row in enumerate(help_rows):
            for text, x in row:
                label(text, left+x, help_y+index*text_height, self.text, _GOLD if state.error else _MUTED)
        y += header
        for index, size in enumerate(RENDER_RESOLUTIONS):
            row = pygame.Rect(left, y, right-left, row_height)
            self.options[f"resolution_{index}"] = row
            selected, highlighted = index == state.selected, index == state.highlighted
            disabled = index in state.unavailable
            fill = (36, 54, 65) if highlighted else (19, 35, 44)
            pygame.draw.rect(surf, fill, row, border_radius=radius)
            if highlighted:
                pygame.draw.line(surf, _GOLD, row.topleft, (row.left, row.bottom-1), max(1, round(2*s)))
            if size is None:
                width_px, height_px = state.output_size or self.size
                caption = f"Нативное · {width_px} × {height_px}"
            else:
                width_px, height_px = size
                caption = f"{height_px}p · {width_px} × {height_px}"
            color = _GOLD if selected else (135, 155, 164) if disabled else _INK
            y_text = row.centery - text_height//2
            label(caption, left+2*gap, y_text, self.text, color)
            status = "Текущее" if selected else "Недоступно" if disabled else ""
            if status:
                label(status, right-2*gap, y_text, self.text, color, right=True)
            y = row.bottom

        hint_y = y + gap
        label("↑↓ · Enter  применить", left, hint_y, self.symbols, _MUTED)
        close = "G / Esc  закрыть"
        close_width = self.symbols.size(close)[0]
        self.options["close_menu"] = pygame.Rect(right-close_width-gap, y, close_width+2*gap, footer)
        label(close, right, hint_y, self.symbols, _INK, right=True)
        surf.set_clip(clip)

    def _source_menu(self, surf, scene, ship, label, pad, y):
        """Селекторы остаются на карте и в осмотре. Карточка I показывает полные условия."""
        s = self.s
        w = self.size[0]
        x = pad
        self.options.clear()
        buttons = (("models", "M", "Модели", SOURCE_NAMES[scene.model_source]),
                   ("textures", "T", "Текстуры", TEXTURE_NAMES[scene.texture_source]),
                   ("credits", "I", "", "Авторы"))
        if scene.camera_mode == "flight":
            buttons += (("flight_profile", "C", "Схема", self._flight_profile_name(scene)),)
        buttons += (("resolution", "G", "", "Разрешение"),)
        for key, hotkey, title, value in buttons:
            caption = f"{hotkey}  {title} · {value}" if title and w >= 740*s else f"{hotkey}  {value}"
            width = min(w-2*pad, self.text.size(caption)[0] + round((32 if key in ("models", "textures") else 19)*s))
            if x > pad and x + width > w - pad:
                x, y = pad, y + round(34*s)
            rect = pygame.Rect(x, y, width, round(29*s))
            action = "cycle_flight_profile" if key == "flight_profile" else "menu_" + key
            self.options[action] = rect
            pygame.draw.rect(surf, (12, 23, 31, 225), rect, border_radius=round(3*s))
            label(caption, x+9*s, y+3*s, color=_GOLD if self.menu == key or key == "flight_profile" else _INK)
            if key in ("models", "textures"):
                tip = rect.right - 11*s
                pygame.draw.lines(surf, _MUTED, False, ((tip-3*s, y+12*s), (tip, y+15*s), (tip+3*s, y+12*s)))
            x = rect.right + round(5*s)
        if self.menu == "resolution":
            self._paint_resolution(surf)
            return
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

    def draw(self, scene, size, hover, vp, *, pixel_ratio=1.0):
        if self.size != size or self.pixel_ratio != pixel_ratio:
            self._resize(size, pixel_ratio)
        # Секунды и паузу обновляем независимо от выбора корпуса; карту — при смене проекции.
        is_flyby = getattr(scene, "is_flyby", False)
        playback = (scene.paused, int(scene.elapsed)) if is_flyby else None
        profile = getattr(scene, "flight_profile", "modern")
        flight = ((profile, self._flight_speed_text(scene) if profile == "elite" else None)
                  if scene.camera_mode == "flight" else None)
        resolution = None
        if self.menu == "resolution":
            state = self.resolution
            resolution = (state.selected, state.highlighted, frozenset(state.unavailable), state.error,
                          state.output_size)
        key = (size, pixel_ratio, is_flyby, scene.camera_mode, scene.selected, hover, scene.model_source, scene.texture_source,
               self.menu, getattr(scene, "auto_sources", None), getattr(scene, "empty_slot", False),
               playback, flight, resolution, getattr(scene, "effects_enabled", False),
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
