"""Pygame только показывает рассчитанные вручную пиксели и учебный интерфейс.

Запуск: python -m elite_lab
Вкладки показывают выбранную модель, пиксельный цикл и каталог оригинала.
Ни один отрезок не рисуется через pygame.draw.line / lines / aaline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from .controls import pose_from_angles
from .framebuffer import Framebuffer
from .model import COBRA_ID, DODO_ID, MODELS, initial_depth, minimum_depth
from .raster import LineTrace, trace_line
from .renderer import render_geometry
from .types import ProjectedVertex


PAPER = (8, 13, 21)
BG = (14, 21, 32)
PANEL = (21, 31, 45)
INK = (225, 237, 244)
MUTED = (150, 170, 190)
CYAN = (72, 220, 229)
AMBER = (255, 190, 85)
GRID = (38, 53, 70)


def _surface(framebuffer: Framebuffer, ink=INK) -> pygame.Surface:
    # frombuffer копирует/показывает уже рассчитанные пиксели. Здесь нет
    # ни преобразования 3D, ни выбора пикселей линии.
    return pygame.image.frombuffer(framebuffer.rgb_bytes(ink, PAPER), (256, 128), "RGB")


def _ship_lines(geometry) -> dict[int, LineTrace]:
    return {edge.index: trace_line(edge.start, edge.end)
            for edge in geometry.edge_decisions if edge.draw}


def _ship_frame(lines: dict[int, LineTrace]) -> Framebuffer:
    frame = Framebuffer()
    for trace in lines.values():
        for pixel in trace.steps:
            if pixel.in_bitmap:
                frame.set_pixel(pixel.x, pixel.y)
    return frame


def _initial_angles(model_id: int) -> tuple[float, float, float]:
    # Вход Dodo находится на стороне +Z. Открываем станцию доком к зрителю.
    return (155.0 if model_id == DODO_ID else 24.0), 22.0, 0.0


class Viewer:
    def __init__(self, size=(1280, 830), start_mode="ship") -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("Elite — ручной рендерер и пиксельный цикл")
        font_path = pygame.font.match_font("dejavusans")
        # На macOS pygame может подставить Courier без кириллицы вместо DejaVu.
        mono_path = pygame.font.match_font("menlo,monaco,dejavusansmono")
        self.font = pygame.font.Font(font_path, 17)
        self.small = pygame.font.Font(font_path, 14)
        self.title_font = pygame.font.Font(font_path, 25)
        self.mono = pygame.font.Font(mono_path, 16)
        self.mode = start_mode
        self.model_id = COBRA_ID
        self.vertex_tests = [model.vertex_test_default for model in MODELS]
        self.catalog_cursor = COBRA_ID
        self.thumbnails: list[pygame.Surface] = []
        self.yaw, self.pitch, self.roll = _initial_angles(self.model_id)
        self.depth = 768
        self.edge_index = 7
        self.rotate_running = False
        self.line_running = False
        self.line_count = 0
        self.line_timer = 0.0
        self.line_points = [ProjectedVertex.from_screen(34, 32),
                            ProjectedVertex.from_screen(214, 91)]
        self.line_source = "Свободный отрезок"
        self.preset_index = -1
        self.dragging: int | None = None
        self.grid = True
        self.geometry = None
        self.ship_frame = Framebuffer()
        self.line_trace: LineTrace = trace_line(*self.line_points)
        self.viewport = pygame.Rect(20, 110, 768, 384)
        self.scale = 3
        self.tabs: list[pygame.Rect] = []
        self.rebuild_ship()

    def text(self, text: str, xy, color=INK, font=None) -> None:
        self.screen.blit((font or self.font).render(text, True, color), xy)

    def wrapped(self, text: str, x: int, y: int, width: int, color=MUTED, font=None) -> int:
        font = font or self.small
        words = text.split()
        line = ""
        for word in words:
            trial = f"{line} {word}".strip()
            if line and font.size(trial)[0] > width:
                self.text(line, (x, y), color, font)
                y += font.get_linesize() + 3
                line = word
            else:
                line = trial
        if line:
            self.text(line, (x, y), color, font)
            y += font.get_linesize() + 3
        return y

    def rebuild_ship(self) -> None:
        pose = pose_from_angles(self.yaw, self.pitch, self.roll, self.depth)
        self.geometry = render_geometry(self.model, pose, use_face_vertices=self.use_face_vertices)
        self.ship_lines = _ship_lines(self.geometry)
        self.ship_frame = _ship_frame(self.ship_lines)

    @property
    def model(self):
        return MODELS[self.model_id]

    @property
    def use_face_vertices(self):
        return self.vertex_tests[self.model_id]

    def choose_model(self, index: int) -> None:
        self.mode = "ship"
        self.catalog_cursor = index
        if self.model_id == index:
            return
        self.model_id = index
        if index == DODO_ID:
            self.yaw, self.pitch, self.roll = _initial_angles(index)
        self.depth = initial_depth(self.model_id)
        self.rebuild_ship()
        self.edge_index = next((index for index, trace in self.ship_lines.items() if trace.steps), 0)

    def zoom(self, notches: int) -> None:
        """Одинаковый ход колеса и клавиш. Вблизи шаг уменьшается до 16."""
        nearest = minimum_depth(self.model_id)
        for _ in range(abs(notches)):
            if notches > 0:
                depth = max(256, self.depth - 128) if self.depth > 256 else max(nearest, self.depth - 16)
            else:
                depth = min(256, self.depth + 16) if self.depth < 256 else min(6000, self.depth + 128)
            if self.depth == depth:
                break
            self.depth = depth
        self.rebuild_ship()

    def show_clipping_example(self, model_id: int) -> None:
        self.choose_model(model_id)
        if model_id == 0:
            # Исходный пример Coriolis: точка A ребра 02 выше экрана, Z80 начинает с B.
            self.yaw, self.depth, self.edge_index = 155.0, 640, 2
            self.rebuild_ship()
            return
        # Начинаем с порога $ACA7. Первый шаг переведёт модель в ближний диапазон.
        self.depth = min(self.depth, 768)
        while self.depth > minimum_depth(model_id):
            self.zoom(1)
            for index, line in self.ship_lines.items():
                if line.clip_status == "clipped":
                    self.edge_index = index
                    return

    def show_face_example(self) -> None:
        report = Path(__file__).resolve().parent.parent / "face_comparison.json"
        example = json.loads(report.read_text(encoding="utf-8"))[self.model_id]["example"]
        self.yaw, self.pitch, self.roll, self.depth = example["view"]
        self.edge_index = example["edge"]
        self.rotate_running = False
        self.rebuild_ship()

    def rebuild_line(self) -> None:
        self.line_trace = trace_line(*self.line_points)
        self.line_count = min(self.line_count, len(self.line_trace.steps))

    def copy_ship_edge(self) -> None:
        decision = self.geometry.edge_decisions[self.edge_index]
        if decision.draw:
            self.line_points = [decision.start, decision.end]
            self.line_source = f"{self.model.name}, ребро {self.edge_index}"
            self.rebuild_line()
            self.line_count = 0
            self.line_running = False
            self.mode = "line"

    def layout(self) -> tuple[int, int]:
        width, height = self.screen.get_size()
        self.scale = max(1, min(4, (width - 420) // 256, (height - 300) // 128))
        self.viewport = pygame.Rect(20, 114, 256 * self.scale, 128 * self.scale)
        return self.viewport.right + 24, width - self.viewport.right - 44

    def draw_header(self) -> None:
        self.text("ELITE / ПИКСЕЛЬ ЗА ПИКСЕЛЕМ", (20, 18), INK, self.title_font)
        self.text("ZX Spectrum → читаемый Python · pygame показывает готовый bitmap", (20, 53), MUTED, self.small)
        self.tabs = [pygame.Rect(20, 80, 185, 27), pygame.Rect(215, 80, 245, 27),
                     pygame.Rect(470, 80, 220, 27)]
        for rect, mode, label in zip(self.tabs, ("ship", "line", "catalog"),
                                    ("1  МОДЕЛЬ", "2  ШАГИ ЛИНИИ", f"3  КАТАЛОГ · {len(MODELS)}")):
            pygame.draw.rect(self.screen, PANEL if self.mode != mode else (29, 63, 76), rect, border_radius=4)
            self.text(label, (rect.x + 9, rect.y + 3), CYAN if self.mode == mode else MUTED, self.small)

    def draw_bitmap(self, fb: Framebuffer, ink=INK) -> None:
        self.screen.blit(pygame.transform.scale(_surface(fb, ink), self.viewport.size), self.viewport)
        pygame.draw.rect(self.screen, GRID, self.viewport.inflate(2, 2), 1)

    def mark(self, point, color, label: str | None = None) -> None:
        x, y = point
        if not (0 <= x < 256 and 0 <= y < 128):
            return
        rect = pygame.Rect(self.viewport.x + x * self.scale, self.viewport.y + y * self.scale,
                           self.scale, self.scale)
        self.screen.set_clip(self.viewport)
        pygame.draw.rect(self.screen, color, rect.inflate(8, 8), 1)
        if label:
            label_x = min(rect.x + 8, self.viewport.right - self.small.size(label)[0] - 2)
            label_y = max(rect.y - 20, self.viewport.top + 2)
            self.text(label, (label_x, label_y), color, self.small)
        self.screen.set_clip(None)

    def mark_line(self, trace: LineTrace) -> None:
        if trace.clip_status == "inside":
            self.mark(trace.start, CYAN, "A")
            self.mark(trace.end, AMBER, "B")
        elif trace.steps:
            first = trace.steps[0]
            last = trace.last_visible
            self.mark((first.x, first.y), CYAN, "1")
            self.mark((last.x, last.y), AMBER, "n")

    def draw_ship(self, panel_x: int, panel_width: int) -> None:
        self.draw_bitmap(self.ship_frame)
        selected = self.geometry.edge_decisions[self.edge_index]
        trace = self.ship_lines.get(selected.index)
        if trace:
            # Подсветка использует те же ПИКСЕЛИ, а не второй line-алгоритм.
            for step in trace.steps:
                if not step.in_bitmap:
                    continue
                pygame.draw.rect(self.screen, CYAN,
                                 (self.viewport.x + step.x * self.scale,
                                  self.viewport.y + step.y * self.scale,
                                  self.scale, self.scale))
            self.mark_line(trace)
        y = 118
        self.text(self.model.name, (panel_x, y), CYAN, self.title_font)
        y += 40
        drawn = sum(bool(line.steps) for line in self.ship_lines.values())
        clipped = sum(line.clip_status == "clipped" for line in self.ship_lines.values())
        precision = self.geometry.details["vertex_precision"]
        near_count = sum(shift > 0 for shift, _ in precision.values())
        for line in (f"Вершин: {len(self.model.vertices)} · рёбер: {len(self.model.edges)} · граней: {len(self.model.faces)}",
                     f"Рисуется рёбер: {drawn} / {len(self.model.edges)} · обрезано: {clipped}",
                     f"Глубина Z: {self.depth} · LOD: {self.geometry.distance}",
                     f"Проекция вершин: $AA2B — {near_count} · $AB73 — {len(precision) - near_count}",
                     f"Yaw {self.yaw % 360:.0f}°   Pitch {self.pitch % 360:.0f}°   Roll {self.roll % 360:.0f}°"):
            self.text(line, (panel_x, y), MUTED, self.small)
            y += 24
        if self.model.face_vertices:
            method = "через вершину грани" if self.use_face_vertices else "оригинальный тест Elite"
            self.text(f"Видимость: {method}  ·  F — сменить", (panel_x, y), CYAN, self.small)
            y += 24
        y += 14
        self.text(f"РЕБРО {self.edge_index:02d}", (panel_x, y), INK)
        y += 28
        edge = self.model.edges[self.edge_index]
        a, b = edge.faces
        face_label = f"Грани: {a}, {b}"
        if self.model.face_vertices and self.use_face_vertices:
            anchors = self.model.face_vertices
            face_label += f"  ·  опорные вершины: {anchors[a]}, {anchors[b]}"
        for line in (f"Вершины: {edge.vertex_a} → {edge.vertex_b}",
                     face_label,
                     f"Флаги: {self.geometry.face_flags[a]}, {self.geometry.face_flags[b]}",
                     f"Порог дальности: {edge.distance_limit}"):
            self.text(line, (panel_x, y), MUTED)
            y += 27
        reason = {
            "draw": "Проверки пройдены: передать весь отрезок на рисование.",
            "distance": "Дальность достигла порога этой детали.",
            "faces": "Оба связанных флага граней равны нулю.",
            "vertex_not_projected": "Одна из вершин не была спроецирована.",
            "behind_camera": "Конец отрезка помечен недопустимым.",
        }.get(selected.reason, selected.reason)
        status = "РИСУЕТСЯ" if selected.draw else "ПРОПУЩЕНО"
        if trace and trace.clip_status == "clipped":
            status = "ОБРЕЗАНО"
            last = trace.last_visible
            reason = (f"Пиксели в окне: 1 = {(trace.steps[0].x, trace.steps[0].y)}, n = {(last.x, last.y)}."
                      " Ниже — исходные проекции.")
        elif trace and not trace.steps:
            status = "ОТСЕЧЕНО"
            reason = "Клиппер $B044 завершился до первой записи пикселя."
        self.text(status, (panel_x, y), CYAN if status == "РИСУЕТСЯ" else AMBER)
        y = self.wrapped(reason, panel_x, y + 27, panel_width)
        if selected.start and selected.end:
            self.text(f"A ({selected.start.x}, {selected.start.y})", (panel_x, y + 7), CYAN, self.mono)
            self.text(f"B ({selected.end.x}, {selected.end.y})", (panel_x, y + 31), AMBER, self.mono)
            y += 64
        if selected.draw:
            y = self.wrapped("Enter — разобрать это ребро по пикселям", panel_x, y + 8, panel_width, CYAN)
        self.wrapped("Флаги граней выбирают ребро целиком. Перекрытие его середины другими поверхностями не проверяется.",
                     panel_x, y + 10, panel_width)

        base = self.viewport.bottom + 18
        self.text("← → / ↑ ↓ — вращение    Q / E — крен    колесо / + − — расстояние", (20, base), MUTED, self.small)
        self.text("[ ] — выбрать ребро    Enter — его пиксели    Space — автоповорот    R — сброс", (20, base + 26), MUTED, self.small)
        if self.model.face_vertices:
            self.text("P — ракурс с расхождением тестов граней", (20, base + 50), CYAN, self.small)
        self.wrapped(f"Фронтальный вид · Z от {minimum_depth(self.model_id)} до 6000. У границы окна 256 × 128 метки 1 и n показывают первый и последний видимые пиксели обрезанного ребра. Клавиша 3 открывает каталог моделей.",
                     20, base + 69, self.viewport.width, MUTED)
        y = base + 143
        self.text("ФЛАГИ ГРАНЕЙ" if self.model.faces else "КОНТУР БЕЗ ГРАНЕЙ", (20, y), INK, self.small)
        for index, value in enumerate(self.geometry.face_flags[:len(self.model.faces)]):
            rect = pygame.Rect(20 + index * 44, y + 28, 36, 37)
            pygame.draw.rect(self.screen, (29, 68, 76) if value else PANEL, rect, border_radius=3)
            self.text(str(index), (rect.x + 6, rect.y + 3), CYAN if value else MUTED, self.small)
            self.text(str(value), (rect.x + 11, rect.y + 19), CYAN if value else MUTED, self.small)

    def draw_zoom(self, fb: Framebuffer, selected, x: int, y: int, width: int) -> None:
        self.text("ПИКСЕЛЬНАЯ СЕТКА", (x, y), MUTED, self.small)
        y += 25
        cell = max(8, min(16, width // 19))
        cols, rows = 19, 9
        if selected:
            cx, cy = selected.x, selected.y
        elif self.line_trace.steps:
            cx, cy = self.line_trace.steps[0].x, self.line_trace.steps[0].y
        else:
            cx, cy = self.line_trace.start
        origin_x = max(0, min(256 - cols, cx - cols // 2))
        origin_y = max(0, min(128 - rows, cy - rows // 2))
        for gy in range(rows):
            for gx in range(cols):
                px, py = origin_x + gx, origin_y + gy
                rect = pygame.Rect(x + gx * cell, y + gy * cell, cell - 1, cell - 1)
                color = CYAN if fb.get_pixel(px, py) else PAPER
                pygame.draw.rect(self.screen, color, rect)
                if selected and px == selected.x and py == selected.y:
                    pygame.draw.rect(self.screen, AMBER, rect, 2)
        self.text(f"Окрестность ({cx}, {cy})" if 0 <= cx < 256 and 0 <= cy < 128 else "Точка вне видеобуфера", (x, y + rows * cell + 6), MUTED, self.small)

    def draw_line(self, panel_x: int, panel_width: int) -> None:
        trace = self.line_trace
        fb = Framebuffer()
        for step in trace.steps[:self.line_count]:
            if step.in_bitmap:
                fb.set_pixel(step.x, step.y)
        self.draw_bitmap(fb, CYAN)
        if self.grid and self.scale >= 3:
            # Это сетка интерфейса из прямоугольников, не рендер линии.
            for x in range(0, 256, 8):
                pygame.draw.rect(self.screen, (18, 29, 41),
                                 (self.viewport.x + x * self.scale, self.viewport.y, 1, self.viewport.height))
            for y in range(0, 128, 8):
                pygame.draw.rect(self.screen, (18, 29, 41),
                                 (self.viewport.x, self.viewport.y + y * self.scale, self.viewport.width, 1))
            # Пиксели заново поверх сетки, чтобы она не закрыла результат.
            for step in trace.steps[:self.line_count]:
                if not step.in_bitmap:
                    continue
                pygame.draw.rect(self.screen, CYAN,
                                 (self.viewport.x + step.x * self.scale, self.viewport.y + step.y * self.scale,
                                  self.scale, self.scale))
        self.mark_line(trace)
        current = trace.steps[self.line_count - 1] if self.line_count else None
        if current:
            self.mark((current.x, current.y), AMBER)

        y = 118
        self.text("РУЧНОЙ ПИКСЕЛЬНЫЙ ЦИКЛ", (panel_x, y), CYAN)
        self.text(self.line_source, (panel_x, y + 30), MUTED, self.small)
        y += 65
        lines = [f"A {trace.start}   B {trace.end}", f"ΔX {trace.delta_x}   ΔY {trace.delta_y}"]
        if trace.steps:
            lines.extend((f"Ведущая ось: {trace.major_axis.upper()}", f"Шаг дроби: {trace.slope} / 256"))
        lines.append(f"Записей: {self.line_count} / {len(trace.steps)}")
        for line in lines:
            self.text(line, (panel_x, y), INK, self.mono)
            y += 27
        y += 10
        if not trace.steps:
            y = self.wrapped("Полностью отсечено: пиксельный цикл не запускался.", panel_x, y, panel_width, AMBER)
        elif current:
            self.text(f"Пиксель ({current.x}, {current.y})", (panel_x, y), CYAN, self.mono)
            y += 29
            self.text(f"${current.address:04X}  mask {current.mask:08b}" + ("" if current.in_bitmap else "  вне буфера"), (panel_x, y), MUTED, self.mono)
            y += 29
            if current.has_next:
                self.text(f"E: {current.error_before} + {trace.slope}", (panel_x, y), INK, self.mono)
                self.text(f"→ {current.error_after}   carry = {int(current.carry)}", (panel_x, y + 27), AMBER, self.mono)
                y += 62
                action = "Шаг по двум осям" if current.carry else f"Шаг только по {trace.major_axis.upper()}"
                self.text(action, (panel_x, y), AMBER, self.small)
                y += 27
            else:
                y = self.wrapped("Последний пиксель: возврат до обновления E. Конечная точка B обычно не рисуется.", panel_x, y, panel_width, AMBER)
        else:
            y = self.wrapped("Начальное E = 128. Нажмите → для первой записи пикселя.", panel_x, y, panel_width, AMBER)
        self.draw_zoom(fb, current, panel_x, max(y + 16, 502), panel_width)

        base = self.viewport.bottom + 18
        self.text("← → — один шаг    Home / End — начало / всё    Space — анимация    R — сброс", (20, base), MUTED, self.small)
        self.text("Мышь — двигать A / B    P — учебные примеры    G — сетка", (20, base + 26), MUTED, self.small)
        self.text("Оригинальный цикл: $B434 / $B464 / $B485 / $B4B1", (20, base + 69), CYAN, self.small)
        code = ["записать пиксель: framebuffer[byte] |= mask",
                "если пиксели закончились: return",
                "total = E + slope;   carry = total >= 256",
                "E = total & 255;     шаг по ведущей оси",
                "если carry:         шаг по второй оси"]
        for i, line in enumerate(code):
            self.text(line, (20, base + 99 + i * 25), INK, self.mono)

    def catalog_cards(self) -> list[pygame.Rect]:
        width, height = self.screen.get_size()
        gap = 8
        card_w = (width - 40 - 4 * gap) // 5
        card_h = (height - 178 - 3 * gap) // 4
        return [pygame.Rect(20 + (i % 5) * (card_w + gap), 152 + (i // 5) * (card_h + gap),
                            card_w, card_h) for i in range(len(MODELS))]

    def draw_catalog(self) -> None:
        if not self.thumbnails:
            # Те же пиксели, что и в основном окне. Для карточки обрезаем пустые поля.
            for index, model in enumerate(MODELS):
                pose = pose_from_angles(*_initial_angles(index), initial_depth(index))
                geometry = render_geometry(model, pose, use_face_vertices=self.vertex_tests[index])
                surface = _surface(_ship_frame(_ship_lines(geometry)))
                surface.set_colorkey(PAPER)
                self.thumbnails.append(surface.subsurface(surface.get_bounding_rect()).copy())
        self.text("Щёлкните по модели или выберите её стрелками и нажмите Enter", (20, 120), MUTED, self.small)
        for i, (rect, model, thumb) in enumerate(zip(self.catalog_cards(), MODELS, self.thumbnails)):
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=4)
            if i == self.catalog_cursor:
                pygame.draw.rect(self.screen, CYAN, rect, 2, border_radius=4)
            self.text(f"{i:02d}", (rect.x + 10, rect.y + 7), MUTED, self.small)
            label = f"{model.name} (торговая)" if i == COBRA_ID else model.name
            self.text(label, (rect.x + 10, rect.bottom - 29), INK)
            w, h = thumb.get_size()
            scale = max(1, min((rect.width - 24) // w, (rect.height - 58) // h))
            thumb_scaled = pygame.transform.scale(thumb, (w * scale, h * scale))
            self.screen.blit(thumb_scaled, thumb_scaled.get_rect(center=(rect.centerx, rect.centery - 3)))

    def render(self) -> None:
        panel_x, panel_width = self.layout()
        self.screen.fill(BG)
        self.draw_header()
        if self.mode == "ship":
            self.draw_ship(panel_x, panel_width)
        elif self.mode == "catalog":
            self.draw_catalog()
        else:
            self.draw_line(panel_x, panel_width)
        pygame.display.flip()

    def move_point(self, pos) -> None:
        x = max(0, min(254, (pos[0] - self.viewport.x) // self.scale))
        y = max(0, min(126, (pos[1] - self.viewport.y) // self.scale))
        if self.dragging is None:
            self.dragging = min(range(2), key=lambda i:
                                (self.line_points[i].x - x) ** 2 + (self.line_points[i].y - y) ** 2)
        self.line_points[self.dragging] = ProjectedVertex.from_screen(x, y)
        self.line_source = "Свободный отрезок"
        self.line_count = 0
        self.line_running = False
        self.rebuild_line()

    def event(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(1220, event.w), max(800, event.h)), pygame.RESIZABLE)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, tab in enumerate(self.tabs):
                if tab.collidepoint(event.pos):
                    self.mode = ("ship", "line", "catalog")[index]
                    self.catalog_cursor = self.model_id
                    return True
            if self.mode == "catalog":
                for index, card in enumerate(self.catalog_cards()):
                    if card.collidepoint(event.pos):
                        self.choose_model(index)
                        return True
            if self.mode == "line" and self.viewport.collidepoint(event.pos):
                self.move_point(event.pos)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = None
        if event.type == pygame.MOUSEMOTION and self.mode == "line" and self.dragging is not None:
            self.move_point(event.pos)
        if event.type == pygame.MOUSEWHEEL and self.mode == "ship":
            self.zoom(event.y)
        if event.type != pygame.KEYDOWN:
            return True
        key = event.key
        if key == pygame.K_ESCAPE:
            if self.mode == "catalog":
                self.mode = "ship"
                return True
            return False
        if key in (pygame.K_TAB, pygame.K_1, pygame.K_2):
            self.mode = ("line" if self.mode == "ship" else "ship") if key == pygame.K_TAB else ("ship" if key == pygame.K_1 else "line")
        elif key == pygame.K_3:
            self.mode = "catalog"
            self.catalog_cursor = self.model_id
        elif self.mode == "catalog":
            moves = {pygame.K_LEFT: -1, pygame.K_RIGHT: 1, pygame.K_UP: -5, pygame.K_DOWN: 5}
            if key in moves:
                self.catalog_cursor = max(0, min(len(MODELS) - 1, self.catalog_cursor + moves[key]))
            elif key == pygame.K_RETURN:
                self.choose_model(self.catalog_cursor)
        elif self.mode == "ship":
            if key == pygame.K_SPACE:
                self.rotate_running = not self.rotate_running
            elif key == pygame.K_f and self.model.face_vertices:
                self.vertex_tests[self.model_id] = not self.use_face_vertices
                self.rotate_running = False
                self.thumbnails.clear()
            elif key == pygame.K_p and self.model.face_vertices:
                self.show_face_example()
                return True
            elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                self.edge_index = (self.edge_index + (-1 if key == pygame.K_LEFTBRACKET else 1)) % len(self.model.edges)
            elif key == pygame.K_RETURN:
                self.copy_ship_edge()
            elif key == pygame.K_r:
                self.yaw, self.pitch, self.roll = _initial_angles(self.model_id)
                self.depth = initial_depth(self.model_id)
                self.rotate_running = False
            elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.zoom(1)
                return True
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.zoom(-1)
                return True
            self.rebuild_ship()
        else:
            if key == pygame.K_SPACE:
                if not self.line_trace.steps:
                    return True
                if self.line_count == len(self.line_trace.steps):
                    self.line_count = 0
                self.line_running = not self.line_running
            elif key in (pygame.K_RIGHT, pygame.K_n):
                self.line_running = False
                self.line_count = min(len(self.line_trace.steps), self.line_count + 1)
            elif key == pygame.K_LEFT:
                self.line_running = False
                self.line_count = max(0, self.line_count - 1)
            elif key in (pygame.K_HOME, pygame.K_r):
                self.line_running = False
                self.line_count = 0
            elif key == pygame.K_END:
                self.line_running = False
                self.line_count = len(self.line_trace.steps)
            elif key == pygame.K_g:
                self.grid = not self.grid
            elif key == pygame.K_p:
                presets = [
                    ((100, 50), (110, 55), "Пример: slope 128, перенос через шаг"),
                    ((100, 50), (104, 55), "Особенность: ΔX=4, ΔY=5, ведущая X"),
                    ((127, 20), (127, 108), "Вертикаль: slope 0"),
                    ((220, 106), (30, 30), "Обратное направление"),
                    ((127, 63), (127, 63), "Совпавшие концы: один пиксель"),
                    ((34, 32), (214, 91), "Свободный отрезок"),
                ]
                self.preset_index = (self.preset_index + 1) % len(presets)
                a, b, self.line_source = presets[self.preset_index]
                self.line_points = [ProjectedVertex.from_screen(*a), ProjectedVertex.from_screen(*b)]
                self.line_count = 0
                self.line_running = False
                self.rebuild_line()
        return True

    def update(self, seconds: float) -> None:
        if self.mode == "ship":
            keys = pygame.key.get_pressed()
            dy = (int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])) * seconds * 60
            dp = (int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP])) * seconds * 60
            dr = (int(keys[pygame.K_e]) - int(keys[pygame.K_q])) * seconds * 60
            if self.rotate_running:
                dy += seconds * 22
            if dy or dp or dr:
                self.yaw += dy
                self.pitch += dp
                self.roll += dr
                self.rebuild_ship()
        elif self.mode == "line" and self.line_running:
            self.line_timer += seconds
            if self.line_timer >= 0.07:
                count = int(self.line_timer / 0.07)
                self.line_timer %= 0.07
                self.line_count = min(len(self.line_trace.steps), self.line_count + count)
                if self.line_count == len(self.line_trace.steps):
                    self.line_running = False


def main(argv=None) -> None:
    from math import isfinite
    from .space_assets import AssetCatalog, AssetError, FLYBY_SOURCE_NAMES, SOURCE_NAMES, TEXTURE_NAMES

    def nonnegative_seconds(value):
        try:
            seconds = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Ожидается число секунд") from exc
        if not isfinite(seconds) or seconds < 0:
            raise argparse.ArgumentTypeError("Число секунд должно быть конечным и неотрицательным")
        return seconds

    parser = argparse.ArgumentParser(description="Учебный ручной рендерер ZX Spectrum Elite")
    parser.add_argument("--mode", choices=("ship", "line", "catalog", "clip", "faces", "space", "flyby"), default="ship")
    parser.add_argument("--model", type=int, metavar="ID",
                        help="Номер модели в каталоге, например 19 для Dodo")
    parser.add_argument("--screenshot", type=Path, help="Сохранить кадр UI и выйти (для проверки)")
    parser.add_argument("--headless", action="store_true", help="Рендер без окна (для space/flyby — отдельный контекст OpenGL)")
    parser.add_argument("--no-hud", action="store_true", help="Скрыть интерфейс space/flyby")
    parser.add_argument("--model-source", choices=tuple(FLYBY_SOURCE_NAMES),
                        help="Набор геометрии для space/flyby; zx доступен только в flyby")
    parser.add_argument("--texture-source", choices=tuple(TEXTURE_NAMES), help="Родные карты модели или процедурная обшивка атласа")
    parser.add_argument("--source-interval", type=nonnegative_seconds, metavar="СЕК",
                        help="flyby: интервал смены наборов (по умолчанию 8 с; 0 — вручную)")
    parser.add_argument("--at", type=nonnegative_seconds, metavar="СЕК",
                        help="flyby: время воспроизводимого кадра с --headless или --screenshot")
    parser.add_argument("--list-sources", action="store_true", help="Показать установленные наборы, авторов, лицензии и номера моделей")
    args = parser.parse_args(argv)
    if args.mode != "flyby" and (args.source_interval is not None or args.at is not None):
        parser.error("--source-interval и --at относятся только к автопролёту (--mode flyby)")
    if args.at is not None and not (args.headless or args.screenshot):
        parser.error("--at требует --headless или --screenshot")
    if args.model_source == "zx" and args.mode != "flyby" and not args.list_sources:
        parser.error("Пиксельный каркас --model-source zx доступен только в --mode flyby")
    if args.list_sources:
        catalog = AssetCatalog()
        print("zx — только flyby: ZX · пиксельный каркас, модели 0–19")
        print("elite — Elite · наша обшивка, модели 0–19, процедурные материалы")
        for source in catalog.pack_ids:
            try:
                pack = catalog.pack(source)
            except AssetError as exc:
                parser.error(str(exc))
            print(f"{source} — {SOURCE_NAMES[source]}, модели {', '.join(map(str, sorted(pack)))}")
            for author, license_name in dict.fromkeys((s.author, s.license) for s in pack.values()):
                print(f"  {author} · {license_name}")
        if catalog.pack_ids:
            print("mixed — сборный флот из установленных наборов")
        print("flyby: порядок M и таймера — " + " → ".join(("zx", *catalog.sources, "zx")))
        print("flyby: место 20 остаётся пустым без гостевой модели Pioneer")
        return
    if args.mode in ("space", "flyby"):
        try:
            from .space_viewer import run_space
        except ModuleNotFoundError as exc:
            if exc.name not in ("numpy", "moderngl", "glcontext"):
                raise
            parser.exit(1, "Для space/flyby установите зависимости: python -m pip install -r requirements-space.txt\n")
        try:
            run_space(args.model, args.screenshot, args.headless, not args.no_hud,
                      args.model_source or ("zx" if args.mode == "flyby" else "elite"),
                      args.texture_source or "author", mode=args.mode,
                      source_interval=8.0 if args.source_interval is None else args.source_interval, at=args.at)
        except AssetError as exc:
            parser.error(str(exc))
        return
    if args.model_source or args.texture_source:
        parser.error("Выбор источника относится к режимам --mode space/flyby")
    if args.model is not None and args.model not in range(len(MODELS)):
        parser.error(f"Номер модели должен быть от 0 до {len(MODELS)-1}")
    if args.no_hud:
        parser.error("--no-hud относится к режимам --mode space/flyby")
    if args.mode == "faces" and args.model is not None and not MODELS[args.model].face_vertices:
        parser.error(f"У модели {MODELS[args.model].name} нет граней для сравнения")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    viewer = Viewer(start_mode="ship" if args.mode in ("clip", "faces") else args.mode)
    if args.mode == "clip":
        viewer.show_clipping_example(0 if args.model is None else args.model)
    elif args.mode == "faces":
        viewer.choose_model(DODO_ID if args.model is None else args.model)
        viewer.show_face_example()
    elif args.model is not None:
        viewer.choose_model(args.model)
        viewer.mode = args.mode
    if args.screenshot:
        if args.mode == "line":
            viewer.line_count = 87
        viewer.render()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(viewer.screen, str(args.screenshot))
        pygame.quit()
        return
    clock = pygame.time.Clock()
    running = True
    while running:
        seconds = min(clock.tick(60) / 1000.0, 0.1)
        for event in pygame.event.get():
            running = viewer.event(event)
            if not running:
                break
        viewer.update(seconds)
        viewer.render()
    pygame.quit()
