"""Материалы смешанного флота и детерминированное звёздное окружение.

Новый OpenGL-режим использует исходные фасеты, но самостоятельное оформление:
собственные панели и ливреи, без заимствованных эмблем и текстур. Цвета материалов
и HDR-cubemap хранятся в линейном пространстве; шейдеры выводят готовый sRGB.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import moderngl
import numpy as np
import pygame

from .model import COBRA_ID


_SHADERS = Path(__file__).with_name("shaders")
_SUN = np.array((-0.62, 0.48, 0.62), dtype="f4")
_SUN /= np.linalg.norm(_SUN)
SUN_DIRECTION = tuple(float(value) for value in _SUN)
_SKY_SIZE = 1536
_SURFACE_SIZE = 2048

# Место бортового номера на правом плече корпуса (X, Z, ширина трафарета).
_REGISTRATION = {8: (.24, .01, .25), 9: (.55, -.11, .30), 12: (.55, -.11, .30),
                 10: (.21, .13, .27), 11: (.21, .13, .27), 13: (.66, -.31, .26),
                 14: (.21, -.64, .23), 16: (.45, -.14, .26), 17: (.27, -.08, .27)}


def _stencils(ctx):
    # Собственные бортовые номера. Шрифт входит в pygame и одинаков на всех ОС.
    pygame.font.init()
    sheet = pygame.Surface((512, 512))
    font = pygame.font.Font(None, 56)
    for index in _REGISTRATION:
        x, y = index % 4 * 128, index // 4 * 64
        number = font.render(f"{index:02d}", True, (255, 255, 255))
        sheet.blit(number, (x + 36, y + 6))
        for dx in (9, 17):
            pygame.draw.polygon(sheet, (196, 196, 196),
                                ((x+dx, y+18), (x+dx+5, y+18), (x+dx+12, y+40), (x+dx+7, y+40)))
        pygame.draw.line(sheet, (228, 228, 228), (x+36, y+51), (x+103, y+51), 2)
        for dx in (90, 97, 104, 111):
            pygame.draw.line(sheet, (205, 205, 205), (x+dx, y+19), (x+dx, y+28), 2)
    red = np.frombuffer(pygame.image.tobytes(sheet, "RGB", True), dtype="u1")[::3]
    texture = ctx.texture(sheet.get_size(), 1, red.tobytes())
    texture.build_mipmaps()
    texture.anisotropy = min(8, ctx.max_anisotropy)
    return texture


@lru_cache(maxsize=128)
def _linear(hex_color: str) -> tuple[float, float, float]:
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype="f4") / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return tuple(float(value) for value in rgb)


@dataclass(frozen=True)
class _Style:
    hull: str
    secondary: str
    accent: str
    wear: float
    roughness: float
    metallic: float
    stripe: int
    scale: float = 1.0
    engine: str = "8edff8"
    windows: str = "ffe2af"
    metal: str = "999d9d"


# Индексы совпадают с MODELS; одинаковые каркасы торговцев и пиратов отличаются
# также материалом, насыщенностью, масштабом панелей и характером износа.
_STYLES = (
    _Style("c8c9bf", "898f91", "b95341", .32, .56, .31, 2, 2.0),  # Coriolis
    _Style("dfdccb", "a8afb0", "dc7d33", .13, .40, .28, 2, .68),  # Missile
    _Style("e3d6b6", "a49e8a", "d06332", .22, .60, .18, 0, .72),  # Escape pod
    _Style("84745f", "4f5658", "a58650", .98, .78, .48, 3, 1.1),  # Splinter
    _Style("b4a274", "747e7a", "c4963e", .70, .68, .40, 2, 1.2),  # Canister
    _Style("827b69", "514f4b", "8f826a", .95, .93, .03, 0),  # Boulder
    _Style("777579", "625f64", "857e74", .95, .93, .03, 0),  # Asteroid
    _Style("8b8171", "53575a", "c39d59", .83, .85, .12, 2, 1.4),  # Hermit
    _Style("a9bac6", "627788", "28566f", .13, .39, .35, 1, 1.3),  # Viper
    _Style("d6d6c6", "9aaba9", "337d7f", .20, .48, .32, 1, 1.15),  # Cobra
    _Style("b7a78a", "877d6c", "964c3a", .69, .68, .43, 0, 1.5),  # Python
    _Style("585f61", "858471", "b0904b", .94, .73, .52, 3, 1.5),  # Python pirate
    _Style("68656c", "928a86", "9f493d", .86, .65, .47, 1, 1.15),  # Cobra pirate
    _Style("c3d3c8", "859f9d", "c67842", .16, .42, .27, 0, 1.25),  # Asp
    _Style("d7d5cb", "8b9da8", "294a67", .09, .32, .42, 3, 1.05),  # FdL
    _Style("96957a", "555f53", "60774c", .34, .50, .60, 2, 1.6, "9bf0a1", "c2eda6"),
    _Style("92a4ac", "65787e", "ba7447", .48, .57, .43, 0, 1.1),  # Sidewinder
    _Style("c4c2ae", "8d9291", "b56136", .63, .59, .37, 1, 1.0),  # Krait
    _Style("999575", "606852", "5d8161", .47, .53, .62, 2, .82, "9befac", "c2eda6"),
    _Style("c9d0c6", "839997", "3d7776", .26, .51, .29, 2, 2.2),  # Dodo
)


@dataclass(frozen=True)
class _ClassicFinish:
    """Авторские покрытия только исходных каркасов, отдельно от импортных T/M.

    layout: семейство силовых секций, доля металла, масштаб и сдвиг.
    response: шероховатость металла и стекла.
    damage: сколы кромок и обтирка краски.
    """
    layout: tuple[float, float, float, float]
    response: tuple[float, float]
    damage: tuple[float, float]
    paint_roughness: float
    wear: float
    metal: str
    glass: str


# Цвета и трафареты прежней ливреи остаются в _STYLES. Эта таблица описывает
# назначение деталей, а не перекрашивает весь флот одним материалом. В частности,
# грузовые продольные накладки Python отличаются от поперечных ремонтных поясов
# пирата; у полицейского корабля металл узкий и целый, у разведчика — на плечах.
_CLASSIC_FINISHES = (
    _ClassicFinish((4, .27, 1.22, .00), (.40, .49), (.55, .95), .79, .40, "8b959b", "65766e"),  # Coriolis
    _ClassicFinish((1, .34, 1.45, -.34), (.31, .40), (.24, .32), .70, .13, "b8b7ae", "647681"),  # Missile: крепления и обтекатель
    _ClassicFinish((1, .14, .68, .12), (.49, .53), (.39, .64), .82, .25, "9b9c8b", "8b9276"),  # Escape pod: теплозащита
    _ClassicFinish((4, .00, 1.00, .00), (.74, .50), (.00, .00), .93, .98, "81766a", "707575"),  # Splinter: природный камень
    _ClassicFinish((3, .47, .82, .17), (.54, .52), (.86, 1.30), .83, .82, "9a9785", "687b78"),  # Canister: грузовые рёбра
    _ClassicFinish((4, .00, 1.00, .00), (.90, .50), (.00, .00), .93, .95, "81766a", "707575"),  # Boulder
    _ClassicFinish((4, .00, 1.00, .00), (.90, .50), (.00, .00), .93, .95, "81766a", "707575"),  # Asteroid
    _ClassicFinish((4, .10, .85, .02), (.58, .54), (.71, 1.30), .86, .83, "8e8c79", "8c9471"),  # Hermit: камень, металл только в шахте
    _ClassicFinish((2, .13, 1.50, .20), (.25, .37), (.20, .24), .73, .12, "aebbc3", "83939f"),  # Viper: чистые носовые усиления
    _ClassicFinish((0, 1.00, 1.00, .00), (.29, .46), (.46, .42), .76, .20, "999d9d", "5f676a"),  # Cobra: исходные плечи
    _ClassicFinish((3, .45, .80, -.08), (.51, .50), (.75, 1.45), .84, .82, "a4987e", "7a8981"),  # Python: обтирка при грузовых работах
    _ClassicFinish((1, .65, 1.10, -.18), (.43, .56), (1.50, .82), .87, .95, "9b9c8d", "777b68"),  # Python pirate: перебитые пояса и сколы
    _ClassicFinish((2, .48, .74, -.15), (.36, .52), (1.55, .78), .81, .91, "a39591", "7f7377"),  # Cobra pirate: диагональные боевые сколы
    _ClassicFinish((2, .26, 1.03, -.06), (.33, .43), (.35, .64), .77, .28, "9db3ad", "829b8f"),  # Asp: экспедиционные плечи
    _ClassicFinish((1, .12, 1.30, .24), (.21, .34), (.12, .18), .65, .08, "b7c0c6", "77858d"),  # Fer-de-Lance: ухоженные редкие вставки
    _ClassicFinish((4, .52, 1.18, .07), (.37, .59), (.55, .48), .74, .35, "a9ac80", "8eab76"),  # Thargoid: радиальные сенсорные секции
    _ClassicFinish((1, .33, .79, -.21), (.46, .51), (.72, .87), .80, .55, "9da8aa", "8c8a78"),  # Sidewinder: поперечные рабочие секции
    _ClassicFinish((2, .35, .88, .01), (.42, .48), (.95, .78), .79, .68, "aaa18a", "8f8173"),  # Krait: крылья и треугольные стёкла
    _ClassicFinish((4, .68, .73, -.12), (.32, .62), (.62, .43), .78, .47, "a0a47b", "88a571"),  # Thargon: сенсор, без человеческой кабины
    _ClassicFinish((4, .36, 1.65, .13), (.43, .50), (.48, .88), .79, .32, "91a5a2", "7b9487"),  # Dodo: технологические секторы станции
)


def _noise2(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Периодический плавный шум без scipy и зависимости от платформы."""
    grid = rng.random((cells, cells), dtype="f4")
    coordinates = np.arange(size, dtype="f4") * (cells / size)
    base = coordinates.astype(np.int32)
    f = coordinates - base
    f = f * f * (3.0 - 2.0 * f)
    along_x = grid[:, base] * (1.0 - f) + grid[:, (base + 1) % cells] * f
    return along_x[base, :] * (1.0 - f[:, None]) + along_x[(base + 1) % cells, :] * f[:, None]


def _surface_bytes() -> bytes:
    """RGBA: тон панели, рельеф, повреждения, углубления оборудования.

    Двоичное разбиение даёт крупные и мелкие прямоугольные секции вместо сетки
    одинаковых плиток. Мелкие детали запекаются до создания mipmap-цепочки.
    """
    size = _SURFACE_SIZE
    rng = np.random.default_rng(0xE117E)
    image = np.empty((size, size, 4), dtype=np.uint8)
    image[:] = (102, 76, 65, 213)
    clouds = _noise2(size, 23, rng) * .64 + _noise2(size, 79, rng) * .36
    grain = rng.random((size, size), dtype="f4")
    image[:, :, 2] = np.clip((clouds ** 2 * .63 + grain * .13) * 255, 0, 255).astype("u1")
    leaves = []

    def split(x: int, y: int, w: int, h: int) -> None:
        if max(w, h) < 385 and (min(w, h) < 133 or rng.random() < .66):
            leaves.append((x, y, w, h))
            return
        vertical = w > h * 1.12 or (w > h * .8 and rng.random() > .5)
        extent = w if vertical else h
        if extent < 150:
            leaves.append((x, y, w, h))
            return
        cut = int(np.clip(int(extent * rng.uniform(.31, .69) / 8) * 8, 56, extent - 56))
        if vertical:
            split(x, y, cut, h)
            split(x + cut, y, w - cut, h)
        else:
            split(x, y, w, cut)
            split(x, y + cut, w, h - cut)

    split(0, 0, size, size)
    for x, y, w, h in leaves:
        gap = 1 if rng.random() < .83 else 2
        x0, y0, x1, y1 = x + gap, y + gap, x + w - gap, y + h - gap
        tone = int(rng.integers(123, 179))
        if rng.random() < .14:
            tone = int(rng.integers(111, 194))
        patch = image[y0:y1, x0:x1]
        patch[:, :, 0] = np.clip(tone + (grain[y0:y1, x0:x1] - .5) * 9 +
                                    (clouds[y0:y1, x0:x1] - .5) * 8, 0, 255).astype("u1")
        patch[:, :, 1] = (157 + grain[y0:y1, x0:x1] * 4).astype("u1")
        patch[:, :, 3] = 255
        # Узкие фрезерованные стыки: крупная секция остаётся визуально цельной.
        image[y0:y0 + 1, x0:x1, 1] = 143
        image[y1 - 1:y1, x0:x1, 1] = 172
        image[y0:y1, x0:x0 + 1, 1] = 143
        image[y0:y1, x1 - 1:x1, 1] = 172
        if rng.random() < .58:
            # Внутри крупных листов — неполные тонкие линии, а не новая рамка.
            sy = y0 + h // 3
            end = min(x1 - 5, x0 + int(w * rng.uniform(.3, .8)))
            image[sy:sy + 1, x0 + 5:end, 1] = 128
            image[sy:sy + 1, x0 + 5:end, 3] = 226
        if rng.random() < .72:
            # Двойные сервисные защёлки — короткие прямоугольники, не «заклёпочная сетка».
            for cx, cy in ((x0 + 9, y0 + 9), (x1 - 13, y1 - 12)):
                image[cy:cy + 2, cx:cx + 3, (0, 1, 3)] = (110, 104, 158)
                image[cy - 1:cy, cx:cx + 3, 1] = 182
        if min(w, h) > 130 and rng.random() < .52:
            # Небольшие плотные «острова» механики разделены спокойной обшивкой.
            bx = x0 + int((x1 - x0) * rng.uniform(.15, .4))
            by = y0 + int((y1 - y0) * rng.uniform(.2, .5))
            bw = min(int(rng.integers(65, 191)), x1 - bx - 10)
            bh = min(int(rng.integers(45, 131)), y1 - by - 10)
            image[by:by + bh, bx:bx + bw, (0, 1, 3)] = (118, 88, 150)
            # Рамка, шины, рёбра охлаждения и ряд отдельных кассет.
            image[by + 2:by + 4, bx + 2:bx + bw - 2, (1, 3)] = (182, 245)
            image[by + bh - 4:by + bh - 2, bx + 2:bx + bw - 2, (1, 3)] = (182, 245)
            for sx in range(bx + 5, bx + bw // 2, 5):
                image[by + 7:by + bh - 6, sx:sx + 2, (0, 1, 3)] = (154, 175, 224)
            for sy in range(by + 7, by + bh - 9, 11):
                cx = bx + bw // 2 + 3
                image[sy:sy + 7, cx:bx + bw - 6, (0, 1, 3)] = (139, 140, 220)
                image[sy + 2:sy + 3, cx + 3:bx + bw - 9, 3] = 97
                image[sy + 4:sy + 6, cx + 2:cx + 4, 0] = 191
        if rng.random() < .59:
            # Малый сервисный люк с четырьмя креплениями, без жирного контура.
            bx, by = x0 + 14, y0 + 20
            bw, bh = min(int(rng.integers(27, 65)), w // 3), min(int(rng.integers(21, 46)), h // 3)
            image[by:by + bh, bx:bx + bw, 0] = int(rng.integers(126, 185))
            image[by:by + bh, bx:bx + bw, 1] = 168
            image[by:by + 1, bx:bx + bw, 1] = 117
            image[by:by + bh, bx:bx + 1, 1] = 117
            image[by + bh // 2:by + bh // 2 + 2, bx + bw // 2 - 3:bx + bw // 2 + 3, 3] = 118
            for cx, cy in ((bx + 3, by + 3), (bx + bw - 4, by + 3),
                           (bx + 3, by + bh - 4), (bx + bw - 4, by + bh - 4)):
                image[cy:cy + 1, cx:cx + 2, (1, 3)] = (127, 160)
        if rng.random() < .63:
            # Собственные цифровые трафареты и сервисные подписи, без логотипов.
            glyphs = ("111101101101111", "010110010010111", "111001111100111",
                      "111001111001111", "101101111001001", "111100111001111",
                      "111100111101111", "111001010010010", "111101111101111", "111101111001111")
            bx, by = x1 - min(65, w // 2), y1 - 29
            for digit in range(3):
                glyph = glyphs[int(rng.integers(0, 10))]
                for gy in range(5):
                    for gx in range(3):
                        if glyph[gy * 3 + gx] == "1":
                            image[by + gy * 2:by + gy * 2 + 2,
                                  bx + digit * 9 + gx * 2:bx + digit * 9 + gx * 2 + 2, 3] = 138
            for line in range(3):
                length = int(rng.integers(5, 17))
                image[by + line * 3:by + line * 3 + 1, bx + 30:bx + 30 + length, 3] = 181
        # Сколы не обводят целиком каждую панель: несколько коротких участков.
        for _ in range(int(rng.integers(1, 5))):
            cx = int(rng.integers(x0, max(x0 + 1, x1 - 10)))
            length = min(int(rng.integers(4, 34)), x1 - cx)
            cy = y0 + 2 if rng.random() < .5 else y1 - 3
            image[cy:cy + 1, cx:cx + length, 2] = int(rng.integers(215, 254))

    # Тонкие случайные следы обслуживания и направленные потёртости.
    for _ in range(3100):
        x, y = rng.integers(2, size - 50, size=2)
        length = int(rng.integers(2, 17))
        image[y:y + 1, x:x + length, 2] = int(rng.integers(186, 250))
    return image.tobytes()


def _surface_slope_bytes(surface: bytes) -> bytes:
    """Непрерывно интерполируемый наклон исходного G-height, dH/dUV.

    Центральный stencil даёт симметричную фаску с тем же перепадом высоты.
    Производная вычисляется в пространстве карты до экранной фильтрации:
    билинейная интерполяция slope не повторяет ступени экранных 2×2 quad.
    Периодические края и единицы UV согласованы с исходной repeat-текстурой.
    """
    height = np.frombuffer(surface, dtype="u1").reshape(_SURFACE_SIZE, _SURFACE_SIZE, 4)[:, :, 1]
    height = height.astype("f4") / 255.0
    du = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    dv = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    return (np.stack((du, dv), axis=-1) * (_SURFACE_SIZE * 0.5)).astype("f2").tobytes()


def _noise3(x: np.ndarray, y: np.ndarray, z: np.ndarray, grid: np.ndarray) -> np.ndarray:
    size = grid.shape[0]
    ix, iy, iz = np.floor(x).astype("i2"), np.floor(y).astype("i2"), np.floor(z).astype("i2")
    fx, fy, fz = x - ix.astype("f4"), y - iy.astype("f4"), z - iz.astype("f4")
    fx, fy, fz = (value * value * (3.0 - 2.0 * value) for value in (fx, fy, fz))
    x0, x1 = ix % size, (ix + 1) % size
    y0, y1 = iy % size, (iy + 1) % size
    z0, z1 = iz % size, (iz + 1) % size
    a = grid[x0, y0, z0] * (1 - fx) + grid[x1, y0, z0] * fx
    b = grid[x0, y1, z0] * (1 - fx) + grid[x1, y1, z0] * fx
    c = grid[x0, y0, z1] * (1 - fx) + grid[x1, y0, z1] * fx
    d = grid[x0, y1, z1] * (1 - fx) + grid[x1, y1, z1] * fx
    return ((a * (1 - fy) + b * fy) * (1 - fz) + (c * (1 - fy) + d * fy) * fz).astype("f4")


def _cube_direction(face: int, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    one = np.ones_like(s)
    # Порядок граней и оси в точности соответствуют samplerCube OpenGL.
    axes = ((one, -t, -s), (-one, -t, s), (s, one, t),
            (s, -one, -t), (s, -t, one), (-s, -t, -one))
    direction = np.stack(axes[face], axis=-1)
    return direction / np.linalg.norm(direction, axis=-1, keepdims=True)


def _cube_project(face: int, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = directions.T
    mappings = ((-z, -y, x), (z, -y, -x), (x, z, y),
                (x, -z, -y), (x, -y, z), (-x, -y, -z))
    s, t, depth = mappings[face]
    divisor = np.where(depth > 0, depth, 1)
    return s / divisor, t / divisor, depth


def _splat_stars(image: np.ndarray, face: int, stars: np.ndarray,
                 magnitudes: np.ndarray, colors: np.ndarray) -> None:
    size = image.shape[0]
    s, t, depth = _cube_project(face, stars)
    visible = (depth > 0) & (np.abs(s) < 1.025) & (np.abs(t) < 1.025)
    for px, py, magnitude, color in zip((s[visible] + 1) * .5 * size - .5,
                                       (t[visible] + 1) * .5 * size - .5,
                                       magnitudes[visible], colors[visible]):
        sigma = .25 if magnitude < .8 else .36 + min(float(magnitude) ** .45 * .045, .18)
        radius = 1 if magnitude < .8 else min(11, int(4 + magnitude ** .5))
        x0, x1 = max(0, int(px) - radius), min(size, int(px) + radius + 2)
        y0, y1 = max(0, int(py) - radius), min(size, int(py) + radius + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        dy, dx = np.mgrid[y0:y1, x0:x1].astype("f4")
        dx, dy = dx - px, dy - py
        rr = dx * dx + dy * dy
        psf = np.exp(-rr / (2 * sigma * sigma))
        psf /= max(float(psf.sum()), .00001)
        if magnitude > .8:
            psf += .007 * np.exp(-rr / (2 * (sigma * 4.5) ** 2))
            psf += .010 * np.exp(-np.abs(dx) * 4.0 - np.abs(dy) / 2.1)
            psf += .010 * np.exp(-np.abs(dy) * 4.0 - np.abs(dx) / 2.1)
        image[y0:y1, x0:x1] += psf[:, :, None] * color * magnitude


def _sky_faces(size: int):
    """Шесть HDR-граней из единого сферического поля; нет швов туманности."""
    rng = np.random.default_rng(0x5AACE)
    grid = rng.random((32, 32, 32), dtype="f4")
    galactic_normal = np.array((.88, .46, .11), dtype="f4")
    galactic_normal /= np.linalg.norm(galactic_normal)
    stars = rng.normal(size=(10500, 3)).astype("f4")
    stars /= np.linalg.norm(stars, axis=1, keepdims=True)
    # Много слабых звёзд, небольшая популяция отчётливых ярких светил.
    magnitudes = (0.008 + rng.random(len(stars), dtype="f4") ** 7 * .42)
    bright = rng.choice(len(stars), 38, replace=False)
    magnitudes[bright] = rng.uniform(.9, 6.0, len(bright))
    color_choices = np.array(((.80, .88, 1.0), (1.0, .94, .81), (1.0, .77, .55),
                              (.69, .82, 1.0), (1.0, .99, .94)), dtype="f4")
    colors = color_choices[rng.integers(0, len(color_choices), len(stars))]
    # Бледкая мелкая звёздная пыль внутри галактической полосы.
    latitude_stars = np.abs(stars @ galactic_normal)
    magnitudes *= .72 + np.exp(-(latitude_stars / .22) ** 2) * .58
    coordinates = (np.arange(size, dtype="f4") + .5) * (2.0 / size) - 1.0
    s, t = np.meshgrid(coordinates, coordinates)
    for face in range(6):
        direction = _cube_direction(face, s, t)
        x, y, z = (direction[:, :, index] for index in range(3))
        latitude = x * galactic_normal[0] + y * galactic_normal[1] + z * galactic_normal[2]
        large = _noise3(x * 4.1 + 9.0, y * 4.1 + 7.0, z * 4.1 + 4.0, grid)
        medium = _noise3(x * 11.7 + 2.0, y * 11.7 + 6.0, z * 11.7 + 8.0, grid)
        small = _noise3(x * 29.5 + 2.0, y * 29.5 + 6.0, z * 29.5 + 8.0, grid)
        fine = _noise3(x * 68.0, y * 68.0, z * 68.0, grid)
        cloud = large * .48 + medium * .29 + small * .16 + fine * .07
        warp = (large - .5) * .16 + (medium - .5) * .045
        band = np.exp(-((latitude + warp) / .24) ** 2)
        dust = np.exp(-((latitude + warp * 1.65 + .025) / .040) ** 2)
        density = band * np.maximum(cloud - .19, 0) ** 1.25 * (1 - dust * .87)
        rgb = np.empty((size, size, 3), dtype="f4")
        rgb[:] = (.00018, .00025, .00045)
        rgb += density[:, :, None] * np.array((.024, .035, .059), dtype="f4")
        # Сдержанные пылевые облака: сине-серый и едва заметный тёплый водород.
        warm = band * np.maximum(large - .46, 0) * medium
        rgb += warm[:, :, None] * np.array((.058, .024, .013), dtype="f4")
        _splat_stars(rgb, face, stars, magnitudes, colors)

        cosine = np.clip(x * _SUN[0] + y * _SUN[1] + z * _SUN[2], -1, 1)
        angle = np.arccos(cosine)
        corona = np.exp(-angle / .018) * .22 + np.exp(-(angle / .065) ** 2) * .010
        disk = np.clip((.0064 - angle) / .0012, 0, 1)
        rgb += corona[:, :, None] * np.array((1.0, .69, .37), dtype="f4")
        rgb += disk[:, :, None] * np.array((22.0, 19.0, 14.0), dtype="f4")
        # Узкая корона, без крупной фотографической линзы поверх кораблей.
        rgba = np.ones((size, size, 4), dtype="f2")
        rgba[:, :, :3] = rgb
        yield rgba.tobytes()


class SpaceLook:
    """Контракт визуального слоя для OpenGL-атласа.

    ``hull`` принимает N×9 float32: position, normal, uv, material. Рендерер
    выставляет u_view_proj/u_model/u_eye. После configure_model() можно рисовать
    корпус. Матрица draw_sky() — обычная математическая numpy-матрица 4×4.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.hull = ctx.program(
            vertex_shader=(_SHADERS / "space_hull.vert").read_text(),
            fragment_shader=(_SHADERS / "space_hull.frag").read_text(),
        )
        self._sky_program = ctx.program(
            vertex_shader=(_SHADERS / "space_sky.vert").read_text(),
            fragment_shader=(_SHADERS / "space_sky.frag").read_text(),
        )
        self._sky_vao = ctx.vertex_array(self._sky_program, [])
        self._stencils = _stencils(ctx)
        surface = _surface_bytes()
        self._surface = ctx.texture((_SURFACE_SIZE, _SURFACE_SIZE), 4, surface)
        self._surface.repeat_x = self._surface.repeat_y = True
        self._surface.build_mipmaps()
        self._surface.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._surface.anisotropy = min(8.0, ctx.max_anisotropy)
        self._surface_slope = ctx.texture((_SURFACE_SIZE, _SURFACE_SIZE), 2,
                                          _surface_slope_bytes(surface), dtype="f2")
        self._surface_slope.repeat_x = self._surface_slope.repeat_y = True
        self._surface_slope.build_mipmaps()
        self._surface_slope.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._surface_slope.anisotropy = min(8.0, ctx.max_anisotropy)
        # На Apple Metal RGB16F + NPOT + glGenerateMipmap повреждает часть
        # базового уровня +X. RGBA16F проверен кодированием направлений сферы;
        # дополнительный канал сохраняет корректными и base, и mip-уровни.
        self._sky = ctx.texture_cube((_SKY_SIZE, _SKY_SIZE), 4, dtype="f2")
        for face, pixels in enumerate(_sky_faces(_SKY_SIZE)):
            self._sky.write(face, pixels)
        self._sky.build_mipmaps()
        self._sky.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        # GL_TEXTURE_CUBE_MAP_SEAMLESS входит в Core начиная с 3.2.
        ctx.enable_direct(0x884F)
        self.hull["u_surface"].value = 0
        if "u_surface_slope" in self.hull:
            self.hull["u_surface_slope"].value = 9
        self.hull["u_stencils"].value = 3
        self._imported_textures = OrderedDict()
        self._imported_texture_bytes = 0
        self._imported_texture_budget = 384 * 1024 * 1024
        self._imported_texture_limit = 256
        self._white = ctx.texture((1, 1), 3, bytes((255, 255, 255)))
        for unit, name in enumerate(("diffuse", "normal", "specular", "emission", "gloss"), 4):
            self.hull[f"u_{name}_map"].value = unit
            self._white.use(unit)
        self.hull["u_sky"].value = 1
        self.hull["u_sun_direction"].value = SUN_DIRECTION
        self.hull["u_shadow"].value = 2
        self.hull["u_shadow_enabled"].value = False
        self._sky_program["u_sky"].value = 1
        self.configure_model(9)

    def configure_model(self, model_id: int, *, imported=False, authored=False, cross_bitangent=False) -> None:
        """Установить самостоятельную ливрею модели и привязать общие текстуры."""
        style = _STYLES[model_id] if model_id < len(_STYLES) else _STYLES[13]
        classic = 0 <= model_id < len(_CLASSIC_FINISHES) and not (imported or authored)
        finish = _CLASSIC_FINISHES[model_id] if classic else _CLASSIC_FINISHES[13]
        uniforms = {
            "u_imported": imported,
            "u_author_maps": authored,
            "u_cross_bitangent": cross_bitangent,
            # Только кабина №9 имеет собственную модель интерьера; покрытия
            # всех остальных исходных каркасов задаются отдельным профилем.
            "u_cobra_finishes": model_id == COBRA_ID and not (imported or authored),
            "u_classic_finishes": classic,
            "u_finish_layout": finish.layout,
            "u_finish_response": finish.response,
            "u_damage_profile": finish.damage,
            "u_glass_tint": _linear(finish.glass),
            "u_part_type": 0,
            "u_hull_color": _linear(style.hull),
            "u_secondary_color": _linear(style.secondary),
            "u_accent_color": _linear(style.accent),
            "u_metal_color": _linear(finish.metal if classic else style.metal),
            "u_engine_color": _linear(style.engine),
            "u_window_color": _linear(style.windows),
            "u_uv_offset": ((model_id * .173) % 1, (model_id * .317) % 1),
            "u_patch": _REGISTRATION.get(model_id, (0., 0., 0.)),
            "u_stencil_tile": (model_id % 4, 7 - model_id // 4),
            "u_texture_scale": style.scale,
            "u_wear": finish.wear if classic else style.wear,
            "u_roughness": finish.paint_roughness if classic else style.roughness,
            "u_metallic": style.metallic,
            "u_marking_style": float(style.stripe),
            "u_seed": model_id * .73,
        }
        for name, value in uniforms.items():
            self.hull[name].value = value
        self._surface.use(location=0)
        self._surface_slope.use(location=9)
        self._sky.use(location=1)
        self._stencils.use(location=3)

    def use_material(self, material):
        maps = ((material.diffuse, None), (material.normal, (128, 128, 255)),
                (material.specular, (255, 255, 255)), (material.emission, (0, 0, 0)),
                (material.gloss_map, (255, 255, 255)))
        for unit, (path, neutral) in enumerate(maps, 4):
            key = path or neutral
            if key not in self._imported_textures:
                if path:
                    image = pygame.image.load(str(path))
                    # UV сетки имеют начало снизу слева, строки PNG/JPEG идут сверху.
                    tex = self.ctx.texture(image.get_size(), 3, pygame.image.tobytes(image, "RGB", True))
                    tex.build_mipmaps()
                    tex.anisotropy = min(8, self.ctx.max_anisotropy)
                else:
                    tex = self.ctx.texture((1, 1), 3, bytes(neutral))
                # Учитываем RGB8 как RGBA8 с mipmap: драйвер вправе хранить
                # выравненный канал. Бюджета хватает всему текущему mixed,
                # поэтому обычный обзор не вытесняет сам себя каждый кадр.
                size = tex.width * tex.height * 4
                self._imported_texture_bytes += size * 4 // 3 if path else size
                self._imported_textures[key] = tex
            self._imported_textures.move_to_end(key)
            self._imported_textures[key].use(unit)
        protected = {path or neutral for path, neutral in maps}
        for key in tuple(self._imported_textures):
            if (self._imported_texture_bytes <= self._imported_texture_budget and
                    len(self._imported_textures) <= self._imported_texture_limit):
                break
            if key not in protected:
                tex = self._imported_textures.pop(key)
                size = tex.width * tex.height * 4
                self._imported_texture_bytes -= size * 4 // 3 if not isinstance(key, tuple) else size
                tex.release()
        self.hull["u_gloss"].value = material.gloss
        self.hull["u_glow_gain"].value = material.emission_strength
        self.hull["u_specular_tint"].value = material.specular_color

    def release_imported_textures(self):
        for texture in self._imported_textures.values():
            texture.release()
        self._imported_textures.clear()
        self._imported_texture_bytes = 0
        # Драйвер Mac проверяет комплектность всех sampler даже в выключенной ветви.
        for unit in range(4, 9):
            self._white.use(unit)

    def draw_sky(self, inv_view_proj: np.ndarray, eye) -> None:
        """Нарисовать cubemap; depth/cull остаются выключенными для вызывающего."""
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        matrix = np.asarray(inv_view_proj, dtype="f4").reshape(4, 4)
        self._sky_program["u_inv_view_proj"].write(matrix.T.tobytes())
        self._sky_program["u_eye"].value = tuple(float(value) for value in eye)
        self._sky.use(location=1)
        self._sky_vao.render(mode=moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        """Освободить GL-ресурсы до закрытия контекста."""
        self.release_imported_textures()
        for resource in (self._sky_vao, self._surface, self._surface_slope, self._stencils,
                         self._white, self._sky, self._sky_program, self.hull):
            resource.release()
