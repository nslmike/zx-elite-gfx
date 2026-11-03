"""Клиппер отрезка Spectrum Elite ($B044–$B432), до первого пикселя.

Результат оригинала — начальный пиксель, направление, slope и число записей.
Даже при двух внешних концах он не строит пару точных вещественных пересечений.
"""

from dataclasses import dataclass
from typing import Protocol

from .arithmetic import multiply_u8
from .types import ProjectedVertex


class ProjectedPoint(Protocol):
    """Те же три байта, что в таблице спроецированных вершин $A8FA."""

    xmag: int
    ymag: int
    flags: int


def screen_xy(point: ProjectedPoint) -> tuple[int, int]:
    """$B308: центр (127, 63); ось экранного Y направлена вниз."""
    x = 127 - point.xmag if point.flags & 2 else 127 + point.xmag
    y = 63 + point.ymag if point.flags & 1 else 63 - point.ymag
    return x, y


def slope_increment(minor: int, major: int) -> int:
    """$B2FE → $AC8B: получить 8 бит дроби minor / major.

    При minor < major результат равен floor(256 * minor / major).
    Для равных приращений исходная процедура даёт 255, а не 256.
    Нулевой знаменатель возможен для вырожденной линии; исходный цикл тоже
    даёт 255, но рисуется единственный пиксель и это значение не применяется.

    Здесь сохранён сам короткий алгоритм деления. Перенос из старшего бита
    remainder нужен: сдвиг восьмибитного регистра может переполниться.
    """
    if not 0 <= minor <= 255 or not 0 <= major <= 255:
        raise ValueError("Приращения должны помещаться в 8 бит")
    remainder = minor
    quotient = 0
    for _ in range(8):
        overflow = bool(remainder & 0x80)
        remainder = (remainder << 1) & 0xFF
        quotient = (quotient << 1) & 0xFF
        if overflow or remainder >= major:
            remainder = (remainder - major) & 0xFF
            quotient |= 1
    return quotient


# $AFF0: номер внешней области по знакам и двум флагам выхода за границу.
_REGIONS = (0, 4, 2, 3, 0, 4, 6, 5, 0, 8, 2, 1, 0, 8, 6, 7)

# $B000: пара внешних областей → отбросить (0), прямо к границе (1)
# или проверить пересечение по наклону к углу (2).
_CASES = (
    (0, 0, 0, 2, 1, 2, 0, 0),
    (0, 0, 0, 2, 2, 1, 2, 2),
    (0, 0, 0, 0, 0, 2, 1, 2),
    (2, 2, 0, 0, 0, 2, 2, 1),
    (1, 2, 0, 0, 0, 0, 0, 2),
    (2, 1, 2, 2, 0, 0, 0, 2),
    (0, 2, 1, 2, 0, 0, 0, 0),
    (0, 2, 2, 1, 2, 2, 0, 0),
)


def _region(p: ProjectedPoint) -> int:
    code = ((p.flags & 3) << 2) | ((p.ymag >= 64) << 1) | (p.xmag >= 128)
    return _REGIONS[code]


@dataclass(frozen=True)
class Direction:
    dx: int
    dy: int
    step_x: int
    step_y: int
    major_axis: str
    slope: int


@dataclass(frozen=True)
class RasterSpan:
    origin: tuple[int, int]
    direction: Direction
    count: int


def _direction(a: ProjectedPoint, b: ProjectedPoint) -> Direction:
    """$B3A7: сравнение половин 9-битных дельт. При переполнении и
    деление для slope выполняется по половинам обеих дельт.
    """
    ax, ay = screen_xy(a)
    bx, by = screen_xy(b)
    dx, dy = abs(bx - ax), abs(by - ay)
    x_major = (dx >> 1) >= (dy >> 1)
    major, minor = (dx, dy) if x_major else (dy, dx)
    if major > 255:
        major >>= 1
        minor >>= 1
    else:
        minor &= 255  # $B426/$B42A берут младший байт.
    # При нулевой дельте Z80 сохраняет направление по знаку первого конца.
    # Это влияет на выбор угла в $B362 даже для горизонтали или вертикали.
    right = bx > ax or (bx == ax and bool(a.flags & 2))
    up = by < ay or (by == ay and bool(a.flags & 1))
    return Direction(dx, dy, 1 if right else -1, -1 if up else 1,
                     "x" if x_major else "y", slope_increment(minor, major))


def _corner_signs(first: int, second: int) -> int:
    """Дерево $B1FE–$B232. Результат — два знака угла (128, 64)."""
    if first + second < 7:
        return 0
    if first + second >= 12:
        return 3
    difference = first ^ second
    if difference >= 8:
        return 2
    if difference < 4:
        return 1
    if difference == 5:
        return 2 if first == 2 else 1
    return 3 if ((first ^ 6) & second) in (0, 6) else 0


def _corner_slope(line: Direction, corner: Direction) -> int:
    return corner.slope if line.major_axis == corner.major_axis else 255


def _entry_point(p: ProjectedPoint, line: Direction, horizontal_border: bool) -> ProjectedVertex:
    """$B274/$B2B2: отступ внешнего конца от входной границы. SUB/NEG
    сохраняют отдельный знак, а сложение модуля производится в одном байте.
    """
    if horizontal_border:
        excess = (p.ymag - 64) & 255
        magnitude, sign_bit, limit = p.xmag, 2, 127
        positive = line.step_x > 0
        multiply = line.major_axis == "y"
    else:
        excess = (p.xmag - 128) & 255
        magnitude, sign_bit, limit = p.ymag, 1, 63
        positive = line.step_y < 0  # Положительный мировой Y направлен вверх.
        multiply = line.major_axis == "x"
    if multiply:
        offset = (multiply_u8(excess, line.slope) + 128) >> 8  # $B2F0
    else:
        offset = slope_increment(excess, line.slope)  # $B2FE
    flags = p.flags & 3
    if positive != bool(flags & sign_bit):
        magnitude = (magnitude + offset) & 255
    else:
        magnitude -= offset
        if magnitude < 0:
            magnitude = -magnitude
            flags ^= sign_bit
    magnitude = min(magnitude, limit)
    if horizontal_border:
        return ProjectedVertex(magnitude, 63, flags)
    return ProjectedVertex(127, magnitude, flags)


def _exit_count(origin: ProjectedPoint, line: Direction) -> int:
    """$B362: длина от точки внутри окна до выходной границы вдоль линии."""
    flags = (2 if line.step_x < 0 else 0) | (1 if line.step_y > 0 else 0)
    corner = _direction(origin, ProjectedVertex(128, 64, flags))
    major, minor = ((corner.dx, corner.dy) if line.major_axis == "x"
                    else (corner.dy, corner.dx))
    count = major
    if line.slope > _corner_slope(line, corner):
        count = slope_increment(max(minor - 1, 0), line.slope)
    return max(count, 2) - 1  # $B0E0, общий вход в пиксельные циклы.


def prepare_line(start: ProjectedPoint, end: ProjectedPoint) -> RasterSpan:
    """$B044: порядок обхода, входная точка и счётчик перед первой записью."""
    for p in (start, end):
        if p.flags & 128:
            raise ValueError("Вершина помечена невидимой: bit 7 установлен")
        if not (0 <= p.xmag <= 255 and 0 <= p.ymag <= 255):
            raise ValueError("Модули спроецированной точки должны помещаться в байт")

    first, last = _region(start), _region(end)
    case = _CASES[first - 1][last - 1] if first and last else 0
    # $B0AB и $B1DB меняют концы местами до вычисления наклона.
    if (first and not last) or (case == 2 and first & 1):
        start, end = end, start
        first, last = last, first
    line = _direction(start, end)
    origin = screen_xy(start)
    if (end.xmag == start.ymag == 255 or end.ymag == start.xmag == 255
            or (first and last and case == 0)):
        return RasterSpan(origin, line, 0)
    if not last:  # Оба внутри, $B0CD. Обычно последняя точка исключена.
        count = line.dx if line.major_axis == "x" else line.dy
        return RasterSpan(origin, line, max(count, 1))
    if first:
        if case == 1 and (first & 1) == 0:
            horizontal = start.ymag >= 64  # $B1CE
        else:
            signs = start.flags & 3 if case == 1 else _corner_signs(first, last)
            slope = _corner_slope(line, _direction(start, ProjectedVertex(128, 64, signs)))
            x_major = line.major_axis == "x"
            if case == 1:
                horizontal = line.slope < slope if x_major else slope < line.slope
            else:  # $B253–$B273, строгие сравнения (равенство отбрасывает).
                horizontal = bool(first & 2)
                crosses = slope < line.slope if x_major == horizontal else line.slope < slope
                if not crosses:
                    return RasterSpan(origin, line, 0)
        start = _entry_point(start, line, horizontal)
    return RasterSpan(screen_xy(start), line, _exit_count(start, line))
