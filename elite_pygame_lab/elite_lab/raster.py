"""Ручная растеризация линий ZX Spectrum Elite — без графических примитивов.

Входной отрезок проходит клиппер $B044 из clipping.py, затем:
    $B3A7: выбор ведущей оси и приращения;
    $B2FE → $AC8B: восьмибитное деление;
    $B0CD → $B0E0: длина и начальная ошибка;
    $B434 / $B464 / $B485 / $B4B1: запись одного пикселя за итерацию.

Источник: Rasmus-M/elite-skoolkit, commit
66ee08efb2aceddbd67d5003c28fed3477b34850, src/elite.skool.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clipping import ProjectedPoint, prepare_line, screen_xy, slope_increment


@dataclass(frozen=True)
class PixelStep:
    """Одна запись пикселя и состояние при переходе к следующему.

    На последнем пикселе оригинал возвращается ДО обновления ошибки.
    Поэтому has_next=False, error_after=error_before, carry=False.
    """

    index: int
    x: int
    y: int
    error_before: int
    error_after: int
    carry: bool
    major_axis: str
    has_next: bool
    address: int
    mask: int

    @property
    def in_bitmap(self) -> bool:
        return 0xC000 <= self.address < 0xD000


@dataclass(frozen=True)
class LineTrace:
    start: tuple[int, int]
    end: tuple[int, int]
    delta_x: int
    delta_y: int
    major_axis: str
    slope: int
    step_x: int
    step_y: int
    steps: tuple[PixelStep, ...]

    @property
    def count(self) -> int:
        return len(self.steps)

    @property
    def clip_status(self) -> str:
        if not self.steps:
            return "rejected"
        if any(not (0 <= x <= 254 and 0 <= y <= 126) for x, y in (self.start, self.end)):
            return "clipped"
        return "inside"

    @property
    def last_visible(self) -> PixelStep | None:
        return next((step for step in reversed(self.steps) if step.in_bitmap), None)

    @property
    def endpoints(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if self.clip_status == "inside":
            return self.start, self.end
        last = self.last_visible
        if last:
            first = self.steps[0]
            return (first.x, first.y), (last.x, last.y)
        return None

    @property
    def pixels(self) -> tuple[tuple[int, int], ...]:
        return tuple((step.x, step.y) for step in self.steps)


def spectrum_address(x: int, y: int) -> tuple[int, int]:
    """Адрес и маска пикселя в исходном буфере $C000..$CFFF.

    Биты Y переставляются так же, как у экранной памяти Spectrum.
    Для нашего обычного массива изображения это не требуется, но адрес полезен
    для сопоставления с Z80 и позволяет увидеть ручную работу с битами.
    """
    address = 0xC000 | ((y & 0xC0) << 5) | ((y & 7) << 8)
    address |= ((y & 0x38) << 2) | (x >> 3)
    mask = 0x80 >> (x & 7)
    return address, mask


def _move_x_address(address: int, x: int, direction: int) -> int:
    # RLC/RRC маски. При переходе в соседний байт INC/DEC L не меняет H.
    if x >> 3 == (x + direction) >> 3:
        return address
    return (address & 0xFF00) | ((address + direction) & 255)


def _move_y_address(address: int, y: int, direction: int) -> int:
    # Внутри группы восьми строк меняется только H. На границе группы
    # $B443/$B470/$B494/$B4BC также сдвигают L на ±32 и учитывают перенос.
    if y >> 3 == (y + direction) >> 3:
        return (address + direction * 256) & 0xFFFF
    low = (address & 255) + direction * 32
    high = (address >> 8) - direction * 7 + (low // 256) * 8
    return ((high & 255) << 8) | (low & 255)


def _write_xy(address: int, logical_x: int) -> tuple[int, int]:
    # Клиппер приблизительный. При переходе через край байт может оказаться
    # в другой строке или даже вне bitmap. В трассе сохраняем фактическую запись.
    offset = address - 0xC000
    x = (offset & 31) * 8 + (logical_x & 7)
    y = (offset // 2048) * 64 + ((offset >> 2) & 56) + ((offset >> 8) & 7)
    return x, y


def trace_line(start: ProjectedPoint, end: ProjectedPoint) -> LineTrace:
    """Ручные записи $B434 и родственных циклов после клиппинга $B044.

    start соответствует DE, end — HL. Полностью отсечённое ребро имеет
    пустую последовательность шагов. Координаты start/end в трассе исходные.
    """
    span = prepare_line(start, end)
    x, y = span.origin
    major_axis, slope = span.direction.major_axis, span.direction.slope
    step_x, step_y = span.direction.step_x, span.direction.step_y
    pixel_count = span.count
    error = 0x80  # $B0EA: середина восьмибитного диапазона.
    steps: list[PixelStep] = []

    address, _ = spectrum_address(x, y)
    for index in range(pixel_count):
        # Сначала запоминаем текущий пиксель. В игре: LD A,(HL); OR C;
        # LD (HL),A. Viewer применяет эти же шаги к своему framebuffer.
        mask = 0x80 >> (x & 7)
        pixel_x, pixel_y = _write_xy(address, x)
        has_next = index + 1 < pixel_count
        error_before = error
        carry = False

        if has_next:
            total = error + slope
            carry = total >= 256  # Флаг переноса Z80 после ADD.
            error = total & 0xFF  # В регистре остаются младшие 8 бит.

        steps.append(PixelStep(
            index=index, x=pixel_x, y=pixel_y,
            error_before=error_before, error_after=error,
            carry=carry, major_axis=major_axis, has_next=has_next,
            address=address, mask=mask,
        ))

        if not has_next:
            break  # Оригинальный RET Z стоит раньше обновления ошибки.

        # По ведущей оси идём всегда; по второй — только при переносе.
        # Четыре оригинальных цикла отличаются направлением Y и ведущей
        # осью; направление X исходник меняет самомодификацией RLC/RRC.
        # Здесь все восемь октантов выражены двумя знаками шага.
        if major_axis == "x":
            if carry:
                address = _move_y_address(address, y, step_y)
                y += step_y
            address = _move_x_address(address, x, step_x)
            x += step_x
        else:
            if carry:
                address = _move_x_address(address, x, step_x)
                x += step_x
            address = _move_y_address(address, y, step_y)
            y += step_y

    return LineTrace(
        start=screen_xy(start), end=screen_xy(end),
        delta_x=span.direction.dx, delta_y=span.direction.dy, major_axis=major_axis,
        slope=slope, step_x=step_x, step_y=step_y, steps=tuple(steps),
    )


def apply_trace(buffer: bytearray, trace: LineTrace, *, xor: bool = False) -> None:
    """Применить шаги к 4096 байтам оригинального экранного буфера.

    По умолчанию — OR, обычный режим линий корабля. XOR можно включить явно
    для изучения записи битов. Это не алгоритм устранения перекрытий.
    """
    if len(buffer) != 4096:
        raise ValueError("Нужен буфер 4096 байт ($C000..$CFFF)")
    for step in trace.steps:
        if not step.in_bitmap:
            continue
        offset = step.address - 0xC000
        if xor:
            buffer[offset] ^= step.mask
        else:
            buffer[offset] |= step.mask
