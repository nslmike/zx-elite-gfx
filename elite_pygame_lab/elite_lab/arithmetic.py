"""Целочисленная математика Spectrum Elite, с адресами исходных процедур.

Это не обычная fixed-point матрица: оригинал использует разность квадратов,
отдельный знак и модуль, а в нескольких местах сохраняет даже отрицательный
ноль. Поэтому внутри этого модуля SignMag хранит знак независимо от величины.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SignMag:
    magnitude: int
    negative: bool = False

    @classmethod
    def from_int(cls, value: int) -> "SignMag":
        return cls(abs(value), value < 0)

    @property
    def signed(self) -> int:
        return -self.magnitude if self.negative else self.magnitude

    @property
    def byte(self) -> int:
        return (0x80 if self.negative else 0) | (self.magnitude & 0x7F)


def multiply_u8(a: int, b: int) -> int:
    """$AB34: разность ДВУХ целых квадратов, не всегда обычное a*b."""
    assert 0 <= a <= 255 and 0 <= b <= 255
    return ((a + b) // 2) ** 2 - (abs(a - b) // 2) ** 2


def multiply_high_u8(a: int, b: int) -> int:
    """$AB0B: вычитание старших байтов квадратов, не multiply_u8() >> 8.

    Усечение каждого квадрата происходит ДО вычитания. Эта последовательность
    округлений заметна и в проекции, и в моменте переключения граней.
    """
    assert 0 <= a <= 255 and 0 <= b <= 255
    larger_square = ((a + b) // 2) ** 2
    smaller_square = (abs(a - b) // 2) ** 2
    return (larger_square >> 8) - (smaller_square >> 8)


def add_sign_magnitude(left: SignMag, right: SignMag, bits: int = 8) -> SignMag:
    """$AAFB и знаковая часть $AFCD: знак нуля берётся от левого аргумента."""
    if left.negative == right.negative:
        return SignMag((left.magnitude + right.magnitude) & ((1 << bits) - 1),
                       left.negative)
    difference = left.magnitude - right.magnitude
    if difference >= 0:
        return SignMag(difference, left.negative)
    return SignMag(-difference, right.negative)


def cross_component(x1: int, x2: int, y1: int, y2: int) -> SignMag:
    """$AF22: старший байт 2*(y1*y2 - x1*x2), включая знак нуля.

    Вызов $AECC подставляет компоненты так, чтобы получить roof × nose.
    Пары аргументов сначала удваиваются, поэтому эти табличные умножения
    точны: оба операнда чётные. Здесь оставлен сам табличный вызов.
    """
    x_negative = (x1 < 0) != (x2 < 0)
    y_negative = (y1 < 0) != (y2 < 0)
    x_product = multiply_u8(2 * abs(x1), 2 * abs(x2))
    y_product = multiply_u8(2 * abs(y1), 2 * abs(y2))
    negative = not x_negative
    if x_negative != y_negative:
        magnitude = min(65535, x_product + y_product)
    else:
        difference = x_product - y_product
        magnitude = abs(difference)
        if difference < 0:
            negative = not negative
    return SignMag((magnitude >> 1) >> 8, negative)


def make_basis(nose: tuple[int, int, int], roof: tuple[int, int, int]):
    """$AECC: строки матрицы (side, roof, nose) для выходных X, Y, Z."""
    nx, ny, nz = nose
    rx, ry, rz = roof
    side = (
        cross_component(ny, rz, nz, ry),
        cross_component(nz, rx, nx, rz),
        cross_component(nx, ry, ny, rx),
    )
    return tuple((side[axis], SignMag.from_int(roof[axis]),
                  SignMag.from_int(nose[axis])) for axis in range(3))


def rotate_components(vector: tuple[int, int, int], basis, *, cached=False):
    """Поворот в половинном масштабе ($AC01 / $ABC0, вблизи $AA8B / $AA49).

    Оригинал повторно использует произведения для последовательных вершин
    с одинаковыми модулями. Здесь произведения вычислены повторно для простоты,
    но порядок знаковых сложений обеих ветвей сохранён (важен знак нуля).
    """
    outputs = []
    for row in basis:
        products = [SignMag(multiply_high_u8(abs(component), factor.magnitude),
                            (component < 0) != factor.negative)
                    for component, factor in zip(vector, row)]
        x, y, z = products
        if cached:
            total = add_sign_magnitude(add_sign_magnitude(x, y), z)
        else:
            total = add_sign_magnitude(z, add_sign_magnitude(y, x))
        outputs.append(total)
    return tuple(outputs)


def translate_vertex_axis(position: int, rotated: SignMag, shift: int, carry: int) -> tuple[SignMag, int, int]:
    """$AAE3, $ACDF и $AB4F: снять масштаб, удвоить и прибавить позицию.

    Возвращает целое со знаком, младший байт и carry для RLA следующей оси.
    При shift=0 остаток поворота равен нулю, как после $ABB3.
    """
    # SRL H / RR L / RRA: выпавшие из целой части биты уходят в старшие
    # биты A. При 1..7 сдвигах из самого A ещё не выпадает единица.
    fixed = (rotated.magnitude << 8) >> shift
    integer, fraction = divmod((fixed << 1) | carry, 256)
    carry = integer & 1  # Старший бит прежнего остатка, сохранённый PUSH AF.
    place = SignMag.from_int(position)
    if place.negative == rotated.negative:
        result = SignMag((place.magnitude + integer) & 0xFFFF, place.negative)
    elif place.magnitude > integer:
        # $AB66–$AB72: DEC HL выполняется даже при нулевом остатке.
        # NEG изменяет сохранённый AF. Его carry попадёт в следующую ось.
        result = SignMag(place.magnitude - integer - 1, place.negative)
        fraction = (-fraction) & 255
        carry = int(fraction != 0)
    else:
        result = SignMag(integer - place.magnitude, rotated.negative)
    return result, fraction, carry


def translate_component(position: int, rotated: SignMag) -> SignMag:
    """Целая координата обычной ветви $AB73."""
    return translate_vertex_axis(position, rotated, 0, 0)[0]


def perspective_divide(numerator: int, denominator: int,
                       numerator_fraction: int = 0, denominator_fraction: int = 0) -> int:
    """$AC69: до восьми сдвигов DE:C и HL:A, затем деление на старший байт H."""
    dividend = ((numerator & 0xFFFF) << 8) | numerator_fraction
    divisor = ((denominator & 0xFFFF) << 8) | denominator_fraction
    for _ in range(8):
        if divisor & 0x800000:
            break
        if dividend & 0x800000:
            # $AC7B..$AC82: откат сдвига, SCF перед RR H.
            divisor |= 0x800000
            break
        dividend <<= 1
        divisor <<= 1
    divisor_high = divisor >> 16
    if (dividend >> 16) >= divisor_high:
        return 255
    return (dividend >> 8) // divisor_high


def line_of_sight_component(position: int, normal: SignMag, scale: int,
                            vertex_component: SignMag | None = None):
    """$BDB4: нормаль и вектор от камеры к точке грани, в половинном масштабе."""
    doubled = (normal.magnitude << 1) & 0xFF
    if vertex_component is None:
        offset = SignMag(doubled >> scale, normal.negative)
    else:
        # Опорная вершина после поворота имеет половинный масштаб.
        # Масштаб табличной нормали к координатам вершины не применяется.
        offset = SignMag(2 * vertex_component.magnitude, vertex_component.negative)
    # У $BDB4 при точной компенсации знак сохраняется от позиции.
    combined = add_sign_magnitude(SignMag.from_int(position), offset, bits=16)
    return (SignMag(doubled >> 1, normal.negative),
            SignMag(combined.magnitude >> 1, combined.negative))


def multiply_byte_word(byte: SignMag, word: SignMag) -> SignMag:
    """$BD95: произведение normal-byte и sign-magnitude line-of-sight word."""
    doubled = (byte.magnitude << 1) & 0xFF
    high = (word.magnitude >> 8) & 0x7F
    low = word.magnitude & 0xFF
    magnitude = multiply_u8(doubled, high) + multiply_high_u8(doubled, low)
    return SignMag(magnitude & 0x7FFF, byte.negative != word.negative)


def face_dot_sign(normal, line_of_sight) -> bool:
    """$BD5F: знак скалярного произведения, включая отрицательный ноль."""
    x, y, z = [multiply_byte_word(n, p) for n, p in zip(normal, line_of_sight)]
    # Первая пара складывается /2, затем ADC восстанавливает потерянный бит.
    # Последнее деление /2 не влияет на проверяемый бит знака.
    yz = add_sign_magnitude(z, y, bits=16)
    total = add_sign_magnitude(yz, x, bits=16)
    return total.negative
