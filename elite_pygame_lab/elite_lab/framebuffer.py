"""Один монохромный буфер 256x128; отдельный ручной шаг записи пикселя.

В учебном представлении строки хранятся подряд. Только отображение адресов
отличается от Spectrum; OR и бит пикселя сохраняют исходный смысл.
"""

WIDTH = 256
HEIGHT = 128


class Framebuffer:
    def __init__(self) -> None:
        self.data = bytearray(WIDTH * HEIGHT // 8)

    def clear(self) -> None:
        self.data[:] = b"\0" * len(self.data)

    def set_pixel(self, x: int, y: int, xor: bool = False) -> None:
        """Эквивалент OR/XOR с маской пикселя в исходном $B434 и соседях."""
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            raise ValueError(f"Pixel outside viewport: {(x, y)}")
        byte_index = y * (WIDTH // 8) + (x >> 3)
        mask = 0x80 >> (x & 7)
        if xor:
            self.data[byte_index] ^= mask
        else:
            self.data[byte_index] |= mask

    def get_pixel(self, x: int, y: int) -> bool:
        return bool(self.data[y * 32 + (x >> 3)] & (0x80 >> (x & 7)))

    def rgb_bytes(self, ink=(224, 239, 243), paper=(8, 13, 21)) -> bytes:
        """Показ уже рассчитанного bitmap; это не часть 3D-рендера."""
        on, off = bytes(ink), bytes(paper)
        return b"".join(on if byte & mask else off
                        for byte in self.data
                        for mask in (128, 64, 32, 16, 8, 4, 2, 1))

    def lit_pixels(self) -> int:
        return sum(byte.bit_count() for byte in self.data)


def spectrum_address(x: int, y: int) -> tuple[int, int]:
    """Адрес в исходном off-screen буфере $C000 и маска из $B308."""
    address = 0xC000 | ((y & 0xC0) << 5) | ((y & 7) << 8) | ((y & 0x38) << 2) | (x >> 3)
    return address, 0x80 >> (x & 7)
