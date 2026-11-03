"""Независимый эталон: исполняем настоящие инструкции Z80 из Elite.

Этот модуль нужен только для проверки переноса. Окно pygame и учебный
рендерер не импортируют SkoolKit и не исполняют машинный код.

В reference_data сохранены каталог моделей, подпрограммы рендера и таблица
квадратов. Заголовки читает сама процедура по указателю из таблицы $6180.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any


_RETURN = 0x5A00
_STACK = 0x5B00
_SHIP = 0x6159
_DODO = 0x7200
_PIXEL_LOOPS = {0xB434, 0xB464, 0xB485, 0xB4B1}


@dataclass
class ReferenceLine:
    pixels: list[tuple[int, int]]
    writes: list[tuple[int, int]]
    framebuffer: bytes
    instructions: int
    errors: list[int] = field(default_factory=list)


@dataclass
class ReferenceGeometry:
    face_flags: list[int]
    projected: list[tuple[int, int, int] | None]
    edge_indices: list[int]
    framebuffer: bytes
    pixels: list[tuple[int, int]]
    edge_pixels: dict[int, list[tuple[int, int]]]
    distance: int
    instructions: int
    rotation_bytes: list[int] = field(default_factory=list)
    vertex_precision: dict[int, tuple[int, tuple[int, ...]]] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _initial_memory() -> bytes:
    path = Path(__file__).resolve().parent.parent / "reference_data" / "renderer_blocks.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    memory = bytearray(65536)
    for block in fixture["blocks"]:
        data = bytes.fromhex(block["hex"])
        if hashlib.sha256(data).hexdigest() != block["sha256"]:
            raise ValueError(f"Повреждён эталонный блок {block['name']}")
        address = int(block["address"], 16)
        memory[address:address + len(data)] = data
    catalog = path.with_name("blueprints.bin").read_bytes()
    if hashlib.sha256(catalog).hexdigest() != "48be3831b28ae05412f749c812eb2e34d963d8cd71eabd329ac65d6c20c015d9":
        raise ValueError("Повреждена исходная таблица моделей $6180..$6FA2")
    memory[0x6180:0x6FA3] = catalog
    # Свободная область эталонной машины для внешней модели. Кассетные записи
    # и игровой код остаются по исходным адресам.
    dodo = path.with_name("dodo_elite_legend.bin").read_bytes()
    if hashlib.sha256(dodo).hexdigest() != "0e0592876d4bab77e1f1945015ad489186ff356fac2660e95453f73a58363cb3":
        raise ValueError("Повреждена модель Dodo из Elite Legend 1.09")
    memory[_DODO:_DODO + len(dodo)] = dodo
    return bytes(memory)


def _memory() -> list[int]:
    memory = list(_initial_memory())
    memory[_STACK:_STACK + 2] = [_RETURN & 255, _RETURN >> 8]
    # Это операнд OR $C0 в преобразовании адреса, а не переменная режима.
    memory[0xB348] = 0xC0
    # Обычный режим корабельных линий: OR C. Солнце может менять его на XOR.
    for address in (0xB435, 0xB465, 0xB486, 0xB4B2):
        memory[address] = 0xB1
    return memory


def _simulator(memory: list[int], registers: dict[str, int]):
    try:
        from skoolkit.simulator import Simulator
    except ImportError as error:
        raise RuntimeError(
            "Эталонная проверка требует SkoolKit: python3 -m pip install skoolkit==10.1"
        ) from error
    return Simulator(memory, {"SP": _STACK, **registers})


def _point_bytes(point: Any) -> tuple[int, int, int]:
    if hasattr(point, "xmag"):
        return point.xmag, point.ymag, point.flags
    return tuple(point)


def pixel_from_write(address: int, mask: int) -> tuple[int, int]:
    """Перестановка строк Spectrum → обычные координаты пикселя."""
    offset = address - 0xC000
    if not 0 <= address < 65536 or mask not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError(f"Неожиданная запись пикселя ${address:04X}, mask=${mask:02X}")
    y = ((offset >> 8) & 7) | ((offset >> 2) & 56) | ((offset // 2048) * 64)
    x = (offset & 31) * 8 + 8 - mask.bit_length()
    return x, y


def unpack_framebuffer(framebuffer: bytes) -> bytes:
    """4096 байт Spectrum → 256×128 байт, по одному значению 0/1 на пиксель."""
    result = bytearray(256 * 128)
    for offset, byte in enumerate(framebuffer):
        if not byte:
            continue
        for bit in range(8):
            if byte & (1 << bit):
                x, y = pixel_from_write(0xC000 + offset, 1 << bit)
                result[y * 256 + x] = 1
    return bytes(result)


def reference_line(start: Any, end: Any) -> ReferenceLine:
    """Выполнить $B044 с двумя настоящими трёхбайтовыми проекциями."""
    x0, y0, flags0 = _point_bytes(start)
    x1, y1, flags1 = _point_bytes(end)
    memory = _memory()
    sim = _simulator(memory, {
        "PC": 0xB044,
        "DE": (x0 << 8) | y0,
        "HL": (x1 << 8) | y1,
        "BC": ((flags0 & 3) << 2) | (flags1 & 3),
    })
    registers = sim.registers
    writes = []
    errors = []
    for instructions in range(200000):
        pc = registers[24]
        if pc == _RETURN:
            break
        if pc in _PIXEL_LOOPS:
            writes.append(((registers[6] << 8) | registers[7], registers[3]))
            errors.append(registers[5])  # E — восьмибитный аккумулятор ошибки.
        sim.opcodes[memory[pc]]()
    else:
        raise RuntimeError(f"Оригинальный отрезок не завершился: PC=${pc:04X}")
    return ReferenceLine(
        [pixel_from_write(address, mask) for address, mask in writes],
        writes, bytes(memory[0xC000:0xD000]), instructions, errors,
    )


def reference_geometry(pose: Any, blueprint_id: int = 9) -> ReferenceGeometry:
    """Выполнить $A84D для записи из таблицы $6180, фронтальный вид, без выстрела.

    Положение задаётся целыми координатами. nose/roof — знаковые значения
    старших байтов в диапазоне -127..127. Младшие байты равны нулю:
    исследуемый рендер использует здесь именно старшие байты.
    """
    memory = _memory()
    memory[_SHIP:_SHIP + 39] = [0] * 39
    for axis, value in enumerate(pose.position):
        magnitude = abs(value)
        if magnitude > 32767:
            raise ValueError("Положение за диапазоном эталона")
        memory[_SHIP + axis * 3:_SHIP + axis * 3 + 3] = [
            magnitude & 255, magnitude >> 8, 128 if value < 0 else 0,
        ]
    for offset, vector in ((9, pose.nose), (15, pose.roof)):
        for axis, value in enumerate(vector):
            if not -127 <= value <= 127:
                raise ValueError("Компонента nose/roof вне диапазона -127..127")
            memory[_SHIP + offset + axis * 2:_SHIP + offset + axis * 2 + 2] = [
                0, abs(value) | (128 if value < 0 else 0),
            ]
    depth = abs(pose.position[2])
    memory[0xA9B8:0xA9BA] = [depth & 255, depth >> 8]
    memory[0xA9C8] = 1
    memory[0x70E3] = 0  # Front view.
    if blueprint_id == 19:
        address = _DODO
    elif 0 <= blueprint_id < 19:
        pointer = 0x6181 + blueprint_id * 4
        address = memory[pointer] | memory[pointer + 1] << 8
    else:
        raise ValueError("Индекс модели должен быть в диапазоне 0..19")
    vertex_count = memory[address + 3]
    edge_base = address + int.from_bytes(memory[address + 14:address + 16], "little")
    sim = _simulator(memory, {"PC": 0xA84D, "IX": _SHIP, "IY": address})
    registers = sim.registers
    precision = {}
    shift = 0
    edge_indices = []
    edge_pixels = {}
    all_pixels = []
    current_edge = None
    for instructions in range(2000000):
        pc = registers[24]
        if pc == _RETURN:
            break
        if pc == 0xACC4:  # Выбор $AA2B, перед обработкой оси X.
            shift = memory[0xA9C7]
        elif pc == 0xAB73:
            shift = 0
        elif pc == 0xADEE:
            address = (registers[6] << 8) | registers[7]
            precision[(address - 0xA8FA) // 3] = (shift, tuple(memory[0xAC9F:0xACA2]))
        if pc == 0xAE42:
            edge_address = (registers[8] << 8) | registers[9]
            current_edge = (edge_address - edge_base) // 4
            edge_indices.append(current_edge)
            edge_pixels[current_edge] = []
        if pc in _PIXEL_LOOPS:
            pixel = pixel_from_write((registers[6] << 8) | registers[7], registers[3])
            all_pixels.append(pixel)
            if current_edge is not None:
                edge_pixels[current_edge].append(pixel)
        sim.opcodes[memory[pc]]()
    else:
        raise RuntimeError(f"Оригинальный рендер не завершился: PC=${pc:04X}")
    projected = [
        tuple(memory[0xA8FA + i * 3:0xA8FA + i * 3 + 3]) if i in precision else None
        for i in range(vertex_count)
    ]
    return ReferenceGeometry(
        memory[0xA984:0xA994], projected, edge_indices,
        bytes(memory[0xC000:0xD000]), all_pixels, edge_pixels,
        memory[0xA9B8], instructions, memory[0xA994:0xA9B8], precision,
    )
