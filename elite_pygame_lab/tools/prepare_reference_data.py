"""Восстановить локальные данные Elite из TAP, SNA и архива Elite Legend 1.09."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


DATA_DIR = Path(__file__).resolve().parents[1] / "reference_data"
DODO_MEMBER = "EliteLegend_v1_09_source/INCLUDE/Ships/old/d_Dodo.a80"
DODO_SOURCE_HASH = "e4dab9fc9883c3a646eabd3af6d66c6a4c1e2153413377c8153df01f235a73df"
EXPECTED = {
    "blueprints.bin": "48be3831b28ae05412f749c812eb2e34d963d8cd71eabd329ac65d6c20c015d9",
    "dodo_elite_legend.bin": "0e0592876d4bab77e1f1945015ad489186ff356fac2660e95453f73a58363cb3",
    "renderer_blocks.json": "6eab3e111bc4f2bd1bcf404ef8829801b7c035375606a6006ed8abb0cf0b5f2f",
}


def check(data, expected, name):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{name}: SHA-256 отличается от ожидаемого {expected}")
    return data


def dodo_bytes(archive):
    with ZipFile(archive) as source:
        data = check(source.read(DODO_MEMBER), DODO_SOURCE_HASH, DODO_MEMBER)
    source_text = re.sub(r"/\*.*?\*/", "", data.decode("cp1251"), flags=re.DOTALL)
    result = bytearray()
    for line in source_text.splitlines():
        line = line.partition(";")[0].strip()
        if not line or line == "d_Dodo":
            continue
        directive, operands = line.split(None, 1)
        width = {"db": 1, "dw": 2}[directive]
        for operand in operands.split(","):
            value, _, divisor = operand.strip().partition("/")
            value = int(value.replace("#", "0x").replace("%", "0b"), 0)
            result.extend((value // int(divisor or "1")).to_bytes(width, "little"))
    return bytes(result)


def prepare(args):
    meta = json.loads((DATA_DIR / "renderer_blocks.meta.json").read_text(encoding="utf-8"))
    tape = check(args.tape.read_bytes(), meta["tape_sha256"], args.tape)
    snapshot = check(args.snapshot.read_bytes(), meta["snapshot_sha256"], args.snapshot)
    # Адресу $6180 соответствует байт 8798 в TAP. В SNA после 27 байт заголовка начинается RAM с $4000.
    tape_offset = 8798 - 0x6180
    for block in meta["blocks"]:
        address = int(block["address"], 16)
        if block["name"] == "initialized_square_table":
            offset = address - 0x4000 + 27
            data = snapshot[offset:offset + block["length"]]
        else:
            offset = address + tape_offset
            data = tape[offset:offset + block["length"]]
        block["hex"] = check(data, block["sha256"], block["name"]).hex()

    outputs = {
        "blueprints.bin": tape[0x6180 + tape_offset:0x6FA3 + tape_offset],
        "dodo_elite_legend.bin": dodo_bytes(args.legend),
        "renderer_blocks.json": (json.dumps(meta, indent=2) + "\n").encode("utf-8"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as scratch:
        for name, data in outputs.items():
            dest = args.output / name
            check(data, EXPECTED[name], name)
            if dest.exists():
                check(dest.read_bytes(), EXPECTED[name], dest)
            else:
                (Path(scratch) / name).write_bytes(data)
        for file in Path(scratch).iterdir():
            file.replace(args.output / file.name)
    print(f"Проверены {len(outputs)} файла в {args.output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", required=True, type=Path, help="elite.tap")
    parser.add_argument("--snapshot", required=True, type=Path, help="elite-title-loop.sna")
    parser.add_argument("--legend", required=True, type=Path, help="архив исходников Elite Legend 1.09")
    parser.add_argument("--output", type=Path, default=DATA_DIR, help="каталог для трёх локальных файлов")
    try:
        prepare(parser.parse_args())
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        parser.exit(1, f"Не удалось подготовить данные: {exc}\n")
