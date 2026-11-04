# Локальные данные ZX Spectrum Elite

Для запуска просмотрщика нужны каталог моделей `blueprints.bin` и станция
`dodo_elite_legend.bin`. Для сверки с оригинальным Z80 также нужен
`renderer_blocks.json`. Все три файла собираются из следующих источников.

| Что скачать | Источник |
| --- | --- |
| `elite.tap` — образ кассеты ZX Spectrum Elite | [Скачать TAP](https://raw.githubusercontent.com/Rasmus-M/elite-skoolkit/66ee08efb2aceddbd67d5003c28fed3477b34850/tape/elite.tap) |
| `elite-title-loop.sna` — снимок после инициализации таблицы квадратов | [Скачать SNA](https://raw.githubusercontent.com/Rasmus-M/elite-skoolkit/66ee08efb2aceddbd67d5003c28fed3477b34850/snapshot/elite-title-loop.sna) |
| `elitelegend_v1_09_source.zip` — исходники Elite Legend 1.09 с моделью Dodo | [Скачать ZIP](https://idpixel.ru/games/e/elitelegend/files/elitelegend_v1_09_source.zip) · [Страница разработчика](https://zx-pk.ru/threads/33861-elite-legend-128k.html) |

Сохраните исходные файлы, которыми вы вправе пользоваться, например в
`~/Downloads`. Из каталога `elite_pygame_lab` выполните команду. Для неё
достаточно стандартной библиотеки Python 3.10 или новее.

```bash
python3 tools/prepare_reference_data.py \
  --tape ~/Downloads/elite.tap \
  --snapshot ~/Downloads/elite-title-loop.sna \
  --legend ~/Downloads/elitelegend_v1_09_source.zip
```

Скрипт проверяет SHA-256 исходных данных, извлекает диапазоны памяти
и собирает таблицы `db` и `dw` из `INCLUDE/Ships/old/d_Dodo.a80`.
Готовые файлы сохраняются в этот каталог и игнорируются Git. При повторном
запуске проверенные копии сохраняются на месте.

| Локальный файл | SHA-256 |
| --- | --- |
| `blueprints.bin` | `48be3831b28ae05412f749c812eb2e34d963d8cd71eabd329ac65d6c20c015d9` |
| `dodo_elite_legend.bin` | `0e0592876d4bab77e1f1945015ad489186ff356fac2660e95453f73a58363cb3` |
| `renderer_blocks.json` | `6eab3e111bc4f2bd1bcf404ef8829801b7c035375606a6006ed8abb0cf0b5f2f` |

Ревизия кассеты и снимка, адреса блоков и их отдельные контрольные суммы
приведены в [описании эталона](renderer_blocks.meta.json).
