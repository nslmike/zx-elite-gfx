# Работа с elite-gfx

Проект воспроизводит каркасный рендерер ZX Spectrum Elite на Python и Pygame. Команды запуска и управление описаны в [README](elite_pygame_lab/README.md), проверенные выводы и источники — в [FINDINGS.md](FINDINGS.md).

## Исходные материалы

- Дизассемблерный листинг ZX Spectrum Elite закреплён на ревизии `66ee08e`. [Полный листинг](https://github.com/Rasmus-M/elite-skoolkit/blob/66ee08efb2aceddbd67d5003c28fed3477b34850/src/elite.skool) · [Тест граней $BCD4](https://github.com/Rasmus-M/elite-skoolkit/blob/66ee08efb2aceddbd67d5003c28fed3477b34850/src/elite.skool#L17717) · [Отбор рёбер $AE01](https://github.com/Rasmus-M/elite-skoolkit/blob/66ee08efb2aceddbd67d5003c28fed3477b34850/src/elite.skool#L14768-L14804) · [Клиппер $B044](https://github.com/Rasmus-M/elite-skoolkit/blob/66ee08efb2aceddbd67d5003c28fed3477b34850/src/elite.skool#L15263)
- Код и координаты моделей сверены с кассетой, таблица квадратов взята из снимка памяти после инициализации. [Кассета TAP](https://github.com/Rasmus-M/elite-skoolkit/blob/66ee08efb2aceddbd67d5003c28fed3477b34850/tape/elite.tap) · [Снимок SNA](https://github.com/Rasmus-M/elite-skoolkit/blob/66ee08efb2aceddbd67d5003c28fed3477b34850/snapshot/elite-title-loop.sna) · [Локальные блоки Z80 и SHA-256](elite_pygame_lab/reference_data/renderer_blocks.json) · [Кассетный каталог моделей](elite_pygame_lab/reference_data/blueprints.bin)
- Для Dodo используй исходники отдельной версии Elite Legend 1.09, файл `INCLUDE/Ships/old/d_Dodo.a80`. [Архив исходников](https://idpixel.ru/games/e/elitelegend/files/elitelegend_v1_09_source.zip) · [Блок Dodo в демо](elite_pygame_lab/reference_data/dodo_elite_legend.bin)

## При изменении рендерера

- В `elite_pygame_lab/elite_lab/` модули `arithmetic.py` и `renderer.py` отвечают за арифметику, видимость и проекцию. Сохраняй знак-модуль, отрицательный ноль, порядок округлений и дальностные пороги оригинала.
- `clipping.py` обрезает экранный отрезок, `raster.py` вручную вычисляет пиксельные шаги. Пиксельный режим ZX должен оставаться доступным по умолчанию. Альтернативные тесты граней и линии Pygame явно обозначай в интерфейсе.
- Сверяй утверждения об оригинале с Z80-листингом и данными в `elite_pygame_lab/reference_data/`. Для новой находки укажи в `FINDINGS.md` модель, воспроизводимый ракурс и наблюдаемый результат. Пояснения и подсказки пиши по-русски.

## Запуск и проверка

Из корня перейди в `elite_pygame_lab/`. Для первого запуска создай окружение командой `python3.12 -m venv .venv` и установи зависимости через `.venv/bin/python -m pip install -r requirements.txt`.

- Запуск демо — `.venv/bin/python -m elite_lab`.
- Быстрые тесты — `.venv/bin/python -m unittest discover -s tests -v`.
- После изменения интерфейса проверь нужный режим без окна, например `.venv/bin/python -m elite_lab --headless --mode clip --model 17 --screenshot /tmp/elite-krait.png`.
- После изменения численного алгоритма установи `requirements-verify.txt` в то же окружение и выполни `.venv/bin/python tools/verify_reference.py --lines 1000 --poses 128` для сверки с оригинальными инструкциями Z80.
