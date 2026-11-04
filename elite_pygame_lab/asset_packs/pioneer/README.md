# Pioneer — Sinonatrix Courier

Самостоятельный космический курьер Pioneer, объект `20` в атласе.

Автор модели и текстур — **Bálint Szilárd (bszlrd)**. В `AUTHORS.txt` Pioneer
он указан как **Szilárd Bálint**. Благодарности также **Pioneer Developers**.
[Портфолио автора](https://bszlrd.hu/project/pioneer) ·
[Авторское добавление ресурсов](https://github.com/pioneerspacesim/pioneer/commit/54f45c138c7a) ·
[Исходный AUTHORS.txt](source_metadata/AUTHORS.txt)

Модель, текстуры и преобразованные ресурсы пакета распространяются по
**CC BY-SA 3.0 Unported**. Лицензия допускает коммерческое использование
с указанием авторства и сохранением лицензии переработанных ресурсов.
[Условия](https://creativecommons.org/licenses/by-sa/3.0/) ·
[Полный текст](LICENSE-CC-BY-SA-3.0.txt)

Файлы взяты из [официального репозитория Pioneer](https://github.com/pioneerspacesim/pioneer/tree/7506ef474e060c129fe27349f279895b2a5d8ed9/data/models/ships/sinonatrix)
на ревизии `7506ef474e060c129fe27349f279895b2a5d8ed9`.
Пути, SHA-256 и параметры преобразований записаны в
[source_manifest.json](source_manifest.json). Оригинальные определения моделей
сохранены в `source_metadata/`.

| LOD | Файл | Треугольники |
| --- | --- | ---: |
| Высокий | `sinonatrix_hi.npz` | 49 804 |
| Средний | `sinonatrix_med.npz` | 13 264 |
| Низкий | `sinonatrix_low.npz` | 6 742 |

При переносе применены вложенные преобразования сцены и общая нормализация
трёх LOD. Геометрия триангулирована с сохранением UV и авторских нормалей.
Исправлены пропущенные привязки материалов колёс, порядок отдельных мелких
треугольников и один вырожденный треугольник. Шасси зафиксированы в убранном
положении на первом кадре `gear_down`.

Восемь родных карт diffuse, normal, specular и glow корпуса и двигателей
переведены в RGB PNG. Сохранены исходные размеры **2048×1024** и **1024×512**.
В specular-карты включены коэффициенты материалов Pioneer. Shininess
приближённо переведена в GGX gloss атласа.

Стекло высокого LOD показывается непрозрачным с родной картой корпуса.
В Pioneer его непрозрачность составляет `90%`. Отдельный интерьер
`sinonatrix_cockpit.dae` и его текстуры доступны в исходниках Pioneer.
В пакете используется внешний вид корабля.

Повторить импорт можно из `elite_pygame_lab/` с установленными `numpy` и `Pillow`.

```sh
python3 tools/import_pioneer_pack.py --cache /tmp/elite-pioneer-source
```

Скрипт скачивает файлы закреплённой ревизии и проверяет SHA-256.
Параметры `--texture-size 1024 --output /tmp/pioneer-1024` создают отдельный
вариант с уменьшенными картами.
