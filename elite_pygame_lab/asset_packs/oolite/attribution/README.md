# Oolite — источники и атрибуция

Набор содержит штатные модели и текстуры из
[OoliteProject/oolite на ревизии 28ceaaba0d79d0b6ea1c044c6235e3616b170098](https://github.com/OoliteProject/oolite/tree/28ceaaba0d79d0b6ea1c044c6235e3616b170098),
полученные 4 ноября 2026 года и адаптированные для атласа elite-gfx.
Все 20 карточек имеют геометрию и diffuse. Соответствия моделей, материалы и ссылки
на исходные DAT записаны в `pack.json`. [source_manifest.json](source_manifest.json)
содержит пермалинки и SHA-256 использованных исходных файлов.

## Авторы и происхождение

Основной флот создан **Griff**, **CaptSolo** и участниками Oolite.
Карты и материалы дорабатывали **Nikos Barkas (another_commander)** и другие участники.
Уведомление правообладателей Oolite — © 2004–2026 Giles C Williams, Jens Ayton and contributors.

[Официальный список участников](https://github.com/OoliteProject/oolite/blob/28ceaaba0d79d0b6ea1c044c6235e3616b170098/Doc/contributors.txt)
сохранён в [UPSTREAM_CONTRIBUTORS.txt](UPSTREAM_CONTRIBUTORS.txt).
[CHANGELOG Oolite](https://github.com/OoliteProject/oolite/blob/28ceaaba0d79d0b6ea1c044c6235e3616b170098/Doc/CHANGELOG.TXT#L1101-L1102)
называет Griff/CaptSolo авторами замены штатных моделей и текстур.

| Ресурсы | Дополнительные Credits происхождения |
| --- | --- |
| Coriolis и Dodecahedron/Dodo с доками и накладками | Griff создал модели. PAGroove подготовил исходные станционные текстуры, CaptSolo переработал текстуры и карты. См. [страницу CaptSolo](https://wiki.alioth.net/index.php?title=User:CaptSolo&oldid=79001#Alternate_Main_Stations_.26_Solo.27s_Rock_Hermit) и [Credits семейства станций](https://wiki.alioth.net/index.php?title=Solos_Alt_Stations&oldid=82605#Credits). |
| Rock Hermit | Griff, CaptSolo, Killer Wolf и участники Oolite. Происхождение модели и текстур от Killer Wolf указано на [странице CaptSolo](https://wiki.alioth.net/index.php?title=User:CaptSolo&oldid=79001#Alternate_Main_Stations_.26_Solo.27s_Rock_Hermit). |

Конкретные файлы набора взяты из официальной базовой Oolite на указанной ревизии.
Dodo 19 соответствует `oolite_template_dodecahedron-station`.
Python 10 использует `oolite_template_python-trader`, Python 11 — `oolite_template_python`.
Cobra 09 и 12 используют общую сетку и штатные торговую и пиратскую обшивки.

## Лицензия

[LICENSE.md Oolite](https://github.com/OoliteProject/oolite/blob/28ceaaba0d79d0b6ea1c044c6235e3616b170098/LICENSE.md)
разрешает распространять художественные ресурсы и конфигурацию по выбору под
**GPL-2.0-or-later OR CC-BY-NC-SA-3.0**.
Исходное уведомление сохранено в [UPSTREAM_LICENSE.md](UPSTREAM_LICENSE.md).

Для этой адаптации выбрана **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported**,
SPDX **CC-BY-NC-SA-3.0**. Изменённые модели и текстуры распространяются на тех же условиях.
Сохранены [условия лицензии](https://creativecommons.org/licenses/by-nc-sa/3.0/)
и [полный текст](CC-BY-NC-SA-3.0.txt).

## Конверсия

- DAT преобразованы в NPZ с UV каждого угла и слотами исходных материалов. Явные NORMALS сохранены. Для старых DAT использованы нормали граней, при `smooth=yes` усреднённые по площади и группе сглаживания.
- Треугольники приведены к наружному CCW вместе с UV и нормалями. Оси сохраняют +Y вверх и +Z к носу или входу в док. Объекты центрированы по bbox и масштабированы до максимального радиуса 1.
- Доки, накладки и части Rock Hermit собраны по позициям и ориентациям `shipdata.plist`.
- DAT UV преобразованы в `(u,v)=(s,1-t)`. Загрузчик атласа переворачивает изображения по Y один раз.
- Diffuse сохранены в JPEG с качеством 92 и субдискретизацией 4:4:4. Diffuse и normal для Cobra, Python и Krait имеют размер до 1024, остальные карты — до 512. Normal перенормированы, карты данных сохранены в PNG. Общие карты дедуплицированы.
- Specular RGB переведён из gamma 2.2 в линейную яркость с запечённым `specular_modulate_color`. Для таких материалов `specular_color=[1,1,1]`. Alpha specular вынесен в линейную `gloss_map`, значение `gloss` сохранено.
- Alpha diffuse преобразован в emission с исходным оттенком `emission_modulate_color`. Яркость сверхярких станционных огней вынесена в `emission_strength`. Diffuse и emission используют sRGB.

Normal сохраняют исходный зелёный канал и базис Oolite **T=+dP/dU, B=cross(N,T)**.
Поле `normal_basis` задаёт этот базис без множителя ориентации UV-острова.
Расчёт приведён в [OOMesh.m](https://github.com/OoliteProject/oolite/blob/28ceaaba0d79d0b6ea1c044c6235e3616b170098/src/Core/OOMesh.m#L1540)
и [шейдере Oolite](https://github.com/OoliteProject/oolite/blob/28ceaaba0d79d0b6ea1c044c6235e3616b170098/Resources/Shaders/oolite-tangent-space-vertex.vertex#L49-L54).

## Воспроизведение

Нужны Python, NumPy и Pillow. Команды выполняются из корня elite-gfx.

```sh
curl -L --fail https://codeload.github.com/OoliteProject/oolite/tar.gz/28ceaaba0d79d0b6ea1c044c6235e3616b170098 -o /tmp/oolite.tar.gz
mkdir -p /tmp/oolite-source
tar -xzf /tmp/oolite.tar.gz -C /tmp/oolite-source --strip-components=1
python3 elite_pygame_lab/tools/import_oolite_pack.py --source /tmp/oolite-source
```

SHA-256 использованного архива равен
`ea12621028ff0f3e0584938179089cce3a8b62a48d46df13f4a8b72de3fb55b4`.
[Импортёр](../../../tools/import_oolite_pack.py) пересоздаёт сетки, карты и манифест.
Этот справочник и файлы лицензий сопровождают результат.
