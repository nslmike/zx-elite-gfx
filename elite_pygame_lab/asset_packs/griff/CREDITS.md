# Griff SpecGloss

Сетки и исходные карты созданы **Griff**. Импорт для атласа выполнен в
проекте **elite-gfx** из авторских выпусков SpecGloss / HP.

[Публикация Griff](https://bb.oolite.space/viewtopic.php?p=282274#p282274) ·
[публичная папка автора](https://drive.google.com/drive/folders/1_8EtoVvwNyXXvqe5GCGy1HuFRRw3NxB7) ·
[Griff Industries](https://wiki.alioth.net/index.php/Griff_Industries)

## Модели и источники

| ID | Модель | Авторский архив | Треугольники |
| --- | --- | --- | --- |
| 9 | Cobra Mk III, blue Trader | [Cobra Mk III v0.2](https://drive.google.com/file/d/1r3cDqy5qMIo2suZVioNx8HwxxLejrlW_/view) | 11 506 |
| 12 | Cobra Mk III, grey-red Pirate | тот же архив и корпус, отдельные diffuse/specgloss | 11 506 |
| 8 | GalCop Viper | [Viper v0.3](https://drive.google.com/file/d/1yXCDGWhK4yDeL0PQpLEI7LLI2ogLMwFW/view) | 7 800 |
| 16 | Sidewinder | [Sidewinder Materials](https://drive.google.com/file/d/1tacwmQ4WcDr6kM0O4jRWzAJCjyewtvFS/view) | 7 266 |

Использованы родные DAT с явными нормалями и PNG этих выпусков.
В авторском `shipdata.plist` blue назначена торговой Cobra, grey-red —
пиратской. Варианты 9 и 12 разделяют сетку, normal и emission. У Viper
сохранены Hull и LightStrip с картами из исходной конфигурации.

Проверенные [Cobra Resources](https://drive.google.com/file/d/1F-Z_oJjydAlRCdTstnEZJfYcSylbZcDD/view)
и [Viper Modding Resources](https://drive.google.com/file/d/1zDktpa_776PPxUAoCzUixTNgZmiiOjhI/view)
содержат OBJ и многослойные PSD. Для Cobra выбран DAT v0.2 с авторским
исправлением затенения задней части. Более ранний Resources OBJ содержит
11 505 треугольников.

## Лицензии

- **Cobra 9 и 12** распространяются по
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  Условие указано в `licence.txt` выпуска v0.2. Точная копия сохранена
  в `LICENSE-cobra3-upstream.txt`.
- **Viper 8** содержит два уведомления. `manifest.plist` указывает
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
  а `CC License.txt` — [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  Оригиналы сохранены в `MANIFEST-viper-upstream.plist` и
  `LICENSE-viper-upstream.txt`. Обе формулировки отражены в `model.license`.
  При распространении этой адаптации соблюдаются атрибуция,
  некоммерческое использование и ShareAlike.
- **Sidewinder 16** использует разрешение
  [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/),
  которое Griff опубликовал в [сообщении о SpecGloss Sidewinder](https://bb.oolite.space/viewtopic.php?p=282274#p282274),
  описывая Materials и Modding Resources и поясняя отсутствие notices
  в загрузках. В Materials OXP лицензионного файла нет.
  Отдельный [CC License.txt папки Sidewinder](https://drive.google.com/file/d/13iha6rYEWwpO-cqmBfVf2nRTXcsLyJ0j/view)
  содержит [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  Он приложен как `LICENSE-sidewinder-folder-license-upstream.txt`.
  [Корневой CC License.txt](https://drive.google.com/file/d/1J0J7AEgD1cl1znDLOiklBDKSgl5eFnZz/view)
  содержит тот же текст и сохранён в `LICENSE-root-folder-license-upstream.txt`.

## Преобразования

1. Из DAT получены наружные CCW-треугольники в сжатых NPZ с float32
   `(position.xyz, normal.xyz, uv.xy, material_slot)`. Позиции и нормали
   отражены по X в правую систему координат с +Y вверх и +Z к носу.
   Выполнено `V_gl = 1 − V_dat`, вычтен центр bbox и равномерно установлен
   максимальный радиус 1. Авторские нормали сохранены.
2. PNG уменьшены с Lanczos до максимума 1024×1024. LightStrip Viper остался
   512×512. Diffuse/emission сохранены в RGB sRGB. При загрузке в OpenGL
   рендерер один раз переворачивает изображения по вертикали.
3. Цветной RGB SpecGloss переведён в линейный RGB через `pow(rgb/255, 2.2)`.
   Альфа вынесена в линейный `gloss_map`, где белое означает глянец.
   Сохранён `gloss=1`. Множитель specular равен 1, что соответствует
   белому `specular_modulate_color` Oolite при наличии specular map.
4. Normal maps уменьшены, G инвертирован, векторы нормализованы.
   Установлено `normal_basis="cross(N,T)"` согласно исходному базису Oolite.
   Отражение X меняет знак B. На всех невырожденных UV-гранях подтверждены
   `dot(T_gl, M*T_src)≈1` и
   `dot(cross(N_gl,T_gl), M*cross(N_src,T_src))≈−1`, где M отражает X.
   Для 120 UV-вырожденных граней Cobra используется интерполированная нормаль.
   Исходные DAT не содержат TANGENTS. T вычислен из производной позиции по U.

Назначения карт сохранены в `SHIPDATA-*-upstream.plist`, числа проверки и
SHA-256 исходников — в `provenance.json`. Атлас вычисляет tangent
из производных в собственном пиксельном шейдере.

Соглашения сверены с [OBJ→DAT конвертером](https://github.com/OoliteProject/oolite-mesh-converters/blob/master/Obj2DatTexNorm.py),
[вершинным](https://github.com/OoliteProject/oolite/blob/master/Resources/Shaders/oolite-tangent-space-vertex.vertex)
и [пиксельным шейдерами Oolite](https://github.com/OoliteProject/oolite/blob/master/Resources/Shaders/oolite-default-shader.fragment),
а множитель блика — с [OOMaterialConvenienceCreators.m](https://github.com/OoliteProject/oolite/blob/master/src/Core/Materials/OOMaterialConvenienceCreators.m).

## Повторный импорт

Из корня проекта с Python, `numpy` и `Pillow` выполните команду

```sh
python3 elite_pygame_lab/tools/import_griff_pack.py --cache /tmp --download
```

Скрипт загружает три авторских ZIP и два notices, проверяет SHA-256,
пути ZIP, наружную ориентацию и tangent basis. Для локального повторного
импорта оставьте файлы `/tmp/griff-cobra3-oolite.zip`,
`/tmp/griff-viper-oolite.zip`, `/tmp/griff-sidewinder-oolite.zip` и
`/tmp/griff-{sidewinder,root}-folder-license.txt`, затем запустите ту же
команду без `--download`.
