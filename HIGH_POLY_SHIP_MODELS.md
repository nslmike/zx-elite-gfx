# Детализированные модели кораблей Elite и Elite Dangerous

Подборка для OpenGL-атласа и будущей игры. Проверено 4 ноября 2026 года.

В атласе подключены три внешних набора. Базовая Oolite покрывает позиции
00–19, Griff SpecGloss даёт четыре варианта для трёх типов кораблей, а
Sinonatrix Courier из Pioneer занимает отдельную позицию 20. Кнопки
«Модели» и «Текстуры» работают во время осмотра. [Запуск и состав наборов](elite_pygame_lab/README.md#наборы-моделей-и-текстур).

Для первого крупного плана в атласе подойдёт **Cobra Mk III из Griff SpecGloss**. Она сохраняет клиновидный силуэт классической Elite и доступна в OBJ с текстурами. Для игры с возможностью коммерческого выпуска интересен **Sinonatrix Courier из Pioneer**. У него три уровня детализации, а лицензия художественных ресурсов допускает коммерческое использование.

## Griff SpecGloss для Oolite

Griff опубликовал **Cobra Mk III, Cobra Mk I, Sidewinder, Viper, Mamba, Gecko и Adder**. В папке нужного корабля для переноса в другой рендерер следует брать архив **Resources** или **Modding Resources**. В нём лежат сетка в формате Wavefront OBJ и текстуры. Архив с **Materials** в названии подготовлен для установки в саму Oolite. У Sidewinder автор отдельно описывает варианты окраски, свечение двигателей и эффект износа. [Папка с моделями](https://drive.google.com/drive/folders/1_8EtoVvwNyXXvqe5GCGy1HuFRRw3NxB7) · [Объявление Griff, галерея и описание архивов](https://bb.oolite.space/viewtopic.php?t=21170) · [Авторский рендер Cobra Mk III](https://i.postimg.cc/D0DrBQh3/griff-specgloss-cobra3.jpg)

В объявлении Griff указал **CC BY-NC-SA 3.0**. При подключении проверены также уведомления из конкретных архивов. Для Cobra Mk III в архиве указана **CC BY-NC-SA 4.0**. У Viper манифест указывает **CC BY-NC-SA 4.0**, а вложенный файл лицензии — **CC BY-NC 4.0**. В карточке Viper приведены обе записи. Для Sidewinder в атласе указаны условия авторской публикации, **CC BY-NC-SA 3.0**. [Объявление Griff](https://bb.oolite.space/viewtopic.php?t=21170) · [Лицензия из архива Cobra](elite_pygame_lab/asset_packs/griff/LICENSE-cobra3-upstream.txt) · [Уведомление из архива Viper](elite_pygame_lab/asset_packs/griff/LICENSE-viper-upstream.txt) · [Метаданные по моделям](elite_pygame_lab/asset_packs/griff/pack.json).

## Базовый флот Oolite и другие наборы Griff

Для общего вида кольца доступны более лёгкие модели. Числа ниже взяты из полей `NVERTS` и `NFACES` в заголовках файлов **базовой Oolite**. Сетки SpecGloss выпущены отдельно.

| Модель | Вершины | Грани | Файл с геометрией |
| --- | ---: | ---: | --- |
| Cobra Mk III | 1 548 | 1 076 | [oolite_cobra3.dat](https://github.com/OoliteProject/oolite/blob/master/Resources/Models/oolite_cobra3.dat) |
| Krait | 1 023 | 1 509 | [oolite_krait.dat](https://github.com/OoliteProject/oolite/blob/master/Resources/Models/oolite_krait.dat) |

Для моделей и текстур из базовой игры Oolite разрешает выбрать **GPL версии 2 или новее** либо **CC BY-NC-SA 3.0**. У отдельных расширений условия могут отличаться. Например, в манифесте **Griff Krait** указана **CC BY-NC-SA 4.0**. В более широком наборе Griff также есть Python, Asp, Fer-de-Lance, таргоидские корабли и орбитальные станции. [Каталог геометрии Oolite](https://github.com/OoliteProject/oolite/tree/master/Resources/Models) · [Лицензия базовой игры и ресурсов](https://github.com/OoliteProject/oolite/blob/master/LICENSE.md) · [Манифест Griff Krait](https://ooliteproject.github.io/oolite-expansion-catalog/expansions/oolite.oxp.Griff.Krait/oolite.oxp.Griff.Krait.html) · [Галерея и наборы Griff Industries](https://wiki.alioth.net/index.php/Griff_Industries)

## Pioneer для будущей игры

**Sinonatrix Courier** — самостоятельный дизайн космического курьера. В каталоге есть сетки `sinonatrix_hi.dae`, `sinonatrix_med.dae`, `sinonatrix_low.dae` и отдельный кокпит. Для корпуса и двигателей опубликованы карты цвета, нормалей, бликов и свечения. Это позволяет на одной орбите показывать лёгкую сетку издалека и детальную при подлёте. В портфолио художника можно также рассмотреть родственные **Natrix Courier** и **Coronatrix Courier**. [Файлы Sinonatrix](https://github.com/pioneerspacesim/pioneer/tree/master/data/models/ships/sinonatrix) · [Галерея художника Bálint Szilárd](https://bszlrd.hu/project/pioneer) · [Как устроены материалы и уровни детализации Pioneer](https://dev.pioneerspacesim.net/workflow/3d_assets/model_system)

Pioneer публикует художественные ресурсы под **CC BY-SA 3.0**. Лицензия допускает коммерческое использование, требует указания авторства и сохранения лицензии при распространении переработанной модели. [Лицензирование ресурсов Pioneer](https://github.com/pioneerspacesim/pioneer/blob/master/AUTHORS.txt) · [Текст лицензии CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

## Корабли в облике Elite Dangerous

| Работа | Что доступно и для чего пригодится |
| --- | --- |
| **Imperial Eagle**, Jaume González Verge | Фанатская реконструкция на **751,2 тыс. треугольников**. В интерактивном просмотрщике можно рассмотреть крылья, двигатели и стыки панелей. На странице указана возможность просмотра, скачивание не указано. [Открыть модель в 3D](https://sketchfab.com/3d-models/imperial-eagle-spaceship-e9ae31158c7a4f1590d795e02980ff06) |
| **Cobra Mk III**, Kahnindustries | Фанатская модель для 3D-печати. Доступны STL целого корабля и отдельных частей. Геометрию можно изучать как образец формы. Для игрового рендера потребуются материалы и UV-развёртка. Автор указывает некоммерческое использование. [Страница модели и файлы STL](https://cults3d.com/en/3d-model/gadget/cobra-mkiii-2-part-elite-dangerous) |
| **Jameson Crash Site**, Ecthelion | Доступная для скачивания реконструкция места крушения Cobra Mk III по 509 кадрам из игры. **8 млн треугольников относятся ко всей сцене**, включая грунт. Пригодится как образец повреждённой обшивки. На странице указана CC Attribution-NonCommercial. [Осмотреть место крушения в 3D](https://sketchfab.com/3d-models/jameson-crash-site-0ed18332cd5a45839bf6a4bcf55905b5) |
| **Корабли Elite Dangerous**, John Kelly | В портфолио художника игры есть изображения Adder, Fer-de-Lance, Imperial Cutter и других судов. По ним можно сравнить пропорции и отделку с тем, как корабли выглядят в игре. [Галерея художника](https://www.artstation.com/artwork/Xno9Ra) |
