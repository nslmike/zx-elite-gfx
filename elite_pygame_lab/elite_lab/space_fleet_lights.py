"""Авторские схемы огней для собственных сеток цветного атласа."""

from dataclasses import dataclass
from functools import lru_cache
from math import sin, tau

import numpy as np

from .model import COBRA_ID, DODO_ID, KRAIT_ID, MODELS
from .space_mesh import _unit, source_transform


_X = (1., 0., 0.)
_Y = (0., 1., 0.)
_Z = (0., 0., 1.)
_WHITE = (1., 1., 1.)
_AMBER = (2.10, 0.71, 0.13)
_ICE = (0.52, 1.32, 2.10)
_WARM = (1.75, 1.18, 0.57)
_TEAL = (0.14, 1.53, 1.45)
_VIOLET = (0.87, 0.23, 1.63)
_RED = (1.78, 0.13, 0.11)
_GREEN = (0.23, 1.73, 0.45)

# Размеры линзы — половины сторон в единичной сфере модели. Монтаж задают
# три вершины исходного треугольника с наружным обходом, а не проекция на
# обе стороны корпуса. Первые два веса определяют точку внутри треугольника.
MAX_LENSES = 4


@dataclass(frozen=True)
class Beat:
    period: float
    low: float
    high: float
    phase: float = 0.
    shape: int = 1

    def level(self, elapsed):
        # Ограничиваем фазу до вычисления sin, в том числе при очень большом seek.
        crest = .5 + .5 * sin(tau * (elapsed % self.period) / self.period + self.phase)
        return self.low + (self.high - self.low) * crest ** self.shape


@dataclass(frozen=True)
class HullLens:
    vertices: tuple[int, int, int]
    size: tuple[float, float]
    tint: tuple[float, float, float]
    beat: Beat | None = None
    weights: tuple[float, float] = (1 / 3, 1 / 3)
    axis: tuple[float, float, float] = _Z
    vent: bool = False


@dataclass(frozen=True)
class PanelSignal:
    beat: Beat
    tint: tuple[float, float, float]


@dataclass(frozen=True)
class PanelPair:
    first: PanelSignal
    second: PanelSignal
    axis: tuple[float, float, float] = _Y
    split: float = 0.
    min_z: float = -2.


@dataclass(frozen=True)
class FleetLights:
    lenses: tuple[HullLens, ...] = ()
    drive: tuple[Beat, ...] = ()
    drive_tint: tuple[float, float, float] = _WHITE
    panels: PanelPair | None = None
    legacy: int = 0


_CORIOLIS_LEFT = Beat(4.0, .45, 1.25, .3, 3)
_CORIOLIS_RIGHT = Beat(4.0, .45, 1.25, 3.44, 3)
_POD_DISTRESS = Beat(2.9, .27, 1.65, .7, 7)
_HERMIT_ENTRY = Beat(7.05, .52, 1.14, 1.42, 2)
_VIPER_LEFT = Beat(2.25, .24, 1.37, .2, 4)
_VIPER_RIGHT = Beat(2.25, .24, 1.37, 3.34, 4)
_PYTHON_PIRATE = Beat(5.7, .15, 1.3, .66, 7)
_COBRA_PIRATE = Beat(4.8, .28, 1.27, 1.83, 6)
_FDL_DRIVE = Beat(3.7, .56, 1.18, .51)
_THARGOID_GREEN = Beat(4.65, .44, 1.26, .31, 2)
_THARGOID_VIOLET = Beat(4.65, .40, 1.22, 2.45, 2)
_THARGON_CORE = Beat(2.75, .46, 1.28, 1.55, 3)
_DODO_ENTRY = Beat(6.2, .52, 1.22, .20, 2)
_DODO_INNER = Beat(6.2, .52, 1.22, 2.24, 2)
_DODO_READY = Beat(6.2, .45, 1.28, 4.29, 2)


PROFILES = {
    # Coriolis. Попеременные огни на створе; глубинные лампы дока постоянны.
    0: FleetLights((
        HullLens((3, 0, 1), (.026, .012), _WARM, _CORIOLIS_RIGHT, (.30, .40), _X),
        HullLens((1, 2, 3), (.026, .012), _AMBER, _CORIOLIS_LEFT, (.30, .40), _X),
    ), panels=PanelPair(PanelSignal(_CORIOLIS_LEFT, (1.24, .75, .40)),
                        PanelSignal(_CORIOLIS_RIGHT, (.91, 1.02, 1.33)), _X, min_z=.65)),
    # Ракета. Пара узких индикаторов сбоку и быстрый маршевый импульс.
    1: FleetLights((
        HullLens((5, 2, 1), (.026, .009), _AMBER, Beat(1.55, .40, 1.23, .63, 3)),
        HullLens((7, 4, 3), (.026, .009), _AMBER, Beat(1.55, .40, 1.23, .63, 3)),
        HullLens((8, 3, 2), (.018, .010), _WARM, weights=(.13, .48), axis=_X),
    ), (Beat(1.82, .76, 1.28, .40),), (1.63, .91, .57)),
    # Капсула. Тот же спасательный ритм виден с тёмной стороны.
    2: FleetLights((
        HullLens((0, 1, 3), (.024, .014), _AMBER, _POD_DISTRESS, (.25, .29)),
        HullLens((2, 1, 0), (.019, .011), _WARM, _POD_DISTRESS, (.35, .25), _Y),
    ), panels=PanelPair(PanelSignal(_POD_DISTRESS, (1.42, .73, .28)),
                        PanelSignal(_POD_DISTRESS, (1.42, .73, .28)))),
    # Контейнер. Два коротких индикатора обращения и один медленный трекер.
    4: FleetLights((
        HullLens((6, 1, 0), (.014, .019), _TEAL, Beat(5.15, .25, 1.25, .49, 5), (.24, .44), _X),
        HullLens((0, 5, 6), (.011, .017), _AMBER, weights=(.28, .43), axis=_X),
        HullLens((4, 9, 5), (.018, .014), _WARM, Beat(5.15, .31, 1.10, 2.6, 4), (.29, .37), _X),
    )),
    # Убежище в астероиде. Только лампы в искусственной шахте.
    7: FleetLights(panels=PanelPair(PanelSignal(_HERMIT_ENTRY, (1.25, .76, .41)),
                                    PanelSignal(_HERMIT_ENTRY, (1.25, .76, .41)))),
    # Viper. Холодная патрульная пара чередуется; снизу — короткие габариты.
    8: FleetLights((
        HullLens((1, 8, 4), (.034, .011), _ICE, _VIPER_LEFT, (.26, .26)),
        HullLens((3, 7, 1), (.034, .011), (0.65, 0.86, 2.24), _VIPER_RIGHT, (.49, .25)),
        HullLens((4, 2, 0), (.017, .010), _WARM, weights=(.41, .25)),
        HullLens((3, 0, 2), (.017, .010), _ICE, weights=(.41, .34)),
    ), (Beat(2.38, .84, 1.23, .18), Beat(2.38, .84, 1.23, 2.12)), (.89, 1.02, 1.20)),
    COBRA_ID: FleetLights(legacy=1),
    10: FleetLights(legacy=2),
    # Python пиратов. Один редкий красный сигнал, бортовой остаточный огонь.
    11: FleetLights((
        HullLens((4, 3, 1), (.015, .021), _RED, _PYTHON_PIRATE, (.42, .38)),
        HullLens((5, 4, 2), (.030, .010), (1.46, .60, .18), weights=(.41, .32)),
        HullLens((8, 2, 0), (.019, .010), _RED, Beat(5.7, .24, 1.20, 2.10, 6), (.27, .49)),
    ), (Beat(3.72, .69, 1.26, 1.22, 2),), (1.79, .91, .66)),
    # Cobra пиратов. Неровная пара сопел и два одиночных огня у разных кромок.
    12: FleetLights((
        HullLens((3, 5, 8), (.013, .025), _RED, _COBRA_PIRATE, (.43, .32), _X),
        HullLens((0, 4, 6), (.020, .010), _VIOLET, weights=(.36, .34)),
        HullLens((11, 4, 0), (.017, .011), _AMBER, Beat(4.8, .24, 1.06, 3.67, 5), (.35, .40)),
    ), (Beat(2.63, .69, 1.25, .47, 2), Beat(3.92, .76, 1.22, 1.97)), (1.37, .83, 1.09)),
    # Asp. Боковые обзорные огни и спокойный отдельный верхний маяк.
    13: FleetLights((
        HullLens((10, 8, 3), (.030, .012), _ICE, weights=(.30, .37)),
        HullLens((6, 9, 11), (.030, .012), _ICE, weights=(.33, .37)),
        HullLens((12, 11, 10), (.022, .013), _AMBER, Beat(5.9, .54, 1.20, .7), (.49, .29), _X),
        HullLens((0, 7, 6), (.019, .010), _WARM, weights=(.26, .39)),
    ), (Beat(5.35, .82, 1.19, 1.12),), (.77, 1.08, 1.18)),
    # Fer-de-Lance. Две сдержанные маневровые апертуры на кормовой стенке.
    14: FleetLights((
        HullLens((5, 1, 0), (.028, .008), _WARM, weights=(.43, .23)),
        HullLens((0, 4, 8), (.028, .008), _WARM, weights=(.34, .23)),
        HullLens((2, 6, 7), (.032, .012), (1.80, .80, .28), _FDL_DRIVE, (.48, .27), _X, True),
        HullLens((7, 3, 2), (.032, .012), (1.80, .80, .28),
                 Beat(3.7, .56, 1.18, 1.11), (.25, .48), _X, True),
    )),
    # Thargoid. Ось корпуса X. Две штатные полосы −X и три отметки по ободу.
    15: FleetLights((
        HullLens((0, 8, 9), (.026, .014), _GREEN, _THARGOID_GREEN, (.27, .40)),
        HullLens((4, 12, 13), (.026, .014), _VIOLET, _THARGOID_VIOLET, (.26, .41)),
        HullLens((15, 7, 6), (.024, .015), _TEAL, Beat(4.65, .43, 1.15, 4.55, 2), (.41, .28), _Y),
    ), panels=PanelPair(PanelSignal(_THARGOID_GREEN, (.38, 1.25, .67)),
                        PanelSignal(_THARGOID_VIOLET, (1.58, .29, 2.52)))),
    # Sidewinder. Компактный поперечный маяк и узкая кормовая индикация.
    16: FleetLights((
        HullLens((4, 3, 0), (.026, .012), _TEAL, Beat(3.44, .31, 1.28, 2.03, 4), (.25, .33), _X),
        HullLens((1, 2, 4), (.018, .010), _WARM, weights=(.34, .31), axis=_X),
        HullLens((5, 3, 4), (.022, .009), _TEAL, weights=(.21, .57), axis=_X),
    ), (Beat(2.92, .81, 1.24, 2.43),), (.91, 1.12, .97)),
    # Krait. Огни установлены у оснований усов, а не на тонких стержнях.
    KRAIT_ID: FleetLights((
        HullLens((3, 1, 0), (.023, .011), _AMBER, Beat(4.32, .39, 1.25, .65, 3), (.66, .18)),
        HullLens((0, 1, 4), (.023, .011), _AMBER, Beat(4.32, .39, 1.25, 1.63, 3), (.16, .18)),
        HullLens((0, 2, 3), (.018, .009), _WARM, weights=(.39, .31), axis=_X),
    ), (Beat(2.58, .74, 1.29, 1.08), Beat(3.14, .78, 1.23, 3.02)), (1.21, 1.06, .81)),
    # Thargon. Ритм течёт по трём боковым фасетам к панели +X.
    18: FleetLights((
        HullLens((5, 0, 1), (.026, .013), _TEAL, Beat(2.75, .33, 1.23, .04, 3), (.43, .30)),
        HullLens((7, 2, 3), (.027, .012), _ICE, Beat(2.75, .34, 1.23, 2.13, 3), (.45, .30), _Y),
        HullLens((5, 9, 4), (.027, .012), _TEAL, Beat(2.75, .33, 1.23, 4.22, 3), (.31, .40)),
    ), panels=PanelPair(PanelSignal(_THARGON_CORE, (.35, 1.13, 2.13)),
                        PanelSignal(_THARGON_CORE, (.35, 1.13, 2.13)))),
    # Dodo. Медленный проход сигнала от нижней кромки к верхнему огню.
    DODO_ID: FleetLights((
        HullLens((0, 3, 7), (.013, .033), _TEAL, _DODO_ENTRY, (.32, .38), _X),
        HullLens((6, 2, 0), (.013, .033), _ICE, _DODO_INNER, (.30, .38), _X),
        HullLens((0, 7, 6), (.025, .011), _WARM, _DODO_READY, (.72, .14), _X),
    ), panels=PanelPair(PanelSignal(_DODO_ENTRY, (.62, 1.08, 1.60)),
                        PanelSignal(_DODO_INNER, (.81, 1.01, 1.45)), min_z=.72)),
}


@lru_cache(maxsize=len(PROFILES))
def lens_mounts(model_id):
    """Центр, нормаль и продольная ось линз в координатах готовой сетки."""
    origin, radius = source_transform(model_id)
    vertices = MODELS[model_id].vertices
    mounts = []
    for lens in PROFILES[model_id].lenses:
        points = np.array([(vertices[i].x, vertices[i].y, vertices[i].z) for i in lens.vertices])
        points = (points - origin) / radius
        a, b = lens.weights
        center = (a, b, 1. - a - b) @ points
        normal = _unit(np.cross(points[1] - points[0], points[2] - points[0]))
        axis = _unit(lens.axis - np.dot(lens.axis, normal) * normal)
        mounts.append((tuple(center), tuple(normal), tuple(axis)))
    return tuple(mounts)
