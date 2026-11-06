"""Световые профили кораблей и локальное свечение в линейном HDR."""

from functools import lru_cache
from math import sin, tau
from pathlib import Path

import moderngl

from .space_fleet_lights import MAX_LENSES, PROFILES, lens_mounts


_SHADER = Path(__file__).with_name("shaders") / "space_bloom.frag"


def ship_effects_profile(ship):
    # При смене набора часть его моделей может оставаться из каталога Elite.
    if ship.source_id == "elite" and not (ship.meshes or ship.materials):
        return PROFILES.get(ship.model_id)
    return None


def legacy_light_levels(profile, elapsed):
    def wave(period, phase):
        # Сначала берём остаток в double: даже предельное время seek не
        # переполняет вычисление фазы и не попадает в uniform как float32.
        return sin(tau * (elapsed % period) / period + phase)

    if profile == 1:
        marker = .83 + .17 * wave(5.2, .17)
        return (.99 + .24 * wave(2.85, .42), .99 + .23 * wave(3.6, 2.1)), marker
    engine = 1.02 + .28 * wave(4.4, 1.37)
    beacon = .28 + .72 * (.5 + .5 * wave(3.2, .6)) ** 4
    return (engine, engine), beacon


@lru_cache(maxsize=len(PROFILES))
def _lens_uniforms(model_id):
    lenses = PROFILES[model_id].lenses
    anchors, normals, tangents = [], [], []
    for lens, (point, normal, tangent) in zip(lenses, lens_mounts(model_id)):
        anchors.append((*point, lens.size[0]))
        normals.append((*normal, lens.size[1]))
        tangents.append((*tangent, float(lens.vent)))
    padding = [(0., 0., 0., 0.)] * (MAX_LENSES - len(lenses))
    return anchors + padding, normals + padding, tangents + padding


def configure_ship_effects(hull, model_id, profile, elapsed):
    """Применить собственный профиль к текущему корпусу перед draw."""
    if profile is None:
        hull["u_effects_profile"].value = 0
        return
    if profile.legacy:
        hull["u_effects_profile"].value = profile.legacy
        engine, marker = legacy_light_levels(profile.legacy, elapsed)
        hull["u_engine_pulse"].value = engine
        hull["u_beacon_level"].value = marker
        return

    hull["u_effects_profile"].value = 3
    hull["u_engine_pulse"].value = ((profile.drive[0].level(elapsed), profile.drive[-1].level(elapsed))
                                    if profile.drive else (1., 1.))
    hull["u_drive_tint"].value = profile.drive_tint
    hull["u_lens_count"].value = len(profile.lenses)
    if profile.lenses:
        anchors, normals, tangents = _lens_uniforms(model_id)
        hull["u_lens_anchor"].value = anchors
        hull["u_lens_normal"].value = normals
        hull["u_lens_tangent"].value = tangents
        radiance = [(*lens.tint, lens.beat.level(elapsed) if lens.beat else 1.) for lens in profile.lenses]
        hull["u_lens_radiance"].value = radiance + [(0., 0., 0., 0.)] * (MAX_LENSES - len(radiance))
    panels = profile.panels
    hull["u_panel_active"].value = panels is not None
    if panels:
        hull["u_panel_split"].value = (*panels.axis, panels.split)
        hull["u_panel_min_z"].value = panels.min_z
        for name, signal in (("u_panel_first", panels.first), ("u_panel_second", panels.second)):
            level = signal.beat.level(elapsed)
            hull[name].value = tuple(channel * level for channel in signal.tint)


class SpaceBloom:
    """Два уменьшенных буфера и финальный проход перед интерфейсом."""

    def __init__(self, ctx, vertex_shader):
        self.ctx = ctx
        self.resources = []
        source = _SHADER.read_text(encoding="utf-8")
        try:
            self.downsample, self.blur, self.composite, self.composite_msaa = (
                self._pass(stage, source, vertex_shader)
                for stage in ("DOWNSAMPLE", "BLUR", "COMPOSITE", "COMPOSITE_MSAA")
            )
            self.composite[0]["bloom"].value = 1
            self.composite_msaa[0]["bloom"].value = 1
        except Exception:
            self.release()
            raise

    def _pass(self, stage, source, vertex_shader):
        program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=source.replace("#version 330", f"#version 330\n#define {stage}", 1),
        )
        self.resources.append(program)
        vao = self.ctx.vertex_array(program, [])
        self.resources.append(vao)
        program["frame"].value = 0
        return program, vao

    def draw(self, world, target, size):
        ctx = self.ctx
        ctx.enable_only(0)
        first, second = world.bloom_frames
        first.use()
        ctx.viewport = (0, 0, *first.size)
        world.texture.use(0)
        self.downsample[1].render(vertices=3)
        spread = max(1., min(2.5, world.size[1] / 900.))
        blur, quad = self.blur
        for source, dest, axis in ((first, second, (spread / first.width, 0.)),
                                   (second, first, (0., spread / first.height))):
            dest.use()
            source.color_attachments[0].use(0)
            blur["step_uv"].value = axis
            quad.render(vertices=3)
        target.use()
        ctx.viewport = (0, 0, *size)
        first.color_attachments[0].use(1)
        world.color.use(0)
        if world.samples:
            composite, quad = self.composite_msaa
            composite["samples"].value = world.samples
        else:
            composite, quad = self.composite
        composite["bloom_uv_scale"].value = tuple(full / (4 * low)
                                                 for full, low in zip(world.size, first.size))
        quad.render(vertices=3)

    def release(self):
        for resource in reversed(self.resources):
            resource.release()
