"""Фильтрация окружения для отражений на классической обшивке."""

from functools import lru_cache

import numpy as np


REFLECTION_SIZE = (512, 256)
REFLECTION_LEVELS = 6
DFG_SIZE = 64
_FILL_DIRECTION = np.array((.27, .78, .57), dtype="f4")
_FILL_DIRECTION /= np.linalg.norm(_FILL_DIRECTION)


def cube_project(face: int, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = directions.T
    mappings = ((-z, -y, x), (z, -y, -x), (x, z, y),
                (x, -z, -y), (x, -y, z), (-x, -y, -z))
    s, t, depth = mappings[face]
    divisor = np.where(depth > 0, depth, 1)
    return s / divisor, t / divisor, depth


def _grid(width, height):
    longitude = ((np.arange(width, dtype="f4") + .5) / width - .5) * (2 * np.pi)
    latitude = ((np.arange(height, dtype="f4") + .5) / height - .5) * np.pi
    cos_lon, sin_lon = np.cos(longitude), np.sin(longitude)
    cos_lat, sin_lat = np.cos(latitude)[:, None], np.sin(latitude)[:, None]
    directions = np.stack((cos_lat * cos_lon, np.broadcast_to(sin_lat, (height, width)),
                           cos_lat * sin_lon), axis=-1)
    tangent = np.stack((-sin_lon, np.zeros_like(sin_lon), cos_lon), axis=-1)
    return directions, np.broadcast_to(tangent, directions.shape)


def _bilinear(image, x, y, *, wrap_x=False):
    height, width = image.shape[:2]
    ix, iy = np.floor(x).astype("i4"), np.floor(y).astype("i4")
    fx, fy = (x - ix.astype("f4"))[..., None], (y - iy.astype("f4"))[..., None]
    x0, x1 = (ix % width, (ix + 1) % width) if wrap_x else (
        ix.clip(0, width - 1), (ix + 1).clip(0, width - 1))
    y0, y1 = iy.clip(0, height - 1), (iy + 1).clip(0, height - 1)
    a = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    b = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return a * (1 - fy) + b * fy


def _sky_to_latlong(faces, directions):
    flat = directions.reshape(-1, 3)
    axis = np.abs(flat).argmax(axis=1)
    faces_at = axis * 2 + (flat[np.arange(len(flat)), axis] < 0)
    radiance = np.empty_like(flat)
    for index, image in enumerate(faces):
        selected = faces_at == index
        u, v, _ = cube_project(index, flat[selected])
        size = len(image)
        radiance[selected] = _bilinear(image, (u + 1) * (.5 * size) - .5,
                                       (v + 1) * (.5 * size) - .5)
    return radiance.reshape(directions.shape)


def _lighting_environment(sky, directions):
    # Прежняя холодная полусфера и небольшая широкая доля света сверху сбоку.
    # Это только отражённый свет. Видимое небо и прямой диск хранятся отдельно.
    up = directions[..., 1:2] * .5 + .5
    lobe = ((directions @ _FILL_DIRECTION) * .5 + .5)[..., None] ** 6
    return (sky * .30 + np.array((.097, .123, .166), dtype="f4") +
            up * np.array((.110, .149, .197), dtype="f4") +
            lobe * np.array((.102, .106, .110), dtype="f4"))


def _ggx_half_vectors(roughness, count):
    # Последовательность Хаммерсли, полувекторы распределены по GGX NDF.
    indices = np.arange(count, dtype="i4")
    radial = (indices.astype("f4") + .5) / count
    angle = np.zeros(count, dtype="f4")
    for bit in range((count - 1).bit_length()):
        angle += ((indices >> bit) & 1).astype("f4") * (2.0 ** (-bit - 1))
    angle *= 2 * np.pi
    alpha2 = roughness ** 4
    z = np.sqrt((1 - radial) / (1 + (alpha2 - 1) * radial))
    r = np.sqrt(np.maximum(1 - z * z, 0))
    return r * np.cos(angle), r * np.sin(angle), z


def _prefilter(environment, width, height, roughness):
    normals, tangent = _grid(width, height)
    n, t = normals.reshape(-1, 3), tangent.reshape(-1, 3)
    b = np.cross(n, t)
    hx, hy, hz = _ggx_half_vectors(np.float32(roughness), 96)
    # При V=N отражённый луч L=2(H·N)H−N, веса пропорциональны N·L.
    valid = hz * hz > .5
    local_x, local_y = 2 * hz[valid] * hx[valid], 2 * hz[valid] * hy[valid]
    local_z = 2 * hz[valid] ** 2 - 1
    weights = local_z / local_z.sum()
    height_in, width_in = environment.shape[:2]
    filtered = np.empty_like(n)
    for start in range(0, len(n), 1024):
        sl = slice(start, start + 1024)
        rays = (n[sl, None] * local_z[:, None] + t[sl, None] * local_x[:, None] +
                b[sl, None] * local_y[:, None])
        x, y, z = np.moveaxis(rays, -1, 0)
        u = np.arctan2(z, x) / (2 * np.pi) + .5
        v = np.arcsin(y.clip(-1, 1)) / np.pi + .5
        colors = _bilinear(environment, u * width_in - .5, v * height_in - .5, wrap_x=True)
        filtered[sl] = np.einsum("nsc,s->nc", colors, weights)
    return filtered.reshape(height, width, 3)


@lru_cache(maxsize=1)
def reflection_levels(face_size: int, faces: tuple[bytes, ...]) -> tuple[bytes, ...]:
    """Шесть уровней с линейным шагом perceptual roughness в 2D HDR-текстуре."""
    sky = [np.frombuffer(face, dtype="f2").reshape(face_size, face_size, 3).astype("f4")
           for face in faces]
    width, height = REFLECTION_SIZE
    directions, _ = _grid(width, height)
    environment = _lighting_environment(_sky_to_latlong(sky, directions), directions)
    levels = []
    for level in range(REFLECTION_LEVELS):
        rgb = environment if level == 0 else _prefilter(
            environment, width >> level, height >> level, level / (REFLECTION_LEVELS - 1))
        # Clamp по широте смотрит на крайние строки. Каждый полюс должен
        # иметь одно значение независимо от долготы отражённого луча.
        rgb[[0, -1]] = rgb[[0, -1]].mean(axis=1, keepdims=True)
        rgba = np.ones((*rgb.shape[:2], 4), dtype="f2")
        rgba[..., :3] = rgb
        levels.append(rgba.tobytes())
    return tuple(levels)


@lru_cache(maxsize=1)
def dfg_lut() -> bytes:
    """Интеграл GGX/Smith, разложенный на множитель F0 и вклад Френеля."""
    domain = (np.arange(DFG_SIZE, dtype="f4") + .5) / DFG_SIZE
    roughness, nv = domain[:, None, None], domain[None, :, None]
    hx, _, nh = _ggx_half_vectors(roughness, 512)
    vh = np.maximum(np.sqrt(1 - nv * nv) * hx + nv * nh, 0)
    nl = np.maximum(2 * vh * nh - nv, 0)
    alpha2 = roughness ** 4
    smith = nl * np.sqrt(nv * nv * (1 - alpha2) + alpha2) + nv * np.sqrt(
        nl * nl * (1 - alpha2) + alpha2)
    visibility = 2 * nl * vh / (nh * smith)
    fresnel = (1 - vh.clip(0, 1)) ** 5
    a = np.mean((1 - fresnel) * visibility, axis=-1)
    b = np.mean(fresnel * visibility, axis=-1)
    # Конечная выборка немного завышает энергию у самого края.
    energy = np.maximum(a + b, 1)
    return np.stack((a / energy, b / energy), axis=-1).astype("f2").tobytes()
