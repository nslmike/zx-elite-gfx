#version 330

uniform sampler2D u_surface;
uniform sampler2D u_surface_slope;
uniform sampler2D u_stencils;
uniform bool u_author_maps;
uniform bool u_imported;
uniform bool u_cross_bitangent;
uniform bool u_cobra_finishes;
uniform bool u_classic_finishes;
uniform vec4 u_finish_layout;
uniform vec4 u_finish_response;
uniform vec4 u_damage_profile;
uniform vec3 u_glass_tint;
uniform sampler2D u_finish_slope;
uniform mat4 u_model;
uniform sampler2D u_diffuse_map;
uniform sampler2D u_normal_map;
uniform sampler2D u_specular_map;
uniform sampler2D u_emission_map;
uniform sampler2D u_gloss_map;
uniform float u_gloss;
uniform float u_glow_gain;
uniform vec3 u_specular_tint;
uniform vec3 u_patch;
uniform vec2 u_stencil_tile;
uniform samplerCube u_sky;
uniform sampler2DShadow u_shadow;
uniform bool u_shadow_enabled;
uniform vec3 u_eye;
uniform vec3 u_sun_direction;
uniform vec3 u_hull_color;
uniform vec3 u_secondary_color;
uniform vec3 u_accent_color;
uniform vec3 u_metal_color;
uniform vec3 u_engine_color;
uniform vec3 u_window_color;
uniform vec2 u_uv_offset;
uniform float u_texture_scale;
uniform float u_wear;
uniform float u_roughness;
uniform float u_metallic;
uniform float u_marking_style;
uniform float u_seed;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_local;
in vec3 v_local_normal;
in vec2 v_uv;
in vec3 v_shadow;
flat in int v_material;
out vec4 frag_color;

const float PI = 3.14159265359;
// Выбранный для нашей Cobra тёмный графит, в линейном пространстве.
const vec3 COBRA_GLASS_TINT = vec3(0.115, 0.135, 0.145);

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float value_noise(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash31(cell), hash31(cell + vec3(1, 0, 0)), f.x),
                   mix(hash31(cell + vec3(0, 1, 0)), hash31(cell + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash31(cell + vec3(0, 0, 1)), hash31(cell + vec3(1, 0, 1)), f.x),
                   mix(hash31(cell + vec3(0, 1, 1)), hash31(cell + vec3(1, 1, 1)), f.x), f.y), f.z);
}

float band(float p, float center, float width) {
    float edge = max(fwidth(p), 0.0005);
    return 1.0 - smoothstep(width - edge, width + edge, abs(p - center));
}

// Наклон берётся из непрерывно фильтруемой UV-карты, а не из разности
// отфильтрованных высот соседних экранных пикселей. Цепное правило переносит
// dH/dUV в мировую касательную плоскость при любом масштабе/ракурсе корпуса.
vec3 relief_from_gradient(vec3 normal, vec2 height_xy, float strength) {
    vec3 dx = dFdx(v_position), dy = dFdy(v_position);
    vec3 r1 = cross(dy, normal), r2 = cross(normal, dx);
    float det = dot(dx, r1);
    vec3 gradient = sign(det) * (height_xy.x * r1 + height_xy.y * r2);
    return normalize(abs(det) * normal - strength * gradient);
}

// Непрерывная процедурная высота камня использует тот же мировой базис.
vec3 relief_normal(vec3 normal, float height, float strength) {
    return relief_from_gradient(normal, vec2(dFdx(height), dFdy(height)), strength);
}

vec3 fresnel_schlick(float cosine, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 display_color(vec3 color) {
    color = (color * (2.51 * color + 0.03)) /
            (color * (2.43 * color + 0.59) + 0.14);
    return pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
}

struct Surface {
    vec3 normal;
    vec3 albedo;
    float roughness;
    float metallic;
    float occlusion;
    vec3 emission;
};

// Силовые секции идут вдоль двух верхних плеч исходной Cobra. Границы
// принадлежат корпусу, а fwidth сглаживает их и на общем плане, и вблизи.
float cobra_shoulder_metal() {
    float diagonal = abs(v_local.x) + 1.34 * v_local.z;
    float plate = band(diagonal, -0.103, 0.066);
    float aa = max(fwidth(v_local.z), 0.0004);
    plate *= smoothstep(-0.414 - aa, -0.414 + aa, v_local.z);
    plate *= 1.0 - smoothstep(-0.095 - aa, -0.095 + aa, v_local.z);
    float ax = abs(v_local.x);
    float ax_aa = max(fwidth(ax), 0.0004);
    plate *= smoothstep(0.105 - ax_aa, 0.105 + ax_aa, ax);
    return plate * smoothstep(0.80, 0.90, v_local_normal.y);
}

// Четыре пространственные компоновки силовых секций; масштаб, доля и
// смещение принадлежат профилю корабля, без переключения shader по ID.
float classic_metal_span(float expansion) {
    if (u_cobra_finishes) return cobra_shoulder_metal();
    vec3 p = v_local;
    p.z = p.z * u_finish_layout.z - u_finish_layout.w;
    float width = mix(0.020, 0.165, clamp(u_finish_layout.y, 0.0, 1.0)) + expansion;
    float mask;
    if (u_finish_layout.x < 1.5) {
        // Поперечные обоймы у двух концов корпуса, с разной шириной.
        mask = max(band(p.z, -0.38, width), band(p.z, 0.33, width * 0.72));
    } else if (u_finish_layout.x < 2.5) {
        // Парные носовые/плечевые листы, вдоль силовой диагонали.
        float diagonal = abs(p.x) - 0.30 * p.z;
        mask = band(diagonal, 0.32, width);
        float aa = max(fwidth(p.z), 0.0005);
        mask *= smoothstep(-0.34 - aa, -0.34 + aa, p.z);
        mask *= 1.0 - smoothstep(0.66 - aa, 0.66 + aa, p.z);
        mask *= 0.30 + 0.70 * smoothstep(0.25, 0.70, abs(v_local_normal.y));
    } else if (u_finish_layout.x < 3.5) {
        // Продольные грузовые усиления и широкий защитный кормовой срез.
        float rail = abs(p.x) + 0.42 * abs(p.y);
        mask = band(rail, 0.34, width);
        mask *= 1.0 - smoothstep(0.55, 0.88, abs(v_local_normal.z));
        mask = max(mask, band(p.z, -0.43, width * 0.72));
    } else {
        // Радиальный пояс и диагональные технические сектора станций/alien.
        float radius = length(p.xz);
        float ring = band(radius, 0.52, width);
        float spokes = band(abs(p.x) - abs(p.z), 0.0, width * 0.62);
        spokes *= smoothstep(0.18, 0.31, radius);
        mask = max(ring, spokes * smoothstep(0.25, 0.55, abs(v_local_normal.y)));
    }
    return clamp(mask, 0.0, 1.0);
}

float classic_metal_mask() {
    return classic_metal_span(0.0);
}

// Обтирка с обрывами лежит в краске рядом с грузовыми усилениями,
// а боевые сколы открывают отдельные нерегулярные участки сплава.
// Чистые профили сюда не попадают. Крупные спокойные поля остаются целыми.
float classic_paint_loss() {
    float weather = smoothstep(0.45, 0.85, u_wear);
    vec3 p = v_local * vec3(32.0, 19.0, 39.0) + vec3(u_seed * 3.7);
    float breakup = value_noise(p);
    float coarse = value_noise(p * 0.31 + vec3(7.3, 2.8, 4.1));
    float footprint = max(length(dFdx(p)), length(dFdy(p)));
    float resolved = 1.0 - smoothstep(0.25, 0.65, footprint);
    float aa = max(fwidth(breakup), 0.025);
    float broken = smoothstep(0.43 - aa, 0.66 + aa, breakup) * resolved;
    // Дальний mip имеет средний физический износ, а не мерцающие острова.
    broken += (1.0 - resolved) * 0.30;
    float loading_edge = max(classic_metal_span(0.055) - classic_metal_mask(), 0.0);
    float rub = loading_edge * broken * u_damage_profile.y;
    float wounds = smoothstep(0.62 - aa, 0.80 + aa, breakup) *
                   smoothstep(0.42, 0.64, coarse) * resolved;
    wounds *= max(u_damage_profile.x - 0.48, 0.0);
    return clamp(weather * (rub * 1.45 + wounds * 1.85), 0.0, 1.0);
}

// RG общей карты содержит широкую фаску конструктивных швов, BA —
// вмятины с валиками. Это dH/dUV до mip-фильтрации, а не разности
// соседних экранных отсчётов. Глубина задаётся в единицах самого корпуса.
vec3 classic_relief(vec2 uv, vec2 old_height_xy, float damage, float equipment,
                    float metal) {
    vec4 slope = texture(u_finish_slope, uv);
    vec2 dx = dFdx(uv), dy = dFdy(uv);
    vec2 seams = vec2(dot(slope.rg, dx), dot(slope.rg, dy));
    vec2 dents = vec2(dot(slope.ba, dx), dot(slope.ba, dy));
    float scale = max(u_texture_scale, 0.5);
    float model_scale = length(u_model[0].xyz);
    float structural = 0.0140 * u_finish_response.z / scale;
    float dent_depth = 0.0260 * u_finish_response.w * u_damage_profile.z / scale;
    dent_depth *= 0.40 + 0.60 * u_wear;
    // Тонкий родной G сохраняет прежнюю мировую глубину: её усиление
    // давало скачки блика на 1–2 texel. Весь крупный рельеф создают
    // отдельные широкие фаски и заломы с глубиной в единицах корпуса.
    vec2 height_xy = old_height_xy * (0.0028 / (scale * model_scale))
                  + seams * structural + dents * dent_depth;

    // Редкие направленные риски: фаза привязана к UV, амплитуда уходит
    // до Найквиста. Производные берутся в согласованном потоке исполнения.
    float islands = value_noise(vec3(uv * vec2(19.0, 11.0), u_seed + 4.7));
    float q = uv.y * 427.0 + islands * 1.7;
    float resolved = 1.0 - smoothstep(0.20, 0.45, fwidth(q));
    float phase = q * (2.0 * PI);
    float scratch = cos(phase) * resolved;
    float patch = smoothstep(0.48, 0.73, islands) *
                  smoothstep(0.18, 0.55, damage) * smoothstep(0.64, 0.96, equipment);
    float fine_depth = 0.000026 * u_damage_profile.w * u_finish_response.w *
                       (0.35 + u_wear) * patch / scale;
    height_xy += fine_depth * (2.0 * PI) * scratch * vec2(dFdx(q), dFdy(q));
    // Шлифованная силовая секция имеет рельеф, но не шероховатость краски.
    height_xy *= mix(1.0, 0.55, metal);
    return relief_from_gradient(normalize(v_normal), height_xy, model_scale);
}

float frosted_box(vec2 p, vec2 half_size, float scatter) {
    float distance = max(abs(p.x) - half_size.x, abs(p.y) - half_size.y);
    float blur = scatter + fwidth(distance);
    return 1.0 - smoothstep(-blur, blur, distance);
}

// Луч преломляется в толщу фонаря и заканчивается на глухой кабине,
// лежащей над исходной обшивкой. Фон и дальняя сторона корпуса сюда
// никогда не подмешиваются. Мягкие силуэты кресел и пульта размыты стеклом.
vec3 cobra_cabin(vec3 view_direction, float nv) {
    vec3 local_view = normalize(transpose(mat3(u_model)) * view_direction);
    vec3 local_normal = normalize(v_local_normal);
    vec3 ray = refract(-local_view, local_normal, 1.0 / 1.47);
    vec3 floor_normal = normalize(vec3(0.0, 1.0, 0.5));
    float depth = max(dot(v_local, floor_normal) - 0.157, 0.0);
    float travel = depth / max(-dot(ray, floor_normal), 0.22);
    vec2 cabin = (v_local + ray * min(travel, 0.12)).xz;
    vec3 interior = vec3(0.015, 0.022, 0.025);
    float seats = frosted_box(vec2(abs(cabin.x) - 0.037, cabin.y - 0.181),
                              vec2(0.022, 0.027), 0.007);
    float headrests = frosted_box(vec2(abs(cabin.x) - 0.037, cabin.y - 0.162),
                                  vec2(0.016, 0.010), 0.006);
    interior = mix(interior, vec3(0.058, 0.068, 0.067), seats * 0.82);
    interior = mix(interior, vec3(0.085, 0.091, 0.083), headrests * 0.65);
    float console = frosted_box(cabin - vec2(0.0, 0.203), vec2(0.009, 0.031), 0.005);
    interior = mix(interior, vec3(0.010, 0.016, 0.019), console * 0.85);
    float dashboard = frosted_box(cabin - vec2(0.0, 0.237), vec2(0.112, 0.009), 0.006);
    interior = mix(interior, vec3(0.007, 0.012, 0.015), dashboard * 0.92);
    float instruments = frosted_box(vec2(abs(cabin.x) - 0.039, cabin.y - 0.231),
                                    vec2(0.022, 0.002), 0.004);
    interior += vec3(0.035, 0.090, 0.082) * instruments;
    // Более длинный путь на скользящем ракурсе сильнее поглощает свет.
    vec3 transmission = exp(-mix(vec3(0.20), vec3(0.85), 1.0 - COBRA_GLASS_TINT) /
                            max(nv, 0.28));
    return interior * transmission;
}

Surface procedural_surface() {
    vec3 n = normalize(v_normal);
    vec2 uv = v_uv * u_texture_scale + u_uv_offset;
    vec4 surface;
    vec2 height_xy;
    if (u_imported) {
        // Карты атласа лежат в координатах корпуса, независимо от авторских UV-островов.
        vec3 w = pow(abs(normalize(v_local_normal)), vec3(5.0));
        w /= max(dot(w, vec3(1.0)), 0.0001);
        vec3 p = v_local * (0.62 * u_texture_scale);
        surface = w.x * texture(u_surface, p.zy + u_uv_offset)
                + w.y * texture(u_surface, p.xz + u_uv_offset)
                + w.z * texture(u_surface, p.xy + u_uv_offset);
        vec2 sx = texture(u_surface_slope, p.zy + u_uv_offset).rg;
        vec2 sy = texture(u_surface_slope, p.xz + u_uv_offset).rg;
        vec2 sz = texture(u_surface_slope, p.xy + u_uv_offset).rg;
        // Каждая проекция возвращает dH по своим двум осям. Сначала общий
        // локальный градиент, затем его проекция на экран и мировой базис.
        vec3 local_slope = w.x * vec3(0.0, sx.y, sx.x)
                         + w.y * vec3(sy.x, 0.0, sy.y)
                         + w.z * vec3(sz.x, sz.y, 0.0);
        height_xy = vec2(dot(local_slope, dFdx(p)), dot(local_slope, dFdy(p)));
    } else {
        surface = texture(u_surface, uv);
        vec2 slope = texture(u_surface_slope, uv).rg;
        height_xy = vec2(dot(slope, dFdx(uv)), dot(slope, dFdy(uv)));
    }
    float panel_tone = surface.r;
    float panel_height = surface.g;
    float damage = surface.b;
    float equipment = surface.a;
    float wear = smoothstep(0.98 - u_wear * 0.71, 1.0 - u_wear * 0.16, damage);
    float recess = smoothstep(0.18, 0.62, panel_height);
    if (u_classic_finishes && (v_material == 0 || v_material == 1 || v_material == 4)) {
        float edge = 1.0 - smoothstep(0.42, 0.63, panel_height);
        float chips = smoothstep(0.48 - u_wear * 0.19, 0.78 - u_wear * 0.19, damage);
        float abrasion = smoothstep(0.24, 0.58, damage) * (0.35 + 0.65 * (1.0 - equipment));
        wear = clamp(wear * (0.50 + 0.60 * u_damage_profile.x) +
                     u_wear * (chips * u_damage_profile.x * 0.55 +
                               edge * chips * 0.35 + abrasion * u_damage_profile.y * 0.58), 0.0, 1.0);
        wear = max(wear, classic_paint_loss());
    }

    vec3 paint = mix(u_hull_color, u_secondary_color,
                     smoothstep(0.53, 0.73, panel_tone) * 0.61);
    vec3 albedo = paint * mix(0.86, 1.14, panel_tone);
    albedo = mix(albedo, u_metal_color * mix(0.68, 1.05, damage), wear * 0.82);
    albedo *= mix(0.68, 1.0, recess);
    albedo *= mix(0.46, 1.0, equipment);
    albedo *= 1.0 - u_wear * smoothstep(0.28, 0.7, damage) * 0.21 * (1.0 - wear);
    float roughness = clamp(u_roughness + (damage - 0.5) * 0.18 +
                           (panel_tone - 0.5) * 0.25 + wear * 0.14, 0.2, 0.92);
    float metallic = mix(u_metallic, 0.78, wear);
    float occlusion = mix(0.55, 1.0, recess) * mix(0.78, 1.0, equipment);
    vec3 emission = vec3(0.0);

    // Трафаретная ливрея лежит в координатах корабля, а не повторяется на каждой панели.
    float primary = 0.0;
    float fine = 0.0;
    if (u_marking_style < 0.5) {
        primary = band(v_local.z, 0.20, 0.039);
        fine = band(v_local.z, 0.28, 0.004);
    } else if (u_marking_style < 1.5) {
        primary = band(abs(v_local.x), 0.36, 0.038);
        fine = band(abs(v_local.x), 0.435, 0.005);
        primary *= 1.0 - smoothstep(0.46, 0.75, v_local.z);
        fine *= 1.0 - smoothstep(0.46, 0.75, v_local.z);
    } else if (u_marking_style < 2.5) {
        primary = band(v_local.y + v_local.z * 0.18, 0.02, 0.055);
        fine = band(v_local.y + v_local.z * 0.18, 0.12, 0.005);
    } else {
        primary = band(v_local.z + v_local.x * 0.26, -0.18, 0.033);
        fine = band(v_local.z + v_local.x * 0.26, -0.11, 0.004);
    }
    float paint_mask = (primary * 0.76 + fine * 0.54) * (1.0 - wear * 0.88) * recess;
    albedo = mix(albedo, u_accent_color * mix(0.77, 1.12, panel_tone), paint_mask);
    float registration_ink = 0.0;
    vec3 registration_pigment = vec3(0.0);
    bool registration_upper = u_classic_finishes ? v_local_normal.y > 0.25 : v_normal.y > 0.25;
    if (u_patch.z > 0.0 && v_material == 0 && registration_upper) {
        vec2 stamp = 0.5 + (u_patch.yx - v_local.zx) / (u_patch.z * vec2(1.0, 0.50));
        vec2 atlas_uv = (u_stencil_tile + clamp(stamp, 0.01, 0.99)) / vec2(4.0, 8.0);
        float ink = texture(u_stencils, atlas_uv).r * recess;
        float light_paint = smoothstep(0.16, 0.38, dot(u_hull_color, vec3(.22, .70, .08)));
        vec3 pigment = mix(vec3(.64, .57, .42), vec3(.030, .052, .065), light_paint);
        if (u_classic_finishes) {
            registration_ink = ink * (1.0 - wear * 0.35) * 0.87;
            registration_pigment = pigment;
        } else {
            // У внешних процедурных корпусов прежний порядок и износ.
            albedo = mix(albedo, pigment, ink * (1.0 - wear * 0.85) * 0.87);
        }
    }
    n = relief_from_gradient(n, height_xy, 0.0028 / max(u_texture_scale, 0.5));

    if (v_material == 1) {
        albedo = mix(vec3(0.035, 0.044, 0.052), u_metal_color * 0.35, wear);
        albedo *= mix(0.5, 1.0, equipment);
        roughness = 0.52;
        metallic = 0.78;
    } else if (v_material == 2) {
        n = normalize(v_normal);
        albedo = vec3(0.012, 0.035, 0.054) + u_accent_color * 0.018;
        roughness = 0.115;
        metallic = 0.3;
        occlusion = 1.0;
    } else if (v_material == 3) {
        n = normalize(v_normal);
        albedo = vec3(0.03, 0.045, 0.06);
        // У каждой апертуры свои UV: утопленный обод, рёбра решётки и горячая середина.
        vec2 p = v_uv * 2.0 - 1.0;
        float aperture = 1.0 - smoothstep(0.80, 1.0, max(abs(p.x), abs(p.y)));
        float ridge = abs(fract(v_uv.x * 5.0) - 0.5);
        float aa = clamp(fwidth(v_uv.x) * 5.0, 0.005, 0.18);
        float grille = smoothstep(0.055 - aa, 0.10 + aa, ridge);
        float core = exp(-2.5 * p.y * p.y) * grille;
        emission = (u_engine_color + vec3(0.15 * core * core)) * aperture * (0.18 + 1.9 * core);
        roughness = 0.24;
        metallic = 0.35;
    } else if (v_material == 4) {
        albedo = u_accent_color * mix(0.66, 1.03, panel_tone);
        albedo = mix(albedo, u_metal_color, wear * 0.75);
        roughness = 0.45;
        metallic = 0.27;
    } else if (v_material == 5) {
        vec3 p = v_local * 9.0 + vec3(u_seed);
        float stone = value_noise(p) * 0.50 + value_noise(p * 2.07) * 0.28 +
                      value_noise(p * 4.1) * 0.15 + value_noise(p * 8.3) * 0.07;
        float veins = 1.0 - smoothstep(0.018, 0.070, abs(value_noise(p * 0.45) - 0.47));
        albedo = mix(u_secondary_color, u_hull_color, smoothstep(0.18, 0.85, stone));
        albedo *= (0.82 + stone * 0.45) * (1.0 - veins * 0.22);
        n = relief_normal(normalize(v_normal), stone, 0.008);
        roughness = 0.93;
        metallic = 0.035;
        occlusion = 0.94 - veins * 0.07;
    } else if (v_material == 6) {
        n = normalize(v_normal);
        albedo = u_window_color * 0.18;
        emission = u_window_color * 1.1;
        roughness = 0.25;
        metallic = 0.1;
    }

    if (u_classic_finishes) {
        if (v_material == 0) {
            float metal = classic_metal_mask();
            // Пигмент — шероховатый диэлектрик. Царапины прежней карты
            // сохраняют и расположение, и металлическое основание.
            roughness = clamp(u_roughness + (damage - 0.5) * 0.14 +
                              (panel_tone - 0.5) * 0.12 + wear * 0.06, 0.45, 0.94);
            roughness = mix(roughness, 0.52 + damage * 0.16, wear * 0.84);
            metallic = mix(0.025, 0.80, wear);
            vec3 alloy = u_metal_color * mix(0.88, 1.32, panel_tone);
            alloy *= mix(0.68, 1.0, recess) * mix(0.55, 1.0, equipment);
            albedo = mix(albedo, alloy, metal);
            roughness = mix(roughness, clamp(u_finish_response.x + damage * 0.10 + wear * 0.05, 0.19, 0.70), metal);
            metallic = mix(metallic, 0.97, metal);
            n = classic_relief(uv, height_xy, damage, equipment, metal);
        } else if (v_material == 1 || v_material == 4) {
            albedo = u_metal_color * mix(0.52, 1.06, panel_tone);
            albedo *= mix(0.64, 1.0, equipment);
            roughness = clamp(u_finish_response.x + damage * 0.12 + wear * 0.05, 0.19, 0.70);
            metallic = 0.96;
            n = classic_relief(uv, height_xy, damage, equipment, 1.0);
        } else if (v_material == 2) {
            // Матовый фонарь: широкий диэлектрический блик и рассеяние,
            // без металлического F0 и без прозрачности всего корабля.
            n = normalize(v_normal);
            albedo = (u_cobra_finishes ? COBRA_GLASS_TINT : u_glass_tint) * 0.018;
            roughness = u_finish_response.y;
            metallic = 0.0;
            occlusion = 1.0;
        } else if (v_material == 6) {
            // Сенсор/освещённое окно: гладкое защитное стекло над прежней
            // эмиссией. Наклоны швов и вмятины на него не переносятся.
            n = normalize(v_normal);
            albedo = mix(albedo, u_glass_tint * 0.035, 0.75);
            roughness = u_finish_response.y;
            metallic = 0.0;
        }
        // Бортовой номер — слой матовой трафаретной краски поверх и
        // окрашенного листа, и оголённого усиления. Он вращается с бортом.
        albedo = mix(albedo, registration_pigment, registration_ink);
        roughness = mix(roughness, clamp(u_roughness, 0.60, 0.90), registration_ink);
        metallic = mix(metallic, 0.025, registration_ink);
    }

    return Surface(n, albedo, roughness, metallic, occlusion, emission);
}

Surface author_surface() {
    vec3 n = normalize(v_normal);
    vec3 p_x = dFdx(v_position), p_y = dFdy(v_position);
    vec2 uv_x = dFdx(v_uv), uv_y = dFdy(v_uv);
    float det = uv_x.x * uv_y.y - uv_x.y * uv_y.x;
    vec3 tangent = (p_x * uv_y.y - p_y * uv_x.y) * sign(det);
    tangent -= n * dot(n, tangent);
    if (abs(det) > 1e-12 && dot(tangent, tangent) > 1e-18) {
        tangent = normalize(tangent);
        vec3 bitangent = cross(n, tangent) * (u_cross_bitangent ? 1.0 : sign(det));
        vec3 mapped = texture(u_normal_map, v_uv).xyz * 2.0 - 1.0;
        n = normalize(mat3(tangent, bitangent, n) * mapped);
    }
    vec3 albedo = pow(texture(u_diffuse_map, v_uv).rgb, vec3(2.2));
    float gloss = clamp(texture(u_gloss_map, v_uv).r * u_gloss, 0.0, 1.0);
    vec3 glow = pow(texture(u_emission_map, v_uv).rgb, vec3(2.2)) * u_glow_gain;
    return Surface(n, albedo, mix(.85, .19, gloss), .18, 1.0, glow);
}

void main() {
    Surface surface;
    if (u_author_maps) surface = author_surface();
    else surface = procedural_surface();
    vec3 n = surface.normal;
    vec3 albedo = surface.albedo;
    float roughness = surface.roughness;
    float metallic = surface.metallic;
    float occlusion = surface.occlusion;
    vec3 view_direction = normalize(u_eye - v_position);
    vec3 light_direction = u_sun_direction;
    vec3 halfway = normalize(view_direction + light_direction);
    float nl = max(dot(n, light_direction), 0.0);
    float nv = max(dot(n, view_direction), 0.001);
    float nh = max(dot(n, halfway), 0.0);
    float hv = max(dot(halfway, view_direction), 0.0);
    vec3 f0 = mix(vec3(0.045), albedo, metallic);
    if (u_author_maps) {
        vec3 spec = texture(u_specular_map, v_uv).rgb * u_specular_tint;
        f0 = clamp(vec3(.025) + spec * .68, .025, .90);
    }
    bool classic_surface = u_classic_finishes &&
                           (v_material == 0 || v_material == 1 || v_material == 2 ||
                            v_material == 4 || v_material == 6);
    bool classic_glass = u_classic_finishes && (v_material == 2 || v_material == 6);
    bool cobra_glass = u_cobra_finishes && v_material == 2;
    if (classic_glass) f0 = vec3(0.040);
    else if (v_material == 2) f0 = vec3(0.075, 0.085, 0.095);
    vec3 fresnel = fresnel_schlick(hv, f0);
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = nh * nh * (alpha2 - 1.0) + 1.0;
    float distribution = alpha2 / max(PI * denom * denom, 0.00001);
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float geometry_v = nv / (nv * (1.0 - k) + k);
    float geometry_l = nl / (nl * (1.0 - k) + k);
    vec3 specular = distribution * geometry_v * geometry_l * fresnel / max(4.0 * nv * nl, 0.001);
    vec3 diffuse = (1.0 - fresnel) * (1.0 - metallic * 0.74) * albedo / PI;
    if (classic_surface) {
        diffuse = (1.0 - fresnel) * (1.0 - metallic) * albedo / PI;
        if (classic_glass) diffuse *= 0.18;
    }
    float visibility = 1.0;
    if (u_shadow_enabled && all(greaterThan(v_shadow, vec3(0.0))) &&
                            all(lessThan(v_shadow, vec3(1.0)))) {
        float bias = max(0.00006, 0.00035 * (1.0 - dot(normalize(v_normal), light_direction)));
        vec2 texel = 1.0 / vec2(textureSize(u_shadow, 0));
        visibility = 0.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                visibility += texture(u_shadow, vec3(v_shadow.xy + vec2(x, y) * texel * 0.85,
                                                      v_shadow.z - bias));
            }
        }
        visibility /= 9.0;
    }
    vec3 color = (diffuse + specular) * vec3(3.25, 3.02, 2.69) * nl * visibility;

    // Холодное рассеянное небо поддерживает форму даже на полностью теневой стороне.
    float up = n.y * 0.5 + 0.5;
    vec3 ambient = mix(vec3(0.080, 0.102, 0.153), vec3(0.215, 0.254, 0.318), up);
    color += albedo * ambient * occlusion * (1.0 - metallic * 0.46);
    vec3 reflection = reflect(-view_direction, n);
    vec3 reflected_sky = textureLod(u_sky, reflection, roughness * 7.0).rgb;
    vec3 edge_fresnel = fresnel_schlick(nv, f0);
    color += (reflected_sky * 0.30 + vec3(0.028, 0.045, 0.073)) * edge_fresnel * occlusion;
    if (u_author_maps) {
        // Та же холодная полусфера, что у рассеянного света, в отражении металла.
        vec3 reflected_fill = mix(vec3(.065, .078, .10), vec3(.18, .23, .29), reflection.y * .5 + .5);
        color += reflected_fill * edge_fresnel;
    }
    if (classic_surface) {
        // Та же широкая холодная полусфера, что освещает обшивку. Металл
        // отражает её цвет, тогда как матовая краска в основном рассеивает.
        vec3 reflected_fill = mix(vec3(0.065, 0.078, 0.10), vec3(0.18, 0.23, 0.29),
                                  reflection.y * 0.5 + 0.5);
        color += reflected_fill * edge_fresnel * occlusion;
    }
    if (v_material == 2 || (classic_glass && v_material == 6)) {
        if (cobra_glass) {
            color += cobra_cabin(view_direction, nv) * (1.0 - edge_fresnel);
            color += COBRA_GLASS_TINT * vec3(0.022, 0.026, 0.030) *
                     (0.40 + 0.60 * nl * visibility) * (1.0 - edge_fresnel);
        } else if (classic_glass) {
            // Чужой кораблю интерьер не рисуется: рассеянный свет
            // заканчивается на тёмной подложке именно его стекла/сенсора.
            float transmission = 0.30 + 0.70 * nv;
            color += u_glass_tint * 0.11 * transmission *
                     (0.40 + 0.60 * nl * visibility) * (1.0 - edge_fresnel);
        } else {
            color += textureLod(u_sky, reflection, 1.1).rgb * edge_fresnel * 0.5;
            color += vec3(0.035, 0.095, 0.15) * pow(1.0 - nv, 3.0);
        }
    }
    color += surface.emission;
    frag_color = vec4(display_color(color), 1.0);
}
