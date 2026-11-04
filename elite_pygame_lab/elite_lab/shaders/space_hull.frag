#version 330

uniform sampler2D u_surface;
uniform sampler2D u_stencils;
uniform bool u_author_maps;
uniform bool u_imported;
uniform bool u_cross_bitangent;
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

// Производные восстанавливают касательную плоскость без изменения крупных фасетов.
vec3 relief_normal(vec3 normal, float height, float strength) {
    vec3 dx = dFdx(v_position);
    vec3 dy = dFdy(v_position);
    vec3 r1 = cross(dy, normal);
    vec3 r2 = cross(normal, dx);
    float det = dot(dx, r1);
    vec3 gradient = sign(det) * (dFdx(height) * r1 + dFdy(height) * r2);
    return normalize(abs(det) * normal - strength * gradient);
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

Surface procedural_surface() {
    vec3 n = normalize(v_normal);
    vec2 uv = v_uv * u_texture_scale + u_uv_offset;
    vec4 surface = texture(u_surface, uv);
    if (u_imported) {
        // Карты атласа лежат в координатах корпуса, независимо от авторских UV-островов.
        vec3 w = pow(abs(normalize(v_local_normal)), vec3(5.0));
        w /= max(dot(w, vec3(1.0)), 0.0001);
        vec3 p = v_local * (0.62 * u_texture_scale);
        surface = w.x * texture(u_surface, p.zy + u_uv_offset)
                + w.y * texture(u_surface, p.xz + u_uv_offset)
                + w.z * texture(u_surface, p.xy + u_uv_offset);
    }
    float panel_tone = surface.r;
    float panel_height = surface.g;
    float damage = surface.b;
    float equipment = surface.a;
    float wear = smoothstep(0.98 - u_wear * 0.71, 1.0 - u_wear * 0.16, damage);
    float recess = smoothstep(0.18, 0.62, panel_height);

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
    if (u_patch.z > 0.0 && v_material == 0 && v_normal.y > 0.25) {
        vec2 stamp = 0.5 + (u_patch.yx - v_local.zx) / (u_patch.z * vec2(1.0, 0.50));
        vec2 atlas_uv = (u_stencil_tile + clamp(stamp, 0.01, 0.99)) / vec2(4.0, 8.0);
        float ink = texture(u_stencils, atlas_uv).r * recess * (1.0 - wear * 0.85);
        float light_paint = smoothstep(0.16, 0.38, dot(u_hull_color, vec3(.22, .70, .08)));
        vec3 pigment = mix(vec3(.64, .57, .42), vec3(.030, .052, .065), light_paint);
        albedo = mix(albedo, pigment, ink * 0.87);
    }
    n = relief_normal(n, panel_height, 0.0028 / max(u_texture_scale, 0.5));

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
    if (v_material == 2) f0 = vec3(0.075, 0.085, 0.095);
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
    if (v_material == 2) {
        color += textureLod(u_sky, reflection, 1.1).rgb * edge_fresnel * 0.5;
        color += vec3(0.035, 0.095, 0.15) * pow(1.0 - nv, 3.0);
    }
    color += surface.emission;
    frag_color = vec4(display_color(color), 1.0);
}
