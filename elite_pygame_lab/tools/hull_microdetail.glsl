// Изолированная проба. Вставляется скриптом после штатного relief_normal().
uniform float u_microdetail;

const float DETAIL_TAU = 6.28318530718;

// Амплитуда исчезает до Найквиста. Координаты принадлежат поверхности,
// производные измеряют реальный footprint, в том числе при скользящем ракурсе.
float detail_filter(float cycles) {
    return 1.0 - smoothstep(0.22, 0.48, fwidth(cycles));
}

vec3 detail_screen_gradient(vec3 normal, float hx, float hy) {
    vec3 dx = dFdx(v_position), dy = dFdy(v_position);
    vec3 r1 = cross(dy, normal), r2 = cross(normal, dx);
    float det = dot(dx, r1);
    return sign(det) * (hx * r1 + hy * r2) / max(abs(det), 1e-15);
}

vec3 detail_box(vec2 p, vec2 halfsize, float bevel) {
    vec2 edge = abs(p) - halfsize;
    float d = max(edge.x, edge.y);
    float aa = max(fwidth(d), 0.00003);
    float width = bevel + 2.0 * aa;
    float t = clamp((d + bevel + aa) / width, 0.0, 1.0);
    float mask = 1.0 - t * t * (3.0 - 2.0 * t);
    // Производная только по положению; footprint — фиксированная величина.
    // Вложенных dFdx(fwidth(...)) и других производных второго порядка нет.
    float slope = -6.0 * t * (1.0 - t) / width;
    vec2 direction = edge.x > edge.y ? vec2(sign(p.x), 0.0) : vec2(0.0, sign(p.y));
    return vec3(mask, slope * direction);
}

Surface hull_microdetail(Surface base) {
    // Эти условия uniform: выход безопасен до всех производных и избавляет
    // выключенную пробу и импортированные материалы от любых вычислений добавки.
    if (u_microdetail == 0.0 || u_imported || u_author_maps)
        return base;
    // Никакого воздействия на стекло, двигатели, окантовки и чужие карты.
    // Uniform меняет исключительно эту добавку, старый G-height уже применён.
    vec3 n = base.normal;
    vec2 p = v_local.xz;
    vec2 uv = v_uv * u_texture_scale + u_uv_offset;
    // Силовые металлические секции №9 используют тот же material 0,
    // что и краска. Общая сглаженная маска штатного shader оставляет B
    // только на пигменте, включая частичное покрытие на границе секции.
    float paint = u_cobra_finishes ? 1.0 - cobra_shoulder_metal() : 1.0;

    // Два симметричных сервисных участка на плечах Cobra. Без новой сетки
    // панелей: спокойная основная обшивка остаётся открытой.
    vec2 service = vec2(abs(p.x) - 0.590, p.y + 0.320);
    vec3 box = detail_box(service, vec2(0.033, 0.020), 0.0030);
    float patch = box.x;
    float upper = smoothstep(0.65, 0.85, v_local_normal.y);
    float slots = service.y / 0.008;
    float resolved = detail_filter(slots);
    float ribs = cos(DETAIL_TAU * slots) * resolved;
    float height = -0.00015 + 0.00008 * ribs;
    vec2 meso_slope = upper * (height * box.yz + patch *
        vec2(0.0, -0.00008 * DETAIL_TAU / 0.008 * sin(DETAIL_TAU * slots) * resolved));
    vec3 meso = detail_screen_gradient(n, dot(meso_slope, dFdx(service)),
                                          dot(meso_slope, dFdy(service)));

    // Длинная направленная микрошлифовка вместо изотропного «песка».
    // Пара несоизмеримых частот даёт неравномерный шаг резца. Фаза привязана
    // к исходным UV фасета, не к экрану; её низкая модуляция тоже объектная.
    float bend1 = 1.2 * value_noise(vec3(uv * vec2(89.0, 7.0), 3.73));
    float bend2 = 1.4 * value_noise(vec3(uv * vec2(127.0, 13.0), 7.29));
    float q1 = uv.y * 900.0 + bend1;
    float q2 = uv.y * 1471.0 + bend2;
    float f1 = detail_filter(q1), f2 = detail_filter(q2);
    float slope1 = cos(DETAIL_TAU * q1) * f1;
    float slope2 = cos(DETAIL_TAU * q2) * f2;
    // Родная карта управляет характером покрытия: часть листов спокойная,
    // встроенная механика не получает дополнительную шлифовку. Карта mipmapped.
    vec4 old_surface = texture(u_surface, uv);
    float finish = smoothstep(0.83, 0.99, old_surface.a) *
        mix(0.15, 1.0, smoothstep(0.53, 0.72, old_surface.r));
    float stroke = mix(0.30, 1.0, value_noise(vec3(uv * vec2(130.0, 47.0), 11.2)));
    float height_scale = 0.000027 * finish * stroke / max(u_texture_scale, 0.5);
    float hx = height_scale * DETAIL_TAU *
        (0.58 * slope1 * dFdx(q1) + 0.42 * slope2 * dFdx(q2));
    float hy = height_scale * DETAIL_TAU *
        (0.58 * slope1 * dFdy(q1) + 0.42 * slope2 * dFdy(q2));
    vec3 brush = detail_screen_gradient(n, hx, hy);

    // Работаем поверх штатной нормали; не сбрасываем её к v_normal.
    // Все производные уже вычислены в согласованном потоке исполнения.
    // Строгий bypass сохраняет даже исходную длину normal и roughness стекла.
    // Bypass после производных: соседние фрагменты на границе краски и
    // металла вычисляют dFdx/dFdy в согласованном потоке исполнения.
    if (v_material != 0 || paint <= 0.0)
        return base;
    base.normal = normalize(n - u_microdetail * paint * (meso + brush));
    float brushed = finish * stroke * (0.58 * sin(DETAIL_TAU * q1) * f1
                                    + 0.42 * sin(DETAIL_TAU * q2) * f2);
    base.roughness = clamp(base.roughness + u_microdetail * paint *
        (0.024 * brushed + upper * patch * (0.014 - 0.018 * ribs)), 0.2, 0.92);
    return base;
}
