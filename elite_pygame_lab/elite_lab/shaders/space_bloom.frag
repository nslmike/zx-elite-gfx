#version 330

#ifdef COMPOSITE_MSAA
uniform sampler2DMS frame;
#else
uniform sampler2D frame;
#endif
in vec2 uv;
out vec4 color;

#ifdef DOWNSAMPLE
void main() {
    ivec2 origin = 4 * ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(frame, 0);
    vec3 radiance = vec3(0.0);
    const vec3 luminance = vec3(0.2126, 0.7152, 0.0722);
    // Усредняем все пиксели блока, в том числе одиночный далёкий огонь.
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            ivec2 p = origin + ivec2(x, y);
            if (all(lessThan(p, size))) {
                vec4 s = texelFetch(frame, p, 0);
                // Цвет берём из видимого источника, яркость из его отдельной
                // линейной маски, чтобы после MSAA не возводить покрытие в квадрат.
                radiance += s.rgb * (s.a / max(dot(s.rgb, luminance), 0.00001));
            }
        }
    }
    color = vec4(radiance / 16.0, 1.0);
}
#endif

#ifdef BLUR
uniform vec2 step_uv;

void main() {
    // Симметричный гауссов фильтр: соседние пары объединены билинейно.
    vec3 radiance = texture(frame, uv).rgb * 0.227027;
    radiance += (texture(frame, uv + 1.384615 * step_uv).rgb +
                 texture(frame, uv - 1.384615 * step_uv).rgb) * 0.316216;
    radiance += (texture(frame, uv + 3.230769 * step_uv).rgb +
                 texture(frame, uv - 3.230769 * step_uv).rgb) * 0.070270;
    color = vec4(radiance, 1.0);
}
#endif

#if defined(COMPOSITE) || defined(COMPOSITE_MSAA)
uniform sampler2D bloom;
uniform vec2 bloom_uv_scale;

vec3 display_color(vec3 hdr) {
    vec3 mapped = (hdr * (2.51 * hdr + 0.03)) /
                  (hdr * (2.43 * hdr + 0.59) + 0.14);
    return pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
}
#endif

#ifdef COMPOSITE
void main() {
    vec3 glow = texture(bloom, uv * bloom_uv_scale).rgb * 0.15;
    color = vec4(display_color(texture(frame, uv).rgb + glow), 1.0);
}
#endif

#ifdef COMPOSITE_MSAA
uniform int samples;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 glow = texture(bloom, uv * bloom_uv_scale).rgb * 0.15;
    // Как в исходном RGBA8 MSAA, усредняем уже отображаемые сэмплы.
    // В частности, яркая звезда за краем корпуса не меняет его сглаживание.
    vec3 first = texelFetch(frame, p, 0).rgb;
    vec3 encoded = display_color(first + glow);
    vec3 sum = roundEven(encoded * 255.0);
    for (int sample_index = 1; sample_index < samples; ++sample_index) {
        vec3 value = texelFetch(frame, p, sample_index).rgb;
        vec3 mapped = all(equal(value, first)) ? encoded : display_color(value + glow);
        sum += roundEven(mapped * 255.0);
    }
    color = vec4(sum / (255.0 * float(samples)), 1.0);
}
#endif
