#version 330

uniform samplerCube u_sky;
uniform mat4 u_inv_view_proj;
uniform vec3 u_eye;
in vec2 v_clip;
out vec4 frag_color;

vec3 display_color(vec3 color) {
    color = (color * (2.51 * color + 0.03)) /
            (color * (2.43 * color + 0.59) + 0.14);
    return pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
}

void main() {
    vec4 far_point = u_inv_view_proj * vec4(v_clip, 1.0, 1.0);
    vec3 direction = normalize(far_point.xyz / far_point.w - u_eye);
    frag_color = vec4(display_color(textureLod(u_sky, direction, 0.0).rgb), 1.0);
}
