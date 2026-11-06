#version 330

uniform samplerCube u_sky;
uniform mat4 u_inv_view_proj;
uniform vec3 u_eye;
uniform bool u_linear_output;
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
    vec3 sky = textureLod(u_sky, direction, 0.0).rgb;
    frag_color = u_linear_output ? vec4(sky, 0.0) : vec4(display_color(sky), 1.0);
}
