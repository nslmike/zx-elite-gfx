#version 330

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in float in_material;

uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform mat4 u_shadow_vp;
uniform bool u_imported;
uniform int u_part_type;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_local;
out vec3 v_local_normal;
out vec2 v_uv;
out vec3 v_shadow;
flat out int v_material;

void main() {
    vec4 position = u_model * vec4(in_position, 1.0);
    v_position = position.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_local = in_position;
    v_local_normal = in_normal;
    v_uv = in_uv;
    v_shadow = (u_shadow_vp * position).xyz * 0.5 + 0.5;
    v_material = u_imported ? u_part_type : int(in_material + 0.5);
    gl_Position = u_view_proj * position;
}
