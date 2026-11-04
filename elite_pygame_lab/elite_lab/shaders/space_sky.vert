#version 330

out vec2 v_clip;

void main() {
    vec2 position = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_clip = position * 2.0 - 1.0;
    gl_Position = vec4(v_clip, 1.0, 1.0);
}
