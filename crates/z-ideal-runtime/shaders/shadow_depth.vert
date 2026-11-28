#version 450
layout(location = 0) in vec3 in_position;
layout(push_constant) uniform ShadowTransform { mat4 light_view_projection_model; } shadow_transform;
void main() { gl_Position = shadow_transform.light_view_projection_model * vec4(in_position, 1.0); }
