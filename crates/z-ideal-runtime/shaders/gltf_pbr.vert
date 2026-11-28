#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv0;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec4 in_tangent;
layout(location = 4) in vec2 in_uv1;

layout(location = 0) out vec3 world_position;
layout(location = 1) out vec3 world_normal;
layout(location = 2) out vec4 world_tangent;
layout(location = 3) out vec2 uv0;
layout(location = 4) out vec2 uv1;

layout(push_constant) uniform TransformData {
    mat4 mvp;
    mat4 model;
} transform_data;

void main() {
    vec4 world = transform_data.model * vec4(in_position, 1.0);
    mat3 normal_matrix = transpose(inverse(mat3(transform_data.model)));
    world_position = world.xyz;
    world_normal = normalize(normal_matrix * in_normal);
    float model_handedness = determinant(mat3(transform_data.model)) < 0.0 ? -1.0 : 1.0;
    world_tangent = vec4(
        normalize(mat3(transform_data.model) * in_tangent.xyz),
        in_tangent.w * model_handedness
    );
    uv0 = in_uv0;
    uv1 = in_uv1;
    gl_Position = transform_data.mvp * vec4(in_position, 1.0);
}
