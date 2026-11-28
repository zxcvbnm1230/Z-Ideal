#version 450

layout(location = 0) in vec3 world_position;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec4 world_tangent;
layout(location = 3) in vec2 uv0;
layout(location = 4) in vec2 uv1;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0, std140) uniform MaterialData {
    vec4 base_color_factor;
    vec4 emissive_alpha_cutoff;
    vec4 metallic_roughness_normal_occlusion;
    uvec4 material_flags;
    uvec4 texture_coordinates_0;
    uvec4 texture_coordinates_1;
} material;

layout(set = 0, binding = 1) uniform sampler2D base_color_texture;
layout(set = 0, binding = 2) uniform sampler2D metallic_roughness_texture;
layout(set = 0, binding = 3) uniform sampler2D normal_texture;
layout(set = 0, binding = 4) uniform sampler2D occlusion_texture;
layout(set = 0, binding = 5) uniform sampler2D emissive_texture;

struct LocalLight {
    vec4 position_range;
    vec4 color_intensity;
    vec4 direction_outer_cos;
    vec4 type_inner_cos;
};

layout(set = 1, binding = 0, std140) uniform SceneData {
    vec4 camera_position;
    vec4 light_direction_intensity;
    vec4 light_color;
    vec4 ambient_color;
    vec4 environment_intensity_max_lod;
    uvec4 local_light_count;
    LocalLight local_lights[4];
} scene;

layout(set = 1, binding = 1) uniform samplerCube diffuse_irradiance_map;
layout(set = 1, binding = 2) uniform samplerCube prefiltered_specular_map;

vec2 material_uv(uint set_index) {
    return set_index == 1u ? uv1 : uv0;
}

vec3 mapped_normal(vec2 coordinates) {
    vec3 n = normalize(world_normal);
    vec3 sampled = texture(normal_texture, coordinates).xyz * 2.0 - 1.0;
    sampled.xy *= material.metallic_roughness_normal_occlusion.z;
    vec3 t;
    vec3 b;
    if (material.texture_coordinates_0.z == 0u) {
        t = normalize(world_tangent.xyz - n * dot(n, world_tangent.xyz));
        b = normalize(cross(n, t)) * world_tangent.w;
    } else {
        vec3 dp1 = dFdx(world_position);
        vec3 dp2 = dFdy(world_position);
        vec2 duv1 = dFdx(coordinates);
        vec2 duv2 = dFdy(coordinates);
        float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
        if (abs(determinant) <= 1e-8) {
            return n;
        }
        float inverse_determinant = 1.0 / determinant;
        t = normalize((dp1 * duv2.y - dp2 * duv1.y) * inverse_determinant);
        b = normalize((-dp1 * duv2.x + dp2 * duv1.x) * inverse_determinant);
    }
    return normalize(mat3(t, b, n) * sampled);
}

float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(n, h), 0.0);
    float denominator = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denominator * denominator, 1e-6);
}

float geometry_schlick_ggx(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6);
}

vec3 fresnel_schlick(float cosine, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 fresnel_schlick_roughness(float cosine, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0)
         * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

vec3 evaluate_direct_brdf(vec3 n, vec3 v, vec3 l, vec3 base_color,
                          vec3 f0, float metallic, float roughness,
                          vec3 radiance) {
    float n_dot_l = max(dot(n, l), 0.0);
    float n_dot_v = max(dot(n, v), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3(0.0);
    }
    vec3 h = normalize(v + l);
    vec3 f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    float d = distribution_ggx(n, h, roughness);
    float g = geometry_schlick_ggx(n_dot_v, roughness)
            * geometry_schlick_ggx(n_dot_l, roughness);
    vec3 specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-5);
    vec3 diffuse_weight = (1.0 - f) * (1.0 - metallic);
    vec3 diffuse = diffuse_weight * base_color / 3.14159265;
    return (diffuse + specular) * radiance * n_dot_l;
}

void main() {
    vec4 base = texture(base_color_texture, material_uv(material.texture_coordinates_0.x))
              * material.base_color_factor;
    if (material.material_flags.x == 1u && base.a < material.emissive_alpha_cutoff.w) {
        discard;
    }

    vec4 mr = texture(metallic_roughness_texture,
                      material_uv(material.texture_coordinates_0.y));
    float metallic = clamp(mr.b * material.metallic_roughness_normal_occlusion.x, 0.0, 1.0);
    float roughness = clamp(mr.g * material.metallic_roughness_normal_occlusion.y, 0.04, 1.0);
    vec3 n = mapped_normal(material_uv(material.texture_coordinates_0.z));
    if (material.material_flags.y != 0u && !gl_FrontFacing) {
        n = -n;
    }
    vec3 v = normalize(scene.camera_position.xyz - world_position);
    vec3 f0 = mix(vec3(0.04), base.rgb, metallic);
    vec3 directional_l = normalize(-scene.light_direction_intensity.xyz);
    vec3 direct = evaluate_direct_brdf(
        n, v, directional_l, base.rgb, f0, metallic, roughness,
        scene.light_color.rgb * scene.light_direction_intensity.w);

    // The fixed-capacity array preserves caller order and bounds fragment work.
    for (uint index = 0u; index < min(scene.local_light_count.x, 4u); ++index) {
        LocalLight light = scene.local_lights[index];
        vec3 to_light = light.position_range.xyz - world_position;
        float distance_to_light = length(to_light);
        float range = light.position_range.w;
        if (distance_to_light > 1e-4 && distance_to_light < range) {
            vec3 local_l = to_light / distance_to_light;
            float normalized_distance = distance_to_light / range;
            float range_fade = pow(max(1.0 - pow(normalized_distance, 4.0), 0.0), 2.0);
            float attenuation = range_fade / max(distance_to_light * distance_to_light, 0.01);
            float spot_factor = 1.0;
            if (light.type_inner_cos.x > 0.5) {
                float cone_cosine = dot(normalize(light.direction_outer_cos.xyz), -local_l);
                spot_factor = smoothstep(light.direction_outer_cos.w,
                                         light.type_inner_cos.y, cone_cosine);
            }
            vec3 radiance = light.color_intensity.rgb
                          * light.color_intensity.w * attenuation * spot_factor;
            direct += evaluate_direct_brdf(
                n, v, local_l, base.rgb, f0, metallic, roughness, radiance);
        }
    }
    float n_dot_v = max(dot(n, v), 0.0);
    vec3 environment_fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    vec3 environment_diffuse_weight = (1.0 - environment_fresnel) * (1.0 - metallic);

    float occlusion_sample = texture(occlusion_texture,
                                     material_uv(material.texture_coordinates_0.w)).r;
    float occlusion = mix(1.0, occlusion_sample,
                          material.metallic_roughness_normal_occlusion.w);
    vec3 ambient = scene.ambient_color.rgb * base.rgb * environment_diffuse_weight * occlusion;
    float environment_intensity = scene.environment_intensity_max_lod.x;
    float max_specular_lod = scene.environment_intensity_max_lod.y;
    vec3 irradiance = texture(diffuse_irradiance_map, n).rgb;
    vec3 diffuse_ibl = irradiance * base.rgb * environment_diffuse_weight
                     / 3.14159265;
    vec3 reflection = reflect(-v, n);
    vec3 prefiltered = textureLod(prefiltered_specular_map, reflection,
                                  roughness * max_specular_lod).rgb;
    // Analytic split-sum approximation; a BRDF integration LUT is not required.
    float split_sum_a = 1.0 - 0.5 * roughness;
    float split_sum_b = 0.04 * roughness;
    vec3 specular_ibl = prefiltered
                      * (f0 * split_sum_a + vec3(split_sum_b));
    vec3 environment = (diffuse_ibl + specular_ibl) * environment_intensity * occlusion;
    vec3 emissive = texture(emissive_texture,
                            material_uv(material.texture_coordinates_1.x)).rgb
                  * material.emissive_alpha_cutoff.rgb;
    out_color = vec4(direct + ambient + environment + emissive, base.a);
}
