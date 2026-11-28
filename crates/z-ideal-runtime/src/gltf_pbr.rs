//! Compiled glTF metallic-roughness PBR shader sources.
use z_ideal_renderer::{
    CompiledShader, ShaderCompileConfig, ShaderStage, VulkanError, compile_glsl,
};

/// Vertex shader source for the standard glTF material pipeline.
pub const GLTF_PBR_VERTEX_SHADER: &str = include_str!("../shaders/gltf_pbr.vert");
/// Fragment shader source for the standard glTF material pipeline.
pub const GLTF_PBR_FRAGMENT_SHADER: &str = include_str!("../shaders/gltf_pbr.frag");

/// Compiles and reflects both standard glTF PBR shaders for Vulkan 1.0.
pub fn compile_gltf_pbr_shaders() -> Result<(CompiledShader, CompiledShader), VulkanError> {
    let mut vertex_config = ShaderCompileConfig::new(ShaderStage::Vertex);
    vertex_config.source_name = "z-ideal/gltf_pbr.vert".to_owned();
    let mut fragment_config = ShaderCompileConfig::new(ShaderStage::Fragment);
    fragment_config.source_name = "z-ideal/gltf_pbr.frag".to_owned();
    Ok((
        compile_glsl(GLTF_PBR_VERTEX_SHADER, &vertex_config)?,
        compile_glsl(GLTF_PBR_FRAGMENT_SHADER, &fragment_config)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    #[test]
    fn pbr_shader_pair_compiles_and_reflects_expected_vertex_and_descriptor_contracts() {
        let (vertex, fragment) = compile_gltf_pbr_shaders().expect("compile glTF PBR shaders");
        assert_eq!(vertex.reflection().push_constant_size, Some(128));
        assert_eq!(
            vertex.reflection().vertex_inputs,
            vec![
                z_ideal_renderer::ShaderVertexInput {
                    location: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                },
                z_ideal_renderer::ShaderVertexInput {
                    location: 1,
                    format: vk::Format::R32G32_SFLOAT,
                },
                z_ideal_renderer::ShaderVertexInput {
                    location: 2,
                    format: vk::Format::R32G32B32_SFLOAT,
                },
                z_ideal_renderer::ShaderVertexInput {
                    location: 3,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                },
                z_ideal_renderer::ShaderVertexInput {
                    location: 4,
                    format: vk::Format::R32G32_SFLOAT,
                },
            ]
        );
        let mut descriptors = fragment.reflection().descriptors.clone();
        descriptors.sort_unstable_by_key(|binding| (binding.set, binding.binding));
        assert_eq!(descriptors.len(), 9);
        assert!(descriptors.iter().any(|binding| {
            binding.set == 0
                && binding.binding == 0
                && binding.descriptor_type == vk::DescriptorType::UNIFORM_BUFFER
        }));
        for binding in 1..=5 {
            assert!(descriptors.iter().any(|descriptor| {
                descriptor.set == 0
                    && descriptor.binding == binding
                    && descriptor.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }));
        }
        assert!(descriptors.iter().any(|binding| {
            binding.set == 1
                && binding.binding == 0
                && binding.descriptor_type == vk::DescriptorType::UNIFORM_BUFFER
        }));
        for binding in 1..=2 {
            assert!(descriptors.iter().any(|descriptor| {
                descriptor.set == 1
                    && descriptor.binding == binding
                    && descriptor.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }));
        }
    }
}
