//! Depth-only shadow-map rendering for a caller-owned light camera and depth target.
use crate::{RenderSceneSnapshot, VulkanSceneMeshes, VulkanSceneMeshesError, VulkanSceneViewport};
use thiserror::Error;
use z_ideal_renderer::{
    CommandBufferRecording, CompiledShader, GraphicsPipelineConfig, ShaderCompileConfig,
    ShaderStage, VulkanError, VulkanGraphicsPipeline, VulkanPipelineLayout, VulkanRenderPass,
    VulkanShaderModule, compile_glsl,
};

/// Vertex shader for a depth-only shadow pass. Its 64-byte push constant is light VP * model.
pub const SHADOW_VERTEX_SHADER: &str = include_str!("../shaders/shadow_depth.vert");
/// Empty fragment shader required by the initial graphics-pipeline API.
pub const SHADOW_FRAGMENT_SHADER: &str = include_str!("../shaders/shadow_depth.frag");

/// Compiles and reflects the depth-only shadow shader pair.
pub fn compile_shadow_shaders() -> Result<(CompiledShader, CompiledShader), VulkanError> {
    let mut vertex = ShaderCompileConfig::new(ShaderStage::Vertex);
    vertex.source_name = "z-ideal/shadow_depth.vert".to_owned();
    let mut fragment = ShaderCompileConfig::new(ShaderStage::Fragment);
    fragment.source_name = "z-ideal/shadow_depth.frag".to_owned();
    Ok((
        compile_glsl(SHADOW_VERTEX_SHADER, &vertex)?,
        compile_glsl(SHADOW_FRAGMENT_SHADER, &fragment)?,
    ))
}

/// Errors while creating or recording a depth-only shadow pass.
#[derive(Debug, Error)]
pub enum VulkanShadowRendererError {
    /// The render pass must have exactly one depth attachment and no color attachments.
    #[error("shadow renderer requires a depth-only render-pass subpass")]
    InvalidDepthOnlyRenderPass,
    /// Shadow pipeline must enable depth testing and depth writes.
    #[error("shadow pipeline requires depth testing and depth writes")]
    InvalidDepthState,
    /// Pipeline creation or command recording failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    /// Scene mesh recording failed.
    #[error(transparent)]
    SceneMeshes(#[from] VulkanSceneMeshesError),
}

/// Reusable depth-only shadow renderer.
///
/// The caller owns the depth image, depth view, comparison sampler, render pass,
/// framebuffer, light-camera snapshot, and synchronization. The renderer owns only
/// its graphics pipeline and borrows the pipeline layout, shaders, and render pass.
/// The light-camera snapshot's projection * view matrix is used for shadow projection.
/// Use a perspective snapshot for spot lights or an orthographic snapshot for directional
/// lights. Point-light cubemap shadowing is not provided by this single-view pass.
pub struct VulkanShadowRenderer<'pipeline, 'layouts, 'device> {
    layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
    pipeline: VulkanGraphicsPipeline<'pipeline, 'layouts, 'device>,
}

impl<'pipeline, 'layouts, 'device> VulkanShadowRenderer<'pipeline, 'layouts, 'device> {
    /// Creates a depth-only pipeline compatible with the supplied shadow render pass.
    ///
    /// base_config must use the uploaded mesh vertex layout and set depth testing/writes.
    /// Color blend attachments must be empty because the target has no color outputs.
    pub fn new(
        layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
        vertex_shader: &'pipeline VulkanShaderModule<'device>,
        fragment_shader: &'pipeline VulkanShaderModule<'device>,
        render_pass: &'pipeline VulkanRenderPass<'device>,
        base_config: &GraphicsPipelineConfig,
    ) -> Result<Self, VulkanShadowRendererError> {
        let subpass = base_config.subpass;
        if render_pass.config().attachments.len() != 1
            || render_pass.color_attachment_count(subpass) != Some(0)
            || render_pass.subpass_has_depth_stencil(subpass) != Some(true)
            || !base_config.color_blend_attachments.is_empty()
        {
            return Err(VulkanShadowRendererError::InvalidDepthOnlyRenderPass);
        }
        if !base_config.depth_test_enable || !base_config.depth_write_enable {
            return Err(VulkanShadowRendererError::InvalidDepthState);
        }
        let pipeline = VulkanGraphicsPipeline::new(
            layout,
            vertex_shader,
            fragment_shader,
            render_pass,
            base_config,
        )?;
        Ok(Self { layout, pipeline })
    }

    /// Records all snapshot primitives into the active depth-only shadow render pass.
    ///
    /// The snapshot must have been extracted using the light's view/projection camera.
    /// Call mesh upload barriers before beginning the render pass. Bind a compatible
    /// depth-only framebuffer before this call. Returns the number of primitive draws.
    pub fn record<'recording, 'pool>(
        &'recording self,
        meshes: &'recording VulkanSceneMeshes<'device>,
        light_snapshot: &RenderSceneSnapshot,
        recording: &mut CommandBufferRecording<'recording, 'pool, 'device>,
        viewport: VulkanSceneViewport,
    ) -> Result<usize, VulkanShadowRendererError> {
        recording.bind_graphics_pipeline(&self.pipeline)?;
        recording.set_viewport(viewport.viewport)?;
        recording.set_scissor(viewport.scissor)?;
        Ok(meshes.record_shadow_draws(light_snapshot, recording, self.layout)?)
    }
}

#[cfg(test)]
mod tests {
    use super::compile_shadow_shaders;
    use ash::vk;
    #[test]
    fn shadow_shader_pair_compiles_without_descriptors_and_uses_one_mvp_push_constant() {
        let (vertex, fragment) = compile_shadow_shaders().expect("compile shadow shaders");
        assert_eq!(vertex.reflection().push_constant_size, Some(64));
        assert_eq!(vertex.reflection().descriptors, Vec::new());
        assert_eq!(fragment.reflection().descriptors, Vec::new());
        assert_eq!(fragment.reflection().push_constant_size, None);
        assert_eq!(
            vertex.reflection().vertex_inputs,
            vec![z_ideal_renderer::ShaderVertexInput {
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
            }]
        );
    }
}
