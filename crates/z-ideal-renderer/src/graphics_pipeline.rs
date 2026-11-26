//! Owned Vulkan graphics pipelines with validated render-pass and shader interfaces.
use ash::vk;

use crate::{
    VulkanDevice, VulkanError, VulkanPipelineLayout, VulkanRenderPass, VulkanShaderModule,
    api_error,
};

/// Configurable fixed-function state for the initial graphics pipeline.
#[derive(Debug, Clone)]
pub struct GraphicsPipelineConfig {
    /// Render-pass subpass this pipeline targets.
    pub subpass: u32,
    /// Primitive topology.
    pub topology: vk::PrimitiveTopology,
    /// Whether primitive restart is enabled.
    pub primitive_restart_enable: bool,
    /// Vertex-buffer binding descriptions.
    pub vertex_bindings: Vec<vk::VertexInputBindingDescription>,
    /// Vertex attribute descriptions.
    pub vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
    /// Rasterizer polygon mode; the initial API supports fill mode.
    pub polygon_mode: vk::PolygonMode,
    /// Face-culling mode.
    pub cull_mode: vk::CullModeFlags,
    /// Front-face winding convention.
    pub front_face: vk::FrontFace,
    /// Enables depth comparisons.
    pub depth_test_enable: bool,
    /// Enables depth writes.
    pub depth_write_enable: bool,
    /// Depth comparison operation.
    pub depth_compare_op: vk::CompareOp,
    /// Color blend states, one per subpass color attachment. Empty selects disabled blending.
    pub color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
}

impl Default for GraphicsPipelineConfig {
    fn default() -> Self {
        Self {
            subpass: 0,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_test_enable: false,
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::LESS,
            color_blend_attachments: Vec::new(),
        }
    }
}

/// Owns a graphics pipeline and retains its layout, shader modules, and render pass.
pub struct VulkanGraphicsPipeline<'pipeline, 'layouts, 'device> {
    pub(super) device: &'device VulkanDevice,
    handle: vk::Pipeline,
    pub(super) layout_handle: vk::PipelineLayout,
    pub(super) render_pass: vk::RenderPass,
    pub(super) subpass: u32,
    _layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
    _vertex_shader: &'pipeline VulkanShaderModule<'device>,
    _fragment_shader: &'pipeline VulkanShaderModule<'device>,
    _render_pass: &'pipeline VulkanRenderPass<'device>,
}

impl<'pipeline, 'layouts, 'device> VulkanGraphicsPipeline<'pipeline, 'layouts, 'device> {
    /// Creates a graphics pipeline after validating shader interfaces and fixed-function state.
    pub fn new(
        layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
        vertex_shader: &'pipeline VulkanShaderModule<'device>,
        fragment_shader: &'pipeline VulkanShaderModule<'device>,
        render_pass: &'pipeline VulkanRenderPass<'device>,
        config: &GraphicsPipelineConfig,
    ) -> Result<Self, VulkanError> {
        if !std::ptr::eq(layout.device, vertex_shader.device)
            || !std::ptr::eq(layout.device, fragment_shader.device)
            || !std::ptr::eq(layout.device, render_pass.device)
        {
            return Err(VulkanError::ShaderModuleDeviceMismatch);
        }
        if vertex_shader.reflection().stage != crate::ShaderStage::Vertex
            || fragment_shader.reflection().stage != crate::ShaderStage::Fragment
        {
            return Err(VulkanError::PipelineShaderStageMismatch);
        }
        layout.validate_shader_interface(vertex_shader.reflection())?;
        layout.validate_shader_interface(fragment_shader.reflection())?;
        let color_count = render_pass
            .color_attachment_count(config.subpass)
            .ok_or(VulkanError::InvalidGraphicsPipelineConfig)?;
        let has_depth = render_pass
            .subpass_has_depth_stencil(config.subpass)
            .ok_or(VulkanError::InvalidGraphicsPipelineConfig)?;
        validate_graphics_pipeline_config(
            config,
            color_count,
            has_depth,
            &vertex_shader.reflection().vertex_inputs,
        )?;
        let sample_count = render_pass
            .subpass_sample_count(config.subpass)
            .ok_or(VulkanError::InvalidGraphicsPipelineConfig)?;

        let blend_attachments = effective_blend_attachments(config, color_count);
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader.handle())
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader.handle())
                .name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&config.vertex_bindings)
            .vertex_attribute_descriptions(&config.vertex_attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(config.topology)
            .primitive_restart_enable(config.primitive_restart_enable);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(config.polygon_mode)
            .cull_mode(config.cull_mode)
            .front_face(config.front_face)
            .depth_bias_enable(false)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(sample_count)
            .sample_shading_enable(false);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(config.depth_test_enable)
            .depth_write_enable(config.depth_write_enable)
            .depth_compare_op(config.depth_compare_op)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout.handle())
            .render_pass(render_pass.handle())
            .subpass(config.subpass)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        // SAFETY: All pipeline state arrays remain alive through creation. The
        // wrapper retains the layout, shaders, render pass, and device afterward.
        let handle = match unsafe {
            layout.device.handle.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        } {
            Ok(mut pipelines) => pipelines.pop().ok_or(VulkanError::Api {
                operation: "create Vulkan graphics pipeline",
                result: vk::Result::ERROR_UNKNOWN,
            })?,
            Err((partial, error)) => {
                // SAFETY: Vulkan returned ownership of any partially created handles.
                unsafe {
                    for pipeline in partial {
                        if pipeline != vk::Pipeline::null() {
                            layout.device.handle.destroy_pipeline(pipeline, None);
                        }
                    }
                }
                return Err(api_error("create Vulkan graphics pipeline", error));
            }
        };

        Ok(Self {
            device: layout.device,
            handle,
            layout_handle: layout.handle(),
            render_pass: render_pass.handle(),
            subpass: config.subpass,
            _layout: layout,
            _vertex_shader: vertex_shader,
            _fragment_shader: fragment_shader,
            _render_pass: render_pass,
        })
    }

    /// Returns the raw Vulkan pipeline handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

impl Drop for VulkanGraphicsPipeline<'_, '_, '_> {
    fn drop(&mut self) {
        // SAFETY: Recorded command buffers retain this pipeline until their submission fence
        // completes; raw Vulkan users synchronize manually.
        unsafe {
            self.device.handle.destroy_pipeline(self.handle, None);
        }
    }
}

fn validate_graphics_pipeline_config(
    config: &GraphicsPipelineConfig,
    color_attachment_count: u32,
    has_depth_attachment: bool,
    vertex_inputs: &[crate::ShaderVertexInput],
) -> Result<(), VulkanError> {
    if !matches!(
        config.topology,
        vk::PrimitiveTopology::POINT_LIST
            | vk::PrimitiveTopology::LINE_LIST
            | vk::PrimitiveTopology::LINE_STRIP
            | vk::PrimitiveTopology::TRIANGLE_LIST
            | vk::PrimitiveTopology::TRIANGLE_STRIP
            | vk::PrimitiveTopology::TRIANGLE_FAN
    ) || config.polygon_mode != vk::PolygonMode::FILL
        || (config.primitive_restart_enable
            && !matches!(
                config.topology,
                vk::PrimitiveTopology::LINE_STRIP | vk::PrimitiveTopology::TRIANGLE_STRIP
            ))
        || !matches!(
            config.front_face,
            vk::FrontFace::COUNTER_CLOCKWISE | vk::FrontFace::CLOCKWISE
        )
        || !matches!(
            config.depth_compare_op,
            vk::CompareOp::NEVER
                | vk::CompareOp::LESS
                | vk::CompareOp::EQUAL
                | vk::CompareOp::LESS_OR_EQUAL
                | vk::CompareOp::GREATER
                | vk::CompareOp::NOT_EQUAL
                | vk::CompareOp::GREATER_OR_EQUAL
                | vk::CompareOp::ALWAYS
        )
        || (config.depth_test_enable && !has_depth_attachment)
        || (config.depth_write_enable && !config.depth_test_enable)
    {
        return Err(VulkanError::InvalidGraphicsPipelineConfig);
    }
    let valid_cull_modes =
        vk::CullModeFlags::FRONT | vk::CullModeFlags::BACK | vk::CullModeFlags::FRONT_AND_BACK;
    if config.cull_mode.as_raw() & !valid_cull_modes.as_raw() != 0 {
        return Err(VulkanError::InvalidGraphicsPipelineConfig);
    }
    validate_vertex_input(
        &config.vertex_bindings,
        &config.vertex_attributes,
        vertex_inputs,
    )?;
    if !config.color_blend_attachments.is_empty()
        && config.color_blend_attachments.len() != color_attachment_count as usize
    {
        return Err(VulkanError::InvalidGraphicsPipelineConfig);
    }
    let valid_color_mask = vk::ColorComponentFlags::R
        | vk::ColorComponentFlags::G
        | vk::ColorComponentFlags::B
        | vk::ColorComponentFlags::A;
    if config.color_blend_attachments.iter().any(|attachment| {
        (attachment.blend_enable != vk::FALSE && attachment.blend_enable != vk::TRUE)
            || attachment.color_write_mask.as_raw() & !valid_color_mask.as_raw() != 0
    }) {
        return Err(VulkanError::InvalidGraphicsPipelineConfig);
    }
    Ok(())
}

fn validate_vertex_input(
    bindings: &[vk::VertexInputBindingDescription],
    attributes: &[vk::VertexInputAttributeDescription],
    required_inputs: &[crate::ShaderVertexInput],
) -> Result<(), VulkanError> {
    for (index, binding) in bindings.iter().enumerate() {
        if !matches!(
            binding.input_rate,
            vk::VertexInputRate::VERTEX | vk::VertexInputRate::INSTANCE
        ) || bindings[..index]
            .iter()
            .any(|previous| previous.binding == binding.binding)
        {
            return Err(VulkanError::InvalidGraphicsPipelineConfig);
        }
    }
    for (index, attribute) in attributes.iter().enumerate() {
        if attribute.format == vk::Format::UNDEFINED
            || !bindings
                .iter()
                .any(|binding| binding.binding == attribute.binding)
            || attributes[..index]
                .iter()
                .any(|previous| previous.location == attribute.location)
        {
            return Err(VulkanError::InvalidGraphicsPipelineConfig);
        }
    }
    for required in required_inputs {
        if !attributes.iter().any(|attribute| {
            attribute.location == required.location && attribute.format == required.format
        }) {
            return Err(VulkanError::InvalidGraphicsPipelineConfig);
        }
    }
    Ok(())
}

fn effective_blend_attachments(
    config: &GraphicsPipelineConfig,
    color_attachment_count: u32,
) -> Vec<vk::PipelineColorBlendAttachmentState> {
    if config.color_blend_attachments.is_empty() {
        vec![
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                );
            color_attachment_count as usize
        ]
    } else {
        config.color_blend_attachments.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GraphicsPipelineConfig, effective_blend_attachments, validate_graphics_pipeline_config,
        validate_vertex_input,
    };
    use ash::vk;

    #[test]
    fn default_pipeline_state_is_valid_for_a_color_subpass() {
        let config = GraphicsPipelineConfig::default();
        assert!(validate_graphics_pipeline_config(&config, 1, false, &[]).is_ok());
        assert_eq!(effective_blend_attachments(&config, 1).len(), 1);
        assert_eq!(
            effective_blend_attachments(&config, 1)[0].blend_enable,
            vk::FALSE
        );
    }

    #[test]
    fn rejects_depth_state_without_depth_attachment_and_blend_count_mismatch() {
        let mut config = GraphicsPipelineConfig {
            depth_test_enable: true,
            ..GraphicsPipelineConfig::default()
        };
        assert!(validate_graphics_pipeline_config(&config, 1, false, &[]).is_err());
        config.depth_test_enable = false;
        config.color_blend_attachments = vec![vk::PipelineColorBlendAttachmentState::default(); 2];
        assert!(validate_graphics_pipeline_config(&config, 1, false, &[]).is_err());
    }

    #[test]
    fn rejects_invalid_primitive_restart_and_blend_boolean_values() {
        let mut config = GraphicsPipelineConfig {
            primitive_restart_enable: true,
            ..GraphicsPipelineConfig::default()
        };
        assert!(validate_graphics_pipeline_config(&config, 1, false, &[]).is_err());
        config.primitive_restart_enable = false;
        config.color_blend_attachments = vec![vk::PipelineColorBlendAttachmentState {
            blend_enable: 2,
            ..vk::PipelineColorBlendAttachmentState::default()
        }];
        assert!(validate_graphics_pipeline_config(&config, 1, false, &[]).is_err());
    }

    #[test]
    fn validates_vertex_bindings_and_attribute_locations() {
        let binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: 24,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let attribute = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        };
        assert!(validate_vertex_input(&[binding], &[attribute], &[]).is_ok());
        assert!(validate_vertex_input(&[binding, binding], &[attribute], &[]).is_err());
        assert!(validate_vertex_input(&[binding], &[attribute, attribute], &[]).is_err());
        assert!(validate_vertex_input(&[], &[attribute], &[]).is_err());
        let required = crate::ShaderVertexInput {
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
        };
        assert!(validate_vertex_input(&[binding], &[attribute], &[required]).is_ok());
        let mismatched = crate::ShaderVertexInput {
            format: vk::Format::R32G32_SFLOAT,
            ..required
        };
        assert!(validate_vertex_input(&[binding], &[attribute], &[mismatched]).is_err());
    }
}
