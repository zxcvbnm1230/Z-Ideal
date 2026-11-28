//! Independently scheduled depth-based screen-space ambient occlusion compute pass.
use ash::vk;
use thiserror::Error;
use z_ideal_renderer::{
    BufferDescriptorWrite, CommandBufferRecording, CompiledShader, DescriptorResourceLease,
    DescriptorResourceWrite, DescriptorSetLeaseRef, ImageBarrier, ImageDescriptorWrite,
    ShaderCompileConfig, ShaderStage, VulkanBuffer, VulkanComputePipeline, VulkanDescriptorSet,
    VulkanError, VulkanImage, VulkanImageView, VulkanPipelineLayout, VulkanSampler,
    VulkanShaderModule, compile_glsl,
};

/// Compute shader source for the deterministic eight-direction depth-based SSAO pass.
pub const SSAO_COMPUTE_SHADER: &str = include_str!("../shaders/ssao.comp");

/// Compiles and reflects the SSAO compute shader.
pub fn compile_ssao_shader() -> Result<CompiledShader, VulkanError> {
    let mut config = ShaderCompileConfig::new(ShaderStage::Compute);
    config.source_name = "z-ideal/ssao.comp".to_owned();
    compile_glsl(SSAO_COMPUTE_SHADER, &config)
}

/// Validated per-dispatch SSAO parameters and inverse projection matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsaoParameters {
    /// Column-major inverse of the camera's Vulkan zero-to-one projection matrix.
    pub inverse_projection: [[f32; 4]; 4],
    /// Sample radius in pixels; range 1..=64.
    pub radius_pixels: f32,
    /// Maximum sample distance in view-space units; must be positive.
    pub radius_world: f32,
    /// Minimum view-space separation before a sample can occlude.
    pub depth_bias: f32,
    /// Non-negative AO multiplier; range 0..=4.
    pub intensity: f32,
    /// Width and height of the depth and output images.
    pub extent: [u32; 2],
}

impl SsaoParameters {
    fn validate(self) -> Result<(), VulkanSsaoRendererError> {
        let inverse_projection = z_ideal_math::Mat4::from_cols_array_2d(&self.inverse_projection);
        let determinant = inverse_projection.determinant();
        if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
            return Err(VulkanSsaoRendererError::InvalidParameters);
        }
        if self
            .inverse_projection
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
            || !self.radius_pixels.is_finite()
            || !(1.0..=64.0).contains(&self.radius_pixels)
            || !self.radius_world.is_finite()
            || self.radius_world <= 0.0
            || self.radius_world > 1_000.0
            || !self.depth_bias.is_finite()
            || self.depth_bias < 0.0
            || self.depth_bias > 10.0
            || !self.intensity.is_finite()
            || !(0.0..=4.0).contains(&self.intensity)
            || self.extent[0] < 2
            || self.extent[1] < 2
            || self.extent[0] > 16_384
            || self.extent[1] > 16_384
        {
            return Err(VulkanSsaoRendererError::InvalidParameters);
        }
        Ok(())
    }

    /// Encodes the exact 96-byte std140 compute uniform.
    #[must_use]
    pub fn to_std140_bytes(self) -> [u8; 96] {
        let mut bytes = [0_u8; 96];
        for (index, value) in self.inverse_projection.into_iter().flatten().enumerate() {
            let offset = index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
        for (index, value) in [
            self.radius_pixels,
            self.radius_world,
            self.depth_bias,
            self.intensity,
        ]
        .into_iter()
        .enumerate()
        {
            let offset = 64 + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        }
        bytes[80..84].copy_from_slice(&self.extent[0].to_ne_bytes());
        bytes[84..88].copy_from_slice(&self.extent[1].to_ne_bytes());
        bytes
    }
}

/// Borrowed resources used by one SSAO dispatch.
#[derive(Clone, Copy)]
pub struct SsaoDescriptorResources<'resource, 'image, 'device> {
    /// Sampled scene depth image in DEPTH_STENCIL_READ_ONLY_OPTIMAL layout.
    pub depth: &'resource VulkanImageView<'image, 'device>,
    /// Non-comparison sampler for the depth image; texelFetch uses integer coordinates.
    pub depth_sampler: &'resource VulkanSampler<'device>,
    /// R32_SFLOAT storage image view in GENERAL layout.
    pub output: &'resource VulkanImageView<'image, 'device>,
}

/// Writes depth, AO storage image, and parameter descriptors for one compute dispatch.
pub fn write_ssao_descriptors<'set, 'resource, 'image, 'pool, 'layout, 'device>(
    descriptor_set: &'set mut VulkanDescriptorSet<'pool, 'layout, 'device>,
    uniform_buffer: &'resource mut VulkanBuffer<'device>,
    parameters: SsaoParameters,
    resources: SsaoDescriptorResources<'resource, 'image, 'device>,
) -> Result<
    DescriptorResourceLease<'set, 'resource, 'image, 'pool, 'layout, 'device>,
    VulkanSsaoRendererError,
> {
    parameters.validate()?;
    let bindings = &descriptor_set.layout().config().bindings;
    if bindings.len() != 3
        || bindings.iter().any(|binding| {
            let expected = match binding.binding {
                0 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1 => vk::DescriptorType::STORAGE_IMAGE,
                2 => vk::DescriptorType::UNIFORM_BUFFER,
                _ => return true,
            };
            binding.descriptor_type != expected
                || binding.descriptor_count != 1
                || !binding.stage_flags.contains(vk::ShaderStageFlags::COMPUTE)
        })
    {
        return Err(VulkanSsaoRendererError::DescriptorLayoutMismatch);
    }
    if uniform_buffer.size() < 96 {
        return Err(VulkanSsaoRendererError::UniformBufferTooSmall);
    }
    let depth_image = resources.depth.image();
    let output_image = resources.output.image();
    if depth_image.config().extent.width != parameters.extent[0]
        || depth_image.config().extent.height != parameters.extent[1]
        || !depth_image
            .config()
            .usage
            .contains(vk::ImageUsageFlags::SAMPLED)
        || !resources
            .depth
            .config()
            .aspect
            .contains(vk::ImageAspectFlags::DEPTH)
        || resources.depth.config().view_type != vk::ImageViewType::TYPE_2D
        || resources.depth.config().base_mip_level != 0
        || resources.depth.config().level_count != 1
        || resources.depth.config().base_array_layer != 0
        || resources.depth.config().layer_count != 1
        || output_image.config().extent.width != parameters.extent[0]
        || output_image.config().extent.height != parameters.extent[1]
        || output_image.config().format != vk::Format::R32_SFLOAT
        || !output_image
            .config()
            .usage
            .contains(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        || !resources
            .output
            .config()
            .aspect
            .contains(vk::ImageAspectFlags::COLOR)
        || resources.output.config().view_type != vk::ImageViewType::TYPE_2D
        || resources.output.config().base_mip_level != 0
        || resources.output.config().level_count != 1
        || resources.output.config().base_array_layer != 0
        || resources.output.config().layer_count != 1
        || resources.depth_sampler.config().compare_op.is_some()
    {
        return Err(VulkanSsaoRendererError::IncompatibleImages);
    }
    uniform_buffer.write(0, &parameters.to_std140_bytes())?;
    Ok(descriptor_set.write_descriptors(&[
        DescriptorResourceWrite::Image(ImageDescriptorWrite {
            binding: 0,
            array_element: 0,
            view: resources.depth,
            sampler: Some(resources.depth_sampler),
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        }),
        DescriptorResourceWrite::Image(ImageDescriptorWrite {
            binding: 1,
            array_element: 0,
            view: resources.output,
            sampler: None,
            image_layout: vk::ImageLayout::GENERAL,
        }),
        DescriptorResourceWrite::Buffer(BufferDescriptorWrite {
            binding: 2,
            array_element: 0,
            buffer: uniform_buffer,
            offset: 0,
            range: 96,
        }),
    ])?)
}

/// Failures while creating or recording the SSAO compute pass.
#[derive(Debug, Error)]
pub enum VulkanSsaoRendererError {
    /// The descriptor layout does not match the depth/storage/uniform contract.
    #[error(
        "SSAO descriptor layout must contain depth sampler, storage output, and compute uniform bindings"
    )]
    DescriptorLayoutMismatch,
    /// The supplied uniform buffer is smaller than 96 bytes.
    #[error("SSAO uniform buffer must contain at least 96 bytes")]
    UniformBufferTooSmall,
    /// Parameters contain non-finite values, unsupported ranges, or invalid extents.
    #[error("SSAO parameters are invalid")]
    InvalidParameters,
    /// Input/output image formats, usages, aspects, or extents do not match the SSAO contract.
    #[error("SSAO input depth and R32_SFLOAT output images are incompatible")]
    IncompatibleImages,
    /// The dispatch output image is not a single-layer R32_SFLOAT storage and sampled image.
    #[error(
        "SSAO output image must be a single-mip, single-layer R32_SFLOAT storage and sampled image"
    )]
    InvalidOutputImage,
    /// Vulkan pipeline creation, descriptor update, dispatch, or barrier failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
}

/// Reusable compute pipeline for depth-based SSAO.
///
/// The caller owns and transitions the sampled depth input to DEPTH_STENCIL_READ_ONLY_OPTIMAL,
/// transitions a STORAGE|SAMPLED output to GENERAL before dispatch, and owns the descriptor resources.
/// `record` dispatches 8x8 workgroups and transitions the output to SHADER_READ_ONLY_OPTIMAL
/// for a subsequent fragment-shader consumer. Transition it back to GENERAL before reuse.
pub struct VulkanSsaoRenderer<'pipeline, 'layouts, 'device> {
    layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
    pipeline: VulkanComputePipeline<'pipeline, 'layouts, 'device>,
}

impl<'pipeline, 'layouts, 'device> VulkanSsaoRenderer<'pipeline, 'layouts, 'device> {
    /// Creates the compute pipeline from the reflected SSAO shader interface.
    pub fn new(
        layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
        shader: &'pipeline VulkanShaderModule<'device>,
    ) -> Result<Self, VulkanSsaoRendererError> {
        Ok(Self {
            layout,
            pipeline: VulkanComputePipeline::new(layout, shader)?,
        })
    }

    /// Records one SSAO dispatch and makes its output visible to fragment shaders.
    pub fn record<'recording, 'pool>(
        &'recording self,
        descriptor_set: DescriptorSetLeaseRef<'recording>,
        output_image: &'recording VulkanImage<'device>,
        recording: &mut CommandBufferRecording<'recording, 'pool, 'device>,
    ) -> Result<(), VulkanSsaoRendererError> {
        let config = output_image.config();
        if config.format != vk::Format::R32_SFLOAT
            || !config
                .usage
                .contains(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            || config.mip_levels != 1
            || config.array_layers != 1
        {
            return Err(VulkanSsaoRendererError::InvalidOutputImage);
        }
        recording.bind_compute_pipeline(&self.pipeline)?;
        recording.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            0,
            &[descriptor_set],
            &[],
        )?;
        recording.dispatch(
            config.extent.width.div_ceil(8),
            config.extent.height.div_ceil(8),
            1,
        )?;
        recording.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[ImageBarrier {
                image: output_image,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                source_access: vk::AccessFlags::SHADER_WRITE,
                destination_access: vk::AccessFlags::SHADER_READ,
                aspect: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            }],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{SsaoParameters, compile_ssao_shader};
    use ash::vk;

    fn parameters() -> SsaoParameters {
        SsaoParameters {
            inverse_projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            radius_pixels: 8.0,
            radius_world: 0.5,
            depth_bias: 0.01,
            intensity: 1.0,
            extent: [64, 32],
        }
    }

    #[test]
    fn ssao_shader_compiles_with_expected_compute_descriptor_contract() {
        let shader = compile_ssao_shader().expect("compile SSAO compute shader");
        assert_eq!(
            shader.reflection().stage,
            z_ideal_renderer::ShaderStage::Compute
        );
        assert_eq!(shader.reflection().descriptors.len(), 3);
        assert!(shader.reflection().descriptors.iter().any(|binding| {
            binding.binding == 0
                && binding.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        }));
        assert!(shader.reflection().descriptors.iter().any(|binding| {
            binding.binding == 1 && binding.descriptor_type == vk::DescriptorType::STORAGE_IMAGE
        }));
        assert!(shader.reflection().descriptors.iter().any(|binding| {
            binding.binding == 2 && binding.descriptor_type == vk::DescriptorType::UNIFORM_BUFFER
        }));
    }

    #[test]
    fn ssao_uniform_uses_exact_std140_offsets() {
        let bytes = parameters().to_std140_bytes();
        assert_eq!(bytes.len(), 96);
        assert_eq!(
            f32::from_ne_bytes(bytes[64..68].try_into().expect("radius")),
            8.0
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[80..84].try_into().expect("width")),
            64
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[84..88].try_into().expect("height")),
            32
        );
    }

    #[test]
    fn rejects_invalid_ssao_parameters() {
        let mut params = parameters();
        params.radius_pixels = f32::NAN;
        assert!(params.validate().is_err());
        params = parameters();
        params.extent = [0, 32];
        assert!(params.validate().is_err());
        params = parameters();
        params.intensity = 5.0;
        assert!(params.validate().is_err());
        params = parameters();
        params.inverse_projection[0][0] = f32::INFINITY;
        assert!(params.validate().is_err());
        params = parameters();
        params.inverse_projection = [[0.0; 4]; 4];
        assert!(params.validate().is_err());
    }
}
