//! Owned Vulkan compute pipelines.
use ash::vk;

use crate::{VulkanError, VulkanPipelineLayout, VulkanShaderModule, api_error};

/// Owns a compute pipeline and retains its pipeline layout and shader module.
pub struct VulkanComputePipeline<'pipeline, 'layouts, 'device> {
    device: &'device crate::VulkanDevice,
    handle: vk::Pipeline,
    _layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
    _shader: &'pipeline VulkanShaderModule<'device>,
}

impl<'pipeline, 'layouts, 'device> VulkanComputePipeline<'pipeline, 'layouts, 'device> {
    /// Creates a compute pipeline after validating the shader against its layout.
    pub fn new(
        layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
        shader: &'pipeline VulkanShaderModule<'device>,
    ) -> Result<Self, VulkanError> {
        if !std::ptr::eq(layout.device, shader.device) {
            return Err(VulkanError::ShaderModuleDeviceMismatch);
        }
        if shader.reflection().stage != crate::ShaderStage::Compute {
            return Err(VulkanError::PipelineShaderStageMismatch);
        }
        layout.validate_shader_interface(shader.reflection())?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.handle())
            .name(c"main");
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout.handle());

        // SAFETY: The shader module and pipeline layout remain alive for the
        // returned pipeline's lifetime; the create-info is valid for this call.
        let handle = match unsafe {
            layout.device.handle.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        } {
            Ok(mut pipelines) => pipelines.pop().ok_or(VulkanError::Api {
                operation: "create Vulkan compute pipeline",
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
                return Err(api_error("create Vulkan compute pipeline", error));
            }
        };

        Ok(Self {
            device: layout.device,
            handle,
            _layout: layout,
            _shader: shader,
        })
    }

    /// Returns the raw Vulkan pipeline handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

impl Drop for VulkanComputePipeline<'_, '_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before compute-pipeline destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns the pipeline and retains its device.
        unsafe {
            self.device.handle.destroy_pipeline(self.handle, None);
        }
    }
}
