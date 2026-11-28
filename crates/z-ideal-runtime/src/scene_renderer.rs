//! Runtime-owned orchestration for drawing a scene with the standard glTF PBR pipeline.
use ash::vk;
use thiserror::Error;
use z_ideal_assets::ImportedMaterial;
use z_ideal_renderer::{
    CommandBufferRecording, DescriptorSetLeaseRef, GraphicsPipelineConfig, VulkanError,
    VulkanGraphicsPipeline, VulkanPipelineLayout, VulkanRenderPass, VulkanShaderModule,
};

use crate::gltf_material::apply_gltf_material_pipeline_variant;
use crate::{GltfMaterialError, RenderSceneSnapshot, VulkanSceneMeshes};

/// Failures while creating or recording the runtime PBR scene renderer.
#[derive(Debug, Error)]
pub enum VulkanSceneRendererError {
    /// The selected render-pass subpass has no color attachment.
    #[error("PBR scene renderer requires a render-pass subpass with a color attachment")]
    MissingColorAttachment,
    /// Material pipeline-state configuration failed.
    #[error(transparent)]
    Material(#[from] GltfMaterialError),
    /// Pipeline creation, binding, or command recording failed.
    #[error(transparent)]
    Vulkan(#[from] VulkanError),
    /// Scene mesh lookup or indexed draw recording failed.
    #[error(transparent)]
    SceneMeshes(#[from] crate::VulkanSceneMeshesError),
}

/// Dynamic viewport and scissor state for one scene-rendering pass.
#[derive(Debug, Clone, Copy)]
pub struct VulkanSceneViewport {
    /// Dynamic Vulkan viewport.
    pub viewport: vk::Viewport,
    /// Dynamic Vulkan scissor rectangle.
    pub scissor: vk::Rect2D,
}

/// Reusable runtime PBR renderer with opaque/masked/blended and sidedness variants.
///
/// The renderer borrows the caller's pipeline layout, shader modules, and render pass.
/// It owns four fixed-function pipeline variants and chooses one per primitive from
/// its imported material. Render-target allocation, render-pass/framebuffer lifetime,
/// scene/material descriptor resources, and frame synchronization remain explicit.
pub struct VulkanSceneRenderer<'pipeline, 'layouts, 'device> {
    layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
    pipelines: Vec<VulkanGraphicsPipeline<'pipeline, 'layouts, 'device>>,
}

impl<'pipeline, 'layouts, 'device> VulkanSceneRenderer<'pipeline, 'layouts, 'device> {
    /// Creates opaque/masked and blended variants for single- and double-sided materials.
    pub fn new(
        layout: &'pipeline VulkanPipelineLayout<'layouts, 'device>,
        vertex_shader: &'pipeline VulkanShaderModule<'device>,
        fragment_shader: &'pipeline VulkanShaderModule<'device>,
        render_pass: &'pipeline VulkanRenderPass<'device>,
        base_config: &GraphicsPipelineConfig,
    ) -> Result<Self, VulkanSceneRendererError> {
        let color_attachment_count = render_pass
            .color_attachment_count(base_config.subpass)
            .ok_or(VulkanSceneRendererError::MissingColorAttachment)?;
        let color_attachment_count = usize::try_from(color_attachment_count)
            .map_err(|_| VulkanSceneRendererError::MissingColorAttachment)?;
        if color_attachment_count == 0 {
            return Err(VulkanSceneRendererError::MissingColorAttachment);
        }

        let mut pipelines = Vec::with_capacity(4);
        for (alpha_blend, double_sided) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let mut config = base_config.clone();
            apply_gltf_material_pipeline_variant(
                &mut config,
                alpha_blend,
                double_sided,
                color_attachment_count,
            )?;
            pipelines.push(VulkanGraphicsPipeline::new(
                layout,
                vertex_shader,
                fragment_shader,
                render_pass,
                &config,
            )?);
        }
        Ok(Self { layout, pipelines })
    }

    fn pipeline_for(
        &self,
        material: Option<&ImportedMaterial>,
    ) -> &VulkanGraphicsPipeline<'pipeline, 'layouts, 'device> {
        let alpha_blend = material.is_some_and(|material| {
            material.alpha_mode == z_ideal_assets::MaterialAlphaMode::Blend
        });
        let double_sided = material.is_some_and(|material| material.double_sided);
        let index = match (alpha_blend, double_sided) {
            (false, false) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (true, true) => 3,
        };
        &self.pipelines[index]
    }

    /// Records a snapshot's indexed draws, selecting the matching pipeline per primitive.
    ///
    /// Call inside the compatible render pass after recording mesh upload barriers.
    /// The scene descriptor lease must match pipeline set 1; the material callback
    /// binds the current primitive's set 0 (including an explicit default material
    /// for primitives without an imported material). Viewport/scissor are rebound
    /// after each pipeline switch. If a callback or draw fails, discard the command
    /// buffer rather than submitting a partially recorded frame.
    pub fn record<'recording, 'pool, F>(
        &'recording self,
        meshes: &'recording VulkanSceneMeshes<'device>,
        snapshot: &RenderSceneSnapshot,
        recording: &mut CommandBufferRecording<'recording, 'pool, 'device>,
        scene_descriptor: DescriptorSetLeaseRef<'recording>,
        viewport: VulkanSceneViewport,
        mut bind_material: F,
    ) -> Result<usize, VulkanSceneRendererError>
    where
        F: FnMut(
            Option<&'recording ImportedMaterial>,
            &mut CommandBufferRecording<'recording, 'pool, 'device>,
            &'recording VulkanPipelineLayout<'layouts, 'device>,
        ) -> Result<(), VulkanError>,
    {
        Ok(meshes.record_pbr_draws_with_materials(
            snapshot,
            recording,
            self.layout,
            |material, recording, layout| {
                recording.bind_graphics_pipeline(self.pipeline_for(material))?;
                recording.set_viewport(viewport.viewport)?;
                recording.set_scissor(viewport.scissor)?;
                recording.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    1,
                    &[scene_descriptor],
                    &[],
                )?;
                bind_material(material, recording, layout)
            },
        )?)
    }
}
