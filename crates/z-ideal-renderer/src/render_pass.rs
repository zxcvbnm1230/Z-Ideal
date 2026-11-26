//! Owned Vulkan render passes and framebuffers.
use ash::vk;

use crate::{VulkanDevice, VulkanError, VulkanImageView, VulkanSwapchain, api_error};

/// One attachment description owned by a render-pass configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderPassAttachmentConfig {
    /// Attachment format.
    pub format: vk::Format,
    /// Number of samples in each attachment texel.
    pub samples: vk::SampleCountFlags,
    /// Operation applied to color/depth contents at render-pass start.
    pub load_op: vk::AttachmentLoadOp,
    /// Operation applied to color/depth contents at render-pass end.
    pub store_op: vk::AttachmentStoreOp,
    /// Operation applied to stencil contents at render-pass start.
    pub stencil_load_op: vk::AttachmentLoadOp,
    /// Operation applied to stencil contents at render-pass end.
    pub stencil_store_op: vk::AttachmentStoreOp,
    /// Layout expected when the render pass begins.
    pub initial_layout: vk::ImageLayout,
    /// Layout transitioned to when the render pass completes.
    pub final_layout: vk::ImageLayout,
}

/// Reference to one render-pass attachment in a subpass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderPassAttachmentReference {
    /// Index into [`RenderPassConfig::attachments`].
    pub attachment: u32,
    /// Layout used while this subpass accesses the attachment.
    pub layout: vk::ImageLayout,
}

/// One graphics subpass and its attachment references.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderPassSubpassConfig {
    /// Attachments read as input attachments.
    pub input_attachments: Vec<RenderPassAttachmentReference>,
    /// Color output attachments.
    pub color_attachments: Vec<RenderPassAttachmentReference>,
    /// Optional multisample resolve attachments; when present, length must match colors.
    pub resolve_attachments: Vec<RenderPassAttachmentReference>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<RenderPassAttachmentReference>,
    /// Attachments preserved but not otherwise used by this subpass.
    pub preserve_attachments: Vec<u32>,
}

/// Inter-subpass or external dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderPassDependency {
    /// Source subpass or [`vk::SUBPASS_EXTERNAL`].
    pub source_subpass: u32,
    /// Destination subpass or [`vk::SUBPASS_EXTERNAL`].
    pub destination_subpass: u32,
    /// Pipeline stages that produce the dependency.
    pub source_stage_mask: vk::PipelineStageFlags,
    /// Pipeline stages that consume the dependency.
    pub destination_stage_mask: vk::PipelineStageFlags,
    /// Access types made available by the source.
    pub source_access_mask: vk::AccessFlags,
    /// Access types made visible to the destination.
    pub destination_access_mask: vk::AccessFlags,
    /// Vulkan dependency flags.
    pub flags: vk::DependencyFlags,
}

/// Complete render-pass description.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderPassConfig {
    /// Attachments used by the render pass.
    pub attachments: Vec<RenderPassAttachmentConfig>,
    /// Subpasses executed by the render pass; at least one is required.
    pub subpasses: Vec<RenderPassSubpassConfig>,
    /// Explicit execution and memory dependencies.
    pub dependencies: Vec<RenderPassDependency>,
}

/// Owns a Vulkan render pass and the attachment metadata needed by framebuffers/pipelines.
pub struct VulkanRenderPass<'device> {
    pub(super) device: &'device VulkanDevice,
    handle: vk::RenderPass,
    config: RenderPassConfig,
    attachment_usage: Vec<vk::ImageUsageFlags>,
    subpass_color_counts: Vec<u32>,
    subpass_sample_counts: Vec<vk::SampleCountFlags>,
    subpass_has_depth: Vec<bool>,
}

impl<'device> VulkanRenderPass<'device> {
    /// Creates a render pass after validating all attachment and subpass references.
    pub fn new(
        device: &'device VulkanDevice,
        config: RenderPassConfig,
    ) -> Result<Self, VulkanError> {
        let metadata = validate_render_pass_config(&config)?;
        let attachments: Vec<_> = config
            .attachments
            .iter()
            .map(|attachment| {
                vk::AttachmentDescription::default()
                    .format(attachment.format)
                    .samples(attachment.samples)
                    .load_op(attachment.load_op)
                    .store_op(attachment.store_op)
                    .stencil_load_op(attachment.stencil_load_op)
                    .stencil_store_op(attachment.stencil_store_op)
                    .initial_layout(attachment.initial_layout)
                    .final_layout(attachment.final_layout)
            })
            .collect();
        let input_refs: Vec<Vec<_>> = config
            .subpasses
            .iter()
            .map(|subpass| to_vk_refs(&subpass.input_attachments))
            .collect();
        let color_refs: Vec<Vec<_>> = config
            .subpasses
            .iter()
            .map(|subpass| to_vk_refs(&subpass.color_attachments))
            .collect();
        let resolve_refs: Vec<Vec<_>> = config
            .subpasses
            .iter()
            .map(|subpass| to_vk_refs(&subpass.resolve_attachments))
            .collect();
        let depth_refs: Vec<_> = config
            .subpasses
            .iter()
            .map(|subpass| subpass.depth_stencil_attachment.map(to_vk_ref))
            .collect();
        let preserve_refs: Vec<Vec<_>> = config
            .subpasses
            .iter()
            .map(|subpass| subpass.preserve_attachments.clone())
            .collect();

        let subpasses: Vec<_> = (0..config.subpasses.len())
            .map(|index| {
                let mut info = vk::SubpassDescription::default()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .input_attachments(&input_refs[index])
                    .color_attachments(&color_refs[index])
                    .preserve_attachments(&preserve_refs[index]);
                if !resolve_refs[index].is_empty() {
                    info = info.resolve_attachments(&resolve_refs[index]);
                }
                if let Some(depth) = depth_refs[index].as_ref() {
                    info = info.depth_stencil_attachment(depth);
                }
                info
            })
            .collect();
        let dependencies: Vec<_> = config
            .dependencies
            .iter()
            .map(|dependency| {
                vk::SubpassDependency::default()
                    .src_subpass(dependency.source_subpass)
                    .dst_subpass(dependency.destination_subpass)
                    .src_stage_mask(dependency.source_stage_mask)
                    .dst_stage_mask(dependency.destination_stage_mask)
                    .src_access_mask(dependency.source_access_mask)
                    .dst_access_mask(dependency.destination_access_mask)
                    .dependency_flags(dependency.flags)
            })
            .collect();
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        // SAFETY: Every nested attachment-reference vector remains alive and
        // unmoved through creation; all indices and subpass relationships were validated.
        let handle = unsafe {
            device
                .handle
                .create_render_pass(&info, None)
                .map_err(|error| api_error("create Vulkan render pass", error))?
        };

        Ok(Self {
            device,
            handle,
            config,
            attachment_usage: metadata.attachment_usage,
            subpass_color_counts: metadata.subpass_color_counts,
            subpass_sample_counts: metadata.subpass_sample_counts,
            subpass_has_depth: metadata.subpass_has_depth,
        })
    }

    /// Returns the raw Vulkan render-pass handle.
    #[must_use]
    pub const fn handle(&self) -> vk::RenderPass {
        self.handle
    }

    /// Returns the render-pass configuration.
    #[must_use]
    pub const fn config(&self) -> &RenderPassConfig {
        &self.config
    }

    /// Returns the number of render-pass attachments.
    #[must_use]
    pub fn attachment_count(&self) -> usize {
        self.config.attachments.len()
    }

    /// Returns the number of subpasses.
    #[must_use]
    pub fn subpass_count(&self) -> usize {
        self.config.subpasses.len()
    }

    /// Returns the number of color outputs in a subpass.
    #[must_use]
    pub fn color_attachment_count(&self, subpass: u32) -> Option<u32> {
        self.subpass_color_counts.get(subpass as usize).copied()
    }

    /// Returns the sample count used by a subpass's color/depth attachments.
    #[must_use]
    pub fn subpass_sample_count(&self, subpass: u32) -> Option<vk::SampleCountFlags> {
        self.subpass_sample_counts.get(subpass as usize).copied()
    }

    /// Returns whether a subpass has a depth/stencil attachment.
    #[must_use]
    pub fn subpass_has_depth_stencil(&self, subpass: u32) -> Option<bool> {
        self.subpass_has_depth.get(subpass as usize).copied()
    }
}

impl Drop for VulkanRenderPass<'_> {
    fn drop(&mut self) {
        // SAFETY: Framebuffers and pipelines borrow this render pass, and recorded command
        // buffers retain those wrappers until their submission fences complete.
        unsafe {
            self.device.handle.destroy_render_pass(self.handle, None);
        }
    }
}

/// Owns a framebuffer and retains its render pass and image views.
pub struct VulkanFramebuffer<'render_pass, 'views, 'image, 'device> {
    device: &'device VulkanDevice,
    handle: vk::Framebuffer,
    render_pass: &'render_pass VulkanRenderPass<'device>,
    _attachments: Vec<&'views VulkanImageView<'image, 'device>>,
    extent: vk::Extent2D,
    layers: u32,
}

impl<'render_pass, 'views, 'image, 'device>
    VulkanFramebuffer<'render_pass, 'views, 'image, 'device>
{
    /// Creates a framebuffer from views matching every render-pass attachment.
    pub fn new(
        render_pass: &'render_pass VulkanRenderPass<'device>,
        attachments: &[&'views VulkanImageView<'image, 'device>],
        extent: vk::Extent2D,
        layers: u32,
    ) -> Result<Self, VulkanError> {
        if extent.width == 0
            || extent.height == 0
            || layers == 0
            || attachments.len() != render_pass.config.attachments.len()
        {
            return Err(VulkanError::InvalidFramebufferConfig);
        }
        // SAFETY: Physical-device properties belong to the instance retained by the render pass.
        let limits = unsafe {
            render_pass
                .device
                ._instance
                .handle
                .get_physical_device_properties(render_pass.device.physical_device)
                .limits
        };
        if extent.width > limits.max_framebuffer_width
            || extent.height > limits.max_framebuffer_height
            || layers > limits.max_framebuffer_layers
        {
            return Err(VulkanError::InvalidFramebufferConfig);
        }
        for (index, (view, attachment)) in attachments
            .iter()
            .zip(&render_pass.config.attachments)
            .enumerate()
        {
            if !std::ptr::eq(render_pass.device, view.image().device) {
                return Err(VulkanError::FramebufferDeviceMismatch);
            }
            let image = view.image().config();
            let view_config = view.config();
            let mip_width = (image.extent.width >> view_config.base_mip_level).max(1);
            let mip_height = (image.extent.height >> view_config.base_mip_level).max(1);
            let required_usage = render_pass.attachment_usage[index];
            if image.format != attachment.format
                || image.samples != attachment.samples
                || mip_width < extent.width
                || mip_height < extent.height
                || layers > view_config.layer_count
                || !image.usage.contains(required_usage)
                || (required_usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    && !view_config.aspect.contains(vk::ImageAspectFlags::COLOR))
                || (required_usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    && !view_config
                        .aspect
                        .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL))
            {
                return Err(VulkanError::InvalidFramebufferConfig);
            }
            // SAFETY: The physical device belongs to the instance retained by the render pass.
            let format_properties = unsafe {
                render_pass
                    .device
                    ._instance
                    .handle
                    .get_physical_device_format_properties(
                        render_pass.device.physical_device,
                        image.format,
                    )
            };
            let available_features = if image.tiling == vk::ImageTiling::LINEAR {
                format_properties.linear_tiling_features
            } else {
                format_properties.optimal_tiling_features
            };
            if (required_usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                && !available_features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT))
                || (required_usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    && !available_features
                        .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT))
            {
                return Err(VulkanError::InvalidFramebufferConfig);
            }
        }
        let raw_views: Vec<_> = attachments.iter().map(|view| view.handle()).collect();
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass.handle)
            .attachments(&raw_views)
            .width(extent.width)
            .height(extent.height)
            .layers(layers);
        // SAFETY: Render pass and every image view remain alive for the returned
        // framebuffer lifetime; attachment compatibility was checked above.
        let handle = unsafe {
            render_pass
                .device
                .handle
                .create_framebuffer(&info, None)
                .map_err(|error| api_error("create Vulkan framebuffer", error))?
        };
        Ok(Self {
            device: render_pass.device,
            handle,
            render_pass,
            _attachments: attachments.to_vec(),
            extent,
            layers,
        })
    }

    /// Returns the raw Vulkan framebuffer handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Framebuffer {
        self.handle
    }

    /// Returns the render pass this framebuffer was created for.
    #[must_use]
    pub const fn render_pass(&self) -> &'render_pass VulkanRenderPass<'device> {
        self.render_pass
    }

    /// Returns the framebuffer width and height.
    pub const fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Returns the framebuffer layer count.
    #[must_use]
    pub const fn layers(&self) -> u32 {
        self.layers
    }
}

impl Drop for VulkanFramebuffer<'_, '_, '_, '_> {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the framebuffer and retains all attachments.
        unsafe {
            self.device.handle.destroy_framebuffer(self.handle, None);
        }
    }
}

/// Owns a framebuffer targeting one image from a Vulkan swapchain.
///
/// The framebuffer borrows both the render pass and swapchain. Recreating or dropping either
/// resource is therefore prevented while this framebuffer exists.
pub struct VulkanSwapchainFramebuffer<'render_pass, 'swapchain, 'device> {
    device: &'device VulkanDevice,
    handle: vk::Framebuffer,
    render_pass: &'render_pass VulkanRenderPass<'device>,
    _swapchain: &'swapchain VulkanSwapchain<'device>,
    image_index: u32,
    extent: vk::Extent2D,
}

impl<'render_pass, 'swapchain, 'device>
    VulkanSwapchainFramebuffer<'render_pass, 'swapchain, 'device>
{
    /// Creates a framebuffer for a single-color-attachment render pass and one swapchain image.
    pub fn new(
        render_pass: &'render_pass VulkanRenderPass<'device>,
        swapchain: &'swapchain VulkanSwapchain<'device>,
        image_index: u32,
    ) -> Result<Self, VulkanError> {
        let attachment = render_pass.config.attachments.first();
        let swapchain_format = swapchain.format().ok_or(VulkanError::MissingSwapchain)?;
        if !swapchain.belongs_to(render_pass.device)
            || render_pass.config.attachments.len() != 1
            || !attachment.is_some_and(|attachment| {
                is_compatible_swapchain_attachment(
                    attachment,
                    render_pass.attachment_usage[0],
                    swapchain_format,
                )
            })
        {
            return Err(VulkanError::InvalidSwapchainFramebuffer);
        }
        let extent = swapchain.extent().ok_or(VulkanError::MissingSwapchain)?;
        // SAFETY: Physical-device limits belong to the instance retained by the render pass.
        let limits = unsafe {
            render_pass
                .device
                ._instance
                .handle
                .get_physical_device_properties(render_pass.device.physical_device)
                .limits
        };
        if extent.width > limits.max_framebuffer_width
            || extent.height > limits.max_framebuffer_height
        {
            return Err(VulkanError::InvalidSwapchainFramebuffer);
        }
        let view = swapchain.image_view(image_index)?;
        let attachments = [view];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass.handle)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        // SAFETY: The swapchain owns the image view and is borrowed for this framebuffer's
        // lifetime; the compatible render pass is likewise retained by the returned owner.
        let handle = unsafe {
            render_pass
                .device
                .handle
                .create_framebuffer(&info, None)
                .map_err(|error| api_error("create Vulkan swapchain framebuffer", error))?
        };
        Ok(Self {
            device: render_pass.device,
            handle,
            render_pass,
            _swapchain: swapchain,
            image_index,
            extent,
        })
    }

    /// Returns the raw Vulkan framebuffer handle.
    #[must_use]
    pub const fn handle(&self) -> vk::Framebuffer {
        self.handle
    }

    /// Returns the render pass this framebuffer was created for.
    #[must_use]
    pub const fn render_pass(&self) -> &'render_pass VulkanRenderPass<'device> {
        self.render_pass
    }

    pub(crate) fn swapchain_identity(&self) -> Result<(std::sync::Arc<()>, u64), VulkanError> {
        self._swapchain.identity()
    }

    /// Returns the swapchain image index targeted by this framebuffer.
    #[must_use]
    pub const fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Returns the framebuffer extent.
    pub const fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for VulkanSwapchainFramebuffer<'_, '_, '_> {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the framebuffer and retains its render pass/swapchain.
        unsafe { self.device.handle.destroy_framebuffer(self.handle, None) };
    }
}

fn is_compatible_swapchain_attachment(
    attachment: &RenderPassAttachmentConfig,
    usage: vk::ImageUsageFlags,
    swapchain_format: vk::Format,
) -> bool {
    attachment.format == swapchain_format
        && attachment.samples == vk::SampleCountFlags::TYPE_1
        && usage == vk::ImageUsageFlags::COLOR_ATTACHMENT
        && matches!(
            attachment.initial_layout,
            vk::ImageLayout::UNDEFINED | vk::ImageLayout::PRESENT_SRC_KHR
        )
        && attachment.final_layout == vk::ImageLayout::PRESENT_SRC_KHR
}

struct RenderPassMetadata {
    attachment_usage: Vec<vk::ImageUsageFlags>,
    subpass_color_counts: Vec<u32>,
    subpass_sample_counts: Vec<vk::SampleCountFlags>,
    subpass_has_depth: Vec<bool>,
}

fn validate_render_pass_config(
    config: &RenderPassConfig,
) -> Result<RenderPassMetadata, VulkanError> {
    if config.subpasses.is_empty() {
        return Err(VulkanError::InvalidRenderPassConfig);
    }
    let mut attachment_usage = vec![vk::ImageUsageFlags::empty(); config.attachments.len()];
    for attachment in &config.attachments {
        if attachment.format == vk::Format::UNDEFINED
            || !matches!(
                attachment.samples,
                vk::SampleCountFlags::TYPE_1
                    | vk::SampleCountFlags::TYPE_2
                    | vk::SampleCountFlags::TYPE_4
                    | vk::SampleCountFlags::TYPE_8
                    | vk::SampleCountFlags::TYPE_16
                    | vk::SampleCountFlags::TYPE_32
                    | vk::SampleCountFlags::TYPE_64
            )
            || attachment.final_layout == vk::ImageLayout::UNDEFINED
            || attachment.final_layout == vk::ImageLayout::PREINITIALIZED
        {
            return Err(VulkanError::InvalidRenderPassConfig);
        }
    }

    let mut subpass_color_counts = Vec::with_capacity(config.subpasses.len());
    let mut subpass_sample_counts = Vec::with_capacity(config.subpasses.len());
    let mut subpass_has_depth = Vec::with_capacity(config.subpasses.len());
    for subpass in &config.subpasses {
        if !subpass.resolve_attachments.is_empty()
            && subpass.resolve_attachments.len() != subpass.color_attachments.len()
        {
            return Err(VulkanError::InvalidRenderPassConfig);
        }
        let mut sample_count = None;
        let mut used = Vec::new();
        for reference in &subpass.input_attachments {
            validate_attachment_reference(config, *reference, AttachmentRole::Input)?;
            attachment_usage[reference.attachment as usize] |=
                vk::ImageUsageFlags::INPUT_ATTACHMENT;
            used.push(reference.attachment);
        }
        for reference in &subpass.color_attachments {
            validate_attachment_reference(config, *reference, AttachmentRole::Color)?;
            attachment_usage[reference.attachment as usize] |=
                vk::ImageUsageFlags::COLOR_ATTACHMENT;
            let samples = config.attachments[reference.attachment as usize].samples;
            if sample_count.is_some_and(|previous| previous != samples) {
                return Err(VulkanError::InvalidRenderPassConfig);
            }
            sample_count = Some(samples);
            used.push(reference.attachment);
        }
        for (source, resolve) in subpass
            .color_attachments
            .iter()
            .zip(&subpass.resolve_attachments)
        {
            validate_attachment_reference(config, *resolve, AttachmentRole::Resolve)?;
            attachment_usage[resolve.attachment as usize] |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            if config.attachments[source.attachment as usize].samples
                == vk::SampleCountFlags::TYPE_1
                || config.attachments[resolve.attachment as usize].samples
                    != vk::SampleCountFlags::TYPE_1
                || config.attachments[source.attachment as usize].format
                    != config.attachments[resolve.attachment as usize].format
            {
                return Err(VulkanError::InvalidRenderPassConfig);
            }
            used.push(resolve.attachment);
        }
        if let Some(reference) = subpass.depth_stencil_attachment {
            validate_attachment_reference(config, reference, AttachmentRole::DepthStencil)?;
            attachment_usage[reference.attachment as usize] |=
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            let samples = config.attachments[reference.attachment as usize].samples;
            if sample_count.is_some_and(|previous| previous != samples) {
                return Err(VulkanError::InvalidRenderPassConfig);
            }
            sample_count = Some(samples);
            used.push(reference.attachment);
        }
        used.sort_unstable();
        if used.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(VulkanError::InvalidRenderPassConfig);
        }
        let mut preserve = subpass.preserve_attachments.clone();
        preserve.sort_unstable();
        if preserve.iter().any(|attachment| {
            *attachment as usize >= config.attachments.len()
                || used.binary_search(attachment).is_ok()
        }) || preserve.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(VulkanError::InvalidRenderPassConfig);
        }
        subpass_color_counts.push(subpass.color_attachments.len() as u32);
        subpass_sample_counts.push(sample_count.unwrap_or(vk::SampleCountFlags::TYPE_1));
        subpass_has_depth.push(subpass.depth_stencil_attachment.is_some());
    }

    for usage in &attachment_usage {
        if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            && usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Err(VulkanError::InvalidRenderPassConfig);
        }
    }

    for dependency in &config.dependencies {
        if dependency.source_stage_mask.is_empty()
            || dependency.destination_stage_mask.is_empty()
            || (dependency.source_subpass != vk::SUBPASS_EXTERNAL
                && dependency.source_subpass as usize >= config.subpasses.len())
            || (dependency.destination_subpass != vk::SUBPASS_EXTERNAL
                && dependency.destination_subpass as usize >= config.subpasses.len())
        {
            return Err(VulkanError::InvalidRenderPassConfig);
        }
    }

    Ok(RenderPassMetadata {
        attachment_usage,
        subpass_color_counts,
        subpass_sample_counts,
        subpass_has_depth,
    })
}

#[derive(Clone, Copy)]
enum AttachmentRole {
    Input,
    Color,
    Resolve,
    DepthStencil,
}

fn validate_attachment_reference(
    config: &RenderPassConfig,
    reference: RenderPassAttachmentReference,
    role: AttachmentRole,
) -> Result<(), VulkanError> {
    let Some(attachment) = config.attachments.get(reference.attachment as usize) else {
        return Err(VulkanError::InvalidRenderPassConfig);
    };
    let valid_layout = match role {
        AttachmentRole::Input => matches!(
            reference.layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                | vk::ImageLayout::GENERAL
        ),
        AttachmentRole::Color | AttachmentRole::Resolve => matches!(
            reference.layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL | vk::ImageLayout::GENERAL
        ),
        AttachmentRole::DepthStencil => matches!(
            reference.layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                | vk::ImageLayout::GENERAL
        ),
    };
    if !valid_layout || attachment.format == vk::Format::UNDEFINED {
        return Err(VulkanError::InvalidRenderPassConfig);
    }
    Ok(())
}

fn to_vk_ref(reference: RenderPassAttachmentReference) -> vk::AttachmentReference {
    vk::AttachmentReference::default()
        .attachment(reference.attachment)
        .layout(reference.layout)
}

fn to_vk_refs(references: &[RenderPassAttachmentReference]) -> Vec<vk::AttachmentReference> {
    references.iter().copied().map(to_vk_ref).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        RenderPassAttachmentConfig, RenderPassAttachmentReference, RenderPassConfig,
        RenderPassDependency, RenderPassSubpassConfig, validate_render_pass_config,
    };
    use ash::vk;

    fn color_attachment() -> RenderPassAttachmentConfig {
        RenderPassAttachmentConfig {
            format: vk::Format::R8G8B8A8_UNORM,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    fn color_ref() -> RenderPassAttachmentReference {
        RenderPassAttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    #[test]
    fn accepts_single_color_subpass_and_tracks_attachment_usage() {
        let config = RenderPassConfig {
            attachments: vec![color_attachment()],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![color_ref()],
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![],
        };
        let metadata = validate_render_pass_config(&config).expect("valid render pass");
        assert_eq!(
            metadata.attachment_usage[0],
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        );
        assert_eq!(metadata.subpass_color_counts, vec![1]);
        assert_eq!(
            metadata.subpass_sample_counts,
            vec![vk::SampleCountFlags::TYPE_1]
        );
        assert_eq!(metadata.subpass_has_depth, vec![false]);
    }

    #[test]
    fn accepts_matching_multisample_resolve_attachments() {
        let mut multisample = color_attachment();
        multisample.samples = vk::SampleCountFlags::TYPE_4;
        let config = RenderPassConfig {
            attachments: vec![multisample, color_attachment()],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![color_ref()],
                resolve_attachments: vec![RenderPassAttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }],
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![],
        };
        let metadata = validate_render_pass_config(&config).expect("valid resolve configuration");
        assert_eq!(
            metadata.attachment_usage[1],
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        );
    }

    #[test]
    fn rejects_missing_subpasses_bad_references_and_resolve_mismatch() {
        assert!(validate_render_pass_config(&RenderPassConfig::default()).is_err());
        let bad_reference = RenderPassConfig {
            attachments: vec![color_attachment()],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![RenderPassAttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                }],
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![],
        };
        assert!(validate_render_pass_config(&bad_reference).is_err());
        let resolve_mismatch = RenderPassConfig {
            attachments: vec![color_attachment()],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![color_ref()],
                resolve_attachments: vec![color_ref(), color_ref()],
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![],
        };
        assert!(validate_render_pass_config(&resolve_mismatch).is_err());
    }

    #[test]
    fn validates_sample_counts_preserve_attachments_and_dependencies() {
        let mut multisample = color_attachment();
        multisample.samples = vk::SampleCountFlags::TYPE_4;
        let mut depth = color_attachment();
        depth.format = vk::Format::D32_SFLOAT;
        depth.samples = vk::SampleCountFlags::TYPE_1;
        let mismatched_samples = RenderPassConfig {
            attachments: vec![multisample, depth],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![color_ref()],
                depth_stencil_attachment: Some(RenderPassAttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                }),
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![],
        };
        assert!(validate_render_pass_config(&mismatched_samples).is_err());

        let bad_dependency = RenderPassConfig {
            attachments: vec![color_attachment()],
            subpasses: vec![RenderPassSubpassConfig {
                color_attachments: vec![color_ref()],
                ..RenderPassSubpassConfig::default()
            }],
            dependencies: vec![RenderPassDependency {
                source_subpass: 3,
                destination_subpass: 0,
                source_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                destination_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                source_access_mask: vk::AccessFlags::empty(),
                destination_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                flags: vk::DependencyFlags::empty(),
            }],
        };
        assert!(validate_render_pass_config(&bad_dependency).is_err());
    }
}

#[cfg(test)]
mod swapchain_framebuffer_tests {
    use super::{RenderPassAttachmentConfig, is_compatible_swapchain_attachment};
    use ash::vk;

    fn attachment() -> RenderPassAttachmentConfig {
        RenderPassAttachmentConfig {
            format: vk::Format::B8G8R8A8_SRGB,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    #[test]
    fn accepts_single_sample_color_attachment_with_present_final_layout() {
        assert!(is_compatible_swapchain_attachment(
            &attachment(),
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::Format::B8G8R8A8_SRGB
        ));
    }

    #[test]
    fn rejects_wrong_format_samples_usage_and_final_layout() {
        let mut config = attachment();
        config.format = vk::Format::R8G8B8A8_UNORM;
        assert!(!is_compatible_swapchain_attachment(
            &config,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::Format::B8G8R8A8_SRGB
        ));
        config = attachment();
        config.samples = vk::SampleCountFlags::TYPE_4;
        assert!(!is_compatible_swapchain_attachment(
            &config,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::Format::B8G8R8A8_SRGB
        ));
        config = attachment();
        config.final_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        assert!(!is_compatible_swapchain_attachment(
            &config,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::Format::B8G8R8A8_SRGB
        ));
        config = attachment();
        assert!(!is_compatible_swapchain_attachment(
            &config,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Format::B8G8R8A8_SRGB
        ));
    }

    #[test]
    fn accepts_present_layout_as_the_next_frame_initial_layout() {
        let mut config = attachment();
        config.initial_layout = vk::ImageLayout::PRESENT_SRC_KHR;
        assert!(is_compatible_swapchain_attachment(
            &config,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::Format::B8G8R8A8_SRGB
        ));
    }
}
