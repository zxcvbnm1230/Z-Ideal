//! Owned Vulkan command pools, command-buffer recording, and queue submissions.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use ash::vk;

use crate::{
    DescriptorSetLeaseRef, FenceConfig, FenceWaitResult, VulkanBuffer, VulkanComputePipeline,
    VulkanDevice, VulkanError, VulkanFence, VulkanFramebuffer, VulkanGraphicsPipeline, VulkanImage,
    VulkanPipelineLayout, VulkanRenderPass, VulkanSemaphore, VulkanSwapchainFramebuffer, api_error,
};

trait CommandResource {}
impl CommandResource for VulkanRenderPass<'_> {}
impl CommandResource for VulkanFramebuffer<'_, '_, '_, '_> {}
impl CommandResource for VulkanSwapchainFramebuffer<'_, '_, '_> {}
impl CommandResource for VulkanGraphicsPipeline<'_, '_, '_> {}
impl CommandResource for VulkanComputePipeline<'_, '_, '_> {}
impl CommandResource for VulkanPipelineLayout<'_, '_> {}
impl CommandResource for VulkanBuffer<'_> {}
impl CommandResource for VulkanImage<'_> {}

/// Buffer memory dependency recorded into a command buffer.
pub struct BufferBarrier<'buffer, 'device> {
    /// Buffer whose accesses are synchronized.
    pub buffer: &'buffer VulkanBuffer<'device>,
    /// Accesses made available before the barrier.
    pub source_access: vk::AccessFlags,
    /// Accesses made visible after the barrier.
    pub destination_access: vk::AccessFlags,
    /// Byte offset in the buffer.
    pub offset: vk::DeviceSize,
    /// Byte range; `vk::WHOLE_SIZE` selects the remaining buffer.
    pub size: vk::DeviceSize,
}

/// Image layout and memory dependency recorded into a command buffer.
pub struct ImageBarrier<'image, 'device> {
    /// Image whose layout/accesses are synchronized.
    pub image: &'image VulkanImage<'device>,
    /// Previous image layout, or `UNDEFINED` when prior contents are discarded.
    pub old_layout: vk::ImageLayout,
    /// Layout required after the barrier.
    pub new_layout: vk::ImageLayout,
    /// Accesses made available before the barrier.
    pub source_access: vk::AccessFlags,
    /// Accesses made visible after the barrier.
    pub destination_access: vk::AccessFlags,
    /// Image aspect mask.
    pub aspect: vk::ImageAspectFlags,
    /// First mip level covered by the barrier.
    pub base_mip_level: u32,
    /// Number of mip levels covered by the barrier.
    pub level_count: u32,
    /// First array layer covered by the barrier.
    pub base_array_layer: u32,
    /// Number of array layers covered by the barrier.
    pub layer_count: u32,
}

/// Command-pool creation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandPoolConfig {
    /// Vulkan command-pool creation flags. Individual command-buffer reset is required.
    pub flags: vk::CommandPoolCreateFlags,
}

impl Default for CommandPoolConfig {
    fn default() -> Self {
        Self {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        }
    }
}

/// Owns a command pool for the device's graphics queue family.
pub struct VulkanCommandPool<'device> {
    device: &'device VulkanDevice,
    handle: vk::CommandPool,
    queue_family: u32,
    queue_flags: vk::QueueFlags,
}

impl<'device> VulkanCommandPool<'device> {
    /// Creates a resettable command pool for the selected graphics queue family.
    pub fn new(
        device: &'device VulkanDevice,
        config: CommandPoolConfig,
    ) -> Result<Self, VulkanError> {
        if !validate_command_pool_flags(config.flags) {
            return Err(VulkanError::InvalidCommandPoolConfig);
        }
        // SAFETY: Physical-device metadata belongs to the instance retained by the device.
        let queue_families = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_queue_family_properties(device.physical_device)
        };
        let family = queue_families
            .get(device.graphics_queue_family as usize)
            .ok_or(VulkanError::InvalidCommandPoolConfig)?;
        if family.queue_count == 0 || !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            return Err(VulkanError::InvalidCommandPoolConfig);
        }
        let info = vk::CommandPoolCreateInfo::default()
            .flags(config.flags)
            .queue_family_index(device.graphics_queue_family);
        // SAFETY: The queue family belongs to the retained physical device and is enabled.
        let handle = unsafe {
            device
                .handle
                .create_command_pool(&info, None)
                .map_err(|error| api_error("create Vulkan command pool", error))?
        };
        Ok(Self {
            device,
            handle,
            queue_family: device.graphics_queue_family,
            queue_flags: family.queue_flags,
        })
    }

    /// Allocates one primary command buffer from this pool.
    pub fn allocate_primary(&self) -> Result<VulkanCommandBuffer<'_, 'device>, VulkanError> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: The pool is live and the allocation requests one primary command buffer.
        let mut buffers = unsafe {
            self.device
                .handle
                .allocate_command_buffers(&info)
                .map_err(|error| api_error("allocate Vulkan command buffer", error))?
        };
        let handle = buffers.pop().ok_or(VulkanError::Api {
            operation: "allocate Vulkan command buffer",
            result: vk::Result::ERROR_UNKNOWN,
        })?;
        Ok(VulkanCommandBuffer {
            pool: self,
            handle,
            state: CommandBufferState::Initial,
            current_render_pass: None,
            current_subpass: 0,
            graphics_pipeline: None,
            compute_pipeline_layout: None,
            viewport_set: false,
            scissor_set: false,
            index_buffer_bound: false,
        })
    }

    /// Returns the raw Vulkan command-pool handle.
    #[must_use]
    pub const fn handle(&self) -> vk::CommandPool {
        self.handle
    }

    /// Returns the queue-family index this pool submits to.
    #[must_use]
    pub const fn queue_family(&self) -> u32 {
        self.queue_family
    }
}

impl Drop for VulkanCommandPool<'_> {
    fn drop(&mut self) {
        // SAFETY: Every allocated command-buffer wrapper borrows this pool, and submitted
        // command buffers remain borrowed by their fence-backed completion guards.
        unsafe {
            self.device.handle.destroy_command_pool(self.handle, None);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandBufferState {
    Initial,
    Recording,
    Executable,
    Pending,
}

/// Primary command buffer allocated from a borrowed command pool.
pub struct VulkanCommandBuffer<'pool, 'device> {
    pool: &'pool VulkanCommandPool<'device>,
    handle: vk::CommandBuffer,
    state: CommandBufferState,
    current_render_pass: Option<vk::RenderPass>,
    current_subpass: u32,
    graphics_pipeline: Option<(vk::PipelineLayout, vk::RenderPass, u32)>,
    compute_pipeline_layout: Option<vk::PipelineLayout>,
    viewport_set: bool,
    scissor_set: bool,
    index_buffer_bound: bool,
}

impl<'pool, 'device> VulkanCommandBuffer<'pool, 'device> {
    /// Begins recording and returns a guard that retains every referenced resource and descriptor lease through submission completion.
    pub fn begin<'recording>(
        &'recording mut self,
        flags: vk::CommandBufferUsageFlags,
    ) -> Result<CommandBufferRecording<'recording, 'pool, 'device>, VulkanError> {
        match self.state {
            CommandBufferState::Pending | CommandBufferState::Recording => {
                return Err(VulkanError::InvalidCommandBufferState);
            }
            CommandBufferState::Executable => self.reset_internal()?,
            CommandBufferState::Initial => {}
        }
        let info = vk::CommandBufferBeginInfo::default().flags(flags);
        // SAFETY: The command buffer is initial and belongs to the retained pool/device.
        unsafe {
            self.pool
                .device
                .handle
                .begin_command_buffer(self.handle, &info)
                .map_err(|error| api_error("begin Vulkan command buffer", error))?;
        }
        self.state = CommandBufferState::Recording;
        self.clear_recording_state();
        Ok(CommandBufferRecording {
            command_buffer: Some(self),
            descriptor_leases: Vec::new(),
            resources: Vec::new(),
            swapchain_target: None,
        })
    }

    /// Returns the raw Vulkan command-buffer handle.
    #[must_use]
    pub const fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    fn require_recording(&self) -> Result<(), VulkanError> {
        if self.state == CommandBufferState::Recording {
            Ok(())
        } else {
            Err(VulkanError::InvalidCommandBufferState)
        }
    }

    fn clear_recording_state(&mut self) {
        self.current_render_pass = None;
        self.current_subpass = 0;
        self.graphics_pipeline = None;
        self.compute_pipeline_layout = None;
        self.viewport_set = false;
        self.scissor_set = false;
        self.index_buffer_bound = false;
    }

    fn reset_internal(&mut self) -> Result<(), VulkanError> {
        if matches!(
            self.state,
            CommandBufferState::Recording | CommandBufferState::Pending
        ) {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command pool was created with RESET_COMMAND_BUFFER and the buffer is not pending.
        unsafe {
            self.pool
                .device
                .handle
                .reset_command_buffer(self.handle, vk::CommandBufferResetFlags::empty())
                .map_err(|error| api_error("reset Vulkan command buffer", error))?;
        }
        self.state = CommandBufferState::Initial;
        self.clear_recording_state();
        Ok(())
    }

    /// Abandons an unfinished recording scope and resets the command buffer.
    ///
    /// The recording guard also invokes this automatically when dropped unfinished.
    pub fn abort_recording(&mut self) -> Result<(), VulkanError> {
        if self.state != CommandBufferState::Recording {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is recording; close its active render pass, end it, then reset.
        unsafe {
            if self.current_render_pass.is_some() {
                self.pool.device.handle.cmd_end_render_pass(self.handle);
            }
            self.pool
                .device
                .handle
                .end_command_buffer(self.handle)
                .map_err(|error| api_error("abort Vulkan command buffer recording", error))?;
            self.pool
                .device
                .handle
                .reset_command_buffer(self.handle, vk::CommandBufferResetFlags::empty())
                .map_err(|error| api_error("reset aborted Vulkan command buffer", error))?;
        }
        self.state = CommandBufferState::Initial;
        self.clear_recording_state();
        Ok(())
    }

    fn end_internal(&mut self) -> Result<(), VulkanError> {
        self.require_recording()?;
        if self.current_render_pass.is_some() {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is recording and no render pass remains open.
        unsafe {
            self.pool
                .device
                .handle
                .end_command_buffer(self.handle)
                .map_err(|error| api_error("end Vulkan command buffer", error))?;
        }
        self.state = CommandBufferState::Executable;
        Ok(())
    }
}

/// Active recording scope. Dropping it without `end` safely aborts and resets the buffer.
pub struct CommandBufferRecording<'recording, 'pool, 'device> {
    command_buffer: Option<&'recording mut VulkanCommandBuffer<'pool, 'device>>,
    descriptor_leases: Vec<DescriptorSetLeaseRef<'recording>>,
    resources: Vec<&'recording dyn CommandResource>,
    swapchain_target: Option<(Arc<()>, u64, u32)>,
}

impl<'recording, 'pool, 'device> CommandBufferRecording<'recording, 'pool, 'device> {
    fn buffer_mut(&mut self) -> &mut VulkanCommandBuffer<'pool, 'device> {
        self.command_buffer
            .as_deref_mut()
            .expect("recording guard owns its command buffer")
    }

    /// Ends recording and returns an executable command buffer retaining all bound descriptor leases.
    pub fn end(
        mut self,
    ) -> Result<ExecutableCommandBuffer<'recording, 'pool, 'device>, VulkanError> {
        let command_buffer = self
            .command_buffer
            .take()
            .expect("recording guard owns its command buffer until end");
        if let Err(error) = command_buffer.end_internal() {
            let _ = command_buffer.abort_recording();
            return Err(error);
        }
        Ok(ExecutableCommandBuffer {
            command_buffer,
            _descriptor_leases: std::mem::take(&mut self.descriptor_leases),
            _resources: std::mem::take(&mut self.resources),
            swapchain_target: self.swapchain_target.take(),
        })
    }

    /// Begins a render pass with one clear value per render-pass attachment.
    pub fn begin_render_pass(
        &mut self,
        render_pass: &'recording VulkanRenderPass<'device>,
        framebuffer: &'recording VulkanFramebuffer<'_, '_, '_, 'device>,
        clear_values: &[vk::ClearValue],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_some()
            || !std::ptr::eq(self.buffer_mut().pool.device, render_pass.device)
            || !std::ptr::eq(
                self.buffer_mut().pool.device,
                framebuffer.render_pass().device,
            )
            || framebuffer.render_pass().handle() != render_pass.handle()
            || clear_values.len() != render_pass.attachment_count()
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: framebuffer.extent(),
        };
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(framebuffer.handle())
            .render_area(area)
            .clear_values(clear_values);
        // SAFETY: The render pass/framebuffer are compatible and retained by the caller;
        // the command-buffer recording guard's borrow prevents them being dropped here.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_begin_render_pass(
                self.buffer_mut().handle,
                &info,
                vk::SubpassContents::INLINE,
            );
        }
        self.buffer_mut().current_render_pass = Some(render_pass.handle());
        self.buffer_mut().current_subpass = 0;
        self.buffer_mut().graphics_pipeline = None;
        self.buffer_mut().viewport_set = false;
        self.buffer_mut().scissor_set = false;
        self.resources.push(render_pass);
        self.resources.push(framebuffer);
        Ok(())
    }

    /// Begins a render pass using one image from the active swapchain.
    pub fn begin_swapchain_render_pass(
        &mut self,
        render_pass: &'recording VulkanRenderPass<'device>,
        framebuffer: &'recording VulkanSwapchainFramebuffer<'_, '_, 'device>,
        clear_values: &[vk::ClearValue],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_some()
            || !std::ptr::eq(self.buffer_mut().pool.device, render_pass.device)
            || !std::ptr::eq(
                self.buffer_mut().pool.device,
                framebuffer.render_pass().device,
            )
            || framebuffer.render_pass().handle() != render_pass.handle()
            || clear_values.len() != render_pass.attachment_count()
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        let identity = framebuffer.swapchain_identity()?;
        let target = (identity.0, identity.1, framebuffer.image_index());
        if let Some(existing) = &self.swapchain_target {
            if !Arc::ptr_eq(&existing.0, &target.0)
                || existing.1 != target.1
                || existing.2 != target.2
            {
                return Err(VulkanError::InvalidSwapchainFramebuffer);
            }
        } else {
            self.swapchain_target = Some(target);
        }
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: framebuffer.extent(),
        };
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(framebuffer.handle())
            .render_area(area)
            .clear_values(clear_values);
        // SAFETY: The framebuffer retains the compatible swapchain view and render pass
        // for the recording borrow; the command buffer is in its recording state.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_begin_render_pass(
                self.buffer_mut().handle,
                &info,
                vk::SubpassContents::INLINE,
            );
        }
        self.buffer_mut().current_render_pass = Some(render_pass.handle());
        self.buffer_mut().current_subpass = 0;
        self.buffer_mut().graphics_pipeline = None;
        self.buffer_mut().viewport_set = false;
        self.buffer_mut().scissor_set = false;
        self.resources.push(render_pass);
        self.resources.push(framebuffer);
        Ok(())
    }

    /// Advances to the next subpass in the active render pass.
    pub fn next_subpass(
        &mut self,
        render_pass: &'recording VulkanRenderPass<'device>,
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if !std::ptr::eq(self.buffer_mut().pool.device, render_pass.device)
            || self.buffer_mut().current_render_pass != Some(render_pass.handle())
            || self.buffer_mut().current_subpass + 1 >= render_pass.subpass_count() as u32
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is inside the specified render pass and a next subpass exists.
        unsafe {
            self.buffer_mut()
                .pool
                .device
                .handle
                .cmd_next_subpass(self.buffer_mut().handle, vk::SubpassContents::INLINE);
        }
        self.buffer_mut().current_subpass += 1;
        self.buffer_mut().graphics_pipeline = None;
        self.buffer_mut().viewport_set = false;
        self.buffer_mut().scissor_set = false;
        self.resources.push(render_pass);
        Ok(())
    }

    /// Ends the active render pass.
    pub fn end_render_pass(&mut self) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_none() {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is inside a render pass.
        unsafe {
            self.buffer_mut()
                .pool
                .device
                .handle
                .cmd_end_render_pass(self.buffer_mut().handle);
        }
        self.buffer_mut().current_render_pass = None;
        self.buffer_mut().current_subpass = 0;
        self.buffer_mut().graphics_pipeline = None;
        self.buffer_mut().viewport_set = false;
        self.buffer_mut().scissor_set = false;
        Ok(())
    }

    /// Binds a graphics pipeline compatible with the active render-pass subpass.
    pub fn bind_graphics_pipeline(
        &mut self,
        pipeline: &'recording VulkanGraphicsPipeline<'_, '_, 'device>,
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if !std::ptr::eq(self.buffer_mut().pool.device, pipeline.device)
            || self.buffer_mut().current_render_pass != Some(pipeline.render_pass)
            || self.buffer_mut().current_subpass != pipeline.subpass
        {
            return Err(VulkanError::PipelineShaderStageMismatch);
        }
        // SAFETY: Pipeline and device are compatible and pipeline is retained by the caller.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_bind_pipeline(
                self.buffer_mut().handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            );
        }
        self.buffer_mut().graphics_pipeline = Some((
            pipeline.layout_handle,
            pipeline.render_pass,
            pipeline.subpass,
        ));
        self.buffer_mut().viewport_set = false;
        self.buffer_mut().scissor_set = false;
        self.resources.push(pipeline);
        Ok(())
    }

    /// Binds a compute pipeline outside a render pass.
    pub fn bind_compute_pipeline(
        &mut self,
        pipeline: &'recording VulkanComputePipeline<'_, '_, 'device>,
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_some()
            || !std::ptr::eq(self.buffer_mut().pool.device, pipeline.device)
            || !self
                .buffer_mut()
                .pool
                .queue_flags
                .contains(vk::QueueFlags::COMPUTE)
        {
            return Err(VulkanError::PipelineShaderStageMismatch);
        }
        // SAFETY: Pipeline and device are compatible and pipeline is retained by the caller.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_bind_pipeline(
                self.buffer_mut().handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle(),
            );
        }
        self.buffer_mut().compute_pipeline_layout = Some(pipeline.layout_handle);
        self.resources.push(pipeline);
        Ok(())
    }

    /// Sets the dynamic viewport required by the initial graphics pipeline configuration.
    pub fn set_viewport(&mut self, viewport: vk::Viewport) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if !viewport.x.is_finite()
            || !viewport.y.is_finite()
            || !viewport.width.is_finite()
            || !viewport.height.is_finite()
            || !viewport.min_depth.is_finite()
            || !viewport.max_depth.is_finite()
            || viewport.width <= 0.0
            || viewport.height <= 0.0
            || viewport.min_depth < 0.0
            || viewport.max_depth > 1.0
            || viewport.min_depth > viewport.max_depth
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is recording and viewport values were validated.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_set_viewport(
                self.buffer_mut().handle,
                0,
                &[viewport],
            );
        }
        self.buffer_mut().viewport_set = true;
        Ok(())
    }

    /// Sets the dynamic scissor required by the initial graphics pipeline configuration.
    pub fn set_scissor(&mut self, scissor: vk::Rect2D) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if scissor.extent.width == 0 || scissor.extent.height == 0 {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is recording and scissor extent is nonzero.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_set_scissor(
                self.buffer_mut().handle,
                0,
                &[scissor],
            );
        }
        self.buffer_mut().scissor_set = true;
        Ok(())
    }

    /// Binds vertex buffers after checking device identity, usage, and offsets.
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[&'recording VulkanBuffer<'device>],
        offsets: &[vk::DeviceSize],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        let binding_count =
            u32::try_from(buffers.len()).map_err(|_| VulkanError::InvalidCommandBufferState)?;
        if buffers.is_empty()
            || buffers.len() != offsets.len()
            || first_binding.checked_add(binding_count).is_none()
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        for (buffer, offset) in buffers.iter().zip(offsets) {
            if !std::ptr::eq(self.buffer_mut().pool.device, buffer.device)
                || !buffer.usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER)
                || *offset >= buffer.size()
            {
                return Err(VulkanError::InvalidCommandBufferState);
            }
        }
        let handles: Vec<_> = buffers.iter().map(|buffer| buffer.handle()).collect();
        // SAFETY: Buffers are live, device-compatible, and remain borrowed for this call.
        unsafe {
            self.buffer_mut()
                .pool
                .device
                .handle
                .cmd_bind_vertex_buffers(
                    self.buffer_mut().handle,
                    first_binding,
                    &handles,
                    offsets,
                );
        }
        self.resources
            .extend(buffers.iter().map(|buffer| *buffer as &dyn CommandResource));
        Ok(())
    }

    /// Binds an index buffer after checking device identity, usage, offset, and index type.
    pub fn bind_index_buffer(
        &mut self,
        buffer: &'recording VulkanBuffer<'device>,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if !std::ptr::eq(self.buffer_mut().pool.device, buffer.device)
            || !buffer.usage.contains(vk::BufferUsageFlags::INDEX_BUFFER)
            || offset >= buffer.size()
            || !matches!(index_type, vk::IndexType::UINT16 | vk::IndexType::UINT32)
            || (index_type == vk::IndexType::UINT16 && offset % 2 != 0)
            || (index_type == vk::IndexType::UINT32 && offset % 4 != 0)
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The buffer is live, device-compatible, and has index-buffer usage.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_bind_index_buffer(
                self.buffer_mut().handle,
                buffer.handle(),
                offset,
                index_type,
            );
        }
        self.buffer_mut().index_buffer_bound = true;
        self.resources.push(buffer);
        Ok(())
    }

    /// Binds descriptor-set leases and validates dynamic offsets and pipeline-layout compatibility.
    pub fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: &'recording VulkanPipelineLayout<'_, 'device>,
        first_set: u32,
        leases: &[DescriptorSetLeaseRef<'recording>],
        dynamic_offsets: &[u32],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        let lease_count =
            u32::try_from(leases.len()).map_err(|_| VulkanError::InvalidCommandBufferState)?;
        if !std::ptr::eq(self.buffer_mut().pool.device, layout.device)
            || leases.is_empty()
            || first_set
                .checked_add(lease_count)
                .is_none_or(|end| end as usize > layout.set_layouts().len())
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        let expected_layout = match bind_point {
            vk::PipelineBindPoint::GRAPHICS if self.buffer_mut().current_render_pass.is_some() => {
                self.buffer_mut()
                    .graphics_pipeline
                    .map(|pipeline| pipeline.0)
            }
            vk::PipelineBindPoint::COMPUTE if self.buffer_mut().current_render_pass.is_none() => {
                self.buffer_mut().compute_pipeline_layout
            }
            _ => None,
        };
        if expected_layout != Some(layout.handle()) {
            return Err(VulkanError::InvalidCommandBufferState);
        }

        let mut required_dynamic_count = 0_usize;
        for (index, lease) in leases.iter().enumerate() {
            let set_index = first_set as usize + index;
            let set_layout = layout
                .set_layouts()
                .get(set_index)
                .ok_or(VulkanError::InvalidCommandBufferState)?;
            if !std::ptr::eq(self.buffer_mut().pool.device, lease.device)
                || set_layout.handle() != lease.layout_handle
            {
                return Err(VulkanError::InvalidCommandBufferState);
            }
            let mut dynamic_bindings: Vec<_> = set_layout
                .config()
                .bindings
                .iter()
                .filter(|binding| {
                    matches!(
                        binding.descriptor_type,
                        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                            | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    )
                })
                .collect();
            dynamic_bindings.sort_unstable_by_key(|binding| binding.binding);
            let expected_slots: Vec<_> = dynamic_bindings
                .iter()
                .flat_map(|binding| {
                    (0..binding.descriptor_count).map(move |array_element| {
                        (binding.binding, array_element, binding.descriptor_type)
                    })
                })
                .collect();
            if lease.dynamic_offsets.len() != expected_slots.len() {
                return Err(VulkanError::InvalidDynamicDescriptorOffset);
            }
            for (requirement, (binding, array_element, descriptor_type)) in
                lease.dynamic_offsets.iter().zip(expected_slots)
            {
                if requirement.binding != binding
                    || requirement.array_element != array_element
                    || requirement.descriptor_type != descriptor_type
                {
                    return Err(VulkanError::InvalidDynamicDescriptorOffset);
                }
                let offset = *dynamic_offsets
                    .get(required_dynamic_count)
                    .ok_or(VulkanError::InvalidDynamicDescriptorOffset)?;
                let (uniform_alignment, storage_alignment) = (
                    self.buffer_mut()
                        .pool
                        .device
                        .min_uniform_buffer_offset_alignment,
                    self.buffer_mut()
                        .pool
                        .device
                        .min_storage_buffer_offset_alignment,
                );
                validate_dynamic_offset_requirement(
                    requirement,
                    offset,
                    uniform_alignment,
                    storage_alignment,
                )?;
                required_dynamic_count += 1;
            }
        }
        if required_dynamic_count != dynamic_offsets.len() {
            return Err(VulkanError::InvalidDynamicDescriptorOffset);
        }

        let handles: Vec<_> = leases.iter().map(|lease| lease.handle).collect();
        // SAFETY: Every descriptor set is retained by a lease stored in this recording
        // guard and has been checked against the pipeline layout and dynamic offsets.
        unsafe {
            self.buffer_mut()
                .pool
                .device
                .handle
                .cmd_bind_descriptor_sets(
                    self.buffer_mut().handle,
                    bind_point,
                    layout.handle(),
                    first_set,
                    &handles,
                    dynamic_offsets,
                );
        }
        self.descriptor_leases.extend_from_slice(leases);
        self.resources.push(layout);
        Ok(())
    }

    /// Copies one or more aligned ranges between distinct transfer-capable buffers.
    ///
    /// The source and destination must have `TRANSFER_SRC` and `TRANSFER_DST` usage respectively.
    /// This records no memory barriers; callers establish producer/consumer dependencies explicitly.
    pub fn copy_buffer(
        &mut self,
        source: &'recording VulkanBuffer<'device>,
        destination: &'recording VulkanBuffer<'device>,
        regions: &[vk::BufferCopy],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_some()
            || !std::ptr::eq(source.device, self.buffer_mut().pool.device)
            || !std::ptr::eq(destination.device, self.buffer_mut().pool.device)
            || source.handle() == destination.handle()
            || !source.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC)
            || !destination
                .usage
                .contains(vk::BufferUsageFlags::TRANSFER_DST)
            || regions.is_empty()
            || regions.iter().any(|region| {
                !validate_buffer_copy_region(region, source.size(), destination.size())
            })
        {
            return Err(VulkanError::InvalidCopyRegion);
        }
        let destination_ranges: Vec<_> = regions
            .iter()
            .map(|region| (region.dst_offset, region.dst_offset + region.size))
            .collect();
        if ranges_overlap(&destination_ranges) {
            return Err(VulkanError::InvalidCopyRegion);
        }
        // SAFETY: Buffer usage, device identity, offsets, sizes, and destination overlap
        // were validated; both resources are retained until submitted work completes.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_copy_buffer(
                self.buffer_mut().handle,
                source.handle(),
                destination.handle(),
                regions,
            );
        }
        self.resources.push(source);
        self.resources.push(destination);
        Ok(())
    }

    /// Copies buffer data into an image in `TRANSFER_DST_OPTIMAL` or `GENERAL` layout.
    ///
    /// Supports explicitly enumerated uncompressed color, BC/ETC2/EAC/ASTC compressed,
    /// and depth/stencil formats. Compressed copies enforce texel-block alignment and
    /// edge rules; depth/stencil copies select exactly one aspect per region.
    pub fn copy_buffer_to_image(
        &mut self,
        source: &'recording VulkanBuffer<'device>,
        destination: &'recording VulkanImage<'device>,
        destination_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_some()
            || !std::ptr::eq(source.device, self.buffer_mut().pool.device)
            || !std::ptr::eq(destination.device, self.buffer_mut().pool.device)
            || !source.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC)
            || !destination
                .config()
                .usage
                .contains(vk::ImageUsageFlags::TRANSFER_DST)
            || !is_transfer_destination_layout(destination_layout)
            || regions.is_empty()
        {
            return Err(VulkanError::InvalidCopyRegion);
        }
        for region in regions {
            validate_buffer_image_copy_region(region, source.size(), destination.config())?;
        }
        // SAFETY: Buffer/image usage, layouts, format-block/aspect support, subresources, extents,
        // and required buffer byte ranges were validated; both resources are retained.
        unsafe {
            self.buffer_mut()
                .pool
                .device
                .handle
                .cmd_copy_buffer_to_image(
                    self.buffer_mut().handle,
                    source.handle(),
                    destination.handle(),
                    destination_layout,
                    regions,
                );
        }
        self.resources.push(source);
        self.resources.push(destination);
        Ok(())
    }

    /// Copies an image into a buffer in `TRANSFER_SRC_OPTIMAL` or `GENERAL` layout.
    ///
    /// Supports the same explicit format/aspect set as [`Self::copy_buffer_to_image`].
    pub fn copy_image_to_buffer(
        &mut self,
        source: &'recording VulkanImage<'device>,
        source_layout: vk::ImageLayout,
        destination: &'recording VulkanBuffer<'device>,
        regions: &[vk::BufferImageCopy],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_some()
            || !std::ptr::eq(source.device, self.buffer_mut().pool.device)
            || !std::ptr::eq(destination.device, self.buffer_mut().pool.device)
            || !source
                .config()
                .usage
                .contains(vk::ImageUsageFlags::TRANSFER_SRC)
            || !destination
                .usage
                .contains(vk::BufferUsageFlags::TRANSFER_DST)
            || !is_transfer_source_layout(source_layout)
            || regions.is_empty()
        {
            return Err(VulkanError::InvalidCopyRegion);
        }
        for region in regions {
            validate_buffer_image_copy_region(region, destination.size(), source.config())?;
        }
        // SAFETY: Buffer/image usage, layouts, format-block/aspect support, subresources, extents,
        // and required buffer byte ranges were validated; both resources are retained.
        unsafe {
            self.buffer_mut()
                .pool
                .device
                .handle
                .cmd_copy_image_to_buffer(
                    self.buffer_mut().handle,
                    source.handle(),
                    source_layout,
                    destination.handle(),
                    regions,
                );
        }
        self.resources.push(source);
        self.resources.push(destination);
        Ok(())
    }

    /// Copies matching uncompressed color regions between distinct images.
    ///
    /// Both images must have the same format and sample count, and their layouts must be
    /// `TRANSFER_SRC_OPTIMAL`/`TRANSFER_DST_OPTIMAL` or `GENERAL` as appropriate.
    pub fn copy_image(
        &mut self,
        source: &'recording VulkanImage<'device>,
        source_layout: vk::ImageLayout,
        destination: &'recording VulkanImage<'device>,
        destination_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        let source_config = source.config();
        let destination_config = destination.config();
        if format_bytes_per_texel(source_config.format).is_none() {
            return Err(VulkanError::UnsupportedCopyFormat);
        }
        if self.buffer_mut().current_render_pass.is_some()
            || !std::ptr::eq(source.device, self.buffer_mut().pool.device)
            || !std::ptr::eq(destination.device, self.buffer_mut().pool.device)
            || source.handle() == destination.handle()
            || !source_config
                .usage
                .contains(vk::ImageUsageFlags::TRANSFER_SRC)
            || !destination_config
                .usage
                .contains(vk::ImageUsageFlags::TRANSFER_DST)
            || source_config.format != destination_config.format
            || source_config.samples != destination_config.samples
            || source_config.image_type != destination_config.image_type
            || !is_transfer_source_layout(source_layout)
            || !is_transfer_destination_layout(destination_layout)
            || regions.is_empty()
            || regions.iter().any(|region| {
                !validate_image_copy_region(region, source_config, destination_config)
            })
        {
            return Err(VulkanError::InvalidCopyRegion);
        }
        // SAFETY: Image usage, format/sample compatibility, layouts, subresources and extents
        // were validated; both images are retained until submitted work completes.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_copy_image(
                self.buffer_mut().handle,
                source.handle(),
                source_layout,
                destination.handle(),
                destination_layout,
                regions,
            );
        }
        self.resources.push(source);
        self.resources.push(destination);
        Ok(())
    }

    /// Records buffer/image memory barriers and image layout transitions.
    ///
    /// Queue-family ownership transfers are not exposed yet; all barriers use
    /// `QUEUE_FAMILY_IGNORED` because this initial queue wrapper submits to one family.
    pub fn pipeline_barrier(
        &mut self,
        source_stages: vk::PipelineStageFlags,
        destination_stages: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        buffers: &[BufferBarrier<'recording, 'device>],
        images: &[ImageBarrier<'recording, 'device>],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if source_stages.is_empty()
            || destination_stages.is_empty()
            || (buffers.is_empty() && images.is_empty())
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        let mut raw_buffers = Vec::with_capacity(buffers.len());
        for barrier in buffers {
            validate_buffer_barrier(barrier, self.buffer_mut().pool.device)?;
            raw_buffers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(barrier.source_access)
                    .dst_access_mask(barrier.destination_access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(barrier.buffer.handle())
                    .offset(barrier.offset)
                    .size(if barrier.size == vk::WHOLE_SIZE {
                        barrier.buffer.size() - barrier.offset
                    } else {
                        barrier.size
                    }),
            );
        }
        let mut raw_images = Vec::with_capacity(images.len());
        for barrier in images {
            validate_image_barrier(barrier, self.buffer_mut().pool.device)?;
            raw_images.push(
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(barrier.source_access)
                    .dst_access_mask(barrier.destination_access)
                    .old_layout(barrier.old_layout)
                    .new_layout(barrier.new_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(barrier.image.handle())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(barrier.aspect)
                            .base_mip_level(barrier.base_mip_level)
                            .level_count(barrier.level_count)
                            .base_array_layer(barrier.base_array_layer)
                            .layer_count(barrier.layer_count),
                    ),
            );
        }
        // SAFETY: Every resource is device-compatible and retained by this recording guard;
        // all subresource/range bounds and stage masks were validated above.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_pipeline_barrier(
                self.buffer_mut().handle,
                source_stages,
                destination_stages,
                dependency_flags,
                &[],
                &raw_buffers,
                &raw_images,
            );
        }
        self.resources.extend(
            buffers
                .iter()
                .map(|barrier| barrier.buffer as &dyn CommandResource),
        );
        self.resources.extend(
            images
                .iter()
                .map(|barrier| barrier.image as &dyn CommandResource),
        );
        Ok(())
    }

    /// Pushes bytes to a validated push-constant range for the active pipeline layout.
    pub fn push_constants(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: &'recording VulkanPipelineLayout<'_, 'device>,
        stages: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        let expected_layout = match bind_point {
            vk::PipelineBindPoint::GRAPHICS => self.buffer_mut().graphics_pipeline.map(|p| p.0),
            vk::PipelineBindPoint::COMPUTE => self.buffer_mut().compute_pipeline_layout,
            _ => None,
        };
        let size = u32::try_from(data.len()).map_err(|_| VulkanError::InvalidPushConstantWrite)?;
        let end = offset
            .checked_add(size)
            .ok_or(VulkanError::InvalidPushConstantWrite)?;
        if !std::ptr::eq(self.buffer_mut().pool.device, layout.device)
            || expected_layout != Some(layout.handle())
            || stages.is_empty()
            || offset % 4 != 0
            || size == 0
            || size % 4 != 0
            || !push_constant_range_covers(layout.push_constant_ranges(), stages, offset, end)
        {
            return Err(VulkanError::InvalidPushConstantWrite);
        }
        // SAFETY: The pipeline layout is live and every requested stage/byte is covered.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_push_constants(
                self.buffer_mut().handle,
                layout.handle(),
                stages,
                offset,
                data,
            );
        }
        self.resources.push(layout);
        Ok(())
    }

    /// Records a non-indexed draw in the active render-pass subpass.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_none()
            || self.buffer_mut().graphics_pipeline.is_none()
            || !self.buffer_mut().viewport_set
            || !self.buffer_mut().scissor_set
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is inside a compatible render pass with pipeline and dynamic state bound.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_draw(
                self.buffer_mut().handle,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
        Ok(())
    }

    /// Records an indexed draw in the active render-pass subpass.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_none()
            || self.buffer_mut().graphics_pipeline.is_none()
            || !self.buffer_mut().viewport_set
            || !self.buffer_mut().scissor_set
            || !self.buffer_mut().index_buffer_bound
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer has a compatible graphics pipeline, dynamic state, and index buffer bound.
        unsafe {
            self.buffer_mut().pool.device.handle.cmd_draw_indexed(
                self.buffer_mut().handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
        Ok(())
    }

    /// Records a compute dispatch outside a render pass.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> Result<(), VulkanError> {
        self.buffer_mut().require_recording()?;
        if self.buffer_mut().current_render_pass.is_some()
            || self.buffer_mut().compute_pipeline_layout.is_none()
            || !self
                .buffer_mut()
                .pool
                .queue_flags
                .contains(vk::QueueFlags::COMPUTE)
        {
            return Err(VulkanError::InvalidCommandBufferState);
        }
        // SAFETY: The command buffer is recording outside a render pass with a compute pipeline bound.
        unsafe {
            self.buffer_mut()
                .pool
                .device
                .handle
                .cmd_dispatch(self.buffer_mut().handle, x, y, z);
        }
        Ok(())
    }
}

impl Drop for CommandBufferRecording<'_, '_, '_> {
    fn drop(&mut self) {
        if let Some(command_buffer) = self.command_buffer.as_deref_mut() {
            if command_buffer.state == CommandBufferState::Recording {
                if let Err(error) = command_buffer.abort_recording() {
                    tracing::warn!(
                        ?error,
                        "failed to abort unfinished Vulkan command-buffer recording"
                    );
                }
            }
        }
    }
}

/// Executable command buffer retaining every resource referenced during recording.
pub struct ExecutableCommandBuffer<'recording, 'pool, 'device> {
    command_buffer: &'recording mut VulkanCommandBuffer<'pool, 'device>,
    _descriptor_leases: Vec<DescriptorSetLeaseRef<'recording>>,
    _resources: Vec<&'recording dyn CommandResource>,
    swapchain_target: Option<(Arc<()>, u64, u32)>,
}

impl Drop for ExecutableCommandBuffer<'_, '_, '_> {
    fn drop(&mut self) {
        if self.command_buffer.state == CommandBufferState::Executable {
            // SAFETY: The command buffer is executable and not pending; its pool allows individual reset.
            if let Err(error) = unsafe {
                self.command_buffer.pool.device.handle.reset_command_buffer(
                    self.command_buffer.handle,
                    vk::CommandBufferResetFlags::empty(),
                )
            } {
                tracing::warn!(
                    ?error,
                    "failed to reset abandoned executable command buffer"
                );
            }
            self.command_buffer.state = CommandBufferState::Initial;
            self.command_buffer.clear_recording_state();
        }
    }
}

impl ExecutableCommandBuffer<'_, '_, '_> {
    /// Returns the raw executable command-buffer handle.
    #[must_use]
    pub const fn handle(&self) -> vk::CommandBuffer {
        self.command_buffer.handle
    }

    pub(crate) fn targets_swapchain(&self, identity: &(Arc<()>, u64), image_index: u32) -> bool {
        swapchain_target_matches(self.swapchain_target.as_ref(), identity, image_index)
    }
}

/// Submits one executable primary command buffer to the device graphics queue.
pub struct VulkanQueue<'device> {
    device: &'device VulkanDevice,
    handle: vk::Queue,
    family: u32,
}

impl<'device> VulkanQueue<'device> {
    /// Creates a queue wrapper for the selected graphics queue.
    #[must_use]
    pub const fn graphics(device: &'device VulkanDevice) -> Self {
        Self {
            device,
            handle: device.graphics_queue,
            family: device.graphics_queue_family,
        }
    }

    /// Submits an executable command buffer and returns a fence-backed completion guard.
    pub fn submit<'recording, 'pool>(
        &self,
        executable: ExecutableCommandBuffer<'recording, 'pool, 'device>,
        wait_semaphores: &[&VulkanSemaphore<'device>],
        wait_stage_masks: &[vk::PipelineStageFlags],
        signal_semaphores: &[&VulkanSemaphore<'device>],
    ) -> Result<VulkanSubmission<'recording, 'pool, 'device>, VulkanError> {
        if !std::ptr::eq(self.device, executable.command_buffer.pool.device)
            || executable.command_buffer.pool.queue_family != self.family
            || executable.command_buffer.state != CommandBufferState::Executable
            || wait_semaphores.len() != wait_stage_masks.len()
            || wait_stage_masks.iter().any(|stage| stage.is_empty())
            || wait_semaphores
                .iter()
                .chain(signal_semaphores)
                .any(|semaphore| !semaphore.belongs_to(self.device))
        {
            return Err(VulkanError::InvalidQueueSubmission);
        }
        let wait_handles: Vec<_> = wait_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect();
        let signal_handles: Vec<_> = signal_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect();
        if !validate_semaphore_handle_sets(&wait_handles, &signal_handles) {
            return Err(VulkanError::InvalidQueueSubmission);
        }
        let fence = Arc::new(VulkanFence::new(
            self.device,
            FenceConfig { signaled: false },
        )?);
        self.submit_with_fence(
            executable,
            wait_semaphores,
            wait_stage_masks,
            signal_semaphores,
            fence,
            None,
        )
    }

    pub(crate) fn submit_with_fence<'recording, 'pool>(
        &self,
        executable: ExecutableCommandBuffer<'recording, 'pool, 'device>,
        wait_semaphores: &[&VulkanSemaphore<'device>],
        wait_stage_masks: &[vk::PipelineStageFlags],
        signal_semaphores: &[&VulkanSemaphore<'device>],
        fence: Arc<VulkanFence<'device>>,
        slot_lease: Option<Arc<AtomicBool>>,
    ) -> Result<VulkanSubmission<'recording, 'pool, 'device>, VulkanError> {
        if !fence.belongs_to(self.device)
            || fence.is_signaled()?
            || !std::ptr::eq(self.device, executable.command_buffer.pool.device)
            || executable.command_buffer.pool.queue_family != self.family
            || executable.command_buffer.state != CommandBufferState::Executable
            || wait_semaphores.len() != wait_stage_masks.len()
            || wait_stage_masks.iter().any(|stage| stage.is_empty())
            || wait_semaphores
                .iter()
                .chain(signal_semaphores)
                .any(|semaphore| !semaphore.belongs_to(self.device))
        {
            return Err(VulkanError::InvalidQueueSubmission);
        }
        let wait_handles: Vec<_> = wait_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect();
        let signal_handles: Vec<_> = signal_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect();
        if !validate_semaphore_handle_sets(&wait_handles, &signal_handles) {
            return Err(VulkanError::InvalidQueueSubmission);
        }
        let command_handles = [executable.command_buffer.handle];
        let info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_handles)
            .wait_dst_stage_mask(wait_stage_masks)
            .command_buffers(&command_handles)
            .signal_semaphores(&signal_handles);
        let _queue_guard = self
            .device
            .queue_submit_lock
            .lock()
            .map_err(|_| VulkanError::QueueSynchronizationPoisoned)?;
        // SAFETY: All submit arrays remain alive through the call; the queue is externally
        // synchronized by the device mutex and every handle belongs to this device.
        unsafe {
            self.device
                .handle
                .queue_submit(self.handle, &[info], fence.handle())
                .map_err(|error| api_error("submit Vulkan command buffer", error))?;
        }
        executable.command_buffer.state = CommandBufferState::Pending;
        if let Some(lease) = &slot_lease {
            lease.store(true, Ordering::Release);
        }
        Ok(VulkanSubmission {
            executable: Some(executable),
            fence,
            completed: false,
            slot_lease,
        })
    }

    pub(crate) fn belongs_to(&self, device: &VulkanDevice) -> bool {
        std::ptr::eq(self.device, device)
    }

    /// Returns the queue-family index used by this wrapper.
    #[must_use]
    pub const fn family(&self) -> u32 {
        self.family
    }
}

/// Retains a submitted command buffer and its descriptor leases until its fence signals.
pub struct VulkanSubmission<'recording, 'pool, 'device> {
    executable: Option<ExecutableCommandBuffer<'recording, 'pool, 'device>>,
    fence: Arc<VulkanFence<'device>>,
    completed: bool,
    slot_lease: Option<Arc<AtomicBool>>,
}

impl VulkanSubmission<'_, '_, '_> {
    /// Waits for submission completion or returns `Timeout` without releasing resources.
    pub fn wait(&mut self, timeout_nanoseconds: u64) -> Result<FenceWaitResult, VulkanError> {
        let result = self.fence.wait(timeout_nanoseconds)?;
        if result == FenceWaitResult::Signaled {
            self.mark_completed();
        }
        Ok(result)
    }

    /// Returns whether the submission fence has signaled.
    pub fn is_complete(&mut self) -> Result<bool, VulkanError> {
        if self.fence.is_signaled()? {
            self.mark_completed();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Returns the fence handle associated with this submission.
    ///
    /// The handle is only associated with this submission until completion is acknowledged;
    /// a frame scheduler may reset and reuse it after `wait` reports `Signaled`.
    #[must_use]
    pub fn fence(&self) -> vk::Fence {
        self.fence.handle()
    }

    fn mark_completed(&mut self) {
        if let Some(executable) = &mut self.executable {
            executable.command_buffer.state = CommandBufferState::Executable;
        }
        self.completed = true;
        if let Some(lease) = &self.slot_lease {
            lease.store(false, Ordering::Release);
        }
    }
}

impl Drop for VulkanSubmission<'_, '_, '_> {
    fn drop(&mut self) {
        if !self.completed {
            match self.fence.wait(u64::MAX) {
                Ok(FenceWaitResult::Signaled) => self.mark_completed(),
                Ok(FenceWaitResult::Timeout) => {
                    let idle_result = self
                        .executable
                        .as_ref()
                        .map(|executable| executable.command_buffer.pool.device.wait_idle());
                    if matches!(idle_result, Some(Ok(()))) {
                        self.mark_completed();
                    } else {
                        tracing::error!(
                            ?idle_result,
                            "infinite Vulkan submission-fence wait timed out and device-idle fallback failed"
                        );
                    }
                }
                Err(error) => {
                    let idle_result = self
                        .executable
                        .as_ref()
                        .map(|executable| executable.command_buffer.pool.device.wait_idle());
                    if matches!(idle_result, Some(Ok(()))) {
                        self.mark_completed();
                    } else {
                        tracing::error!(
                            ?error,
                            ?idle_result,
                            "failed to establish Vulkan completion before releasing submission resources"
                        );
                    }
                }
            }
        }
    }
}

fn validate_buffer_copy_region(
    region: &vk::BufferCopy,
    source_size: vk::DeviceSize,
    destination_size: vk::DeviceSize,
) -> bool {
    region.size > 0
        && region.src_offset % 4 == 0
        && region.dst_offset % 4 == 0
        && region.size % 4 == 0
        && region
            .src_offset
            .checked_add(region.size)
            .is_some_and(|end| end <= source_size)
        && region
            .dst_offset
            .checked_add(region.size)
            .is_some_and(|end| end <= destination_size)
}

fn ranges_overlap(ranges: &[(u64, u64)]) -> bool {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable_by_key(|range| range.0);
    sorted.windows(2).any(|pair| pair[0].1 > pair[1].0)
}

fn is_transfer_source_layout(layout: vk::ImageLayout) -> bool {
    matches!(
        layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL | vk::ImageLayout::GENERAL
    )
}

fn is_transfer_destination_layout(layout: vk::ImageLayout) -> bool {
    matches!(
        layout,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL | vk::ImageLayout::GENERAL
    )
}

fn mip_dimension(base: u32, level: u32) -> u32 {
    base.checked_shr(level).unwrap_or(0).max(1)
}

fn format_bytes_per_texel(format: vk::Format) -> Option<u64> {
    use vk::Format as F;
    Some(match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT => 1,
        F::R8G8_UNORM
        | F::R8G8_SNORM
        | F::R8G8_UINT
        | F::R8G8_SINT
        | F::R16_UNORM
        | F::R16_SNORM
        | F::R16_UINT
        | F::R16_SINT
        | F::R16_SFLOAT => 2,
        F::R8G8B8_UNORM
        | F::R8G8B8_SNORM
        | F::R8G8B8_UINT
        | F::R8G8B8_SINT
        | F::B8G8R8_UNORM
        | F::B8G8R8_SNORM
        | F::B8G8R8_UINT
        | F::B8G8R8_SINT => 3,
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_SRGB
        | F::B8G8R8A8_UNORM
        | F::B8G8R8A8_SRGB
        | F::A2B10G10R10_UNORM_PACK32
        | F::A2R10G10B10_UNORM_PACK32
        | F::B10G11R11_UFLOAT_PACK32
        | F::E5B9G9R9_UFLOAT_PACK32
        | F::R16G16_UNORM
        | F::R16G16_SNORM
        | F::R16G16_UINT
        | F::R16G16_SINT
        | F::R16G16_SFLOAT
        | F::R32_UINT
        | F::R32_SINT
        | F::R32_SFLOAT => 4,
        F::R16G16B16_UNORM
        | F::R16G16B16_SNORM
        | F::R16G16B16_UINT
        | F::R16G16B16_SINT
        | F::R16G16B16_SFLOAT => 6,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT
        | F::R32G32_UINT
        | F::R32G32_SINT
        | F::R32G32_SFLOAT => 8,
        F::R32G32B32_UINT | F::R32G32B32_SINT | F::R32G32B32_SFLOAT => 12,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FormatCopyBlockInfo {
    width: u32,
    height: u32,
    depth: u32,
    bytes: u64,
}

impl FormatCopyBlockInfo {
    const fn texel(bytes: u64) -> Self {
        Self {
            width: 1,
            height: 1,
            depth: 1,
            bytes,
        }
    }

    const fn compressed(width: u32, height: u32, bytes: u64) -> Self {
        Self {
            width,
            height,
            depth: 1,
            bytes,
        }
    }
}

/// Returns the byte/block geometry for one copy aspect. Combined depth/stencil
/// formats must be copied one aspect at a time, as required by Vulkan.
fn format_copy_block_info(
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Option<FormatCopyBlockInfo> {
    use vk::{Format as F, ImageAspectFlags as A};
    if aspect.as_raw().count_ones() != 1 {
        return None;
    }
    if aspect == A::COLOR {
        if let Some(bytes) = format_bytes_per_texel(format) {
            return Some(FormatCopyBlockInfo::texel(bytes));
        }
        return Some(match format {
            F::BC1_RGB_UNORM_BLOCK
            | F::BC1_RGB_SRGB_BLOCK
            | F::BC1_RGBA_UNORM_BLOCK
            | F::BC1_RGBA_SRGB_BLOCK
            | F::BC4_UNORM_BLOCK
            | F::BC4_SNORM_BLOCK
            | F::ETC2_R8G8B8_UNORM_BLOCK
            | F::ETC2_R8G8B8_SRGB_BLOCK
            | F::ETC2_R8G8B8A1_UNORM_BLOCK
            | F::ETC2_R8G8B8A1_SRGB_BLOCK
            | F::EAC_R11_UNORM_BLOCK
            | F::EAC_R11_SNORM_BLOCK => FormatCopyBlockInfo::compressed(4, 4, 8),
            F::BC2_UNORM_BLOCK
            | F::BC2_SRGB_BLOCK
            | F::BC3_UNORM_BLOCK
            | F::BC3_SRGB_BLOCK
            | F::BC5_UNORM_BLOCK
            | F::BC5_SNORM_BLOCK
            | F::BC6H_UFLOAT_BLOCK
            | F::BC6H_SFLOAT_BLOCK
            | F::BC7_UNORM_BLOCK
            | F::BC7_SRGB_BLOCK
            | F::ETC2_R8G8B8A8_UNORM_BLOCK
            | F::ETC2_R8G8B8A8_SRGB_BLOCK
            | F::EAC_R11G11_UNORM_BLOCK
            | F::EAC_R11G11_SNORM_BLOCK => FormatCopyBlockInfo::compressed(4, 4, 16),
            F::ASTC_4X4_UNORM_BLOCK | F::ASTC_4X4_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(4, 4, 16)
            }
            F::ASTC_5X4_UNORM_BLOCK | F::ASTC_5X4_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(5, 4, 16)
            }
            F::ASTC_5X5_UNORM_BLOCK | F::ASTC_5X5_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(5, 5, 16)
            }
            F::ASTC_6X5_UNORM_BLOCK | F::ASTC_6X5_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(6, 5, 16)
            }
            F::ASTC_6X6_UNORM_BLOCK | F::ASTC_6X6_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(6, 6, 16)
            }
            F::ASTC_8X5_UNORM_BLOCK | F::ASTC_8X5_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(8, 5, 16)
            }
            F::ASTC_8X6_UNORM_BLOCK | F::ASTC_8X6_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(8, 6, 16)
            }
            F::ASTC_8X8_UNORM_BLOCK | F::ASTC_8X8_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(8, 8, 16)
            }
            F::ASTC_10X5_UNORM_BLOCK | F::ASTC_10X5_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(10, 5, 16)
            }
            F::ASTC_10X6_UNORM_BLOCK | F::ASTC_10X6_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(10, 6, 16)
            }
            F::ASTC_10X8_UNORM_BLOCK | F::ASTC_10X8_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(10, 8, 16)
            }
            F::ASTC_10X10_UNORM_BLOCK | F::ASTC_10X10_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(10, 10, 16)
            }
            F::ASTC_12X10_UNORM_BLOCK | F::ASTC_12X10_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(12, 10, 16)
            }
            F::ASTC_12X12_UNORM_BLOCK | F::ASTC_12X12_SRGB_BLOCK => {
                FormatCopyBlockInfo::compressed(12, 12, 16)
            }
            _ => return None,
        });
    }
    if aspect == A::DEPTH {
        return match format {
            F::D16_UNORM | F::D16_UNORM_S8_UINT => Some(FormatCopyBlockInfo::texel(2)),
            F::X8_D24_UNORM_PACK32
            | F::D32_SFLOAT
            | F::D24_UNORM_S8_UINT
            | F::D32_SFLOAT_S8_UINT => Some(FormatCopyBlockInfo::texel(4)),
            _ => None,
        };
    }
    if aspect == A::STENCIL {
        return match format {
            F::S8_UINT | F::D16_UNORM_S8_UINT | F::D24_UNORM_S8_UINT | F::D32_SFLOAT_S8_UINT => {
                Some(FormatCopyBlockInfo::texel(1))
            }
            _ => None,
        };
    }
    None
}

fn div_ceil(value: u64, divisor: u64) -> Option<u64> {
    if divisor == 0 {
        return None;
    }
    value
        .checked_add(divisor - 1)
        .map(|rounded| rounded / divisor)
}

fn validate_buffer_image_copy_region(
    region: &vk::BufferImageCopy,
    buffer_size: vk::DeviceSize,
    image: crate::ImageConfig,
) -> Result<(), VulkanError> {
    let subresource = region.image_subresource;
    let extent = region.image_extent;
    let offset = region.image_offset;
    let block = format_copy_block_info(image.format, subresource.aspect_mask)
        .ok_or(VulkanError::UnsupportedCopyFormat)?;
    if image.samples != vk::SampleCountFlags::TYPE_1
        || subresource.mip_level >= image.mip_levels
        || subresource.layer_count == 0
        || subresource
            .base_array_layer
            .checked_add(subresource.layer_count)
            .is_none_or(|end| end > image.array_layers)
        || extent.width == 0
        || extent.height == 0
        || extent.depth == 0
        || offset.x < 0
        || offset.y < 0
        || offset.z < 0
        || (region.buffer_row_length != 0 && region.buffer_row_length < extent.width)
        || (region.buffer_image_height != 0 && region.buffer_image_height < extent.height)
        || (region.buffer_row_length != 0 && region.buffer_row_length % block.width != 0)
        || (region.buffer_image_height != 0 && region.buffer_image_height % block.height != 0)
        || region.buffer_offset % 4 != 0
        || region.buffer_offset % block.bytes != 0
    {
        return Err(VulkanError::InvalidCopyRegion);
    }
    let compressed = block.width > 1 || block.height > 1 || block.depth > 1;
    if compressed && image.image_type != vk::ImageType::TYPE_2D {
        return Err(VulkanError::InvalidCopyRegion);
    }
    match image.image_type {
        vk::ImageType::TYPE_1D
            if offset.y != 0 || extent.height != 1 || offset.z != 0 || extent.depth != 1 =>
        {
            return Err(VulkanError::InvalidCopyRegion);
        }
        vk::ImageType::TYPE_2D if offset.z != 0 || extent.depth != 1 => {
            return Err(VulkanError::InvalidCopyRegion);
        }
        vk::ImageType::TYPE_3D
            if subresource.base_array_layer != 0 || subresource.layer_count != 1 =>
        {
            return Err(VulkanError::InvalidCopyRegion);
        }
        _ => {}
    }
    let mip_extent = vk::Extent3D {
        width: mip_dimension(image.extent.width, subresource.mip_level),
        height: mip_dimension(image.extent.height, subresource.mip_level),
        depth: mip_dimension(image.extent.depth, subresource.mip_level),
    };
    let end_x = (offset.x as u32)
        .checked_add(extent.width)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let end_y = (offset.y as u32)
        .checked_add(extent.height)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let end_z = (offset.z as u32)
        .checked_add(extent.depth)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    if end_x > mip_extent.width
        || end_y > mip_extent.height
        || end_z > mip_extent.depth
        || (offset.x as u32) % block.width != 0
        || (offset.y as u32) % block.height != 0
        || (offset.z as u32) % block.depth != 0
        || (extent.width % block.width != 0 && end_x != mip_extent.width)
        || (extent.height % block.height != 0 && end_y != mip_extent.height)
        || (extent.depth % block.depth != 0 && end_z != mip_extent.depth)
    {
        return Err(VulkanError::InvalidCopyRegion);
    }

    let row_texels = u64::from(if region.buffer_row_length == 0 {
        extent.width
    } else {
        region.buffer_row_length
    });
    let image_rows = u64::from(if region.buffer_image_height == 0 {
        extent.height
    } else {
        region.buffer_image_height
    });
    let row_blocks =
        div_ceil(row_texels, u64::from(block.width)).ok_or(VulkanError::InvalidCopyRegion)?;
    let image_block_rows =
        div_ceil(image_rows, u64::from(block.height)).ok_or(VulkanError::InvalidCopyRegion)?;
    let copy_block_rows = div_ceil(u64::from(extent.height), u64::from(block.height))
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let copy_block_slices = div_ceil(u64::from(extent.depth), u64::from(block.depth))
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let final_row_blocks = div_ceil(u64::from(extent.width), u64::from(block.width))
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let row_bytes = row_blocks
        .checked_mul(block.bytes)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let slice_bytes = row_bytes
        .checked_mul(image_block_rows)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let layer_bytes = slice_bytes
        .checked_mul(copy_block_slices)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let preceding_layers = layer_bytes
        .checked_mul(u64::from(subresource.layer_count - 1))
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let preceding_depth = slice_bytes
        .checked_mul(copy_block_slices - 1)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let preceding_rows = row_bytes
        .checked_mul(copy_block_rows - 1)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let final_row = final_row_blocks
        .checked_mul(block.bytes)
        .ok_or(VulkanError::InvalidCopyRegion)?;
    let required = preceding_layers
        .checked_add(preceding_depth)
        .and_then(|value| value.checked_add(preceding_rows))
        .and_then(|value| value.checked_add(final_row))
        .ok_or(VulkanError::InvalidCopyRegion)?;
    if region
        .buffer_offset
        .checked_add(required)
        .is_none_or(|end| end > buffer_size)
    {
        return Err(VulkanError::InvalidCopyRegion);
    }
    Ok(())
}

fn validate_image_copy_region(
    region: &vk::ImageCopy,
    source: crate::ImageConfig,
    destination: crate::ImageConfig,
) -> bool {
    let src = region.src_subresource;
    let dst = region.dst_subresource;
    let extent = region.extent;
    let src_offset = region.src_offset;
    let dst_offset = region.dst_offset;
    if src.aspect_mask != vk::ImageAspectFlags::COLOR
        || dst.aspect_mask != vk::ImageAspectFlags::COLOR
        || src.mip_level >= source.mip_levels
        || dst.mip_level >= destination.mip_levels
        || src.layer_count == 0
        || src.layer_count != dst.layer_count
        || src
            .base_array_layer
            .checked_add(src.layer_count)
            .is_none_or(|end| end > source.array_layers)
        || dst
            .base_array_layer
            .checked_add(dst.layer_count)
            .is_none_or(|end| end > destination.array_layers)
        || extent.width == 0
        || extent.height == 0
        || extent.depth == 0
        || src_offset.x < 0
        || src_offset.y < 0
        || src_offset.z < 0
        || dst_offset.x < 0
        || dst_offset.y < 0
        || dst_offset.z < 0
    {
        return false;
    }
    if source.image_type == vk::ImageType::TYPE_3D {
        if src.base_array_layer != 0
            || dst.base_array_layer != 0
            || src.layer_count != 1
            || dst.layer_count != 1
        {
            return false;
        }
    } else if src_offset.z != 0 || dst_offset.z != 0 || extent.depth != 1 {
        return false;
    }
    if source.image_type == vk::ImageType::TYPE_1D
        && (src_offset.y != 0 || dst_offset.y != 0 || extent.height != 1)
    {
        return false;
    }
    let src_extent = vk::Extent3D {
        width: mip_dimension(source.extent.width, src.mip_level),
        height: mip_dimension(source.extent.height, src.mip_level),
        depth: mip_dimension(source.extent.depth, src.mip_level),
    };
    let dst_extent = vk::Extent3D {
        width: mip_dimension(destination.extent.width, dst.mip_level),
        height: mip_dimension(destination.extent.height, dst.mip_level),
        depth: mip_dimension(destination.extent.depth, dst.mip_level),
    };
    (src_offset.x as u32)
        .checked_add(extent.width)
        .is_some_and(|end| end <= src_extent.width)
        && (src_offset.y as u32)
            .checked_add(extent.height)
            .is_some_and(|end| end <= src_extent.height)
        && (src_offset.z as u32)
            .checked_add(extent.depth)
            .is_some_and(|end| end <= src_extent.depth)
        && (dst_offset.x as u32)
            .checked_add(extent.width)
            .is_some_and(|end| end <= dst_extent.width)
        && (dst_offset.y as u32)
            .checked_add(extent.height)
            .is_some_and(|end| end <= dst_extent.height)
        && (dst_offset.z as u32)
            .checked_add(extent.depth)
            .is_some_and(|end| end <= dst_extent.depth)
}

fn validate_command_pool_flags(flags: vk::CommandPoolCreateFlags) -> bool {
    let allowed =
        vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
    flags.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        && flags.as_raw() & !allowed.as_raw() == 0
}

fn validate_buffer_barrier_range(
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    buffer_size: vk::DeviceSize,
) -> bool {
    let range = if size == vk::WHOLE_SIZE {
        buffer_size.checked_sub(offset)
    } else {
        Some(size)
    };
    range.is_some_and(|range| {
        range > 0
            && offset
                .checked_add(range)
                .is_some_and(|end| end <= buffer_size)
    })
}

#[derive(Clone, Copy)]
struct ImageBarrierSubresourceRange {
    aspect: vk::ImageAspectFlags,
    base_mip_level: u32,
    level_count: u32,
    mip_levels: u32,
    base_array_layer: u32,
    layer_count: u32,
    array_layers: u32,
}

fn validate_image_barrier_range(
    range: ImageBarrierSubresourceRange,
    new_layout: vk::ImageLayout,
) -> bool {
    !range.aspect.is_empty()
        && range.level_count > 0
        && range.layer_count > 0
        && range
            .base_mip_level
            .checked_add(range.level_count)
            .is_some_and(|end| end <= range.mip_levels)
        && range
            .base_array_layer
            .checked_add(range.layer_count)
            .is_some_and(|end| end <= range.array_layers)
        && new_layout != vk::ImageLayout::UNDEFINED
        && new_layout != vk::ImageLayout::PREINITIALIZED
}

fn validate_buffer_barrier(
    barrier: &BufferBarrier<'_, '_>,
    device: &VulkanDevice,
) -> Result<(), VulkanError> {
    if !std::ptr::eq(device, barrier.buffer.device)
        || !validate_buffer_barrier_range(barrier.offset, barrier.size, barrier.buffer.size())
    {
        return Err(VulkanError::InvalidCommandBufferState);
    }
    Ok(())
}

fn validate_image_barrier(
    barrier: &ImageBarrier<'_, '_>,
    device: &VulkanDevice,
) -> Result<(), VulkanError> {
    let config = barrier.image.config();
    if !std::ptr::eq(device, barrier.image.device)
        || !validate_image_barrier_range(
            ImageBarrierSubresourceRange {
                aspect: barrier.aspect,
                base_mip_level: barrier.base_mip_level,
                level_count: barrier.level_count,
                mip_levels: config.mip_levels,
                base_array_layer: barrier.base_array_layer,
                layer_count: barrier.layer_count,
                array_layers: config.array_layers,
            },
            barrier.new_layout,
        )
    {
        return Err(VulkanError::InvalidCommandBufferState);
    }
    Ok(())
}

fn validate_dynamic_offset_requirement(
    requirement: &crate::descriptor::DynamicOffsetRequirement,
    dynamic_offset: u32,
    uniform_alignment: vk::DeviceSize,
    storage_alignment: vk::DeviceSize,
) -> Result<(), VulkanError> {
    let alignment = match requirement.descriptor_type {
        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => uniform_alignment,
        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => storage_alignment,
        _ => return Err(VulkanError::InvalidDynamicDescriptorOffset),
    };
    let effective_offset = requirement
        .base_offset
        .checked_add(u64::from(dynamic_offset))
        .ok_or(VulkanError::InvalidDynamicDescriptorOffset)?;
    let end = effective_offset
        .checked_add(requirement.range)
        .ok_or(VulkanError::InvalidDynamicDescriptorOffset)?;
    if (alignment > 1 && u64::from(dynamic_offset) % alignment != 0)
        || end > requirement.buffer_size
    {
        return Err(VulkanError::InvalidDynamicDescriptorOffset);
    }
    Ok(())
}

fn validate_semaphore_handle_sets(
    wait_handles: &[vk::Semaphore],
    signal_handles: &[vk::Semaphore],
) -> bool {
    if wait_handles
        .iter()
        .any(|handle| *handle == vk::Semaphore::null())
        || signal_handles
            .iter()
            .any(|handle| *handle == vk::Semaphore::null())
    {
        return false;
    }
    for (index, handle) in wait_handles.iter().enumerate() {
        if wait_handles[..index].contains(handle) || signal_handles.contains(handle) {
            return false;
        }
    }
    for (index, handle) in signal_handles.iter().enumerate() {
        if signal_handles[..index].contains(handle) {
            return false;
        }
    }
    true
}

fn push_constant_range_covers(
    ranges: &[vk::PushConstantRange],
    stages: vk::ShaderStageFlags,
    offset: u32,
    end: u32,
) -> bool {
    let supported = vk::ShaderStageFlags::VERTEX
        | vk::ShaderStageFlags::FRAGMENT
        | vk::ShaderStageFlags::COMPUTE;
    if stages.as_raw() & !supported.as_raw() != 0 {
        return false;
    }
    for stage in [
        vk::ShaderStageFlags::VERTEX,
        vk::ShaderStageFlags::FRAGMENT,
        vk::ShaderStageFlags::COMPUTE,
    ] {
        if !stages.contains(stage) {
            continue;
        }
        let mut stage_ranges: Vec<_> = ranges
            .iter()
            .filter(|range| range.stage_flags.contains(stage))
            .collect();
        stage_ranges.sort_unstable_by_key(|range| range.offset);
        let mut covered = offset;
        for range in stage_ranges {
            let range_end = match range.offset.checked_add(range.size) {
                Some(end) => end,
                None => return false,
            };
            if range_end <= covered || range.offset > covered {
                continue;
            }
            covered = covered.max(range_end);
            if covered >= end {
                break;
            }
        }
        if covered < end {
            return false;
        }
    }
    true
}

fn swapchain_target_matches(
    target: Option<&(Arc<()>, u64, u32)>,
    identity: &(Arc<()>, u64),
    image_index: u32,
) -> bool {
    target.is_some_and(|target| {
        Arc::ptr_eq(&target.0, &identity.0) && target.1 == identity.1 && target.2 == image_index
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ImageBarrierSubresourceRange, format_bytes_per_texel, format_copy_block_info,
        push_constant_range_covers, ranges_overlap, swapchain_target_matches,
        validate_buffer_barrier_range, validate_buffer_copy_region,
        validate_buffer_image_copy_region, validate_command_pool_flags,
        validate_dynamic_offset_requirement, validate_image_barrier_range,
        validate_image_copy_region, validate_semaphore_handle_sets,
    };
    use crate::descriptor::DynamicOffsetRequirement;
    use ash::vk;
    use std::sync::Arc;

    #[test]
    fn validates_command_pool_flags_and_barrier_ranges() {
        assert!(validate_command_pool_flags(
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                | vk::CommandPoolCreateFlags::TRANSIENT
        ));
        assert!(!validate_command_pool_flags(
            vk::CommandPoolCreateFlags::empty()
        ));
        assert!(!validate_command_pool_flags(
            vk::CommandPoolCreateFlags::PROTECTED
        ));

        assert!(validate_buffer_barrier_range(128, 256, 1024));
        assert!(validate_buffer_barrier_range(256, vk::WHOLE_SIZE, 1024));
        assert!(!validate_buffer_barrier_range(1024, vk::WHOLE_SIZE, 1024));
        assert!(!validate_buffer_barrier_range(u64::MAX - 2, 8, u64::MAX));
        assert!(!validate_buffer_barrier_range(0, 0, 1024));

        let valid_range = ImageBarrierSubresourceRange {
            aspect: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 2,
            mip_levels: 4,
            base_array_layer: 0,
            layer_count: 1,
            array_layers: 1,
        };
        assert!(validate_image_barrier_range(
            valid_range,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ));
        assert!(!validate_image_barrier_range(
            ImageBarrierSubresourceRange {
                aspect: vk::ImageAspectFlags::empty(),
                level_count: 1,
                ..valid_range
            },
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ));
        assert!(!validate_image_barrier_range(
            ImageBarrierSubresourceRange {
                base_mip_level: 3,
                level_count: 2,
                ..valid_range
            },
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ));
        assert!(!validate_image_barrier_range(
            valid_range,
            vk::ImageLayout::UNDEFINED,
        ));
    }

    #[test]
    fn validates_dynamic_offset_alignment_and_effective_buffer_bounds() {
        let requirement = DynamicOffsetRequirement {
            binding: 0,
            array_element: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            base_offset: 256,
            range: 256,
            buffer_size: 1024,
        };
        assert!(validate_dynamic_offset_requirement(&requirement, 256, 256, 16).is_ok());
        assert!(validate_dynamic_offset_requirement(&requirement, 128, 256, 16).is_err());
        assert!(validate_dynamic_offset_requirement(&requirement, 768, 256, 16).is_err());
        let overflow = DynamicOffsetRequirement {
            base_offset: u64::MAX - 4,
            range: 16,
            ..requirement
        };
        assert!(validate_dynamic_offset_requirement(&overflow, 0, 1, 1).is_err());
    }

    #[test]
    fn rejects_duplicate_or_null_semaphores_in_a_queue_submission() {
        use ash::vk::Handle;
        let first = vk::Semaphore::from_raw(1);
        let second = vk::Semaphore::from_raw(2);
        assert!(validate_semaphore_handle_sets(&[first], &[second]));
        assert!(!validate_semaphore_handle_sets(&[first, first], &[]));
        assert!(!validate_semaphore_handle_sets(&[first], &[first]));
        assert!(!validate_semaphore_handle_sets(
            &[],
            &[vk::Semaphore::null()]
        ));
    }

    fn test_color_image_config() -> crate::ImageConfig {
        crate::ImageConfig {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            flags: vk::ImageCreateFlags::empty(),
        }
    }

    fn test_buffer_image_region() -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            })
    }

    #[test]
    fn validates_buffer_copy_alignment_bounds_and_overlap() {
        let valid = vk::BufferCopy::default()
            .src_offset(4)
            .dst_offset(8)
            .size(16);
        assert!(validate_buffer_copy_region(&valid, 64, 64));
        assert!(!validate_buffer_copy_region(&valid, 19, 64));
        assert!(!validate_buffer_copy_region(&valid, 64, 23));
        assert!(!validate_buffer_copy_region(
            &vk::BufferCopy::default().size(0),
            64,
            64
        ));
        let misaligned = vk::BufferCopy::default()
            .src_offset(2)
            .dst_offset(0)
            .size(4);
        assert!(!validate_buffer_copy_region(&misaligned, 64, 64));
        let overflow = vk::BufferCopy::default()
            .src_offset(u64::MAX - 3)
            .dst_offset(0)
            .size(8);
        assert!(!validate_buffer_copy_region(&overflow, u64::MAX, 64));
        assert!(!ranges_overlap(&[(0, 4), (4, 8)]));
        assert!(ranges_overlap(&[(0, 8), (4, 12)]));
    }

    #[test]
    fn image_copy_texel_size_support_is_explicit() {
        assert_eq!(format_bytes_per_texel(vk::Format::R8_UNORM), Some(1));
        assert_eq!(format_bytes_per_texel(vk::Format::R8G8B8A8_UNORM), Some(4));
        assert_eq!(
            format_bytes_per_texel(vk::Format::R32G32B32_SFLOAT),
            Some(12)
        );
        assert_eq!(
            format_bytes_per_texel(vk::Format::BC1_RGBA_UNORM_BLOCK),
            None
        );
        assert_eq!(format_bytes_per_texel(vk::Format::D32_SFLOAT), None);
    }

    #[test]
    fn copy_block_metadata_covers_compressed_and_depth_stencil_aspects() {
        assert_eq!(
            format_copy_block_info(
                vk::Format::BC1_RGBA_UNORM_BLOCK,
                vk::ImageAspectFlags::COLOR,
            ),
            Some(super::FormatCopyBlockInfo::compressed(4, 4, 8))
        );
        assert_eq!(
            format_copy_block_info(
                vk::Format::ASTC_10X6_SRGB_BLOCK,
                vk::ImageAspectFlags::COLOR,
            ),
            Some(super::FormatCopyBlockInfo::compressed(10, 6, 16))
        );
        assert_eq!(
            format_copy_block_info(vk::Format::D24_UNORM_S8_UINT, vk::ImageAspectFlags::DEPTH,),
            Some(super::FormatCopyBlockInfo::texel(4))
        );
        assert_eq!(
            format_copy_block_info(vk::Format::D24_UNORM_S8_UINT, vk::ImageAspectFlags::STENCIL,),
            Some(super::FormatCopyBlockInfo::texel(1))
        );
        assert_eq!(
            format_copy_block_info(
                vk::Format::D24_UNORM_S8_UINT,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            ),
            None
        );
        assert_eq!(
            format_copy_block_info(vk::Format::D32_SFLOAT, vk::ImageAspectFlags::COLOR,),
            None
        );
    }

    #[test]
    fn validates_buffer_image_copy_required_bytes_and_subresource_bounds() {
        let image = test_color_image_config();
        let region = test_buffer_image_region();
        assert!(validate_buffer_image_copy_region(&region, 64, image).is_ok());
        assert!(validate_buffer_image_copy_region(&region, 63, image).is_err());
        let outside = vk::BufferImageCopy::default()
            .image_subresource(region.image_subresource)
            .image_offset(vk::Offset3D { x: 1, y: 0, z: 0 })
            .image_extent(region.image_extent);
        assert!(validate_buffer_image_copy_region(&outside, 128, image).is_err());
        let short_row = vk::BufferImageCopy::default()
            .buffer_row_length(3)
            .image_subresource(region.image_subresource)
            .image_extent(region.image_extent);
        assert!(validate_buffer_image_copy_region(&short_row, 128, image).is_err());
        let misaligned = vk::BufferImageCopy::default()
            .buffer_offset(2)
            .image_subresource(region.image_subresource)
            .image_extent(region.image_extent);
        assert!(validate_buffer_image_copy_region(&misaligned, 128, image).is_err());
        let compressed = crate::ImageConfig {
            format: vk::Format::BC1_RGBA_UNORM_BLOCK,
            ..image
        };
        let compressed_region = vk::BufferImageCopy {
            image_extent: vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            },
            ..region
        };
        assert!(validate_buffer_image_copy_region(&compressed_region, 8, compressed).is_ok());
        assert!(validate_buffer_image_copy_region(&compressed_region, 7, compressed).is_err());
        let compressed_edge_image = crate::ImageConfig {
            extent: vk::Extent3D {
                width: 5,
                height: 4,
                depth: 1,
            },
            ..compressed
        };
        let compressed_edge_region = vk::BufferImageCopy {
            image_extent: vk::Extent3D {
                width: 5,
                height: 4,
                depth: 1,
            },
            ..compressed_region
        };
        assert!(
            validate_buffer_image_copy_region(&compressed_edge_region, 16, compressed_edge_image)
                .is_ok()
        );
        let bad_compressed_row = vk::BufferImageCopy {
            buffer_row_length: 6,
            ..compressed_region
        };
        assert!(validate_buffer_image_copy_region(&bad_compressed_row, 32, compressed).is_err());
        let bad_compressed_offset = vk::BufferImageCopy {
            image_offset: vk::Offset3D { x: 1, y: 0, z: 0 },
            ..compressed_region
        };
        assert!(validate_buffer_image_copy_region(&bad_compressed_offset, 32, compressed).is_err());

        let depth_stencil = crate::ImageConfig {
            format: vk::Format::D24_UNORM_S8_UINT,
            ..image
        };
        let depth_region = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                ..region.image_subresource
            },
            ..region
        };
        let stencil_region = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::STENCIL,
                ..region.image_subresource
            },
            ..region
        };
        assert!(validate_buffer_image_copy_region(&depth_region, 64, depth_stencil).is_ok());
        assert!(validate_buffer_image_copy_region(&stencil_region, 16, depth_stencil).is_ok());
        let combined_aspect = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                ..region.image_subresource
            },
            ..region
        };
        assert!(matches!(
            validate_buffer_image_copy_region(&combined_aspect, 64, depth_stencil),
            Err(crate::VulkanError::UnsupportedCopyFormat)
        ));

        let unsupported = crate::ImageConfig {
            format: vk::Format::R64G64B64_SFLOAT,
            ..image
        };
        assert!(matches!(
            validate_buffer_image_copy_region(&region, 128, unsupported),
            Err(crate::VulkanError::UnsupportedCopyFormat)
        ));
    }

    #[test]
    fn validates_image_copy_source_and_destination_extents() {
        let config = test_color_image_config();
        let subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);
        let region = vk::ImageCopy::default()
            .src_subresource(subresource)
            .dst_subresource(subresource)
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .extent(vk::Extent3D {
                width: 4,
                height: 4,
                depth: 1,
            });
        assert!(validate_image_copy_region(&region, config, config));
        let outside = vk::ImageCopy {
            dst_offset: vk::Offset3D { x: 1, y: 0, z: 0 },
            ..region
        };
        assert!(!validate_image_copy_region(&outside, config, config));
        let bad_layers = vk::ImageCopy {
            dst_subresource: vk::ImageSubresourceLayers {
                layer_count: 0,
                ..subresource
            },
            ..region
        };
        assert!(!validate_image_copy_region(&bad_layers, config, config));
    }

    #[test]
    fn swapchain_command_targets_must_match_owner_generation_and_image_index() {
        let token = Arc::new(());
        let identity = (Arc::clone(&token), 4);
        let target = (Arc::clone(&token), 4, 2);
        assert!(swapchain_target_matches(Some(&target), &identity, 2));
        assert!(!swapchain_target_matches(Some(&target), &identity, 1));
        assert!(!swapchain_target_matches(
            Some(&target),
            &(Arc::clone(&token), 5),
            2
        ));
        assert!(!swapchain_target_matches(
            Some(&target),
            &(Arc::new(()), 4),
            2
        ));
        assert!(!swapchain_target_matches(None, &identity, 2));
    }

    #[test]
    fn validates_push_constant_write_coverage_for_each_stage() {
        let ranges = [
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(16),
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size(32),
        ];
        assert!(push_constant_range_covers(
            &ranges,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            16,
        ));
        assert!(!push_constant_range_covers(
            &ranges,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            24,
        ));
    }
}
