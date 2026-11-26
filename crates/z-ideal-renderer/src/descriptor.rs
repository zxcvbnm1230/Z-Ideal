//! Explicitly owned Vulkan descriptor layouts and pools.
use ash::vk;
use std::collections::HashSet;

use crate::{VulkanBuffer, VulkanDevice, VulkanError, VulkanImageView, VulkanSampler, api_error};

/// One binding in a descriptor-set layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBinding {
    /// Binding number visible to shaders.
    pub binding: u32,
    /// Kind of resource accepted by this binding.
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors in this binding.
    pub descriptor_count: u32,
    /// Shader stages allowed to access this binding.
    pub stage_flags: vk::ShaderStageFlags,
}

/// Configuration for a descriptor-set layout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DescriptorSetLayoutConfig {
    /// Bindings in the layout; an empty layout is valid.
    pub bindings: Vec<DescriptorBinding>,
}

/// One buffer descriptor update for a uniform or storage buffer binding.
#[derive(Clone, Copy)]
pub struct BufferDescriptorWrite<'resource, 'device> {
    /// Binding number in the descriptor-set layout.
    pub binding: u32,
    /// Array element within the binding.
    pub array_element: u32,
    /// Buffer retained by the returned resource lease.
    pub buffer: &'resource VulkanBuffer<'device>,
    /// Byte offset into the buffer; must satisfy the device's descriptor alignment.
    pub offset: vk::DeviceSize,
    /// Byte range; `vk::WHOLE_SIZE` selects the remaining buffer range.
    pub range: vk::DeviceSize,
}

/// One image or sampler descriptor update.
#[derive(Clone, Copy)]
pub struct ImageDescriptorWrite<'resource, 'image, 'device> {
    /// Binding number in the descriptor-set layout.
    pub binding: u32,
    /// Array element within the binding.
    pub array_element: u32,
    /// Image view retained by the returned resource lease.
    pub view: &'resource VulkanImageView<'image, 'device>,
    /// Sampler for combined-image-sampler descriptors; must be absent otherwise.
    pub sampler: Option<&'resource VulkanSampler<'device>>,
    /// Layout the image will be in when accessed by the shader.
    pub image_layout: vk::ImageLayout,
}

/// One sampler-only descriptor update.
#[derive(Clone, Copy)]
pub struct SamplerDescriptorWrite<'resource, 'device> {
    /// Binding number in the descriptor-set layout.
    pub binding: u32,
    /// Array element within the binding.
    pub array_element: u32,
    /// Sampler retained by the returned resource lease.
    pub sampler: &'resource VulkanSampler<'device>,
}

/// One resource write in a descriptor-set update batch.
#[derive(Clone, Copy)]
pub enum DescriptorResourceWrite<'resource, 'image, 'device> {
    /// Uniform or storage buffer descriptor.
    Buffer(BufferDescriptorWrite<'resource, 'device>),
    /// Sampled, combined-image-sampler, or storage-image descriptor.
    Image(ImageDescriptorWrite<'resource, 'image, 'device>),
    /// Sampler-only descriptor.
    Sampler(SamplerDescriptorWrite<'resource, 'device>),
}

/// Owns a descriptor-set layout and borrows its creating logical device.
pub struct VulkanDescriptorSetLayout<'device> {
    pub(super) device: &'device VulkanDevice,
    handle: vk::DescriptorSetLayout,
    config: DescriptorSetLayoutConfig,
}

impl<'device> VulkanDescriptorSetLayout<'device> {
    /// Creates a descriptor-set layout after validating binding declarations.
    pub fn new(
        device: &'device VulkanDevice,
        config: DescriptorSetLayoutConfig,
    ) -> Result<Self, VulkanError> {
        validate_layout_config(&config)?;
        let bindings: Vec<_> = config
            .bindings
            .iter()
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(binding.descriptor_type)
                    .descriptor_count(binding.descriptor_count)
                    .stage_flags(binding.stage_flags)
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        // SAFETY: The device is live and all binding data remains valid for the call.
        let handle = unsafe {
            device
                .handle
                .create_descriptor_set_layout(&info, None)
                .map_err(|error| api_error("create Vulkan descriptor-set layout", error))?
        };

        Ok(Self {
            device,
            handle,
            config,
        })
    }

    /// Returns the raw layout handle for pipeline-layout creation.
    #[must_use]
    pub const fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }

    /// Returns the validated layout configuration.
    #[must_use]
    pub const fn config(&self) -> &DescriptorSetLayoutConfig {
        &self.config
    }
}

impl Drop for VulkanDescriptorSetLayout<'_> {
    fn drop(&mut self) {
        // SAFETY: Descriptor sets and pipeline layouts borrow this layout; recorded command
        // buffers retain their users until submission completion.
        unsafe {
            self.device
                .handle
                .destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

/// Descriptor counts available from a descriptor pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorPoolSize {
    /// Descriptor kind accounted for by this pool entry.
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors of this kind.
    pub descriptor_count: u32,
}

/// Configuration for a descriptor pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorPoolConfig {
    /// Maximum number of descriptor sets that may be allocated.
    pub max_sets: u32,
    /// Descriptor capacities by descriptor kind.
    pub pool_sizes: Vec<DescriptorPoolSize>,
}

/// Owns a Vulkan descriptor pool and all sets allocated from it.
///
/// Individual sets are not freed. They borrow this pool and are released together
/// when the pool is destroyed.
pub struct VulkanDescriptorPool<'device> {
    device: &'device VulkanDevice,
    handle: vk::DescriptorPool,
}

impl<'device> VulkanDescriptorPool<'device> {
    /// Creates a descriptor pool with explicit set and descriptor capacities.
    pub fn new(
        device: &'device VulkanDevice,
        config: DescriptorPoolConfig,
    ) -> Result<Self, VulkanError> {
        validate_pool_config(&config)?;
        let pool_sizes: Vec<_> = config
            .pool_sizes
            .iter()
            .map(|size| {
                vk::DescriptorPoolSize::default()
                    .ty(size.descriptor_type)
                    .descriptor_count(size.descriptor_count)
            })
            .collect();
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(config.max_sets)
            .pool_sizes(&pool_sizes);

        // SAFETY: The device is live and pool-size data remains valid for the call.
        let handle = unsafe {
            device
                .handle
                .create_descriptor_pool(&info, None)
                .map_err(|error| api_error("create Vulkan descriptor pool", error))?
        };

        Ok(Self { device, handle })
    }

    /// Returns the raw descriptor-pool handle.
    #[must_use]
    pub const fn handle(&self) -> vk::DescriptorPool {
        self.handle
    }

    /// Allocates one descriptor set that borrows both its pool and layout.
    ///
    /// The descriptor set is automatically released with its pool. Resource writes
    /// use [`VulkanDescriptorSet::write_descriptors`] and return a lease that retains
    /// referenced GPU resources until submitted work has completed.
    pub fn allocate_set<'layout>(
        &self,
        layout: &'layout VulkanDescriptorSetLayout<'device>,
    ) -> Result<VulkanDescriptorSet<'_, 'layout, 'device>, VulkanError> {
        if !std::ptr::eq(self.device, layout.device) {
            return Err(VulkanError::DescriptorSetDeviceMismatch);
        }
        let layouts = [layout.handle];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(&layouts);

        // SAFETY: Pool and layout belong to the same live device and remain borrowed
        // for the full lifetime of the returned descriptor-set wrapper.
        let mut sets = unsafe {
            self.device
                .handle
                .allocate_descriptor_sets(&info)
                .map_err(|error| api_error("allocate Vulkan descriptor set", error))?
        };
        let handle = sets.pop().ok_or_else(|| {
            api_error("allocate Vulkan descriptor set", vk::Result::ERROR_UNKNOWN)
        })?;

        Ok(VulkanDescriptorSet {
            _pool: self,
            layout,
            handle,
        })
    }
}

impl Drop for VulkanDescriptorPool<'_> {
    fn drop(&mut self) {
        // SAFETY: This wrapper uniquely owns the pool; destroying it releases all sets.
        unsafe {
            self.device
                .handle
                .destroy_descriptor_pool(self.handle, None)
        };
    }
}

/// A descriptor set allocated from a pool and compatible with a borrowed layout.
///
/// The set itself is owned by the descriptor pool, so this wrapper does not
/// individually free the Vulkan descriptor-set handle.
pub struct VulkanDescriptorSet<'pool, 'layout, 'device> {
    _pool: &'pool VulkanDescriptorPool<'device>,
    layout: &'layout VulkanDescriptorSetLayout<'device>,
    handle: vk::DescriptorSet,
}

impl<'pool, 'layout, 'device> VulkanDescriptorSet<'pool, 'layout, 'device> {
    /// Updates buffer, sampled-image, combined-image-sampler, and storage-image descriptors.
    ///
    /// Every binding array element in the layout must be written exactly once in the
    /// batch. This initial complete-update contract prevents stale descriptor slots
    /// from outliving resources retained by an earlier lease. The returned lease
    /// retains all referenced resources and exclusively borrows this set, preventing
    /// another update until the lease is dropped. Keep it alive through all submissions
    /// that use the set. Command submissions retain the lease until their fence signals.
    /// Descriptor updates do not perform queue synchronization, image layout transitions,
    /// or buffer memory barriers; callers must establish all required dependencies.
    /// Dynamic buffer offsets and final effective ranges must be validated by the
    /// command-binding layer before drawing or dispatching.
    pub fn write_descriptors<'set, 'resource, 'image>(
        &'set mut self,
        writes: &[DescriptorResourceWrite<'resource, 'image, 'device>],
    ) -> Result<
        DescriptorResourceLease<'set, 'resource, 'image, 'pool, 'layout, 'device>,
        VulkanError,
    > {
        let mut seen = HashSet::with_capacity(writes.len());
        let mut buffer_infos = Vec::with_capacity(writes.len());
        let mut image_infos = Vec::with_capacity(writes.len());
        let mut prepared = Vec::with_capacity(writes.len());
        let mut retained_resources = Vec::with_capacity(writes.len().saturating_mul(2));
        let mut dynamic_offsets = Vec::new();

        for write in writes {
            let (binding_number, array_element) = match write {
                DescriptorResourceWrite::Buffer(write) => (write.binding, write.array_element),
                DescriptorResourceWrite::Image(write) => (write.binding, write.array_element),
                DescriptorResourceWrite::Sampler(write) => (write.binding, write.array_element),
            };
            validate_unique_descriptor_write(&mut seen, binding_number, array_element)?;
            let binding = self
                .layout
                .config
                .bindings
                .iter()
                .find(|binding| binding.binding == binding_number)
                .ok_or(VulkanError::DescriptorBindingNotFound)?;
            if array_element >= binding.descriptor_count {
                return Err(VulkanError::DescriptorArrayElementOutOfBounds);
            }

            match write {
                DescriptorResourceWrite::Buffer(write) => {
                    if !std::ptr::eq(self._pool.device, write.buffer.device) {
                        return Err(VulkanError::DescriptorSetDeviceMismatch);
                    }
                    validate_buffer_descriptor_write(
                        binding,
                        BufferDescriptorMetadata {
                            array_element: write.array_element,
                            offset: write.offset,
                            range: write.range,
                            buffer_size: write.buffer.size(),
                            usage: write.buffer.usage,
                        },
                        BufferDescriptorLimits {
                            uniform_alignment: self
                                ._pool
                                .device
                                .min_uniform_buffer_offset_alignment,
                            storage_alignment: self
                                ._pool
                                .device
                                .min_storage_buffer_offset_alignment,
                            max_uniform_range: self._pool.device.max_uniform_buffer_range,
                            max_storage_range: self._pool.device.max_storage_buffer_range,
                        },
                    )?;
                    let range = if write.range == vk::WHOLE_SIZE {
                        write.buffer.size() - write.offset
                    } else {
                        write.range
                    };
                    if matches!(
                        binding.descriptor_type,
                        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                            | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    ) {
                        dynamic_offsets.push(DynamicOffsetRequirement {
                            binding: binding_number,
                            array_element,
                            descriptor_type: binding.descriptor_type,
                            base_offset: write.offset,
                            range,
                            buffer_size: write.buffer.size(),
                        });
                    }
                    let index = buffer_infos.len();
                    buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(write.buffer.handle())
                            .offset(write.offset)
                            .range(range),
                    );
                    prepared.push(PreparedDescriptorWrite {
                        binding: binding_number,
                        array_element,
                        descriptor_type: binding.descriptor_type,
                        info: DescriptorInfoIndex::Buffer(index),
                    });
                    retained_resources.push(RetainedDescriptorResource::Buffer(write.buffer));
                }
                DescriptorResourceWrite::Image(write) => {
                    if !std::ptr::eq(self._pool.device, write.view.image().device) {
                        return Err(VulkanError::DescriptorSetDeviceMismatch);
                    }
                    if write
                        .sampler
                        .is_some_and(|sampler| !std::ptr::eq(self._pool.device, sampler.device))
                    {
                        return Err(VulkanError::DescriptorSetDeviceMismatch);
                    }
                    validate_image_descriptor_write(self._pool.device, binding, write)?;
                    let index = image_infos.len();
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .sampler(write.sampler.map_or(vk::Sampler::null(), |s| s.handle()))
                            .image_view(write.view.handle())
                            .image_layout(write.image_layout),
                    );
                    prepared.push(PreparedDescriptorWrite {
                        binding: binding_number,
                        array_element,
                        descriptor_type: binding.descriptor_type,
                        info: DescriptorInfoIndex::Image(index),
                    });
                    retained_resources.push(RetainedDescriptorResource::ImageView(write.view));
                    if let Some(sampler) = write.sampler {
                        retained_resources.push(RetainedDescriptorResource::Sampler(sampler));
                    }
                }
                DescriptorResourceWrite::Sampler(write) => {
                    validate_sampler_descriptor_binding(binding)?;
                    if !std::ptr::eq(self._pool.device, write.sampler.device) {
                        return Err(VulkanError::DescriptorSetDeviceMismatch);
                    }
                    let index = image_infos.len();
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .sampler(write.sampler.handle())
                            .image_view(vk::ImageView::null())
                            .image_layout(vk::ImageLayout::UNDEFINED),
                    );
                    prepared.push(PreparedDescriptorWrite {
                        binding: binding_number,
                        array_element,
                        descriptor_type: binding.descriptor_type,
                        info: DescriptorInfoIndex::Image(index),
                    });
                    retained_resources.push(RetainedDescriptorResource::Sampler(write.sampler));
                }
            }
        }
        validate_complete_descriptor_write_set(&self.layout.config, &seen)?;
        dynamic_offsets
            .sort_unstable_by_key(|requirement| (requirement.binding, requirement.array_element));

        let descriptor_writes: Vec<_> = prepared
            .iter()
            .map(|write| {
                let base = vk::WriteDescriptorSet::default()
                    .dst_set(self.handle)
                    .dst_binding(write.binding)
                    .dst_array_element(write.array_element)
                    .descriptor_type(write.descriptor_type);
                match write.info {
                    DescriptorInfoIndex::Buffer(index) => {
                        base.buffer_info(std::slice::from_ref(&buffer_infos[index]))
                    }
                    DescriptorInfoIndex::Image(index) => {
                        base.image_info(std::slice::from_ref(&image_infos[index]))
                    }
                }
            })
            .collect();

        // SAFETY: Each descriptor slot is unique and complete. Resource device
        // identities, descriptor types, usages, ranges, image layouts, and sampler
        // requirements were validated above. Info arrays remain alive for the call.
        unsafe {
            self._pool
                .device
                .handle
                .update_descriptor_sets(&descriptor_writes, &[]);
        }

        Ok(DescriptorResourceLease {
            set: self,
            retained_resources,
            dynamic_offsets,
        })
    }

    /// Updates a descriptor set containing only uniform/storage buffer descriptors.
    ///
    /// This convenience wrapper uses the same complete-update and resource-lease
    /// contract as [`Self::write_descriptors`].
    pub fn write_buffer_descriptors<'set, 'resource>(
        &'set mut self,
        writes: &[BufferDescriptorWrite<'resource, 'device>],
    ) -> Result<
        DescriptorResourceLease<'set, 'resource, 'resource, 'pool, 'layout, 'device>,
        VulkanError,
    > {
        let resources: Vec<_> = writes
            .iter()
            .copied()
            .map(DescriptorResourceWrite::Buffer)
            .collect();
        self.write_descriptors(&resources)
    }

    /// Returns the layout used to allocate this descriptor set.
    #[must_use]
    pub const fn layout(&self) -> &'layout VulkanDescriptorSetLayout<'device> {
        self.layout
    }
}

/// Keeps every resource referenced by a descriptor set alive while it is in use.
///
/// This guard prevents descriptor updates and retains referenced resources. Safe
/// command buffers retain the guard through submission completion, so dropping the
/// final lease does not require a device-wide idle wait.
pub struct DescriptorResourceLease<'set, 'resource, 'image, 'pool, 'layout, 'device> {
    set: &'set mut VulkanDescriptorSet<'pool, 'layout, 'device>,
    retained_resources: Vec<RetainedDescriptorResource<'resource, 'image, 'device>>,
    dynamic_offsets: Vec<DynamicOffsetRequirement>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DynamicOffsetRequirement {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_type: vk::DescriptorType,
    pub base_offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
    pub buffer_size: vk::DeviceSize,
}

trait DescriptorLeaseMarker {}
impl DescriptorLeaseMarker for DescriptorResourceLease<'_, '_, '_, '_, '_, '_> {}

/// A borrowable descriptor-set lease reference for command-buffer recording.
#[derive(Clone, Copy)]
pub struct DescriptorSetLeaseRef<'lease> {
    _lease: &'lease dyn DescriptorLeaseMarker,
    pub(super) device: &'lease VulkanDevice,
    pub(super) handle: vk::DescriptorSet,
    pub(super) layout_handle: vk::DescriptorSetLayout,
    pub(super) dynamic_offsets: &'lease [DynamicOffsetRequirement],
}

enum RetainedDescriptorResource<'resource, 'image, 'device> {
    Buffer(&'resource VulkanBuffer<'device>),
    ImageView(&'resource VulkanImageView<'image, 'device>),
    Sampler(&'resource VulkanSampler<'device>),
}

enum DescriptorInfoIndex {
    Buffer(usize),
    Image(usize),
}

struct PreparedDescriptorWrite {
    binding: u32,
    array_element: u32,
    descriptor_type: vk::DescriptorType,
    info: DescriptorInfoIndex,
}

impl DescriptorResourceLease<'_, '_, '_, '_, '_, '_> {
    /// Creates a lightweight borrow that keeps this descriptor lease alive while recording/submitting commands.
    #[must_use]
    pub fn as_ref(&self) -> DescriptorSetLeaseRef<'_> {
        DescriptorSetLeaseRef {
            _lease: self,
            device: self.set._pool.device,
            handle: self.set.handle,
            layout_handle: self.set.layout.handle(),
            dynamic_offsets: &self.dynamic_offsets,
        }
    }

    /// Returns the descriptor-set handle while its referenced resources are leased.
    #[must_use]
    pub const fn handle(&self) -> vk::DescriptorSet {
        self.set.handle
    }

    /// Returns the number of retained buffer resources.
    #[must_use]
    pub fn retained_buffer_count(&self) -> usize {
        self.retained_resources
            .iter()
            .filter_map(|resource| match resource {
                RetainedDescriptorResource::Buffer(buffer) => {
                    let _ = buffer.handle();
                    Some(())
                }
                _ => None,
            })
            .count()
    }

    /// Returns the number of retained image views.
    #[must_use]
    pub fn retained_image_view_count(&self) -> usize {
        self.retained_resources
            .iter()
            .filter_map(|resource| match resource {
                RetainedDescriptorResource::ImageView(view) => {
                    let _ = view.handle();
                    Some(())
                }
                _ => None,
            })
            .count()
    }

    /// Returns the number of retained samplers.
    #[must_use]
    pub fn retained_sampler_count(&self) -> usize {
        self.retained_resources
            .iter()
            .filter_map(|resource| match resource {
                RetainedDescriptorResource::Sampler(sampler) => {
                    let _ = sampler.handle();
                    Some(())
                }
                _ => None,
            })
            .count()
    }
}

impl Drop for DescriptorResourceLease<'_, '_, '_, '_, '_, '_> {
    fn drop(&mut self) {}
}

fn validate_complete_descriptor_write_set(
    layout: &DescriptorSetLayoutConfig,
    seen: &HashSet<(u32, u32)>,
) -> Result<(), VulkanError> {
    let mut expected_count = 0_usize;
    for binding in &layout.bindings {
        if !matches!(
            binding.descriptor_type,
            vk::DescriptorType::UNIFORM_BUFFER
                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                | vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                | vk::DescriptorType::SAMPLED_IMAGE
                | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                | vk::DescriptorType::STORAGE_IMAGE
                | vk::DescriptorType::SAMPLER
        ) {
            return Err(VulkanError::UnsupportedDescriptorType);
        }
        expected_count = expected_count
            .checked_add(binding.descriptor_count as usize)
            .ok_or(VulkanError::IncompleteDescriptorWrite)?;
        for array_element in 0..binding.descriptor_count {
            if !seen.contains(&(binding.binding, array_element)) {
                return Err(VulkanError::IncompleteDescriptorWrite);
            }
        }
    }
    if seen.len() != expected_count {
        return Err(VulkanError::IncompleteDescriptorWrite);
    }
    Ok(())
}

fn validate_sampler_descriptor_binding(binding: &DescriptorBinding) -> Result<(), VulkanError> {
    if binding.descriptor_type == vk::DescriptorType::SAMPLER {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedDescriptorType)
    }
}

fn validate_image_descriptor_write(
    device: &VulkanDevice,
    binding: &DescriptorBinding,
    write: &ImageDescriptorWrite<'_, '_, '_>,
) -> Result<(), VulkanError> {
    let image = write.view.image();
    validate_image_descriptor_metadata(
        binding,
        write.sampler.is_some(),
        image.config().usage,
        write.image_layout,
        write.view.config().aspect,
    )?;

    let required_feature = match binding.descriptor_type {
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER | vk::DescriptorType::SAMPLED_IMAGE => {
            vk::FormatFeatureFlags::SAMPLED_IMAGE
        }
        vk::DescriptorType::STORAGE_IMAGE => vk::FormatFeatureFlags::STORAGE_IMAGE,
        _ => return Err(VulkanError::UnsupportedDescriptorType),
    };
    // SAFETY: The physical device is owned by the instance retained by this live device.
    let format_properties = unsafe {
        device
            ._instance
            .handle
            .get_physical_device_format_properties(device.physical_device, image.config().format)
    };
    let available_features = if image.config().tiling == vk::ImageTiling::LINEAR {
        format_properties.linear_tiling_features
    } else {
        format_properties.optimal_tiling_features
    };
    validate_image_format_feature(required_feature, available_features)
}

fn validate_image_format_feature(
    required: vk::FormatFeatureFlags,
    available: vk::FormatFeatureFlags,
) -> Result<(), VulkanError> {
    if available.contains(required) {
        Ok(())
    } else {
        Err(VulkanError::UnsupportedImageFormatFeature)
    }
}

fn validate_image_descriptor_metadata(
    binding: &DescriptorBinding,
    has_sampler: bool,
    usage: vk::ImageUsageFlags,
    image_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) -> Result<(), VulkanError> {
    if image_layout == vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        && !aspect.intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
    {
        return Err(VulkanError::InvalidImageDescriptor);
    }
    match binding.descriptor_type {
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
            if !has_sampler {
                return Err(VulkanError::DescriptorSamplerRequired);
            }
            if !usage.contains(vk::ImageUsageFlags::SAMPLED)
                || !matches!(
                    image_layout,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        | vk::ImageLayout::GENERAL
                )
            {
                return Err(VulkanError::InvalidImageDescriptor);
            }
        }
        vk::DescriptorType::SAMPLED_IMAGE => {
            if has_sampler
                || !usage.contains(vk::ImageUsageFlags::SAMPLED)
                || !matches!(
                    image_layout,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        | vk::ImageLayout::GENERAL
                )
            {
                return Err(VulkanError::InvalidImageDescriptor);
            }
        }
        vk::DescriptorType::STORAGE_IMAGE => {
            if has_sampler
                || !usage.contains(vk::ImageUsageFlags::STORAGE)
                || image_layout != vk::ImageLayout::GENERAL
            {
                return Err(VulkanError::InvalidImageDescriptor);
            }
        }
        _ => return Err(VulkanError::UnsupportedDescriptorType),
    }
    Ok(())
}

fn validate_unique_descriptor_write(
    seen: &mut HashSet<(u32, u32)>,
    binding: u32,
    array_element: u32,
) -> Result<(), VulkanError> {
    if seen.insert((binding, array_element)) {
        Ok(())
    } else {
        Err(VulkanError::DuplicateDescriptorWrite)
    }
}

#[derive(Clone, Copy)]
struct BufferDescriptorMetadata {
    array_element: u32,
    offset: vk::DeviceSize,
    range: vk::DeviceSize,
    buffer_size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
}

#[derive(Clone, Copy)]
struct BufferDescriptorLimits {
    uniform_alignment: vk::DeviceSize,
    storage_alignment: vk::DeviceSize,
    max_uniform_range: vk::DeviceSize,
    max_storage_range: vk::DeviceSize,
}

fn validate_buffer_descriptor_write(
    binding: &DescriptorBinding,
    metadata: BufferDescriptorMetadata,
    limits: BufferDescriptorLimits,
) -> Result<(), VulkanError> {
    if metadata.array_element >= binding.descriptor_count {
        return Err(VulkanError::DescriptorArrayElementOutOfBounds);
    }

    let (required_usage, alignment, max_range) = match binding.descriptor_type {
        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => (
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            limits.uniform_alignment,
            limits.max_uniform_range,
        ),
        vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => (
            vk::BufferUsageFlags::STORAGE_BUFFER,
            limits.storage_alignment,
            limits.max_storage_range,
        ),
        _ => return Err(VulkanError::UnsupportedBufferDescriptorType),
    };
    if !metadata.usage.contains(required_usage) {
        return Err(VulkanError::BufferDescriptorUsageMismatch);
    }
    if metadata.offset > metadata.buffer_size || (alignment > 1 && metadata.offset % alignment != 0)
    {
        return Err(VulkanError::InvalidBufferDescriptorRange);
    }
    let effective_range = if metadata.range == vk::WHOLE_SIZE {
        metadata.buffer_size - metadata.offset
    } else {
        metadata.range
    };
    let Some(end) = metadata.offset.checked_add(effective_range) else {
        return Err(VulkanError::InvalidBufferDescriptorRange);
    };
    if effective_range == 0 || end > metadata.buffer_size || effective_range > max_range {
        return Err(VulkanError::InvalidBufferDescriptorRange);
    }
    Ok(())
}

fn validate_layout_config(config: &DescriptorSetLayoutConfig) -> Result<(), VulkanError> {
    for (index, binding) in config.bindings.iter().enumerate() {
        if binding.descriptor_count == 0 || binding.stage_flags.is_empty() {
            return Err(VulkanError::InvalidDescriptorSetLayoutBinding);
        }
        if config.bindings[..index]
            .iter()
            .any(|previous| previous.binding == binding.binding)
        {
            return Err(VulkanError::InvalidDescriptorSetLayoutBinding);
        }
    }
    Ok(())
}

fn validate_pool_config(config: &DescriptorPoolConfig) -> Result<(), VulkanError> {
    if config.max_sets == 0 || config.pool_sizes.is_empty() {
        return Err(VulkanError::InvalidDescriptorPoolConfig);
    }
    for (index, size) in config.pool_sizes.iter().enumerate() {
        if size.descriptor_count == 0
            || config.pool_sizes[..index]
                .iter()
                .any(|previous| previous.descriptor_type == size.descriptor_type)
        {
            return Err(VulkanError::InvalidDescriptorPoolConfig);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BufferDescriptorLimits, BufferDescriptorMetadata, DescriptorBinding, DescriptorPoolConfig,
        DescriptorPoolSize, DescriptorSetLayoutConfig, validate_buffer_descriptor_write,
        validate_complete_descriptor_write_set, validate_image_descriptor_metadata,
        validate_image_format_feature, validate_layout_config, validate_pool_config,
        validate_sampler_descriptor_binding, validate_unique_descriptor_write,
    };
    use crate::VulkanError;
    use ash::vk;
    use std::collections::HashSet;

    fn uniform_binding() -> DescriptorBinding {
        DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 2,
            stage_flags: vk::ShaderStageFlags::VERTEX,
        }
    }

    #[test]
    fn requires_every_buffer_descriptor_slot_to_be_rewritten() {
        let layout = DescriptorSetLayoutConfig {
            bindings: vec![uniform_binding()],
        };
        assert!(
            validate_complete_descriptor_write_set(&layout, &HashSet::from([(0, 0), (0, 1)]))
                .is_ok()
        );
        assert!(matches!(
            validate_complete_descriptor_write_set(&layout, &HashSet::from([(0, 0)])),
            Err(VulkanError::IncompleteDescriptorWrite)
        ));
        let mixed_layout = DescriptorSetLayoutConfig {
            bindings: vec![
                uniform_binding(),
                DescriptorBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                },
            ],
        };
        assert!(matches!(
            validate_complete_descriptor_write_set(&mixed_layout, &HashSet::from([(0, 0), (0, 1)]),),
            Err(VulkanError::IncompleteDescriptorWrite)
        ));
        assert!(
            validate_complete_descriptor_write_set(
                &mixed_layout,
                &HashSet::from([(0, 0), (0, 1), (1, 0)]),
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_duplicate_binding_array_element_updates() {
        let mut seen = HashSet::new();
        assert!(validate_unique_descriptor_write(&mut seen, 2, 0).is_ok());
        assert!(validate_unique_descriptor_write(&mut seen, 2, 1).is_ok());
        assert!(matches!(
            validate_unique_descriptor_write(&mut seen, 2, 0),
            Err(VulkanError::DuplicateDescriptorWrite)
        ));
    }

    #[test]
    fn validates_uniform_buffer_descriptor_usage_alignment_and_bounds() {
        assert!(
            validate_buffer_descriptor_write(
                &uniform_binding(),
                BufferDescriptorMetadata {
                    array_element: 1,
                    offset: 256,
                    range: 512,
                    buffer_size: 1024,
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                },
                BufferDescriptorLimits {
                    uniform_alignment: 256,
                    storage_alignment: 16,
                    max_uniform_range: 65536,
                    max_storage_range: 1 << 20,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn validates_storage_buffer_alignment_and_dynamic_descriptor_type() {
        let binding = DescriptorBinding {
            binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        };
        assert!(
            validate_buffer_descriptor_write(
                &binding,
                BufferDescriptorMetadata {
                    array_element: 0,
                    offset: 32,
                    range: 128,
                    buffer_size: 256,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                },
                BufferDescriptorLimits {
                    uniform_alignment: 256,
                    storage_alignment: 32,
                    max_uniform_range: 65536,
                    max_storage_range: 256,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn accepts_whole_size_when_the_remaining_buffer_range_is_valid() {
        assert!(
            validate_buffer_descriptor_write(
                &uniform_binding(),
                BufferDescriptorMetadata {
                    array_element: 0,
                    offset: 256,
                    range: vk::WHOLE_SIZE,
                    buffer_size: 1024,
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                },
                BufferDescriptorLimits {
                    uniform_alignment: 256,
                    storage_alignment: 16,
                    max_uniform_range: 768,
                    max_storage_range: 1 << 20,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_invalid_buffer_descriptor_array_usage_alignment_and_range() {
        let binding = uniform_binding();
        let cases = [
            (2, 0, 256, 1024, vk::BufferUsageFlags::UNIFORM_BUFFER),
            (0, 0, 256, 1024, vk::BufferUsageFlags::STORAGE_BUFFER),
            (0, 4, 256, 1024, vk::BufferUsageFlags::UNIFORM_BUFFER),
            (0, 256, 0, 1024, vk::BufferUsageFlags::UNIFORM_BUFFER),
            (0, 768, 512, 1024, vk::BufferUsageFlags::UNIFORM_BUFFER),
        ];
        for (element, offset, range, size, usage) in cases {
            assert!(matches!(
                validate_buffer_descriptor_write(
                    &binding,
                    BufferDescriptorMetadata {
                        array_element: element,
                        offset,
                        range,
                        buffer_size: size,
                        usage,
                    },
                    BufferDescriptorLimits {
                        uniform_alignment: 256,
                        storage_alignment: 16,
                        max_uniform_range: 65536,
                        max_storage_range: 1 << 20,
                    },
                ),
                Err(VulkanError::DescriptorArrayElementOutOfBounds)
                    | Err(VulkanError::BufferDescriptorUsageMismatch)
                    | Err(VulkanError::InvalidBufferDescriptorRange)
            ));
        }
    }

    #[test]
    fn rejects_unsupported_descriptor_types_and_ranges_over_device_limit() {
        let sampler_binding = DescriptorBinding {
            descriptor_type: vk::DescriptorType::SAMPLER,
            ..uniform_binding()
        };
        assert!(matches!(
            validate_buffer_descriptor_write(
                &sampler_binding,
                BufferDescriptorMetadata {
                    array_element: 0,
                    offset: 0,
                    range: 16,
                    buffer_size: 1024,
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                },
                BufferDescriptorLimits {
                    uniform_alignment: 1,
                    storage_alignment: 1,
                    max_uniform_range: 8,
                    max_storage_range: 8,
                },
            ),
            Err(VulkanError::UnsupportedBufferDescriptorType)
        ));
        assert!(matches!(
            validate_buffer_descriptor_write(
                &uniform_binding(),
                BufferDescriptorMetadata {
                    array_element: 0,
                    offset: 0,
                    range: 16,
                    buffer_size: 1024,
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                },
                BufferDescriptorLimits {
                    uniform_alignment: 1,
                    storage_alignment: 1,
                    max_uniform_range: 8,
                    max_storage_range: 1024,
                },
            ),
            Err(VulkanError::InvalidBufferDescriptorRange)
        ));
    }

    #[test]
    fn validates_combined_sampled_and_storage_image_descriptor_contracts() {
        let combined = DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
        };
        assert!(
            validate_image_descriptor_metadata(
                &combined,
                true,
                vk::ImageUsageFlags::SAMPLED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_image_descriptor_metadata(
                &combined,
                false,
                vk::ImageUsageFlags::SAMPLED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            ),
            Err(VulkanError::DescriptorSamplerRequired)
        ));

        let sampled = DescriptorBinding {
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            ..combined
        };
        assert!(
            validate_image_descriptor_metadata(
                &sampled,
                false,
                vk::ImageUsageFlags::SAMPLED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_image_descriptor_metadata(
                &sampled,
                true,
                vk::ImageUsageFlags::SAMPLED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            ),
            Err(VulkanError::InvalidImageDescriptor)
        ));

        assert!(matches!(
            validate_image_descriptor_metadata(
                &sampled,
                false,
                vk::ImageUsageFlags::SAMPLED,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            ),
            Err(VulkanError::InvalidImageDescriptor)
        ));

        let storage = DescriptorBinding {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ..combined
        };
        assert!(
            validate_image_descriptor_metadata(
                &storage,
                false,
                vk::ImageUsageFlags::STORAGE,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_image_descriptor_metadata(
                &storage,
                false,
                vk::ImageUsageFlags::SAMPLED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            ),
            Err(VulkanError::InvalidImageDescriptor)
        ));
    }

    #[test]
    fn validates_sampler_only_descriptor_bindings() {
        let sampler = DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
        };
        assert!(validate_sampler_descriptor_binding(&sampler).is_ok());
        assert!(matches!(
            validate_sampler_descriptor_binding(&uniform_binding()),
            Err(VulkanError::UnsupportedDescriptorType)
        ));
    }

    #[test]
    fn validates_physical_device_image_format_features() {
        assert!(
            validate_image_format_feature(
                vk::FormatFeatureFlags::SAMPLED_IMAGE,
                vk::FormatFeatureFlags::SAMPLED_IMAGE
                    | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_image_format_feature(
                vk::FormatFeatureFlags::STORAGE_IMAGE,
                vk::FormatFeatureFlags::SAMPLED_IMAGE,
            ),
            Err(VulkanError::UnsupportedImageFormatFeature)
        ));
    }

    #[test]
    fn accepts_empty_descriptor_set_layout() {
        assert!(validate_layout_config(&DescriptorSetLayoutConfig::default()).is_ok());
    }

    #[test]
    fn accepts_valid_descriptor_set_layout_bindings() {
        let config = DescriptorSetLayoutConfig {
            bindings: vec![
                DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                },
                DescriptorBinding {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 4,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                },
            ],
        };
        assert!(validate_layout_config(&config).is_ok());
    }

    #[test]
    fn rejects_zero_count_empty_stages_and_duplicate_bindings() {
        let invalid = [
            DescriptorSetLayoutConfig {
                bindings: vec![DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 0,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                }],
            },
            DescriptorSetLayoutConfig {
                bindings: vec![DescriptorBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::empty(),
                }],
            },
            DescriptorSetLayoutConfig {
                bindings: vec![
                    DescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX,
                    },
                    DescriptorBinding {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                ],
            },
        ];
        for config in invalid {
            assert!(matches!(
                validate_layout_config(&config),
                Err(VulkanError::InvalidDescriptorSetLayoutBinding)
            ));
        }
    }

    #[test]
    fn accepts_valid_descriptor_pool_configuration() {
        let config = DescriptorPoolConfig {
            max_sets: 8,
            pool_sizes: vec![
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 8,
                },
                DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 16,
                },
            ],
        };
        assert!(validate_pool_config(&config).is_ok());
    }

    #[test]
    fn rejects_empty_or_invalid_descriptor_pool_configuration() {
        let invalid = [
            DescriptorPoolConfig {
                max_sets: 0,
                pool_sizes: vec![DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                }],
            },
            DescriptorPoolConfig {
                max_sets: 1,
                pool_sizes: vec![],
            },
            DescriptorPoolConfig {
                max_sets: 1,
                pool_sizes: vec![DescriptorPoolSize {
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 0,
                }],
            },
            DescriptorPoolConfig {
                max_sets: 1,
                pool_sizes: vec![
                    DescriptorPoolSize {
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                    },
                    DescriptorPoolSize {
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 2,
                    },
                ],
            },
        ];
        for config in invalid {
            assert!(matches!(
                validate_pool_config(&config),
                Err(VulkanError::InvalidDescriptorPoolConfig)
            ));
        }
    }
}
