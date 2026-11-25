//! Explicitly owned Vulkan descriptor layouts and pools.
use ash::vk;

use crate::{VulkanBuffer, VulkanDevice, VulkanError, api_error};

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
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before descriptor-set-layout destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns the layout and its device is still borrowed.
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
    /// The descriptor set is automatically released with its pool. This initial
    /// API intentionally exposes allocation only; resource writes are handled by
    /// a later descriptor-update layer with explicit resource-lifetime contracts.
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
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before descriptor-pool destruction"
            );
        }
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
    /// Updates uniform/storage buffer descriptors and returns a resource lease.
    ///
    /// The lease retains every referenced buffer and exclusively borrows this set,
    /// preventing another update while the descriptors may be in use. This initial
    /// buffer-only path requires every binding and every array element in the layout
    /// to be a supported uniform/storage buffer descriptor and rewritten in this batch.
    /// This avoids stale descriptor slots outliving the resources from a previous lease.
    /// Keep the
    /// lease alive through all submissions that use the set. Dropping the lease
    /// waits for the device to become idle before releasing its buffer references.
    /// Descriptor updates do not perform queue synchronization or buffer barriers;
    /// callers must establish the required memory dependencies separately. For
    /// dynamic buffer descriptor types, the eventual command-binding layer must
    /// validate dynamic offsets and the final effective buffer range.
    pub fn write_buffer_descriptors<'set, 'resource>(
        &'set mut self,
        writes: &[BufferDescriptorWrite<'resource, 'device>],
    ) -> Result<DescriptorResourceLease<'set, 'resource, 'pool, 'layout, 'device>, VulkanError>
    {
        let mut seen = Vec::with_capacity(writes.len());
        let mut buffer_infos = Vec::with_capacity(writes.len());
        let mut descriptor_types = Vec::with_capacity(writes.len());
        let mut retained_buffers = Vec::with_capacity(writes.len());

        for write in writes {
            validate_unique_descriptor_write(&mut seen, write.binding, write.array_element)?;

            let binding = self
                .layout
                .config
                .bindings
                .iter()
                .find(|binding| binding.binding == write.binding)
                .ok_or(VulkanError::DescriptorBindingNotFound)?;
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
                    uniform_alignment: self._pool.device.min_uniform_buffer_offset_alignment,
                    storage_alignment: self._pool.device.min_storage_buffer_offset_alignment,
                    max_uniform_range: self._pool.device.max_uniform_buffer_range,
                    max_storage_range: self._pool.device.max_storage_buffer_range,
                },
            )?;

            let range = if write.range == vk::WHOLE_SIZE {
                write.buffer.size() - write.offset
            } else {
                write.range
            };
            descriptor_types.push(binding.descriptor_type);
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(write.buffer.handle())
                    .offset(write.offset)
                    .range(range),
            );
            retained_buffers.push(write.buffer);
        }
        validate_complete_buffer_write_set(&self.layout.config, &seen)?;

        let descriptor_writes: Vec<_> = writes
            .iter()
            .zip(&buffer_infos)
            .zip(&descriptor_types)
            .map(|((write, info), descriptor_type)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.handle)
                    .dst_binding(write.binding)
                    .dst_array_element(write.array_element)
                    .descriptor_type(*descriptor_type)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        // SAFETY: All descriptor types, binding indices, array elements, buffer
        // ranges, usages, and device identities were validated above. The info
        // vectors remain alive and unmoved for the duration of this call.
        unsafe {
            self._pool
                .device
                .handle
                .update_descriptor_sets(&descriptor_writes, &[]);
        }

        Ok(DescriptorResourceLease {
            set: self,
            retained_buffers,
        })
    }

    /// Returns the layout used to allocate this descriptor set.
    #[must_use]
    pub const fn layout(&self) -> &'layout VulkanDescriptorSetLayout<'device> {
        self.layout
    }
}

/// Keeps a descriptor set's referenced buffers alive while the set is in use.
///
/// This guard also prevents descriptor updates until it is dropped. Drop waits
/// for the owning device to become idle before releasing retained buffers.
pub struct DescriptorResourceLease<'set, 'resource, 'pool, 'layout, 'device> {
    set: &'set mut VulkanDescriptorSet<'pool, 'layout, 'device>,
    retained_buffers: Vec<&'resource VulkanBuffer<'device>>,
}

impl DescriptorResourceLease<'_, '_, '_, '_, '_> {
    /// Returns the descriptor-set handle while its referenced resources are leased.
    #[must_use]
    pub const fn handle(&self) -> vk::DescriptorSet {
        self.set.handle
    }

    /// Returns the number of buffer resources retained by this lease.
    #[must_use]
    pub fn retained_buffer_count(&self) -> usize {
        self.retained_buffers.len()
    }
}

impl Drop for DescriptorResourceLease<'_, '_, '_, '_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.set._pool.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before descriptor resource lease release"
            );
        }
    }
}

fn validate_complete_buffer_write_set(
    layout: &DescriptorSetLayoutConfig,
    seen: &[(u32, u32)],
) -> Result<(), VulkanError> {
    let mut expected_count = 0_usize;
    for binding in &layout.bindings {
        if !matches!(
            binding.descriptor_type,
            vk::DescriptorType::UNIFORM_BUFFER
                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                | vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        ) {
            return Err(VulkanError::UnsupportedBufferDescriptorType);
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

fn validate_unique_descriptor_write(
    seen: &mut Vec<(u32, u32)>,
    binding: u32,
    array_element: u32,
) -> Result<(), VulkanError> {
    if seen.contains(&(binding, array_element)) {
        return Err(VulkanError::DuplicateDescriptorWrite);
    }
    seen.push((binding, array_element));
    Ok(())
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
        validate_complete_buffer_write_set, validate_layout_config, validate_pool_config,
        validate_unique_descriptor_write,
    };
    use crate::VulkanError;
    use ash::vk;

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
        assert!(validate_complete_buffer_write_set(&layout, &[(0, 0), (0, 1)]).is_ok());
        assert!(matches!(
            validate_complete_buffer_write_set(&layout, &[(0, 0)]),
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
            validate_complete_buffer_write_set(&mixed_layout, &[(0, 0), (0, 1)]),
            Err(VulkanError::UnsupportedBufferDescriptorType)
        ));
    }

    #[test]
    fn rejects_duplicate_binding_array_element_updates() {
        let mut seen = Vec::new();
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
