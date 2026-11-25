//! Explicitly owned Vulkan buffer and backing-memory resources.
use crate::{VulkanDevice, VulkanError, api_error};
use ash::vk;
use std::ptr;

/// Memory placement policy for a buffer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferMemoryLocation {
    /// Requires device-local memory for GPU resources. Host access is not guaranteed.
    DeviceLocal,
    /// Host-visible memory for CPU-to-GPU uploads.
    Upload,
    /// Host-visible, preferably cached memory for GPU-to-CPU readback.
    Readback,
}

/// Buffer creation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferConfig {
    /// Buffer size in bytes; must be nonzero.
    pub size: vk::DeviceSize,
    /// Required Vulkan buffer usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Requested memory placement policy.
    pub memory: BufferMemoryLocation,
}

/// Owns a Vulkan buffer and its dedicated memory allocation.
///
/// Host access is bounds checked and handles non-coherent cache maintenance.
/// Device-local allocations are not guaranteed to be host-visible. Upload and
/// readback allocations require host-visible memory.
///
/// Read and write operations do not synchronize queues or insert memory barriers;
/// callers must establish the required GPU/host dependencies. For non-coherent
/// memory, synchronization must account for the atom-aligned cache-maintenance range,
/// which can extend beyond the requested byte range. Drop waits for device idle
/// before destroying the buffer and memory; this is deliberately conservative until
/// deferred resource retirement exists.
pub struct VulkanBuffer<'device> {
    pub(super) device: &'device VulkanDevice,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    pub(super) usage: vk::BufferUsageFlags,
    allocation_size: vk::DeviceSize,
    memory_flags: vk::MemoryPropertyFlags,
    atom_size: vk::DeviceSize,
    location: BufferMemoryLocation,
}

impl<'device> VulkanBuffer<'device> {
    /// Creates a buffer and dedicated allocation on the supplied device.
    pub fn new(device: &'device VulkanDevice, config: BufferConfig) -> Result<Self, VulkanError> {
        if config.size == 0 {
            return Err(VulkanError::InvalidBufferSize);
        }
        if config.usage.is_empty() {
            return Err(VulkanError::InvalidBufferUsage);
        }
        let info = vk::BufferCreateInfo::default()
            .size(config.size)
            .usage(config.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: The device is live and the create info contains no borrowed pointers.
        let handle = unsafe {
            device
                .handle
                .create_buffer(&info, None)
                .map_err(|error| api_error("create Vulkan buffer", error))?
        };
        // SAFETY: The buffer belongs to this live device.
        let requirements = unsafe { device.handle.get_buffer_memory_requirements(handle) };
        // SAFETY: These physical-device metadata queries use the retained instance/device.
        let properties = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_memory_properties(device.physical_device)
        };
        // SAFETY: These physical-device metadata queries use the retained instance/device.
        let limits = unsafe {
            device
                ._instance
                .handle
                .get_physical_device_properties(device.physical_device)
        }
        .limits;

        let (required, preferred) = memory_preferences(config.memory);
        let Some(memory_type_index) = select_memory_type(
            &properties,
            requirements.memory_type_bits,
            required,
            preferred,
        ) else {
            // SAFETY: The buffer is uniquely owned and has no bound memory.
            unsafe { device.handle.destroy_buffer(handle, None) };
            return Err(VulkanError::NoCompatibleMemoryType { required });
        };
        let memory_flags = properties.memory_types[memory_type_index as usize].property_flags;
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        // SAFETY: Allocation size and memory type come from this buffer's requirements.
        let memory = match unsafe { device.handle.allocate_memory(&allocation_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                // SAFETY: The buffer has no bound memory.
                unsafe { device.handle.destroy_buffer(handle, None) };
                return Err(api_error("allocate Vulkan buffer memory", error));
            }
        };
        // SAFETY: The selected memory type satisfies the buffer's memory requirements.
        if let Err(error) = unsafe { device.handle.bind_buffer_memory(handle, memory, 0) } {
            // SAFETY: Binding failed; neither object is in use by submitted work.
            unsafe {
                device.handle.free_memory(memory, None);
                device.handle.destroy_buffer(handle, None);
            }
            return Err(api_error("bind Vulkan buffer memory", error));
        }
        Ok(Self {
            device,
            handle,
            memory,
            size: config.size,
            usage: config.usage,
            allocation_size: requirements.size,
            memory_flags,
            atom_size: limits.non_coherent_atom_size.max(1),
            location: config.memory,
        })
    }

    /// Returns the raw handle for command recording and queue submission.
    #[must_use]
    pub const fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Returns the logical buffer size in bytes.
    #[must_use]
    pub const fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Returns the requested memory placement policy.
    #[must_use]
    pub const fn memory_location(&self) -> BufferMemoryLocation {
        self.location
    }

    /// Returns the actual Vulkan memory properties selected for the allocation.
    #[must_use]
    pub const fn memory_properties(&self) -> vk::MemoryPropertyFlags {
        self.memory_flags
    }

    /// Copies bytes into a host-visible buffer.
    ///
    /// This operation does not synchronize queue work or insert a Vulkan memory
    /// barrier. The caller must ensure the GPU is not accessing the target range,
    /// then establish the required device-side dependency before consuming the data.
    /// For non-coherent memory, synchronization must also account for the atom-aligned
    /// range used by cache maintenance, which can extend beyond the requested bytes.
    pub fn write(&mut self, offset: vk::DeviceSize, bytes: &[u8]) -> Result<(), VulkanError> {
        self.validate_range(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.ensure_host_visible()?;
        let pointer_offset = self.pointer_offset(offset, bytes.len())?;
        let byte_length =
            u64::try_from(bytes.len()).map_err(|_| VulkanError::BufferOffsetTooLarge)?;
        // SAFETY: The allocation is live and not mapped elsewhere through this exclusive borrow.
        let mapped = unsafe {
            self.device
                .handle
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|error| api_error("map Vulkan buffer memory for writing", error))?
        };
        // SAFETY: The byte range was bounds checked and the full allocation is mapped.
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.cast::<u8>().add(pointer_offset),
                bytes.len(),
            );
        }
        let result = self.cache_operation(offset, byte_length, true);
        // SAFETY: This allocation was successfully mapped above.
        unsafe { self.device.handle.unmap_memory(self.memory) };
        result
    }

    /// Copies bytes from a host-visible buffer.
    ///
    /// The caller must wait for all GPU writes to the requested range to complete
    /// and establish the required memory dependency before reading. This operation
    /// does not wait for queues or insert a Vulkan memory barrier. For non-coherent
    /// memory, synchronization must account for the atom-aligned invalidation range.
    pub fn read(&mut self, offset: vk::DeviceSize, output: &mut [u8]) -> Result<(), VulkanError> {
        self.validate_range(offset, output.len())?;
        if output.is_empty() {
            return Ok(());
        }
        self.ensure_host_visible()?;
        let pointer_offset = self.pointer_offset(offset, output.len())?;
        let byte_length =
            u64::try_from(output.len()).map_err(|_| VulkanError::BufferOffsetTooLarge)?;
        // SAFETY: The allocation is live and not mapped elsewhere through this exclusive borrow.
        let mapped = unsafe {
            self.device
                .handle
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|error| api_error("map Vulkan buffer memory for reading", error))?
        };
        if let Err(error) = self.cache_operation(offset, byte_length, false) {
            // SAFETY: This allocation was successfully mapped above.
            unsafe { self.device.handle.unmap_memory(self.memory) };
            return Err(error);
        }
        // SAFETY: The range was bounds checked and invalidated if memory is non-coherent.
        unsafe {
            ptr::copy_nonoverlapping(
                mapped.cast::<u8>().add(pointer_offset),
                output.as_mut_ptr(),
                output.len(),
            );
        }
        // SAFETY: This allocation was successfully mapped above.
        unsafe { self.device.handle.unmap_memory(self.memory) };
        Ok(())
    }

    fn validate_range(&self, offset: u64, length: usize) -> Result<(), VulkanError> {
        let length = u64::try_from(length).map_err(|_| VulkanError::BufferOffsetTooLarge)?;
        if offset.checked_add(length).is_none_or(|end| end > self.size) {
            return Err(VulkanError::BufferRangeOutOfBounds {
                offset,
                length,
                capacity: self.size,
            });
        }
        Ok(())
    }

    fn pointer_offset(&self, offset: u64, length: usize) -> Result<usize, VulkanError> {
        let length = u64::try_from(length).map_err(|_| VulkanError::BufferOffsetTooLarge)?;
        let end = offset
            .checked_add(length)
            .ok_or(VulkanError::BufferOffsetTooLarge)?;
        if end > isize::MAX as u64 {
            return Err(VulkanError::BufferOffsetTooLarge);
        }
        usize::try_from(offset).map_err(|_| VulkanError::BufferOffsetTooLarge)
    }

    fn ensure_host_visible(&self) -> Result<(), VulkanError> {
        if self
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            Ok(())
        } else {
            Err(VulkanError::BufferNotHostVisible)
        }
    }

    fn cache_operation(&self, offset: u64, length: u64, flush: bool) -> Result<(), VulkanError> {
        if self
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(());
        }
        let (range_offset, range_size) =
            mapped_memory_range(offset, length, self.allocation_size, self.atom_size)
                .ok_or(VulkanError::BufferMemoryRangeOverflow)?;
        let ranges = [vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(range_offset)
            .size(range_size)];
        // SAFETY: Memory is mapped; range is atom-aligned and within the allocation.
        let result = unsafe {
            if flush {
                self.device.handle.flush_mapped_memory_ranges(&ranges)
            } else {
                self.device.handle.invalidate_mapped_memory_ranges(&ranges)
            }
        };
        result.map_err(|error| {
            api_error(
                if flush {
                    "flush non-coherent Vulkan buffer memory"
                } else {
                    "invalidate non-coherent Vulkan buffer memory"
                },
                error,
            )
        })
    }
}

impl Drop for VulkanBuffer<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before buffer destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns both objects; the buffer is destroyed
        // before the memory bound to it is freed.
        unsafe {
            self.device.handle.destroy_buffer(self.handle, None);
            self.device.handle.free_memory(self.memory, None);
        }
    }
}

fn memory_preferences(
    location: BufferMemoryLocation,
) -> (vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) {
    match location {
        BufferMemoryLocation::DeviceLocal => (
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        ),
        BufferMemoryLocation::Upload => (
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ),
        BufferMemoryLocation::Readback => (
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
        ),
    }
}

fn select_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mut selected = None;
    let mut best_score = 0;
    for (index, memory_type) in properties
        .memory_types
        .iter()
        .enumerate()
        .take(properties.memory_type_count as usize)
    {
        if type_bits & (1_u32 << index) == 0 {
            continue;
        }
        let flags = memory_type.property_flags;
        if !flags.contains(required) {
            continue;
        }
        let score = (flags & preferred).as_raw().count_ones();
        if selected.is_none() || score > best_score {
            selected = Some(index as u32);
            best_score = score;
        }
    }
    selected
}

fn mapped_memory_range(
    offset: u64,
    length: u64,
    allocation_size: u64,
    atom_size: u64,
) -> Option<(u64, u64)> {
    if atom_size == 0 {
        return None;
    }
    let end = offset.checked_add(length)?;
    if end > allocation_size {
        return None;
    }
    let aligned_offset = (offset / atom_size) * atom_size;
    // If rounding up would overflow, the requested range is necessarily close to
    // the allocation end; extending to that end is valid because Vulkan permits a
    // non-atom-sized range when it reaches the end of the allocation.
    let aligned_end = match end.checked_add(atom_size - 1) {
        Some(rounded) => rounded
            .checked_div(atom_size)?
            .checked_mul(atom_size)?
            .min(allocation_size),
        None => allocation_size,
    };
    let range_size = if aligned_end == allocation_size {
        vk::WHOLE_SIZE
    } else {
        aligned_end.checked_sub(aligned_offset)?
    };
    Some((aligned_offset, range_size))
}

#[cfg(test)]
mod tests {
    use super::{mapped_memory_range, select_memory_type};
    use ash::vk;

    fn properties(flags: &[vk::MemoryPropertyFlags]) -> vk::PhysicalDeviceMemoryProperties {
        let mut value = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: flags.len() as u32,
            ..Default::default()
        };
        for (index, property_flags) in flags.iter().copied().enumerate() {
            value.memory_types[index] = vk::MemoryType {
                property_flags,
                heap_index: 0,
            };
        }
        value
    }

    #[test]
    fn selection_requires_all_flags_and_respects_type_mask() {
        let props = properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ]);
        assert_eq!(
            select_memory_type(
                &props,
                0b11,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::empty(),
            ),
            Some(1)
        );
        assert_eq!(
            select_memory_type(
                &props,
                0b01,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::empty(),
            ),
            Some(0)
        );
    }

    #[test]
    fn selection_prefers_the_type_with_more_preferred_flags() {
        let props = properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED,
        ]);
        assert_eq!(
            select_memory_type(
                &props,
                0b111,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
            ),
            Some(2)
        );
    }

    #[test]
    fn selection_uses_first_type_for_equal_preference_scores() {
        let props = properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
        ]);
        assert_eq!(
            select_memory_type(
                &props,
                0b11,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
            ),
            Some(0),
        );
    }

    #[test]
    fn selection_reports_no_compatible_type() {
        let props = properties(&[vk::MemoryPropertyFlags::DEVICE_LOCAL]);
        assert_eq!(
            select_memory_type(
                &props,
                0b1,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::empty(),
            ),
            None
        );
    }

    #[test]
    fn non_coherent_ranges_are_atom_aligned() {
        assert_eq!(mapped_memory_range(70, 20, 256, 64), Some((64, 64)));
        assert_eq!(mapped_memory_range(64, 64, 256, 64), Some((64, 64)));
        assert_eq!(
            mapped_memory_range(240, 16, 256, 64),
            Some((192, vk::WHOLE_SIZE))
        );
        assert_eq!(
            mapped_memory_range(192, 58, 250, 64),
            Some((192, vk::WHOLE_SIZE))
        );
        assert_eq!(
            mapped_memory_range(0, 250, 250, 64),
            Some((0, vk::WHOLE_SIZE))
        );
        assert_eq!(mapped_memory_range(u64::MAX, 2, 256, 64), None);
        assert_eq!(mapped_memory_range(250, 16, 256, 64), None);
        assert_eq!(mapped_memory_range(0, 1, 256, 0), None);
        assert_eq!(
            mapped_memory_range(u64::MAX - 15, 10, u64::MAX, 64),
            Some(((u64::MAX - 15) / 64 * 64, vk::WHOLE_SIZE))
        );
    }
}
