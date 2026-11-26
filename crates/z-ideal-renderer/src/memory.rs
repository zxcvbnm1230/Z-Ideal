//! Vulkan memory suballocation shared by buffers and images.
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU32, Ordering},
};

use ash::vk;

use crate::VulkanError;

const DEFAULT_BLOCK_SIZE: vk::DeviceSize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRange {
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

/// Owns memory blocks for one Vulkan device and suballocates compatible ranges.
pub(super) struct MemoryAllocator {
    device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    block_size: vk::DeviceSize,
    max_allocation_count: u32,
    allocation_count: Arc<AtomicU32>,
    pools: Mutex<Vec<Arc<MemoryBlock>>>,
}

impl MemoryAllocator {
    pub(super) fn new(
        device: &ash::Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        max_allocation_count: u32,
    ) -> Self {
        Self {
            device: device.clone(),
            memory_properties,
            block_size: DEFAULT_BLOCK_SIZE,
            max_allocation_count,
            allocation_count: Arc::new(AtomicU32::new(0)),
            pools: Mutex::new(Vec::new()),
        }
    }

    pub(super) const fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    pub(super) fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        memory_type_index: u32,
    ) -> Result<MemoryAllocation, VulkanError> {
        if requirements.size == 0 || requirements.alignment == 0 {
            return Err(VulkanError::InvalidMemoryRequirements);
        }
        if memory_type_index >= self.memory_properties.memory_type_count {
            return Err(VulkanError::InvalidMemoryTypeIndex);
        }

        let use_standalone = requirements.size > self.block_size;
        let mut pools = self
            .pools
            .lock()
            .map_err(|_| VulkanError::MemoryPoolPoisoned)?;

        if !use_standalone {
            for block in pools
                .iter()
                .filter(|block| block.memory_type_index == memory_type_index && !block.standalone)
            {
                if let Some(offset) =
                    block.allocate_range(requirements.size, requirements.alignment)?
                {
                    return Ok(MemoryAllocation {
                        block: Arc::clone(block),
                        offset,
                        size: requirements.size,
                    });
                }
            }
        }

        let (block, pooled) = if use_standalone {
            (
                self.create_block(requirements.size, memory_type_index, true)?,
                false,
            )
        } else {
            match self.create_block(self.block_size, memory_type_index, false) {
                Ok(block) => (block, true),
                Err(error)
                    if matches!(
                        &error,
                        VulkanError::Api {
                            result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                                | vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                            ..
                        }
                    ) =>
                {
                    tracing::warn!(
                        requested_block_size = self.block_size,
                        requested_allocation_size = requirements.size,
                        "Vulkan memory pool block allocation failed; retrying with a standalone allocation"
                    );
                    (
                        self.create_block(requirements.size, memory_type_index, true)?,
                        false,
                    )
                }
                Err(error) => return Err(error),
            }
        };
        let offset = block
            .allocate_range(requirements.size, requirements.alignment)?
            .ok_or(VulkanError::MemorySuballocationFailed)?;
        if pooled {
            pools.push(Arc::clone(&block));
        }
        Ok(MemoryAllocation {
            block,
            offset,
            size: requirements.size,
        })
    }

    fn create_block(
        &self,
        size: vk::DeviceSize,
        memory_type_index: u32,
        standalone: bool,
    ) -> Result<Arc<MemoryBlock>, VulkanError> {
        let current = self.allocation_count.fetch_add(1, Ordering::AcqRel);
        if current >= self.max_allocation_count {
            self.allocation_count.fetch_sub(1, Ordering::AcqRel);
            return Err(VulkanError::MemoryAllocationLimitReached {
                limit: self.max_allocation_count,
            });
        }

        // Vulkan 1.0-compatible allocation path. Large resources receive their own
        // VkDeviceMemory block; smaller resources suballocate from pooled blocks.
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index);
        // SAFETY: The allocation size and memory type were validated against the
        // resource's Vulkan memory requirements and the selected physical device.
        let memory = match unsafe { self.device.allocate_memory(&info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                self.allocation_count.fetch_sub(1, Ordering::AcqRel);
                return Err(crate::api_error("allocate Vulkan device memory", error));
            }
        };
        let property_flags =
            self.memory_properties.memory_types[memory_type_index as usize].property_flags;
        Ok(Arc::new(MemoryBlock {
            device: self.device.clone(),
            memory,
            size,
            memory_type_index,
            property_flags,
            standalone,
            free_ranges: Mutex::new(vec![FreeRange { offset: 0, size }]),
            mapping_lock: Mutex::new(()),
            allocation_count: Arc::clone(&self.allocation_count),
        }))
    }

    /// Releases pooled Vulkan memory before the logical device is destroyed.
    pub(super) fn clear(&mut self) {
        let pools = self
            .pools
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for block in pools.iter() {
            if Arc::strong_count(block) != 1 {
                tracing::error!(
                    strong_references = Arc::strong_count(block),
                    "Vulkan memory block still has live suballocations during device teardown"
                );
            }
        }
        pools.clear();
    }
}

struct MemoryBlock {
    device: ash::Device,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    memory_type_index: u32,
    property_flags: vk::MemoryPropertyFlags,
    standalone: bool,
    free_ranges: Mutex<Vec<FreeRange>>,
    mapping_lock: Mutex<()>,
    allocation_count: Arc<AtomicU32>,
}

impl MemoryBlock {
    fn allocate_range(
        &self,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Result<Option<vk::DeviceSize>, VulkanError> {
        let mut ranges = self
            .free_ranges
            .lock()
            .map_err(|_| VulkanError::MemoryPoolPoisoned)?;
        Ok(allocate_from_ranges(&mut ranges, size, alignment))
    }

    fn free_range(&self, offset: vk::DeviceSize, size: vk::DeviceSize) {
        if self.standalone {
            return;
        }
        match self.free_ranges.lock() {
            Ok(mut ranges) => insert_and_coalesce(&mut ranges, FreeRange { offset, size }),
            Err(_) => tracing::error!(
                "Vulkan memory free-range mutex was poisoned; range retained until device teardown"
            ),
        }
    }

    fn lock_mapping(&self) -> Result<MutexGuard<'_, ()>, VulkanError> {
        self.mapping_lock
            .lock()
            .map_err(|_| VulkanError::MemoryMapLockPoisoned)
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        // SAFETY: The block uniquely owns this allocation. The allocator is cleared
        // before device destruction, and resource lifetimes prevent live allocations
        // from outliving their VulkanDevice.
        unsafe { self.device.free_memory(self.memory, None) };
        self.allocation_count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One aligned suballocation within a shared Vulkan memory block.
pub(super) struct MemoryAllocation {
    block: Arc<MemoryBlock>,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

impl MemoryAllocation {
    pub(super) fn memory(&self) -> vk::DeviceMemory {
        self.block.memory
    }

    pub(super) const fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    pub(super) fn block_size(&self) -> vk::DeviceSize {
        self.block.size
    }

    pub(super) fn property_flags(&self) -> vk::MemoryPropertyFlags {
        self.block.property_flags
    }

    pub(super) fn lock_mapping(&self) -> Result<MutexGuard<'_, ()>, VulkanError> {
        self.block.lock_mapping()
    }
}

impl Drop for MemoryAllocation {
    fn drop(&mut self) {
        self.block.free_range(self.offset, self.size);
    }
}

fn greatest_common_divisor(mut left: vk::DeviceSize, mut right: vk::DeviceSize) -> vk::DeviceSize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

pub(super) fn checked_least_common_multiple(
    left: vk::DeviceSize,
    right: vk::DeviceSize,
) -> Option<vk::DeviceSize> {
    if left == 0 || right == 0 {
        return None;
    }
    (left / greatest_common_divisor(left, right)).checked_mul(right)
}

fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> Option<vk::DeviceSize> {
    if alignment == 0 {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

fn allocate_from_ranges(
    ranges: &mut Vec<FreeRange>,
    size: vk::DeviceSize,
    alignment: vk::DeviceSize,
) -> Option<vk::DeviceSize> {
    let index = ranges
        .iter()
        .position(|range| aligned_range(range.offset, range.size, size, alignment).is_some())?;
    let range = ranges.remove(index);
    let (offset, end) = aligned_range(range.offset, range.size, size, alignment)?;
    let range_end = range.offset.checked_add(range.size)?;
    if offset > range.offset {
        ranges.push(FreeRange {
            offset: range.offset,
            size: offset - range.offset,
        });
    }
    if end < range_end {
        ranges.push(FreeRange {
            offset: end,
            size: range_end - end,
        });
    }
    ranges.sort_unstable_by_key(|entry| entry.offset);
    Some(offset)
}

fn aligned_range(
    range_offset: vk::DeviceSize,
    range_size: vk::DeviceSize,
    size: vk::DeviceSize,
    alignment: vk::DeviceSize,
) -> Option<(vk::DeviceSize, vk::DeviceSize)> {
    let offset = align_up(range_offset, alignment)?;
    let end = offset.checked_add(size)?;
    let range_end = range_offset.checked_add(range_size)?;
    (end <= range_end).then_some((offset, end))
}

fn insert_and_coalesce(ranges: &mut Vec<FreeRange>, freed: FreeRange) {
    ranges.push(freed);
    ranges.sort_unstable_by_key(|range| range.offset);
    let mut merged: Vec<FreeRange> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(previous) = merged.last_mut() {
            let previous_end = previous.offset.saturating_add(previous.size);
            if range.offset <= previous_end {
                let range_end = range.offset.saturating_add(range.size);
                previous.size = previous_end.max(range_end).saturating_sub(previous.offset);
                continue;
            }
        }
        merged.push(range);
    }
    *ranges = merged;
}

#[cfg(test)]
mod tests {
    use super::{
        FreeRange, align_up, aligned_range, allocate_from_ranges, checked_least_common_multiple,
        insert_and_coalesce,
    };

    #[test]
    fn aligns_offsets_for_non_power_of_two_requirements() {
        assert_eq!(align_up(17, 6), Some(18));
        assert_eq!(align_up(18, 6), Some(18));
        assert_eq!(align_up(u64::MAX, 2), None);
        assert_eq!(align_up(1, 0), None);
    }

    #[test]
    fn combines_resource_and_non_coherent_atom_alignment_without_overflow() {
        assert_eq!(checked_least_common_multiple(24, 64), Some(192));
        assert_eq!(checked_least_common_multiple(256, 64), Some(256));
        assert_eq!(checked_least_common_multiple(u64::MAX, 2), None);
        assert_eq!(checked_least_common_multiple(0, 64), None);
    }

    #[test]
    fn checks_suballocation_fit_without_overflow() {
        assert_eq!(aligned_range(3, 20, 8, 8), Some((8, 16)));
        assert_eq!(aligned_range(3, 12, 8, 8), None);
        assert_eq!(aligned_range(u64::MAX - 4, 8, 8, 1), None);
    }

    #[test]
    fn allocation_splits_prefix_and_suffix_and_reuses_freed_ranges() {
        let mut ranges = vec![FreeRange {
            offset: 0,
            size: 512,
        }];
        assert_eq!(allocate_from_ranges(&mut ranges, 100, 64), Some(0));
        assert_eq!(
            ranges,
            vec![FreeRange {
                offset: 100,
                size: 412
            }]
        );
        assert_eq!(allocate_from_ranges(&mut ranges, 80, 64), Some(128));
        assert_eq!(
            ranges,
            vec![
                FreeRange {
                    offset: 100,
                    size: 28
                },
                FreeRange {
                    offset: 208,
                    size: 304
                },
            ]
        );
        insert_and_coalesce(
            &mut ranges,
            FreeRange {
                offset: 0,
                size: 100,
            },
        );
        assert_eq!(
            ranges[0],
            FreeRange {
                offset: 0,
                size: 128
            }
        );
        assert_eq!(allocate_from_ranges(&mut ranges, 120, 64), Some(0));
    }

    #[test]
    fn freed_ranges_coalesce_in_both_directions() {
        let mut ranges = vec![
            FreeRange {
                offset: 16,
                size: 16,
            },
            FreeRange {
                offset: 48,
                size: 16,
            },
        ];
        insert_and_coalesce(
            &mut ranges,
            FreeRange {
                offset: 32,
                size: 16,
            },
        );
        assert_eq!(
            ranges,
            vec![FreeRange {
                offset: 16,
                size: 48
            }]
        );
        insert_and_coalesce(
            &mut ranges,
            FreeRange {
                offset: 0,
                size: 16,
            },
        );
        assert_eq!(
            ranges,
            vec![FreeRange {
                offset: 0,
                size: 64
            }]
        );
    }

    #[test]
    fn adjacent_ranges_coalesce_but_gaps_remain_separate() {
        let mut ranges = vec![
            FreeRange {
                offset: 0,
                size: 16,
            },
            FreeRange {
                offset: 32,
                size: 16,
            },
        ];
        insert_and_coalesce(
            &mut ranges,
            FreeRange {
                offset: 16,
                size: 16,
            },
        );
        assert_eq!(
            ranges,
            vec![FreeRange {
                offset: 0,
                size: 48
            }]
        );
        insert_and_coalesce(
            &mut ranges,
            FreeRange {
                offset: 64,
                size: 16,
            },
        );
        assert_eq!(ranges.len(), 2);
    }
}
