//! Explicitly owned Vulkan pipeline layouts.
use ash::vk;

use crate::{VulkanDescriptorSetLayout, VulkanDevice, VulkanError, api_error};

/// Owns a Vulkan pipeline layout and borrows all descriptor-set layouts it uses.
///
/// The Rust lifetime prevents a referenced descriptor-set layout from being
/// destroyed before this pipeline layout. Destruction waits for device idle so
/// submitted work cannot continue using the Vulkan pipeline-layout handle.
pub struct VulkanPipelineLayout<'layouts, 'device> {
    device: &'device VulkanDevice,
    handle: vk::PipelineLayout,
    set_layouts: Vec<&'layouts VulkanDescriptorSetLayout<'device>>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl<'layouts, 'device> VulkanPipelineLayout<'layouts, 'device> {
    /// Creates a pipeline layout from descriptor-set layouts and push-constant ranges.
    pub fn new(
        device: &'device VulkanDevice,
        set_layouts: &[&'layouts VulkanDescriptorSetLayout<'device>],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self, VulkanError> {
        if set_layouts
            .iter()
            .any(|layout| !std::ptr::eq(device, layout.device))
        {
            return Err(VulkanError::PipelineLayoutDeviceMismatch);
        }
        validate_push_constant_ranges(push_constant_ranges, device.max_push_constants_size)?;

        let raw_layouts: Vec<_> = set_layouts.iter().map(|layout| layout.handle()).collect();
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&raw_layouts)
            .push_constant_ranges(push_constant_ranges);

        // SAFETY: The device and every descriptor-set layout remain live for the
        // returned wrapper's lifetime; both input arrays live through this call.
        let handle = unsafe {
            device
                .handle
                .create_pipeline_layout(&info, None)
                .map_err(|error| api_error("create Vulkan pipeline layout", error))?
        };

        Ok(Self {
            device,
            handle,
            set_layouts: set_layouts.to_vec(),
            push_constant_ranges: push_constant_ranges.to_vec(),
        })
    }

    /// Returns the raw handle for graphics or compute pipeline creation.
    #[must_use]
    pub const fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }

    /// Returns the descriptor-set layouts retained by this pipeline layout.
    #[must_use]
    pub fn set_layouts(&self) -> &[&'layouts VulkanDescriptorSetLayout<'device>] {
        &self.set_layouts
    }

    /// Returns the validated push-constant ranges.
    #[must_use]
    pub fn push_constant_ranges(&self) -> &[vk::PushConstantRange] {
        &self.push_constant_ranges
    }
}

impl Drop for VulkanPipelineLayout<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before pipeline-layout destruction"
            );
        }
        // SAFETY: This wrapper uniquely owns the pipeline layout and retains its device.
        unsafe {
            self.device
                .handle
                .destroy_pipeline_layout(self.handle, None);
        }
    }
}

fn validate_push_constant_ranges(
    ranges: &[vk::PushConstantRange],
    max_size: u32,
) -> Result<(), VulkanError> {
    for (index, range) in ranges.iter().enumerate() {
        let Some(end) = range.offset.checked_add(range.size) else {
            return Err(VulkanError::InvalidPushConstantRange);
        };
        if range.stage_flags.is_empty()
            || range.size == 0
            || range.offset % 4 != 0
            || range.size % 4 != 0
            || end > max_size
        {
            return Err(VulkanError::InvalidPushConstantRange);
        }
        for previous in &ranges[..index] {
            let previous_end = previous
                .offset
                .checked_add(previous.size)
                .ok_or(VulkanError::InvalidPushConstantRange)?;
            let overlaps = range.offset < previous_end && previous.offset < end;
            if overlaps && range.stage_flags.intersects(previous.stage_flags) {
                return Err(VulkanError::OverlappingPushConstantRanges);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_push_constant_ranges;
    use crate::VulkanError;
    use ash::vk;

    fn range(offset: u32, size: u32, stages: vk::ShaderStageFlags) -> vk::PushConstantRange {
        vk::PushConstantRange::default()
            .offset(offset)
            .size(size)
            .stage_flags(stages)
    }

    #[test]
    fn accepts_empty_and_valid_push_constant_ranges() {
        assert!(validate_push_constant_ranges(&[], 128).is_ok());
        assert!(
            validate_push_constant_ranges(
                &[
                    range(0, 16, vk::ShaderStageFlags::VERTEX),
                    range(16, 16, vk::ShaderStageFlags::FRAGMENT)
                ],
                128,
            )
            .is_ok()
        );
    }

    #[test]
    fn accepts_overlapping_ranges_for_disjoint_shader_stages() {
        assert!(
            validate_push_constant_ranges(
                &[
                    range(0, 16, vk::ShaderStageFlags::VERTEX),
                    range(8, 16, vk::ShaderStageFlags::FRAGMENT)
                ],
                128,
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_empty_stage_masks_zero_sizes_and_unaligned_ranges() {
        for invalid in [
            range(0, 4, vk::ShaderStageFlags::empty()),
            range(0, 0, vk::ShaderStageFlags::VERTEX),
            range(2, 4, vk::ShaderStageFlags::VERTEX),
            range(0, 6, vk::ShaderStageFlags::VERTEX),
        ] {
            assert!(matches!(
                validate_push_constant_ranges(&[invalid], 128),
                Err(VulkanError::InvalidPushConstantRange)
            ));
        }
    }

    #[test]
    fn rejects_ranges_exceeding_device_limit_or_overflowing() {
        for invalid in [
            range(124, 8, vk::ShaderStageFlags::VERTEX),
            range(u32::MAX - 3, 8, vk::ShaderStageFlags::VERTEX),
        ] {
            assert!(matches!(
                validate_push_constant_ranges(&[invalid], 128),
                Err(VulkanError::InvalidPushConstantRange)
            ));
        }
    }

    #[test]
    fn rejects_overlapping_ranges_with_shared_shader_stages() {
        assert!(matches!(
            validate_push_constant_ranges(
                &[
                    range(0, 16, vk::ShaderStageFlags::VERTEX),
                    range(
                        8,
                        16,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
                    )
                ],
                128,
            ),
            Err(VulkanError::OverlappingPushConstantRanges)
        ));
    }
}
