//! Explicitly owned Vulkan synchronization primitives.
use ash::vk;

use crate::{VulkanDevice, VulkanError, api_error};

/// Fence creation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FenceConfig {
    /// Create the fence in the signaled state.
    pub signaled: bool,
}

/// Result of waiting for a fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceWaitResult {
    /// The fence became signaled before the timeout.
    Signaled,
    /// The timeout elapsed while the fence remained unsignaled.
    Timeout,
}

/// Owns a Vulkan fence and borrows the logical device that created it.
///
/// Dropping the owner waits for device idle before destroying the fence. This is
/// intentionally conservative; higher-level frame scheduling should retain and
/// recycle fences rather than create and destroy them in the frame-critical path.
pub struct VulkanFence<'device> {
    device: &'device VulkanDevice,
    handle: vk::Fence,
}

impl<'device> VulkanFence<'device> {
    /// Creates a fence owned by the supplied logical device.
    pub fn new(device: &'device VulkanDevice, config: FenceConfig) -> Result<Self, VulkanError> {
        let flags = if config.signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let info = vk::FenceCreateInfo::default().flags(flags);
        // SAFETY: The device is live and the create info contains no borrowed pointers.
        let handle = unsafe {
            device
                .handle()
                .create_fence(&info, None)
                .map_err(|error| api_error("create Vulkan fence", error))?
        };
        Ok(Self { device, handle })
    }

    /// Returns the raw fence handle for queue submission and presentation operations.
    #[must_use]
    pub const fn handle(&self) -> vk::Fence {
        self.handle
    }

    /// Waits until this fence is signaled or the timeout expires.
    pub fn wait(&self, timeout_nanoseconds: u64) -> Result<FenceWaitResult, VulkanError> {
        // SAFETY: The fence is live and belongs to the retained logical device.
        match unsafe {
            self.device
                .handle()
                .wait_for_fences(&[self.handle], true, timeout_nanoseconds)
        } {
            Ok(()) => Ok(FenceWaitResult::Signaled),
            Err(vk::Result::TIMEOUT) => Ok(FenceWaitResult::Timeout),
            Err(error) => Err(api_error("wait for Vulkan fence", error)),
        }
    }

    /// Resets this fence to the unsignaled state.
    ///
    /// The caller must ensure the fence is not associated with pending work.
    pub fn reset(&self) -> Result<(), VulkanError> {
        // SAFETY: The fence is live and belongs to the retained logical device.
        unsafe {
            self.device
                .handle()
                .reset_fences(&[self.handle])
                .map_err(|error| api_error("reset Vulkan fence", error))
        }
    }

    /// Returns whether this fence is currently signaled.
    pub fn is_signaled(&self) -> Result<bool, VulkanError> {
        // SAFETY: The fence is live and belongs to the retained logical device.
        match unsafe { self.device.handle().get_fence_status(self.handle) } {
            Ok(signaled) => Ok(signaled),
            Err(vk::Result::NOT_READY) => Ok(false),
            Err(error) => Err(api_error("query Vulkan fence status", error)),
        }
    }
}

impl Drop for VulkanFence<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before fence destruction"
            );
        }
        // SAFETY: This owner uniquely owns the fence and retains its parent device.
        unsafe { self.device.handle().destroy_fence(self.handle, None) };
    }
}

/// Owns a Vulkan binary semaphore and borrows its creating logical device.
///
/// Dropping the owner waits for device idle before destroying the semaphore.
/// Retain and recycle semaphores in frame scheduling rather than creating them
/// repeatedly during rendering.
pub struct VulkanSemaphore<'device> {
    device: &'device VulkanDevice,
    handle: vk::Semaphore,
}

impl<'device> VulkanSemaphore<'device> {
    /// Creates a binary semaphore owned by the supplied logical device.
    pub fn new(device: &'device VulkanDevice) -> Result<Self, VulkanError> {
        let info = vk::SemaphoreCreateInfo::default();
        // SAFETY: The device is live and the create info contains no borrowed pointers.
        let handle = unsafe {
            device
                .handle()
                .create_semaphore(&info, None)
                .map_err(|error| api_error("create Vulkan semaphore", error))?
        };
        Ok(Self { device, handle })
    }

    /// Returns the raw semaphore handle for queue submission and presentation operations.
    #[must_use]
    pub const fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for VulkanSemaphore<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::warn!(
                ?error,
                "Vulkan device idle wait failed before semaphore destruction"
            );
        }
        // SAFETY: This owner uniquely owns the semaphore and retains its parent device.
        unsafe { self.device.handle().destroy_semaphore(self.handle, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::{FenceConfig, FenceWaitResult};

    #[test]
    fn fence_defaults_to_unsignaled() {
        assert!(!FenceConfig::default().signaled);
    }

    #[test]
    fn fence_wait_result_distinguishes_timeout() {
        assert_ne!(FenceWaitResult::Signaled, FenceWaitResult::Timeout);
    }
}
