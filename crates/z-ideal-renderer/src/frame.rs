//! Frames-in-flight scheduling and swapchain-image synchronization.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use ash::vk;

use crate::{
    AcquireResult, ExecutableCommandBuffer, FenceConfig, FenceWaitResult, PresentResult,
    VulkanDevice, VulkanError, VulkanFence, VulkanQueue, VulkanSemaphore, VulkanSubmission,
    VulkanSwapchain,
};

#[derive(Debug, Default)]
struct SwapchainImageTracker {
    owners: Vec<Option<usize>>,
}

impl SwapchainImageTracker {
    fn new(image_count: usize) -> Self {
        Self {
            owners: vec![None; image_count],
        }
    }

    fn reset(&mut self, image_count: usize) {
        self.owners = vec![None; image_count];
    }

    fn owner(&self, image_index: usize) -> Option<usize> {
        self.owners.get(image_index).copied().flatten()
    }

    fn mark_submitted(&mut self, image_index: usize, frame_index: usize) -> bool {
        if let Some(owner) = self.owners.get_mut(image_index) {
            *owner = Some(frame_index);
            true
        } else {
            false
        }
    }
}

fn next_frame_index(current: usize, frame_count: usize) -> usize {
    (current + 1) % frame_count
}

fn same_swapchain_identity(left: &(Arc<()>, u64), right: &(Arc<()>, u64)) -> bool {
    Arc::ptr_eq(&left.0, &right.0) && left.1 == right.1
}

struct FrameSlot<'device> {
    fence: Arc<VulkanFence<'device>>,
    image_available: VulkanSemaphore<'device>,
    submission_live: Arc<AtomicBool>,
}

/// Coordinates a bounded number of frames in flight and tracks which frame last used each swapchain image.
///
/// A submission guard must be waited or dropped before its frame slot can be recycled. This
/// requirement prevents a fence from being reset while an older guard still refers to it.
pub struct VulkanFrameScheduler<'device> {
    device: &'device VulkanDevice,
    slots: Vec<FrameSlot<'device>>,
    current_frame: usize,
    image_tracker: SwapchainImageTracker,
    // Presentation wait semaphores are per swapchain image. A graphics fence does not prove
    // that the presentation engine has consumed a semaphore passed to queue_present.
    render_finished: Vec<VulkanSemaphore<'device>>,
    swapchain_identity: Option<(Arc<()>, u64)>,
    pending_acquire: Option<(u32, bool)>,
    pending_swapchain: Option<(Arc<()>, u64)>,
    poisoned: bool,
}

/// Outcome of beginning a frame.
pub enum FrameStart<'scheduler, 'swapchain, 'device> {
    /// A swapchain image is acquired and ready for command recording/submission.
    Ready(FrameTicket<'scheduler, 'swapchain, 'device>),
    /// The acquire operation reported an out-of-date swapchain; recreate it and retry.
    OutOfDate,
    /// The current frame fence did not signal before the supplied timeout.
    Timeout,
    /// An earlier submission guard still owns this frame slot or image; release it and retry.
    Pending,
}

/// Exclusive ticket for one acquired swapchain image.
///
/// Keep the ticket until `submit_and_present` succeeds. Dropping it without submission preserves
/// the pending acquisition so the same image-available semaphore is not signaled a second time.
#[must_use = "an acquired frame ticket must be submitted and presented before acquiring another frame"]
pub struct FrameTicket<'scheduler, 'swapchain, 'device> {
    scheduler: &'scheduler mut VulkanFrameScheduler<'device>,
    swapchain: &'swapchain VulkanSwapchain<'device>,
    image_index: u32,
    suboptimal: bool,
}

impl<'device> VulkanFrameScheduler<'device> {
    /// Creates a scheduler with reusable synchronization objects for each frame slot.
    pub fn new(
        device: &'device VulkanDevice,
        frames_in_flight: usize,
    ) -> Result<Self, VulkanError> {
        if frames_in_flight == 0 {
            return Err(VulkanError::InvalidFrameSchedulerConfig);
        }
        let mut slots = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            slots.push(FrameSlot {
                fence: Arc::new(VulkanFence::new(device, FenceConfig { signaled: true })?),
                image_available: VulkanSemaphore::new(device)?,
                submission_live: Arc::new(AtomicBool::new(false)),
            });
        }
        Ok(Self {
            device,
            slots,
            current_frame: 0,
            image_tracker: SwapchainImageTracker::new(0),
            render_finished: Vec::new(),
            swapchain_identity: None,
            pending_acquire: None,
            pending_swapchain: None,
            poisoned: false,
        })
    }

    /// Returns the number of reusable frame slots.
    #[must_use]
    pub fn frames_in_flight(&self) -> usize {
        self.slots.len()
    }

    /// Returns the index of the frame slot that will be used next.
    #[must_use]
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Waits for the current slot, acquires an image, and waits for any earlier frame using it.
    ///
    /// `timeout_nanoseconds` applies to the current slot and image acquisition. Once an image has
    /// been acquired, waiting for its prior owner is unbounded: returning early would strand a
    /// signaled acquire semaphore and an acquired image without a cancellation operation.
    pub fn begin_frame<'scheduler, 'swapchain>(
        &'scheduler mut self,
        swapchain: &'swapchain VulkanSwapchain<'device>,
        timeout_nanoseconds: u64,
    ) -> Result<FrameStart<'scheduler, 'swapchain, 'device>, VulkanError> {
        if self.poisoned {
            return Err(VulkanError::FrameSchedulerPoisoned);
        }
        if !swapchain.belongs_to(self.device) {
            return Err(VulkanError::FrameSwapchainDeviceMismatch);
        }
        let swapchain_identity = swapchain.identity()?;
        if let Some(pending_identity) = &self.pending_swapchain {
            if !same_swapchain_identity(pending_identity, &swapchain_identity) {
                return Err(VulkanError::PendingFrameSwapchainChanged);
            }
        } else if self
            .swapchain_identity
            .as_ref()
            .is_none_or(|identity| !same_swapchain_identity(identity, &swapchain_identity))
        {
            if self
                .slots
                .iter()
                .any(|frame| frame.submission_live.load(Ordering::Acquire))
            {
                return Ok(FrameStart::Pending);
            }
            self.device.wait_idle()?;
            let mut render_finished = Vec::with_capacity(swapchain.image_count());
            for _ in 0..swapchain.image_count() {
                render_finished.push(VulkanSemaphore::new(self.device)?);
            }
            self.render_finished = render_finished;
            self.image_tracker.reset(swapchain.image_count());
            self.swapchain_identity = Some(swapchain_identity.clone());
        }

        let slot_index = self.current_frame;
        let slot = &self.slots[slot_index];
        if slot.fence.wait(timeout_nanoseconds)? == FenceWaitResult::Timeout {
            return Ok(FrameStart::Timeout);
        }
        if slot.submission_live.load(Ordering::Acquire) {
            return Ok(FrameStart::Pending);
        }

        if self.pending_acquire.is_none() {
            match swapchain.acquire_next_image(
                timeout_nanoseconds,
                slot.image_available.handle(),
                vk::Fence::null(),
            )? {
                AcquireResult::OutOfDate => return Ok(FrameStart::OutOfDate),
                AcquireResult::Timeout => return Ok(FrameStart::Timeout),
                AcquireResult::Acquired {
                    image_index,
                    suboptimal,
                } => {
                    self.pending_acquire = Some((image_index, suboptimal));
                    self.pending_swapchain = Some(swapchain_identity.clone());
                }
            }
        }

        let (image_index, suboptimal) =
            self.pending_acquire.ok_or(VulkanError::MissingSwapchain)?;
        let image_count = swapchain.image_count();
        if image_index as usize >= image_count {
            return Err(VulkanError::InvalidSwapchainImageIndex {
                index: image_index,
                image_count,
            });
        }
        if self.image_tracker.owners.len() != image_count
            || self.render_finished.len() != image_count
        {
            return Err(VulkanError::FrameSchedulerPoisoned);
        }
        if let Some(previous_frame) = self.image_tracker.owner(image_index as usize)
            && previous_frame != slot_index
        {
            let previous = &self.slots[previous_frame];
            if previous.fence.wait(u64::MAX)? == FenceWaitResult::Timeout {
                return Err(VulkanError::Api {
                    operation: "wait for prior swapchain image owner",
                    result: vk::Result::TIMEOUT,
                });
            }
            if previous.submission_live.load(Ordering::Acquire) {
                return Ok(FrameStart::Pending);
            }
        }

        Ok(FrameStart::Ready(FrameTicket {
            scheduler: self,
            swapchain,
            image_index,
            suboptimal,
        }))
    }
}

impl<'scheduler, 'swapchain, 'device> FrameTicket<'scheduler, 'swapchain, 'device> {
    /// Returns the acquired swapchain image index.
    #[must_use]
    pub const fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Whether acquisition reported a suboptimal swapchain.
    #[must_use]
    pub const fn is_suboptimal(&self) -> bool {
        self.suboptimal
    }

    /// Submits the executable command buffer, signals presentation, and advances the frame ring.
    ///
    /// The returned submission guard retains all command resources. It must be waited or dropped
    /// before this frame slot can be reused. A successful queue submission advances the ring even
    /// when presentation reports out-of-date, because the frame's fence is now associated with work.
    pub fn submit_and_present<'recording, 'pool>(
        self,
        queue: &VulkanQueue<'device>,
        executable: ExecutableCommandBuffer<'recording, 'pool, 'device>,
    ) -> Result<(PresentResult, VulkanSubmission<'recording, 'pool, 'device>), VulkanError> {
        if !queue.belongs_to(self.scheduler.device) {
            return Err(VulkanError::FrameSwapchainDeviceMismatch);
        }
        let current_swapchain_identity = self.swapchain.identity()?;
        if self.scheduler.pending_acquire != Some((self.image_index, self.suboptimal))
            || !self
                .scheduler
                .pending_swapchain
                .as_ref()
                .is_some_and(|identity| {
                    same_swapchain_identity(identity, &current_swapchain_identity)
                })
        {
            return Err(VulkanError::InvalidQueueSubmission);
        }
        if !executable.targets_swapchain(&current_swapchain_identity, self.image_index) {
            return Err(VulkanError::FrameCommandBufferTargetMismatch);
        }
        let slot_index = self.scheduler.current_frame;
        let frame_count = self.scheduler.slots.len();
        let render_finished = &self.scheduler.render_finished[self.image_index as usize];
        let render_finished_handle = render_finished.handle();
        let submission = {
            let slot = &mut self.scheduler.slots[slot_index];
            if let Err(error) = slot.fence.reset() {
                self.scheduler.poisoned = true;
                return Err(error);
            }
            let wait_semaphores = [&slot.image_available];
            let signal_semaphores = [render_finished];
            let stage_masks = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            match queue.submit_with_fence(
                executable,
                &wait_semaphores,
                &stage_masks,
                &signal_semaphores,
                Arc::clone(&slot.fence),
                Some(Arc::clone(&slot.submission_live)),
            ) {
                Ok(submission) => submission,
                Err(error) => {
                    match VulkanFence::new(self.scheduler.device, FenceConfig { signaled: true }) {
                        Ok(replacement) => slot.fence = Arc::new(replacement),
                        Err(_) => self.scheduler.poisoned = true,
                    }
                    return Err(error);
                }
            }
        };
        let tracked = self
            .scheduler
            .image_tracker
            .mark_submitted(self.image_index as usize, slot_index);
        if !tracked {
            self.scheduler.poisoned = true;
            return Err(VulkanError::FrameSchedulerPoisoned);
        }
        self.scheduler.pending_acquire = None;
        self.scheduler.pending_swapchain = None;
        self.scheduler.current_frame = next_frame_index(slot_index, frame_count);
        let present_result = self
            .swapchain
            .present(self.image_index, &[render_finished_handle])?;
        Ok((present_result, submission))
    }
}

#[cfg(test)]
mod tests {
    use super::{SwapchainImageTracker, next_frame_index, same_swapchain_identity};
    use std::sync::Arc;

    #[test]
    fn frame_ring_advances_and_wraps_for_multiple_slot_counts() {
        assert_eq!(next_frame_index(0, 1), 0);
        assert_eq!(next_frame_index(0, 2), 1);
        assert_eq!(next_frame_index(1, 2), 0);
        assert_eq!(next_frame_index(2, 3), 0);
    }

    #[test]
    fn image_tracker_keeps_independent_frame_owners() {
        let mut tracker = SwapchainImageTracker::new(3);
        assert_eq!(tracker.owner(0), None);
        assert!(tracker.mark_submitted(0, 1));
        assert!(tracker.mark_submitted(2, 0));
        assert_eq!(tracker.owner(0), Some(1));
        assert_eq!(tracker.owner(1), None);
        assert_eq!(tracker.owner(2), Some(0));
    }

    #[test]
    fn swapchain_identity_detects_replacement_and_recreation_even_if_handles_are_reused() {
        let token = Arc::new(());
        let original = (Arc::clone(&token), 0);
        let recreated = (Arc::clone(&token), 1);
        let replacement = (Arc::new(()), 0);
        assert!(same_swapchain_identity(&original, &original));
        assert!(!same_swapchain_identity(&original, &recreated));
        assert!(!same_swapchain_identity(&original, &replacement));
    }

    #[test]
    fn image_tracker_reassigns_reused_images_and_resets_on_swapchain_replacement() {
        let mut tracker = SwapchainImageTracker::new(2);
        assert!(tracker.mark_submitted(1, 0));
        assert!(tracker.mark_submitted(1, 2));
        assert_eq!(tracker.owner(1), Some(2));
        tracker.reset(3);
        assert_eq!(tracker.owners, vec![None, None, None]);
        assert!(!tracker.mark_submitted(3, 0));
    }
}
