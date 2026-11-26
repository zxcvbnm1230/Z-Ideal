//! Deterministic application-level redraw, pause, and frame-pacing policy.
use std::time::{Duration, Instant};

use thiserror::Error;
use winit::event_loop::ControlFlow;

use crate::WindowEventOutcome;

/// Determines whether frames are produced continuously or only when requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameMode {
    /// Produce frames at the configured interval, or as quickly as possible when unthrottled.
    Continuous,
    /// Produce a frame only after a redraw or surface/focus change is observed.
    OnDemand,
}

/// Application-level frame scheduling configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePolicyConfig {
    /// Continuous rendering or event-driven rendering.
    pub mode: FrameMode,
    /// Minimum interval between frame starts; `None` disables pacing.
    pub target_interval: Option<Duration>,
    /// Suspend rendering while the window is unfocused.
    pub pause_when_unfocused: bool,
}

impl Default for FramePolicyConfig {
    fn default() -> Self {
        Self {
            mode: FrameMode::Continuous,
            target_interval: Some(Duration::from_nanos(16_666_667)),
            pause_when_unfocused: false,
        }
    }
}

/// Why the frame policy currently suppresses rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSuspensionReason {
    /// The application explicitly paused rendering.
    Paused,
    /// The native window has no drawable area.
    Minimized,
    /// The policy is configured to pause when the window is unfocused.
    Unfocused,
}

/// Decision for the application's next event-loop iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDecision {
    /// Request a redraw now.
    RenderNow,
    /// No redraw is currently required.
    Wait,
    /// A redraw is required, but frame pacing delays it until this instant.
    WaitUntil(Instant),
    /// Rendering is suspended until the relevant state changes.
    Suspended(FrameSuspensionReason),
}

impl FrameDecision {
    /// Whether the application should call `Window::request_redraw`.
    #[must_use]
    pub const fn should_request_redraw(self) -> bool {
        matches!(self, Self::RenderNow)
    }

    /// Maps the decision to winit's event-loop control flow.
    #[must_use]
    pub const fn control_flow(self) -> ControlFlow {
        match self {
            Self::WaitUntil(deadline) => ControlFlow::WaitUntil(deadline),
            Self::RenderNow | Self::Wait | Self::Suspended(_) => ControlFlow::Wait,
        }
    }
}

/// Errors returned by the application frame policy.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum FramePolicyError {
    /// A configured frame interval must be greater than zero.
    #[error("target frame interval must be greater than zero")]
    ZeroFrameInterval,
    /// The requested frame deadline exceeds the representable `Instant` range.
    #[error("target frame deadline exceeds the platform clock range")]
    DeadlineOverflow,
}

/// Pure per-window frame pacing state that applications can use from `about_to_wait`.
///
/// Create one policy per window. The policy does not own a native event loop or Vulkan scheduler.
/// Applications pass the
/// current drawable size/focus state, translate [`FrameDecision`] into a redraw request and
/// `ControlFlow`, and call [`Self::frame_started`] when handling `WindowEvent::RedrawRequested`.
/// A pending
/// redraw is preserved while minimized, paused, or unfocused so restoration does not lose it.
#[derive(Debug, Clone)]
pub struct FramePolicy {
    config: FramePolicyConfig,
    paused: bool,
    redraw_pending: bool,
    next_frame_deadline: Option<Instant>,
}

impl FramePolicy {
    /// Creates a frame policy after validating its pacing configuration.
    pub fn new(config: FramePolicyConfig) -> Result<Self, FramePolicyError> {
        if config.target_interval == Some(Duration::ZERO) {
            return Err(FramePolicyError::ZeroFrameInterval);
        }
        Ok(Self {
            config,
            paused: false,
            redraw_pending: config.mode == FrameMode::Continuous,
            next_frame_deadline: None,
        })
    }

    /// Returns the immutable policy configuration.
    #[must_use]
    pub const fn config(&self) -> FramePolicyConfig {
        self.config
    }

    /// Whether rendering is explicitly paused.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    /// Requests a redraw; repeated requests coalesce into one pending frame.
    pub fn request_redraw(&mut self) {
        self.redraw_pending = true;
    }

    /// Observes platform events that should schedule a redraw.
    ///
    /// Surface and focus changes schedule a frame. A `RedrawRequested` event means the
    /// platform has already delivered a redraw; handle it by starting the frame rather than
    /// queueing another request. A pending request remains while rendering is suspended.
    pub fn observe_window_event(&mut self, outcome: WindowEventOutcome) {
        if outcome.surface_changed || outcome.focus_changed {
            self.redraw_pending = true;
        }
    }

    /// Pauses or resumes rendering. Resuming schedules an initial redraw.
    pub fn set_paused(&mut self, paused: bool) {
        if self.paused && !paused {
            self.redraw_pending = true;
        }
        self.paused = paused;
    }

    /// Computes the next action from a monotonic timestamp and current window state.
    ///
    /// `physical_size` is the drawable extent in physical pixels. Zero in either dimension
    /// suspends rendering. `focused` is consulted only when `pause_when_unfocused` is enabled.
    #[must_use]
    pub fn decide(&self, now: Instant, physical_size: [u32; 2], focused: bool) -> FrameDecision {
        if self.paused {
            return FrameDecision::Suspended(FrameSuspensionReason::Paused);
        }
        if physical_size[0] == 0 || physical_size[1] == 0 {
            return FrameDecision::Suspended(FrameSuspensionReason::Minimized);
        }
        if self.config.pause_when_unfocused && !focused {
            return FrameDecision::Suspended(FrameSuspensionReason::Unfocused);
        }
        if self.config.mode == FrameMode::OnDemand && !self.redraw_pending {
            return FrameDecision::Wait;
        }
        if let Some(deadline) = self.next_frame_deadline
            && now < deadline
        {
            return FrameDecision::WaitUntil(deadline);
        }
        FrameDecision::RenderNow
    }

    /// Marks the start of a frame, consumes the pending redraw, and schedules the next deadline.
    ///
    /// Call this only after the application receives/starts a redraw. The deadline is based on
    /// frame-start time, so slow frames do not accumulate a backlog of missed ticks.
    pub fn frame_started(&mut self, now: Instant) -> Result<(), FramePolicyError> {
        let next_deadline = self
            .config
            .target_interval
            .map(|interval| {
                now.checked_add(interval)
                    .ok_or(FramePolicyError::DeadlineOverflow)
            })
            .transpose()?;
        self.redraw_pending = false;
        self.next_frame_deadline = next_deadline;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FrameDecision, FrameMode, FramePolicy, FramePolicyConfig, FramePolicyError,
        FrameSuspensionReason,
    };
    use crate::WindowEventOutcome;
    use std::time::{Duration, Instant};
    use winit::event_loop::ControlFlow;

    fn on_demand_config() -> FramePolicyConfig {
        FramePolicyConfig {
            mode: FrameMode::OnDemand,
            target_interval: None,
            pause_when_unfocused: false,
        }
    }

    #[test]
    fn continuous_default_starts_immediately_and_paces_subsequent_frames() {
        let mut policy = FramePolicy::new(FramePolicyConfig::default()).expect("valid policy");
        let start = Instant::now();
        assert_eq!(
            policy.decide(start, [1280, 720], true),
            FrameDecision::RenderNow
        );
        policy.frame_started(start).expect("start frame");
        let deadline = start + Duration::from_nanos(16_666_667);
        assert_eq!(
            policy.decide(start + Duration::from_millis(4), [1280, 720], true),
            FrameDecision::WaitUntil(deadline)
        );
        assert_eq!(
            policy.decide(deadline, [1280, 720], true),
            FrameDecision::RenderNow
        );
    }

    #[test]
    fn on_demand_mode_waits_until_a_redraw_is_requested() {
        let mut policy = FramePolicy::new(on_demand_config()).expect("valid policy");
        let now = Instant::now();
        assert_eq!(policy.decide(now, [800, 600], true), FrameDecision::Wait);
        policy.request_redraw();
        assert_eq!(
            policy.decide(now, [800, 600], true),
            FrameDecision::RenderNow
        );
        policy.frame_started(now).expect("start frame");
        assert_eq!(policy.decide(now, [800, 600], true), FrameDecision::Wait);
    }

    #[test]
    fn surface_and_focus_events_schedule_redraws_but_delivered_redraws_do_not_requeue() {
        let mut policy = FramePolicy::new(on_demand_config()).expect("valid policy");
        let now = Instant::now();
        policy.observe_window_event(WindowEventOutcome {
            redraw_requested: true,
            ..WindowEventOutcome::default()
        });
        assert_eq!(policy.decide(now, [800, 600], true), FrameDecision::Wait);
        policy.observe_window_event(WindowEventOutcome {
            surface_changed: true,
            ..WindowEventOutcome::default()
        });
        assert_eq!(
            policy.decide(now, [800, 600], true),
            FrameDecision::RenderNow
        );
    }

    #[test]
    fn minimized_paused_and_unfocused_states_suspend_without_losing_redraws() {
        let mut policy = FramePolicy::new(FramePolicyConfig {
            pause_when_unfocused: true,
            ..on_demand_config()
        })
        .expect("valid policy");
        policy.request_redraw();
        let now = Instant::now();
        assert_eq!(
            policy.decide(now, [0, 0], true),
            FrameDecision::Suspended(FrameSuspensionReason::Minimized)
        );
        assert_eq!(
            policy.decide(now, [800, 600], false),
            FrameDecision::Suspended(FrameSuspensionReason::Unfocused)
        );
        policy.set_paused(true);
        assert_eq!(
            policy.decide(now, [800, 600], true),
            FrameDecision::Suspended(FrameSuspensionReason::Paused)
        );
        policy.set_paused(false);
        assert_eq!(
            policy.decide(now, [800, 600], true),
            FrameDecision::RenderNow
        );
    }

    #[test]
    fn focus_and_surface_restoration_schedule_on_demand_redraws() {
        let mut policy = FramePolicy::new(FramePolicyConfig {
            pause_when_unfocused: true,
            ..on_demand_config()
        })
        .expect("valid policy");
        let now = Instant::now();
        assert_eq!(
            policy.decide(now, [0, 0], true),
            FrameDecision::Suspended(FrameSuspensionReason::Minimized)
        );
        policy.observe_window_event(WindowEventOutcome {
            surface_changed: true,
            focus_changed: true,
            ..WindowEventOutcome::default()
        });
        assert_eq!(
            policy.decide(now, [800, 600], true),
            FrameDecision::RenderNow
        );
    }

    #[test]
    fn zero_interval_is_rejected_and_event_loop_control_flow_is_explicit() {
        assert_eq!(
            FramePolicy::new(FramePolicyConfig {
                target_interval: Some(Duration::ZERO),
                ..FramePolicyConfig::default()
            })
            .err(),
            Some(FramePolicyError::ZeroFrameInterval)
        );
        let deadline = Instant::now() + Duration::from_millis(5);
        assert_eq!(
            FrameDecision::WaitUntil(deadline).control_flow(),
            ControlFlow::WaitUntil(deadline)
        );
        assert_eq!(FrameDecision::RenderNow.control_flow(), ControlFlow::Wait);
        assert!(FrameDecision::RenderNow.should_request_redraw());
        assert!(!FrameDecision::Wait.should_request_redraw());
    }
}
