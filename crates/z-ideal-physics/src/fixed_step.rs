//! Deterministic fixed-step time accumulation for physics simulation.

use std::time::Duration;

/// Configuration errors for [`FixedStepConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FixedStepConfigError {
    /// A simulation step must have a positive duration.
    #[error("fixed simulation step duration must be greater than zero")]
    ZeroStepDuration,
    /// At least one simulation step must be permitted per frame.
    #[error("maximum catch-up steps must be greater than zero")]
    ZeroCatchUpSteps,
}

/// Fixed-step simulation policy.
///
/// The default is a 60 Hz simulation with at most eight steps per rendered
/// frame. The default step is the nearest nanosecond (16,666,667 ns), not an
/// exact rational 1/60 second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedStepConfig {
    step_duration: Duration,
    max_catch_up_steps: u32,
}

impl Default for FixedStepConfig {
    fn default() -> Self {
        Self {
            step_duration: Duration::from_nanos(16_666_667),
            max_catch_up_steps: 8,
        }
    }
}

impl FixedStepConfig {
    /// Creates a validated fixed-step policy.
    pub fn new(
        step_duration: Duration,
        max_catch_up_steps: u32,
    ) -> Result<Self, FixedStepConfigError> {
        if step_duration.is_zero() {
            return Err(FixedStepConfigError::ZeroStepDuration);
        }
        if max_catch_up_steps == 0 {
            return Err(FixedStepConfigError::ZeroCatchUpSteps);
        }
        Ok(Self {
            step_duration,
            max_catch_up_steps,
        })
    }

    /// Returns the duration of one simulation step.
    #[must_use]
    pub const fn step_duration(self) -> Duration {
        self.step_duration
    }

    /// Returns the maximum number of steps emitted by one advance call.
    #[must_use]
    pub const fn max_catch_up_steps(self) -> u32 {
        self.max_catch_up_steps
    }
}

/// Failure while accumulating elapsed frame time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FixedStepError {
    /// Adding the supplied frame duration would overflow [`Duration`].
    #[error("fixed-step elapsed time overflowed Duration")]
    ElapsedTimeOverflow,
}

/// Work scheduled for one rendered frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedStepPlan {
    /// Number of fixed simulation steps to execute, in order.
    pub steps: u32,
    /// Fractional progress toward the next step, always in `[0, 1)`.
    pub interpolation_alpha: f64,
    /// Whole fixed-step time discarded after reaching the catch-up limit.
    ///
    /// While paused, this is the frame duration ignored by the accumulator.
    pub dropped_time: Duration,
}

/// Converts variable frame durations into bounded, fixed simulation work.
///
/// Excess whole steps are dropped when the catch-up limit is reached; the
/// fractional remainder is retained for interpolation. Pausing clears pending
/// accumulated time, and elapsed time supplied while paused is ignored. The
/// caller executes returned steps sequentially and uses interpolation only
/// for presentation, never as simulation input.
#[derive(Debug, Clone)]
pub struct FixedStepAccumulator {
    config: FixedStepConfig,
    accumulated: Duration,
    paused: bool,
}

impl FixedStepAccumulator {
    /// Creates an accumulator with no pending time and simulation unpaused.
    #[must_use]
    pub const fn new(config: FixedStepConfig) -> Self {
        Self {
            config,
            accumulated: Duration::ZERO,
            paused: false,
        }
    }

    /// Returns the configured fixed-step policy.
    #[must_use]
    pub const fn config(&self) -> FixedStepConfig {
        self.config
    }

    /// Returns whether simulation time is currently paused.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }

    /// Pauses or resumes simulation time.
    ///
    /// Entering the paused state clears pending time, preventing a catch-up
    /// burst when simulation resumes.
    pub fn set_paused(&mut self, paused: bool) {
        if paused && !self.paused {
            self.accumulated = Duration::ZERO;
        }
        self.paused = paused;
    }

    /// Clears pending time without changing the pause state.
    pub fn reset(&mut self) {
        self.accumulated = Duration::ZERO;
    }

    /// Adds one frame duration and returns the bounded simulation work plan.
    pub fn advance(&mut self, frame_delta: Duration) -> Result<FixedStepPlan, FixedStepError> {
        if self.paused {
            return Ok(FixedStepPlan {
                steps: 0,
                interpolation_alpha: 0.0,
                dropped_time: frame_delta,
            });
        }

        let accumulated = self
            .accumulated
            .checked_add(frame_delta)
            .ok_or(FixedStepError::ElapsedTimeOverflow)?;
        let step_nanos = self.config.step_duration.as_nanos();
        let available_steps = accumulated.as_nanos() / step_nanos;
        let steps = available_steps.min(u128::from(self.config.max_catch_up_steps)) as u32;
        let consumed = self.config.step_duration.saturating_mul(steps);
        let after_steps = accumulated - consumed;
        let remainder = duration_from_nanos(after_steps.as_nanos() % step_nanos);
        let dropped_time = after_steps - remainder;

        self.accumulated = remainder;
        let interpolation_alpha = remainder.as_secs_f64() / self.config.step_duration.as_secs_f64();

        Ok(FixedStepPlan {
            steps,
            interpolation_alpha,
            dropped_time,
        })
    }

    /// Returns the currently retained fractional simulation time.
    #[must_use]
    pub const fn accumulated_time(&self) -> Duration {
        self.accumulated
    }
}

fn duration_from_nanos(nanos: u128) -> Duration {
    let seconds = nanos / 1_000_000_000;
    let subsecond_nanos = (nanos % 1_000_000_000) as u32;
    Duration::new(seconds as u64, subsecond_nanos)
}

#[cfg(test)]
mod tests {
    use super::{FixedStepAccumulator, FixedStepConfig, FixedStepConfigError, FixedStepError};
    use std::time::Duration;

    fn config(step_ms: u64, max_steps: u32) -> FixedStepConfig {
        FixedStepConfig::new(Duration::from_millis(step_ms), max_steps)
            .expect("valid fixed-step config")
    }

    #[test]
    fn default_policy_is_sixty_hertz_with_eight_catch_up_steps() {
        let config = FixedStepConfig::default();
        assert_eq!(config.step_duration(), Duration::from_nanos(16_666_667));
        assert_eq!(config.max_catch_up_steps(), 8);
    }

    #[test]
    fn rejects_zero_step_duration_and_zero_catch_up_limit() {
        assert_eq!(
            FixedStepConfig::new(Duration::ZERO, 1),
            Err(FixedStepConfigError::ZeroStepDuration)
        );
        assert_eq!(
            FixedStepConfig::new(Duration::from_millis(1), 0),
            Err(FixedStepConfigError::ZeroCatchUpSteps)
        );
    }

    #[test]
    fn emits_no_step_until_one_full_step_has_accumulated() {
        let mut accumulator = FixedStepAccumulator::new(config(10, 4));
        let plan = accumulator
            .advance(Duration::from_millis(9))
            .expect("advance");
        assert_eq!(plan.steps, 0);
        assert!((plan.interpolation_alpha - 0.9).abs() < 1.0e-12);
        assert_eq!(accumulator.accumulated_time(), Duration::from_millis(9));
    }

    #[test]
    fn emits_multiple_steps_and_preserves_fractional_remainder() {
        let mut accumulator = FixedStepAccumulator::new(config(10, 4));
        let plan = accumulator
            .advance(Duration::from_millis(25))
            .expect("advance");
        assert_eq!(plan.steps, 2);
        assert_eq!(plan.dropped_time, Duration::ZERO);
        assert!((plan.interpolation_alpha - 0.5).abs() < 1.0e-12);
        assert_eq!(accumulator.accumulated_time(), Duration::from_millis(5));
    }

    #[test]
    fn caps_catch_up_work_and_drops_only_excess_whole_steps() {
        let mut accumulator = FixedStepAccumulator::new(config(10, 3));
        let plan = accumulator
            .advance(Duration::from_millis(47))
            .expect("advance");
        assert_eq!(plan.steps, 3);
        assert_eq!(plan.dropped_time, Duration::from_millis(10));
        assert!((plan.interpolation_alpha - 0.7).abs() < 1.0e-12);
        assert_eq!(accumulator.accumulated_time(), Duration::from_millis(7));
    }

    #[test]
    fn frame_partitioning_preserves_total_step_count_without_drops() {
        let mut whole = FixedStepAccumulator::new(config(10, 8));
        let mut split = FixedStepAccumulator::new(config(10, 8));
        let whole_plan = whole
            .advance(Duration::from_millis(35))
            .expect("whole frame");
        let first = split
            .advance(Duration::from_millis(12))
            .expect("first frame");
        let second = split
            .advance(Duration::from_millis(23))
            .expect("second frame");
        assert_eq!(whole_plan.steps, first.steps + second.steps);
        assert_eq!(whole.accumulated_time(), split.accumulated_time());
    }

    #[test]
    fn pausing_clears_remainder_and_ignores_elapsed_time() {
        let mut accumulator = FixedStepAccumulator::new(config(10, 4));
        accumulator
            .advance(Duration::from_millis(7))
            .expect("initial frame");
        accumulator.set_paused(true);
        assert!(accumulator.is_paused());
        assert_eq!(accumulator.accumulated_time(), Duration::ZERO);
        let paused = accumulator
            .advance(Duration::from_secs(2))
            .expect("paused frame");
        assert_eq!(paused.steps, 0);
        assert_eq!(paused.interpolation_alpha, 0.0);
        assert_eq!(paused.dropped_time, Duration::from_secs(2));
        accumulator.set_paused(false);
        let resumed = accumulator
            .advance(Duration::from_millis(5))
            .expect("resumed frame");
        assert_eq!(resumed.steps, 0);
        assert_eq!(resumed.interpolation_alpha, 0.5);
    }

    #[test]
    fn reset_discards_pending_time_without_changing_pause_state() {
        let mut accumulator = FixedStepAccumulator::new(config(10, 4));
        accumulator
            .advance(Duration::from_millis(8))
            .expect("frame");
        accumulator.reset();
        assert_eq!(accumulator.accumulated_time(), Duration::ZERO);
        assert!(!accumulator.is_paused());
        accumulator.set_paused(true);
        accumulator.reset();
        assert!(accumulator.is_paused());
    }

    #[test]
    fn reports_duration_overflow_without_mutating_accumulator() {
        let policy = FixedStepConfig::new(Duration::MAX, 1).expect("valid policy");
        let mut accumulator = FixedStepAccumulator::new(policy);
        accumulator
            .advance(Duration::MAX - Duration::from_nanos(1))
            .expect("first large duration");
        assert_eq!(
            accumulator.advance(Duration::from_nanos(2)),
            Err(FixedStepError::ElapsedTimeOverflow)
        );
        assert_eq!(
            accumulator.accumulated_time(),
            Duration::MAX - Duration::from_nanos(1)
        );
    }

    #[test]
    fn interpolation_fraction_is_always_below_one() {
        let policy = config(7, 5);
        let mut accumulator = FixedStepAccumulator::new(policy);
        for millis in 0..100 {
            let plan = accumulator
                .advance(Duration::from_millis(millis))
                .expect("advance");
            assert!((0.0..1.0).contains(&plan.interpolation_alpha));
            assert!(accumulator.accumulated_time() < policy.step_duration());
        }
    }
}
