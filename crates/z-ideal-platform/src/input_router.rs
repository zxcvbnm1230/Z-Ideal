//! Centralized routing of native window events into independent input states.
use std::collections::HashMap;

use thiserror::Error;
use winit::event::WindowEvent;

use crate::InputState;

/// Stable router-owned identity for one registered native window.
///
/// The platform application stores this token alongside its native
/// `winit::window::WindowId` and uses it to route that window's events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputWindowId(u64);

impl InputWindowId {
    /// Returns the numeric value for diagnostics and external bookkeeping.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Failures returned by centralized input routing.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum InputRouterError {
    /// The monotonically increasing window identity space was exhausted.
    #[error("input window identity space is exhausted")]
    WindowIdExhausted,
    /// An event or query referenced a window that is not registered.
    #[error("input window {0} is not registered")]
    UnknownWindow(u64),
}

/// Owns independent input state for every registered native window.
///
/// Register each native window once and retain the returned token in the
/// platform application's native-window map. Route events using that token.
/// Call `begin_frame` exactly once at each engine frame boundary; it clears
/// transient key/button transitions and accumulated wheel deltas for every
/// registered window while preserving held state.
#[derive(Debug, Default)]
pub struct InputRouter {
    next_window_id: u64,
    windows: HashMap<InputWindowId, InputState>,
}

impl InputRouter {
    /// Registers a window and returns its stable router identity.
    ///
    /// Identities are monotonically allocated and are never reused during this
    /// router's lifetime, so a stale token cannot address a newly registered
    /// window.
    pub fn register_window(&mut self) -> Result<InputWindowId, InputRouterError> {
        let id = self
            .next_window_id
            .checked_add(1)
            .ok_or(InputRouterError::WindowIdExhausted)?;
        let window_id = InputWindowId(id);
        let _ = self.windows.insert(window_id, InputState::default());
        self.next_window_id = id;
        Ok(window_id)
    }

    /// Unregisters a window and discards its input state.
    pub fn unregister_window(
        &mut self,
        window_id: InputWindowId,
    ) -> Result<(), InputRouterError> {
        self.windows
            .remove(&window_id)
            .map(|_| ())
            .ok_or(InputRouterError::UnknownWindow(window_id.get()))
    }

    /// Applies a native event to the registered window identified by `window_id`.
    pub fn process_window_event(
        &mut self,
        window_id: InputWindowId,
        event: &WindowEvent,
    ) -> Result<(), InputRouterError> {
        self.windows
            .get_mut(&window_id)
            .ok_or(InputRouterError::UnknownWindow(window_id.get()))?
            .process_window_event(event);
        Ok(())
    }

    /// Starts a new input frame for every registered window.
    pub fn begin_frame(&mut self) {
        for input in self.windows.values_mut() {
            input.begin_frame();
        }
    }

    /// Returns the input snapshot for a registered window.
    #[must_use]
    pub fn input(&self, window_id: InputWindowId) -> Option<&InputState> {
        self.windows.get(&window_id)
    }

    /// Returns the number of currently registered windows.
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Whether no windows are currently registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{InputRouter, InputRouterError};

    #[test]
    fn registered_windows_have_independent_input_state() {
        let mut router = InputRouter::default();
        let first = router.register_window().expect("first window identity");
        let second = router.register_window().expect("second window identity");

        router
            .process_window_event(first, &winit::event::WindowEvent::Focused(true))
            .expect("first window event");

        assert!(router.input(first).expect("first input").focused());
        assert!(!router.input(second).expect("second input").focused());
        assert_eq!(router.window_count(), 2);
    }

    #[test]
    fn frame_boundary_visits_every_registered_window() {
        let mut router = InputRouter::default();
        let first = router.register_window().expect("first window identity");
        let second = router.register_window().expect("second window identity");

        router
            .process_window_event(first, &winit::event::WindowEvent::Focused(true))
            .expect("first window event");
        router
            .process_window_event(second, &winit::event::WindowEvent::Focused(true))
            .expect("second window event");
        router.begin_frame();

        assert!(router.input(first).expect("first input").focused());
        assert!(router.input(second).expect("second input").focused());
    }

    #[test]
    fn unregister_rejects_stale_identity_and_does_not_reuse_it() {
        let mut router = InputRouter::default();
        let removed = router.register_window().expect("window identity");
        router
            .unregister_window(removed)
            .expect("registered window removal");

        assert!(router.is_empty());
        assert_eq!(
            router.unregister_window(removed),
            Err(InputRouterError::UnknownWindow(removed.get()))
        );

        let replacement = router.register_window().expect("replacement identity");
        assert_ne!(removed, replacement);
        assert_eq!(router.window_count(), 1);
    }

    #[test]
    fn events_for_unregistered_windows_are_rejected() {
        let mut router = InputRouter::default();
        let window = router.register_window().expect("window identity");
        router
            .unregister_window(window)
            .expect("registered window removal");

        assert_eq!(
            router.process_window_event(
                window,
                &winit::event::WindowEvent::Focused(true)
            ),
            Err(InputRouterError::UnknownWindow(window.get()))
        );
    }
}
