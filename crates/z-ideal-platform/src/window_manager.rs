//! Native-window ownership, input routing, and surface-change tracking.
use std::collections::HashMap;

use thiserror::Error;
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    window::{Window, WindowId},
};

use crate::{InputRouter, InputRouterError, InputState, InputWindowId};

/// Errors returned by the native-window registry.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum WindowManagerError {
    /// A native window was registered more than once.
    #[error("native window is already registered")]
    DuplicateWindow,
    /// An operation referenced a native window that is not registered.
    #[error("native window is not registered")]
    UnknownWindow,
    /// Input routing rejected an operation.
    #[error("input routing failed: {0}")]
    InputRouter(#[from] InputRouterError),
    /// A window's surface generation counter was exhausted.
    #[error("window surface generation counter is exhausted")]
    SurfaceGenerationExhausted,
}

/// Current physical-pixel and focus state for a native window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowMetrics {
    physical_size: [u32; 2],
    scale_factor: f64,
    focused: bool,
    surface_generation: u64,
}

impl WindowMetrics {
    fn new(size: PhysicalSize<u32>, scale_factor: f64) -> Self {
        Self {
            physical_size: [size.width, size.height],
            scale_factor,
            focused: false,
            surface_generation: 0,
        }
    }

    /// Current drawable extent in physical pixels.
    #[must_use]
    pub const fn physical_size(&self) -> [u32; 2] {
        self.physical_size
    }

    /// Current native scale factor.
    #[must_use]
    pub const fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Whether the native window currently has focus.
    #[must_use]
    pub const fn focused(&self) -> bool {
        self.focused
    }

    /// Whether the window currently has a zero-sized drawable extent.
    #[must_use]
    pub const fn minimized(&self) -> bool {
        self.physical_size[0] == 0 || self.physical_size[1] == 0
    }

    /// Monotonically increasing generation for surface-relevant metric changes.
    ///
    /// Renderer integrations can compare this value with the generation last
    /// used to create a swapchain and recreate only when it changes.
    #[must_use]
    pub const fn surface_generation(&self) -> u64 {
        self.surface_generation
    }

    fn update_surface(
        &mut self,
        size: PhysicalSize<u32>,
        scale_factor: f64,
    ) -> Result<bool, WindowManagerError> {
        let next_size = [size.width, size.height];
        if self.physical_size == next_size && self.scale_factor == scale_factor {
            return Ok(false);
        }

        self.surface_generation = self
            .surface_generation
            .checked_add(1)
            .ok_or(WindowManagerError::SurfaceGenerationExhausted)?;
        self.physical_size = next_size;
        self.scale_factor = scale_factor;
        Ok(true)
    }
}

/// Changes observed while processing one native window event.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WindowEventOutcome {
    /// A close request was received; the application decides whether to close.
    pub close_requested: bool,
    /// A redraw request was received.
    pub redraw_requested: bool,
    /// Surface-relevant metrics changed and presentation resources may need recreation.
    pub surface_changed: bool,
    /// Keyboard focus changed.
    pub focus_changed: bool,
}

/// Owns native windows and routes their events to per-window input state.
///
/// All methods must be called on the native event-loop thread, matching winit's
/// window ownership requirements. The manager owns each registered `Window`;
/// removing a window returns ownership to the caller. Vulkan remains outside
/// this crate: consumers observe `surface_generation` and decide when and how
/// to recreate presentation resources.
#[derive(Default)]
pub struct WindowManager {
    windows: HashMap<WindowId, ManagedWindow>,
    input: InputRouter,
}

struct ManagedWindow {
    window: Window,
    input_id: InputWindowId,
    metrics: WindowMetrics,
}

impl WindowManager {
    /// Creates an empty native-window registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers and takes ownership of a native window.
    ///
    /// The returned input token is stable until the window is removed. Native
    /// `WindowId` values are used for event lookup; the separate input token
    /// prevents the platform API from exposing the router's internal map.
    pub fn register_window(&mut self, window: Window) -> Result<InputWindowId, WindowManagerError> {
        let native_id = window.id();
        if self.windows.contains_key(&native_id) {
            return Err(WindowManagerError::DuplicateWindow);
        }

        let input_id = self.input.register_window()?;
        let size = window.inner_size();
        let metrics = WindowMetrics::new(size, window.scale_factor());
        let _ = self.windows.insert(
            native_id,
            ManagedWindow {
                window,
                input_id,
                metrics,
            },
        );
        Ok(input_id)
    }

    /// Removes a native window and returns ownership to the caller.
    pub fn remove_window(&mut self, native_id: WindowId) -> Result<Window, WindowManagerError> {
        let input_id = self
            .windows
            .get(&native_id)
            .ok_or(WindowManagerError::UnknownWindow)?
            .input_id;
        self.input.unregister_window(input_id)?;
        self.windows
            .remove(&native_id)
            .map(|managed| managed.window)
            .ok_or(WindowManagerError::UnknownWindow)
    }

    /// Processes a native event, updating input and window metrics.
    pub fn process_window_event(
        &mut self,
        native_id: WindowId,
        event: &WindowEvent,
    ) -> Result<WindowEventOutcome, WindowManagerError> {
        let managed = self
            .windows
            .get_mut(&native_id)
            .ok_or(WindowManagerError::UnknownWindow)?;

        self.input.process_window_event(managed.input_id, event)?;

        let mut outcome = WindowEventOutcome::default();
        match event {
            WindowEvent::Resized(size) => {
                outcome.surface_changed = managed
                    .metrics
                    .update_surface(*size, managed.metrics.scale_factor)?;
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                outcome.surface_changed = managed
                    .metrics
                    .update_surface(managed.window.inner_size(), *scale_factor)?;
            }
            WindowEvent::Focused(focused) => {
                outcome.focus_changed = managed.metrics.focused != *focused;
                managed.metrics.focused = *focused;
            }
            WindowEvent::CloseRequested => outcome.close_requested = true,
            WindowEvent::RedrawRequested => outcome.redraw_requested = true,
            _ => {}
        }
        Ok(outcome)
    }

    /// Starts a new input frame for every registered window.
    pub fn begin_frame(&mut self) {
        self.input.begin_frame();
    }

    /// Returns the native window, if registered.
    #[must_use]
    pub fn window(&self, native_id: WindowId) -> Option<&Window> {
        self.windows.get(&native_id).map(|managed| &managed.window)
    }

    /// Returns the input snapshot associated with a native window.
    #[must_use]
    pub fn input(&self, native_id: WindowId) -> Option<&InputState> {
        self.windows
            .get(&native_id)
            .and_then(|managed| self.input.input(managed.input_id))
    }

    /// Returns current drawable and focus metrics for a native window.
    #[must_use]
    pub fn metrics(&self, native_id: WindowId) -> Option<WindowMetrics> {
        self.windows.get(&native_id).map(|managed| managed.metrics)
    }

    /// Requests redraw for one registered window.
    pub fn request_redraw(&self, native_id: WindowId) -> Result<(), WindowManagerError> {
        self.windows
            .get(&native_id)
            .ok_or(WindowManagerError::UnknownWindow)?
            .window
            .request_redraw();
        Ok(())
    }

    /// Requests redraw for every registered window.
    pub fn request_redraw_all(&self) {
        for managed in self.windows.values() {
            managed.window.request_redraw();
        }
    }

    /// Returns the number of registered native windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Whether no native windows are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{WindowManagerError, WindowMetrics};
    use winit::dpi::PhysicalSize;

    #[test]
    fn initial_metrics_are_not_minimized_and_generation_starts_at_zero() {
        let metrics = WindowMetrics::new(PhysicalSize::new(1280, 720), 1.0);
        assert_eq!(metrics.physical_size(), [1280, 720]);
        assert_eq!(metrics.scale_factor(), 1.0);
        assert!(!metrics.minimized());
        assert_eq!(metrics.surface_generation(), 0);
    }

    #[test]
    fn zero_extent_marks_window_minimized_and_advances_generation() {
        let mut metrics = WindowMetrics::new(PhysicalSize::new(1280, 720), 1.0);
        assert!(
            metrics
                .update_surface(PhysicalSize::new(0, 0), 1.0)
                .expect("surface metrics update")
        );

        assert!(metrics.minimized());
        assert_eq!(metrics.physical_size(), [0, 0]);
        assert_eq!(metrics.surface_generation(), 1);
    }

    #[test]
    fn nonzero_extent_restores_window_and_advances_generation() {
        let mut metrics = WindowMetrics::new(PhysicalSize::new(0, 0), 1.0);
        assert!(
            metrics
                .update_surface(PhysicalSize::new(1920, 1080), 1.0)
                .expect("surface metrics update")
        );

        assert!(!metrics.minimized());
        assert_eq!(metrics.physical_size(), [1920, 1080]);
        assert_eq!(metrics.surface_generation(), 1);
    }

    #[test]
    fn unchanged_metrics_do_not_advance_generation() {
        let mut metrics = WindowMetrics::new(PhysicalSize::new(800, 600), 1.0);
        assert!(
            !metrics
                .update_surface(PhysicalSize::new(800, 600), 1.0)
                .expect("unchanged surface metrics")
        );

        assert_eq!(metrics.surface_generation(), 0);
    }

    #[test]
    fn scale_factor_change_advances_surface_generation() {
        let mut metrics = WindowMetrics::new(PhysicalSize::new(800, 600), 1.0);
        assert!(
            metrics
                .update_surface(PhysicalSize::new(800, 600), 2.0)
                .expect("scale factor update")
        );

        assert_eq!(metrics.scale_factor(), 2.0);
        assert_eq!(metrics.surface_generation(), 1);
    }

    #[test]
    fn surface_generation_overflow_is_reported_without_partial_update() {
        let mut metrics = WindowMetrics::new(PhysicalSize::new(800, 600), 1.0);
        metrics.surface_generation = u64::MAX;

        assert_eq!(
            metrics.update_surface(PhysicalSize::new(1024, 768), 1.0),
            Err(WindowManagerError::SurfaceGenerationExhausted)
        );
        assert_eq!(metrics.physical_size(), [800, 600]);
        assert_eq!(metrics.surface_generation(), u64::MAX);
    }
}
