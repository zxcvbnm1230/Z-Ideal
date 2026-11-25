//! Cross-platform native window and event-loop integration.
#![forbid(unsafe_code)]

use thiserror::Error;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    error::EventLoopError,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

/// Stable package identifier.
pub const CRATE_NAME: &str = "z-ideal-platform";

/// Validated initial native-window configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    /// Window title.
    pub title: String,
    /// Initial logical width.
    pub width: u32,
    /// Initial logical height.
    pub height: u32,
    /// Initial visibility.
    pub visible: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Z Ideal".to_owned(),
            width: 1280,
            height: 720,
            visible: true,
        }
    }
}

impl WindowConfig {
    /// Rejects invalid dimensions and blank titles.
    pub fn validate(&self) -> Result<(), PlatformError> {
        if self.title.trim().is_empty() {
            return Err(PlatformError::InvalidWindowConfig(
                "title must not be empty".to_owned(),
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err(PlatformError::InvalidWindowConfig(
                "width and height must both be greater than zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Converts this configuration to winit window attributes.
    pub fn attributes(&self) -> Result<WindowAttributes, PlatformError> {
        self.validate()?;
        Ok(Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(self.width, self.height))
            .with_visible(self.visible))
    }
}

/// Native platform startup and application callback errors.
#[derive(Debug, Error)]
pub enum PlatformError {
    /// Window configuration was invalid.
    #[error("invalid window configuration: {0}")]
    InvalidWindowConfig(String),
    /// Native event-loop creation or execution failed.
    #[error("window event loop failed: {0}")]
    EventLoop(String),
    /// An application callback failed.
    #[error("platform application failed: {0}")]
    Application(String),
}

impl From<EventLoopError> for PlatformError {
    fn from(error: EventLoopError) -> Self {
        Self::EventLoop(error.to_string())
    }
}

/// Callbacks invoked on the native event-loop thread.
pub trait PlatformApplication: 'static {
    /// Called when native windows may be created or restored.
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) -> Result<(), PlatformError> {
        Ok(())
    }

    /// Receives events for native windows.
    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        _event: WindowEvent,
    ) -> Result<(), PlatformError> {
        Ok(())
    }

    /// Called before the event loop waits for more events.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) -> Result<(), PlatformError> {
        Ok(())
    }

    /// Called while the event loop exits.
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {}
}

struct Runner<A> {
    application: A,
    error: Option<PlatformError>,
}

impl<A: PlatformApplication> Runner<A> {
    fn finish_callback(&mut self, event_loop: &ActiveEventLoop, result: Result<(), PlatformError>) {
        if let Err(error) = result {
            self.error = Some(error);
            event_loop.exit();
        }
    }
}

impl<A: PlatformApplication> ApplicationHandler for Runner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = self.application.resumed(event_loop);
        self.finish_callback(event_loop, result);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let result = self.application.window_event(event_loop, window_id, event);
        self.finish_callback(event_loop, result);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let result = self.application.about_to_wait(event_loop);
        self.finish_callback(event_loop, result);
    }

    fn exiting(&mut self, event_loop: &ActiveEventLoop) {
        self.application.exiting(event_loop);
    }
}

/// Runs an application using the native platform event loop.
pub struct PlatformEventLoop;

impl PlatformEventLoop {
    /// Creates and runs the event loop.
    pub fn run<A: PlatformApplication>(application: A) -> Result<(), PlatformError> {
        let event_loop = EventLoop::new()?;
        let mut runner = Runner {
            application,
            error: None,
        };
        event_loop.run_app(&mut runner)?;
        match runner.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Creates a native window from an active event-loop callback.
pub fn create_window(
    event_loop: &ActiveEventLoop,
    config: &WindowConfig,
) -> Result<Window, PlatformError> {
    event_loop
        .create_window(config.attributes()?)
        .map_err(|error| PlatformError::EventLoop(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{CRATE_NAME, PlatformError, WindowConfig};

    #[test]
    fn crate_name_is_stable() {
        assert_eq!(CRATE_NAME, "z-ideal-platform");
    }

    #[test]
    fn default_window_configuration_is_valid() {
        assert!(WindowConfig::default().validate().is_ok());
    }

    #[test]
    fn rejects_blank_title() {
        let config = WindowConfig {
            title: " ".to_owned(),
            ..WindowConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(PlatformError::InvalidWindowConfig(_))
        ));
    }

    #[test]
    fn rejects_zero_dimensions() {
        let config = WindowConfig {
            height: 0,
            ..WindowConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(PlatformError::InvalidWindowConfig(_))
        ));
    }
}
