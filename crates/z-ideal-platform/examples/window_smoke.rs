//! Manual native-window lifecycle and resize smoke test.
use std::time::{Duration, Instant};

use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow},
    window::WindowId,
};
use z_ideal_platform::{
    PlatformApplication, PlatformError, PlatformEventLoop, WindowConfig, WindowManager,
    create_window,
};

const RESIZED_EXTENT: [u32; 2] = [96, 64];

#[derive(Default)]
struct ResizeSmoke {
    windows: WindowManager,
    window_id: Option<WindowId>,
    deadline: Option<Instant>,
}

impl PlatformApplication for ResizeSmoke {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) -> Result<(), PlatformError> {
        if self.window_id.is_some() {
            return Ok(());
        }

        let window = create_window(
            event_loop,
            &WindowConfig {
                title: "Z Ideal native window resize validation".to_owned(),
                width: 64,
                height: 64,
                visible: true,
            },
        )?;
        let window_id = window.id();
        self.windows
            .register_window(window)
            .map_err(|error| PlatformError::Application(error.to_string()))?;
        self.window_id = Some(window_id);
        self.deadline = Some(Instant::now() + Duration::from_secs(5));

        let window = self.windows.window(window_id).ok_or_else(|| {
            PlatformError::Application("registered native window was not found".to_owned())
        })?;
        if let Some(actual) =
            window.request_inner_size(PhysicalSize::new(RESIZED_EXTENT[0], RESIZED_EXTENT[1]))
            && [actual.width, actual.height] != RESIZED_EXTENT
        {
            return Err(PlatformError::Application(format!(
                "window manager acknowledged unexpected resize: {}x{}",
                actual.width, actual.height
            )));
        }
        Ok(())
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) -> Result<(), PlatformError> {
        if self.window_id != Some(window_id) {
            return Ok(());
        }

        self.windows
            .process_window_event(window_id, &event)
            .map_err(|error| PlatformError::Application(error.to_string()))?;

        if let WindowEvent::Resized(size) = event
            && [size.width, size.height] == RESIZED_EXTENT
        {
            let metrics = self.windows.metrics(window_id).ok_or_else(|| {
                PlatformError::Application("resized window metrics were not found".to_owned())
            })?;
            if metrics.physical_size() != RESIZED_EXTENT || metrics.surface_generation() == 0 {
                return Err(PlatformError::Application(
                    "native resize did not advance the surface generation".to_owned(),
                ));
            }

            let window = self
                .windows
                .remove_window(window_id)
                .map_err(|error| PlatformError::Application(error.to_string()))?;
            drop(window);
            self.window_id = None;
            event_loop.exit();
        }
        Ok(())
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) -> Result<(), PlatformError> {
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(PlatformError::Application(
                    "timed out waiting for the native resize event".to_owned(),
                ));
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
        }
        Ok(())
    }
}

fn main() {
    PlatformEventLoop::run(ResizeSmoke::default())
        .expect("native window startup, resize, and shutdown smoke test");
}
