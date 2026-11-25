//! Backend-normalized per-frame keyboard and mouse state.
use std::collections::HashSet;

use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

/// Accumulated mouse-wheel movement since the last frame boundary.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ScrollDelta {
    /// Wheel movement in logical lines.
    pub lines: (f32, f32),
    /// Wheel movement in physical pixels.
    pub pixels: (f64, f64),
}

/// Input state for one native window.
///
/// Call begin_frame once at the start of each engine frame, then feed every
/// WindowEvent to process_window_event. Pressed state persists across frames;
/// edge transitions and wheel deltas do not.
#[derive(Debug, Default, Clone)]
pub struct InputState {
    focused: bool,
    pressed_keys: HashSet<KeyCode>,
    just_pressed_keys: HashSet<KeyCode>,
    just_released_keys: HashSet<KeyCode>,
    pressed_mouse_buttons: HashSet<MouseButton>,
    just_pressed_mouse_buttons: HashSet<MouseButton>,
    just_released_mouse_buttons: HashSet<MouseButton>,
    cursor_position: Option<(f64, f64)>,
    scroll: ScrollDelta,
}

impl InputState {
    /// Starts a new input frame, clearing transient transitions and wheel movement.
    pub fn begin_frame(&mut self) {
        self.just_pressed_keys.clear();
        self.just_released_keys.clear();
        self.just_pressed_mouse_buttons.clear();
        self.just_released_mouse_buttons.clear();
        self.scroll = ScrollDelta::default();
    }

    /// Applies a native window event to this window's input state.
    pub fn process_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::Focused(focused) => {
                self.focused = *focused;
                if !focused {
                    self.release_all_inputs();
                    self.cursor_position = None;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    self.update_key(key, event.state);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.update_mouse_button(*button, *state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = Some((position.x, position.y));
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor_position = None;
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::LineDelta(x, y) => {
                    self.scroll.lines.0 += *x;
                    self.scroll.lines.1 += *y;
                }
                MouseScrollDelta::PixelDelta(position) => {
                    self.scroll.pixels.0 += position.x;
                    self.scroll.pixels.1 += position.y;
                }
            },
            _ => {}
        }
    }

    /// Whether the window currently has keyboard focus.
    #[must_use]
    pub const fn focused(&self) -> bool { self.focused }

    /// Whether a physical keyboard key is currently held.
    #[must_use]
    pub fn key_down(&self, key: KeyCode) -> bool { self.pressed_keys.contains(&key) }

    /// Whether a physical keyboard key transitioned to pressed this frame.
    #[must_use]
    pub fn key_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed_keys.contains(&key)
    }

    /// Whether a physical keyboard key transitioned to released this frame.
    #[must_use]
    pub fn key_just_released(&self, key: KeyCode) -> bool {
        self.just_released_keys.contains(&key)
    }

    /// Whether a mouse button is currently held.
    #[must_use]
    pub fn mouse_button_down(&self, button: MouseButton) -> bool {
        self.pressed_mouse_buttons.contains(&button)
    }

    /// Whether a mouse button transitioned to pressed this frame.
    #[must_use]
    pub fn mouse_button_just_pressed(&self, button: MouseButton) -> bool {
        self.just_pressed_mouse_buttons.contains(&button)
    }

    /// Whether a mouse button transitioned to released this frame.
    #[must_use]
    pub fn mouse_button_just_released(&self, button: MouseButton) -> bool {
        self.just_released_mouse_buttons.contains(&button)
    }

    /// Current cursor position in physical window pixels, if inside the window.
    #[must_use]
    pub const fn cursor_position(&self) -> Option<(f64, f64)> { self.cursor_position }

    /// Wheel movement accumulated since the last call to begin_frame.
    #[must_use]
    pub const fn scroll_delta(&self) -> ScrollDelta { self.scroll }

    fn update_key(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if self.pressed_keys.insert(key) {
                    self.just_pressed_keys.insert(key);
                }
            }
            ElementState::Released => {
                if self.pressed_keys.remove(&key) {
                    self.just_released_keys.insert(key);
                }
            }
        }
    }

    fn update_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if self.pressed_mouse_buttons.insert(button) {
                    self.just_pressed_mouse_buttons.insert(button);
                }
            }
            ElementState::Released => {
                if self.pressed_mouse_buttons.remove(&button) {
                    self.just_released_mouse_buttons.insert(button);
                }
            }
        }
    }

    fn release_all_inputs(&mut self) {
        self.just_released_keys.extend(self.pressed_keys.drain());
        self.just_released_mouse_buttons.extend(self.pressed_mouse_buttons.drain());
    }
}

#[cfg(test)]
mod tests {
    use super::InputState;
    use winit::{event::ElementState, keyboard::KeyCode};

    #[test]
    fn key_edges_are_emitted_once_and_persist_until_released() {
        let mut input = InputState::default();
        input.update_key(KeyCode::KeyW, ElementState::Pressed);
        assert!(input.key_down(KeyCode::KeyW));
        assert!(input.key_just_pressed(KeyCode::KeyW));

        input.update_key(KeyCode::KeyW, ElementState::Pressed);
        assert!(input.key_just_pressed(KeyCode::KeyW));

        input.begin_frame();
        assert!(input.key_down(KeyCode::KeyW));
        assert!(!input.key_just_pressed(KeyCode::KeyW));

        input.update_key(KeyCode::KeyW, ElementState::Released);
        assert!(!input.key_down(KeyCode::KeyW));
        assert!(input.key_just_released(KeyCode::KeyW));
    }

    #[test]
    fn duplicate_release_does_not_emit_an_edge() {
        let mut input = InputState::default();
        input.update_key(KeyCode::Space, ElementState::Released);
        assert!(!input.key_just_released(KeyCode::Space));
    }

    #[test]
    fn focus_loss_releases_held_keyboard_and_mouse_inputs() {
        let mut input = InputState::default();
        input.update_key(KeyCode::KeyA, ElementState::Pressed);
        input.update_mouse_button(winit::event::MouseButton::Left, ElementState::Pressed);
        input.release_all_inputs();

        assert!(!input.key_down(KeyCode::KeyA));
        assert!(input.key_just_released(KeyCode::KeyA));
        assert!(!input.mouse_button_down(winit::event::MouseButton::Left));
        assert!(input.mouse_button_just_released(winit::event::MouseButton::Left));
    }

    #[test]
    fn frame_boundary_clears_transient_input() {
        let mut input = InputState::default();
        input.update_key(KeyCode::KeyA, ElementState::Pressed);
        input.scroll.lines = (1.0, -2.0);
        input.begin_frame();

        assert!(!input.key_just_pressed(KeyCode::KeyA));
        assert_eq!(input.scroll_delta(), super::ScrollDelta::default());
    }
}
