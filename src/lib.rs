mod config;
mod pipeline;
mod state;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

static PAUSED: AtomicBool = AtomicBool::new(false);
static RESTART: AtomicBool = AtomicBool::new(false);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn toggle_pause() {
    PAUSED.store(!PAUSED.load(Ordering::Relaxed), Ordering::Relaxed);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn restart_simulation() {
    RESTART.store(true, Ordering::Relaxed);
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

struct App {
    window: Option<Arc<Window>>,
    state: Rc<RefCell<Option<state::State>>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let mut attributes = Window::default_attributes()
                .with_title("Magnetohydrodynamic Mayhem")
                .with_inner_size(winit::dpi::PhysicalSize::new(800, 450));

            #[cfg(target_arch = "wasm32")]
            {
                use wasm_bindgen::JsCast;
                use winit::platform::web::WindowAttributesExtWebSys;

                let web_window = web_sys::window().expect("No global window found");
                let document = web_window.document().expect("No document found");

                let canvas = document
                    .get_element_by_id("mhdm-canvas")
                    .expect("Canvas with id 'mhdm-canvas' not found")
                    .dyn_into::<web_sys::HtmlCanvasElement>()
                    .unwrap();

                attributes = attributes.with_canvas(Some(canvas));
            }
            let window = Arc::new(event_loop.create_window(attributes).unwrap());
            self.window = Some(window.clone());

            #[cfg(not(target_arch = "wasm32"))]
            {
                let new_state = pollster::block_on(state::State::new(window));
                *self.state.borrow_mut() = Some(new_state);
            }

            #[cfg(target_arch = "wasm32")]
            {
                let state_clone = self.state.clone();
                let window_clone = window.clone();

                wasm_bindgen_futures::spawn_local(async move {
                    let new_state = state::State::new(window_clone).await;
                    *state_clone.borrow_mut() = Some(new_state);
                });
            }
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(_window) = &self.window {
            event_loop.exit();
        }
        self.window = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(window) = &self.window {
            if window.id() == window_id {
                let mut state_ref = self.state.borrow_mut();
                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::Resized(new_size) => {
                        if let Some(state) = state_ref.as_mut() {
                            state.resize(new_size);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        if let Some(state) = state_ref.as_mut() {
                            if RESTART.swap(false, Ordering::Relaxed) {
                                state.reset();
                            }
                            let is_paused = PAUSED.load(Ordering::Relaxed);

                            match state.render(is_paused) {
                                Ok(_) => {
                                    window.request_redraw();
                                }
                                Err(e) => eprintln!("{:?}", e),
                            }
                        } else {
                            window.request_redraw();
                        }
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        event_loop.exit();
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if let Some(state) = state_ref.as_mut() {
                            state.update_cursor_position(position.x as f32, position.y as f32);
                        }
                    }
                    WindowEvent::MouseInput {
                        state: button_state,
                        button,
                        ..
                    } => {
                        if let Some(state) = state_ref.as_mut() {
                            let is_pressed = button_state == ElementState::Pressed;
                            match button {
                                MouseButton::Left => state.update_mouse_click(0, is_pressed),
                                MouseButton::Right => state.update_mouse_click(1, is_pressed),
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        state: Rc::new(RefCell::new(None)),
    };

    event_loop.run_app(&mut app).unwrap();
}
