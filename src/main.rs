mod config;
mod pipeline;
mod state;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<Arc<Window>>,
    state: Option<state::State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
            self.window = Some(window.clone());

            self.state = Some(pollster::block_on(state::State::new(window)));
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
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
                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::Resized(new_size) => {
                        if let Some(state) = &mut self.state {
                            state.resize(new_size);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        if let Some(state) = &mut self.state {
                            match state.render() {
                                Ok(_) => {}
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        state: None,
    };

    event_loop.run_app(&mut app).unwrap();
}
