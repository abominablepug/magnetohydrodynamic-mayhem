mod pipeline;
mod state;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct App<'a> {
    window: Option<Window>,
    state: Option<state::State<'a>>,
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .expect("Failed to create window");
        self.window = Some(window);
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

    app.state = Some(pollster::block_on(state::State::new(
        app.window.as_ref().unwrap(),
    )));

    event_loop.run_app(&mut app);
}
