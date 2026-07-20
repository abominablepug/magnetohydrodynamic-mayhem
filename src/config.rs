pub const CELL_SIZE: u32 = 4;
pub const MAX_SCREEN_WIDTH: u32 = 3840;
pub const MAX_SCREEN_HEIGHT: u32 = 2160;

pub const MAX_COLS: u32 = MAX_SCREEN_WIDTH / CELL_SIZE;
pub const MAX_ROWS: u32 = MAX_SCREEN_HEIGHT / CELL_SIZE;
pub const MAX_CELLS: u32 = MAX_COLS * MAX_ROWS;

pub const DT: f32 = 0.01;
pub const FLUID_DENSITY: f32 = 1.0;
pub const VISCOSITY: f32 = 0.1;
