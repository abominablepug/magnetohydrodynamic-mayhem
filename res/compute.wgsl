struct Cell {
    u_left: f32,
    u_right: f32,
    v_bottom: f32,
    v_top: f32,
    bx_left: f32,
    bx_right: f32,
    by_bottom: f32,
    by_top: f32,
    p: f32,
    _padding: vec3<f32>,
}

struct SimParams {
    dt: f32,
    density: f32,
    viscosity: f32,
    active_cols: u32,
    active_rows: u32,
    cell_size: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> sim_params: SimParams;
