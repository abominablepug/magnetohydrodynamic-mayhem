struct SimParams {
    dt: f32,
    density: f32,
    viscosity: f32,
    active_cols: u32,
    active_rows: u32,
    cell_size: u32,
    fluid_density: f32,
    _padding: u32,
}

struct Cell {
    u: f32,
    v: f32,
    bx: f32,
    by: f32,
    p: f32,
    divergence: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read> grid_in: array<Cell>;
@group(0) @binding(2) var<storage, read_write> grid_out: array<Cell>;

fn get_cell_index(row: i32, col: i32) -> u32 {
    let c = clamp(col, 0, i32(sim_params.active_cols) - 1);
    let r = clamp(row, 0, i32(sim_params.active_rows) - 1);
    return u32(r) * sim_params.active_cols + u32(c);
}

fn bilinear_interpolation_u(x: f32, y: f32) -> f32 {
    let grid_x = x / f32(sim_params.cell_size);
    let grid_y = (y / f32(sim_params.cell_size)) - 0.5;

    let col = i32(floor(grid_x));
    let row = i32(floor(grid_y));

    let tx = fract(grid_x);
    let ty = fract(grid_y);

    let v00 = grid_in[get_cell_index(row, col)].u;
    let v10 = grid_in[get_cell_index(row, col + 1)].u;
    let v01 = grid_in[get_cell_index(row + 1, col)].u;
    let v11 = grid_in[get_cell_index(row + 1, col + 1)].u;

    let mix_bottom = mix(v00, v10, tx);
    let mix_top = mix(v01, v11, tx);
    return mix(mix_bottom, mix_top, ty);
}

fn bilinear_interpolation_v(x: f32, y: f32) -> f32 {
    let grid_x = (x / f32(sim_params.cell_size)) - 0.5;
    let grid_y = y / f32(sim_params.cell_size);

    let col = i32(floor(grid_x));
    let row = i32(floor(grid_y));

    let tx = fract(grid_x);
    let ty = fract(grid_y);

    let v00 = grid_in[get_cell_index(row, col)].v;
    let v10 = grid_in[get_cell_index(row, col + 1)].v;
    let v01 = grid_in[get_cell_index(row + 1, col)].v;
    let v11 = grid_in[get_cell_index(row + 1, col + 1)].v;

    let mix_bottom = mix(v00, v10, tx);
    let mix_top = mix(v01, v11, tx);
    return mix(mix_bottom, mix_top, ty);
}

@compute
@workgroup_size(8, 8, 1)
fn fluid_advection_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= sim_params.active_cols || row >= sim_params.active_rows {
        return;
    }

    let index = get_cell_index(i32(row), i32(col));
    var cell = grid_in[index];

    let u_x = f32(col) * f32(sim_params.cell_size);
    let u_y = (f32(row) + 0.5) * f32(sim_params.cell_size);

    let local_u = cell.u;
    let local_v = bilinear_interpolation_v(u_x, u_y);

    let u_old_x = u_x - local_u * sim_params.dt;
    let u_old_y = u_y - local_v * sim_params.dt;

    cell.u = bilinear_interpolation_u(u_old_x, u_old_y);

    let v_x = (f32(col) + 0.5) * f32(sim_params.cell_size);
    let v_y = f32(row) * f32(sim_params.cell_size);

    let local_u_v = bilinear_interpolation_u(v_x, v_y);
    let local_v_v = cell.v;

    let v_old_x = v_x - local_u_v * sim_params.dt;
    let v_old_y = v_y - local_v_v * sim_params.dt;

    cell.v = bilinear_interpolation_v(v_old_x, v_old_y);

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn compute_divergence_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= sim_params.active_cols || row >= sim_params.active_rows {
        return;
    }

    let x = i32(col);
    let y = i32(row);
    let delta = f32(sim_params.cell_size);

    let index = get_cell_index(y, x);
    var cell = grid_in[index];
    var cell_right = grid_in[get_cell_index(y, x + 1)];
    var cell_up = grid_in[get_cell_index(y + 1, x)];

    let div_u = (cell_right.u - cell.u) / delta;
    let div_v = (cell_up.v - cell.v) / delta;

    cell.divergence = div_u + div_v;
}
