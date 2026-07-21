struct SimParams {
    dt: f32,
    fluid_density: f32,
    viscosity: f32,
    active_cols: u32,
    active_rows: u32,
    cell_size: u32,
    _padding: vec2<u32>,
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

fn bilinear_prep(x: f32, y: f32, is_x: bool) -> vec4<f32> {
    var grid_x: f32;
    var grid_y: f32;

    if is_x {
        var grid_x = (x / f32(sim_params.cell_size)) - 0.5;
        var grid_y = y / f32(sim_params.cell_size);
    } else {
        var grid_x = x / f32(sim_params.cell_size);
        var grid_y = (y / f32(sim_params.cell_size)) - 0.5;
    }

    let col = i32(floor(grid_x));
    let row = i32(floor(grid_y));

    let tx = fract(grid_x);
    let ty = fract(grid_y);

    return vec4<f32>(f32(row), f32(col), tx, ty);
}

fn bilinear_interpolation_u(x: f32, y: f32) -> f32 {
    let prep = bilinear_prep(x, y, true);
    let row = i32(prep.x);
    let col = i32(prep.y);
    let tx = prep.z;
    let ty = prep.w;

    let u00 = grid_in[get_cell_index(row, col)].u;
    let u10 = grid_in[get_cell_index(row, col + 1)].u;
    let u01 = grid_in[get_cell_index(row + 1, col)].u;
    let u11 = grid_in[get_cell_index(row + 1, col + 1)].u;

    let mix_bottom = mix(u00, u10, tx);
    let mix_top = mix(u01, u11, tx);
    return mix(mix_bottom, mix_top, ty);
}

fn bilinear_interpolation_v(x: f32, y: f32) -> f32 {
    let prep = bilinear_prep(x, y, false);
    let row = i32(prep.x);
    let col = i32(prep.y);
    let tx = prep.z;
    let ty = prep.w;

    let v00 = grid_in[get_cell_index(row, col)].v;
    let v10 = grid_in[get_cell_index(row, col + 1)].v;
    let v01 = grid_in[get_cell_index(row + 1, col)].v;
    let v11 = grid_in[get_cell_index(row + 1, col + 1)].v;

    let mix_bottom = mix(v00, v10, tx);
    let mix_top = mix(v01, v11, tx);
    return mix(mix_bottom, mix_top, ty);
}

fn bilinear_interpolation_bx(x: f32, y: f32) -> f32 {
    let prep = bilinear_prep(x, y, true);
    let row = i32(prep.x);
    let col = i32(prep.y);
    let tx = prep.z;
    let ty = prep.w;

    let bx00 = grid_in[get_cell_index(row, col)].bx;
    let bx10 = grid_in[get_cell_index(row, col + 1)].bx;
    let bx01 = grid_in[get_cell_index(row + 1, col)].bx;
    let bx11 = grid_in[get_cell_index(row + 1, col + 1)].bx;

    let mix_bottom = mix(bx00, bx10, tx);
    let mix_top = mix(bx01, bx11, tx);
    return mix(mix_bottom, mix_top, ty);
}

fn bilinear_interpolation_by(x: f32, y: f32) -> f32 {
    let prep = bilinear_prep(x, y, false);
    let row = i32(prep.x);
    let col = i32(prep.y);
    let tx = prep.z;
    let ty = prep.w;

    let by00 = grid_in[get_cell_index(row, col)].by;
    let by10 = grid_in[get_cell_index(row, col + 1)].by;
    let by01 = grid_in[get_cell_index(row + 1, col)].by;
    let by11 = grid_in[get_cell_index(row + 1, col + 1)].by;

    let mix_bottom = mix(by00, by10, tx);
    let mix_top = mix(by01, by11, tx);
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
    let delta = f32(sim_params.cell_size);
    let dt = sim_params.dt;

    let u_x = f32(col) * delta;
    let u_y = (f32(row) + 0.5) * delta;

    var local_u = cell.u;
    var local_v = bilinear_interpolation_v(u_x, u_y);

    let u_old_x = u_x - local_u * dt;
    let u_old_y = u_y - local_v * dt;

    cell.u = bilinear_interpolation_u(u_old_x, u_old_y);

    let v_x = (f32(col) + 0.5) * delta;
    let v_y = f32(row) * delta;

    local_u = bilinear_interpolation_u(v_x, v_y);
    local_v = cell.v;

    let v_old_x = v_x - local_u * dt;
    let v_old_y = v_y - local_v * dt;

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
    var u_right = grid_in[get_cell_index(y, x + 1)].u;
    var v_up = grid_in[get_cell_index(y + 1, x)].v;

    if x == i32(sim_params.active_cols - 1) {
        u_right = 0.0;
    }
    if y == i32(sim_params.active_rows - 1) {
        v_up = 0.0;
    }

    let div_u = (u_right - cell.u) / delta;
    let div_v = (v_up - cell.v) / delta;

    cell.divergence = div_u + div_v;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn jacobi_iteration_step(@builtin(global_invocation_id) id: vec3<u32>) {
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
    var p_left = grid_in[get_cell_index(y, x - 1)].p;
    var p_right = grid_in[get_cell_index(y, x + 1)].p;
    var p_down = grid_in[get_cell_index(y - 1, x)].p;
    var p_up = grid_in[get_cell_index(y + 1, x)].p;

    let b = (sim_params.fluid_density * delta * delta * cell.divergence) / sim_params.dt;

    let p_new = (p_left + p_right + p_down + p_up - b) / 4.0;

    cell.p = p_new;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn pressure_gradient_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= sim_params.active_cols || row >= sim_params.active_rows {
        return;
    }

    let x = i32(col);
    let y = i32(row);
    let delta = f32(sim_params.cell_size);
    let fd = sim_params.fluid_density;

    let index = get_cell_index(y, x);
    var cell = grid_in[index];
    var p_left = grid_in[get_cell_index(y, x - 1)].p;
    var p_down = grid_in[get_cell_index(y - 1, x)].p;

    let gradient_p = vec2<f32>(
        (cell.p - p_left) / (delta * fd),
        (cell.p - p_down) / (delta * fd)
    ) * sim_params.dt;

    cell.u -= gradient_p.x;
    cell.v -= gradient_p.y;

    if x == 0 {
        cell.u = 0.0;
    }
    if y == 0 {
        cell.v = 0.0;
    }

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn magnetic_induction_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= sim_params.active_cols || row >= sim_params.active_rows {
        return;
    }

    let index = get_cell_index(i32(row), i32(col));
    var cell = grid_in[index];
    let cell_right = grid_in[get_cell_index(i32(row), i32(col + 1))];
    let cell_up = grid_in[get_cell_index(i32(row + 1), i32(col))];
    let delta = f32(sim_params.cell_size);
    let dt = sim_params.dt;

    // stretching term x-component

    let bx_x = f32(col) * delta;
    let bx_y = (f32(row) + 0.5) * delta;

    var local_bx = cell.bx;
    var local_by = bilinear_interpolation_by(bx_x, bx_y);

    let du_dx = (cell_right.u - cell.u) / delta;
    let du_dy = (cell_up.u - cell.u) / delta;

    let b_stretch_x = (local_bx * du_dx + local_by * du_dy) * dt;

    // advection term x-component

    var local_u = cell.u;
    var local_v = bilinear_interpolation_v(bx_x, bx_y);

    let bx_old_x = bx_x - local_u * dt;
    let bx_old_y = bx_y - local_v * dt;

    cell.bx = bilinear_interpolation_bx(bx_old_x, bx_old_y) + b_stretch_x;

    // stretching term y-component

    let by_x = (f32(col) + 0.5) * delta;
    let by_y = f32(row) * delta;

    local_bx = bilinear_interpolation_bx(by_x, by_y);
    local_by = cell.by;

    let dv_dx = (cell_right.v - cell.v) / delta;
    let dv_dy = (cell_up.v - cell.v) / delta;

    let b_stretch_y = (local_bx * dv_dx + local_by * dv_dy) * dt;

    // advection term y-component

    local_u = bilinear_interpolation_u(by_x, by_y);
    local_v = cell.v;

    let by_old_x = by_x - local_u * dt;
    let by_old_y = by_y - local_v * dt;

    cell.by = bilinear_interpolation_by(by_old_x, by_old_y) + b_stretch_y;

    grid_out[index] = cell;
}
