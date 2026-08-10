struct SimParams {
    dt: f32,
    fluid_density: f32,
    viscosity: f32,
    active_cols: u32,
    active_rows: u32,
    cell_size: u32,
    mouse_x: f32,
    mouse_y: f32,
    mouse_left_clicked: u32,
    mouse_right_clicked: u32,
    _padding: vec2<u32>,
}

struct Cell {
    u: f32,
    v: f32,
    bx: f32,
    by: f32,
    p: f32,
    fluid_divergence: f32,
    phi: f32,
    magnetic_divergence: f32,
    current_density: f32,
    dye: f32,
    _padding: array<f32, 2>,
}

struct Particle {
    x: f32,
    y: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read> grid_in: array<Cell>;
@group(0) @binding(2) var<storage, read_write> grid_out: array<Cell>;
@group(0) @binding(3) var<storage, read_write> particles: array<Particle>;

fn get_cell_index(row: i32, col: i32) -> u32 {
    let c = clamp(col, 0, i32(sim_params.active_cols) - 1);
    let r = clamp(row, 0, i32(sim_params.active_rows) - 1);
    return u32(r) * sim_params.active_cols + u32(c);
}

fn bilinear_prep(x: f32, y: f32, is_x: bool) -> vec4<f32> {
    var grid_x: f32;
    var grid_y: f32;

    if is_x {
        grid_x = (x / f32(sim_params.cell_size)) - 0.5;
        grid_y = y / f32(sim_params.cell_size);
    } else {
        grid_x = x / f32(sim_params.cell_size);
        grid_y = (y / f32(sim_params.cell_size)) - 0.5;
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

    let u00 = get_u(row, col);
    let u10 = get_u(row, col + 1);
    let u01 = get_u(row + 1, col);
    let u11 = get_u(row + 1, col + 1);

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

    let v00 = get_v(row, col);
    let v10 = get_v(row, col + 1);
    let v01 = get_v(row + 1, col);
    let v11 = get_v(row + 1, col + 1);

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

    let bx00 = get_bx(row, col);
    let bx10 = get_bx(row, col + 1);
    let bx01 = get_bx(row + 1, col);
    let bx11 = get_bx(row + 1, col + 1);

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

    let by00 = get_by(row, col);
    let by10 = get_by(row, col + 1);
    let by01 = get_by(row + 1, col);
    let by11 = get_by(row + 1, col + 1);

    let mix_bottom = mix(by00, by10, tx);
    let mix_top = mix(by01, by11, tx);
    return mix(mix_bottom, mix_top, ty);
}

fn bilinear_interpolation_dye(x: f32, y: f32) -> f32 {
    let prep = bilinear_prep(x, y, false);
    let row = i32(prep.x);
    let col = i32(prep.y);
    let tx = prep.z;
    let ty = prep.w;

    let dye00 = grid_in[get_cell_index(row, col)].dye;
    let dye10 = grid_in[get_cell_index(row, col + 1)].dye;
    let dye01 = grid_in[get_cell_index(row + 1, col)].dye;
    let dye11 = grid_in[get_cell_index(row + 1, col + 1)].dye;

    let mix_bottom = mix(dye00, dye10, tx);
    let mix_top = mix(dye01, dye11, tx);
    return mix(mix_bottom, mix_top, ty);
}

fn get_u(row: i32, col: i32) -> f32 {
    if col <= 0 || col >= i32(sim_params.active_cols) {
        return 0.0;
    }

    if row < 0 {
        return -grid_in[get_cell_index(0, col)].u;
    } else if row >= i32(sim_params.active_rows) {
        return -grid_in[get_cell_index(i32(sim_params.active_rows - 1), col)].u;
    }

    return grid_in[get_cell_index(row, col)].u;
}

fn get_v(row: i32, col: i32) -> f32 {
    if row <= 0 || row >= i32(sim_params.active_rows) {
        return 0.0;
    }

    if col < 0 {
        return -grid_in[get_cell_index(row, 0)].v;
    } else if col >= i32(sim_params.active_cols) {
        return -grid_in[get_cell_index(row, i32(sim_params.active_cols - 1))].v;
    }

    return grid_in[get_cell_index(row, col)].v;
}

fn get_bx(row: i32, col: i32) -> f32 {
    if col <= 0 || col >= i32(sim_params.active_cols) {
        return 0.0;
    }

    if row < 0 {
        return grid_in[get_cell_index(0, col)].bx;
    } else if row >= i32(sim_params.active_rows) {
        return grid_in[get_cell_index(i32(sim_params.active_rows - 1), col)].bx;
    }

    return grid_in[get_cell_index(row, col)].bx;
}

fn get_by(row: i32, col: i32) -> f32 {
    if row <= 0 || row >= i32(sim_params.active_rows) {
        return 0.0;
    }

    if col < 0 {
        return grid_in[get_cell_index(row, 0)].by;
    } else if col >= i32(sim_params.active_cols) {
        return grid_in[get_cell_index(row, i32(sim_params.active_cols - 1))].by;
    }

    return grid_in[get_cell_index(row, col)].by;
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

    let speed_mult = 100.0;

    let center_x = (f32(col) + 0.5) * delta;
    let center_y = (f32(row) + 0.5) * delta;

    let center_u = bilinear_interpolation_u(center_x, center_y);
    let center_v = bilinear_interpolation_v(center_x, center_y);

    let dye_old_x = center_x - center_u * dt * speed_mult;
    let dye_old_y = center_y - center_v * dt * speed_mult;

    cell.dye = bilinear_interpolation_dye(dye_old_x, dye_old_y) * 0.9999;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn fluid_divergence_step(@builtin(global_invocation_id) id: vec3<u32>) {
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

    let u_left = cell.u;
    let u_right = get_u(y, x + 1);

    let v_down = cell.v;
    var v_up = get_v(y + 1, x);

    let div_u = (u_right - u_left) / delta;
    let div_v = (v_up - v_down) / delta;

    cell.fluid_divergence = div_u + div_v;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn fluid_jacobi_step(@builtin(global_invocation_id) id: vec3<u32>) {
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

    let b = (sim_params.fluid_density * delta * delta * cell.fluid_divergence) / sim_params.dt;

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

    let x = i32(col);
    let y = i32(row);
    let index = get_cell_index(i32(row), i32(col));
    var cell = grid_in[index];
    let delta = f32(sim_params.cell_size);
    let dt = sim_params.dt;

    let u_right = get_u(y, x + 1);
    let u_up = get_u(y + 1, x);
    let v_right = get_v(y, x + 1);
    var v_up = get_v(y + 1, x);

    // stretching term x-component

    let bx_x = f32(col) * delta;
    let bx_y = (f32(row) + 0.5) * delta;

    var local_bx = cell.bx;
    var local_by = bilinear_interpolation_by(bx_x, bx_y);

    let du_dx = (u_right - cell.u) / delta;
    let du_dy = (u_up - cell.u) / delta;

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

    let dv_dx = (v_right - cell.v) / delta;
    let dv_dy = (v_up - cell.v) / delta;

    let b_stretch_y = (local_bx * dv_dx + local_by * dv_dy) * dt;

    // advection term y-component

    local_u = bilinear_interpolation_u(by_x, by_y);
    local_v = cell.v;

    let by_old_x = by_x - local_u * dt;
    let by_old_y = by_y - local_v * dt;

    cell.by = bilinear_interpolation_by(by_old_x, by_old_y) + b_stretch_y;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn magnetic_divergence_step(@builtin(global_invocation_id) id: vec3<u32>) {
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

    let bx_left = cell.bx;
    var bx_right = get_bx(y, x + 1);

    let by_down = cell.by;
    let by_up = get_by(y + 1, x);

    let div_bx = (bx_right - bx_left) / delta;
    let div_by = (by_up - by_down) / delta;

    cell.magnetic_divergence = div_bx + div_by;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn magnetic_jacobi_step(@builtin(global_invocation_id) id: vec3<u32>) {
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
    var phi_left = grid_in[get_cell_index(y, x - 1)].phi;
    var phi_right = grid_in[get_cell_index(y, x + 1)].phi;
    var phi_down = grid_in[get_cell_index(y - 1, x)].phi;
    var phi_up = grid_in[get_cell_index(y + 1, x)].phi;

    let b = delta * delta * cell.magnetic_divergence;

    let phi_new = (phi_left + phi_right + phi_down + phi_up - b) / 4.0;

    cell.phi = phi_new;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn electric_potential_gradient_step(@builtin(global_invocation_id) id: vec3<u32>) {
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
    var phi_left = grid_in[get_cell_index(y, x - 1)].phi;
    var phi_down = grid_in[get_cell_index(y - 1, x)].phi;

    let gradient_phi = vec2<f32>(
        (cell.phi - phi_left) / delta,
        (cell.phi - phi_down) / delta
    );

    cell.bx -= gradient_phi.x;
    cell.by -= gradient_phi.y;

    if x == 0 {
        cell.bx = 0.0;
    }
    if y == 0 {
        cell.by = 0.0;
    }

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn current_density_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= sim_params.active_cols || row >= sim_params.active_rows {
        return;
    }

    let index = get_cell_index(i32(row), i32(col));
    var cell = grid_in[index];
    let delta = f32(sim_params.cell_size);
    let dt = sim_params.dt;

    var by_x = (f32(col) + 1.0) * delta;
    var by_y = (f32(row) + 0.5) * delta;

    let by_right = bilinear_interpolation_by(by_x, by_y);

    by_x = f32(col) * delta;

    let by_left = bilinear_interpolation_by(by_x, by_y);

    var bx_x = (f32(col) + 0.5) * delta;
    var bx_y = (f32(row) + 1.0) * delta;

    let bx_up = bilinear_interpolation_bx(bx_x, bx_y);

    bx_y = f32(row) * delta;

    let bx_down = bilinear_interpolation_bx(bx_x, bx_y);

    cell.current_density = ((by_right - by_left) - (bx_up - bx_down)) / delta;

    grid_out[index] = cell;
}

@compute
@workgroup_size(8, 8, 1)
fn lorentz_force_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;

    if col >= sim_params.active_cols || row >= sim_params.active_rows {
        return;
    }

    let index = get_cell_index(i32(row), i32(col));
    var cell = grid_in[index];
    let current_density_left = grid_in[get_cell_index(i32(row), i32(col - 1))].current_density;
    let current_density_down = grid_in[get_cell_index(i32(row - 1), i32(col))].current_density;
    let delta = f32(sim_params.cell_size);
    let dt = sim_params.dt;

    let j_x = (cell.current_density + current_density_left) / 2.0;
    let j_y = (cell.current_density + current_density_down) / 2.0;

    var x = f32(col) * delta;
    var y = (f32(row) + 0.5) * delta;

    let local_by = bilinear_interpolation_by(x, y);

    x = (f32(col) + 0.5) * delta;
    y = f32(row) * delta;

    let local_bx = bilinear_interpolation_bx(x, y);

    let lorentz_force = vec2<f32>(
        -j_x * local_by,
        j_y * local_bx
    ) / sim_params.fluid_density * dt;

    cell.u += lorentz_force.x;
    cell.v += lorentz_force.y;

    grid_out[index] = cell;
}

@compute @workgroup_size(8, 8, 1)
fn user_interaction_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let col = id.x;
    let row = id.y;
    if col >= sim_params.active_cols || row >= sim_params.active_rows { return; }

    let index = get_cell_index(i32(row), i32(col));
    var cell = grid_in[index];

    let cell_x = f32(col * sim_params.cell_size);
    let cell_y = f32(row * sim_params.cell_size);

    let dx = cell_x - sim_params.mouse_x;
    let dy = cell_y - sim_params.mouse_y;
    let dist = max(sqrt(dx * dx + dy * dy), 0.0001);

    if sim_params.mouse_left_clicked == 1u && dist < 30.0 {
        let force = 100.0 * (1.0 - dist / 30.0);
        cell.u += force;

        cell.dye += 60.0 * (1.0 - dist / 30.0);
    }

    if sim_params.mouse_right_clicked == 1u && dist < 50.0 {
        let b_force = 100.0 * (1.0 - dist / 50.0);
        cell.by += b_force;
    }

    grid_out[index] = cell;
}

@compute @workgroup_size(64, 1, 1)
fn particle_update_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let particle_index = id.x;
    if particle_index >= arrayLength(&particles) { return; }

    var particle = particles[particle_index];

    let u = bilinear_interpolation_u(particle.x, particle.y);
    let v = bilinear_interpolation_v(particle.x, particle.y);

    let speed_mult = 50.0;

    particle.x += u * sim_params.dt * speed_mult;
    particle.y += v * sim_params.dt * speed_mult;

    if particle.x < 0.0 {
        particle.x += f32(sim_params.active_cols * sim_params.cell_size);
    } else if particle.x >= f32(sim_params.active_cols * sim_params.cell_size) {
        particle.x -= f32(sim_params.active_cols * sim_params.cell_size);
    }

    if particle.y < 0.0 {
        particle.y += f32(sim_params.active_rows * sim_params.cell_size);
    } else if particle.y >= f32(sim_params.active_rows * sim_params.cell_size) {
        particle.y -= f32(sim_params.active_rows * sim_params.cell_size);
    }

    particles[particle_index] = particle;
}
