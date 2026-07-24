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
    _padding: array<f32, 3>,
}

@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read> grid: array<Cell>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

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

    let u00 = grid[get_cell_index(row, col)].u;
    let u10 = grid[get_cell_index(row, col + 1)].u;
    let u01 = grid[get_cell_index(row + 1, col)].u;
    let u11 = grid[get_cell_index(row + 1, col + 1)].u;

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

    let v00 = grid[get_cell_index(row, col)].v;
    let v10 = grid[get_cell_index(row, col + 1)].v;
    let v01 = grid[get_cell_index(row + 1, col)].v;
    let v11 = grid[get_cell_index(row + 1, col + 1)].v;

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

    let bx00 = grid[get_cell_index(row, col)].bx;
    let bx10 = grid[get_cell_index(row, col + 1)].bx;
    let bx01 = grid[get_cell_index(row + 1, col)].bx;
    let bx11 = grid[get_cell_index(row + 1, col + 1)].bx;

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

    let by00 = grid[get_cell_index(row, col)].by;
    let by10 = grid[get_cell_index(row, col + 1)].by;
    let by01 = grid[get_cell_index(row + 1, col)].by;
    let by11 = grid[get_cell_index(row + 1, col + 1)].by;

    let mix_bottom = mix(by00, by10, tx);
    let mix_top = mix(by01, by11, tx);
    return mix(mix_bottom, mix_top, ty);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);

    output.clip_position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);

    return output;
}

@fragment
fn fluid_background(input: VertexOutput) -> @location(0) vec4<f32> {
    let col = clamp(u32(input.uv.x * f32(sim_params.active_cols)), 0u, sim_params.active_cols - 1u);
    let row = clamp(u32(input.uv.y * f32(sim_params.active_rows)), 0u, sim_params.active_rows - 1u);

    let index = row * sim_params.active_cols + col;
    let cell = grid[index];
    let delta = f32(sim_params.cell_size);

    // velocity curl

    var x = f32(col) * delta;
    var y = (f32(row) + 0.5) * delta;

    let v_left = bilinear_interpolation_v(x, y);
    x += delta;
    let v_right = bilinear_interpolation_v(x, y);

    x = (f32(col) + 0.5) * delta;
    y = f32(row) * delta;

    let u_down = bilinear_interpolation_u(x, y);
    y += delta;
    let u_up = bilinear_interpolation_u(x, y);

    let curl = (v_right - v_left) / delta - (u_up - u_down) / delta;

    // color based on curl (dark blue palette)
    let color = vec3<f32>(0.0, 0.0, 0.2) + vec3<f32>(0.0, 0.0, 0.4) * clamp(curl * 10.0, -1.0, 1.0);

    return vec4<f32>(color, 1.0);
}

@fragment
fn magnetic_field_lines(input: VertexOutput) -> @location(0) vec4<f32> {
    let col_f = input.uv.x * f32(sim_params.active_cols);
    let row_f = input.uv.y * f32(sim_params.active_rows);

    let col = clamp(u32(col_f), 0u, sim_params.active_cols - 1u);
    let row = clamp(u32(row_f), 0u, sim_params.active_rows - 1u);

    let delta = f32(sim_params.cell_size);

    let x = (f32(col) + 0.5) * delta;
    let y = (f32(row) + 0.5) * delta;

    let bx = bilinear_interpolation_bx(x, y);
    let by = bilinear_interpolation_by(x, y);

    let b_vector = vec2<f32>(bx, by);
    let b_magnitude = length(b_vector);

    let epsilon = 1e-7;
    let b_direction = normalize(b_vector + vec2<f32>(epsilon, epsilon));

    let local_p = vec2<f32>(fract(col_f) - 0.5, fract(row_f) - 0.5);

    let proj_length = dot(local_p, b_direction);
    let proj_point = b_direction * proj_length;

    let dist_to_line = length(local_p - proj_point);

    let thickness = 0.06;
    let line_length_half = 0.35;
    let edge_softness = 0.03;

    let alpha_thickness = 1.0 - smoothstep(thickness - edge_softness, thickness + edge_softness, dist_to_line);
    let alpha_length = 1.0 - smoothstep(line_length_half - edge_softness, line_length_half + edge_softness, abs(proj_length));

    let shape_alpha = alpha_thickness * alpha_length;

    let max_b = 1.0;
    let intensity = smoothstep(0.0, max_b, b_magnitude);

    // royal purple accent
    let royal_purple = vec3<f32>(0.4, 0.1, 0.86);

    let final_color = royal_purple * intensity;
    let final_alpha = shape_alpha * intensity;

    return vec4<f32>(final_color, final_alpha);
}

@fragment
fn interaction_density_glow(input: VertexOutput) -> @location(0) vec4<f32> {
    let col = clamp(u32(input.uv.x * f32(sim_params.active_cols)), 0u, sim_params.active_cols - 1u);
    let row = clamp(u32(input.uv.y * f32(sim_params.active_rows)), 0u, sim_params.active_rows - 1u);

    let index = row * sim_params.active_cols + col;
    let cell = grid[index];

    // current glow
    let tension_glow = vec3<f32>(0.9, 0.95, 1.0);

    let max_current: f32 = 1.0;

    let normalized_j = saturate(abs(cell.current_density) / max_current);

    let t_snap = pow(normalized_j, 2.0);

    let color = tension_glow * t_snap;

    return vec4<f32>(color, t_snap);
}
