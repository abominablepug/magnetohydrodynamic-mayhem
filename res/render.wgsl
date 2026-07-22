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
    fluid_divergence: f32,
    phi: f32,
    magnetic_divergence: f32,
    current_density: f32,
    _padding: array<f32, 3>,
}

@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let col = clamp(u32(input.uv.x * f32(sim_params.active_cols)), 0u, sim_params.active_cols - 1u);
    let row = clamp(u32(input.uv.y * f32(sim_params.active_rows)), 0u, sim_params.active_rows - 1u);

    let index = row * sim_params.active_cols + col;
    let cell = cells[index];

    let velocity_magnitude = length(vec2<f32>(cell.u, cell.v));

    let color = vec3<f32>(
        abs(cell.bx) * 10.0,
        velocity_magnitude * 0.1,
        abs(cell.by) * 10.0
    );

    return vec4<f32>(color, 1.0);
}
