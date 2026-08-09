use crate::config::{CELL_SIZE, DT, FLUID_DENSITY, VISCOSITY};
use crate::pipeline::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
    _padding: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    x: f32,
    y: f32,
    _padding: [f32; 2],
}

struct ComputeResources {
    interaction_pipeline: wgpu::ComputePipeline,
    advection_pipeline: wgpu::ComputePipeline,
    induction_pipeline: wgpu::ComputePipeline,
    current_pipeline: wgpu::ComputePipeline,
    lorentz_pipeline: wgpu::ComputePipeline,
    fluid_divergence_pipeline: wgpu::ComputePipeline,
    fluid_jacobi_pipeline: wgpu::ComputePipeline,
    fluid_gradient_pipeline: wgpu::ComputePipeline,
    mag_divergence_pipeline: wgpu::ComputePipeline,
    mag_jacobi_pipeline: wgpu::ComputePipeline,
    mag_gradient_pipeline: wgpu::ComputePipeline,
    particle_pipeline: wgpu::ComputePipeline,

    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,

    active_cols: u32,
    active_rows: u32,
}

struct RenderResources {
    render_pipeline: wgpu::RenderPipeline,
    particle_render_pipeline: wgpu::RenderPipeline,
    bind_group_reading_a: wgpu::BindGroup,
    bind_group_reading_b: wgpu::BindGroup,
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<winit::window::Window>,
    compute: ComputeResources,
    render: RenderResources,
    current_sim_params: SimParams,
    pub sim_params_buffer: wgpu::Buffer,
    pub particles_buffer: wgpu::Buffer,
    pub particle_count: u32,
}

impl State {
    pub async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
            flags: wgpu::InstanceFlags::all(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_limits: wgpu::Limits::default(),
                required_features: wgpu::Features::empty(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .expect("Failed to create device");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            color_space: wgpu::SurfaceColorSpace::Auto,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let active_cols = size.width / CELL_SIZE;
        let active_rows = size.height / CELL_SIZE;
        let sim_params = SimParams {
            dt: DT,
            fluid_density: FLUID_DENSITY,
            viscosity: VISCOSITY,
            active_cols,
            active_rows,
            cell_size: CELL_SIZE,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_left_clicked: 0,
            mouse_right_clicked: 0,
            _padding: [0; 2],
        };

        let mut grid = vec![
            Cell {
                u: 0.0,
                v: 0.0,
                bx: 0.0,
                by: 0.0,
                p: 1.0,
                fluid_divergence: 0.0,
                phi: 0.0,
                magnetic_divergence: 0.0,
                current_density: 0.0,
                dye: 0.0,
                _padding: [0.0; 2],
            };
            (active_cols * active_rows) as usize
        ];

        let particle_count = 15000;
        let mut particles = Vec::with_capacity(particle_count as usize);

        let mut seed = 1512u32;
        for _ in 0..particle_count {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let rand_x = (seed as f32) / (u32::MAX as f32);

            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let rand_y = (seed as f32) / (u32::MAX as f32);

            particles.push(Particle {
                x: rand_x * (active_cols * CELL_SIZE) as f32,
                y: rand_y * (active_rows * CELL_SIZE) as f32,
                _padding: [0.0; 2],
            });
        }

        let center_y = active_rows as f32 / 2.0;

        let center_x_left = active_cols as f32 * 0.35;
        let center_x_right = active_cols as f32 * 0.65;

        let vortex_radius = 35.0;
        let vortex_strength = 150.0;

        for row in 0..active_rows {
            for col in 0..active_cols {
                let index = (row * active_cols + col) as usize;

                grid[index].bx = 50.0;
                grid[index].by = 0.0;

                let dx_left = col as f32 - center_x_left;
                let dy_left = row as f32 - center_y;
                let dist_left = (dx_left * dx_left + dy_left * dy_left).sqrt();

                let dx_right = col as f32 - center_x_right;
                let dy_right = row as f32 - center_y;
                let dist_right = (dx_right * dx_right + dy_right * dy_right).sqrt();

                grid[index].u = 0.0;
                grid[index].v = 0.0;

                if dist_left < vortex_radius && dist_left > 1.0 {
                    let falloff = 1.0 - (dist_left / vortex_radius);
                    grid[index].u += (-dy_left / dist_left) * vortex_strength * falloff;
                    grid[index].v += (dx_left / dist_left) * vortex_strength * falloff;
                }

                if dist_right < vortex_radius && dist_right > 1.0 {
                    let falloff = 1.0 - (dist_right / vortex_radius);
                    grid[index].u += (dy_right / dist_right) * vortex_strength * falloff;
                    grid[index].v += (-dx_right / dist_right) * vortex_strength * falloff;
                }
            }
        }

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shared Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shared Compute Pipeline Layout"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });

        let interaction_pipeline = create_compute_pipeline(
            &device,
            Some("Interaction Compute Pipeline"),
            Some("user_interaction_step"),
            Some(&compute_pipeline_layout),
        );

        let advection_pipeline = create_compute_pipeline(
            &device,
            Some("Advection Compute Pipeline"),
            Some("fluid_advection_step"),
            Some(&compute_pipeline_layout),
        );

        let induction_pipeline = create_compute_pipeline(
            &device,
            Some("Induction Compute Pipeline"),
            Some("magnetic_induction_step"),
            Some(&compute_pipeline_layout),
        );

        let current_pipeline = create_compute_pipeline(
            &device,
            Some("Current Density Compute Pipeline"),
            Some("current_density_step"),
            Some(&compute_pipeline_layout),
        );

        let lorentz_pipeline = create_compute_pipeline(
            &device,
            Some("Lorentz Force Compute Pipeline"),
            Some("lorentz_force_step"),
            Some(&compute_pipeline_layout),
        );

        let fluid_divergence_pipeline = create_compute_pipeline(
            &device,
            Some("Fluid Divergence Compute Pipeline"),
            Some("fluid_divergence_step"),
            Some(&compute_pipeline_layout),
        );

        let fluid_jacobi_pipeline = create_compute_pipeline(
            &device,
            Some("Fluid Jacobi Compute Pipeline"),
            Some("fluid_jacobi_step"),
            Some(&compute_pipeline_layout),
        );

        let fluid_gradient_pipeline = create_compute_pipeline(
            &device,
            Some("Fluid Gradient Compute Pipeline"),
            Some("pressure_gradient_step"),
            Some(&compute_pipeline_layout),
        );

        let mag_divergence_pipeline = create_compute_pipeline(
            &device,
            Some("Magnetic Divergence Compute Pipeline"),
            Some("magnetic_divergence_step"),
            Some(&compute_pipeline_layout),
        );

        let mag_jacobi_pipeline = create_compute_pipeline(
            &device,
            Some("Magnetic Jacobi Compute Pipeline"),
            Some("magnetic_jacobi_step"),
            Some(&compute_pipeline_layout),
        );

        let mag_gradient_pipeline = create_compute_pipeline(
            &device,
            Some("Magnetic Gradient Compute Pipeline"),
            Some("electric_potential_gradient_step"),
            Some(&compute_pipeline_layout),
        );

        let particle_pipeline = create_compute_pipeline(
            &device,
            Some("Particle Compute Pipeline"),
            Some("particle_update_step"),
            Some(&compute_pipeline_layout),
        );

        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params Buffer"),
            contents: bytemuck::cast_slice(&[sim_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grid_in_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Buffer"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let grid_out_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Buffer"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::VERTEX,
        });

        let bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_in_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_in_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        let compute = ComputeResources {
            interaction_pipeline,
            advection_pipeline,
            induction_pipeline,
            current_pipeline,
            lorentz_pipeline,
            fluid_divergence_pipeline,
            fluid_jacobi_pipeline,
            fluid_gradient_pipeline,
            mag_divergence_pipeline,
            mag_jacobi_pipeline,
            mag_gradient_pipeline,
            particle_pipeline,

            bind_group_a,
            bind_group_b,

            active_cols,
            active_rows,
        };

        let render_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&render_bind_layout)],
                immediate_size: 0,
            });

        let render_pipeline = create_render_pipeline(
            &device,
            config.format,
            Some(&render_pipeline_layout),
            Some("vs_main"),
            Some("fs_main"),
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../res/render.wgsl").into()),
        });

        let particle_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Particle Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_particle"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x4],
                    })],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_particle"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let bind_group_reading_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group A"),
            layout: &render_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_in_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_reading_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group B"),
            layout: &render_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_out_buffer.as_entire_binding(),
                },
            ],
        });

        let render = RenderResources {
            render_pipeline,
            particle_render_pipeline,
            bind_group_reading_a,
            bind_group_reading_b,
        };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            compute,
            render,
            current_sim_params: sim_params,
            sim_params_buffer,
            particles_buffer: particle_buffer,
            particle_count,
        }
    }

    pub fn update(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        self.queue.write_buffer(
            &self.sim_params_buffer,
            0,
            bytemuck::cast_slice(&[self.current_sim_params]),
        );

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MHD Compute Pass"),
            timestamp_writes: None,
        });

        let workgroups_x = (self.compute.active_cols + 7) / 8;
        let workgroups_y = (self.compute.active_rows + 7) / 8;

        let mut is_a = true;

        let mut dispatch = |compute_pass: &mut wgpu::ComputePass,
                            pipeline: &wgpu::ComputePipeline,
                            is_a: &mut bool| {
            compute_pass.set_pipeline(pipeline);
            let bind_group = if *is_a {
                &self.compute.bind_group_a
            } else {
                &self.compute.bind_group_b
            };
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            *is_a = !*is_a;
        };

        dispatch(
            &mut compute_pass,
            &self.compute.interaction_pipeline,
            &mut is_a,
        );

        dispatch(
            &mut compute_pass,
            &self.compute.advection_pipeline,
            &mut is_a,
        );
        dispatch(
            &mut compute_pass,
            &self.compute.induction_pipeline,
            &mut is_a,
        );

        dispatch(&mut compute_pass, &self.compute.current_pipeline, &mut is_a);
        dispatch(&mut compute_pass, &self.compute.lorentz_pipeline, &mut is_a);

        dispatch(
            &mut compute_pass,
            &self.compute.fluid_divergence_pipeline,
            &mut is_a,
        );
        for _ in 0..40 {
            dispatch(
                &mut compute_pass,
                &self.compute.fluid_jacobi_pipeline,
                &mut is_a,
            );
        }
        dispatch(
            &mut compute_pass,
            &self.compute.fluid_gradient_pipeline,
            &mut is_a,
        );

        dispatch(
            &mut compute_pass,
            &self.compute.mag_divergence_pipeline,
            &mut is_a,
        );
        for _ in 0..40 {
            dispatch(
                &mut compute_pass,
                &self.compute.mag_jacobi_pipeline,
                &mut is_a,
            );
        }
        dispatch(
            &mut compute_pass,
            &self.compute.mag_gradient_pipeline,
            &mut is_a,
        );

        compute_pass.set_pipeline(&self.compute.particle_pipeline);
        let bind_group = if is_a {
            &self.compute.bind_group_a
        } else {
            &self.compute.bind_group_b
        };
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups((self.particle_count + 63) / 64, 1, 1);

        is_a
    }

    pub fn update_cursor_position(&mut self, x: f32, y: f32) {
        self.current_sim_params.mouse_x = x;
        self.current_sim_params.mouse_y = y;
    }

    pub fn update_mouse_click(&mut self, button: u32, is_pressed: bool) {
        let state_val = if is_pressed { 1 } else { 0 };

        if button == 0 {
            self.current_sim_params.mouse_left_clicked = state_val;
        } else if button == 1 {
            self.current_sim_params.mouse_right_clicked = state_val;
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                anyhow::bail!("Lost device");
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let final_buffer_b = self.update(&mut encoder);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let bind_group = if final_buffer_b {
                &self.render.bind_group_reading_a
            } else {
                &self.render.bind_group_reading_b
            };

            rpass.set_pipeline(&self.render.render_pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.draw(0..3, 0..1);

            rpass.set_pipeline(&self.render.particle_render_pipeline);
            rpass.set_vertex_buffer(0, self.particles_buffer.slice(..));
            rpass.draw(0..4, 0..self.particle_count);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);

        Ok(())
    }
}
