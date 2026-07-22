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
    _padding: [f32; 3],
}

struct ComputeResources {
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

    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,

    active_cols: u32,
    active_rows: u32,
}

struct RenderResources {
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
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
            _padding: [0; 2],
        };

        let grid = vec![
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
                _padding: [0.0; 3],
            };
            (active_cols * active_rows) as usize
        ];

        let advection_pipeline = create_compute_pipeline(
            &device,
            Some("Advection Compute Pipeline"),
            Some("fluid_advection_step"),
        );

        let induction_pipeline = create_compute_pipeline(
            &device,
            Some("Induction Compute Pipeline"),
            Some("magnetic_induction_step"),
        );

        let current_pipeline = create_compute_pipeline(
            &device,
            Some("Current Density Compute Pipeline"),
            Some("current_density_step"),
        );

        let lorentz_pipeline = create_compute_pipeline(
            &device,
            Some("Lorentz Force Compute Pipeline"),
            Some("lorentz_force_step"),
        );

        let fluid_divergence_pipeline = create_compute_pipeline(
            &device,
            Some("Fluid Divergence Compute Pipeline"),
            Some("fluid_divergence_step"),
        );

        let fluid_jacobi_pipeline = create_compute_pipeline(
            &device,
            Some("Fluid Jacobi Compute Pipeline"),
            Some("fluid_jacobi_step"),
        );

        let fluid_gradient_pipeline = create_compute_pipeline(
            &device,
            Some("Fluid Gradient Compute Pipeline"),
            Some("fluid_gradient_step"),
        );

        let mag_divergence_pipeline = create_compute_pipeline(
            &device,
            Some("Magnetic Divergence Compute Pipeline"),
            Some("magnetic_divergence_step"),
        );

        let mag_jacobi_pipeline = create_compute_pipeline(
            &device,
            Some("Magnetic Jacobi Compute Pipeline"),
            Some("magnetic_jacobi_step"),
        );

        let mag_gradient_pipeline = create_compute_pipeline(
            &device,
            Some("Magnetic Gradient Compute Pipeline"),
            Some("magnetic_gradient_step"),
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

        let bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group A"),
            layout: &advection_pipeline.get_bind_group_layout(0),
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
            ],
        });

        let bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group B"),
            layout: &advection_pipeline.get_bind_group_layout(0),
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
            ],
        });

        let render_pipeline = create_render_pipeline(&device);

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_pipeline.get_bind_group_layout(0),
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

        let compute = ComputeResources {
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

            bind_group_a,
            bind_group_b,

            active_cols,
            active_rows,
        };

        let render = RenderResources {
            render_pipeline,
            bind_group: render_bind_group,
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
        }
    }

    pub fn update(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
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

        is_a
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

            rpass.set_pipeline(&self.render.render_pipeline);
            rpass.set_bind_group(0, &self.render.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);

        Ok(())
    }
}
