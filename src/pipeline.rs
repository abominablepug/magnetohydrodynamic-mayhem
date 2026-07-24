pub fn create_render_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: Option<&wgpu::PipelineLayout>,
    vertex_shader_entry: Option<&str>,
    fragment_shader_entry: Option<&str>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../res/render.wgsl").into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: layout,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: vertex_shader_entry,
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: fragment_shader_entry,
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        cache: None,
        multiview_mask: None,
    })
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    label: Option<&str>,
    entry_point: Option<&str>,
    pipeline_layout: Option<&wgpu::PipelineLayout>,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../res/compute.wgsl").into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label,
        layout: pipeline_layout,
        module: &shader,
        entry_point,
        compilation_options: Default::default(),
        cache: None,
    })
}
