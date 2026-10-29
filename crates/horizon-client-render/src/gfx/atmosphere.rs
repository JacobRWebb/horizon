//! Ray-marched atmosphere: Horizon's own air, off by default. A fullscreen pass after the world
//! marches each pixel's ray to the terrain's depth through the dimension's gas
//! (`atmosphere.wgsl`): sun-lit ground fog and sun shafts in the Overworld, a lava-lit ash haze
//! with embers in the Nether, nebula gas in the End. It shares the clouds' 3D noise.
use glam::{DVec3, Mat4, Vec3, Vec4};

/// The camera's x and z wrap to this many blocks, a multiple of every scale the shader samples
/// the tiling noise at, so the noise does not jump and `f32` keeps its precision far out.
const WRAP: f64 = 12000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dimension {
    Overworld,
    Nether,
    End,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_view_proj: [[f32; 4]; 4],
    sun: [f32; 4],
    shaft: [f32; 4],
    cam: [f32; 4],
    fog: [f32; 4],
    misc: [f32; 4],
}

/// The player's settings for the look of the air (`graphics::Graphics`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// Percent of the default density.
    pub density: u32,
    /// Percent of the default sun shaft strength.
    pub shafts: u32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            density: 100,
            shafts: 100,
        }
    }
}

/// What a frame gives the pass.
pub struct Scene {
    /// The world's projection (with the TAA jitter) and view.
    pub view_proj: Mat4,
    pub camera: DVec3,
    /// Seconds of game time.
    pub time: f64,
    /// `Sky.sun_angle`, in degrees.
    pub sun_angle: f32,
    pub fog_color: [f32; 3],
    /// How far the ray reaches: the render distance, in blocks.
    pub distance: f32,
    pub dimension: Dimension,
    pub tuning: Tuning,
}

pub struct Atmosphere {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    noise: wgpu::TextureView,
    frame: u32,
}

impl Atmosphere {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        noise: wgpu::TextureView,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray-marched atmosphere"),
            source: wgpu::ShaderSource::Wgsl(include_str!("atmosphere.wgsl").into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ray-marched atmosphere"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(
                    1,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    2,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                ),
                entry(
                    3,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray-marched atmosphere"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ray-marched atmosphere"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_post"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_atmosphere"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Premultiplied over the world; the alpha (the emissive mark) stays.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ray-marched atmosphere"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Ray-marched atmosphere"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            uniform,
            sampler,
            noise,
            frame: 0,
        }
    }

    /// Draws the air over `target`, which `depth` (reversed, `Depth32Float`) belongs to.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        scene: &Scene,
    ) {
        self.frame = self.frame.wrapping_add(1);
        let angle = scene.sun_angle.to_radians();
        // The sun turns around the X axis; angle 0 is noon.
        let sun = Vec3::new(0.0, angle.cos(), -angle.sin());
        let clip = scene.view_proj * Vec4::new(sun.x, sun.y, sun.z, 0.0);
        let in_front = clip.w > 1.0e-4;
        let ndc = if in_front {
            [clip.x / clip.w, clip.y / clip.w]
        } else {
            [0.0, 0.0]
        };
        let daylight = ((sun.y + 0.05) / 0.25).clamp(0.0, 1.0);
        let mode = match scene.dimension {
            Dimension::Overworld => 0.0,
            Dimension::Nether => 1.0,
            Dimension::End => 2.0,
        };
        // Shafts belong to the Overworld's sun.
        let shafts = if scene.dimension == Dimension::Overworld {
            scene.tuning.shafts as f32 / 100.0
        } else {
            0.0
        };
        let params = Params {
            inv_view_proj: scene.view_proj.inverse().to_cols_array_2d(),
            sun: [sun.x, sun.y, sun.z, daylight],
            shaft: [ndc[0], ndc[1], f32::from(in_front), shafts],
            cam: [
                scene.camera.x.rem_euclid(WRAP) as f32,
                scene.camera.y as f32,
                scene.camera.z.rem_euclid(WRAP) as f32,
                (scene.time.rem_euclid(1.0e5)) as f32,
            ],
            fog: [
                scene.fog_color[0],
                scene.fog_color[1],
                scene.fog_color[2],
                scene.tuning.density as f32 / 100.0,
            ],
            misc: [
                mode,
                32.0,
                (self.frame % 64) as f32,
                scene.distance.clamp(64.0, 512.0),
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray-marched atmosphere"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.noise),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Ray-marched atmosphere"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("atmosphere.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn wrap_divides_the_shader_noise_scales() {
        // The scales atmosphere.wgsl hands `noise_at`, each divided by its position multiplier.
        for period in [150.0, 120.0, 300.0, 750.0, 600.0, 400.0, 240.0, 24.0] {
            assert_eq!(WRAP % period, 0.0, "{period}");
        }
    }
}
