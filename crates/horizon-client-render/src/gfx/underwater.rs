//! The underwater look of native water: with the camera under the surface, two fullscreen passes
//! after the world (`underwater.wgsl`). One dims the frame by how much light survives the water
//! between the camera and each pixel, red first; the other adds what the water scatters back: its
//! own colour, shafts of sunlight and caustics on what faces up. The surface seen from below is
//! the water shader's (`water.wgsl`); this uses its colours.
use glam::{DVec3, Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_view_proj: [[f32; 4]; 4],
    camera: [f32; 4],
    sun: [f32; 4],
    body: [f32; 4],
    look: [f32; 4],
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
    /// The water fog colour the game computed (the biome's tint).
    pub water_fog: [f32; 3],
    /// `Tuning::clarity`, in blocks.
    pub clarity: u32,
    /// `Tuning::underwater`, in percent.
    pub strength: u32,
}

/// The sun's direction (it turns around the X axis; angle 0 is noon) and how day it is.
pub fn sun(sun_angle: f32) -> (Vec3, f32) {
    let angle = sun_angle.to_radians();
    let sun = Vec3::new(0.0, angle.cos(), -angle.sin());
    let t = ((sun.y + 0.1) / 0.35).clamp(0.0, 1.0);
    (sun, t * t * (3.0 - 2.0 * t))
}

/// The colour of the water seen far away: the biome's tint over a clear blue-green, as bright as
/// the day.
pub fn body_color(water_fog: [f32; 3], day: f32) -> [f32; 3] {
    let clear = [0.04, 0.26, 0.34];
    let light = 0.22 + 0.78 * day;
    [0, 1, 2].map(|i| (0.5 * clear[i] + 0.5 * (water_fog[i] * 1.5).min(0.6)) * light)
}

pub struct Underwater {
    transmit: wgpu::RenderPipeline,
    light: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
}

impl Underwater {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Underwater"),
            source: wgpu::ShaderSource::Wgsl(include_str!("underwater.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Underwater"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Underwater"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = |entry: &str, src: wgpu::BlendFactor, dst: wgpu::BlendFactor| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Underwater"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_underwater"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: src,
                                dst_factor: dst,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        // The alpha carries the emissive mark through to HDR output: keep it.
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            // The frame times what survives: the source colour is the factor.
            transmit: pipeline(
                "fs_transmit",
                wgpu::BlendFactor::Zero,
                wgpu::BlendFactor::Src,
            ),
            light: pipeline("fs_light", wgpu::BlendFactor::One, wgpu::BlendFactor::One),
            layout,
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Underwater"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Draws the underwater look over `target`, which `depth` (reversed, `Depth32Float`) belongs to.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        scene: &Scene,
    ) {
        let (sun, day) = sun(scene.sun_angle);
        let body = body_color(scene.water_fog, day);
        let period = 256.0;
        let params = Params {
            inv_view_proj: scene.view_proj.inverse().to_cols_array_2d(),
            camera: [
                scene.camera.x.rem_euclid(period) as f32,
                0.0,
                scene.camera.z.rem_euclid(period) as f32,
                (scene.time % 3600.0) as f32,
            ],
            sun: [sun.x, sun.y, sun.z, day],
            body: [body[0], body[1], body[2], scene.strength as f32 / 100.0],
            look: [scene.clarity as f32 * 1.8, 160.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Underwater"),
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
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Underwater"),
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
        pass.set_bind_group(0, &group, &[]);
        pass.set_pipeline(&self.transmit);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.light);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("underwater.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn it_is_darker_at_night_and_the_sun_is_up_at_noon() {
        let (noon, day) = sun(0.0);
        assert!(noon.y > 0.99 && day > 0.99);
        let (_, night) = sun(180.0);
        assert!(night < 0.01);
        let bright = body_color([0.2, 0.3, 0.6], 1.0);
        let dark = body_color([0.2, 0.3, 0.6], 0.0);
        assert!(bright.iter().zip(dark).all(|(b, d)| *b > d));
    }
}
