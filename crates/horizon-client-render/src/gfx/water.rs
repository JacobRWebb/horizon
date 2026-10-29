//! Native water: Horizon's own look for the surface of water, off by default. The world pass is
//! split before translucent terrain, a copy of the frame and its depth is taken, and the water's
//! top faces are drawn by `water.wgsl` (appended to `terrain.wgsl`): colour from how deep the
//! water is, refraction of the copy, animated wave normals, a Fresnel sky reflection, a sun glint
//! and foam where the water meets terrain. Other translucent terrain draws as vanilla.
use crate::mesher::Vertex;
use glam::{DVec3, Mat4};

/// How many waves `water.wgsl` sums.
const WAVES: usize = 8;
/// Direction (radians) and wavelength (blocks) of each wave.
const WAVE_TABLE: [(f64, f64); WAVES] = [
    (0.30, 23.0),
    (1.70, 14.0),
    (2.60, 9.5),
    (4.00, 6.3),
    (5.20, 4.1),
    (0.90, 2.9),
    (3.30, 2.1),
    (5.90, 1.5),
];
/// The foam noise repeats every this many cells.
const FOAM_PERIOD: f64 = 256.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_view_proj: [[f32; 4]; 4],
    layers: [u32; 4],
    screen: [f32; 4],
    misc: [f32; 4],
    foam_offset: [f32; 4],
    look_a: [f32; 4],
    look_b: [f32; 4],
    look_c: [f32; 4],
    under: [f32; 4],
    waves: [[f32; 4]; WAVES],
}

/// The player's settings for the look of the water (`graphics::Graphics`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// Blocks of water that take the clear colour to the deep one.
    pub clarity: u32,
    /// Percent of the default darkness of deep water.
    pub depth_color: u32,
    /// Percent of the default wave height.
    pub wave_height: u32,
    /// Percent of the default wavelength.
    pub wave_size: u32,
    /// Percent of the default wave speed.
    pub wave_speed: u32,
    /// Percent of the default strength of the sky's reflection.
    pub reflection: u32,
    /// Percent of the default sun glint.
    pub glint: u32,
    /// Percent of the default refraction.
    pub refraction: u32,
    /// Percent of the default foam.
    pub foam: u32,
    /// Foam width, in tenths of a block.
    pub foam_width: u32,
    /// Steps a reflected ray marches through the frame (0: the sky only).
    pub reflection_steps: u32,
    /// How far a reflected ray looks, in blocks.
    pub reflection_distance: u32,
    /// Percent of the default underwater look (0: vanilla's).
    pub underwater: u32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            clarity: 14,
            depth_color: 100,
            wave_height: 100,
            wave_size: 100,
            wave_speed: 100,
            reflection: 100,
            glint: 100,
            refraction: 100,
            foam: 100,
            foam_width: 16,
            reflection_steps: 48,
            reflection_distance: 64,
            underwater: 100,
        }
    }
}

/// What a frame gives the pass.
pub struct Scene {
    /// The world's projection (with the TAA jitter) and view.
    pub view_proj: Mat4,
    pub camera: DVec3,
    /// Seconds of game time: the waves move with it.
    pub time: f64,
    /// The sprite layers of `water_still` and `water_flow`.
    pub layers: [u32; 2],
    pub tuning: Tuning,
    /// While the camera is under water: the colour of the water far away and the strength of
    /// the underwater look; zero otherwise.
    pub under: [f32; 4],
}

struct Snapshot {
    size: [u32; 2],
    format: wgpu::TextureFormat,
    color: wgpu::Texture,
    depth: wgpu::Texture,
    /// The player's own model, mirrored in the water and drawn from the real camera: where a
    /// pixel of it lands is where the water there shows the reflection.
    own_color: wgpu::TextureView,
    own_depth: wgpu::TextureView,
    group: wgpu::BindGroup,
}

pub struct Water {
    pipeline: wgpu::RenderPipeline,
    flat_pipeline: Option<wgpu::RenderPipeline>,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    snapshot: Option<Snapshot>,
    /// Stands in at group 1 for the draw bindings that only flat draws have.
    empty: wgpu::BindGroup,
}

/// terrain.wgsl with the water's entry points after it.
pub fn source() -> String {
    format!(
        "{}\n{}",
        include_str!("../terrain.wgsl"),
        include_str!("water.wgsl")
    )
}

impl Water {
    /// `terrain` is terrain.wgsl's bindings (group 0); `flat` the flat draws' (group 1);
    /// `sprites` its pipeline constants.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        terrain: &wgpu::BindGroupLayout,
        flat: Option<&wgpu::BindGroupLayout>,
        sprites: [f64; 2],
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        flat_buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Native water"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Native water"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Native water, no draw bindings"),
            entries: &[],
        });
        let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Native water, no draw bindings"),
            layout: &empty_layout,
            entries: &[],
        });
        let constants = [("SPRITE_ARRAYS", sprites[0]), ("SPRITE_SPLIT", sprites[1])];
        let pipeline = |layouts: [Option<&wgpu::BindGroupLayout>; 3],
                        vertex: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Native water"),
                bind_group_layouts: &layouts,
                ..Default::default()
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Native water"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_water"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            pipeline: pipeline(
                [Some(terrain), Some(&empty_layout), Some(&layout)],
                "vs_water",
                buffers,
            ),
            flat_pipeline: flat.map(|flat| {
                pipeline(
                    [Some(terrain), Some(flat), Some(&layout)],
                    "vs_water_flat",
                    flat_buffers,
                )
            }),
            layout,
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Native water"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Native water"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            snapshot: None,
            empty,
        }
    }

    /// The pipeline for flat draws (`flat`) or instanced ones.
    pub fn pipeline(&self, flat: bool) -> Option<&wgpu::RenderPipeline> {
        if flat {
            self.flat_pipeline.as_ref()
        } else {
            Some(&self.pipeline)
        }
    }

    /// Binds the snapshot for the draws that follow; `flat` draws bind group 1 themselves.
    pub fn bind(&self, pass: &mut wgpu::RenderPass<'_>, flat: bool) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        if !flat {
            pass.set_bind_group(1, &self.empty, &[]);
        }
        pass.set_bind_group(2, &snapshot.group, &[]);
    }

    /// Copies the frame and its depth so far, and writes this frame's parameters. The textures
    /// need `COPY_SRC`.
    pub fn capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        depth: &wgpu::Texture,
        scene: &Scene,
    ) {
        let size = [color.width(), color.height()];
        let format = color.format();
        if self
            .snapshot
            .as_ref()
            .is_none_or(|s| s.size != size || s.format != format)
        {
            let make = |label, format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            let copy_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
            let own_usage =
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT;
            let color_copy = make("Water scene colour", format, copy_usage);
            let depth_copy = make(
                "Water scene depth",
                wgpu::TextureFormat::Depth32Float,
                copy_usage,
            );
            let own_color = make("Water mirrored player colour", format, own_usage);
            let own_depth = make(
                "Water mirrored player depth",
                wgpu::TextureFormat::Depth32Float,
                own_usage,
            );
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Native water"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_copy.create_view(
                            &wgpu::TextureViewDescriptor {
                                aspect: wgpu::TextureAspect::DepthOnly,
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &color_copy.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&own_depth.create_view(
                            &wgpu::TextureViewDescriptor {
                                aspect: wgpu::TextureAspect::DepthOnly,
                                ..Default::default()
                            },
                        )),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(
                            &own_color.create_view(&Default::default()),
                        ),
                    },
                ],
            });
            self.snapshot = Some(Snapshot {
                size,
                format,
                color: color_copy,
                depth: depth_copy,
                own_color: own_color.create_view(&Default::default()),
                own_depth: own_depth.create_view(&Default::default()),
                group,
            });
        }
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_texture(
            color.as_image_copy(),
            snapshot.color.as_image_copy(),
            extent,
        );
        encoder.copy_texture_to_texture(
            depth.as_image_copy(),
            snapshot.depth.as_image_copy(),
            extent,
        );
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&self.params(scene, size)),
        );
    }

    /// The attachments the player's mirrored model is drawn into (colour, depth), once a frame
    /// has been captured.
    pub fn own_attachments(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        let s = self.snapshot.as_ref()?;
        Some((&s.own_color, &s.own_depth))
    }

    fn params(&self, scene: &Scene, size: [u32; 2]) -> Params {
        let t = scene.tuning;
        let time = scene.time;
        let speed = f64::from(t.wave_speed) / 100.0;
        let scale = f64::from(t.wave_size).max(10.0) / 100.0;
        let mut waves = [[0.0; 4]; WAVES];
        for (wave, (angle, length)) in waves.iter_mut().zip(WAVE_TABLE) {
            let k = std::f64::consts::TAU / (length * scale);
            let (kx, kz) = (k * angle.cos(), k * angle.sin());
            // The wave travels as deep water waves do, faster the longer it is.
            let omega = 0.9 * speed * (k * 9.81).sqrt() * 0.5;
            // What the camera's position and the wave's travel add to the phase, exactly.
            let phase = (kx * scene.camera.x + kz * scene.camera.z + omega * time)
                .rem_euclid(std::f64::consts::TAU);
            *wave = [kx as f32, kz as f32, omega as f32, phase as f32];
        }
        let foam = |scale: f64, v: f64| (v * scale).rem_euclid(FOAM_PERIOD) as f32;
        Params {
            inv_view_proj: scene.view_proj.inverse().to_cols_array_2d(),
            layers: [scene.layers[0], scene.layers[1], 0, 0],
            screen: [size[0] as f32, size[1] as f32, 0.0, 0.0],
            misc: [(time % 3600.0) as f32, 0.0, foam(2.0, scene.camera.y), 0.0],
            foam_offset: [
                foam(2.0, scene.camera.x),
                foam(2.0, scene.camera.z),
                foam(5.0, scene.camera.x),
                foam(5.0, scene.camera.z),
            ],
            look_a: [
                t.clarity as f32,
                t.depth_color as f32 / 100.0,
                t.wave_height as f32 / 100.0,
                t.refraction as f32 / 100.0,
            ],
            look_b: [
                t.reflection as f32 / 100.0,
                t.glint as f32 / 100.0,
                t.foam as f32 / 100.0,
                t.foam_width as f32 / 10.0,
            ],
            look_c: [
                t.reflection_steps as f32,
                t.reflection_distance as f32,
                0.0,
                0.0,
            ],
            under: scene.under,
            waves,
        }
    }
}

/// `vertices` (camera-relative quads) turned over in the plane `plane` blocks above the camera:
/// the model's reflection, wound the right way round again and lit from the other side.
pub fn mirror(vertices: &[Vertex], plane: f32, out: &mut Vec<Vertex>) {
    out.clear();
    out.reserve(vertices.len());
    for quad in vertices.as_chunks::<4>().0 {
        let mut turned = *quad;
        for v in &mut turned {
            v.pos[1] = 2.0 * plane - v.pos[1];
            v.normal[1] = v.normal[1].saturating_neg();
        }
        // Mirroring reverses a quad's winding; swapping two corners restores it.
        turned.swap(1, 3);
        out.extend_from_slice(&turned);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirroring_flips_height_and_keeps_the_winding() {
        let v = |x: f32, y: f32| Vertex {
            pos: [x, y, 0.0],
            uv: [0.0; 2],
            layer: 0,
            color: 0,
            light: 0,
            normal: [0, 127, 0, 0],
            glint_uv: [0.0; 2],
        };
        let quad = [v(0.0, 1.0), v(1.0, 1.0), v(1.0, 2.0), v(0.0, 2.0)];
        let mut out = Vec::new();
        mirror(&quad, -1.0, &mut out);
        // y -> -2 - y; corners 1 and 3 swapped.
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].pos, [0.0, -3.0, 0.0]);
        assert_eq!(out[1].pos, [0.0, -4.0, 0.0]);
        assert_eq!(out[2].pos, [1.0, -4.0, 0.0]);
        assert_eq!(out[3].pos, [1.0, -3.0, 0.0]);
        assert_eq!(out[0].normal[1], -127);
    }

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(&source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn params_match_the_shader_layout() {
        // 4 x 4 matrix, eight vec4s, eight waves.
        assert_eq!(
            std::mem::size_of::<Params>(),
            64 + 16 * 6 + 16 + 16 + 16 * WAVES
        );
    }
}
