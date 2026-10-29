//! Ray-marched clouds: Horizon's own cloud look, off by default. A fullscreen pass after the
//! world marches each pixel's ray through a thick cloud slab, stopping at the terrain's depth
//! (`raymarch_clouds.wgsl`). The vanilla cloud map (`clouds.png`) decides where there are clouds,
//! so they stay where vanilla puts them and move with its wind; a tiling 3D noise shapes them.
use crate::clouds;
use glam::{Mat4, Vec3};

/// Texels of the 3D noise along each axis.
const NOISE_SIZE: u32 = 64;
/// Vanilla's cloud height: the slab's bottom.
const BOTTOM: f32 = 192.33;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_view_proj: [[f32; 4]; 4],
    origin: [f32; 4],
    sun: [f32; 4],
    color: [f32; 4],
    fog: [f32; 4],
    slab: [f32; 4],
    misc: [f32; 4],
    tune: [f32; 4],
}

/// The player's settings for the look of the clouds (`graphics::Graphics`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// The slab's thickness in blocks (vanilla's cells are 4 thick).
    pub thickness: u32,
    /// Percent of the default density.
    pub density: u32,
    /// Percent of the default erosion by the noise: how ragged and billowy.
    pub detail: u32,
    /// Percent of the default sunlight.
    pub sun: u32,
    /// Percent of the default sky light.
    pub ambient: u32,
    /// The most steps along a ray.
    pub steps: u32,
    /// How far banks of cloud sit higher or lower than the cloud height, in blocks.
    pub variety: u32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            thickness: 18,
            density: 100,
            detail: 100,
            sun: 100,
            ambient: 100,
            steps: 128,
            variety: 6,
        }
    }
}

/// What a frame gives the pass.
pub struct Scene<'a> {
    pub input: &'a clouds::Input<'a>,
    /// The world's projection (with the TAA jitter) and view.
    pub view_proj: Mat4,
    /// `Sky.sun_angle`, in degrees.
    pub sun_angle: f32,
    pub fog_color: [f32; 3],
    /// `Sky.cloud_fog_end`.
    pub fog_end: f32,
    pub tuning: Tuning,
}

pub struct RaymarchClouds {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    noise: wgpu::TextureView,
    coverage: Option<((u32, u32), wgpu::TextureView)>,
    frame: u32,
}

/// The 3D noise the clouds and the atmosphere march through (`noise_texels`).
pub fn noise_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: NOISE_SIZE,
        height: NOISE_SIZE,
        depth_or_array_layers: NOISE_SIZE,
    };
    let noise = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Cloud noise"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        noise.as_image_copy(),
        &noise_texels(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(NOISE_SIZE * 4),
            rows_per_image: Some(NOISE_SIZE),
        },
        extent,
    );
    noise.create_view(&Default::default())
}

impl RaymarchClouds {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        noise: wgpu::TextureView,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray-marched clouds"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raymarch_clouds.wgsl").into()),
        });
        let texture_entry = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ray-marched clouds"),
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
                texture_entry(2, wgpu::TextureViewDimension::D2),
                texture_entry(3, wgpu::TextureViewDimension::D3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray-marched clouds"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ray-marched clouds"),
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
                entry_point: Some("fs_clouds"),
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
            label: Some("Ray-marched clouds"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Ray-marched clouds"),
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
            coverage: None,
            frame: 0,
        }
    }

    /// Draws the clouds over `target`, which `depth` (reversed, `Depth32Float`) belongs to.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        scene: &Scene<'_>,
    ) {
        let input = scene.input;
        let (w, h, cells) = input.texture;
        if self
            .coverage
            .as_ref()
            .is_none_or(|(size, _)| *size != (*w, *h))
        {
            let extent = wgpu::Extent3d {
                width: *w,
                height: *h,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Cloud map"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let texels: Vec<u8> = cells.iter().map(|&c| if c { 255 } else { 0 }).collect();
            queue.write_texture(
                texture.as_image_copy(),
                &texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(*w),
                    rows_per_image: Some(*h),
                },
                extent,
            );
            self.coverage = Some(((*w, *h), texture.create_view(&Default::default())));
        }
        let Some((_, coverage)) = &self.coverage else {
            return;
        };
        self.frame = self.frame.wrapping_add(1);
        let sun_angle = scene.sun_angle.to_radians();
        // The sun turns around the X axis; angle 0 is noon.
        let sun = Vec3::new(0.0, sun_angle.cos(), -sun_angle.sin());
        let [x, z] = input.map_origin();
        let drift = (input.time.rem_euclid(1 << 20) as f32 + input.partial_tick as f32) * 0.05;
        let fog_end = scene.fog_end.max(64.0);
        let params = Params {
            inv_view_proj: scene.view_proj.inverse().to_cols_array_2d(),
            origin: [x, input.camera.y as f32, z, drift],
            sun: [sun.x, sun.y, sun.z, ((sun.y + 0.05) / 0.25).clamp(0.0, 1.0)],
            color: input.color,
            fog: [
                scene.fog_color[0],
                scene.fog_color[1],
                scene.fog_color[2],
                fog_end,
            ],
            slab: [
                BOTTOM,
                1.0 / (*w as f32 * 12.0),
                1.0 / (*h as f32 * 12.0),
                scene.tuning.variety as f32,
            ],
            misc: [
                scene.tuning.thickness as f32,
                fog_end,
                scene.tuning.steps as f32,
                (self.frame % 64) as f32,
            ],
            tune: [
                scene.tuning.density as f32 / 100.0,
                scene.tuning.detail as f32 / 100.0,
                scene.tuning.sun as f32 / 100.0,
                scene.tuning.ambient as f32 / 100.0,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray-marched clouds"),
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
                    resource: wgpu::BindingResource::TextureView(coverage),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.noise),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Ray-marched clouds"),
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

fn hash(x: u32, y: u32, z: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(0x8da6_b343)
        ^ y.wrapping_mul(0xd816_3841)
        ^ z.wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x9e37_79b9);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}

/// Value noise that repeats every `period` lattice cells.
fn value_noise(p: [f32; 3], period: u32, seed: u32) -> f32 {
    let cell = p.map(f32::floor);
    let f = [0, 1, 2].map(|i| {
        let t = p[i] - cell[i];
        t * t * (3.0 - 2.0 * t)
    });
    let c = cell.map(|v| v as i64);
    let corner = |dx: i64, dy: i64, dz: i64| {
        let wrap = |v: i64| v.rem_euclid(i64::from(period)) as u32;
        hash(wrap(c[0] + dx), wrap(c[1] + dy), wrap(c[2] + dz), seed)
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x = |dy, dz| lerp(corner(0, dy, dz), corner(1, dy, dz), f[0]);
    let y = |dz| lerp(x(0, dz), x(1, dz), f[1]);
    lerp(y(0), y(1), f[2])
}

/// Cellular noise that repeats every `period` cells: 1 at a feature point, falling to 0 a cell away.
fn worley(p: [f32; 3], period: u32, seed: u32) -> f32 {
    let cell = p.map(f32::floor);
    let c = cell.map(|v| v as i64);
    let mut nearest = f32::MAX;
    for dz in -1..=1i64 {
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let wrap = |v: i64| v.rem_euclid(i64::from(period)) as u32;
                let (x, y, z) = (wrap(c[0] + dx), wrap(c[1] + dy), wrap(c[2] + dz));
                let point = [
                    (c[0] + dx) as f32 + hash(x, y, z, seed),
                    (c[1] + dy) as f32 + hash(x, y, z, seed + 101),
                    (c[2] + dz) as f32 + hash(x, y, z, seed + 202),
                ];
                let d: f32 = (0..3).map(|i| (point[i] - p[i]).powi(2)).sum();
                nearest = nearest.min(d);
            }
        }
    }
    (1.0 - nearest.sqrt()).clamp(0.0, 1.0)
}

/// The cloud noise, tiling over the texture: red is smooth fractal noise, green billowy cellular
/// noise for the clouds' shape and blue finer cellular noise for their edges.
fn noise_texels() -> Vec<u8> {
    let n = NOISE_SIZE;
    let mut out = Vec::with_capacity((n * n * n * 4) as usize);
    let octaves = |u: [f32; 3], noise: fn([f32; 3], u32, u32) -> f32, list: &[(u32, f32)], seed| {
        list.iter()
            .enumerate()
            .map(|(i, &(freq, weight))| {
                weight * noise(u.map(|v| v * freq as f32), freq, seed + i as u32)
            })
            .sum::<f32>()
    };
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let u = [x, y, z].map(|v| v as f32 / n as f32);
                let smooth = octaves(u, value_noise, &[(4, 0.5), (8, 0.3), (16, 0.2)], 1);
                let billow = octaves(u, worley, &[(3, 0.6), (6, 0.28), (12, 0.12)], 10);
                let fine = octaves(u, worley, &[(8, 0.6), (16, 0.4)], 20);
                out.extend([byte(smooth), byte(billow), byte(fine), 255]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("raymarch_clouds.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn noise_tiles_and_stays_in_range() {
        let a = value_noise([0.3, 1.7, 2.2], 4, 1);
        let b = value_noise([4.3, 1.7, 6.2], 4, 1);
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
        let texels = noise_texels();
        assert_eq!(texels.len(), NOISE_SIZE.pow(3) as usize * 4);
        for channel in 0..3 {
            let values: Vec<f32> = texels
                .iter()
                .skip(channel)
                .step_by(4)
                .map(|&t| f32::from(t))
                .collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            assert!(
                (30.0..226.0).contains(&mean),
                "channel {channel} mean {mean}"
            );
            assert!(
                values.iter().any(|&v| v < 100.0) && values.iter().any(|&v| v > 156.0),
                "channel {channel} is flat"
            );
        }
    }
}
