// Ray-marched clouds (`raymarch_clouds.rs`): a fullscreen pass after the world.
// Each pixel marches a ray through the cloud slab, stopping at the terrain's
// depth, and blends the cloud's light over the frame.
struct Params {
    // Clip space (with the TAA jitter) to camera-relative world.
    inv_view_proj: mat4x4<f32>,
    // x, z: the camera in the cloud map's space (blocks, wrapped, with the wind);
    // y: the camera's height; w: the shape's slow drift.
    origin: vec4<f32>,
    // xyz: towards the sun; w: how far above the horizon it stands (0..1).
    sun: vec4<f32>,
    color: vec4<f32>,
    // rgb: the fog's colour; w: where cloud fog ends, in blocks.
    fog: vec4<f32>,
    // x: the slab's bottom height; yz: 1 / the map's size in blocks; w: the height variety.
    slab: vec4<f32>,
    // x: the slab's thickness; y: the longest ray; z: the most steps; w: the frame count.
    misc: vec4<f32>,
    // x: density; y: how much the noise erodes the shape; z: sunlight; w: sky light. 1 is default.
    tune: vec4<f32>,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var coverage: texture_2d<f32>;
// r: smooth fractal noise; g: billowy cellular noise; b: fine cellular noise.
@group(0) @binding(3) var noise: texture_3d<f32>;
@group(0) @binding(4) var samp: sampler;

// Extinction per block at full density.
const SIGMA: f32 = 0.2;
const MAX_STEPS: i32 = 256;

struct PostOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex fn vs_post(@builtin(vertex_index) i: u32) -> PostOut {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: PostOut;
    out.clip = vec4(q * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let h = p.inv_view_proj * vec4(ndc, z, 1.0);
    return h.xyz / h.w;
}

fn gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

fn noise_at(pos: vec3<f32>, scale: f32) -> vec3<f32> {
    return textureSampleLevel(noise, samp, pos / scale, 0.0).rgb;
}

// The slab reaches below the nominal cloud height for ragged bases, and by the height
// variety (`slab.w`) either way for banks that sit higher or lower.
fn margin() -> f32 {
    return p.misc.x * 0.45 + p.slab.w;
}

fn span() -> f32 {
    return p.misc.x + p.misc.x * 0.45 + 2.0 * p.slab.w;
}

// The cloud's density at `pos`; `detail` adds the fine erosion the light march skips.

fn density(pos: vec3<f32>, detail: bool) -> f32 {
    let m = margin();
    let y = pos.y - (p.slab.x - m);
    if y <= 0.0 || y >= span() {
        return 0.0;
    }
    let drift = vec3(p.origin.w, 0.0, p.origin.w * 0.6);
    // Warp the vanilla map so its round cells get ragged edges.
    let warp = noise_at(vec3(pos.x, 0.0, pos.z) + drift, 140.0).r - 0.5;
    let cover = textureSampleLevel(coverage, samp, (pos.xz + warp * 36.0) * p.slab.yz, 0.0).r;
    if cover <= 0.02 {
        return 0.0;
    }
    // Taller in the middle of a cloud than at its edge: flat bottoms, rising tops.
    // Each column has its own base height and top: no flat floor, tops rising in the middle.
    let base_m = p.misc.x * 0.45;
    let lift = (noise_at(vec3(pos.x, 0.0, pos.z) + drift * 0.3, 320.0).r - 0.5) * 4.0 * p.slab.w;
    let floor_y = base_m * (1.0 - noise_at(vec3(pos.x, 0.0, pos.z) + drift, 75.0).r * 1.2) + p.slab.w + lift;
    let top_y = m + p.misc.x * mix(0.3, 1.0, cover * cover) + lift;
    let h = (y - floor_y) / (top_y - floor_y);
    if h <= 0.0 || h >= 1.0 {
        return 0.0;
    }
    let profile = smoothstep(0.0, 0.3, h) * (1.0 - smoothstep(0.45, 1.0, h));
    let base = cover * profile;
    let shape = noise_at(pos + drift, 110.0).g;
    var erode = (1.0 - shape) * 0.72;
    if detail {
        let fine = noise_at(pos + drift * 2.0, 34.0).b;
        erode += (1.0 - fine) * 0.34 * (0.5 + 0.5 * h);
        // Wispier undersides.
        erode += (1.0 - h) * 0.12;
    }
    return clamp((base * 1.3 - erode * p.tune.y) * 1.8, 0.0, 1.0);
}

// Optical depth from `pos` towards the sun.
fn sun_depth(pos: vec3<f32>) -> f32 {
    var tau = 0.0;
    var reach = 0.0;
    for (var i = 0; i < 5; i++) {
        let len = 2.0 + f32(i) * 2.5;
        reach += len * 0.5;
        tau += density(pos + p.sun.xyz * reach, false) * len * SIGMA * p.tune.x;
        reach += len * 0.5;
    }
    return tau;
}

fn forward_phase(c: f32) -> f32 {
    let g = 0.62;
    return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * c, 1.5);
}

@fragment fn fs_clouds(in: PostOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(depth_tex));
    let uv = in.clip.xy / size;
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // The ray starts on the near plane: view bobbing moves the eye off the camera's position.
    let ro = unproject(ndc, 1.0);
    let rd = normalize(unproject(ndc, 0.01) - ro);
    // Reversed depth: 0 is the far plane, where nothing was drawn.
    let depth = textureLoad(depth_tex, vec2<i32>(in.clip.xy), 0);
    var t_scene = 1.0e9;
    if depth > 0.0 {
        t_scene = length(unproject(ndc, depth) - ro);
    }

    let bottom = p.slab.x - margin() - (p.origin.y + ro.y);
    let top = bottom + span();
    var t0 = 0.0;
    var t1 = 1.0e9;
    if abs(rd.y) < 1.0e-4 {
        if bottom > 0.0 || top < 0.0 {
            return vec4(0.0);
        }
    } else {
        let ta = bottom / rd.y;
        let tb = top / rd.y;
        t0 = max(0.0, min(ta, tb));
        t1 = max(ta, tb);
    }
    t1 = min(t1, min(t_scene, p.misc.y));
    if t1 <= t0 {
        return vec4(0.0);
    }

    let fog_end = p.fog.w;
    let phase = mix(1.0, forward_phase(dot(rd, p.sun.xyz)), 0.5);
    let sun_color = mix(vec3(1.0, 0.62, 0.38), vec3(1.0, 0.97, 0.9), smoothstep(0.0, 0.45, p.sun.y));
    let shadow_color = vec3(0.60, 0.68, 0.82);
    let sky_color = vec3(0.86, 0.91, 1.0);
    let origin = vec3(p.origin.x, p.origin.y, p.origin.z) + ro;
    let jitter = gradient_noise(in.clip.xy + p.misc.w * 5.588238);
    var t = t0 + jitter * clamp(t0 * 0.01, 1.0, 16.0);
    let max_steps = i32(p.misc.z);
    var trans = 1.0;
    var light = vec3(0.0);
    for (var i = 0; i < MAX_STEPS; i++) {
        if t >= t1 || i >= max_steps {
            break;
        }
        let dt = clamp(t * 0.01, 1.0, 16.0);
        let pos = origin + rd * t;
        let d = density(pos, true);
        if d > 0.004 {
            let fog = smoothstep(0.35 * fog_end, fog_end, t);
            let sigma = d * SIGMA * p.tune.x * (1.0 - fog);
            let h = clamp((pos.y - (p.slab.x - margin())) / span(), 0.0, 1.0);
            let tau = sun_depth(pos);
            // Direct light, with a little multiple scattering so deep clouds are not black.
            let direct = p.sun.w * p.tune.z * (exp(-tau) + 0.5 * exp(-tau * 0.2)) * phase;
            let ambient = mix(shadow_color, sky_color, smoothstep(0.0, 0.8, h));
            var lit = p.color.rgb * (ambient * 0.78 * p.tune.w + sun_color * direct * 0.7);
            // Keep the brightest light from clipping to a flat white.
            lit = lit / (1.0 + max(lit - vec3(0.85), vec3(0.0)));
            let step_trans = exp(-sigma * dt);
            light += trans * (1.0 - step_trans) * mix(lit, p.fog.rgb, fog);
            trans *= step_trans;
            if trans < 0.02 {
                break;
            }
            t += dt;
        } else {
            // Empty space: step further.
            t += dt * 1.25;
        }
    }
    return vec4(min(light, vec3(1.0)), 1.0 - trans) * p.color.a;
}
