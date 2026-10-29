// Ray-marched atmosphere (`atmosphere.rs`): a fullscreen pass after the world. Each pixel
// marches a ray from the eye to the terrain's depth through the dimension's own air:
//   0 Overworld: ground fog lit by the sun, and sun shafts marched through the depth buffer;
//   1 Nether: ash haze glowing from the lava sea, with embers drifting up;
//   2 End: pale mist, and nebula gas marched through the void behind the sky.
struct Params {
    // Clip space (with the TAA jitter) to camera-relative world.
    inv_view_proj: mat4x4<f32>,
    // xyz: towards the sun; w: daylight (0..1).
    sun: vec4<f32>,
    // xy: the sun on screen in NDC; z: 1 when it is in front of the camera; w: shaft strength.
    shaft: vec4<f32>,
    // xyz: the camera (x and z wrapped to the noise's period); w: seconds.
    cam: vec4<f32>,
    // rgb: the fog's colour; w: density, 1 is default.
    fog: vec4<f32>,
    // x: 0 Overworld, 1 Nether, 2 End; y: most steps; z: the frame count; w: the longest ray.
    misc: vec4<f32>,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
// r: smooth fractal noise; g: billowy cellular noise; b: fine cellular noise.
@group(0) @binding(2) var noise: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;

const MAX_STEPS: i32 = 64;

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

// The scales below divide 12000, the period the camera is wrapped to.
fn noise_at(pos: vec3<f32>, scale: f32) -> vec3<f32> {
    return textureSampleLevel(noise, samp, pos / scale, 0.0).rgb;
}

fn hg_phase(c: f32, g: f32) -> f32 {
    return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * c, 1.5);
}

fn sun_color() -> vec3<f32> {
    return mix(vec3(1.0, 0.62, 0.38), vec3(1.0, 0.95, 0.85), smoothstep(0.0, 0.45, p.sun.y));
}

// Sun shafts: marches towards the sun on the screen, counting the open sky it sees.
// Terrain and leaves in the depth buffer cut the shafts.
fn shafts(pix: vec2<f32>, size: vec2<f32>, rd: vec3<f32>) -> f32 {
    if p.shaft.z < 0.5 || p.shaft.w <= 0.0 || p.sun.w <= 0.0 {
        return 0.0;
    }
    let facing = dot(rd, p.sun.xyz);
    if facing <= 0.0 {
        return 0.0;
    }
    let sun_px = vec2(p.shaft.x * 0.5 + 0.5, 0.5 - p.shaft.y * 0.5) * size;
    // A fixed share of the way there, so shafts reach the same distance at any resolution.
    var delta = (sun_px - pix) * 0.85;
    let reach = length(delta);
    let longest = size.y * 0.6;
    if reach > longest {
        delta = delta * (longest / reach);
    }
    let n = 28;
    let step = delta / f32(n);
    let jitter = gradient_noise(pix + p.misc.z * 5.588238);
    var s = pix + step * jitter;
    var open = 0.0;
    var weight = 1.0;
    var total = 0.0;
    for (var i = 0; i < n; i++) {
        s += step;
        let inside = all(s >= vec2(0.0)) && all(s < size);
        let d = textureLoad(depth_tex, vec2<i32>(clamp(s, vec2(0.0), size - 1.0)), 0);
        if inside && d <= 0.0 {
            open += weight;
        }
        total += weight;
        weight *= 0.965;
    }
    let f = smoothstep(0.1, 1.0, facing);
    return open / total * f * f * p.shaft.w * p.sun.w;
}

@fragment fn fs_atmosphere(in: PostOut) -> @location(0) vec4<f32> {
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
    let sky = depth <= 0.0;

    let mode = i32(p.misc.x);
    let origin = p.cam.xyz + ro;
    let time = p.cam.w;
    let density = p.fog.w;
    let max_steps = i32(p.misc.y);
    let jitter = gradient_noise(in.clip.xy + p.misc.z * 5.588238);

    var t0 = 0.0;
    var t1 = min(t_scene, p.misc.w);
    if mode == 2 && sky {
        // The nebula hangs in a shell far out in the void.
        t0 = 120.0;
        t1 = 720.0;
    }
    var light = vec3(0.0);
    var trans = 1.0;
    if t1 > t0 {
        let min_dt = (t1 - t0) / (f32(max_steps) * 0.9);
        var t = t0 + jitter * min_dt;
        for (var i = 0; i < MAX_STEPS; i++) {
            if t >= t1 || i >= max_steps {
                break;
            }
            let pos = origin + rd * t;
            var dt = min_dt;
            if mode == 1 {
                // Finer steps close up, where the embers are.
                dt = max(min(t * 0.05, 3.0), min_dt * 0.35);
            }
            if mode == 0 {
                // Ground fog: thickest in valleys and over water, thinning into the sky,
                // drifting with the wind.
                let h = max(pos.y - 63.0, 0.0);
                let wisp = noise_at(pos + vec3(time * 1.2, 0.0, time * 0.7), 120.0).r;
                let bank = noise_at(pos * 0.4 + vec3(time * 0.4, 0.0, 0.0), 300.0).r;
                let d = exp(-h / 22.0) * (0.25 + 1.5 * wisp) * (0.5 + bank);
                let sigma = d * 0.0065 * density;
                if sigma > 1.0e-5 {
                    let phase = mix(1.0, hg_phase(dot(rd, p.sun.xyz), 0.55), 0.6);
                    let lit = p.fog.rgb * 0.62 + sun_color() * p.sun.w * phase * 0.55;
                    let step_trans = exp(-sigma * dt);
                    light += trans * (1.0 - step_trans) * lit;
                    trans *= step_trans;
                }
            } else if mode == 1 {
                // Nether: ash haze, glowing towards the lava sea at y 31 and the ceiling's
                // lava falls.
                let drift = vec3(time * 1.6, time * 0.9, time * 0.5);
                let wisp = noise_at(pos + drift, 150.0).r;
                let cloud = noise_at(pos * 0.5 + drift * 0.5, 300.0).g;
                let low = exp(-max(pos.y - 31.0, 0.0) / 26.0);
                let d = (0.2 + 1.3 * wisp) * (0.4 + 0.9 * cloud) * (0.35 + 0.65 * low);
                let sigma = d * 0.011 * density;
                let heat = exp(-max(pos.y - 31.0, 0.0) / 11.0);
                let glow = vec3(1.0, 0.3, 0.06) * (0.1 + 2.2 * heat * wisp);
                let lit = p.fog.rgb * 0.9 + glow * 0.5;
                let step_trans = exp(-sigma * dt);
                light += trans * (1.0 - step_trans) * lit;
                // The glow itself, which the haze scatters towards the eye.
                light += trans * glow * sigma * dt * 0.8;
                trans *= step_trans;
                // Embers: sparks rising on the heat, near the eye.
                if t < 56.0 {
                    let rise = vec3(sin(pos.y * 0.7 + time) * 0.6, -time * 2.4, cos(pos.y * 0.5 + time * 0.8) * 0.6);
                    let e = noise_at(pos + rise, 24.0).b;
                    let spark = smoothstep(0.82, 0.96, e);
                    let flicker = 0.55 + 0.45 * sin(time * 9.0 + pos.x * 3.1 + pos.z * 1.7);
                    let fade = 1.0 - smoothstep(24.0, 56.0, t);
                    light += trans * vec3(1.0, 0.5, 0.12) * spark * flicker * fade * min(dt, 1.5) * 2.6 * density;
                }
            } else {
                if sky {
                    // Nebula: gas layered in the void, fixed to the sky as the eye turns.
                    let q = rd * t + vec3(time * 0.8, 0.0, time * 0.3);
                    let a = noise_at(q, 400.0);
                    let b = noise_at(q * 1.7 + vec3(a.r * 40.0), 240.0);
                    let d = smoothstep(0.42, 0.85, a.r * 0.65 + b.g * 0.5);
                    let hue = smoothstep(0.3, 0.7, noise_at(q * 0.5, 600.0).r);
                    let gas = mix(vec3(0.55, 0.16, 0.85), vec3(0.12, 0.62, 0.78), hue);
                    let core = vec3(1.0, 0.72, 0.9) * smoothstep(0.78, 1.0, d) * 0.6;
                    let sigma = d * 0.0045 * density;
                    let step_trans = exp(-sigma * dt);
                    light += trans * ((1.0 - step_trans) * gas * 1.5 + core * sigma * dt * 3.0);
                    trans *= step_trans;
                } else {
                    // A pale violet mist hanging over the islands.
                    let h = max(pos.y - 40.0, 0.0);
                    let wisp = noise_at(pos + vec3(time * 0.8, 0.0, time * 0.4), 150.0).r;
                    let d = exp(-h / 40.0) * (0.2 + 1.3 * wisp);
                    let sigma = d * 0.0035 * density;
                    let lit = vec3(0.5, 0.36, 0.72) * 0.8 + p.fog.rgb * 0.2;
                    let step_trans = exp(-sigma * dt);
                    light += trans * (1.0 - step_trans) * lit;
                    trans *= step_trans;
                }
            }
            if trans < 0.02 {
                break;
            }
            t += dt;
        }
    }
    if mode == 0 {
        // Shafts sit in the fog: always a little, more where the fog is thick.
        let s = shafts(in.clip.xy, size, rd) * (0.3 + 0.7 * (1.0 - trans) * 3.0);
        light += sun_color() * min(s, 1.0) * 0.55;
    }
    return vec4(min(light, vec3(1.0)), 1.0 - trans);
}
