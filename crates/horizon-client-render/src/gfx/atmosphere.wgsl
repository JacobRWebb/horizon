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
    // x: how many fires there are.
    extra: vec4<f32>,
    // The fire blocks near the camera: camera-relative corner, w 1 for fire and 2 for soul fire.
    fires: array<vec4<f32>, 48>,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
// r: smooth fractal noise; g: billowy cellular noise; b: fine cellular noise.
@group(0) @binding(2) var noise: texture_3d<f32>;
@group(0) @binding(3) var samp: sampler;

const MAX_STEPS: i32 = 64;
const MAX_FIRES: i32 = 48;

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

// The void's sea: below the world, a field of black swells marched like a height field.
// The sea level, in blocks, and how far the swells rise and fall.
const SEA_AMP: f32 = 12.0;
fn sea_level(mode: i32) -> f32 {
    return select(-110.0, -40.0, mode == 2);
}
fn sea_height(xz: vec2<f32>, t: f32) -> f32 {
    // Slow rolling swells, a choppier layer on them and a fine ripple, all drifting.
    let swell = noise_at(vec3(xz.x, t * 1.5, xz.y), 240.0).r - 0.5;
    let chop = noise_at(vec3(xz.x + t * 3.0, t * 2.4, xz.y - t * 2.0), 80.0).r - 0.5;
    let ripple = noise_at(vec3(xz.x - t * 4.0, t * 3.0, xz.y), 24.0).r - 0.5;
    return swell * 18.0 + chop * 7.0 + ripple * 1.6;
}
fn sea_normal(xz: vec2<f32>, t: f32, e: f32) -> vec3<f32> {
    let dx = sea_height(xz - vec2(e, 0.0), t) - sea_height(xz + vec2(e, 0.0), t);
    let dz = sea_height(xz - vec2(0.0, e), t) - sea_height(xz + vec2(0.0, e), t);
    return normalize(vec3(dx, 2.0 * e, dz));
}
// The sea seen along `rd` from `ro` (premultiplied, alpha 1 where there is sea).
fn void_sea(ro: vec3<f32>, rd: vec3<f32>, mode: i32, time: f32, jitter: f32) -> vec4<f32> {
    let y0 = sea_level(mode);
    let top = y0 + SEA_AMP;
    let haze = select(p.fog.rgb * 0.25, vec3(0.012, 0.006, 0.028), mode == 2);
    if ro.y <= top || rd.y > -0.004 {
        return vec4(0.0);
    }
    let t_in = (top - ro.y) / rd.y;
    let t_out = (y0 - SEA_AMP - ro.y) / rd.y;
    // Far out the sea is only haze.
    let far = 3200.0;
    if t_in > far {
        return vec4(haze, 1.0);
    }
    let n = 30;
    let span = min(t_out, far) - t_in;
    var t = t_in + span / f32(n) * jitter;
    let dt = span / f32(n);
    var hit = -1.0;
    var last = t_in;
    for (var i = 0; i < n; i++) {
        let pos = ro + rd * t;
        if pos.y < y0 + sea_height(pos.xz, time) {
            hit = t;
            break;
        }
        last = t;
        t += dt;
    }
    if hit < 0.0 {
        return vec4(haze, 1.0);
    }
    // Bisect between the last step above the swells and the first below.
    var lo = last;
    var hi = hit;
    for (var i = 0; i < 5; i++) {
        let mid = 0.5 * (lo + hi);
        let pos = ro + rd * mid;
        if pos.y < y0 + sea_height(pos.xz, time) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let ts = 0.5 * (lo + hi);
    let pos = ro + rd * ts;
    let nrm = sea_normal(pos.xz, time, clamp(ts * 0.004, 0.5, 12.0));
    let fres = 0.03 + 0.97 * pow(1.0 - max(dot(nrm, -rd), 0.0), 4.0);
    let refl = reflect(rd, nrm);
    var sky = vec3(0.0);
    var glint = vec3(0.0);
    var deep = vec3(0.0);
    var crest_color = vec3(0.0);
    if mode == 2 {
        // Reflects the dark violet of the void and the nebula's gas.
        let gas = noise_at(refl * 400.0, 400.0).r;
        sky = mix(vec3(0.02, 0.008, 0.05), vec3(0.34, 0.12, 0.6), pow(clamp(gas * 1.4 - 0.2, 0.0, 1.0), 2.0));
        deep = vec3(0.002, 0.001, 0.006);
        crest_color = vec3(0.45, 0.14, 0.85);
    } else {
        sky = p.fog.rgb * 0.7;
        glint = vec3(1.0, 0.9, 0.75) * pow(max(dot(refl, p.sun.xyz), 0.0), 220.0) * p.sun.w * 4.0;
        deep = vec3(0.004, 0.01, 0.016);
        crest_color = p.fog.rgb * 0.9;
    }
    // Crests catch the light: thin lines along the tops of the swells.
    let rise = (pos.y - y0) / SEA_AMP;
    let sheen = noise_at(vec3(pos.x * 1.7 + time * 2.0, 0.0, pos.z * 1.7), 120.0).r;
    let crest = smoothstep(0.35, 0.85, rise) * smoothstep(0.45, 0.8, sheen);
    var color = deep + sky * fres + glint + crest_color * crest * 0.55;
    // A dark mist hangs over the swells, thicker in the troughs.
    let t_mist = max((y0 + 60.0 - ro.y) / rd.y, 0.0);
    let mist_len = max(ts - t_mist, 0.0);
    let mist_noise = 0.4 + 1.2 * noise_at(pos * 0.5 + vec3(time * 2.0, 0.0, time), 300.0).r;
    let mist = 1.0 - exp(-mist_len * 0.006 * mist_noise * p.fog.w);
    color = mix(color, haze * 3.0 + crest_color * 0.04, mist);
    // Distance melts into haze.
    color = mix(color, haze, 1.0 - exp(-ts / 1500.0));
    return vec4(color, 1.0);
}

// Fire: each fire block near the camera is a volume of flame with a plume of smoke over it,
// marched as the ray crosses their boxes. `p.fires` holds the blocks' camera-relative corners.
fn ray_box(o: vec3<f32>, d: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / select(d, vec3(1.0e-6), abs(d) < vec3(1.0e-6));
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let near = min(a, b);
    let far = max(a, b);
    return vec2(max(max(near.x, near.y), near.z), min(min(far.x, far.y), far.z));
}

// A flame's colour by how hot it burns: dark red at the tips, orange, yellow, and a pale core.
fn flame_color(heat: f32, soul: bool) -> vec3<f32> {
    if soul {
        var c = mix(vec3(0.02, 0.1, 0.4), vec3(0.08, 0.5, 1.0), smoothstep(0.0, 0.55, heat));
        return mix(c, vec3(0.7, 1.0, 1.0), smoothstep(0.85, 1.25, heat));
    }
    var c = mix(vec3(0.45, 0.03, 0.0), vec3(1.0, 0.3, 0.02), smoothstep(0.0, 0.45, heat));
    c = mix(c, vec3(1.0, 0.72, 0.14), smoothstep(0.4, 0.9, heat));
    return mix(c, vec3(1.0, 0.92, 0.65), smoothstep(0.95, 1.3, heat));
}

fn flame_volume(o: vec3<f32>, rd: vec3<f32>, t_a: f32, t_b: f32, seed: vec3<f32>, soul: bool,
    jitter: f32, light: ptr<function, vec3<f32>>, trans: ptr<function, f32>) {
    let time = p.cam.w;
    let n = 14;
    let dt = (t_b - t_a) / f32(n);
    var t = t_a + dt * jitter;
    for (var i = 0; i < n; i++) {
        let l = o + rd * t;
        let y = max(l.y, 0.0);
        let c = l.xz - 0.5;
        let s = seed + l;
        // The flame sways as it rises.
        let sway = vec3(sin(time * 2.9 + y * 3.7 + s.z * 5.0), 0.0, sin(time * 2.3 + y * 3.1 + s.x * 5.0))
            * (0.07 * y);
        let q = s + sway;
        // Tongues: noise stretched up and sliding up, a finer layer racing faster.
        let n1 = noise_at(vec3(q.x * 2.0, q.y * 0.7 - time * 2.2, q.z * 2.0), 6.0).r;
        let n2 = noise_at(vec3(q.x * 4.5 + 7.0, q.y * 1.6 - time * 4.5, q.z * 4.5), 6.0).r;
        let shape = n1 * 0.6 + n2 * 0.4;
        let d = shape * 1.55 - 0.32 - y * 0.85 - dot(c, c) * 2.3
            + 0.3 * (1.0 - smoothstep(0.0, 0.5, y));
        if d > 0.0 {
            let heat = clamp(d * 1.25 + (1.0 - y / 1.7) * 0.3, 0.0, 1.2);
            let sigma = d * 15.0;
            let a = 1.0 - exp(-sigma * dt);
            *light += *trans * flame_color(heat, soul) * a * 1.35;
            *trans *= exp(-sigma * dt * 0.3);
        }
        t += dt;
    }
}

fn smoke_volume(o: vec3<f32>, rd: vec3<f32>, t_a: f32, t_b: f32, seed: vec3<f32>, soul: bool,
    jitter: f32, light: ptr<function, vec3<f32>>, trans: ptr<function, f32>) {
    let time = p.cam.w;
    let n = 12;
    let dt = (t_b - t_a) / f32(n);
    var t = t_a + dt * jitter;
    for (var i = 0; i < n; i++) {
        let l = o + rd * t;
        let y = l.y - 0.9;
        if y > 0.0 {
            let s = seed + l;
            // The plume leans downwind and widens as it climbs.
            let lean = vec2(0.3, 0.12) * y;
            let off = l.xz - 0.5 - lean;
            let radius = 0.32 + 0.24 * y;
            let env = 1.0 - smoothstep(0.0, radius, length(off));
            let n1 = noise_at(vec3(s.x * 1.2, s.y * 0.8 - time * 1.0, s.z * 1.2), 6.0).r;
            let n2 = noise_at(vec3(s.x * 2.6 + 3.0, s.y * 1.8 - time * 1.9, s.z * 2.6), 6.0).r;
            let n3 = n1 * 0.6 + n2 * 0.4;
            let d = smoothstep(0.38, 0.76, n3) * env * smoothstep(0.0, 0.7, y)
                * (1.0 - smoothstep(2.8, 5.0, y));
            if d > 0.001 {
                let sigma = d * 2.6;
                let step_trans = exp(-sigma * dt);
                // Black soot, lit orange from below by the fire and faintly by the sky.
                var smoke = vec3(0.05, 0.045, 0.045) + p.fog.rgb * 0.05;
                var glow = vec3(1.0, 0.34, 0.06);
                if soul {
                    smoke = vec3(0.03, 0.05, 0.07) + p.fog.rgb * 0.04;
                    glow = vec3(0.08, 0.45, 0.9);
                }
                smoke += glow * exp(-y * 0.9) * 0.4;
                *light += *trans * (1.0 - step_trans) * smoke;
                *trans *= step_trans;
            }
        }
        t += dt;
    }
}

fn fires(ro: vec3<f32>, rd: vec3<f32>, t_scene: f32, jitter: f32,
    light: ptr<function, vec3<f32>>, trans: ptr<function, f32>) {
    let count = i32(p.extra.x);
    for (var i = 0; i < MAX_FIRES; i++) {
        if i >= count {
            break;
        }
        let f = p.fires[i];
        let o = ro - f.xyz;
        let soul = f.w > 1.5;
        let seed = p.cam.xyz + f.xyz;
        let flame = ray_box(o, rd, vec3(-0.05, 0.0, -0.05), vec3(1.05, 1.7, 1.05));
        let smoke = ray_box(o, rd, vec3(-1.6, 0.9, -1.6), vec3(2.6, 5.2, 2.6));
        let fa = max(flame.x, 0.0);
        let fb = min(flame.y, t_scene);
        let sa = max(smoke.x, 0.0);
        let sb = min(smoke.y, t_scene);
        let has_flame = fb > fa;
        let has_smoke = sb > sa;
        // Whichever the ray reaches first goes first.
        if has_smoke && (!has_flame || sa < fa) {
            smoke_volume(o, rd, sa, sb, seed, soul, jitter, light, trans);
            if has_flame {
                flame_volume(o, rd, fa, fb, seed, soul, jitter, light, trans);
            }
        } else {
            if has_flame {
                flame_volume(o, rd, fa, fb, seed, soul, jitter, light, trans);
            }
            if has_smoke {
                smoke_volume(o, rd, sa, sb, seed, soul, jitter, light, trans);
            }
        }
        if *trans < 0.02 {
            break;
        }
    }
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
    // Mode 3 is no air at all: only the fires.
    var t1 = select(min(t_scene, p.misc.w), 0.0, mode == 3);
    if mode == 2 && sky {
        // The nebula hangs in a shell far out in the void.
        t0 = 120.0;
        t1 = 720.0;
    }
    var light = vec3(0.0);
    var trans = 1.0;
    // Below the world, where nothing was drawn, lies the void's sea.
    var behind = vec4(0.0);
    if sky && (mode == 0 || mode == 2) {
        behind = void_sea(origin, rd, mode, time, jitter);
    }
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
                    let d_sky = d * smoothstep(-0.45, 0.05, rd.y);
                    let hue = smoothstep(0.3, 0.7, noise_at(q * 0.5, 600.0).r);
                    let gas = mix(vec3(0.55, 0.16, 0.85), vec3(0.12, 0.62, 0.78), hue);
                    let core = vec3(1.0, 0.72, 0.9) * smoothstep(0.78, 1.0, d) * 0.6;
                    let sigma = d_sky * 0.0045 * density;
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
    fires(ro, rd, t_scene, jitter, &light, &trans);
    light += trans * behind.rgb;
    trans *= 1.0 - behind.a;
    if mode == 0 {
        // Shafts sit in the fog: always a little, more where the fog is thick.
        let s = shafts(in.clip.xy, size, rd) * (0.3 + 0.7 * (1.0 - trans) * 3.0);
        light += sun_color() * min(s, 1.0) * 0.55;
    }
    return vec4(min(light, vec3(1.0)), 1.0 - trans);
}
