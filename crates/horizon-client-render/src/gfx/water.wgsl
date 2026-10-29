// Native water (gfx/water.rs): appended to terrain.wgsl, so it shares its Globals, sprite sampling,
// lighting, fog and sky. The water's top faces are redrawn with depth-based colour, refraction of
// the frame behind them, animated wave normals, a Fresnel sky reflection, a sun glint and shore
// foam. Everything else in the translucent terrain pass (glass, ice, water's sides) keeps the
// vanilla look.
struct WaterParams {
    // Camera-relative clip space back to camera-relative world space.
    inv_view_proj: mat4x4<f32>,
    // x, y: the water sprite layers
    layers: vec4<u32>,
    // x, y: the target's size in pixels
    screen: vec4<f32>,
    // x: time in seconds; z: the camera's height times 2, modulo 256 (the noise's lattice period)
    misc: vec4<f32>,
    // The camera's position times 2 and times 5, modulo 256 (the foam noise's lattice period).
    foam_offset: vec4<f32>,
    // x: clarity in blocks, y: deep water darkness, z: wave height, w: refraction
    look_a: vec4<f32>,
    // x: reflection, y: sun glint, z: foam, w: foam width in blocks
    look_b: vec4<f32>,
    // x: reflection march steps (0: sky only), y: how far a reflected ray looks, in blocks
    look_c: vec4<f32>,
    // While the camera is under water: rgb the colour of the water far away, w the strength of
    // the underwater look (zero above the surface).
    under: vec4<f32>,
    // Waves: wave vector, angular speed, and the phase the camera's position adds.
    waves: array<vec4<f32>, 8>,
}
@group(2) @binding(0) var<uniform> water: WaterParams;
@group(2) @binding(1) var scene_depth: texture_depth_2d;
@group(2) @binding(2) var scene_color: texture_2d<f32>;
@group(2) @binding(3) var scene_sampler: sampler;
// The player's model mirrored in the water, drawn from the real camera (premultiplied).
@group(2) @binding(4) var own_depth: texture_depth_2d;
@group(2) @binding(5) var own_color: texture_2d<f32>;

struct WaterOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
    @location(2) color: vec4<f32>,
    @location(3) spherical: f32,
    @location(4) cylindrical: f32,
    @location(5) @interpolate(flat) emission: u32,
    @location(6) rel: vec3<f32>,
    @location(7) lit: vec3<f32>,
    @location(8) tint: vec3<f32>,
    @location(9) sky: f32,
}
fn water_vertex(position: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>,
    light: vec2<u32>, origin: vec3<f32>) -> WaterOut {
    let t = terrain_vertex(position, uv, layer, color, light, origin);
    var out: WaterOut;
    out.clip = t.clip;
    out.uv = t.uv;
    out.layer = t.layer;
    out.color = t.color;
    out.spherical = t.spherical;
    out.cylindrical = t.cylindrical;
    out.emission = t.emission;
    out.rel = position + origin;
    let block = max(f32(light.y), dynamic_block_light(position + origin));
    out.lit = sample_lightmap(f32(light.x) / 16.0, block / 16.0).rgb;
    out.tint = color.rgb;
    out.sky = f32(light.x) / 240.0;
    return out;
}
@vertex fn vs_water(
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>,
    @location(2) layer: u32, @location(3) color: vec4<f32>, @location(4) light: vec2<u32>,
    @location(5) origin: vec4<f32>) -> WaterOut {
    return water_vertex(position, terrain_uv(uv, origin.w), layer, color, light, origin.xyz);
}
@vertex fn vs_water_flat(
    @builtin(vertex_index) vertex: u32,
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>,
    @location(2) layer: u32, @location(3) color: vec4<f32>, @location(4) light: vec2<u32>
) -> WaterOut {
    let origin = slot_origins[block_slots[vertex >> 8u]];
    return water_vertex(position, terrain_uv(uv, origin.w), layer, color, light, origin.xyz);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}
fn hash_lattice(p: vec2<i32>) -> f32 {
    var h = u32(p.x & 255) * 374761393u + u32(p.y & 255) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xffffu) / 65535.0;
}
// Value noise on a lattice that repeats every 256 cells.
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c = vec2<i32>(i);
    let a = hash_lattice(c);
    let b = hash_lattice(c + vec2(1, 0));
    let d = hash_lattice(c + vec2(0, 1));
    let e = hash_lattice(c + vec2(1, 1));
    return mix(mix(a, b, u.x), mix(d, e, u.x), u.y);
}

// How steeply each wave tilts the surface.
const WAVE_SLOPE: array<f32, 8> = array<f32, 8>(0.11, 0.10, 0.09, 0.08, 0.07, 0.055, 0.045, 0.035);
fn wave_normal(rel: vec3<f32>, dist: f32) -> vec3<f32> {
    let t = water.misc.x;
    var slope = vec2(0.0);
    for (var i = 0; i < 8; i++) {
        let w = water.waves[i];
        let k = length(w.xy);
        // Phase: the wave vector against the camera-relative position, plus what the camera's
        // own position adds (exact on the CPU), plus the wave's travel. A crest is a little
        // sharper than a sine.
        let ph = dot(w.xy, rel.xz) + w.w;
        let fade = 1.0 / (1.0 + dist * k * 0.05);
        slope += (w.xy / k) * (cos(ph + 0.5 * sin(ph)) * (1.0 + 0.5 * cos(ph))
            * WAVE_SLOPE[i] * fade);
    }
    slope *= water.look_a.z;
    return normalize(vec3(-slope.x, 1.0, -slope.y));
}

fn scene_depth_at(pixel: vec2<i32>) -> f32 {
    return textureLoad(scene_depth, pixel, 0);
}
// The camera-relative position of what the frame drew at `pixel` (the sky's far plane for 0).
fn scene_position(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    if depth <= 0.0 { return vec3(0.0, -1000.0, 0.0); }
    let uv = (vec2<f32>(pixel) + 0.5) / water.screen.xy;
    let ndc = vec3(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let p = water.inv_view_proj * vec4(ndc, 1.0);
    return p.xyz / p.w;
}

// Screen-space reflections: the reflected ray marched against the copy of the frame's depth.
struct Hit {
    color: vec3<f32>,
    weight: f32,
}
fn no_hit() -> Hit {
    return Hit(vec3(0.0), 0.0);
}
fn to_pixels(c: vec4<f32>) -> vec2<f32> {
    return vec2(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5) * water.screen.xy;
}
fn edge_fade(p: vec2<f32>) -> f32 {
    let uv = p / water.screen.xy;
    let e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return smoothstep(0.0, 0.1, e);
}
fn depth_at(p: vec2<f32>) -> f32 {
    let size = vec2<i32>(water.screen.xy);
    return textureLoad(scene_depth, clamp(vec2<i32>(p), vec2(0), size - 1), 0);
}
// Where the frame shows the sky in direction `dir`, if it does (the clouds are in it, when they
// were drawn before the copy).
fn screen_sky(dir: vec3<f32>) -> Hit {
    let c = globals.view_proj * vec4(dir, 0.0);
    if c.w < 0.05 { return no_hit(); }
    let p = to_pixels(c);
    let fade = edge_fade(p);
    if fade <= 0.0 { return no_hit(); }
    let open = max(max(depth_at(p), depth_at(p + vec2(3.0, 0.0))),
        max(max(depth_at(p - vec2(3.0, 0.0)), depth_at(p + vec2(0.0, 3.0))),
        depth_at(p - vec2(0.0, 3.0))));
    if open > 0.0 { return no_hit(); }
    let color = textureSampleLevel(scene_color, scene_sampler, p / water.screen.xy, 0.0).rgb;
    return Hit(color, fade);
}
// Marches from `origin` along `dir` (camera-relative): linear steps in screen space, then a
// binary search between the last two. Depth is reversed (nearer is larger) and linear in screen
// space, so it is interpolated as is; its view-space distance is 1 / (a * depth + b).
fn trace_reflection(origin: vec3<f32>, dir: vec3<f32>, jitter: f32) -> Hit {
    let steps = i32(water.look_c.x);
    if steps <= 0 { return no_hit(); }
    let c0 = globals.view_proj * vec4(origin, 1.0);
    var c1 = globals.view_proj * vec4(origin + dir * water.look_c.y, 1.0);
    if c0.w < 0.05 { return no_hit(); }
    if c1.w < 0.05 { c1 = mix(c0, c1, (0.05 - c0.w) / (c1.w - c0.w)); }
    let k0 = 1.0 / c0.w;
    let k1 = 1.0 / c1.w;
    let s0 = to_pixels(c0);
    let s1 = to_pixels(c1);
    let d0 = c0.z * k0;
    let d1 = c1.z * k1;
    let delta = s1 - s0;
    let len = max(abs(delta.x), abs(delta.y));
    if len < 2.0 { return no_hit(); }
    let count = min(f32(steps), len / 2.0);
    let dt = 1.0 / count;
    let a = water.inv_view_proj[2][3];
    let b = water.inv_view_proj[3][3];
    var previous = 0.0;
    for (var i = 0; i < 96; i++) {
        if f32(i) >= count { break; }
        let t = min((f32(i) + 1.0 + jitter) * dt, 1.0);
        let p = s0 + delta * t;
        if p.x < 0.0 || p.y < 0.0 || p.x >= water.screen.x || p.y >= water.screen.y { break; }
        let ray = mix(d0, d1, t);
        let seen = depth_at(p);
        if seen > ray && seen > 0.0 {
            let z_ray = 1.0 / max(a * ray + b, 1e-6);
            let z_seen = 1.0 / max(a * seen + b, 1e-6);
            if z_ray - z_seen < 0.7 + 0.04 * z_ray {
                var lo = previous;
                var hi = t;
                for (var j = 0; j < 5; j++) {
                    let mid = 0.5 * (lo + hi);
                    if depth_at(s0 + delta * mid) > mix(d0, d1, mid) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let q = s0 + delta * hi;
                let color = textureSampleLevel(scene_color, scene_sampler, q / water.screen.xy,
                    0.0).rgb;
                return Hit(color, edge_fade(q) * (1.0 - smoothstep(0.7, 1.0, hi)));
            }
        }
        previous = t;
    }
    return no_hit();
}

// The surface seen from below: a window onto the sky where light can leave the water (Snell's
// window), and a mirror of the dark water everywhere else.
fn underside(in: WaterOut) -> vec4<f32> {
    let dist = length(in.rel);
    let incident = in.rel / dist;
    let n = -wave_normal(in.rel, dist);
    let t = refract(incident, n, 1.33);
    var window = vec3(0.0);
    var mirror_weight = 1.0;
    if dot(t, t) > 1.0e-4 {
        let air = normalize(t);
        let cos_air = clamp(dot(air, -n), 0.0, 1.0);
        mirror_weight = 0.02 + 0.98 * pow(1.0 - cos_air, 5.0);
        let sun = globals.sun.xyz;
        let shimmer = 0.9 + 0.2 * value_noise(in.rel.xz * 0.8 + vec2(water.misc.x * 0.4, 0.0));
        window = (sky_along_ray(air) * 1.15
            + vec3(1.0, 0.95, 0.8) * pow(max(dot(air, sun), 0.0), 500.0) * 10.0) * shimmer;
    }
    let mirrored = water.under.rgb * 0.9;
    return vec4(mix(window, mirrored, mirror_weight), 1.0);
}

// Water falling down a face: streaks that run down, whiter where they are thick, over the
// water's own colour, with a sheen of the sky. The streaks are fixed to the world (height
// included), so they do not slide when the camera moves.
fn waterfall(in: WaterOut, base: vec4<f32>, face: vec3<f32>) -> vec4<f32> {
    let t = water.misc.x;
    let world = in.rel.xz * 2.0 + water.foam_offset.xy;
    // The face runs along x or along z.
    let across = select(world.y, world.x, abs(face.z) > abs(face.x));
    let height = in.rel.y * 2.0 + water.misc.z;
    let fall = value_noise(vec2(across * 1.4, height * 0.3 + t * 8.0)) * 0.55
        + value_noise(vec2(across * 3.3 + 9.0, height * 0.45 + t * 13.0)) * 0.45;
    let streak = smoothstep(0.5, 0.85, fall);
    let tint = clamp(in.tint, vec3(0.05), vec3(1.0));
    let body = mix(base.rgb, mix(tint, vec3(0.35, 0.8, 0.85), 0.5), 0.55) * in.lit;
    let white = vec3(0.95, 0.98, 1.0) * (0.4 + 0.6 * luma(in.lit));
    var color = mix(body, white, streak * 0.8);
    // A sheen of the sky, stronger at grazing angles.
    let v = -normalize(in.rel);
    let n = normalize(face * select(-1.0, 1.0, dot(face, v) > 0.0));
    let fresnel = 0.04 + 0.96 * pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 5.0);
    color = mix(color, sky_along_ray(reflect(-v, n)) * (0.35 + 0.65 * smoothstep(0.25, 0.95, in.sky)),
        fresnel * 0.5 * water.look_b.x);
    let alpha = mix(0.6, 0.97, streak);
    return apply_fog(vec4(color, alpha), in.spherical, in.cylindrical);
}

@fragment fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    // Sampled in uniform control flow, before anything branches.
    let base = sprite(in.uv, in.layer & 0x7fffu) * in.color;
    let face = normalize(cross(dpdx(in.rel), dpdy(in.rel)));
    let is_water = in.layer == water.layers.x || in.layer == water.layers.y;
    if base.a < 0.01 { discard; }
    if !is_water {
        return apply_fog(base, in.spherical, in.cylindrical);
    }
    if abs(face.y) < 0.6 {
        return waterfall(in, base, face);
    }

    if water.under.w > 0.0 && in.rel.y > 0.0 {
        return underside(in);
    }

    let dist = length(in.rel);
    let v = -in.rel / dist;
    var n = wave_normal(in.rel, dist);
    // Flowing water on a slope: ripples stretched along the way down, carried downstream, with
    // foam in the faster streaks. The pattern keeps to the world's axes (the lattice repeats
    // along them), so it is stretched along the one the flow is nearer to.
    var stream = 0.0;
    if in.layer == water.layers.y {
        let up_face = select(-face, face, face.y > 0.0);
        let slope = length(up_face.xz);
        if slope > 0.03 {
            let dir = up_face.xz / slope;
            let speed = clamp(slope * 6.0, 0.35, 1.0);
            let world = in.rel.xz * 2.0 + water.foam_offset.xy;
            let stretch = vec2(mix(1.0, 0.3, smoothstep(0.55, 0.85, abs(dir.x))),
                mix(1.0, 0.3, smoothstep(0.55, 0.85, abs(dir.y))));
            let p = world * 0.9 * stretch - dir * (water.misc.x * 2.2 * speed);
            let streaks = value_noise(p) * 0.6 + value_noise(p * 2.1 + vec2(17.0, 5.0)) * 0.4;
            n = normalize(n + vec3(dir.x, 0.0, dir.y) * (streaks - 0.5) * 0.35 * speed);
            stream = smoothstep(0.6, 0.86, streaks) * speed;
        }
    }
    let pixel = vec2<i32>(in.clip.xy);
    let size = vec2<i32>(water.screen.xy);

    // Refraction: look through the surface where the wave tilts it, as long as what that
    // reaches is behind the water.
    let depth0 = scene_depth_at(clamp(pixel, vec2(0), size - 1));
    let near = scene_position(pixel, depth0);
    let vertical0 = clamp(in.rel.y - near.y, 0.0, 64.0);
    let shallow_fade = clamp(vertical0 * 0.25, 0.15, 1.0);
    let shift = n.xz * (0.09 * water.look_a.w * shallow_fade / (1.0 + dist * 0.03));
    var pixel2 = clamp(pixel + vec2<i32>(shift * water.screen.xy * vec2(1.0, -1.0)),
        vec2(0), size - 1);
    var depth = scene_depth_at(pixel2);
    if depth > in.clip.z - 0.00001 * in.clip.z {
        pixel2 = clamp(pixel, vec2(0), size - 1);
        depth = depth0;
    }
    let behind = textureLoad(scene_color, pixel2, 0).rgb;
    let floor_pos = scene_position(pixel2, depth);
    // How far the view ray travels under the surface, and how deep the water is straight down.
    var path = 64.0;
    var vertical = 64.0;
    if depth > 0.0 {
        path = min(length(floor_pos - in.rel), 64.0);
        vertical = clamp(in.rel.y - floor_pos.y, 0.0, 64.0);
    }
    let into = max(abs(v.y), 0.2);
    path = min(path, vertical / into + 0.25);

    // Colour of the water body: the biome's tint, darkening and saturating with depth.
    let clarity = max(water.look_a.x, 1.0);
    let optical = path / clarity;
    let transmit = exp(-optical * vec3(2.4, 0.85, 0.5));
    let tint = clamp(in.tint, vec3(0.02), vec3(1.0));
    let deep = pow(tint, vec3(1.5)) * 0.30 * water.look_a.y;
    let clear = mix(tint, vec3(0.45, 0.95, 0.9), 0.55) * 0.42;
    let body = mix(clear, deep, smoothstep(0.0, 1.6, optical)) * in.lit;
    var color = behind * transmit + body * (1.0 - transmit);

    // Fresnel: the sky and sun in the surface, strongest at grazing angles.
    let ndv = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = 0.03 + 0.97 * pow(1.0 - ndv, 5.0);
    let skylit = smoothstep(0.25, 0.95, in.sky);
    let r = reflect(-v, n);
    let r_up = normalize(vec3(r.x, abs(r.y) + 0.02, r.z));
    // The sky: the frame's own (clouds and the sun's disc, where it shows that direction), else
    // the analytic one. Then what the reflected ray meets in the frame, over it.
    let sky_screen = screen_sky(r_up);
    let sky = mix(sky_along_ray(r_up), sky_screen.color, sky_screen.weight)
        * (0.35 + 0.65 * skylit);
    let jitter = fract(52.9829189 * fract(dot(in.clip.xy, vec2(0.06711056, 0.00583715))));
    let seen = trace_reflection(in.rel + n * 0.05, r, jitter);
    let graze = 1.0 - smoothstep(0.5, 0.9, dot(r, v));
    var reflected = mix(sky, seen.color, seen.weight * graze * step(0.0, r.y));
    // Your own reflection: the mirrored model where it lands on this pixel (rippled by the waves),
    // if it is behind the surface.
    let ripple = vec2<i32>(n.xz * vec2(1.0, -1.0) * water.screen.xy * 0.012);
    let mirror_pixel = clamp(pixel + ripple, vec2(0), size - 1);
    let own_z = textureLoad(own_depth, mirror_pixel, 0);
    let own = textureLoad(own_color, mirror_pixel, 0);
    let own_seen = select(0.0, 1.0, own_z > 0.0 && own_z <= in.clip.z);
    reflected = reflected * (1.0 - own.a * own_seen) + own.rgb * own_seen;
    let strength = clamp(fresnel * water.look_b.x, 0.0, 0.97);
    color = mix(color, reflected, strength);

    // The sun's glint: a tight lobe and a soft sheen around it, in the sun's colour.
    let sun = globals.sun.xyz;
    let sun_up = smoothstep(-0.02, 0.14, sun.y);
    let along = max(dot(r_up, sun), 0.0);
    var sun_color = vec3(1.0, 0.96, 0.86);
    sun_color = mix(sun_color, globals.sunrise.rgb * 1.2 + vec3(0.2, 0.1, 0.0),
        clamp(globals.sunrise.a, 0.0, 1.0) * 0.8);
    let glint = pow(along, 900.0) * 28.0 + pow(along, 90.0) * 0.9 + pow(along, 12.0) * 0.06;
    color += sun_color * glint * sun_up * skylit * water.look_b.y * (0.4 + 0.6 * fresnel * 4.0);

    // Foam where the water gets thin against the shore (and around anything standing in it).
    let width = max(water.look_b.w, 0.05);
    let wp2 = in.rel.xz * 2.0 + water.foam_offset.xy;
    let wp5 = in.rel.xz * 5.0 + water.foam_offset.zw;
    let t = water.misc.x;
    let lace = value_noise(wp2 + vec2(t * 0.35, t * 0.2)) * 0.6
        + value_noise(wp5 - vec2(t * 0.5, t * 0.3)) * 0.4;
    let edge = 1.0 - clamp(vertical0 / width, 0.0, 1.0);
    let band = smoothstep(0.32, 0.72, edge * 1.25 + lace * 0.55 - 0.22);
    let line = 1.0 - smoothstep(0.0, 0.22, vertical0 + (lace - 0.5) * 0.22);
    let foam = max(band * 0.85, line) * clamp(water.look_b.z, 0.0, 1.5)
        * select(0.0, 1.0, depth0 > 0.0);
    let foam_color = vec3(0.93, 0.97, 1.0) * (0.35 + 0.65 * luma(in.lit));
    color = mix(color, foam_color, clamp(max(foam, stream * 0.6), 0.0, 1.0));

    // Alpha 1: the frame behind is already in the colour. Soft at the very rim so a shore
    // that the refraction could not reach still blends.
    let rim = smoothstep(0.0, 0.08, vertical0 + 0.02);
    return apply_fog(vec4(color, 1.0), in.spherical, in.cylindrical) * vec4(1.0, 1.0, 1.0, rim);
}
