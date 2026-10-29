// The underwater look (gfx/underwater.rs): two fullscreen passes after the world, with the camera
// under the surface. `fs_transmit` multiplies the frame by how much light survives the water between
// the camera and what each pixel shows (red first, then green); `fs_light` adds what the water
// scatters back: its own colour, shafts of sunlight, and caustics on what the light reaches.
struct Params {
    // Clip space (with the TAA jitter) back to camera-relative world space.
    inv_view_proj: mat4x4<f32>,
    // xz: the camera's position modulo 256 (the noise's lattice period); w: time in seconds.
    camera: vec4<f32>,
    // xyz: towards the sun; w: how far above the horizon it stands (0..1).
    sun: vec4<f32>,
    // rgb: the colour of the water seen far away; w: strength (1 is the default look).
    body: vec4<f32>,
    // x: clarity, the blocks over which light dims by a factor e; y: the furthest distance used.
    look: vec4<f32>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var depth_tex: texture_depth_2d;

struct Out {
    @builtin(position) clip: vec4<f32>,
}
@vertex fn vs_underwater(@builtin(vertex_index) i: u32) -> Out {
    let q = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: Out;
    out.clip = vec4(q * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let h = p.inv_view_proj * vec4(ndc, z, 1.0);
    return h.xyz / h.w;
}
fn hash2(c: vec2<i32>) -> vec2<f32> {
    var h = vec2<u32>(u32(c.x & 255), u32(c.y & 255)) * vec2(374761393u, 668265263u);
    h = (h ^ (h.yx >> vec2(13u))) * vec2(1274126177u, 2246822519u);
    h = h ^ (h >> vec2(16u));
    return vec2<f32>(h & vec2(0xffffu)) / 65535.0;
}
fn hash1(c: vec2<i32>) -> f32 {
    return hash2(c).x;
}
// The gap between the nearest and the second-nearest of a moving field of points: near zero along
// the lines between cells, which is what a caustic network looks like.
fn cell_lines(q: vec2<f32>, t: f32) -> f32 {
    let base = floor(q);
    let f = q - base;
    var first = 8.0;
    var second = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let cell = vec2<i32>(base) + vec2(x, y);
            let h = hash2(cell);
            let point = vec2<f32>(f32(x), f32(y)) + 0.5 + 0.42 * sin(t * (0.6 + h.yx) + h * 6.2831);
            let d = length(point - f);
            if d < first {
                second = first;
                first = d;
            } else if d < second {
                second = d;
            }
        }
    }
    return second - first;
}
fn caustic(xz: vec2<f32>, t: f32) -> f32 {
    // Wide, soft lines: a bright network that reads as light, not as a drawn web.
    let a = 1.0 - smoothstep(0.0, 0.42, cell_lines(xz * 0.4, t * 0.8));
    let b = 1.0 - smoothstep(0.0, 0.5, cell_lines(xz * 0.7 + vec2(31.7, 12.3), t * 1.1 + 4.0));
    return a * a * 0.6 + b * b * 0.4;
}

struct View {
    rd: vec3<f32>,
    ro: vec3<f32>,
    dist: f32,
    far: bool,
    rel: vec3<f32>,
}
fn view_at(clip: vec4<f32>) -> View {
    let size = vec2<f32>(textureDimensions(depth_tex));
    let uv = clip.xy / size;
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ro = unproject(ndc, 1.0);
    let rd = normalize(unproject(ndc, 0.01) - ro);
    let depth = textureLoad(depth_tex, vec2<i32>(clip.xy), 0);
    var v: View;
    v.rd = rd;
    v.ro = ro;
    v.far = depth <= 0.0;
    v.rel = vec3(0.0);
    v.dist = p.look.y;
    if !v.far {
        v.rel = unproject(ndc, depth);
        v.dist = min(length(v.rel - ro), p.look.y);
    }
    return v;
}

// What survives the water along `dist`: red goes first, then green, blue lasts.
fn transmittance(dist: f32) -> vec3<f32> {
    return exp(-dist / max(p.look.x, 1.0) * vec3(2.4, 0.85, 0.5) * p.body.w);
}

@fragment fn fs_transmit(in: Out) -> @location(0) vec4<f32> {
    let v = view_at(in.clip);
    return vec4(transmittance(v.dist), 1.0);
}

@fragment fn fs_light(in: Out) -> @location(0) vec4<f32> {
    // The derivatives need uniform control flow, so they come first.
    let v = view_at(in.clip);
    let face = cross(dpdx(v.rel), dpdy(v.rel));
    let t_now = p.camera.w;
    let t = transmittance(v.dist);
    let day = p.sun.w;

    // The water's own colour, filling in what the light lost on the way.
    var light = p.body.rgb * (vec3(1.0) - t);

    // Caustics: sunlight focused by the surface onto what faces up, fading with depth.
    if !v.far && day > 0.01 {
        var n = normalize(face + vec3(1.0e-6));
        if dot(n, v.rel) > 0.0 {
            n = -n;
        }
        // Only what really faces up, and not at depth edges, where the surface's tilt is
        // garbage: the screen-space steps there are far longer than a pixel's footprint.
        let up = smoothstep(0.6, 0.92, n.y);
        let footprint = v.dist * 0.0012 + 0.004;
        let jump = max(length(dpdx(v.rel)), length(dpdy(v.rel))) / footprint;
        let flat_face = 1.0 - smoothstep(6.0, 14.0, jump);
        let xz = v.rel.xz + p.camera.xz;
        let c = caustic(xz, t_now);
        let reach = exp(-max(-v.rel.y, 0.0) * 0.05) * smoothstep(1.0, -1.5, v.rel.y)
            * exp(-v.dist * 0.03);
        light += vec3(0.7, 0.95, 1.0) * c * up * flat_face * reach * day * 0.3 * t.g * p.body.w;
    }

    // Shafts: the sun's light through the surface, seen as it scatters along the view. Their
    // pattern is where the ray crosses a plane above the camera.
    let toward = max(dot(v.rd, p.sun.xyz), 0.0);
    if v.rd.y > 0.08 && day > 0.01 {
        let hit = v.rd.xz * (7.0 / v.rd.y) + p.camera.xz;
        // Soft bands that drift: a few broad beams, not a pattern.
        let a = 0.5 + 0.5 * sin(hit.x * 0.35 + sin(hit.y * 0.25 + t_now * 0.4) * 1.2 + t_now * 0.3);
        let b = 0.5 + 0.5 * sin(hit.y * 0.3 - hit.x * 0.12 + t_now * 0.25);
        let streak = smoothstep(0.35, 1.0, a * 0.7 + b * 0.3);
        let reach = 1.0 - exp(-v.dist * 0.035);
        light += vec3(0.6, 0.95, 0.9) * pow(toward, 2.0) * streak * reach * day * 0.3 * p.body.w
            * smoothstep(0.08, 0.5, v.rd.y);
    }
    return vec4(light, 0.0);
}
