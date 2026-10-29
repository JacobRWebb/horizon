// Terrain, sky and outline shaders, ported from the pinned vanilla
// terrain.vsh/fsh, fog.glsl and sample_lightmap.glsl.
struct Globals {
    view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    // environmental start/end, render-distance start/end
    fog: vec4<f32>,
    sky_color: vec4<f32>,
    // x: sky darkening, y: game time, z: aspect, w: tan(fov/2)
    misc: vec4<f32>,
    inverse_view: mat4x4<f32>,
    sunrise: vec4<f32>,
    // xyz: sun direction, w: star brightness
    sun: vec4<f32>,
    // x: sky fog end, y: cloud fog end, z: dark disc flag
    fog_ends: vec4<f32>,
    // Atlas rects (u0, v0, width, height) of end_sky.png and end_portal.png.
    end_portal_sprites: array<vec4<f32>, 2>,
    // x: vanilla's GameTime ((game time % 24000 + partial tick) / 24000), z: how many
    // dynamic_lights are set
    time: vec4<f32>,
    // Atlas rect of the current destroy stage (read by entity.wgsl).
    crumbling_sprite: vec4<f32>,
    outline_params: vec4<f32>,
    // Horizon Extras' dynamic lights: camera-relative position and level (dynamic_light.rs).
    dynamic_lights: array<vec4<f32>, 16>,
    // The sunrise fan's 16 triangles (gfx/sunrise_fan.rs): xyz the apex weight's normal, w the
    // triangle's volume with the camera; then each rim vertex's normal against the apex.
    sunrise_fan: array<vec4<f32>, 32>,
    // x, y, z: the sprite layers, plus 1 (0: none), of nether_portal, lava_still and lava_flow.
    special: vec4<f32>,
    // x, y, z, w: the same for fire_0, fire_1, soul_fire_0 and soul_fire_1.
    special_b: vec4<f32>,
    // xyz: the camera's position modulo 256 (the procedural noise's period); w: seconds.
    camera: vec4<f32>,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var sprites: texture_2d_array<f32>;
@group(0) @binding(2) var sprite_sampler: sampler;
@group(0) @binding(3) var lightmap: texture_2d<f32>;
@group(0) @binding(4) var lightmap_sampler: sampler;
// Layers past the first array's SPRITE_SPLIT (2048, gfx.rs): mods' sprites.
@group(0) @binding(5) var sprites_1: texture_2d_array<f32>;
@group(0) @binding(6) var sprites_2: texture_2d_array<f32>;
@group(0) @binding(7) var sprites_3: texture_2d_array<f32>;
@group(0) @binding(8) var sprites_4: texture_2d_array<f32>;
@group(0) @binding(9) var sprites_5: texture_2d_array<f32>;
@group(0) @binding(10) var sprites_6: texture_2d_array<f32>;
@group(0) @binding(11) var sprites_7: texture_2d_array<f32>;
// How many sprite arrays hold layers (a pipeline constant): 1 for vanilla,
// which then samples exactly as before.
override SPRITE_ARRAYS: u32 = 1u;
override SPRITE_SPLIT: u32 = 2048u;
fn sprite(uv: vec2<f32>, layer: u32) -> vec4<f32> {
    if SPRITE_ARRAYS == 1u {
        return textureSample(sprites, sprite_sampler, uv, i32(layer));
    }
    // The array is chosen per fragment: sample with the derivatives taken here.
    let dx = dpdx(uv);
    let dy = dpdy(uv);
    let l = i32(layer % SPRITE_SPLIT);
    switch layer / SPRITE_SPLIT {
        case 0u: { return textureSampleGrad(sprites, sprite_sampler, uv, l, dx, dy); }
        case 1u: { return textureSampleGrad(sprites_1, sprite_sampler, uv, l, dx, dy); }
        case 2u: { return textureSampleGrad(sprites_2, sprite_sampler, uv, l, dx, dy); }
        case 3u: { return textureSampleGrad(sprites_3, sprite_sampler, uv, l, dx, dy); }
        case 4u: { return textureSampleGrad(sprites_4, sprite_sampler, uv, l, dx, dy); }
        case 5u: { return textureSampleGrad(sprites_5, sprite_sampler, uv, l, dx, dy); }
        case 6u: { return textureSampleGrad(sprites_6, sprite_sampler, uv, l, dx, dy); }
        default: { return textureSampleGrad(sprites_7, sprite_sampler, uv, l, dx, dy); }
    }
}

struct TerrainOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
    @location(2) color: vec4<f32>,
    @location(3) spherical: f32,
    @location(4) cylindrical: f32,
    // The light the face emits, 0 to 15 (mesher.rs's `EMISSION_BITS`).
    @location(5) @interpolate(flat) emission: u32,
    // The camera-relative position.
    @location(6) rel: vec3<f32>,
}
// Horizon Extras' dynamic lights (dynamic_light.rs): the block light, in the vertex's units
// (16 per level), a vertex at camera-relative `pos` gets from the frame's light sources, one
// level less per block. time.z of them are set: none when the extra is off, so no loop runs.
fn dynamic_block_light(pos: vec3<f32>) -> f32 {
    let count = min(u32(globals.time.z), 16u);
    var level = 0.0;
    for (var i = 0u; i < count; i++) {
        let l = globals.dynamic_lights[i];
        level = max(level, l.w - distance(pos, l.xyz));
    }
    return level * 16.0;
}
fn sample_lightmap(sky: f32, block: f32) -> vec4<f32> {
    let uv = vec2(block, sky) * 16.0;
    return textureSampleLevel(lightmap, lightmap_sampler,
        clamp(uv / 256.0 + 0.5 / 16.0, vec2(0.5 / 16.0), vec2(15.5 / 16.0)), 0.0);
}
// mesher.rs's TerrainVertex holds the UV in fixed point; the instance's
// origin.w is the step: 1/32768 per unit for sections (a power of two on an
// integer below 2^24, so it decodes exactly), 1/32 per block for distant
// terrain tiles (TerrainVertex::tiled).
fn terrain_uv(packed: vec2<u32>, step: f32) -> vec2<f32> {
    return vec2<f32>(packed) * step;
}
@vertex fn vs_terrain(
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>,
    @location(2) layer: u32, @location(3) color: vec4<f32>, @location(4) light: vec2<u32>,
    @location(5) origin: vec4<f32>) -> TerrainOut {
    return terrain_vertex(position, terrain_uv(uv, origin.w), layer, color, light, origin.xyz);
}
// Flat section draws (gfx/flat.rs): one draw per arena page over the frame's
// index list. A section's vertices fill whole 256-vertex blocks of its page:
// the block gives its resident slot, the slot its camera-relative origin.
@group(1) @binding(0) var<storage, read> block_slots: array<u32>;
@group(1) @binding(1) var<storage, read> slot_origins: array<vec4<f32>>;
@vertex fn vs_terrain_flat(
    @builtin(vertex_index) vertex: u32,
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>,
    @location(2) layer: u32, @location(3) color: vec4<f32>, @location(4) light: vec2<u32>
) -> TerrainOut {
    let origin = slot_origins[block_slots[vertex >> 8u]];
    return terrain_vertex(position, terrain_uv(uv, origin.w), layer, color, light, origin.xyz);
}
fn terrain_vertex(position: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>,
    light: vec2<u32>, origin: vec3<f32>) -> TerrainOut {
    let pos = position + origin;
    var out: TerrainOut;
    out.clip = globals.view_proj * vec4(pos, 1.0);
    out.uv = uv;
    out.layer = layer;
    let block = max(f32(light.y), dynamic_block_light(pos));
    out.color = vec4(color.rgb, 1.0) * sample_lightmap(f32(light.x) / 16.0, block / 16.0);
    out.emission = 15u - (u32(round(color.a * 255.0)) & 15u);
    out.rel = pos;
    out.spherical = length(pos);
    out.cylindrical = max(length(pos.xz), abs(pos.y));
    return out;
}
fn linear_fog(d: f32, start: f32, end: f32) -> f32 {
    if d <= start { return 0.0; }
    if d >= end { return 1.0; }
    return (d - start) / (end - start);
}
fn apply_fog(color: vec4<f32>, spherical: f32, cylindrical: f32) -> vec4<f32> {
    let f = max(linear_fog(spherical, globals.fog.x, globals.fog.y),
                linear_fog(cylindrical, globals.fog.z, globals.fog.w));
    return vec4(mix(color.rgb, globals.fog_color.rgb, f * globals.fog_color.a), color.a);
}
// Procedural noise for the portal and the lava: value noise on a lattice that repeats every 64
// cells, sampled at multiples of 1/4 per block, so it repeats every 256 blocks like the
// camera position it is offset by (`globals.camera`).
fn hash3(p: vec3<i32>) -> f32 {
    var h = u32(p.x & 63) * 374761393u + u32(p.y & 63) * 668265263u + u32(p.z & 63) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xffffu) / 65535.0;
}
fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c = vec3<i32>(i);
    let x0 = mix(hash3(c), hash3(c + vec3(1, 0, 0)), u.x);
    let x1 = mix(hash3(c + vec3(0, 1, 0)), hash3(c + vec3(1, 1, 0)), u.x);
    let x2 = mix(hash3(c + vec3(0, 0, 1)), hash3(c + vec3(1, 0, 1)), u.x);
    let x3 = mix(hash3(c + vec3(0, 1, 1)), hash3(c + vec3(1, 1, 1)), u.x);
    return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}
// Three octaves, from one cell per 4 blocks: 0 to about 0.9.
fn fbm3(p: vec3<f32>) -> f32 {
    return vnoise3(p * 0.25) * 0.5 + vnoise3(p * 0.5) * 0.3 + vnoise3(p) * 0.2;
}
fn world_position(rel: vec3<f32>) -> vec3<f32> {
    return rel + globals.camera.xyz;
}
// Lava, ray-marched a little way into the surface: slow plates of crust drifting on a warped
// flow, with white-hot cracks between them that burn deeper than the crust is thick.
fn lava_surface(rel: vec3<f32>, flowing: bool) -> vec3<f32> {
    let t = globals.camera.w;
    let dir = normalize(rel);
    let world = world_position(rel);
    let drift = select(vec3(0.03, 0.0, 0.02), vec3(0.0, -0.16, 0.0), flowing) * t;
    let warp = vec3(
        fbm3(world * 0.5 + drift + vec3(0.0, t * 0.02, 0.0)),
        fbm3(world * 0.5 + drift + vec3(17.0, 0.0, t * 0.02)),
        fbm3(world * 0.5 + drift + vec3(5.0, 11.0, 0.0)),
    );
    var crack = 0.0;
    var halo = 0.0;
    var crust = 0.0;
    let p = world + (warp - 0.4) * 3.0;
    for (var i = 0; i < 4; i++) {
        let deep = f32(i);
        // Looking in at an angle sees the cracks shift: deeper samples are offset along the view.
        let q = p + dir * (deep * 0.18);
        let n = fbm3(q + drift * 2.0);
        // The cracks run where the noise crosses its middle; they widen a little with depth.
        let d = abs(n - 0.5);
        crack += smoothstep(0.045 + deep * 0.01, 0.0, d) * (1.0 - deep * 0.18);
        halo += smoothstep(0.2, 0.0, d);
        if i == 0 {
            crust = fbm3(q * 2.0 + vec3(0.0, t * 0.05, 0.0));
        }
    }
    crack = clamp(crack * 0.5, 0.0, 1.0);
    halo = clamp(halo * 0.25, 0.0, 1.0);
    let pulse = 0.85 + 0.15 * sin(t * 1.4 + warp.x * 9.0 + crust * 6.0);
    // Cooled crust, near black and reddish, with plates of lighter rock.
    let rock = vec3(0.075, 0.02, 0.012) + crust * vec3(0.12, 0.03, 0.012);
    // The glow bleeds out of the cracks into the crust beside them.
    let ember = vec3(0.7, 0.1, 0.01) * pow(halo, 2.0);
    let fire = mix(vec3(1.0, 0.32, 0.02), vec3(1.0, 0.72, 0.2), crack);
    return (rock + ember * pulse) + fire * crack * pulse * 1.15;
}
// The Nether portal as a volume of gas: marched a block and a half behind the face, ribbons of
// violet and magenta churning and swaying, with pale cores where the gas is thickest.
fn portal_gas(rel: vec3<f32>) -> vec4<f32> {
    let t = globals.camera.w;
    let dir = normalize(rel);
    let world = world_position(rel);
    var trans = 1.0;
    var acc = vec3(0.0);
    let steps = 18;
    let dt = 1.6 / f32(steps);
    for (var i = 0; i < steps; i++) {
        var p = world + dir * (f32(i) + 0.5) * dt;
        // Waves: the whole sheet sways.
        p += 0.22 * vec3(
            sin(p.y * 2.9 + t * 1.3 + p.z * 1.7),
            sin(p.z * 2.3 + t * 0.9 + p.x * 1.9),
            sin(p.x * 2.7 + t * 1.1 + p.y * 1.3),
        );
        let flow = vec3(t * 0.12, -t * 0.34, t * 0.09);
        let n = fbm3(p * 1.5 + flow);
        let m = fbm3(p * 2.5 - flow * 1.6 + vec3(n * 3.0));
        let density = smoothstep(0.28, 0.85, n * 0.7 + m * 0.5);
        let ribbon = pow(1.0 - abs(2.0 * m - 0.9), 3.0);
        let a = 1.0 - exp(-density * dt * 5.0);
        let violet = vec3(0.30, 0.04, 0.62);
        let magenta = vec3(0.92, 0.22, 1.0);
        var c = mix(violet, magenta, clamp(ribbon + n * 0.4, 0.0, 1.0));
        c = mix(c, vec3(1.0, 0.82, 1.0), pow(density, 3.0) * 0.7);
        acc += trans * a * c;
        trans *= 1.0 - a;
    }
    let alpha = 1.0 - trans;
    // A faint glow where the gas is thin, so the doorway never reads as empty.
    let base = vec3(0.2, 0.03, 0.4);
    let color = (acc + trans * base * 0.5) / max(alpha + trans * 0.5, 0.001);
    return vec4(color * 1.15, mix(0.55, 0.96, alpha));
}
// Fire as a volume of flame: marched half a block through the quad it is drawn on, tongues
// licking up on a rising, swaying noise, hottest and whitest at the base and cooling through
// yellow and orange to dark red at the tips. The quad has no soft edge, so the flame is dithered
// (TAA smooths it).
fn fire_flame(rel: vec3<f32>, uv: vec2<f32>, soul: bool) -> vec4<f32> {
    let t = globals.camera.w;
    let dir = normalize(rel);
    let world = world_position(rel);
    // Fire fills its block's height: the quad's top is uv.y = 0.
    let h0 = clamp(1.0 - uv.y, 0.0, 1.0);
    let base_y = world.y - h0;
    let edge = smoothstep(0.55, 1.0, abs(uv.x * 2.0 - 1.0));
    var trans = 1.0;
    var acc = vec3(0.0);
    let steps = 8;
    let dt = 0.5 / f32(steps);
    for (var i = 0; i < steps; i++) {
        let p = world + dir * (f32(i) + 0.5) * dt;
        let h = clamp(p.y - base_y, 0.0, 1.3);
        // Tongues: stretched up, sliding up, swaying from side to side.
        let sway = vec3(sin(t * 3.1 + p.y * 4.0 + p.z * 3.0), 0.0, sin(t * 2.6 + p.y * 3.5 + p.x * 3.0))
            * (0.12 * h);
        let q = (p + sway) * vec3(4.0, 1.4, 4.0) + vec3(0.0, -t * 3.2, 0.0);
        let n = fbm3(q) * 1.25;
        let fine = vnoise3(q * 2.0 + vec3(0.0, -t * 5.0, 0.0));
        let density = n + fine * 0.25 - h * 1.05 + 0.5 - edge * 0.7;
        let heat = clamp(density * 1.3 + (1.0 - h) * 0.45, 0.0, 1.5);
        var c: vec3<f32>;
        if soul {
            c = mix(vec3(0.02, 0.12, 0.45), vec3(0.1, 0.55, 1.0), smoothstep(0.0, 0.6, heat));
            c = mix(c, vec3(0.75, 1.0, 1.0), smoothstep(0.7, 1.4, heat));
        } else {
            c = mix(vec3(0.5, 0.04, 0.0), vec3(1.0, 0.33, 0.02), smoothstep(0.0, 0.5, heat));
            c = mix(c, vec3(1.0, 0.8, 0.2), smoothstep(0.5, 1.0, heat));
            c = mix(c, vec3(1.0, 0.97, 0.8), smoothstep(1.0, 1.5, heat));
        }
        let a = 1.0 - exp(-clamp(density, 0.0, 2.0) * dt * 22.0);
        acc += trans * a * c;
        trans *= 1.0 - a;
    }
    let alpha = 1.0 - trans;
    return vec4(acc / max(alpha, 0.001) * 1.25, alpha);
}
@fragment fn fs_solid(in: TerrainOut) -> @location(0) vec4<f32> {
    // Bit 15 of the layer marks opaque leaves (forceOpaque): no alpha cutout.
    let color = sprite(in.uv, in.layer & 0x7fffu) * in.color;
    if color.a < 0.5 && (in.layer & 0x8000u) == 0u { discard; }
    let own_layer = (in.layer & 0x7fffu) + 1u;
    if own_layer == u32(globals.special.y) || own_layer == u32(globals.special.z) {
        let lit = lava_surface(in.rel, own_layer == u32(globals.special.z))
            * (0.6 + 0.4 * in.color.rgb);
        return apply_fog(vec4(lit, 0.0), in.spherical, in.cylindrical);
    }
    let fire_layer = vec4<u32>(globals.special_b);
    if any(vec4<u32>(own_layer) == fire_layer) {
        // Near the camera the flames are a volume (gfx/atmosphere.wgsl).
        if length(in.rel) < globals.special.w { discard; }
        let soul = own_layer == fire_layer.z || own_layer == fire_layer.w;
        let flame = fire_flame(in.rel, in.uv, soul);
        let dither = fract(52.9829189 * fract(dot(in.clip.xy + globals.camera.w * 61.0,
            vec2(0.06711056, 0.00583715))));
        if flame.a <= dither { discard; }
        return apply_fog(vec4(flame.rgb, 0.0), in.spherical, in.cylindrical);
    }
    // Alpha 0 tells HDR output (hdr.wgsl) a light-emitting block's bright pixels
    // may shine. Nothing else reads the frame's alpha.
    let emissive = select(1.0, 0.0, in.emission > 0u);
    return apply_fog(vec4(color.rgb, emissive), in.spherical, in.cylindrical);
}
@fragment fn fs_translucent(in: TerrainOut) -> @location(0) vec4<f32> {
    let color = sprite(in.uv, in.layer & 0x7fffu) * in.color;
    if color.a < 0.01 { discard; }
    if (in.layer & 0x7fffu) + 1u == u32(globals.special.x) {
        let gas = portal_gas(in.rel);
        return apply_fog(vec4(gas.rgb * (0.75 + 0.25 * in.color.rgb), gas.a),
            in.spherical, in.cylindrical);
    }
    return apply_fog(color, in.spherical, in.cylindrical);
}

// Sky (SkyRenderer): the 10-vertex sky disc fan at y = 16 (radius 512) in
// the sky colour with sky fog (sky.fsh: fog 0..FogSkyEnd on the spherical
// distance, a step at FogSkyEnd on the cylindrical one), evaluated per pixel
// along the view ray with the fan's interpolated vertex distances; the
// clear colour (fog colour) elsewhere; the sunrise/sunset fan blended over
// it; the dark disc (y = -4) below the horizon when the camera is low.
struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}
@vertex fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    let p = vec2(f32(i32(i & 1u) * 4 - 1), f32(i32(i >> 1u) * 4 - 1));
    var out: SkyOut;
    out.clip = vec4(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}
fn view_ray(ndc: vec2<f32>) -> vec3<f32> {
    return normalize((globals.inverse_view * vec4(ndc.x * globals.misc.z * globals.misc.w,
        ndc.y * globals.misc.w, -1.0, 0.0)).xyz);
}
const PI: f32 = 3.14159265;
// Interpolated (spherical, cylindrical) distance where the ray meets a disc
// fan at height y, or negative when it misses.
fn disc_distance(d: vec3<f32>, y: f32) -> vec2<f32> {
    if d.y * y <= 0.0 { return vec2(-1.0); }
    let p = d * (y / d.y);
    let h = length(p.xz);
    let phi = atan2(p.z, p.x);
    let sector = PI / 4.0;
    let delta = phi - (floor(phi / sector) * sector + sector * 0.5);
    let edge = 512.0 * cos(sector * 0.5) / cos(delta);
    if h > edge { return vec2(-1.0); }
    let s = h / edge;
    let ay = abs(y);
    let rim = sqrt(512.0 * 512.0 + y * y);
    return vec2(ay + s * (rim - ay), ay + s * (512.0 - ay));
}
fn sky_fog(color: vec3<f32>, dist: vec2<f32>) -> vec3<f32> {
    let end = globals.fog_ends.x;
    let f = max(linear_fog(dist.x, 0.0, end), linear_fog(dist.y, end, end));
    return mix(color, globals.fog_color.rgb, f * globals.fog_color.a);
}
// Weight of the fan centre where the ray crosses the sunrise fan (0 when missed):
// Moller-Trumbore from the origin along d, over normals built once a frame
// (gfx/sunrise_fan.rs). The barycentrics are apex/s, k1/s and -k0/s; s is -det.
fn sunrise_weight(d: vec3<f32>) -> f32 {
    if globals.sunrise.a <= 0.001 { return 0.0; }
    var best = 0.0;
    var k0 = dot(d, globals.sunrise_fan[16].xyz);
    for (var i = 0; i < 16; i++) {
        let triangle = globals.sunrise_fan[i];
        let k1 = dot(d, globals.sunrise_fan[16 + (i + 1) % 16].xyz);
        // A hit needs s on the side of w, so k0 and k1 on either side of 0.
        if k0 * triangle.w <= 0.0 && k1 * triangle.w >= 0.0 {
            let apex = dot(d, triangle.xyz);
            let s = apex + k1 - k0;
            if abs(s) >= 1e-6 && s * triangle.w > 0.0 && apex * s >= 0.0 && k1 * s >= 0.0
                && k0 * s <= 0.0 {
                best = max(best, apex / s);
            }
        }
        k0 = k1;
    }
    return best;
}
// The sky's colour along the view ray `d` (a shader pack's sky stage gets it as its input:
// gfx/native_sky.rs builds that stage from this section).
fn sky_along_ray(d: vec3<f32>) -> vec3<f32> {
    var color = globals.fog_color.rgb;
    let dist = disc_distance(d, 16.0);
    if dist.x >= 0.0 {
        color = sky_fog(globals.sky_color.rgb, dist);
    }
    let w = sunrise_weight(d);
    if w > 0.0 {
        color = mix(color, globals.sunrise.rgb, w * globals.sunrise.a);
    }
    return color;
}
@fragment fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    return vec4(sky_along_ray(view_ray(in.ndc)), 1.0);
}
@fragment fn fs_dark_disc(in: SkyOut) -> @location(0) vec4<f32> {
    let d = view_ray(in.ndc);
    let dist = disc_distance(d, -4.0);
    if dist.x < 0.0 { discard; }
    return vec4(sky_fog(vec3(0.0), dist), 1.0);
}

// rendertype_lines.vsh: screen-space stroke and camera-space depth separation.
fn outline_clip(vertex: u32, start: vec3<f32>, end: vec3<f32>, width: f32) -> vec4<f32> {
    let position = select(start, end, vertex >= 2u);
    let normal = normalize(end - start);
    let view_shrink = 1.0 - 1.0 / 256.0;
    let line_start = globals.view_proj * vec4((position * globals.outline_params.w) * view_shrink, 1.0);
    let line_end = globals.view_proj * vec4(((position + normal) * globals.outline_params.w) * view_shrink, 1.0);
    let ndc1 = line_start.xyz / line_start.w;
    let ndc2 = line_end.xyz / line_end.w;
    let screen = globals.outline_params.xy;
    let direction = normalize((ndc2.xy - ndc1.xy) * screen);
    var offset = vec2(-direction.y, direction.x) * width / screen;
    if offset.x < 0.0 { offset = -offset; }
    if vertex % 2u != 0u { offset = -offset; }
    return vec4((ndc1 + vec3(offset, 0.0)) * line_start.w, line_start.w);
}
struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) spherical: f32,
    @location(1) cylindrical: f32,
    @location(2) color: vec4<f32>,
}
@vertex fn vs_line(@builtin(vertex_index) vertex: u32,
    @location(0) start: vec3<f32>, @location(1) end: vec3<f32>,
    @location(2) argb: vec4<f32>, @location(3) width: f32) -> LineOut {
    let position = select(start, end, vertex >= 2u);
    var out: LineOut;
    out.clip = outline_clip(vertex, start, end, width);
    out.spherical = length(position);
    out.cylindrical = max(length(position.xz), abs(position.y));
    // ARGB in a little-endian u32 reads as B, G, R, A.
    out.color = argb.zyxw;
    return out;
}
@fragment fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return apply_fog(in.color, in.spherical, in.cylindrical);
}
// Gizmo lines (rendertype_lines with a per-vertex colour and LineWidth in
// pixels; no layering transform).
struct DebugLineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) spherical: f32,
    @location(1) cylindrical: f32,
    @location(2) color: vec4<f32>,
}
@vertex fn vs_debug_line(@builtin(vertex_index) vertex: u32,
    @location(0) start: vec3<f32>, @location(1) end: vec3<f32>,
    @location(2) argb: vec4<f32>, @location(3) width: f32) -> DebugLineOut {
    let position = select(start, end, vertex >= 2u);
    let view_shrink = 1.0 - 1.0 / 256.0;
    let line_start = globals.view_proj * vec4(position * view_shrink, 1.0);
    let line_end = globals.view_proj * vec4((position + (end - start)) * view_shrink, 1.0);
    let ndc1 = line_start.xyz / line_start.w;
    let ndc2 = line_end.xyz / line_end.w;
    let screen = globals.outline_params.xy;
    let direction = normalize((ndc2.xy - ndc1.xy) * screen);
    var offset = vec2(-direction.y, direction.x) * width / screen;
    if offset.x < 0.0 { offset = -offset; }
    if vertex % 2u != 0u { offset = -offset; }
    var out: DebugLineOut;
    out.clip = vec4((ndc1 + vec3(offset, 0.0)) * line_start.w, line_start.w);
    out.spherical = length(position);
    out.cylindrical = max(length(position.xz), abs(position.y));
    // ARGB in a little-endian u32 reads as B, G, R, A.
    out.color = argb.zyxw;
    return out;
}
@fragment fn fs_debug_line(in: DebugLineOut) -> @location(0) vec4<f32> {
    return apply_fog(in.color, in.spherical, in.cylindrical);
}
