//! Ray-marched fire (`gfx::atmosphere`): the fire blocks near the camera, which the atmosphere
//! pass draws as volumes of flame under a plume of smoke instead of flat sprites. Nothing is
//! gathered while the portal, lava and fire look is off.
use super::Game;
use glam::DVec3;
use horizon_client::gfx::{FIRE_VOLUME_RADIUS, atmosphere::MAX_FIRES};

/// How often the blocks around the camera are looked through again, in ticks.
const RESCAN_TICKS: u64 = 4;

impl Game<'_> {
    /// The fire blocks within [`FIRE_VOLUME_RADIUS`] of the camera, nearest first (at most
    /// [`MAX_FIRES`]): camera-relative corner, and 1 for fire, 2 for soul fire. Also tells the
    /// particles which smoke the plumes replace.
    pub(super) fn gather_fires(&mut self, camera: DVec3) -> Vec<[f32; 4]> {
        if !self.settings.horizon.portal_lava {
            self.fires.clear();
            self.fire_scan = None;
            self.particles.hidden_smoke.clear();
            return Vec::new();
        }
        let at = [camera.x, camera.y, camera.z].map(|v| (v / 2.0).floor() as i32);
        let stale = self
            .fire_scan
            .is_none_or(|(key, tick)| key != at || self.ticks.saturating_sub(tick) >= RESCAN_TICKS);
        if stale {
            let r = FIRE_VOLUME_RADIUS as i32;
            let (cx, cy, cz) = (
                camera.x.floor() as i32,
                camera.y.floor() as i32,
                camera.z.floor() as i32,
            );
            let mut found: Vec<(f64, [f64; 4])> = Vec::new();
            for x in cx - r..=cx + r {
                for z in cz - r..=cz + r {
                    for y in cy - r..=cy + r {
                        let name = &self.res.facts().get(self.world.block(x, y, z)).name;
                        let kind = match &**name {
                            "minecraft:fire" => 1.0,
                            "minecraft:soul_fire" => 2.0,
                            _ => continue,
                        };
                        let d = (f64::from(x) + 0.5 - camera.x).powi(2)
                            + (f64::from(y) + 0.5 - camera.y).powi(2)
                            + (f64::from(z) + 0.5 - camera.z).powi(2);
                        if d < f64::from(FIRE_VOLUME_RADIUS).powi(2) {
                            found.push((d, [f64::from(x), f64::from(y), f64::from(z), kind]));
                        }
                    }
                }
            }
            found.sort_by(|a, b| a.0.total_cmp(&b.0));
            found.truncate(MAX_FIRES);
            self.fires = found.into_iter().map(|(_, f)| f).collect();
            self.fire_scan = Some((at, self.ticks));
            self.particles.hidden_smoke = self.fires.iter().map(|f| [f[0], f[1], f[2]]).collect();
        }
        self.fires
            .iter()
            .map(|f| {
                [
                    (f[0] - camera.x) as f32,
                    (f[1] - camera.y) as f32,
                    (f[2] - camera.z) as f32,
                    f[3] as f32,
                ]
            })
            .collect()
    }
}
