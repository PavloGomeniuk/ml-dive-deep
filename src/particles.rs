use crate::combat::xorshift;

#[derive(Clone, Copy, Default)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub life: f32,
    pub max_life: f32,
    pub active: bool,
}

pub struct ParticlePool {
    pub particles: [Particle; 64],
    next: usize,
}

impl ParticlePool {
    pub fn new() -> Self {
        ParticlePool {
            particles: [Particle::default(); 64],
            next: 0,
        }
    }

    pub fn spawn_burst(&mut self, x: f32, y: f32, count: usize, rng: &mut u32) {
        for _ in 0..count {
            let idx = self.next;
            self.next = (self.next + 1) % 64;

            let angle_raw = xorshift(rng);
            let angle = (angle_raw % 628) as f32 / 100.0;
            let speed = 40.0 + (xorshift(rng) % 80) as f32;

            let jitter_x = ((xorshift(rng) % 20) as f32) - 10.0;
            let jitter_y = ((xorshift(rng) % 20) as f32) - 10.0;

            self.particles[idx] = Particle {
                x: x + jitter_x,
                y: y + jitter_y,
                vx: angle.cos() * speed,
                vy: angle.sin() * speed,
                life: 0.6,
                max_life: 0.6,
                active: true,
            };
        }
    }

    pub fn update(&mut self, dt: f32) {
        for p in self.particles.iter_mut() {
            if !p.active {
                continue;
            }
            p.vy += 80.0 * dt; // gravity
            p.vx *= 0.95_f32.powf(dt * 60.0); // friction
            p.vy *= 0.95_f32.powf(dt * 60.0);
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.life -= dt;
            if p.life <= 0.0 {
                p.active = false;
            }
        }
    }
}
