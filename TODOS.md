# TODOS

## Post-Hackathon

### Seeded deterministic RNG
**What:** Replace `getrandom` (non-deterministic) with a seeded RNG (e.g., `SmallRng` from
the `rand` crate or a hand-rolled xorshift with explicit seed).
**Why:** Non-deterministic randomness makes bugs non-reproducible. A seeded RNG allows
replays, deterministic debugging, and eventually replay-based testing.
**How to apply:** Add seed parameter to `GameState::new(seed: u64)`. Seed can be
random on first run but stored in game state for debug output.
**Depends on:** Game logic stable (post-hackathon).

### wasm-opt post-build size reduction
**What:** Add `wasm-opt -Oz` step to the build chain after `wasm-bindgen`.
**Why:** binaryen's wasm-opt typically reduces WASM binary another 20-30% beyond
Rust's own `opt-level="z"`. Combined with current release profile, could reach <300 KB.
**How:** `apt install binaryen` or `brew install binaryen`, then add to build script:
`wasm-opt -Oz static/pkg/game_bg.wasm -o static/pkg/game_bg.wasm`
**Not npm.** System package only.
**Depends on:** Game at stable size (Day 3 or post-hackathon).

### Demon Eye enemy (Type B)
**What:** Ranged enemy. 15 HP, aggro radius 300px, ranged projectile attack, cooldown 2s,
damage 6-10, move speed 40px/s. Color #aa2222 (blood red).
**Why:** Second enemy type adds combat variety and tests the enemy state machine
for non-melee behaviors (projectile spawn, range check different from melee range).
**How:** Add `EnemyKind::DemonEye` variant, projectile struct in entities.rs,
projectile movement + collision in combat.rs.
**Depends on:** Day 2 combat systems stable, dungeon generator working.
