# DESIGN.md — Medieval WASM RPG

Design system for the Rust/WASM Diablo-like. Pure Canvas 2D. No CSS frameworks.
Every pixel is drawn from `renderer.rs`. This file is the source of truth for
visual decisions — update it when the code changes.

---

## Palette

All colors live in `src/renderer.rs` as `const` strings.

### Environment

| Token | Hex | Usage |
|-------|-----|-------|
| `C_BG` | `#0d0d1a` | Canvas background, outside dungeon |
| `C_WALL` | `#1a1a2e` | Room outer wall fill |
| `C_WALL_EDGE` | `#2a2a4e` | Room wall border (2px stroke) |
| `C_FLOOR` | `#1e1e1e` | Dungeon floor fill |
| `C_FLOOR_GRID` | `#252525` | Floor grid lines (0.5px, 40px cells) |

### Characters

| Token | Hex | Usage |
|-------|-----|-------|
| `C_PLAYER` | `#c8b89a` | Player body + head (tan/leather) |
| `C_PLAYER_SWORD` | `#aaaacc` | Warrior sword (steel blue-grey) |
| `C_ENEMY` | `#8888aa` | Skeleton body + eye socket fill |
| `C_ENEMY_SKULL` | `#ccccee` | Skeleton skull head (lighter) |

New in v2:
- Magician body: `#9988cc` (purple robes, narrower 16×26px)
- Magician hat: `#4444aa`
- Magician staff orb: `#aaaaff` with `shadowColor: #8888ff`
- Warrior helmet: `#888888`
- Warrior shield: `#888888`

### UI / HUD

| Token | Hex | Usage |
|-------|-----|-------|
| `C_HUD_BG` | `#0a0a14` | HUD strip background |
| `C_HUD_BORDER` | `#5a3a15` | HUD top border, orb rings, slot borders |
| `C_GOLD` | `#d4af37` | Gold display, title text, card highlights |
| `C_DAMAGE` | `#ffee44` | Floating damage numbers |
| `C_AGGRO` | `#ff4444` | Aggro indicator dot above enemy |

### Orbs

| Token | Hex | Usage |
|-------|-----|-------|
| `C_HP_ORB` / `C_HP_BAR` | `#cc2222` | HP orb fill, HP bars |
| `C_HP_BAR_BG` | `#330000` | HP bar background |
| `C_MANA_ORB` | `#2244cc` | Mana orb fill |
| `C_PARTICLE` | `#cc2222` | Combat blood particles |

### Screens / Overlays

| Token | Hex | Usage |
|-------|-----|-------|
| `C_OVERLAY_BG` | `rgba(0,0,0,0.75)` | WASM stats overlay bg |
| `C_OVERLAY_TEXT` | `#aaffaa` | WASM stats text (green-white) |
| title bg | `#0d0d1a` | Same as C_BG — consistency |
| title vignette center | `#1a1420` | Radial gradient center for title screen |
| title dividers | `#5a3a15` | Same as C_HUD_BORDER |
| title subtitle | `#5a3a15` | Same bronze tone |
| title flavor text | `#3a3a5a` | Dark blue-grey, barely visible |
| card default border | `#5a3a15` | Class cards, unselected |
| card selected glow | `#d4af37` + `box-shadow: 0 0 15px rgba(212,175,55,0.2)` | Highlighted class card |
| card stats text | `#8888aa` | Same as C_ENEMY — recycled for muted text |
| card ability bg | `#3a3a5a` | Ability box inside card |
| item label | `#c8b89a` | Same as C_PLAYER — neutral tan |
| HUD flash text | `#ffee44` | Same as C_DAMAGE — "nothing to equip" warning |

### Items on Floor

| Item kind | Hex | Shape |
|-----------|-----|-------|
| Sword | `#aaaacc` | Diamond (rotated 10×10px rect) |
| Staff | `#8866aa` | Diamond |
| Tome | `#aa6622` | Diamond |

---

## Typography

Single font everywhere: `'Courier New', monospace`

| Size | Weight | Color | Usage |
|------|--------|-------|-------|
| `bold 48px` | bold | `#cc2222` | YOU DIED screen |
| `bold 16px` | bold | `#ffee44` | Floating damage numbers |
| `bold 14px` | bold | `#d4af37` | Gold counter |
| `bold 11px` | bold | `#aaffaa` | WASM overlay header |
| `18px` | normal | `#888888` | Game over sub-text |
| `11px` | normal | `#aaffaa` | WASM overlay lines |
| `11px` | normal | `#cc4444` / `#4466cc` | HP / MP orb labels |
| `10px` | normal | `#88cc88` | WASM overlay footer |

New in v2 (CSS/canvas):
| Size | Letter-spacing | Color | Usage |
|------|---------------|-------|-------|
| `2.8rem` | `0.25em` | `#d4af37` | Title "MEDIEVAL RPG" |
| `0.9rem` | `0.2em` | `#d4af37` | Character select header |
| `1rem` | — | `#d4af37` | Class name in card |
| `0.75rem` | `0.08em` | `#5a3a15` | Title subtitle |
| `0.7rem` | — | `#8888aa` | Class stat text in card |
| `0.65rem` | — | `#3a3a5a` | "ENTER IF YOU DARE" flavor |
| `0.6rem` | — | `#d4af37` | Door label, item label |
| `0.6rem` | — | `#3a3a5a` | Key hints in HUD |

All uppercase for headers and labels. Sentence case only for flavor text.

---

## Layout

### Canvas

Logical size: **800 × 500px** (never changes — this is the coordinate system)

- Game area: `0, 0` to `800, 400` (HUD_Y = 400)
- HUD strip: `0, 400` to `800, 500` (HUD_H = 100)

Fullscreen: CSS scaling layer only.
```css
canvas:-webkit-full-screen,
canvas:fullscreen {
  width: 100vw;
  height: 100vh;
  object-fit: contain;
}
```

### Dungeon Room (single room, v1)

Room rect: positioned by `dungeon.rs` Room struct
- Floor inset: 20px from room edges (wall thickness)
- Grid: 40px cells, `#252525`, 0.5px lines

### HUD Strip (y=400, h=100)

```
[HP orb]  [slot][slot][slot][slot]  [gold]  [MP orb]
  x=80        center ~380              x=370  x=720
```

- Orbs: radius 38px, centered at y=450
- Ability slots: 40×40px, gap 8px, centered horizontally
- Slot bg: `#111122`, border `#5a3a15` 2px
- HUD border: 3px `#5a3a15` line at y=400

New in v2:
- Equipment slots: 3 × 40×40px right of MP orb
  - Unequipped: `#5a3a15` border
  - Equipped: `#d4af37` border + "E" label top-right
- Ability slot: 44×44px, right of equipment slots
  - On cooldown: gray overlay + seconds remaining in center
- Key hints: `[ E ] equip  [ Space ] ability` — 0.6rem `#3a3a5a`, right of ability slot
- HUD flash text: centered in HUD, `#ffee44`, fades over 60 frames

---

## Components

### Orb (HP / Mana)

```
draw_orb(ctx, cx, cy, radius=38, color, fraction)
```
1. Clip to circle
2. Fill `#111122` background
3. Fill color from bottom up by `fraction`
4. Unclip, stroke `#5a3a15` ring (3px)

### HP Bar (on entity)

Width: 30px, height: 4px, placed at entity bottom + 18px
- Background: `C_HP_BAR_BG` (`#330000`)
- Fill: `C_HP_BAR` (`#cc2222`), scaled by `hp / max_hp`

### Damage Number

- Float upward 2px/frame
- Alpha = `life / 30` (fades over 30 frames)
- Font: `bold 16px monospace`, color `rgba(255,238,68,alpha)`

### Particle

- Circle radius 3px
- Color `rgba(204,34,34,alpha)`, alpha = `life / max_life`
- 8-particle burst on hit, gold burst on loot drop (`#d4af37`)

### Aggro Dot

- Circle radius 4px, `#ff4444`
- Drawn 34px above enemy center when state != Patrolling

### Door Arch (new in v2)

Position: bottom-center of floor, inset 5px from floor bottom edge
- Two vertical pillar rects: 8px wide each
- Arc for doorway top
- Fill: `#3a2a0a`, border: 2px `#5a3a15`, frame: 1px `#d4af37`
- Label: "LOC 2 ▼" or "LOC 1 ▲" — 0.6rem `#d4af37` centered above arch
- Trigger zone: 30px radius around door center

### Item Diamond (new in v2)

- Rotated 45° rect, 10×10px
- Color by kind (see Items table above)
- Label: "[E] {name}" — 0.6rem `#c8b89a`, above diamond, fades after 3s

### Class Card (new in v2)

Size: 280×310px, positioned 80px from canvas edges
- Default: 2px `#5a3a15` border, `#0f0f1e` background
- Hover/selected: 2px `#d4af37` border + `box-shadow: 0 0 15px rgba(212,175,55,0.2)`
- Layout top to bottom:
  1. Sprite area: 120px height — canvas drawing (Warrior or Magician)
  2. Class name: 1rem `#d4af37`
  3. Stats: 0.7rem `#8888aa`
  4. Ability box: `#3a3a5a` bg, bordered, ability name + key hint

---

## Sprite Specs

### Warrior

```
Body:       rect 22×28px, fill #c8b89a (same as C_PLAYER)
Head:       circle r=9px, fill #c8b89a
Helmet:     rect 22×6px on top of head, fill #888888
Sword:      line (x+11,y-5)→(x+28,y-22), 3px, #aaaacc
Crossguard: horizontal 8px at sword midpoint
Shield:     rect 8×10px left of body, fill #888888
```

### Magician

```
Body:  rect 16×26px, fill #9988cc (narrower, purple robes)
Head:  circle r=8px, fill #c8b89a
Hat:   trapezoid/triangle above head, fill #4444aa (pointed top)
Staff: vertical line (x-14,y-28)→(x-14,y+20), 2px, #8888aa
Orb:   circle r=4px at staff top, fill #aaaaff, shadowColor #8888ff
```

---

## Screen Flows

```
[ TITLE ]
  click "NEW GAME"
    → [ CHARACTER SELECT ]
      click card → highlight
      ENTER or second click → [ PLAYING ]
      ESC → [ TITLE ]

[ PLAYING ]
  player dies → [ DEAD ]
    click → [ TITLE ]
  walk to door → [ LOCATION 2 ]
    walk back → [ PLAYING / Location 1 ]

[ TITLE ]
  ESC → no-op
```

### Title Screen

- Background: `#0d0d1a` + radial gradient vignette (center `#1a1420`)
- Title: "MEDIEVAL RPG" — 2.8rem, `#d4af37`, letter-spacing 0.25em
- Subtitle: "built with Rust + WebAssembly · zero npm" — 0.75rem, `#5a3a15`
- CTA button: "⚔ NEW GAME" — `#d4af37` border, transparent bg, hover `rgba(212,175,55,0.1)`
- Dividers: 1px `#5a3a15`, 400px wide, centered above/below title
- Flavor text: "☠ ENTER IF YOU DARE ☠" — 0.65rem, `#3a3a5a`, bottom

### Character Select

- Header: "CHOOSE YOUR FATE" — 0.9rem, `#d4af37`, letter-spacing 0.2em
- Divider: 1px `#5a3a15` below header
- Two class cards side-by-side (see Class Card component above)
- Hint bar: "click to select · ENTER to begin · ESC for title"

---

## Room Transition

16-frame black fade (~270ms at 60fps):
1. Frames 16→9: alpha `(transition_fade - 8) / 8.0` (black overlay fades in)
2. Frame 8: switch room, reposition player at opposite door
3. Frames 8→1: alpha fades from 1.0 to 0.0 (fade in to new room)

Field: `transition_fade: u8` on `GameState`, decrements each tick.

---

## Fullscreen Button

Position: top-right of `#canvas-wrap` div
- Icon: `⛶`
- Size: 28×28px
- Background: `rgba(0,0,0,0.5)`
- Border: 1px `#5a3a15`
- `aria-label="Toggle fullscreen"`

Canvas element: `tabIndex="-1"` to prevent tab-trapping.

---

## Key Bindings

| Key | Action |
|-----|--------|
| Left-click | Move to position or attack enemy |
| `E` | Equip nearest item on floor |
| `Space` | Use class ability (Cleave / Frost Nova) |
| `ESC` | Return to title screen |
| `I` | (reserved: inventory) |

---

## Abilities

### Warrior — Cleave

AoE melee hit, 80px radius around player, 4s cooldown.
All enemies in radius take damage roll.

### Magician — Frost Nova

AoE freeze, 70px radius, 3s cooldown.
Slows/stops enemies in radius for 2s.

---

## Sound Design (Web Audio API, no files)

All sounds generated at runtime using oscillators/noise. No audio files needed.

| Event | Wave | Frequency | Duration |
|-------|------|-----------|----------|
| Hit | square | 200Hz → 50Hz | 80ms |
| Death | noise + low tone | — | 300ms |
| Equip | ascending arpeggio | C4→E4→G4 | 150ms |
| Room transition | chord swell | — | 270ms (matches fade) |

Master gain: 0.3. AudioContext initialized on first user interaction.

---

## Anti-patterns

Things explicitly NOT done:

- No WebGL. Canvas 2D only.
- No CSS framework. All game visuals drawn via canvas API.
- No font loading. `'Courier New', monospace` only — it's a system font.
- No image assets. Every sprite is drawn with `fillRect` + `arc` + lines.
- No external color variables in JS/CSS. All colors are `const` in `renderer.rs`.
- No mana system (yet). Both classes use cooldowns only. Mana = future feature.
- No item rarities (yet). Weapon types only: Sword, Staff, Tome.
