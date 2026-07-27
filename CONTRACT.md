# BLACKSITE — Module Contract (READ FIRST, FOLLOW EXACTLY)

Target: a first-person shooter in Three.js whose **rendered image and game feel** stand next to
Call of Duty: Modern Warfare II/III. Art direction: late-afternoon golden hour, dusty
Middle-Eastern urban compound ("Blacksite"), warm key light / cool sky bounce, heavy
atmospheric perspective, filmic grade.

## Hard rules

1. **You own exactly ONE file.** Never create, edit, or delete any other file. Other agents are
   editing other files in the same repo at the same time. Touching their file destroys their work.
2. **Never edit** `src/main.js`, `src/core/Config.js`, `src/core/Bus.js`, `index.html`, `CONTRACT.md`.
3. Export the named class the contract specifies: `export class X { constructor(game) {} async init() {}
   update(dt, t) {} resize(w, h) {} }`. `init` may be async. All four are optional except the constructor
   and the export name.
4. `game` gives you: `game.THREE`, `game.scene`, `game.camera`, `game.config`, `game.bus`,
   `game.registry`, plus every module constructed *before* yours in `main.js`'s step list
   (e.g. `game.materials`, `game.level`). Modules constructed *after* you must be accessed lazily
   inside `update()`, never in the constructor.
5. Communicate across modules through `game.bus` events listed in `src/core/Bus.js`. Do not import
   sibling modules directly.
6. Read tunables from `game.config`. Do not hardcode a value that already has a Config key.
7. `import * as THREE from 'three'` and `import { X } from 'three/addons/...'` both work (Vite +
   importmap resolve them). three r180 is installed. Check `node_modules/three/examples/jsm/` for the
   exact addon path before importing it — a wrong path is a hard boot failure.
8. **No external asset downloads.** No CDN, no .glb/.hdr/.jpg fetches. Every texture, mesh, sound and
   animation is generated procedurally in code. This is a hard constraint — a 404 is a broken build.
9. Must run at **≥60 fps at 1920×1080** on an integrated GPU budget. Instance aggressively, share
   geometry/materials, cap draw calls, pool everything. No per-frame allocation in `update()`.
10. Your file must be self-contained and must not throw if a module you'd like to use is missing —
    guard with `?.`.

## Quality bar (the critic will judge against real CoD screenshots)

- No flat/untextured surfaces. Everything gets albedo + normal + roughness + AO, with grime,
  edge wear, and large-scale variation breakup so tiling is invisible.
- No pure blacks, no blown highlights. Shadows keep colored ambient bounce.
- Silhouettes read at distance; surfaces read up close. Add trim, bevels, and greeble — CoD
  environments never have a bare untrimmed box.
- Motion is animated on curves with weight, never linear lerps.

## Debug API (needed by the automated screenshot critic — implement if your module owns it)

`window.__GAME` is the game. The screenshot tool calls:
- `__GAME.controller.teleport(x, y, z, yaw, pitch)` — move the player camera (Controller owns).
- `__GAME.weapon.setAds(bool)` — force ADS state (Weapon owns).
- `__GAME.hud.setDemoState()` — populate HUD with representative values (HUD owns).
- `__GAME.config.debug` — flags.

## Verifying your work

Dev server: `npx vite --port 5173` from the project root (may already be running).
Screenshot: `node tools/shoot.mjs <name> [--cam preset]` writes `shots/<name>-*.png`.
Console errors are captured to `window.__ERRORS` and printed by the shoot tool — **a build with any
console error is a failed build.** Always run the shoot tool before you report done, and iterate
until your module renders clean.
