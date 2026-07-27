// ============================================================================
// BLACKSITE — src/fx/Particles.js
// Owner: Particles agent.  ALL particle FX in the game.
//
// Design
// ------
// Every class is ONE InstancedBufferGeometry driven by ONE interleaved instance
// buffer and a custom shader.  The simulation is analytic and runs entirely in
// the vertex shader:
//
//     v(t) = v0 * e^(-k t) + g * (1 - e^(-k t)) / k
//     p(t) = p0 + v0 * (1 - e^(-k t)) / k + g * (t - (1 - e^(-k t)) / k) / k
//
// so a particle costs 24 floats of upload ONCE at spawn and nothing at all for
// the rest of its life.  Dead particles collapse to an off-clip vertex, and an
// idle pool drops its instanceCount to 0 so it costs a skipped draw call.
//
// Soft particles: smoke / fire / blood mist fade against a scene depth buffer so
// they never show a hard intersection line.  Depth comes from PostFX if it
// publishes one (game.post.depthTexture), otherwise from a half-res depth-only
// prepass this module renders ITSELF — and only on frames where soft particles
// are alive, so an idle scene pays nothing.  The prepass' view-projection matrix
// travels with the depth texture into the shader, so the screen-space lookup is
// exact even though the camera moves between the prepass and the beauty pass.
//
// Public API (collaborators — this is stable)
// ------------------------------------------
//   game.particles.impact(point, normal, surface, scale)
//        surface: 'concrete'|'metal'|'sand'|'wood'|'glass'|'flesh' (aliases ok)
//   game.particles.explosion(point, radius, power)
//   game.particles.muzzle(position, direction, scale)
//   game.particles.footstep(position, surface, speed)
//   game.particles.blood(point, normal, amount, headshot)
//   game.particles.flashLight(position, colorHex, intensity, seconds, distance)
//   game.particles.smokePuff(px,py,pz, vx,vy,vz, o)     low level, allocation free
//   game.particles.sparkBurst(px,py,pz, nx,ny,nz, count, o)
//   game.particles.debrisBurst(px,py,pz, nx,ny,nz, count, o)
//   game.particles.setQuality('low'|'high'|'ultra')
//   game.particles.setWind(x, z)
//   game.particles.test()        fire one of everything in front of the camera
//   game.particles.stats  -> { live, softPasses, pools:{...} }
//
// Bus events consumed: 'impact', 'shot', 'explosion', 'footstep', 'hit', 'kill'.
// Everything is guarded with ?. — a missing collaborator never throws.
// ============================================================================

import * as THREE from 'three';

/* ==========================================================================
 *  build-time noise helpers (never called after init)
 * ========================================================================== */

function mulberry32(a) {
  a = (a >>> 0) || 0x9e3779b9;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function ihash(x, y, s) {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(s | 0, 1442695041);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function vnoise(x, y, s) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const a = ihash(ix, iy, s), b = ihash(ix + 1, iy, s);
  const c = ihash(ix, iy + 1, s), d = ihash(ix + 1, iy + 1, s);
  const top = a + (b - a) * ux;
  const bot = c + (d - c) * ux;
  return top + (bot - top) * uy;
}

function fbm(x, y, oct, gain, s) {
  let amp = 1, sum = 0, norm = 0, f = 1;
  for (let i = 0; i < oct; i++) {
    sum += vnoise(x * f, y * f, s + i * 619) * amp;
    norm += amp; amp *= gain; f *= 2;
  }
  return sum / norm;
}

const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const sstep = (e0, e1, x) => { let t = (x - e0) / (e1 - e0); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); };
const rnd = Math.random;
const rr = (a, b) => a + (b - a) * Math.random();

/* ==========================================================================
 *  shared GLSL
 * ========================================================================== */

const GLSL_COMMON = /* glsl */`
#define BS_TAU 6.283185307179586

vec2 bsRot( vec2 p, float a ) {
  float s = sin( a ), c = cos( a );
  return vec2( p.x * c - p.y * s, p.x * s + p.y * c );
}

vec3 bsIntegrate( vec3 p0, vec3 v0, float k, vec3 g, float age ) {
  if ( k > 0.0005 ) {
    float e = exp( -k * age );
    float a = ( 1.0 - e ) / k;
    return p0 + v0 * a + g * ( age - a ) / k;
  }
  return p0 + v0 * age + 0.5 * g * age * age;
}

vec3 bsVelocity( vec3 v0, float k, vec3 g, float age ) {
  if ( k > 0.0005 ) {
    float e = exp( -k * age );
    return v0 * e + g * ( 1.0 - e ) / k;
  }
  return v0 + g * age;
}
`;

const GLSL_SOFT_V = /* glsl */`
uniform mat4 uDepthVP;
varying vec4 vDepthClip;
`;

const GLSL_SOFT_F = /* glsl */`
#include <packing>
uniform sampler2D tDepth;
uniform vec2 uDepthNearFar;
uniform float uHasDepth;
varying vec4 vDepthClip;

float bsSoft( float softness ) {
  if ( uHasDepth < 0.5 || vDepthClip.w <= 0.0 ) return 1.0;
  vec2 duv = vDepthClip.xy / vDepthClip.w * 0.5 + 0.5;
  if ( duv.x < 0.0 || duv.x > 1.0 || duv.y < 0.0 || duv.y > 1.0 ) return 1.0;
  float d = texture2D( tDepth, duv ).x;
  if ( d >= 0.99999 ) return 1.0;
  float sceneZ = perspectiveDepthToViewZ( d, uDepthNearFar.x, uDepthNearFar.y );
  float partZ  = -vDepthClip.w;
  return clamp( ( partZ - sceneZ ) / max( softness, 1e-3 ), 0.0, 1.0 );
}
`;

/* ==========================================================================
 *  quality presets
 * ========================================================================== */

const QUALITY = {
  low:   { smoke: 260,  fire: 140, spark: 900,  debris: 480,  ring: 12, motes: 800,  sand: 300,  litter: 14, burst: 0.45, soft: false },
  high:  { smoke: 640,  fire: 320, spark: 2600, debris: 1100, ring: 24, motes: 1800, sand: 800,  litter: 30, burst: 0.75, soft: true  },
  ultra: { smoke: 1100, fire: 560, spark: 4600, debris: 2000, ring: 32, motes: 3000, sand: 1500, litter: 48, burst: 1.00, soft: true  },
};

// interleaved instance layout, 24 floats:
//   0..3   iSpawn  ( px, py, pz, spawnTime )
//   4..7   iVel    ( vx, vy, vz, life )
//   8..11  iParam  ( a, b, seed, drag )
//   12..15 iColA   ( r, g, b, alpha )
//   16..19 iColB   ( r, g, b, alpha )
//   20..23 iMisc   ( gravityMul, rotSpeed, extra, extra2 )
const STRIDE = 24;
const GRAV = -9.81;

const SURFACE_ALIAS = {
  concrete: 'concrete', plaster: 'concrete', brick: 'concrete', stone: 'concrete',
  asphalt: 'concrete', rock: 'concrete', cement: 'concrete',
  metal: 'metal', rustmetal: 'metal', steel: 'metal', gunmetal: 'metal', iron: 'metal',
  sand: 'sand', dirt: 'sand', dust: 'sand', ground: 'sand', gravel: 'sand', soil: 'sand',
  wood: 'wood', plank: 'wood', crate: 'wood', foliage: 'wood',
  glass: 'glass', window: 'glass',
  flesh: 'flesh', body: 'flesh', head: 'flesh', skin: 'flesh',
  fabric: 'wood', cloth_tan: 'wood', rubber: 'concrete', polymer: 'concrete',
};

export class Particles {

  constructor(game) {
    this.g = game;
    this.THREE = THREE;

    const q = game?.config?.quality;
    this.quality = QUALITY[q] ? q : 'ultra';
    this.preset = QUALITY[this.quality];
    this.burstScale = this.preset.burst;

    this.group = new THREE.Group();
    this.group.name = 'fx_particles';
    this.group.matrixAutoUpdate = false;
    this.group.frustumCulled = false;

    this.wind = new THREE.Vector3(0.86, 0, 0.34);

    this._t = 0;
    this._ready = false;
    this._sys = {};
    this._systems = [];
    this._ambient = [];
    this._unsub = [];

    this.stats = { live: 0, softPasses: 0, pools: {} };

    // ---- scratch (zero allocation in update / spawn) ----------------------
    this._n = new THREE.Vector3(0, 1, 0);
    this._t1 = new THREE.Vector3(1, 0, 0);
    this._t2 = new THREE.Vector3(0, 0, 1);
    this._dir = new THREE.Vector3();
    this._tmp = new THREE.Vector3();
    this._sunView = new THREE.Vector3(0, 0, 1);
    this._sunDir = new THREE.Vector3(-0.62, 0.28, -0.73);
    this._tanA = new THREE.Vector3(1, 0, 0);
    this._tanB = new THREE.Vector3(0, 0, 1);
    this._depthVP = new THREE.Matrix4();
    this._dbSize = new THREE.Vector2(1920, 1080);

    // ---- shared uniform objects (one object, referenced by every material) --
    this.U = {
      uTime:        { value: 0 },
      uWind:        { value: this.wind },
      uSunDir:      { value: this._sunDir },
      uSunView:     { value: this._sunView },
      uSunColor:    { value: new THREE.Color(1.00, 0.84, 0.62) },
      uSkyColor:    { value: new THREE.Color(0.42, 0.55, 0.76) },
      uPxScale:     { value: 0.0018 },
      tDepth:        { value: null },
      uDepthNearFar: { value: new THREE.Vector2(0.05, 900) },
      uHasDepth:     { value: 0 },
      uDepthVP:      { value: this._depthVP },
      uTanA:   { value: this._tanA },
      uTanB:   { value: this._tanB },
      uGroundY: { value: -0.30 },
    };

    // ---- depth prepass state ---------------------------------------------
    this._softEnabled = this.preset.soft;
    this._depthRT = null;
    this._depthMat = null;
    this._depthOk = true;
    this._hidden = new Array(2048);
    this._hiddenCount = 0;
    this._hiddenFrame = -1e9;
    this._collect = this._collect.bind(this);
    this._viewW = 1920;
    this._viewH = 1080;

    this._lights = [];
  }

  /* ======================================================================== *
   *  LIFECYCLE
   * ======================================================================== */

  async init() {
    const step = () => new Promise(r => setTimeout(r, 0));

    this._texSmoke = this._buildSmokeAtlas(this.quality === 'low' ? 128 : 256);
    await step();
    this._texDebris = this._buildDebrisAtlas(128);
    this._texLitter = this._buildLitterAtlas(64);
    await step();

    this._buildSmokeSystem();
    this._buildFireSystem();
    this._buildSparkSystem();
    this._buildDebrisSystem();
    this._buildRingSystem();
    await step();

    this._buildMotes();
    this._buildWindSand();
    this._buildLitter();
    await step();

    this._buildFlashLights();
    this._buildDepthResources();

    this.g?.scene?.add(this.group);
    this._syncAtmosphere();
    this._ready = true;
    this._bind();
  }

  _bind() {
    const bus = this.g?.bus;
    if (!bus) return;
    const on = (evt, fn) => { const off = bus.on(evt, fn); if (typeof off === 'function') this._unsub.push(off); };

    on('impact', (e) => {
      if (!e?.point) return;
      const surf = e.surface || this.g?.materials?.surfaceFor?.(e.material) || 'concrete';
      this.impact(e.point, e.normal, surf, e.scale ?? 1);
    });

    on('shot', (e) => {
      if (!e?.origin || !e?.dir) return;
      this.muzzle(e.origin, e.dir, e.scale ?? 1);
    });

    on('explosion', (e) => {
      if (!e?.point) return;
      this.explosion(e.point, e.radius ?? 4, e.power ?? 1);
    });

    on('footstep', (e) => {
      if (!e) return;
      const p = e.point || this.g?.controller?.pos || this.g?.camera?.position;
      if (!p) return;
      this.footstep(p, e.surface || 'sand', e.speed ?? 3, !e.point);
    });

    on('hit', (e) => {
      if (!e?.point) return;
      this.blood(e.point, e.normal, Math.min(2, (e.damage ?? 30) / 30), !!e.headshot);
    });

    on('kill', (e) => {
      const p = e?.point || e?.enemy?.position;
      if (!p) return;
      this.blood(p, null, 1.5, !!e?.headshot);
    });
  }

  dispose() {
    for (const off of this._unsub) { try { off(); } catch (e) { /* bus gone */ } }
    this._unsub.length = 0;
    for (const s of this._systems.concat(this._ambient)) {
      s.mesh?.geometry?.dispose();
      s.mesh?.material?.dispose();
    }
    this._texSmoke?.dispose();
    this._texDebris?.dispose();
    this._texLitter?.dispose();
    this._depthRT?.depthTexture?.dispose();
    this._depthRT?.dispose();
    this._depthMat?.dispose();
    for (const L of this._lights) L.light.parent?.remove(L.light);
    this.group.parent?.remove(this.group);
  }

  /* ======================================================================== *
   *  TEXTURES  (procedural, generated once at init — no file loads)
   * ======================================================================== */

  _mkTex(data, w, h, srgb) {
    const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.generateMipmaps = true;
    t.anisotropy = Math.min(4, this.g?.config?.render?.maxAnisotropy ?? 4);
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.needsUpdate = true;
    return t;
  }

  /**
   * 2x2 atlas of smoke puffs.  RGB = a baked normal (so the low sun genuinely
   * rakes across the volume and the sun side silvers), A = density.
   */
  _buildSmokeAtlas(TILE) {
    const N = TILE * 2;
    const data = new Uint8Array(N * N * 4);
    const dens = new Float32Array(TILE * TILE);

    for (let ty = 0; ty < 2; ty++) {
      for (let tx = 0; tx < 2; tx++) {
        const seed = 7001 + ty * 1301 + tx * 977;
        const prng = mulberry32(seed);

        const nl = 5 + ((seed >> 3) % 4);
        const lx = new Float32Array(nl), ly = new Float32Array(nl), lr = new Float32Array(nl);
        for (let i = 0; i < nl; i++) {
          const a = prng() * Math.PI * 2, d = Math.pow(prng(), 0.7) * 0.19;
          lx[i] = 0.5 + Math.cos(a) * d;
          ly[i] = 0.5 + Math.sin(a) * d;
          lr[i] = 0.17 + prng() * 0.15;
        }

        const ns = 4.5 + (seed % 3);
        for (let y = 0; y < TILE; y++) {
          const v = (y + 0.5) / TILE;
          for (let x = 0; x < TILE; x++) {
            const u = (x + 0.5) / TILE;
            let d = 0;
            for (let i = 0; i < nl; i++) {
              const dx = u - lx[i], dy = v - ly[i];
              const r = Math.sqrt(dx * dx + dy * dy) / lr[i];
              const c = 1 - sstep(0.22, 1.0, r);
              if (c > d) d = c;
            }
            const n = fbm(u * ns, v * ns, 5, 0.55, seed);
            d *= 0.28 + 1.10 * n;
            const w = fbm(u * ns * 2.7 + 11, v * ns * 2.7 - 7, 3, 0.5, seed + 41);
            d -= (1 - n) * 0.20 * w;
            const rx = (u - 0.5) * 2, ry = (v - 0.5) * 2;
            d *= sstep(1.0, 0.52, Math.sqrt(rx * rx + ry * ry));
            dens[y * TILE + x] = clamp01(d);
          }
        }

        const K = TILE * 0.055;
        for (let y = 0; y < TILE; y++) {
          for (let x = 0; x < TILE; x++) {
            const xm = x > 0 ? x - 1 : 0, xp = x < TILE - 1 ? x + 1 : TILE - 1;
            const ym = y > 0 ? y - 1 : 0, yp = y < TILE - 1 ? y + 1 : TILE - 1;
            const gx = dens[y * TILE + xp] - dens[y * TILE + xm];
            const gy = dens[yp * TILE + x] - dens[ym * TILE + x];
            let nx = -gx * K, ny = -gy * K;
            let inv = 1 / Math.sqrt(nx * nx + ny * ny + 1);
            nx *= inv; ny *= inv;
            const nz = inv;

            const sx = ((x + 0.5) / TILE - 0.5) * 2 * 0.92;
            const sy = ((y + 0.5) / TILE - 0.5) * 2 * 0.92;
            const sz = Math.sqrt(Math.max(0.02, 1 - sx * sx - sy * sy));
            const m = 0.58;
            let fx = nx * (1 - m) + sx * m;
            let fy = ny * (1 - m) + sy * m;
            let fz = nz * (1 - m) + sz * m;
            inv = 1 / Math.max(1e-5, Math.sqrt(fx * fx + fy * fy + fz * fz));
            fx *= inv; fy *= inv; fz *= inv;

            const o = (((ty * TILE + y) * N) + (tx * TILE + x)) * 4;
            data[o]     = ((fx * 0.5 + 0.5) * 255) | 0;
            data[o + 1] = ((fy * 0.5 + 0.5) * 255) | 0;
            data[o + 2] = ((fz * 0.5 + 0.5) * 255) | 0;
            data[o + 3] = (dens[y * TILE + x] * 255) | 0;
          }
        }
      }
    }
    return this._mkTex(data, N, N, false);
  }

  /** 2x2 atlas: 0 rock chip, 1 wood splinter, 2 glass shard, 3 droplet/grain. */
  _buildDebrisAtlas(TILE) {
    const N = TILE * 2;
    const data = new Uint8Array(N * N * 4);
    const SS = 3, isub = 1 / (SS * SS);

    const polys = [
      [[0.22, 0.12], [0.66, 0.06], [0.92, 0.38], [0.84, 0.80], [0.44, 0.94], [0.12, 0.62]],
      [[0.40, 0.02], [0.60, 0.07], [0.56, 0.62], [0.50, 0.98], [0.44, 0.60]],
      [[0.50, 0.03], [0.93, 0.72], [0.62, 0.96], [0.10, 0.55]],
      null,
    ];

    const inside = (poly, x, y) => {
      let c = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
        if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) c = !c;
      }
      return c;
    };

    for (let ti = 0; ti < 4; ti++) {
      const tx = ti & 1, ty = ti >> 1;
      const poly = polys[ti];
      const seed = 3300 + ti * 811;
      for (let y = 0; y < TILE; y++) {
        for (let x = 0; x < TILE; x++) {
          let cov = 0;
          for (let sy = 0; sy < SS; sy++) {
            for (let sx = 0; sx < SS; sx++) {
              const u = (x + (sx + 0.5) / SS) / TILE;
              const v = (y + (sy + 0.5) / SS) / TILE;
              if (poly) { if (inside(poly, u, v)) cov++; }
              else {
                const du = (u - 0.5) / 0.32;
                const dv = (v - 0.44) / (0.28 + 0.26 * clamp01((v - 0.44) * 2.2));
                if (du * du + dv * dv < 1) cov++;
              }
            }
          }
          const a = cov * isub;
          const u = (x + 0.5) / TILE, v = (y + 0.5) / TILE;
          let s = 0.60 + 0.48 * fbm(u * 6, v * 6, 3, 0.5, seed);
          s *= 0.70 + 0.58 * (1 - v);
          if (ti === 2) s = 0.50 + 0.95 * Math.pow(clamp01(1 - Math.abs(u - 0.5) * 2), 1.6);
          if (ti === 3) s = 0.52 + 0.80 * Math.pow(clamp01(1.15 - v), 2.0);
          const o = (((ty * TILE + y) * N) + (tx * TILE + x)) * 4;
          const c = (clamp01(s) * 255) | 0;
          data[o] = data[o + 1] = data[o + 2] = c;
          data[o + 3] = (clamp01(a) * 255) | 0;
        }
      }
    }
    return this._mkTex(data, N, N, false);
  }

  /** 2x2 atlas: 0 paper sheet, 1 leaf, 2 paper strip, 3 cloth scrap. */
  _buildLitterAtlas(TILE) {
    const N = TILE * 2;
    const data = new Uint8Array(N * N * 4);
    for (let ti = 0; ti < 4; ti++) {
      const tx = ti & 1, ty = ti >> 1;
      const seed = 5100 + ti * 353;
      for (let y = 0; y < TILE; y++) {
        const v = (y + 0.5) / TILE;
        for (let x = 0; x < TILE; x++) {
          const u = (x + 0.5) / TILE;
          let a = 0, r = 1, g = 1, b = 1;
          const tear = (fbm(u * 9, v * 9, 3, 0.5, seed) - 0.5) * 0.10;

          if (ti === 0 || ti === 2) {
            const hw = (ti === 0 ? 0.33 : 0.15) + tear;
            const hh = 0.41 + tear;
            a = (Math.abs(u - 0.5) < hw && Math.abs(v - 0.5) < hh) ? 1 : 0;
            const grime = fbm(u * 5, v * 5, 3, 0.5, seed + 7);
            r = 0.90 - grime * 0.17; g = 0.87 - grime * 0.19; b = 0.78 - grime * 0.21;
            if (a > 0 && ((v * 14) % 1) < 0.28 && Math.abs(u - 0.5) < hw * 0.8) { r *= 0.74; g *= 0.74; b *= 0.75; }
          } else if (ti === 1) {
            const du = (u - 0.5) / (0.23 + tear);
            const dv = (v - 0.5) / (0.43 + tear);
            a = (du * du + dv * dv < 1) ? 1 : 0;
            const rib = 1 - clamp01(Math.abs(u - 0.5) * 26);
            r = 0.46 + 0.24 * v + rib * 0.16;
            g = 0.44 + 0.20 * v + rib * 0.14;
            b = 0.20 + 0.10 * v + rib * 0.06;
          } else {
            const rag = 0.35 + (fbm(u * 7, v * 7, 3, 0.5, seed + 3) - 0.5) * 0.30;
            a = (Math.abs(u - 0.5) < rag && Math.abs(v - 0.5) < rag * 1.15) ? 1 : 0;
            const gr = fbm(u * 11, v * 11, 3, 0.5, seed + 19);
            r = 0.62 - gr * 0.17; g = 0.55 - gr * 0.17; b = 0.42 - gr * 0.15;
          }

          const o = (((ty * TILE + y) * N) + (tx * TILE + x)) * 4;
          data[o]     = (clamp01(r) * 255) | 0;
          data[o + 1] = (clamp01(g) * 255) | 0;
          data[o + 2] = (clamp01(b) * 255) | 0;
          data[o + 3] = (a * 255) | 0;
        }
      }
    }
    return this._mkTex(data, N, N, true);
  }

  /* ======================================================================== *
   *  POOL PLUMBING
   * ======================================================================== */

  _fogUniforms() {
    return {
      fogDensity: { value: 0.00025 },
      fogNear: { value: 1 },
      fogFar: { value: 2000 },
      fogColor: { value: new THREE.Color(0xffffff) },
    };
  }

  _quadGeo(geo) {
    if (!this._quadPos) {
      this._quadPos = new THREE.BufferAttribute(new Float32Array([
        -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
      ]), 3);
      this._quadUv = new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2);
      this._quadIdx = new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 0, 2, 3]), 1);
    }
    geo.setAttribute('position', this._quadPos);
    geo.setAttribute('uv', this._quadUv);
    geo.setIndex(this._quadIdx);
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1e7);
    return geo;
  }

  _instanceGeo(capacity) {
    const geo = this._quadGeo(new THREE.InstancedBufferGeometry());
    const array = new Float32Array(capacity * STRIDE);
    const ib = new THREE.InstancedInterleavedBuffer(array, STRIDE, 1);
    ib.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('iSpawn', new THREE.InterleavedBufferAttribute(ib, 4, 0));
    geo.setAttribute('iVel',   new THREE.InterleavedBufferAttribute(ib, 4, 4));
    geo.setAttribute('iParam', new THREE.InterleavedBufferAttribute(ib, 4, 8));
    geo.setAttribute('iColA',  new THREE.InterleavedBufferAttribute(ib, 4, 12));
    geo.setAttribute('iColB',  new THREE.InterleavedBufferAttribute(ib, 4, 16));
    geo.setAttribute('iMisc',  new THREE.InterleavedBufferAttribute(ib, 4, 20));
    return { geo, array, ib };
  }

  /** Build a pooled, ring-buffered instanced system. */
  _pool(name, capacity, material, renderOrder) {
    const { geo, array, ib } = this._instanceGeo(capacity);
    geo.instanceCount = 0;

    const mesh = new THREE.Mesh(geo, material);
    mesh.name = 'fx_' + name;
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.renderOrder = renderOrder;
    mesh.updateMatrix();
    this.group.add(mesh);

    const sys = {
      name, cap: capacity, arr: array, ib, geo, mesh, mat: material,
      head: 0, wrapped: false, maxDeath: -1, dirtyLo: -1, dirtyHi: -1,
    };
    this._sys[name] = sys;
    this._systems.push(sys);
    this.stats.pools[name] = 0;
    return sys;
  }

  /** Grab the next ring slot. Returns the float offset to write STRIDE values at. */
  _alloc(sys, life) {
    const i = sys.head;
    sys.head++;
    if (sys.head >= sys.cap) { sys.head = 0; sys.wrapped = true; }
    const death = this._t + life;
    if (death > sys.maxDeath) sys.maxDeath = death;
    if (sys.dirtyLo < 0) { sys.dirtyLo = i; sys.dirtyHi = i; }
    else { if (i < sys.dirtyLo) sys.dirtyLo = i; if (i > sys.dirtyHi) sys.dirtyHi = i; }
    return i * STRIDE;
  }

  _flush(sys) {
    if (sys.dirtyLo < 0) return;
    const ib = sys.ib;
    ib.clearUpdateRanges();
    if (sys.dirtyLo <= sys.dirtyHi) {
      ib.addUpdateRange(sys.dirtyLo * STRIDE, (sys.dirtyHi - sys.dirtyLo + 1) * STRIDE);
    }
    ib.needsUpdate = true;
    sys.dirtyLo = -1; sys.dirtyHi = -1;
  }

  /** Orthonormal basis around a (already normalised) normal. Allocation free. */
  _basis(nx, ny, nz) {
    const n = this._n.set(nx, ny, nz);
    if (n.lengthSq() < 1e-8) n.set(0, 1, 0); else n.normalize();
    const ax = Math.abs(n.x), ay = Math.abs(n.y), az = Math.abs(n.z);
    const t1 = this._t1;
    if (ax <= ay && ax <= az) t1.set(1, 0, 0);
    else if (ay <= az) t1.set(0, 1, 0);
    else t1.set(0, 0, 1);
    t1.crossVectors(n, t1).normalize();
    this._t2.crossVectors(n, t1);
    return n;
  }

  /** Random direction inside a cone about the current basis. Writes _dir. */
  _cone(spread, bias) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.pow(Math.random(), bias ?? 0.7) * spread;
    const s = Math.sin(r), c = Math.cos(r);
    const ca = Math.cos(a), sa = Math.sin(a);
    const n = this._n, t1 = this._t1, t2 = this._t2;
    return this._dir.set(
      n.x * c + (t1.x * ca + t2.x * sa) * s,
      n.y * c + (t1.y * ca + t2.y * sa) * s,
      n.z * c + (t1.z * ca + t2.z * sa) * s,
    );
  }

  /* ======================================================================== *
   *  SYSTEM: SMOKE / DUST — soft, fake-lit off a baked normal, fogged.
   *    iParam ( size0, size1, seed, drag )
   *    iColA  ( tintR, tintG, tintB, alpha0 )
   *    iColB  ( tintR, tintG, tintB, alpha1 )
   *    iMisc  ( gravityMul, rotSpeed, windSusceptibility, rim )
   * ======================================================================== */

  _buildSmokeSystem() {
    const U = this.U;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.smoke',
      uniforms: Object.assign({
        tSmoke: { value: this._texSmoke },
        uTime: U.uTime, uWind: U.uWind, uSunView: U.uSunView,
        uSunColor: U.uSunColor, uSkyColor: U.uSkyColor,
        tDepth: U.tDepth, uDepthNearFar: U.uDepthNearFar,
        uHasDepth: U.uHasDepth, uDepthVP: U.uDepthVP,
      }, this._fogUniforms()),
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime;
        uniform vec3 uWind, uSunColor, uSkyColor;
        ${GLSL_SOFT_V}
        varying vec2 vUv;
        varying vec3 vLit, vShadow;
        varying float vAlpha, vSoftness, vRim;
        #include <fog_pars_vertex>
        ${GLSL_COMMON}

        void main() {
          vUv = vec2( 0.0 );
          vLit = vec3( 0.0 ); vShadow = vec3( 0.0 );
          vAlpha = 0.0; vSoftness = 1.0; vRim = 0.0;
          vDepthClip = vec4( 0.0, 0.0, 0.0, 1.0 );
          vec4 mvPosition = vec4( 0.0, 0.0, -1.0, 1.0 );

          float age  = uTime - iSpawn.w;
          float life = max( iVel.w, 1e-4 );
          float t    = age / life;
          if ( age < 0.0 || t >= 1.0 ) {
            gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 );
            #include <fog_vertex>
            return;
          }

          float sd = iParam.z * BS_TAU;
          vec3 g = vec3( 0.0, ${GRAV.toFixed(3)} * iMisc.x, 0.0 );
          vec3 p = bsIntegrate( iSpawn.xyz, iVel.xyz, iParam.w, g, age );

          float ws = iMisc.z;
          p += uWind * ( age * age * 0.34 ) * ws;
          p.x += sin( age * 0.85 + sd ) * 0.16 * age * ws;
          p.z += cos( age * 0.71 + sd * 1.7 ) * 0.16 * age * ws;
          p.y += age * 0.10 * ws;

          float st = 1.0 - pow( 1.0 - t, 2.2 );
          float size = mix( iParam.x, iParam.y, st );

          float fadeIn  = smoothstep( 0.0, 0.075, t );
          float fadeOut = 1.0 - smoothstep( 0.34, 1.0, t );
          fadeOut *= fadeOut;
          float alpha = mix( iColA.a, iColB.a, t ) * fadeIn * fadeOut;

          vec3 tint = mix( iColA.rgb, iColB.rgb, t );
          vLit      = tint * uSunColor * 1.45;
          vShadow   = tint * ( uSkyColor * 0.85 + uSunColor * 0.12 );
          vRim      = iMisc.w;
          vSoftness = max( 0.40, size * 0.80 );

          mvPosition = viewMatrix * vec4( p, 1.0 );
          float rot = sd + iMisc.y * age;
          mvPosition.xy += bsRot( position.xy, rot ) * size;

          alpha *= smoothstep( 0.15, 0.90, -mvPosition.z );
          vAlpha = alpha;

          float tile = floor( fract( iParam.z * 7.13 ) * 3.999 );
          vUv = ( uv + vec2( mod( tile, 2.0 ), floor( tile * 0.5 ) ) ) * 0.5;

          gl_Position = projectionMatrix * mvPosition;
          vDepthClip = uDepthVP * vec4( p, 1.0 );
          #include <fog_vertex>
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tSmoke;
        uniform vec3 uSunView, uSunColor;
        ${GLSL_SOFT_F}
        varying vec2 vUv;
        varying vec3 vLit, vShadow;
        varying float vAlpha, vSoftness, vRim;
        #include <fog_pars_fragment>

        void main() {
          vec4 s = texture2D( tSmoke, vUv );
          float a = s.a * vAlpha;
          if ( a <= 0.0035 ) discard;

          vec3 n = normalize( s.rgb * 2.0 - 1.0 );
          float ndl = dot( n, uSunView );
          float lit = smoothstep( -0.70, 0.95, ndl );
          vec3 col = mix( vShadow, vLit, lit );
          float thin = 1.0 - s.a;
          // specular-ish silvering on the sun-facing lobes
          col += uSunColor * pow( max( ndl, 0.0 ), 2.6 ) * thin * vRim * 0.55;
          // forward scatter: with the sun BEHIND the puff the thin edges glow.
          // uSunView.z goes to -1 as the camera turns into the sun.
          float fs = pow( clamp( -uSunView.z, 0.0, 1.0 ), 2.0 );
          col += uSunColor * fs * ( 0.22 + 1.05 * thin ) * vRim;

          a *= bsSoft( vSoftness );
          gl_FragColor = vec4( col, clamp( a, 0.0, 1.0 ) );
          #include <fog_fragment>
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, blending: THREE.NormalBlending,
      fog: true, toneMapped: true,
    });
    this._pool('smoke', this.preset.smoke, mat, 18);
  }

  /* ======================================================================== *
   *  SYSTEM: FIRE — additive HDR, turbulent, soft against geometry.
   *    iParam ( size0, size1, seed, drag )
   *    iMisc  ( gravityMul, rotSpeed, turbulence, brightness )
   * ======================================================================== */

  _buildFireSystem() {
    const U = this.U;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.fire',
      uniforms: {
        tSmoke: { value: this._texSmoke },
        uTime: U.uTime, uWind: U.uWind,
        tDepth: U.tDepth, uDepthNearFar: U.uDepthNearFar,
        uHasDepth: U.uHasDepth, uDepthVP: U.uDepthVP,
      },
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime;
        uniform vec3 uWind;
        ${GLSL_SOFT_V}
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha, vSoftness;
        ${GLSL_COMMON}

        void main() {
          vUv = vec2( 0.0 ); vCol = vec3( 0.0 ); vAlpha = 0.0; vSoftness = 0.5;
          vDepthClip = vec4( 0.0, 0.0, 0.0, 1.0 );

          float age  = uTime - iSpawn.w;
          float life = max( iVel.w, 1e-4 );
          float t    = age / life;
          if ( age < 0.0 || t >= 1.0 ) { gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 ); return; }

          float sd = iParam.z * BS_TAU;
          vec3 g = vec3( 0.0, ${GRAV.toFixed(3)} * iMisc.x, 0.0 );
          vec3 p = bsIntegrate( iSpawn.xyz, iVel.xyz, iParam.w, g, age );
          float tb = iMisc.z;
          p.x += sin( age * 7.3 + sd ) * 0.16 * tb * age;
          p.y += sin( age * 5.1 + sd * 2.3 ) * 0.10 * tb * age;
          p.z += cos( age * 6.7 + sd * 1.4 ) * 0.16 * tb * age;
          p += uWind * age * age * 0.10;

          float st = 1.0 - pow( 1.0 - t, 1.7 );
          float size = mix( iParam.x, iParam.y, st );

          // white-hot -> yellow -> orange -> ember, brightness collapses fast
          vec3 col = mix( iColA.rgb, iColB.rgb, pow( t, 0.55 ) );
          float bright = iMisc.w * ( 1.0 - smoothstep( 0.05, 0.92, t ) );
          bright *= 0.82 + 0.30 * sin( age * 41.0 + sd * 13.0 );
          vCol = col * max( bright, 0.0 );
          vAlpha = ( 1.0 - smoothstep( 0.55, 1.0, t ) ) * iColA.a;
          vSoftness = max( 0.30, size * 0.6 );

          vec4 mvPosition = viewMatrix * vec4( p, 1.0 );
          mvPosition.xy += bsRot( position.xy, sd + iMisc.y * age ) * size;
          vAlpha *= smoothstep( 0.12, 0.7, -mvPosition.z );

          float tile = floor( fract( iParam.z * 3.71 ) * 3.999 );
          vUv = ( uv + vec2( mod( tile, 2.0 ), floor( tile * 0.5 ) ) ) * 0.5;

          gl_Position = projectionMatrix * mvPosition;
          vDepthClip = uDepthVP * vec4( p, 1.0 );
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tSmoke;
        ${GLSL_SOFT_F}
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha, vSoftness;

        void main() {
          float d = texture2D( tSmoke, vUv ).a;
          float a = d * vAlpha;
          if ( a <= 0.004 ) discard;
          // hot core: the densest part of the puff burns brightest
          vec3 col = vCol * ( 0.45 + 1.9 * d * d );
          a *= bsSoft( vSoftness );
          gl_FragColor = vec4( col, clamp( a, 0.0, 1.0 ) );
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      fog: false, toneMapped: true,
    });
    this._pool('fire', this.preset.fire, mat, 21);
  }

  /* ======================================================================== *
   *  SYSTEM: SPARK — additive, HDR, motion-stretched along the live velocity.
   *    iParam ( width, maxLength, seed, drag )
   *    iMisc  ( gravityMul, roundness, flickerHz, brightness )
   * ======================================================================== */

  _buildSparkSystem() {
    const U = this.U;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.spark',
      uniforms: { uTime: U.uTime, uStretch: { value: 0.055 } },
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime, uStretch;
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha, vRound;
        ${GLSL_COMMON}

        void main() {
          vUv = uv; vCol = vec3( 0.0 ); vAlpha = 0.0; vRound = 0.0;

          float age  = uTime - iSpawn.w;
          float life = max( iVel.w, 1e-4 );
          float t    = age / life;
          if ( age < 0.0 || t >= 1.0 ) { gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 ); return; }

          float sd = iParam.z * BS_TAU;
          vec3 g = vec3( 0.0, ${GRAV.toFixed(3)} * iMisc.x, 0.0 );
          vec3 p = bsIntegrate( iSpawn.xyz, iVel.xyz, iParam.w, g, age );
          vec3 v = bsVelocity( iVel.xyz, iParam.w, g, age );

          vec4 mvPosition = viewMatrix * vec4( p, 1.0 );
          vec3 vView = ( viewMatrix * vec4( v, 0.0 ) ).xyz;

          float w = iParam.x * ( 1.0 - t * 0.35 );
          vec2 d2 = vView.xy;
          float sp = length( d2 );
          vec2 dir = sp > 1e-4 ? d2 / sp : vec2( 0.0, 1.0 );
          vec2 perp = vec2( -dir.y, dir.x );
          float len = mix( w, w + min( sp * uStretch, iParam.y ), 1.0 - iMisc.y );

          mvPosition.xy += dir * ( position.y * len ) + perp * ( position.x * w );

          vec3 col = mix( iColA.rgb, iColB.rgb, pow( t, 0.62 ) );
          float fl = 0.70 + 0.30 * sin( age * iMisc.z + sd * 17.0 );
          // glitter twinkles hard; a spark just flickers
          float tw = mix( 1.0, 0.18 + 0.95 * pow( abs( sin( age * 13.0 + sd * 7.0 ) ), 4.0 ), iMisc.y );
          float decay = 1.0 - smoothstep( 0.35, 1.0, t );
          vCol = col * iMisc.w * fl * tw * decay;
          vAlpha = iColA.a * ( 1.0 - smoothstep( 0.55, 1.0, t ) );
          vAlpha *= smoothstep( 0.08, 0.45, -mvPosition.z );
          // cheap aerial-perspective attenuation (additive can't use scene fog)
          vAlpha *= exp( mvPosition.z * 0.0065 );
          vRound = iMisc.y;

          gl_Position = projectionMatrix * mvPosition;
        }`,
      fragmentShader: /* glsl */`
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha, vRound;
        void main() {
          vec2 c = vUv * 2.0 - 1.0;
          float cross = pow( max( 0.0, 1.0 - abs( c.x ) ), 2.0 );
          float along = pow( clamp( vUv.y, 0.0, 1.0 ), 1.5 );
          float streak = cross * mix( along, 1.0, 0.12 );
          float dot_ = pow( max( 0.0, 1.0 - length( c ) ), 2.6 );
          float m = mix( streak, dot_, vRound );
          float a = m * vAlpha;
          if ( a <= 0.004 ) discard;
          gl_FragColor = vec4( vCol * ( 0.35 + 1.5 * m ), clamp( a, 0.0, 1.0 ) );
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      fog: false, toneMapped: true,
    });
    this._pool('spark', this.preset.spark, mat, 23);
  }

  /* ======================================================================== *
   *  SYSTEM: DEBRIS — chips / splinters / shards / droplets. Textured atlas,
   *  tumbling, lit by the key + sky so they read as solid objects.
   *    iParam ( size, aspect, seed, drag )
   *    iMisc  ( gravityMul, rotSpeed, atlasTile, floorY )
   * ======================================================================== */

  _buildDebrisSystem() {
    const U = this.U;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.debris',
      uniforms: Object.assign({
        tDebris: { value: this._texDebris },
        uTime: U.uTime, uSunView: U.uSunView,
        uSunColor: U.uSunColor, uSkyColor: U.uSkyColor,
      }, this._fogUniforms()),
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime;
        uniform vec3 uSunView, uSunColor, uSkyColor;
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha;
        #include <fog_pars_vertex>
        ${GLSL_COMMON}

        void main() {
          vUv = vec2( 0.0 ); vCol = vec3( 0.0 ); vAlpha = 0.0;
          vec4 mvPosition = vec4( 0.0, 0.0, -1.0, 1.0 );

          float age  = uTime - iSpawn.w;
          float life = max( iVel.w, 1e-4 );
          float t    = age / life;
          if ( age < 0.0 || t >= 1.0 ) {
            gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 );
            #include <fog_vertex>
            return;
          }

          float sd = iParam.z * BS_TAU;
          vec3 g = vec3( 0.0, ${GRAV.toFixed(3)} * iMisc.x, 0.0 );
          vec3 p = bsIntegrate( iSpawn.xyz, iVel.xyz, iParam.w, g, age );
          // soft floor: debris settles instead of sinking through the world
          p.y = max( p.y, iMisc.w );

          mvPosition = viewMatrix * vec4( p, 1.0 );
          float rot = sd + iMisc.y * age;
          vec2 q = position.xy * vec2( 1.0, iParam.y );
          mvPosition.xy += bsRot( q, rot ) * iParam.x;

          // fake tumbling normal so the lighting flickers as the chip spins
          vec3 n = normalize( vec3( sin( rot ) * 0.85, cos( rot * 0.77 ) * 0.55, 0.72 ) );
          float ndl = max( dot( n, uSunView ), 0.0 );
          vCol = iColA.rgb * ( uSunColor * ( 0.22 + 1.25 * ndl ) + uSkyColor * 0.42 );

          vAlpha = iColA.a * ( 1.0 - smoothstep( 0.72, 1.0, t ) );
          vAlpha *= smoothstep( 0.10, 0.45, -mvPosition.z );

          float tile = clamp( floor( iMisc.z + 0.5 ), 0.0, 3.0 );
          vUv = ( uv + vec2( mod( tile, 2.0 ), floor( tile * 0.5 ) ) ) * 0.5;

          gl_Position = projectionMatrix * mvPosition;
          #include <fog_vertex>
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tDebris;
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha;
        #include <fog_pars_fragment>
        void main() {
          vec4 s = texture2D( tDebris, vUv );
          float a = s.a * vAlpha;
          if ( a <= 0.02 ) discard;
          gl_FragColor = vec4( vCol * s.r, clamp( a, 0.0, 1.0 ) );
          #include <fog_fragment>
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, blending: THREE.NormalBlending,
      fog: true, toneMapped: true,
    });
    this._pool('debris', this.preset.debris, mat, 19);
  }

  /* ======================================================================== *
   *  SYSTEM: RING — expanding shockwave. Camera-facing flash ring, or a
   *  ground-aligned ring for the blast wave skimming the deck.
   *    iParam ( r0, r1, seed, thickness )
   *    iMisc  ( orientation 0=camera 1=ground, -, -, brightness )
   * ======================================================================== */

  _buildRingSystem() {
    const U = this.U;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.ring',
      uniforms: {
        uTime: U.uTime,
        tDepth: U.tDepth, uDepthNearFar: U.uDepthNearFar,
        uHasDepth: U.uHasDepth, uDepthVP: U.uDepthVP,
      },
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime;
        ${GLSL_SOFT_V}
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha, vThick;
        ${GLSL_COMMON}

        void main() {
          vUv = uv; vCol = vec3( 0.0 ); vAlpha = 0.0; vThick = 0.2;
          vDepthClip = vec4( 0.0, 0.0, 0.0, 1.0 );

          float age  = uTime - iSpawn.w;
          float life = max( iVel.w, 1e-4 );
          float t    = age / life;
          if ( age < 0.0 || t >= 1.0 ) { gl_Position = vec4( 2.0, 2.0, 2.0, 1.0 ); return; }

          float ex = 1.0 - pow( 1.0 - t, 2.8 );
          float r  = mix( iParam.x, iParam.y, ex );

          vec3 p = iSpawn.xyz + iVel.xyz * age;
          vec4 mvPosition;
          if ( iMisc.x > 0.5 ) {
            vec3 off = vec3( position.x, 0.0, position.y ) * ( r * 2.0 );
            mvPosition = viewMatrix * vec4( p + off, 1.0 );
          } else {
            mvPosition = viewMatrix * vec4( p, 1.0 );
            mvPosition.xy += position.xy * ( r * 2.0 );
          }

          vThick = mix( iParam.w, iParam.w * 0.22, ex );
          vCol = mix( iColA.rgb, iColB.rgb, ex ) * iMisc.w * ( 1.0 - smoothstep( 0.0, 1.0, t ) );
          vAlpha = iColA.a * ( 1.0 - smoothstep( 0.10, 1.0, t ) );

          gl_Position = projectionMatrix * mvPosition;
          vDepthClip = uDepthVP * vec4( p, 1.0 );
        }`,
      fragmentShader: /* glsl */`
        ${GLSL_SOFT_F}
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha, vThick;
        void main() {
          float r = length( vUv * 2.0 - 1.0 );
          float inner = 1.0 - clamp( vThick, 0.02, 0.85 );
          float band = smoothstep( inner - 0.07, inner + 0.03, r )
                     * ( 1.0 - smoothstep( 0.88, 1.0, r ) );
          float a = band * vAlpha;
          if ( a <= 0.004 ) discard;
          a *= bsSoft( 0.6 );
          gl_FragColor = vec4( vCol * band, clamp( a, 0.0, 1.0 ) );
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      fog: false, toneMapped: true,
    });
    this._pool('ring', this.preset.ring, mat, 22);
  }

  /* ======================================================================== *
   *  AMBIENT SYSTEMS — never spawned, never updated on the CPU. The whole
   *  field wraps around the camera in the vertex shader, so a handful of
   *  thousand motes follow the player across the entire level for free.
   * ======================================================================== */

  _ambientPool(name, count, material, renderOrder, fill) {
    const { geo, array } = this._instanceGeo(count);
    for (let i = 0; i < count; i++) fill(array, i * STRIDE, i);
    geo.instanceCount = count;
    const mesh = new THREE.Mesh(geo, material);
    mesh.name = 'fx_' + name;
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.renderOrder = renderOrder;
    mesh.updateMatrix();
    this.group.add(mesh);
    const sys = { name, cap: count, geo, mesh, mat: material, ambient: true };
    this._sys[name] = sys;
    this._ambient.push(sys);
    return sys;
  }

  /** Sunlit dust motes drifting through the whole level, banded into shafts. */
  _buildMotes() {
    const U = this.U;
    const BOX = 40, SLAB = 13;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.motes',
      uniforms: {
        uTime: U.uTime, uWind: U.uWind, uSunDir: U.uSunDir,
        uSunColor: U.uSunColor, uSkyColor: U.uSkyColor,
        uTanA: U.uTanA, uTanB: U.uTanB, uPxScale: U.uPxScale,
        uGroundY: U.uGroundY,
        uBox: { value: BOX }, uSlab: { value: SLAB },
        uOpacity: { value: 1.25 },
      },
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime, uBox, uSlab, uPxScale, uOpacity, uGroundY;
        uniform vec3 uWind, uSunDir, uSunColor, uSkyColor, uTanA, uTanB;
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha;
        ${GLSL_COMMON}

        void main() {
          vUv = uv;
          float sd = iParam.z * BS_TAU;

          vec3 p = iSpawn.xyz;
          p.xz += uWind.xz * uTime * 0.30;
          p.x += sin( uTime * 0.41 + sd ) * 0.85;
          p.z += cos( uTime * 0.36 + sd * 1.7 ) * 0.85;
          p.y += sin( uTime * 0.22 + sd * 2.3 ) * 0.40 + uTime * 0.035;

          vec2 rel = p.xz - cameraPosition.xz + uBox * 0.5;
          rel = mod( rel, uBox ) - uBox * 0.5;
          float h = mod( p.y, uSlab );
          vec3 world = vec3( cameraPosition.x + rel.x, h + uGroundY, cameraPosition.z + rel.y );

          // god-ray shafts: band the density on the plane perpendicular to the sun
          float a1 = dot( world, uTanA );
          float a2 = dot( world, uTanB );
          float shaft = 0.5 + 0.5 * sin( a1 * 0.46 + 1.3 ) * sin( a2 * 0.33 - 0.7 );
          shaft = pow( clamp( shaft, 0.0, 1.0 ), 2.4 );
          shaft *= 0.55 + 0.45 * ( 0.5 + 0.5 * sin( a1 * 1.9 + a2 * 0.63 ) );
          shaft = mix( 0.22, 1.0, clamp( shaft, 0.0, 1.0 ) );

          vec3 toCam = world - cameraPosition;
          float dist = max( length( toCam ), 0.05 );
          vec3 vdir = toCam / dist;
          float back = pow( max( dot( vdir, uSunDir ), 0.0 ), 3.0 );

          vCol = mix( uSkyColor * 0.95, uSunColor, 0.22 + back * 0.78 ) * ( 0.62 + back * 3.30 ) * shaft;

          float wsize = iParam.x;
          float minSize = dist * uPxScale * 3.1;
          float size = max( wsize, minSize );

          float hFade   = exp( -h * 0.115 );
          float nearF   = smoothstep( 0.30, 2.2, dist );
          float farF    = 1.0 - smoothstep( uBox * 0.30, uBox * 0.47, dist );
          float energy  = mix( 1.0, wsize / size, 0.72 );
          vAlpha = uOpacity * iColA.a * hFade * nearF * farF * energy;

          vec4 mvPosition = viewMatrix * vec4( world, 1.0 );
          mvPosition.xy += bsRot( position.xy, sd ) * size;
          gl_Position = projectionMatrix * mvPosition;
        }`,
      fragmentShader: /* glsl */`
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha;
        void main() {
          float r = length( vUv * 2.0 - 1.0 );
          float m = pow( max( 0.0, 1.0 - r ), 1.6 );
          float a = m * vAlpha;
          if ( a <= 0.0025 ) discard;
          gl_FragColor = vec4( vCol * m, clamp( a, 0.0, 1.0 ) );
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      fog: false, toneMapped: true,
    });

    this._ambientPool('motes', this.preset.motes, mat, 14, (a, o) => {
      a[o]     = (Math.random() - 0.5) * BOX;
      a[o + 1] = Math.pow(Math.random(), 1.75) * SLAB;
      a[o + 2] = (Math.random() - 0.5) * BOX;
      a[o + 3] = 0;
      a[o + 7] = 1e9;                                   // life: effectively immortal
      a[o + 8] = 0.014 + Math.pow(Math.random(), 2.2) * 0.090;   // world size
      a[o + 9] = 1;
      a[o + 10] = Math.random();                        // seed
      a[o + 11] = 0;
      a[o + 15] = 0.35 + Math.random() * 0.65;          // per-mote alpha
    });
  }

  /** Wind-driven sand skimming the deck: fast, stretched, warm, low. */
  _buildWindSand() {
    const U = this.U;
    const BOX = 36, SLAB = 2.6;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.sand',
      uniforms: {
        uTime: U.uTime, uWind: U.uWind, uSunDir: U.uSunDir,
        uSunColor: U.uSunColor, uSkyColor: U.uSkyColor, uGroundY: U.uGroundY,
        uPxScale: U.uPxScale,
        uBox: { value: BOX }, uSlab: { value: SLAB },
        uOpacity: { value: 0.80 },
      },
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime, uBox, uSlab, uOpacity, uGroundY, uPxScale;
        uniform vec3 uWind, uSunDir, uSunColor, uSkyColor;
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha;
        ${GLSL_COMMON}

        void main() {
          vUv = uv;
          float sd = iParam.z * BS_TAU;
          float gust = 0.75 + 0.55 * sin( uTime * 0.27 + sd * 0.7 );

          vec3 p = iSpawn.xyz;
          p.xz += uWind.xz * uTime * ( 2.4 + iParam.y * 2.6 ) * gust;
          p.y  += sin( uTime * 1.7 + sd * 3.1 ) * 0.22;

          vec2 rel = p.xz - cameraPosition.xz + uBox * 0.5;
          rel = mod( rel, uBox ) - uBox * 0.5;
          float h = mod( p.y, uSlab );
          vec3 world = vec3( cameraPosition.x + rel.x, h + uGroundY, cameraPosition.z + rel.y );

          vec3 toCam = world - cameraPosition;
          float dist = max( length( toCam ), 0.05 );
          vec3 vdir = toCam / dist;
          float back = pow( max( dot( vdir, uSunDir ), 0.0 ), 2.4 );
          vCol = mix( uSkyColor * 0.70, uSunColor, 0.34 + back * 0.66 ) * ( 0.42 + back * 2.30 );

          vec4 mvPosition = viewMatrix * vec4( world, 1.0 );
          vec3 wView = ( viewMatrix * vec4( normalize( uWind + vec3( 0.0, 0.001, 0.0 ) ), 0.0 ) ).xyz;
          vec2 d2 = wView.xy;
          float sp = length( d2 );
          vec2 dir = sp > 1e-4 ? d2 / sp : vec2( 1.0, 0.0 );
          vec2 perp = vec2( -dir.y, dir.x );

          float w = max( iParam.x, dist * uPxScale * 1.4 );
          float len = w * ( 6.0 + iParam.y * 16.0 );
          mvPosition.xy += dir * ( position.x * len ) + perp * ( position.y * w );

          float hFade = exp( -h * 0.55 );
          float nearF = smoothstep( 0.5, 3.5, dist );
          float farF  = 1.0 - smoothstep( uBox * 0.26, uBox * 0.45, dist );
          vAlpha = uOpacity * iColA.a * hFade * nearF * farF * gust;

          gl_Position = projectionMatrix * mvPosition;
        }`,
      fragmentShader: /* glsl */`
        varying vec2 vUv;
        varying vec3 vCol;
        varying float vAlpha;
        void main() {
          vec2 c = vUv * 2.0 - 1.0;
          float m = pow( max( 0.0, 1.0 - abs( c.y ) ), 1.8 ) * pow( max( 0.0, 1.0 - abs( c.x ) ), 0.9 );
          float a = m * vAlpha;
          if ( a <= 0.002 ) discard;
          gl_FragColor = vec4( vCol * m, clamp( a, 0.0, 1.0 ) );
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: true, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      fog: false, toneMapped: true,
    });

    this._ambientPool('sand', this.preset.sand, mat, 15, (a, o) => {
      a[o]     = (Math.random() - 0.5) * BOX;
      a[o + 1] = Math.pow(Math.random(), 2.2) * SLAB;
      a[o + 2] = (Math.random() - 0.5) * BOX;
      a[o + 7] = 1e9;
      a[o + 8] = 0.006 + Math.random() * 0.016;   // width
      a[o + 9] = Math.random();                   // speed / length variation
      a[o + 10] = Math.random();
      a[o + 15] = 0.30 + Math.random() * 0.70;
    });
  }

  /** Paper and leaf litter tumbling through the compound. */
  _buildLitter() {
    const U = this.U;
    const BOX = 44, SLAB = 3.4;
    const mat = new THREE.ShaderMaterial({
      name: 'fx.litter',
      uniforms: Object.assign({
        tLitter: { value: this._texLitter },
        uTime: U.uTime, uWind: U.uWind, uSunDir: U.uSunDir,
        uSunColor: U.uSunColor, uSkyColor: U.uSkyColor, uGroundY: U.uGroundY,
        uBox: { value: BOX }, uSlab: { value: SLAB },
      }, this._fogUniforms()),
      vertexShader: /* glsl */`
        attribute vec4 iSpawn, iVel, iParam, iColA, iColB, iMisc;
        uniform float uTime, uBox, uSlab, uGroundY;
        uniform vec3 uWind, uSunDir, uSunColor, uSkyColor;
        varying vec2 vUv;
        varying vec3 vLight;
        varying float vFade;
        #include <fog_pars_vertex>
        ${GLSL_COMMON}

        void main() {
          float sd = iParam.z * BS_TAU;

          vec3 p = iSpawn.xyz;
          p.xz += uWind.xz * uTime * ( 1.1 + iParam.y * 1.4 );
          p.x += sin( uTime * 0.62 + sd ) * 1.4;
          p.z += cos( uTime * 0.55 + sd * 1.9 ) * 1.4;
          p.y += sin( uTime * 0.83 + sd * 2.7 ) * 0.9;

          vec2 rel = p.xz - cameraPosition.xz + uBox * 0.5;
          rel = mod( rel, uBox ) - uBox * 0.5;
          float h = mod( p.y, uSlab );
          // hug the ground: most litter skitters, a little of it lofts
          h = pow( h / uSlab, 3.0 ) * uSlab;
          vec3 base = vec3( cameraPosition.x + rel.x, h + uGroundY + 0.04, cameraPosition.z + rel.y );

          float ang = uTime * iMisc.y + sd;
          vec3 axis = normalize( vec3( sin( sd * 3.1 ) + 0.2, cos( sd * 7.7 ), sin( sd * 5.3 ) - 0.2 ) );
          vec3 local = vec3( position.xy * iParam.x, 0.0 );
          float ca = cos( ang ), sa = sin( ang );
          vec3 rotated = local * ca + cross( axis, local ) * sa + axis * dot( axis, local ) * ( 1.0 - ca );
          vec3 world = base + rotated;

          vec3 nrm = vec3( 0.0, 0.0, 1.0 );
          nrm = nrm * ca + cross( axis, nrm ) * sa + axis * dot( axis, nrm ) * ( 1.0 - ca );
          float ndl = abs( dot( nrm, uSunDir ) );
          vLight = uSunColor * ( 0.18 + 1.05 * ndl ) + uSkyColor * 0.40;

          float dist = length( world - cameraPosition );
          vFade = ( 1.0 - smoothstep( uBox * 0.30, uBox * 0.46, dist ) ) * smoothstep( 0.4, 1.6, dist );

          float tile = clamp( floor( iMisc.z + 0.5 ), 0.0, 3.0 );
          vUv = ( uv + vec2( mod( tile, 2.0 ), floor( tile * 0.5 ) ) ) * 0.5;

          vec4 mvPosition = viewMatrix * vec4( world, 1.0 );
          gl_Position = projectionMatrix * mvPosition;
          #include <fog_vertex>
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tLitter;
        varying vec2 vUv;
        varying vec3 vLight;
        varying float vFade;
        #include <fog_pars_fragment>
        void main() {
          vec4 s = texture2D( tLitter, vUv );
          if ( s.a * vFade < 0.45 ) discard;
          gl_FragColor = vec4( s.rgb * vLight, 1.0 );
          #include <fog_fragment>
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
      transparent: false, depthWrite: true, depthTest: true,
      side: THREE.DoubleSide, fog: true, toneMapped: true,
    });

    this._ambientPool('litter', this.preset.litter, mat, 0, (a, o) => {
      a[o]     = (Math.random() - 0.5) * BOX;
      a[o + 1] = Math.random() * SLAB;
      a[o + 2] = (Math.random() - 0.5) * BOX;
      a[o + 7] = 1e9;
      a[o + 8] = 0.07 + Math.random() * 0.11;    // size
      a[o + 9] = Math.random();                  // speed variation
      a[o + 10] = Math.random();
      a[o + 21] = 0.8 + Math.random() * 2.6;     // tumble rate
      a[o + 22] = Math.floor(Math.random() * 4); // atlas tile
    });
  }

  /* ======================================================================== *
   *  SOFT-PARTICLE DEPTH
   * ======================================================================== */

  _buildDepthResources() {
    if (!this._softEnabled) return;
    this._depthMat = new THREE.MeshBasicMaterial({
      name: 'fx.depthOnly', colorWrite: false, side: THREE.DoubleSide, fog: false,
    });
    this._allocDepthRT(this._viewW, this._viewH);
  }

  _allocDepthRT(w, h) {
    if (!this._softEnabled || !this._depthOk) return;
    const dw = Math.max(160, Math.floor(w * 0.5));
    const dh = Math.max(90, Math.floor(h * 0.5));
    if (this._depthRT && this._depthRT.width === dw && this._depthRT.height === dh) return;
    try {
      this._depthRT?.depthTexture?.dispose();
      this._depthRT?.dispose();
      const dt = new THREE.DepthTexture(dw, dh);
      dt.type = THREE.UnsignedIntType;
      dt.format = THREE.DepthFormat;
      dt.minFilter = THREE.NearestFilter;
      dt.magFilter = THREE.NearestFilter;
      dt.generateMipmaps = false;
      dt.name = 'fx.sceneDepth';
      this._depthRT = new THREE.WebGLRenderTarget(dw, dh, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        depthBuffer: true,
        stencilBuffer: false,
        generateMipmaps: false,
        depthTexture: dt,
      });
    } catch (e) {
      console.warn('[Particles] depth target allocation failed — soft particles disabled', e);
      this._depthOk = false;
      this._depthRT = null;
    }
  }

  // Objects that must not write into the soft-particle depth buffer: the sky
  // dome, every transparent surface, points/sprites/lines, anything driven by a
  // custom vertex shader (the override material would collapse it to the origin)
  // and the whole FX group itself.
  _collect(o) {
    if (!o.visible) return;
    if (o === this.group) { this._push(o); return; }
    if (o.isPoints || o.isSprite || o.isLine || o.isLineSegments) { this._push(o); return; }
    if (o.userData?.viewmodel || o.userData?.noDepth) { this._push(o); return; }
    const m = o.material;
    if (!m) return;
    const bad = (mm) => !mm || mm.transparent === true || mm.depthWrite === false ||
      mm.isShaderMaterial === true || mm.isRawShaderMaterial === true;
    if (Array.isArray(m)) {
      for (let i = 0; i < m.length; i++) if (bad(m[i])) { this._push(o); return; }
    } else if (bad(m)) {
      this._push(o);
    }
  }

  _push(o) { if (this._hiddenCount < this._hidden.length) this._hidden[this._hiddenCount++] = o; }

  _renderDepth() {
    const r = this.g?.renderer?.renderer;
    const cam = this.g?.camera;
    const scene = this.g?.scene;
    if (!r || !cam || !scene || !this._depthRT || !this._depthMat) return false;

    try {
      cam.updateMatrixWorld();
      this._depthVP.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);

      if (this.g.frame - this._hiddenFrame > 12 || this._hiddenCount === 0) {
        this._hiddenFrame = this.g.frame;
        this._hiddenCount = 0;
        scene.traverse(this._collect);
      }
      for (let i = 0; i < this._hiddenCount; i++) {
        const o = this._hidden[i];
        if (o) { o.__fxWasVisible = o.visible; o.visible = false; }
      }

      const prevTarget = r.getRenderTarget();
      const prevOverride = scene.overrideMaterial;
      const prevBg = scene.background;
      const prevShadow = r.shadowMap.autoUpdate;
      const prevAutoClear = r.autoClear;

      scene.background = null;
      scene.overrideMaterial = this._depthMat;
      r.shadowMap.autoUpdate = false;
      r.autoClear = true;
      r.setRenderTarget(this._depthRT);
      r.clear(true, true, false);
      r.render(scene, cam);

      scene.overrideMaterial = prevOverride;
      scene.background = prevBg;
      r.shadowMap.autoUpdate = prevShadow;
      r.autoClear = prevAutoClear;
      r.setRenderTarget(prevTarget);

      for (let i = 0; i < this._hiddenCount; i++) {
        const o = this._hidden[i];
        if (o && o.__fxWasVisible !== undefined) o.visible = o.__fxWasVisible;
      }

      this.U.tDepth.value = this._depthRT.depthTexture;
      this.U.uDepthNearFar.value.set(cam.near, cam.far);
      this.U.uHasDepth.value = 1;
      this.stats.softPasses++;
      return true;
    } catch (e) {
      console.warn('[Particles] depth prepass failed — soft particles disabled', e);
      this._depthOk = false;
      this.U.uHasDepth.value = 0;
      return false;
    }
  }

  /* ======================================================================== *
   *  FLASH LIGHTS — two recycled point lights for the metal "ping" and the
   *  explosion flash. Added once at init so the light count (and therefore the
   *  compiled program set) never changes at runtime.
   * ======================================================================== */

  _buildFlashLights() {
    const scene = this.g?.scene;
    if (!scene) return;
    for (let i = 0; i < 2; i++) {
      const l = new THREE.PointLight(0xffb266, 0, 12, 2);
      l.name = 'fx_flash' + i;
      l.castShadow = false;
      l.visible = true;
      scene.add(l);
      this._lights.push({ light: l, t: 0, dur: 0, base: 0 });
      this.g?.registry?.lights?.push?.(l);
    }
  }

  flashLight(pos, colorHex, intensity, seconds, distance) {
    if (!this._lights.length || !pos) return;
    let best = this._lights[0], bestLeft = Infinity;
    for (const L of this._lights) {
      const left = L.dur > 0 ? (L.dur - L.t) : -1;
      if (left < bestLeft) { bestLeft = left; best = L; }
    }
    best.light.position.set(pos.x, pos.y, pos.z);
    best.light.color.setHex(colorHex ?? 0xffb266);
    best.light.distance = distance ?? 10;
    best.base = intensity ?? 8;
    best.dur = Math.max(0.02, seconds ?? 0.1);
    best.t = 0;
    best.light.intensity = best.base;
  }

  /* ======================================================================== *
   *  FRAME
   * ======================================================================== */

  /** Pull the live grade off Sky / Renderer so FX always match the atmosphere. */
  _syncAtmosphere() {
    const sky = this.g?.sky;
    const rend = this.g?.renderer;
    const U = this.U;

    if (sky?.sunDirection) this._sunDir.copy(sky.sunDirection);
    else if (rend?.sunDir) this._sunDir.copy(rend.sunDir);
    if (this._sunDir.lengthSq() < 1e-6) this._sunDir.set(-0.62, 0.28, -0.73);
    this._sunDir.normalize();

    if (sky?.sunColor) U.uSunColor.value.copy(sky.sunColor);
    else if (rend?.sun?.color) U.uSunColor.value.copy(rend.sun.color).multiplyScalar(0.86);
    if (sky?.skyColor) U.uSkyColor.value.copy(sky.skyColor);

    // basis perpendicular to the sun — drives the god-ray shaft banding
    const a = this._tanA, b = this._tanB, d = this._sunDir;
    if (Math.abs(d.y) < 0.95) a.set(0, 1, 0); else a.set(1, 0, 0);
    a.crossVectors(d, a).normalize();
    b.crossVectors(d, a).normalize();

    const cam = this.g?.camera;
    if (cam) this._sunView.copy(this._sunDir).transformDirection(cam.matrixWorldInverse).normalize();

    const r = this.g?.renderer?.renderer;
    if (r && cam) {
      r.getDrawingBufferSize(this._dbSize);
      const fov = (cam.fov || 68) * Math.PI / 180;
      U.uPxScale.value = (2 * Math.tan(fov * 0.5)) / Math.max(1, this._dbSize.y);
    }
  }

  update(dt, t) {
    if (!this._ready) return;
    this._t = t;
    this.U.uTime.value = t;

    this._syncAtmosphere();

    // ---- flash light decay (quadratic falloff reads like a real muzzle flash)
    for (let i = 0; i < this._lights.length; i++) {
      const L = this._lights[i];
      if (L.dur <= 0) continue;
      L.t += dt;
      if (L.t >= L.dur) { L.dur = 0; L.light.intensity = 0; continue; }
      const k = 1 - L.t / L.dur;
      L.light.intensity = L.base * k * k;
    }

    // ---- upload freshly spawned particles, retire idle pools ---------------
    let live = 0, needSoft = false;
    for (let i = 0; i < this._systems.length; i++) {
      const s = this._systems[i];
      this._flush(s);
      let count;
      if (t > s.maxDeath) {
        count = 0;
        s.head = 0; s.wrapped = false;
      } else {
        count = s.wrapped ? s.cap : s.head;
      }
      s.geo.instanceCount = count;
      this.stats.pools[s.name] = count;
      live += count;
      if (count > 0 && (s.name === 'smoke' || s.name === 'fire' || s.name === 'ring')) needSoft = true;
    }
    this.stats.live = live;

    // ---- soft-particle depth ----------------------------------------------
    if (!this._softEnabled || !this._depthOk) {
      this.U.uHasDepth.value = 0;
    } else if (!needSoft) {
      this.U.uHasDepth.value = 0;   // nothing to fade — skip the whole prepass
    } else {
      const ext = this._externalDepth();
      if (ext) {
        const cam = this.g.camera;
        cam.updateMatrixWorld();
        this._depthVP.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
        this.U.tDepth.value = ext;
        this.U.uDepthNearFar.value.set(cam.near, cam.far);
        this.U.uHasDepth.value = 1;
      } else {
        this._renderDepth();
      }
    }
  }

  /**
   * PostFX publishes a scene depth texture, but it is the LIVE depth attachment of the
   * render target the particles are themselves drawn into. Sampling it during that same
   * scene render is a framebuffer/texture feedback loop: WebGL raises
   * GL_INVALID_OPERATION on every instanced particle draw and the result is undefined.
   * Our own half-res depth prepass is the only correct source, so this is deliberately
   * inert. Kept as a seam in case PostFX ever exposes a *resolved copy* of last frame's
   * depth (it would have to be a separate texture, not sceneRT.depthTexture).
   */
  _externalDepth() {
    const post = this.g?.post;
    if (!post) return null;
    const t = post.resolvedDepthTexture || post.sceneDepthCopy || null;
    return (t && t.isTexture) ? t : null;
  }

  resize(w, h) {
    this._viewW = Math.max(1, w);
    this._viewH = Math.max(1, h);
    this._allocDepthRT(this._viewW, this._viewH);
    this._syncAtmosphere();
  }

  setWind(x, z) {
    this.wind.set(x, 0, z);
    return this;
  }

  setQuality(level) {
    const p = QUALITY[level];
    if (!p) return this;
    this.quality = level;
    this.burstScale = p.burst;
    this._softEnabled = p.soft && this._depthOk;
    // ambient pools keep their allocation; we just draw fewer of them
    const clampTo = (name, n) => {
      const s = this._sys[name];
      if (s?.geo) s.geo.instanceCount = Math.min(s.cap, n);
    };
    clampTo('motes', p.motes);
    clampTo('sand', p.sand);
    clampTo('litter', p.litter);
    return this;
  }

  /* ======================================================================== *
   *  LOW-LEVEL WRITERS — positional, allocation free, one typed-array write.
   * ======================================================================== */

  _pushSmoke(px, py, pz, vx, vy, vz, life, s0, s1, drag,
             r, g, b, a, r2, g2, b2, a2, gm, rot, wind, rim) {
    const s = this._sys.smoke; if (!s) return;
    const A = s.arr, o = this._alloc(s, life);
    A[o] = px; A[o + 1] = py; A[o + 2] = pz; A[o + 3] = this._t;
    A[o + 4] = vx; A[o + 5] = vy; A[o + 6] = vz; A[o + 7] = life;
    A[o + 8] = s0; A[o + 9] = s1; A[o + 10] = Math.random(); A[o + 11] = drag;
    A[o + 12] = r; A[o + 13] = g; A[o + 14] = b; A[o + 15] = a;
    A[o + 16] = r2; A[o + 17] = g2; A[o + 18] = b2; A[o + 19] = a2;
    A[o + 20] = gm; A[o + 21] = rot; A[o + 22] = wind; A[o + 23] = rim;
  }

  _pushFire(px, py, pz, vx, vy, vz, life, s0, s1, drag,
            r, g, b, a, r2, g2, b2, gm, rot, turb, bright) {
    const s = this._sys.fire; if (!s) return;
    const A = s.arr, o = this._alloc(s, life);
    A[o] = px; A[o + 1] = py; A[o + 2] = pz; A[o + 3] = this._t;
    A[o + 4] = vx; A[o + 5] = vy; A[o + 6] = vz; A[o + 7] = life;
    A[o + 8] = s0; A[o + 9] = s1; A[o + 10] = Math.random(); A[o + 11] = drag;
    A[o + 12] = r; A[o + 13] = g; A[o + 14] = b; A[o + 15] = a;
    A[o + 16] = r2; A[o + 17] = g2; A[o + 18] = b2; A[o + 19] = 0;
    A[o + 20] = gm; A[o + 21] = rot; A[o + 22] = turb; A[o + 23] = bright;
  }

  _pushSpark(px, py, pz, vx, vy, vz, life, w, maxLen, drag,
             r, g, b, a, r2, g2, b2, gm, round, flick, bright) {
    const s = this._sys.spark; if (!s) return;
    const A = s.arr, o = this._alloc(s, life);
    A[o] = px; A[o + 1] = py; A[o + 2] = pz; A[o + 3] = this._t;
    A[o + 4] = vx; A[o + 5] = vy; A[o + 6] = vz; A[o + 7] = life;
    A[o + 8] = w; A[o + 9] = maxLen; A[o + 10] = Math.random(); A[o + 11] = drag;
    A[o + 12] = r; A[o + 13] = g; A[o + 14] = b; A[o + 15] = a;
    A[o + 16] = r2; A[o + 17] = g2; A[o + 18] = b2; A[o + 19] = 0;
    A[o + 20] = gm; A[o + 21] = round; A[o + 22] = flick; A[o + 23] = bright;
  }

  _pushDebris(px, py, pz, vx, vy, vz, life, size, aspect, drag,
              r, g, b, a, gm, rot, tile, floorY) {
    const s = this._sys.debris; if (!s) return;
    const A = s.arr, o = this._alloc(s, life);
    A[o] = px; A[o + 1] = py; A[o + 2] = pz; A[o + 3] = this._t;
    A[o + 4] = vx; A[o + 5] = vy; A[o + 6] = vz; A[o + 7] = life;
    A[o + 8] = size; A[o + 9] = aspect; A[o + 10] = Math.random(); A[o + 11] = drag;
    A[o + 12] = r; A[o + 13] = g; A[o + 14] = b; A[o + 15] = a;
    A[o + 16] = r; A[o + 17] = g; A[o + 18] = b; A[o + 19] = 0;
    A[o + 20] = gm; A[o + 21] = rot; A[o + 22] = tile; A[o + 23] = floorY;
  }

  _pushRing(px, py, pz, life, r0, r1, thick, orient,
            r, g, b, a, r2, g2, b2, bright) {
    const s = this._sys.ring; if (!s) return;
    const A = s.arr, o = this._alloc(s, life);
    A[o] = px; A[o + 1] = py; A[o + 2] = pz; A[o + 3] = this._t;
    A[o + 4] = 0; A[o + 5] = 0; A[o + 6] = 0; A[o + 7] = life;
    A[o + 8] = r0; A[o + 9] = r1; A[o + 10] = Math.random(); A[o + 11] = thick;
    A[o + 12] = r; A[o + 13] = g; A[o + 14] = b; A[o + 15] = a;
    A[o + 16] = r2; A[o + 17] = g2; A[o + 18] = b2; A[o + 19] = 0;
    A[o + 20] = orient; A[o + 21] = 0; A[o + 22] = 0; A[o + 23] = bright;
  }

  /* ---- public, option-object flavours (convenient for collaborators) ----- */

  smokePuff(px, py, pz, vx, vy, vz, o) {
    o = o || {};
    const c = o.color ?? 0xb0a89a, c2 = o.colorEnd ?? c;
    const r = ((c >> 16) & 255) / 255, g = ((c >> 8) & 255) / 255, b = (c & 255) / 255;
    const r2 = ((c2 >> 16) & 255) / 255, g2 = ((c2 >> 8) & 255) / 255, b2 = (c2 & 255) / 255;
    this._pushSmoke(px, py, pz, vx, vy, vz,
      o.life ?? 1.4, o.size ?? 0.14, o.sizeEnd ?? 0.8, o.drag ?? 2.6,
      r, g, b, o.alpha ?? 0.5, r2, g2, b2, o.alphaEnd ?? 0.0,
      o.gravity ?? 0.15, o.spin ?? (Math.random() - 0.5) * 1.6,
      o.wind ?? 0.7, o.rim ?? 0.6);
    return this;
  }

  sparkBurst(px, py, pz, nx, ny, nz, count, o) {
    o = o || {};
    this._basis(nx ?? 0, ny ?? 1, nz ?? 0);
    const n = Math.max(1, Math.round((count ?? 12) * this.burstScale));
    for (let i = 0; i < n; i++) {
      const d = this._cone(o.spread ?? 1.15, 0.55);
      const sp = rr(o.speedMin ?? 3, o.speedMax ?? 12);
      this._pushSpark(px, py, pz, d.x * sp, d.y * sp, d.z * sp,
        rr(o.lifeMin ?? 0.25, o.lifeMax ?? 0.8),
        o.width ?? 0.014, o.maxLength ?? 0.55, o.drag ?? 1.6,
        o.r ?? 9.0, o.g ?? 5.0, o.b ?? 1.5, 1.0,
        o.r2 ?? 2.4, o.g2 ?? 0.35, o.b2 ?? 0.05,
        o.gravity ?? 1.0, o.round ?? 0, o.flicker ?? 62, o.brightness ?? 1.0);
    }
    return this;
  }

  debrisBurst(px, py, pz, nx, ny, nz, count, o) {
    o = o || {};
    this._basis(nx ?? 0, ny ?? 1, nz ?? 0);
    const n = Math.max(1, Math.round((count ?? 10) * this.burstScale));
    const c = o.color ?? 0x8c8378;
    const r = ((c >> 16) & 255) / 255, g = ((c >> 8) & 255) / 255, b = (c & 255) / 255;
    const floorY = o.floorY ?? -1e6;
    for (let i = 0; i < n; i++) {
      const d = this._cone(o.spread ?? 1.05, 0.6);
      const sp = rr(o.speedMin ?? 2.5, o.speedMax ?? 8);
      this._pushDebris(px, py, pz, d.x * sp, d.y * sp, d.z * sp,
        rr(o.lifeMin ?? 0.5, o.lifeMax ?? 1.2),
        rr(o.sizeMin ?? 0.018, o.sizeMax ?? 0.05), o.aspect ?? 1.0, o.drag ?? 0.55,
        r, g, b, 1.0, o.gravity ?? 1.0,
        (Math.random() - 0.5) * (o.spin ?? 26), o.tile ?? 0, floorY);
    }
    return this;
  }

  /* ======================================================================== *
   *  RECIPES
   * ======================================================================== */

  impact(point, normal, surface, scale) {
    if (!this._ready || !point) return;
    const s = SURFACE_ALIAS[String(surface || 'concrete').toLowerCase()] || 'concrete';
    const k = (scale ?? 1) * this.burstScale;
    const nx = normal?.x ?? 0, ny = normal?.y ?? 1, nz = normal?.z ?? 0;
    const px = point.x + nx * 0.03, py = point.y + ny * 0.03, pz = point.z + nz * 0.03;
    const floorY = (ny > 0.55) ? point.y + 0.01 : -1e6;

    switch (s) {
      case 'metal':   this._fxMetal(px, py, pz, nx, ny, nz, k, floorY); break;
      case 'sand':    this._fxSand(px, py, pz, nx, ny, nz, k, floorY); break;
      case 'wood':    this._fxWood(px, py, pz, nx, ny, nz, k, floorY); break;
      case 'glass':   this._fxGlass(px, py, pz, nx, ny, nz, k, floorY); break;
      case 'flesh':   this.blood(point, normal, scale ?? 1, false); break;
      default:        this._fxConcrete(px, py, pz, nx, ny, nz, k, floorY); break;
    }
  }

  _fxConcrete(px, py, pz, nx, ny, nz, k, floorY) {
    this._basis(nx, ny, nz);
    // grey dust puff
    let n = Math.max(2, Math.round(7 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(0.95, 0.6);
      const sp = rr(1.4, 4.2);
      this._pushSmoke(px, py, pz, d.x * sp, d.y * sp + 0.5, d.z * sp,
        rr(0.85, 1.55), rr(0.09, 0.17), rr(0.50, 0.95), 3.6,
        0.74, 0.70, 0.62, 0.62, 0.60, 0.56, 0.50, 0.0,
        0.22, (Math.random() - 0.5) * 2.2, 0.55, 0.55);
    }
    // chips
    this._basis(nx, ny, nz);
    n = Math.max(3, Math.round(13 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.05, 0.55);
      const sp = rr(3, 9.5);
      this._pushDebris(px, py, pz, d.x * sp, d.y * sp + 0.8, d.z * sp,
        rr(0.5, 1.1), rr(0.014, 0.042), 1.0, 0.5,
        0.56, 0.53, 0.48, 1.0, 1.0, (Math.random() - 0.5) * 30, 0, floorY);
    }
    // lingering wisp that hangs and drifts
    for (let i = 0; i < 2; i++) {
      const d = this._cone(0.6, 0.8);
      this._pushSmoke(px + nx * 0.08, py + ny * 0.08, pz + nz * 0.08,
        d.x * 0.5, d.y * 0.5 + 0.35, d.z * 0.5,
        rr(2.6, 4.0), 0.22, rr(1.1, 1.7), 1.1,
        0.76, 0.72, 0.65, 0.26, 0.66, 0.63, 0.58, 0.0,
        -0.045, (Math.random() - 0.5) * 0.7, 1.0, 0.75);
    }
  }

  _fxMetal(px, py, pz, nx, ny, nz, k, floorY) {
    this._basis(nx, ny, nz);
    const n = Math.max(6, Math.round(22 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.30, 0.45);
      const sp = rr(4, 17);
      this._pushSpark(px, py, pz, d.x * sp, d.y * sp + 0.6, d.z * sp,
        rr(0.22, 0.72), rr(0.008, 0.017), 0.62, 1.7,
        9.5, 5.4, 1.7, 1.0, 2.6, 0.36, 0.04,
        1.0, 0.0, 64, 1.0);
    }
    this._basis(nx, ny, nz);
    for (let i = 0; i < 3; i++) {
      const d = this._cone(0.8, 0.7);
      const sp = rr(1.0, 2.6);
      this._pushSmoke(px, py, pz, d.x * sp, d.y * sp + 0.4, d.z * sp,
        rr(0.5, 0.9), 0.06, rr(0.22, 0.40), 4.5,
        0.62, 0.58, 0.55, 0.34, 0.55, 0.52, 0.50, 0.0,
        0.1, (Math.random() - 0.5) * 3, 0.6, 0.5);
    }
    this._basis(nx, ny, nz);
    for (let i = 0; i < 3; i++) {
      const d = this._cone(1.0, 0.6);
      const sp = rr(2.5, 7);
      this._pushDebris(px, py, pz, d.x * sp, d.y * sp, d.z * sp,
        rr(0.4, 0.8), rr(0.010, 0.024), 1.0, 0.5,
        0.60, 0.58, 0.56, 1.0, 1.0, (Math.random() - 0.5) * 34, 0, floorY);
    }
    this._tmp.set(px + nx * 0.16, py + ny * 0.16, pz + nz * 0.16);
    this.flashLight(this._tmp, 0xffc46a, 5.5, 0.075, 4.5);
  }

  _fxSand(px, py, pz, nx, ny, nz, k, floorY) {
    this._basis(nx, ny, nz);
    let n = Math.max(3, Math.round(11 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(0.85, 0.7);
      const sp = rr(1.1, 3.4);
      this._pushSmoke(px, py, pz, d.x * sp, d.y * sp + 0.6, d.z * sp,
        rr(1.1, 2.1), rr(0.10, 0.20), rr(0.65, 1.25), 2.6,
        0.84, 0.72, 0.53, 0.58, 0.76, 0.66, 0.50, 0.0,
        0.30, (Math.random() - 0.5) * 1.6, 0.85, 0.85);
    }
    this._basis(nx, ny, nz);
    n = Math.max(3, Math.round(9 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.0, 0.6);
      const sp = rr(2, 6.5);
      this._pushDebris(px, py, pz, d.x * sp, d.y * sp + 0.5, d.z * sp,
        rr(0.45, 0.85), rr(0.008, 0.020), 1.0, 0.9,
        0.78, 0.68, 0.50, 1.0, 1.0, (Math.random() - 0.5) * 20, 3, floorY);
    }
    this._pushSmoke(px, py, pz, 0, 0.35, 0,
      rr(2.4, 3.6), 0.30, rr(1.3, 2.0), 1.0,
      0.86, 0.75, 0.57, 0.24, 0.78, 0.70, 0.56, 0.0,
      -0.03, (Math.random() - 0.5) * 0.5, 1.1, 0.95);
  }

  _fxWood(px, py, pz, nx, ny, nz, k, floorY) {
    this._basis(nx, ny, nz);
    let n = Math.max(4, Math.round(15 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.0, 0.5);
      const sp = rr(3, 10);
      this._pushDebris(px, py, pz, d.x * sp, d.y * sp + 0.8, d.z * sp,
        rr(0.55, 1.15), rr(0.020, 0.055), rr(2.2, 4.2), 0.45,
        0.54, 0.39, 0.23, 1.0, 1.0, (Math.random() - 0.5) * 26, 1, floorY);
    }
    this._basis(nx, ny, nz);
    n = Math.max(2, Math.round(5 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(0.8, 0.7);
      const sp = rr(1.0, 2.8);
      this._pushSmoke(px, py, pz, d.x * sp, d.y * sp + 0.4, d.z * sp,
        rr(0.7, 1.3), 0.07, rr(0.30, 0.55), 3.8,
        0.72, 0.62, 0.47, 0.38, 0.62, 0.55, 0.44, 0.0,
        0.18, (Math.random() - 0.5) * 2.4, 0.7, 0.6);
    }
  }

  _fxGlass(px, py, pz, nx, ny, nz, k, floorY) {
    this._basis(nx, ny, nz);
    let n = Math.max(4, Math.round(17 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.15, 0.5);
      const sp = rr(2.5, 9);
      this._pushDebris(px, py, pz, d.x * sp, d.y * sp + 0.6, d.z * sp,
        rr(0.6, 1.3), rr(0.014, 0.048), rr(0.8, 1.6), 0.35,
        0.84, 0.90, 0.94, 1.0, 1.0, (Math.random() - 0.5) * 30, 2, floorY);
    }
    // glitter: hard, twinkling specular sparkle as the shards catch the sun
    this._basis(nx, ny, nz);
    n = Math.max(6, Math.round(20 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.25, 0.5);
      const sp = rr(2, 8);
      this._pushSpark(px, py, pz, d.x * sp, d.y * sp + 0.5, d.z * sp,
        rr(0.45, 1.05), 0.012, 0.10, 0.5,
        5.5, 6.4, 7.2, 1.0, 1.1, 1.5, 2.0,
        1.0, 1.0, 26, 1.0);
    }
    this._basis(nx, ny, nz);
    for (let i = 0; i < 2; i++) {
      const d = this._cone(0.8, 0.8);
      this._pushSmoke(px, py, pz, d.x * 1.2, d.y * 1.2 + 0.3, d.z * 1.2,
        rr(0.5, 0.9), 0.06, 0.26, 4.0,
        0.78, 0.80, 0.82, 0.22, 0.72, 0.74, 0.76, 0.0,
        0.2, (Math.random() - 0.5) * 2, 0.6, 0.5);
    }
  }

  blood(point, normal, amount, headshot) {
    if (!this._ready || !point) return;
    const k = Math.max(0.35, (amount ?? 1)) * this.burstScale * (headshot ? 1.7 : 1);
    const nx = normal?.x ?? 0, ny = normal?.y ?? 0.35, nz = normal?.z ?? 0;
    const px = point.x, py = point.y, pz = point.z;

    // fine mist — soft, unlit, no silver rim (blood does not silver)
    this._basis(nx, ny, nz);
    let n = Math.max(3, Math.round(8 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.15, 0.6);
      const sp = rr(1.2, 4.0);
      this._pushSmoke(px, py, pz, d.x * sp, d.y * sp + 0.3, d.z * sp,
        rr(0.45, 0.85), rr(0.06, 0.12), rr(0.26, 0.52), 5.0,
        0.46, 0.055, 0.040, 0.60, 0.26, 0.030, 0.024, 0.0,
        0.55, (Math.random() - 0.5) * 2.5, 0.25, 0.10);
    }
    // droplets
    this._basis(nx, ny, nz);
    n = Math.max(4, Math.round(14 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.25, 0.5);
      const sp = rr(2.0, 7.5);
      this._pushDebris(px, py, pz, d.x * sp, d.y * sp + 0.4, d.z * sp,
        rr(0.5, 1.15), rr(0.008, 0.024), 1.25, 0.30,
        0.38, 0.035, 0.028, 1.0, 1.0, (Math.random() - 0.5) * 12, 3, -1e6);
    }
    // let the decal system leave the splatter, if it wants it
    try {
      const dec = this.g?.decals;
      dec?.blood?.(point, normal, 0.35 + 0.5 * k);
    } catch (e) { /* Decals owns its own signature */ }
  }

  muzzle(position, direction, scale) {
    if (!this._ready || !position || !direction) return;
    const k = (scale ?? 1);
    const dx = direction.x, dy = direction.y, dz = direction.z;
    const px = position.x + dx * 0.10, py = position.y + dy * 0.10, pz = position.z + dz * 0.10;

    this._basis(dx, dy, dz);
    // lingering propellant smoke — accumulates naturally under sustained fire
    const n = Math.max(2, Math.round(3 * k * this.burstScale));
    for (let i = 0; i < n; i++) {
      const d = this._cone(0.42, 0.8);
      const sp = rr(1.3, 3.0);
      this._pushSmoke(px, py, pz, d.x * sp, d.y * sp + 0.18, d.z * sp,
        rr(1.2, 2.4), rr(0.035, 0.070), rr(0.34, 0.62), 3.1,
        0.74, 0.71, 0.66, 0.20, 0.66, 0.64, 0.60, 0.0,
        -0.055, (Math.random() - 0.5) * 2.0, 1.0, 0.80);
    }
    // one hot flash puff at the crown
    this._pushFire(px, py, pz, dx * 2.2, dy * 2.2 + 0.1, dz * 2.2,
      0.055, 0.10, 0.26, 6.0,
      1.00, 0.72, 0.34, 0.9, 0.85, 0.30, 0.08,
      -0.2, (Math.random() - 0.5) * 4, 0.4, 5.5);
    // unburnt powder
    this._basis(dx, dy, dz);
    for (let i = 0; i < 4; i++) {
      const d = this._cone(0.5, 0.6);
      const sp = rr(3, 9);
      this._pushSpark(px, py, pz, d.x * sp, d.y * sp, d.z * sp,
        rr(0.10, 0.28), 0.010, 0.22, 3.0,
        8.0, 4.2, 1.1, 1.0, 2.0, 0.30, 0.03,
        1.0, 0.0, 70, 0.9);
    }
  }

  footstep(position, surface, speed, atFeet) {
    if (!this._ready || !position) return;
    const s = SURFACE_ALIAS[String(surface || 'sand').toLowerCase()] || 'concrete';
    if (s === 'metal' || s === 'glass' || s === 'flesh') return;
    const dusty = (s === 'sand');
    const k = Math.min(1.6, Math.max(0.3, (speed ?? 3) / 3.4)) * this.burstScale;
    const px = position.x;
    const py = position.y - (atFeet ? (this.g?.config?.player?.eyeHeight ?? 1.62) : 0) + 0.03;
    const pz = position.z;

    const n = Math.max(2, Math.round((dusty ? 5 : 3) * k));
    this._basis(0, 1, 0);
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.35, 0.5);
      const sp = rr(0.3, 1.0) * k;
      this._pushSmoke(px, py, pz, d.x * sp, Math.abs(d.y) * sp * 0.7 + 0.25, d.z * sp,
        rr(0.55, 1.05), rr(0.05, 0.10), rr(0.26, 0.52), 3.2,
        dusty ? 0.84 : 0.74, dusty ? 0.73 : 0.70, dusty ? 0.55 : 0.63,
        (dusty ? 0.30 : 0.18) * k,
        dusty ? 0.78 : 0.68, dusty ? 0.68 : 0.65, dusty ? 0.53 : 0.60, 0.0,
        0.10, (Math.random() - 0.5) * 1.6, 0.9, 0.8);
    }
  }

  explosion(point, radius, power) {
    if (!this._ready || !point) return;
    const R = Math.max(0.6, radius ?? 4);
    const P = Math.max(0.2, power ?? 1);
    const k = this.burstScale;
    const px = point.x, py = point.y, pz = point.z;

    // --- fireball ---------------------------------------------------------
    let n = Math.max(6, Math.round(24 * k));
    for (let i = 0; i < n; i++) {
      this._basis(rr(-1, 1), rr(-0.3, 1), rr(-1, 1));
      const d = this._cone(1.4, 0.5);
      const sp = rr(2, 7.5) * P;
      const jx = rr(-0.3, 0.3) * R, jy = rr(-0.15, 0.35) * R, jz = rr(-0.3, 0.3) * R;
      this._pushFire(px + jx, py + jy, pz + jz,
        d.x * sp, d.y * sp + 1.2, d.z * sp,
        rr(0.30, 0.78), R * rr(0.22, 0.40), R * rr(0.75, 1.25), 2.6,
        1.00, 0.70, 0.30, 1.0, 0.95, 0.20, 0.035,
        -0.30, (Math.random() - 0.5) * 3, 0.75, rr(2.2, 4.4) * P);
    }

    // --- shockwave rings ---------------------------------------------------
    this._pushRing(px, py, pz, 0.36, R * 0.18, R * 1.75, 0.34, 0,
      1.00, 0.86, 0.58, 0.85, 1.00, 0.40, 0.10, 2.6 * P);
    this._pushRing(px, py + 0.10, pz, 0.60, R * 0.28, R * 2.60, 0.26, 1,
      1.00, 0.78, 0.48, 0.75, 0.90, 0.44, 0.18, 1.7 * P);

    // --- dark rolling smoke column ----------------------------------------
    n = Math.max(8, Math.round(28 * k));
    for (let i = 0; i < n; i++) {
      this._basis(rr(-1, 1), rr(0.2, 1), rr(-1, 1));
      const d = this._cone(1.1, 0.6);
      const sp = rr(1.2, 4.5) * P;
      this._pushSmoke(px + rr(-0.4, 0.4) * R, py + rr(-0.1, 0.6) * R, pz + rr(-0.4, 0.4) * R,
        d.x * sp, Math.abs(d.y) * sp + rr(1.0, 3.4), d.z * sp,
        rr(3.2, 6.4), R * rr(0.30, 0.55), R * rr(1.5, 2.7), 1.05,
        0.115, 0.100, 0.092, 0.90, 0.40, 0.36, 0.32, 0.0,
        -0.16, (Math.random() - 0.5) * 1.2, 1.0, 0.55);
    }

    // --- ground dust ring --------------------------------------------------
    n = Math.max(8, Math.round(24 * k));
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + rr(-0.12, 0.12);
      const ca = Math.cos(a), sa = Math.sin(a);
      const sp = rr(5, 11) * P;
      this._pushSmoke(px + ca * R * 0.35, py + 0.10, pz + sa * R * 0.35,
        ca * sp, rr(0.4, 1.4), sa * sp,
        rr(2.0, 3.6), R * 0.28, R * rr(1.0, 1.6), 2.1,
        0.80, 0.70, 0.53, 0.62, 0.72, 0.64, 0.51, 0.0,
        0.10, (Math.random() - 0.5) * 1.4, 0.9, 0.85);
    }

    // --- ejecta ------------------------------------------------------------
    n = Math.max(10, Math.round(34 * k));
    for (let i = 0; i < n; i++) {
      this._basis(rr(-1, 1), rr(0.1, 1), rr(-1, 1));
      const d = this._cone(1.3, 0.5);
      const sp = rr(6, 19) * P;
      this._pushDebris(px, py + 0.1, pz, d.x * sp, Math.abs(d.y) * sp + 2.0, d.z * sp,
        rr(0.9, 2.0), rr(0.02, 0.075), rr(0.8, 2.2), 0.35,
        0.52, 0.47, 0.41, 1.0, 1.0, (Math.random() - 0.5) * 28,
        Math.random() < 0.7 ? 0 : 1, py - 0.05);
    }

    // --- sparks -------------------------------------------------------------
    this._basis(0, 1, 0);
    n = Math.max(10, Math.round(44 * k));
    for (let i = 0; i < n; i++) {
      const d = this._cone(1.5, 0.45);
      const sp = rr(6, 22) * P;
      this._pushSpark(px, py + 0.1, pz, d.x * sp, Math.abs(d.y) * sp * 0.8 + 1.5, d.z * sp,
        rr(0.5, 1.5), rr(0.012, 0.024), 0.85, 0.9,
        10.0, 5.6, 1.6, 1.0, 2.8, 0.40, 0.05,
        1.0, 0.0, 48, 1.0);
    }

    this._tmp.set(px, py + R * 0.3, pz);
    this.flashLight(this._tmp, 0xffb057, 55 * P, 0.45, R * 7);
  }

  /** Debug: fire one of everything just in front of the camera. */
  test() {
    const cam = this.g?.camera;
    if (!cam || !this._ready) return this;
    const p = this._tmp;
    const f = this._dir.set(0, 0, -1).applyQuaternion(cam.quaternion);
    const surfaces = ['concrete', 'metal', 'sand', 'wood', 'glass', 'flesh'];
    for (let i = 0; i < surfaces.length; i++) {
      const off = (i - 2.5) * 1.1;
      p.set(cam.position.x + f.x * 6 - f.z * off, cam.position.y - 0.4 + (i % 2) * 0.5, cam.position.z + f.z * 6 + f.x * off);
      this.impact(p, { x: -f.x, y: 0.25, z: -f.z }, surfaces[i], 1.4);
    }
    p.set(cam.position.x + f.x * 9, cam.position.y - 0.6, cam.position.z + f.z * 9);
    this.explosion(p, 3.2, 1.1);
    p.set(cam.position.x + f.x * 0.6, cam.position.y - 0.15, cam.position.z + f.z * 0.6);
    this.muzzle(p, f, 1);
    return this;
  }
}

export default Particles;
