// ============================================================================
// BLACKSITE — src/art/Materials.js
// Owner: Materials agent.  Procedural PBR material library.
//
// Every surface in the game is generated here, in code, at init time.  No file
// loads, no CDN, no network.  Each surface is built from a HEIGHT FIELD first;
// albedo, roughness, AO and the normal map are all derived from that same
// field so bumps and colour always agree.
//
// Public API (stable — other modules depend on exactly this):
//   game.materials.get('concrete')                      -> THREE.MeshStandardMaterial (cached, shared)
//   game.materials.make('concrete', { repeat:[4,4], tint:0xffffff, roughness:0.9 })
//   game.materials.getEnv()                             -> PMREM env map (THREE.Texture) or null
//   game.materials.keys                                 -> string[] of surface keys
//   game.materials.has(key)                             -> bool
//   game.materials.surfaceFor(materialOrKey)            -> 'concrete'|'metal'|'sand'|'wood'|'flesh'|'glass'
//   game.materials.stats                                -> { ms:{key:ms}, totalMs, textures, missing }
//   game.materials.trimCache()                          -> free noise scratch (auto-runs 6s idle)
//
// Texture layout per surface:
//   map        RGBA8, sRGB          albedo (+ grime, sun bleach, run-off streaks)
//   normalMap  RGBA8, linear        tangent-space, OpenGL/Y+ convention, Sobel of height
//   ormMap     RGBA8, linear        R = AO, G = roughness, B = metalness  (glTF ORM packing)
//              -> bound simultaneously as aoMap / roughnessMap / metalnessMap
//   + a shared 256px detail normal blended in at high tiling frequency by a
//     small shader patch, so surfaces still read at 20cm from the muzzle.
//
// Material scalars are multipliers over the maps (glTF convention): roughness
// and metalness default to 1.0 so the map value is what you get; make() lets a
// caller scale them.
//
// Everything is generated ON DEMAND and cached.  init() only builds the env map
// plus three hero surfaces so the cost lands inside the loading bar.
// ============================================================================

import * as THREE from 'three';

/* ------------------------------------------------------------------ *
 *  PRNG + tiny math
 * ------------------------------------------------------------------ */

function mulberry32(a) {
  a = (a >>> 0) || 0x9e3779b9;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// deterministic 2D integer hash -> [0,1)
function hash2i(x, y, s) {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(s | 0, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function hash1i(x, s) { return hash2i(x, 0x51ed, s); }

const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
function sstep(e0, e1, x) {
  if (e1 === e0) return x < e0 ? 0 : 1;
  let t = (x - e0) / (e1 - e0);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}
// hex -> [r,g,b] floats in *display* (sRGB) space; albedo is authored here
const C = hex => [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];

function normalize01(f) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < f.length; i++) { const v = f[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
  const d = mx - mn;
  if (d < 1e-6) { f.fill(0.5); return f; }
  const inv = 1 / d;
  for (let i = 0; i < f.length; i++) f[i] = (f[i] - mn) * inv;
  return f;
}

// Sobel gain: strength * RES * SOBEL_K.  Tuned so a typical fBm surface lands
// around 15-20 degrees of average slope — bumpy enough to catch the raking sun,
// never the shimmering noise-storm of an untuned normal map.
const SOBEL_K = 0.0028;

/* ------------------------------------------------------------------ *
 *  Surface definitions
 *
 *  res     texture resolution
 *  seed    PRNG seed (stable results between runs)
 *  impact  bus 'impact' surface bucket
 *  cavK    cavity contrast used for AO + dirt occlusion
 *  ao      AO strength baked into the ORM red channel
 *  dirt    how much grime collects in the crevices
 *  streak  vertical run-off streaking from ledges
 *  bleach  sun-bleaching of raised / exposed faces
 *  wear    polish on raised areas (traffic / handling)
 *  nrm     normal map strength
 *  detail  [tilingMultiplier, strength] for the micro detail normal, or null
 * ------------------------------------------------------------------ */

const DEFS = {
  concrete: {
    res: 1024, seed: 1301, build: '_bConcrete', impact: 'concrete',
    cavK: 5.5, ao: 0.95, dirt: 0.85, streak: 0.60, bleach: 0.45, wear: 0.55, nrm: 1.15,
    detail: [11, 0.55], metalness: 0.0, envI: 1.0,
  },
  plaster: {
    res: 1024, seed: 2207, build: '_bPlaster', impact: 'concrete',
    cavK: 6.5, ao: 0.85, dirt: 0.95, streak: 0.95, bleach: 0.70, wear: 0.30, nrm: 0.90,
    detail: [13, 0.45], metalness: 0.0, envI: 1.0,
  },
  brick: {
    res: 1024, seed: 3511, build: '_bBrick', impact: 'concrete',
    cavK: 3.2, ao: 1.15, dirt: 0.90, streak: 0.55, bleach: 0.40, wear: 0.45, nrm: 1.30,
    detail: [10, 0.55], metalness: 0.0, envI: 1.0,
  },
  sand: {
    res: 1024, seed: 4127, build: '_bSand', impact: 'sand',
    cavK: 6.0, ao: 0.70, dirt: 0.35, streak: 0.0, bleach: 0.55, wear: 0.20, nrm: 1.00,
    detail: [14, 0.60], metalness: 0.0, envI: 0.85,
  },
  asphalt: {
    res: 1024, seed: 5303, build: '_bAsphalt', impact: 'concrete',
    cavK: 7.0, ao: 1.0, dirt: 0.55, streak: 0.0, bleach: 0.35, wear: 0.75, nrm: 1.10,
    detail: [12, 0.55], metalness: 0.0, envI: 0.95,
  },
  rustmetal: {
    res: 1024, seed: 6199, build: '_bRustMetal', impact: 'metal',
    cavK: 6.0, ao: 1.0, dirt: 0.75, streak: 1.00, bleach: 0.20, wear: 0.40, nrm: 1.10,
    detail: [12, 0.45], metalness: 1.0, envI: 1.15,
  },
  metal: {
    res: 512, seed: 7027, build: '_bMetal', impact: 'metal',
    cavK: 7.0, ao: 0.75, dirt: 0.45, streak: 0.55, bleach: 0.10, wear: 0.80, nrm: 0.70,
    detail: [16, 0.35], metalness: 1.0, envI: 1.25,
  },
  wood: {
    res: 512, seed: 8093, build: '_bWood', impact: 'wood',
    cavK: 6.0, ao: 0.90, dirt: 0.70, streak: 0.45, bleach: 0.55, wear: 0.65, nrm: 0.95,
    detail: [14, 0.45], metalness: 0.0, envI: 0.95,
  },
  fabric: {
    res: 512, seed: 9151, build: '_bFabric', impact: 'concrete',
    cavK: 6.5, ao: 1.05, dirt: 0.70, streak: 0.30, bleach: 0.45, wear: 0.35, nrm: 0.55,
    detail: [10, 0.30], metalness: 0.0, envI: 0.7,
  },
  cloth_tan: {
    res: 512, seed: 10321, build: '_bClothTan', impact: 'concrete',
    cavK: 6.5, ao: 1.0, dirt: 0.65, streak: 0.25, bleach: 0.50, wear: 0.45, nrm: 0.50,
    detail: [10, 0.30], metalness: 0.0, envI: 0.7,
  },
  rubber: {
    res: 512, seed: 11447, build: '_bRubber', impact: 'concrete',
    cavK: 6.0, ao: 0.85, dirt: 0.50, streak: 0.20, bleach: 0.20, wear: 0.55, nrm: 0.80,
    detail: [15, 0.35], metalness: 0.0, envI: 0.6,
  },
  glass: {
    res: 512, seed: 12503, build: '_bGlass', impact: 'glass',
    cavK: 3.0, ao: 0.25, dirt: 0.55, streak: 0.85, bleach: 0.0, wear: 0.0, nrm: 0.30,
    detail: [18, 0.15], metalness: 0.0, envI: 2.2,
  },
  gunmetal: {
    res: 512, seed: 13649, build: '_bGunmetal', impact: 'metal',
    cavK: 7.5, ao: 0.80, dirt: 0.35, streak: 0.10, bleach: 0.05, wear: 0.90, nrm: 0.65,
    detail: [20, 0.40], metalness: 1.0, envI: 1.35,
  },
  polymer: {
    res: 512, seed: 14771, build: '_bPolymer', impact: 'concrete',
    cavK: 7.0, ao: 0.75, dirt: 0.30, streak: 0.10, bleach: 0.15, wear: 0.70, nrm: 0.70,
    detail: [18, 0.35], metalness: 0.0, envI: 0.9,
  },
  flesh: {
    res: 512, seed: 15877, build: '_bFlesh', impact: 'flesh',
    cavK: 6.0, ao: 0.70, dirt: 0.20, streak: 0.0, bleach: 0.10, wear: 0.30, nrm: 0.60,
    detail: [16, 0.30], metalness: 0.0, envI: 0.8,
  },
  foliage: {
    res: 512, seed: 16963, build: '_bFoliage', impact: 'wood',
    cavK: 5.0, ao: 0.85, dirt: 0.30, streak: 0.0, bleach: 0.60, wear: 0.10, nrm: 0.75,
    detail: [12, 0.35], metalness: 0.0, envI: 0.75,
  },
};

// Friendly aliases so a collaborator asking for a reasonable-sounding surface
// never gets a hard failure or a magenta box.
const ALIAS = {
  stone: 'concrete', cement: 'concrete', rock: 'concrete', kerb: 'concrete', curb: 'concrete',
  wall: 'plaster', stucco: 'plaster', paint: 'plaster', painted: 'plaster', tile: 'plaster',
  ceramic: 'plaster', ceiling: 'plaster',
  masonry: 'brick', clay: 'brick',
  dirt: 'sand', ground: 'sand', gravel: 'sand', dust: 'sand', soil: 'sand',
  road: 'asphalt', tarmac: 'asphalt', street: 'asphalt',
  steel: 'metal', iron: 'metal', aluminum: 'metal', aluminium: 'metal', chrome: 'metal',
  rust: 'rustmetal', corrugated: 'rustmetal', container: 'rustmetal', barrel: 'rustmetal',
  plank: 'wood', crate: 'wood', timber: 'wood', plywood: 'wood',
  cloth: 'cloth_tan', canvas: 'fabric', tarp: 'fabric', carpet: 'fabric', burlap: 'fabric',
  tan: 'cloth_tan', nylon: 'cloth_tan', webbing: 'cloth_tan',
  tyre: 'rubber', tire: 'rubber', mat: 'rubber',
  window: 'glass', mirror: 'glass',
  gun: 'gunmetal', weapon: 'gunmetal',
  plastic: 'polymer', abs: 'polymer',
  skin: 'flesh', body: 'flesh', head: 'flesh',
  leaf: 'foliage', leaves: 'foliage', grass: 'foliage', palm: 'foliage', bush: 'foliage',
};

/* ------------------------------------------------------------------ *
 *  Detail-normal shader patch
 *  Re-implements <normal_fragment_maps> with a second, high-frequency
 *  normal blended in (UDN-style) and faded out with view distance.
 * ------------------------------------------------------------------ */

const DETAIL_PARS = /* glsl */`
uniform sampler2D uDetailNormal;
uniform vec2  uDetailScale;
uniform float uDetailStrength;
uniform float uDetailFade;
`;

const DETAIL_MAPS = /* glsl */`
#ifdef USE_NORMALMAP_OBJECTSPACE

	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;

	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif

	normal = normalize( normalMatrix * normal );

#elif defined( USE_NORMALMAP_TANGENTSPACE )

	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;

	vec3 detN = texture2D( uDetailNormal, vNormalMapUv * uDetailScale ).xyz * 2.0 - 1.0;
	float bsFade = 1.0 - smoothstep( uDetailFade * 0.35, uDetailFade, length( vViewPosition ) );
	mapN.xy += detN.xy * ( uDetailStrength * bsFade );

	normal = normalize( tbn * mapN );

#elif defined( USE_BUMPMAP )

	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );

#endif
`;

/* ================================================================== *
 *  Materials
 * ================================================================== */

export class Materials {
  constructor(game) {
    this.g = game;
    this.THREE = THREE;

    this.keys = Object.keys(DEFS);
    this._mats = new Map();      // key                -> base material (repeat 1,1)
    this._tex = new Map();       // key                -> { map, normalMap, ormMap }
    this._texVar = new Map();    // key|ru|rv          -> cloned texture set
    this._tables = new Map();    // "N|n"              -> interpolation tables
    this._shf = new Map();       // "N|name"           -> shared generic noise field
    this._detail = null;         // shared micro normal
    this._env = null;
    this._envRT = null;
    this._trimT = 0;

    this.stats = { ms: {}, totalMs: 0, textures: 0, missing: [] };

    const cfg = game?.config?.render;
    let maxAniso = cfg?.maxAnisotropy ?? 8;
    const caps = game?.renderer?.renderer?.capabilities;
    if (caps?.getMaxAnisotropy) maxAniso = Math.min(maxAniso, caps.getMaxAnisotropy());
    this.anisotropy = Math.max(1, maxAniso | 0);

    // Every patched material emits byte-identical GLSL, so a single constant
    // cache key lets them all share one compiled program.
    this._cacheKey = () => 'bs-detail-1';
  }

  /* ---------------------------------------------------------------- *
   *  Lifecycle
   * ---------------------------------------------------------------- */

  async init() {
    const t0 = this._now();

    // Environment first — Sky (built after us) may replace scene.environment
    // with its own PMREM; until then every material still gets a correct,
    // golden-hour specular response instead of a flat black reflection.
    const env = this.getEnv();
    const sc = this.g?.scene;
    if (env && sc && !sc.environment) {
      sc.environment = env;
      if ('environmentIntensity' in sc) sc.environmentIntensity = 1.0;
    }

    this._detailTex();

    // Pre-warm the surfaces the level is guaranteed to ask for, so the cost
    // lands inside the loading bar rather than on the first rendered frame.
    for (const k of ['concrete', 'plaster', 'sand']) {
      this.get(k);
      await new Promise(r => setTimeout(r, 0));  // let the boot bar paint
    }

    this.stats.totalMs = +(this._now() - t0).toFixed(1);
  }

  update(/* dt, t */) { /* materials are static — no per-frame work, no allocation */ }

  resize(/* w, h */) { }

  /* ---------------------------------------------------------------- *
   *  Public API
   * ---------------------------------------------------------------- */

  has(key) { return !!DEFS[this._key(key)]; }

  /** Cached, shared material at repeat 1,1. Do not mutate the result. */
  get(key) {
    const k = this._key(key);
    let m = this._mats.get(k);
    if (m) return m;
    m = this._build(k, {});
    this._mats.set(k, m);
    return m;
  }

  /**
   * A fresh material variant. Texture clones share GPU memory with the base.
   *   repeat        [u,v] tiling
   *   tint          hex multiplied over the albedo
   *   roughness     multiplier over the roughness map (default 1)
   *   metalness     multiplier over the metalness map
   *   normalScale   scalar or [x,y]
   *   aoIntensity, envMapIntensity, opacity, transparent, side, alphaTest,
   *   flatShading, depthWrite, detailScale, detailStrength, detailFade, name
   */
  make(key, opts = {}) {
    return this._build(this._key(key), opts || {});
  }

  /** Small procedurally-generated PMREM environment (warm sky + ground bounce). */
  getEnv() {
    if (this._env) return this._env;
    const renderer = this.g?.renderer?.renderer;
    if (!renderer) return null;
    try {
      const eq = this._equirect(256, 128);
      const pmrem = new THREE.PMREMGenerator(renderer);
      pmrem.compileEquirectangularShader();
      const rt = pmrem.fromEquirectangular(eq);
      pmrem.dispose();
      eq.dispose();
      this._envRT = rt;
      this._env = rt.texture;
      return this._env;
    } catch (e) {
      this.stats.missing.push('env');
      return null;
    }
  }

  /** Map any material/key onto the six canonical Bus 'impact' surfaces. */
  surfaceFor(x) {
    if (!x) return 'concrete';
    if (typeof x === 'string') return DEFS[this._key(x)]?.impact ?? 'concrete';
    const k = x.userData?.bsSurface;
    return (k && DEFS[k]?.impact) || 'concrete';
  }

  /**
   * Release noise scratch. By default only the expensive high-resolution pool
   * is dropped (~36MB); the 512px pool is kept so a late, rare surface request
   * during gameplay still costs ~100ms rather than rebuilding everything.
   * trimCache(true) frees the lot.
   */
  trimCache(all = false) {
    if (all) { this._shf.clear(); this._tables.clear(); return; }
    for (const k of [...this._shf.keys()]) {
      if (parseInt(k, 10) > 512) this._shf.delete(k);
    }
  }

  dispose() {
    if (this._trimT) { clearTimeout(this._trimT); this._trimT = 0; }
    for (const set of this._tex.values()) for (const t of Object.values(set)) t?.dispose?.();
    for (const set of this._texVar.values()) for (const t of Object.values(set)) t?.dispose?.();
    for (const m of this._mats.values()) m.dispose?.();
    this._detail?.dispose?.();
    this._envRT?.dispose?.();
    this._tex.clear(); this._texVar.clear(); this._mats.clear();
    this.trimCache();
    this._detail = null; this._env = null; this._envRT = null;
  }

  /* ---------------------------------------------------------------- *
   *  Material construction
   * ---------------------------------------------------------------- */

  _now() { return (typeof performance !== 'undefined' ? performance.now() : Date.now()); }

  _key(key) {
    if (typeof key !== 'string') return 'concrete';
    const k = key.toLowerCase().trim();
    if (DEFS[k]) return k;
    if (ALIAS[k]) return ALIAS[k];
    const cut = k.split(/[^a-z]+/)[0];          // 'concrete_wall' -> 'concrete'
    if (DEFS[cut]) return cut;
    if (ALIAS[cut]) return ALIAS[cut];
    if (!this.stats.missing.includes(key)) this.stats.missing.push(key);
    return 'concrete';
  }

  _build(k, o) {
    const d = DEFS[k];
    const ru = o.repeat ? (o.repeat[0] ?? 1) : 1;
    const rv = o.repeat ? (o.repeat[1] ?? ru) : 1;
    const set = this._texSet(k, ru, rv);

    const mat = new THREE.MeshStandardMaterial({
      name: 'bs:' + k,
      map: set.map,
      normalMap: set.normalMap,
      roughnessMap: set.ormMap,
      metalnessMap: set.ormMap,
      aoMap: set.ormMap,
      // the maps carry the real values; these scalars act as multipliers
      color: new THREE.Color(o.tint ?? o.color ?? 0xffffff),
      roughness: o.roughness ?? 1.0,
      metalness: o.metalness ?? 1.0,
      envMapIntensity: o.envMapIntensity ?? d.envI ?? 1.0,
      aoMapIntensity: o.aoIntensity ?? 1.0,
      dithering: true,
    });

    const ns = o.normalScale;
    if (Array.isArray(ns)) mat.normalScale.set(ns[0], ns[1]);
    else if (typeof ns === 'number') mat.normalScale.set(ns, ns);

    if (k === 'glass') {
      mat.transparent = true;
      mat.opacity = o.opacity ?? 0.30;
      mat.side = o.side ?? THREE.DoubleSide;
      mat.depthWrite = o.depthWrite ?? false;
    } else if (k === 'foliage') {
      mat.side = o.side ?? THREE.DoubleSide;
    }

    if (o.side !== undefined) mat.side = o.side;
    if (o.transparent !== undefined) mat.transparent = o.transparent;
    if (o.opacity !== undefined) { mat.opacity = o.opacity; if (o.opacity < 1) mat.transparent = true; }
    if (o.depthWrite !== undefined) mat.depthWrite = o.depthWrite;
    if (o.alphaTest !== undefined) mat.alphaTest = o.alphaTest;
    if (o.flatShading !== undefined) mat.flatShading = o.flatShading;
    if (o.name) mat.name = o.name;

    mat.userData.bsSurface = k;
    mat.userData.bsImpact = d.impact;

    // Micro-detail normal. onBeforeCompile is invoked as (shader, renderer),
    // so the material is captured in the closure, not read from an argument.
    const det = d.detail;
    if (det) {
      const s = o.detailScale ?? det[0];
      const uni = {
        scale: new THREE.Vector2(s * ru, s * rv),
        strength: o.detailStrength ?? det[1],
        fade: o.detailFade ?? 22.0,
      };
      mat.userData.bsDetail = uni;
      mat.onBeforeCompile = (shader) => {
        shader.uniforms.uDetailNormal = { value: this._detailTex() };
        shader.uniforms.uDetailScale = { value: uni.scale };
        shader.uniforms.uDetailStrength = { value: uni.strength };
        shader.uniforms.uDetailFade = { value: uni.fade };
        shader.fragmentShader = shader.fragmentShader
          .replace('#include <normal_pars_fragment>', '#include <normal_pars_fragment>\n' + DETAIL_PARS)
          .replace('#include <normal_fragment_maps>', DETAIL_MAPS);
      };
      mat.customProgramCacheKey = this._cacheKey;
    }

    return mat;
  }

  _texSet(k, ru, rv) {
    const base = this._baseTex(k);
    if (ru === 1 && rv === 1) return base;
    const id = k + '|' + ru + '|' + rv;
    let v = this._texVar.get(id);
    if (v) return v;
    v = {};
    for (const name of ['map', 'normalMap', 'ormMap']) {
      const t = base[name].clone();
      t.repeat.set(ru, rv);
      t.needsUpdate = true;      // required: clones start at version 0
      v[name] = t;
    }
    this._texVar.set(id, v);
    return v;
  }

  _baseTex(k) {
    let s = this._tex.get(k);
    if (s) return s;
    const t0 = this._now();
    s = this._generate(k);
    this.stats.ms[k] = +(this._now() - t0).toFixed(1);
    this.stats.textures += 3;
    this._tex.set(k, s);
    this._scheduleTrim();
    return s;
  }

  /**
   * Free the big noise pool once the game has actually booted and stopped
   * asking for surfaces. Everything Level/Weapon/Enemies need is generated
   * during their init(), i.e. before game.ready, so this never runs while a
   * module is still streaming its materials in.
   */
  _scheduleTrim() {
    if (typeof setTimeout !== 'function') return;
    if (this._trimT) clearTimeout(this._trimT);
    this._trimT = setTimeout(() => {
      this._trimT = 0;
      if (this.g?.ready) this.trimCache();
      else this._scheduleTrim();
    }, 5000);
  }

  _res(r) {
    const q = this.g?.config?.quality;
    if (q === 'low') return Math.max(256, r >> 1);
    return r;
  }

  _mkTex(data, N, srgb) {
    const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.generateMipmaps = true;
    t.anisotropy = this.anisotropy;
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.needsUpdate = true;
    return t;
  }

  /* ---------------------------------------------------------------- *
   *  Noise / field toolbox
   * ---------------------------------------------------------------- */

  // cached smoothstep interpolation tables for a given (texture size, lattice size)
  _table(N, n) {
    const id = N + '|' + n;
    let t = this._tables.get(id);
    if (t) return t;
    const i0 = new Int32Array(N), i1 = new Int32Array(N), s = new Float32Array(N);
    for (let x = 0; x < N; x++) {
      const f = (x * n) / N;
      const i = Math.floor(f);
      const u = f - i;
      i0[x] = i % n;
      i1[x] = (i + 1) % n;
      s[x] = u * u * (3 - 2 * u);
    }
    t = { i0, i1, s };
    this._tables.set(id, t);
    return t;
  }

  // one tileable value-noise octave, accumulated into `out` (separable upsample)
  _octave(out, N, nx, ny, rnd, amp, ridged) {
    nx = Math.max(1, Math.min(nx | 0, N));
    ny = Math.max(1, Math.min(ny | 0, N));
    const lat = new Float32Array(nx * ny);
    for (let i = 0; i < lat.length; i++) lat[i] = rnd();

    const TX = this._table(N, nx), TY = this._table(N, ny);
    const x0 = TX.i0, x1 = TX.i1, sx = TX.s;
    const y0 = TY.i0, y1 = TY.i1, sy = TY.s;

    const tmp = new Float32Array(ny * N);
    for (let j = 0; j < ny; j++) {
      const ro = j * nx, to = j * N;
      for (let x = 0; x < N; x++) {
        const a = lat[ro + x0[x]], b = lat[ro + x1[x]];
        tmp[to + x] = a + (b - a) * sx[x];
      }
    }
    for (let y = 0; y < N; y++) {
      const r0 = y0[y] * N, r1 = y1[y] * N, t = sy[y], oo = y * N;
      if (ridged) {
        for (let x = 0; x < N; x++) {
          const a = tmp[r0 + x], b = tmp[r1 + x];
          let v = a + (b - a) * t;
          v = 1 - Math.abs(v * 2 - 1);
          out[oo + x] += v * amp;
        }
      } else {
        for (let x = 0; x < N; x++) {
          const a = tmp[r0 + x], b = tmp[r1 + x];
          out[oo + x] += (a + (b - a) * t) * amp;
        }
      }
    }
  }

  /** Multi-octave tileable fBm normalised to exactly [0,1]. */
  _fbm(N, o) {
    const oct = o.oct ?? 5, gain = o.gain ?? 0.5;
    const rnd = mulberry32(((o.seed | 0) * 2654435761) >>> 0);
    const out = new Float32Array(N * N);
    let nx = o.nx ?? 8, ny = o.ny ?? (o.nx ?? 8), amp = 1;
    for (let i = 0; i < oct; i++) {
      this._octave(out, N, nx, ny, rnd, amp, !!o.ridged);
      amp *= gain;
      if (nx >= N && ny >= N) break;
      nx *= 2; ny *= 2;
    }
    return o.raw ? out : normalize01(out);
  }

  /* --- shared generic fields -------------------------------------- *
   * Large / fine breakup noise is not what makes concrete look like
   * concrete — the height field and the colour logic are.  So the generic
   * layers are generated ONCE per resolution and handed to each surface as a
   * cheap toroidal translation, which is still a perfectly tileable, visually
   * uncorrelated field.  This roughly halves total generation time.
   * ---------------------------------------------------------------- */

  _sfOpts(N, name) {
    switch (name) {
      case 'fine': return { oct: 3, nx: Math.max(8, N >> 3), gain: 0.50, seed: 5001 };
      case 'grain': return { oct: 2, nx: Math.max(8, N >> 2), gain: 0.50, seed: 5002 };
      case 'pit': return { oct: 2, nx: Math.max(4, N >> 4), gain: 0.50, seed: 5003 };
      case 'grime': return { oct: 4, nx: 5, gain: 0.55, seed: 5004 };
      case 'grimeF': return { oct: 3, nx: 48, gain: 0.50, seed: 5005 };
      case 'bleach': return { oct: 3, nx: 3, gain: 0.52, seed: 5006 };
      case 'wear': return { oct: 4, nx: 4, gain: 0.55, seed: 5007 };
      case 'big': return { oct: 4, nx: 2, gain: 0.58, seed: 5008 };
      case 'mid': return { oct: 4, nx: 13, gain: 0.55, seed: 5009 };
      default: return { oct: 4, nx: 8, gain: 0.50, seed: 5010 };
    }
  }

  _sf(N, name) {
    const id = N + '|' + name;
    let f = this._shf.get(id);
    if (f) return f;
    f = (name === 'streak') ? this._streaks(N, 5100) : this._fbm(N, this._sfOpts(N, name));
    this._shf.set(id, f);
    return f;
  }

  /** Shared field, translated on the torus by a seed-derived offset. */
  _grab(N, name, seed) {
    const src = this._sf(N, name);
    const dx = 1 + ((hash1i(seed, 7717) * (N - 2)) | 0);
    const dy = ((hash1i(seed, 3313) * N) | 0) % N;
    const out = new Float32Array(N * N);
    const cut = N - dx;
    for (let y = 0; y < N; y++) {
      const sy = (((y + dy) % N) * N), oo = y * N;
      out.set(src.subarray(sy + dx, sy + N), oo);
      out.set(src.subarray(sy, sy + dx), oo + cut);
    }
    return out;
  }

  /** Tileable Worley F1 distance (grid units) + the winning cell's random id. */
  _cells(N, cx, cy, seed) {
    cx = Math.max(1, cx | 0); cy = Math.max(1, cy | 0);
    const rnd = mulberry32(((seed | 0) * 40503) >>> 0);
    const px = new Float32Array(cx * cy), py = new Float32Array(cx * cy), pr = new Float32Array(cx * cy);
    for (let i = 0; i < px.length; i++) { px[i] = 0.12 + rnd() * 0.76; py[i] = 0.12 + rnd() * 0.76; pr[i] = rnd(); }

    const wx = new Int32Array(cx + 2), wy = new Int32Array(cy + 2);
    for (let i = -1; i <= cx; i++) wx[i + 1] = ((i % cx) + cx) % cx;
    for (let i = -1; i <= cy; i++) wy[i + 1] = ((i % cy) + cy) % cy;

    const d = new Float32Array(N * N), id = new Float32Array(N * N);
    const kx = cx / N, ky = cy / N;
    for (let y = 0; y < N; y++) {
      const gy = (y + 0.5) * ky;
      const cy0 = Math.floor(gy);
      const oo = y * N;
      const rb0 = wy[cy0] * cx, rb1 = wy[cy0 + 1] * cx, rb2 = wy[cy0 + 2] * cx;
      for (let x = 0; x < N; x++) {
        const gx = (x + 0.5) * kx;
        const cx0 = Math.floor(gx);
        let best = 1e9, bestR = 0;
        for (let j = 0; j < 3; j++) {
          const cj = cy0 + j - 1;
          const rowBase = j === 0 ? rb0 : (j === 1 ? rb1 : rb2);
          for (let i = 0; i < 3; i++) {
            const ci = cx0 + i - 1;
            const p = rowBase + wx[ci + 1];
            const ddx = (ci + px[p]) - gx;
            const ddy = (cj + py[p]) - gy;
            const dd = ddx * ddx + ddy * ddy;
            if (dd < best) { best = dd; bestR = pr[p]; }
          }
        }
        d[oo + x] = Math.sqrt(best);
        id[oo + x] = bestR;
      }
    }
    return { d, id };
  }

  /** Separable wrapping box blur — used for cavity/AO and grime pooling. */
  _blur(src, N, r) {
    r = Math.max(1, Math.min(r | 0, (N >> 1) - 1));
    const w = 2 * r + 1, inv = 1 / w;
    const tmp = new Float32Array(N * N), out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) {
      const o = y * N;
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += src[o + ((k + N) % N)];
      for (let x = 0; x < N; x++) {
        tmp[o + x] = sum * inv;
        sum += src[o + ((x + r + 1) % N)] - src[o + ((x - r + N) % N)];
      }
    }
    for (let x = 0; x < N; x++) {
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += tmp[((k + N) % N) * N + x];
      for (let y = 0; y < N; y++) {
        out[y * N + x] = sum * inv;
        sum += tmp[((y + r + 1) % N) * N + x] - tmp[((y - r + N) % N) * N + x];
      }
    }
    return out;
  }

  /**
   * Dark run-off streaks descending from ledges: vertical striation × a
   * per-column start height with a falloff running downward (v decreasing).
   */
  _streaks(N, seed) {
    const out = new Float32Array(N * N);
    const vert = this._fbm(N, { oct: 4, nx: Math.max(64, N >> 2), ny: 3, gain: 0.55, seed: seed + 71 });
    const wide = this._fbm(N, { oct: 3, nx: 6, ny: 2, gain: 0.5, seed: seed + 72 });
    for (let x = 0; x < N; x++) {
      const h0 = hash1i(x, seed + 991);
      const h1 = hash1i(x, seed + 992);
      const h2 = hash1i(x, seed + 993);
      if (h2 > 0.55) continue;                       // only ~45% of columns run
      const start = Math.floor((0.45 + 0.5 * h0) * (N - 1));
      const len = Math.max(8, (0.12 + 0.5 * h1) * N);
      const amp = 0.45 + 0.55 * h1;
      const yEnd = Math.max(0, (start - len) | 0);
      for (let y = start; y >= yEnd; y--) {
        const f = (start - y) / len;                 // 0 at the ledge, 1 at the tail
        const fall = (1 - f) * (1 - f * 0.55);
        const i = y * N + x;
        out[i] = amp * fall * (0.35 + 0.9 * vert[i]) * (0.4 + 0.9 * wide[i]);
      }
    }
    return normalize01(out);
  }

  /* ---------------------------------------------------------------- *
   *  Generation pipeline
   * ---------------------------------------------------------------- */

  _generate(k) {
    const d = DEFS[k];
    const N = this._res(d.res);
    const b = this[d.build](N, d.seed);
    return this._finish(d, N, d.seed, b);
  }

  /**
   * Shared finishing pass: cavity AO, grime in crevices, sun bleach on raised
   * faces, run-off streaking, wear polish, ORM packing, Sobel normal.
   */
  _finish(d, N, S, b) {
    const { H, cr, cg, cb, rg } = b;
    const mt = b.mt;                                 // optional per-pixel metalness
    const n = N * N;

    const tight = this._blur(H, N, Math.max(2, N >> 8));
    const wide = this._blur(H, N, Math.max(5, N >> 5));
    const grimeN = this._grab(N, 'grime', S + 401);
    const grimeF = this._grab(N, 'grimeF', S + 402);
    const bleachN = this._grab(N, 'bleach', S + 403);
    const wearN = this._grab(N, 'wear', S + 404);
    const streak = d.streak > 0 ? this._grab(N, 'streak', S + 405) : null;

    // grime is a warm soot/dust brown; bleach is a hot, desaturated sand-white
    const DR = 0.135, DG = 0.115, DB = 0.092;
    const BR = 0.845, BG = 0.812, BB = 0.735;

    const alb = new Uint8Array(n * 4);
    const orm = new Uint8Array(n * 4);

    const cavK = d.cavK, aoK = d.ao, dirtK = d.dirt, strK = d.streak, blK = d.bleach, wrK = d.wear;

    for (let i = 0; i < n; i++) {
      const h = H[i];

      // ---- cavity / ambient occlusion -------------------------------
      const cav = clamp01((tight[i] - h) * cavK);
      const cav2 = clamp01((wide[i] - h) * cavK * 0.55);
      let ao = 1 - (cav * 0.62 + cav2 * 0.62) * aoK;
      ao = ao * (0.90 + 0.10 * grimeN[i]);
      // never let a cavity go to pure black — the grade keeps lifted shadows
      ao = ao < 0.14 ? 0.14 : (ao > 1 ? 1 : ao);

      // ---- grime accumulation ---------------------------------------
      let dirt = clamp01(cav * 1.35 + cav2 * 0.85) * dirtK * (0.35 + 0.85 * grimeN[i]);
      dirt *= (0.55 + 0.65 * grimeF[i]);
      if (streak) dirt += streak[i] * strK * (0.35 + 0.8 * grimeN[i]);
      dirt = clamp01(dirt);

      // ---- sun bleach on the exposed, upward-facing crests -----------
      const expo = clamp01((h - 0.50) * 2.1);
      const bl = clamp01(expo * (0.35 + 0.9 * bleachN[i])) * blK;

      // ---- traffic / handling polish on raised areas ----------------
      const wear = sstep(0.50, 0.93, wearN[i]) * expo * wrK;

      // ---- albedo ----------------------------------------------------
      let R = cr[i], G = cg[i], B = cb[i];
      if (bl > 0.001) {
        const lum = 0.30 * R + 0.60 * G + 0.10 * B;
        const t = bl * 0.55;
        R = lerp(R, lerp(lum, BR, 0.62), t);
        G = lerp(G, lerp(lum, BG, 0.62), t);
        B = lerp(B, lerp(lum, BB, 0.62), t);
      }
      if (dirt > 0.001) {
        const t = dirt * 0.80;
        R = lerp(R, DR, t); G = lerp(G, DG, t); B = lerp(B, DB, t);
      }
      // a touch of cavity darkening in the albedo keeps crevices reading even
      // where strong ambient washes the AO map out
      const cd = 1 - cav * 0.22 * aoK;
      R *= cd; G *= cd; B *= cd;

      const o4 = i << 2;
      alb[o4] = (clamp01(R) * 255) | 0;
      alb[o4 + 1] = (clamp01(G) * 255) | 0;
      alb[o4 + 2] = (clamp01(B) * 255) | 0;
      alb[o4 + 3] = 255;

      // ---- ORM -------------------------------------------------------
      const rough = clamp01(rg[i] + dirt * 0.20 - wear * 0.28 - bl * 0.04);
      const metal = clamp01((mt ? mt[i] : 0) * (1 - dirt * 0.65));

      orm[o4] = (ao * 255) | 0;
      orm[o4 + 1] = (rough * 255) | 0;
      orm[o4 + 2] = (metal * 255) | 0;
      orm[o4 + 3] = 255;
    }

    return {
      map: this._mkTex(alb, N, true),
      normalMap: this._mkTex(this._sobel(H, N, d.nrm), N, false),
      ormMap: this._mkTex(orm, N, false),
    };
  }

  /**
   * Tangent-space normal from the height field.
   *   p = (u, v, h)  ->  n = normalize(-dh/du, -dh/dv, 1)
   * DataTexture is not flipped, so row index increases with v and dh/dv is the
   * row gradient. OpenGL / Y+ green channel, which is what three expects.
   */
  _sobel(H, N, strength) {
    const out = new Uint8Array(N * N * 4);
    const k = (strength ?? 1) * N * SOBEL_K;
    for (let y = 0; y < N; y++) {
      const ym = ((y - 1 + N) % N) * N, yp = ((y + 1) % N) * N, yc = y * N;
      for (let x = 0; x < N; x++) {
        const xm = (x - 1 + N) % N, xp = (x + 1) % N;
        const h00 = H[ym + xm], h10 = H[ym + x], h20 = H[ym + xp];
        const h01 = H[yc + xm], h21 = H[yc + xp];
        const h02 = H[yp + xm], h12 = H[yp + x], h22 = H[yp + xp];
        const gx = (h20 + 2 * h21 + h22) - (h00 + 2 * h01 + h02);
        const gy = (h02 + 2 * h12 + h22) - (h00 + 2 * h10 + h20);
        let nx = -gx * k, ny = -gy * k;
        const inv = 1 / Math.sqrt(nx * nx + ny * ny + 1);
        nx *= inv; ny *= inv;
        const o = (yc + x) << 2;
        out[o] = ((nx * 0.5 + 0.5) * 255) | 0;
        out[o + 1] = ((ny * 0.5 + 0.5) * 255) | 0;
        out[o + 2] = ((inv * 0.5 + 0.5) * 255) | 0;
        out[o + 3] = 255;
      }
    }
    return out;
  }

  /** Shared 256px micro-surface normal, tiled at high frequency by the shader. */
  _detailTex() {
    if (this._detail) return this._detail;
    const N = 256;
    const a = this._fbm(N, { oct: 4, nx: 24, ny: 24, gain: 0.55, seed: 90211 });
    const b = this._fbm(N, { oct: 3, nx: 96, ny: 96, gain: 0.5, seed: 90212 });
    const s1 = this._fbm(N, { oct: 3, nx: 128, ny: 4, gain: 0.5, seed: 90213, ridged: true });
    const s2 = this._fbm(N, { oct: 3, nx: 5, ny: 128, gain: 0.5, seed: 90214, ridged: true });
    const H = new Float32Array(N * N);
    for (let i = 0; i < H.length; i++) {
      H[i] = a[i] * 0.55 + b[i] * 0.35
        - sstep(0.86, 1.0, s1[i]) * 0.30
        - sstep(0.90, 1.0, s2[i]) * 0.22;
    }
    normalize01(H);
    this._detail = this._mkTex(this._sobel(H, N, 1.6), N, false);
    return this._detail;
  }

  /* ---------------------------------------------------------------- *
   *  Environment (equirect -> PMREM)
   * ---------------------------------------------------------------- */

  _equirect(W, Hh) {
    const data = new Float32Array(W * Hh * 4);
    const cfg = this.g?.config?.sky ?? {};
    const tod = cfg.timeOfDay ?? 16.4;

    // 16:20 -> sun low in the west. az 0 = +X, increasing toward +Z.
    const sunEl = THREE.MathUtils.degToRad(THREE.MathUtils.clamp(90 - Math.abs(tod - 12) * 12.5, 6, 60));
    const sunAz = THREE.MathUtils.degToRad(196);
    const sdx = Math.cos(sunEl) * Math.cos(sunAz);
    const sdy = Math.sin(sunEl);
    const sdz = Math.cos(sunEl) * Math.sin(sunAz);

    const ZEN = [0.155, 0.255, 0.470];   // cool sky dome
    const HOR = [0.980, 0.660, 0.365];   // warm dusty horizon
    const GND = [0.330, 0.250, 0.160];   // sun-warmed dust bounce
    const SUN = [1.000, 0.845, 0.620];

    const turb = cfg.turbidity ?? 6.2;
    const hazeBoost = 0.55 + turb * 0.09;

    for (let y = 0; y < Hh; y++) {
      const v = (y + 0.5) / Hh;
      const el = (v - 0.5) * Math.PI;
      const ce = Math.cos(el), se = Math.sin(el);
      for (let x = 0; x < W; x++) {
        const u = (x + 0.5) / W;
        const az = (u - 0.5) * Math.PI * 2;
        const dx = ce * Math.cos(az), dy = se, dz = ce * Math.sin(az);
        const dot = dx * sdx + dy * sdy + dz * sdz;
        const dp = dot > 0 ? dot : 0;

        let r, g, bl;
        if (dy >= 0) {
          const t = Math.pow(clamp01(dy), 0.42);
          r = lerp(HOR[0] * hazeBoost, ZEN[0], t);
          g = lerp(HOR[1] * hazeBoost, ZEN[1], t);
          bl = lerp(HOR[2] * hazeBoost, ZEN[2], t);
        } else {
          // ground: dusty bounce, brightest just below the horizon and sunward
          const t = clamp01(-dy * 2.4);
          const f = (1 - t * 0.72) * (0.72 + 0.55 * dp);
          r = GND[0] * f; g = GND[1] * f; bl = GND[2] * f;
        }

        // broad forward-scatter lobe + tight solar disc
        const sk = Math.pow(dp, 5.5) * 1.35 + Math.pow(dp, 48) * 6.0 + Math.pow(dp, 2600) * 90;
        r += SUN[0] * sk; g += SUN[1] * sk; bl += SUN[2] * sk;

        const o = (y * W + x) * 4;
        data[o] = r; data[o + 1] = g; data[o + 2] = bl; data[o + 3] = 1;
      }
    }

    const t = new THREE.DataTexture(data, W, Hh, THREE.RGBAFormat, THREE.FloatType);
    t.mapping = THREE.EquirectangularReflectionMapping;
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.wrapS = THREE.RepeatWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    t.colorSpace = THREE.LinearSRGBColorSpace;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  }

  /* ================================================================ *
   *  SURFACE BUILDERS
   *  Each returns { H, cr, cg, cb, rg, mt? } as Float32Array(N*N).
   *  H = height (0..1), cr/cg/cb = sRGB albedo, rg = roughness,
   *  mt = optional per-pixel metalness (absent -> dielectric).
   * ================================================================ */

  /* -- CONCRETE: cast, form-marked, aggregate speckle, hairline cracks -- */
  _bConcrete(N, S) {
    const n = N * N;
    const big = this._grab(N, 'big', S + 1);
    const med = this._grab(N, 'mid', S + 2);
    const fine = this._grab(N, 'fine', S + 3);
    const pit = this._grab(N, 'pit', S + 4);
    const stain = this._grab(N, 'bleach', S + 5);
    const crk = this._fbm(N, { oct: 5, nx: 3, ny: 3, gain: 0.58, seed: S + 6, ridged: true });
    const crk2 = this._fbm(N, { oct: 4, nx: 9, ny: 9, gain: 0.55, seed: S + 7, ridged: true });
    const agg = this._cells(N, Math.max(24, N >> 5) * 3, Math.max(24, N >> 5) * 3, S + 8);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const BASE = C(0xa39a8c);
    const LIGHT = C(0xbdb4a4);
    const DARK = C(0x736d64);
    const AGGL = C(0xc8c1b4);
    const AGGD = C(0x5f5b55);
    const CRK = C(0x36322d);

    for (let y = 0; y < N; y++) {
      // form-board seams every third of the tile (cast-in-place panels)
      const sy = ((y / N) * 3) % 1;
      const seam = 1 - sstep(0, 0.012, Math.min(sy, 1 - sy));
      for (let x = 0; x < N; x++) {
        const i = y * N + x;

        const speck = sstep(0.30, 0.045, agg.d[i]);              // pebbles poking through
        const bubble = sstep(0.60, 0.86, pit[i]);                 // trapped air voids
        const crack = sstep(0.885, 0.975, crk[i]) * sstep(0.35, 0.62, big[i]);
        const hair = sstep(0.930, 0.995, crk2[i]) * 0.45;

        let h = 0.50 + (big[i] - 0.5) * 0.30 + (med[i] - 0.5) * 0.24 + (fine[i] - 0.5) * 0.11;
        h += speck * 0.10;
        h -= bubble * 0.20;
        h -= (crack * 0.62 + hair * 0.30);
        h -= seam * 0.16;
        H[i] = clamp01(h);

        // ---- colour
        const t = clamp01(big[i] * 1.25 - 0.12);
        let R = lerp(DARK[0], LIGHT[0], t), G = lerp(DARK[1], LIGHT[1], t), B = lerp(DARK[2], LIGHT[2], t);
        R = lerp(R, BASE[0], 0.42); G = lerp(G, BASE[1], 0.42); B = lerp(B, BASE[2], 0.42);
        // medium-scale mottling with a small warm/cool hue swing
        const mm = (med[i] - 0.5) * 0.11;
        R += mm * 1.15; G += mm; B += mm * 0.80;
        // aggregate: some pale quartz, some dark basalt
        if (speck > 0.01) {
          const dk = agg.id[i] < 0.42;
          const AR = dk ? AGGD[0] : AGGL[0], AG = dk ? AGGD[1] : AGGL[1], AB = dk ? AGGD[2] : AGGL[2];
          const tt = speck * (dk ? 0.62 : 0.70);
          R = lerp(R, AR, tt); G = lerp(G, AG, tt); B = lerp(B, AB, tt);
        }
        const fg = (fine[i] - 0.5) * 0.085;
        R += fg; G += fg; B += fg * 0.92;
        // water staining
        const st = sstep(0.62, 0.95, stain[i]) * 0.16;
        R = lerp(R, R * 0.80, st); G = lerp(G, G * 0.82, st); B = lerp(B, B * 0.88, st);
        // cracks
        const ck = clamp01(crack + hair * 0.7);
        R = lerp(R, CRK[0], ck * 0.78); G = lerp(G, CRK[1], ck * 0.78); B = lerp(B, CRK[2], ck * 0.78);

        cr[i] = R; cg[i] = G; cb[i] = B;

        // ---- roughness: powdery cement matrix, glassier aggregate
        let r = 0.90 + (fine[i] - 0.5) * 0.10 + (med[i] - 0.5) * 0.06;
        r -= speck * 0.20;
        r += bubble * 0.04 + ck * 0.05;
        rg[i] = clamp01(r);
      }
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- PLASTER: troweled render, chipped patches, water stains -- */
  _bPlaster(N, S) {
    const n = N * N;
    const swirl = this._fbm(N, { oct: 4, nx: 20, ny: 5, gain: 0.55, seed: S + 1 });
    const swirl2 = this._fbm(N, { oct: 4, nx: 6, ny: 22, gain: 0.55, seed: S + 2 });
    const crk = this._fbm(N, { oct: 5, nx: 4, ny: 4, gain: 0.58, seed: S + 3, ridged: true });
    const big = this._grab(N, 'big', S + 4);
    const fine = this._grab(N, 'fine', S + 5);
    const chipF = this._grab(N, 'grime', S + 6);
    const chipE = this._grab(N, 'grimeF', S + 7);
    const eff = this._grab(N, 'mid', S + 8);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const BASE = C(0xd2c1a2);
    const WARM = C(0xdfd0b0);
    const COOL = C(0xb8ab93);
    const SUB = C(0x8d7259);   // exposed substrate under the chipped render
    const CRK = C(0x4b4237);

    for (let i = 0; i < n; i++) {
      // trowel arcs = two crossed anisotropic fields
      const sw = (swirl[i] - 0.5) * 0.55 + (swirl2[i] - 0.5) * 0.45;
      const chipMask = sstep(0.66, 0.80, chipF[i] * 0.78 + chipE[i] * 0.22);
      const crack = sstep(0.895, 0.985, crk[i]);

      let h = 0.58 + sw * 0.18 + (big[i] - 0.5) * 0.16 + (fine[i] - 0.5) * 0.06;
      h -= chipMask * 0.34;
      h -= crack * 0.45;
      H[i] = clamp01(h);

      const t = clamp01(big[i] * 1.2 - 0.1);
      let R = lerp(COOL[0], WARM[0], t), G = lerp(COOL[1], WARM[1], t), B = lerp(COOL[2], WARM[2], t);
      R = lerp(R, BASE[0], 0.35); G = lerp(G, BASE[1], 0.35); B = lerp(B, BASE[2], 0.35);

      const sv = sw * 0.055;
      R += sv * 1.1; G += sv; B += sv * 0.85;
      const fg = (fine[i] - 0.5) * 0.055;
      R += fg; G += fg; B += fg;

      // efflorescence — pale salt blooms
      const ef = sstep(0.72, 0.96, eff[i]) * 0.30;
      R = lerp(R, 0.90, ef); G = lerp(G, 0.885, ef); B = lerp(B, 0.845, ef);

      if (chipMask > 0.001) {
        R = lerp(R, SUB[0], chipMask * 0.88); G = lerp(G, SUB[1], chipMask * 0.88); B = lerp(B, SUB[2], chipMask * 0.88);
      }
      R = lerp(R, CRK[0], crack * 0.72); G = lerp(G, CRK[1], crack * 0.72); B = lerp(B, CRK[2], crack * 0.72);

      cr[i] = R; cg[i] = G; cb[i] = B;

      let r = 0.82 + (fine[i] - 0.5) * 0.10 - sw * 0.07;
      r += chipMask * 0.10 + crack * 0.05 + ef * 0.10;
      rg[i] = clamp01(r);
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- BRICK: irregular hand-laid courses, mortar variance, chipped arrises -- */
  _bBrick(N, S) {
    const n = N * N;
    const ROWS = 9, COLS = 4;
    const face = this._fbm(N, { oct: 4, nx: 28, ny: 28, gain: 0.52, seed: S + 1 });
    const fine = this._grab(N, 'fine', S + 2);
    const mortN = this._grab(N, 'mid', S + 3);
    const big = this._grab(N, 'big', S + 4);
    const pit = this._grab(N, 'pit', S + 5);
    const chip = this._grab(N, 'grimeF', S + 6);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    // sun-baked mudbrick / terracotta palette
    const PAL = [
      C(0x9a5f42), C(0xa96f4c), C(0x8b5238), C(0xb07c56),
      C(0x7d4a34), C(0xa2694a), C(0x94614a), C(0xbb8a63),
    ];
    const MORT = C(0xb3a892);
    const MORTD = C(0x8d8474);

    const brickW = N / COLS, brickH = N / ROWS;
    const mPx = Math.max(3, N * 0.011);

    for (let y = 0; y < N; y++) {
      const fyy = (y / N) * ROWS;
      const rr = Math.floor(fyy);
      const ty = fyy - rr;
      const rowOff = (rr & 1 ? 0.5 : 0.0) + (hash1i(rr, S + 31) - 0.5) * 0.05;
      const rowSag = (hash1i(rr, S + 32) - 0.5) * 0.05;

      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        const fxx = (x / N) * COLS + rowOff;
        const cc = Math.floor(fxx);
        const tx = fxx - cc;

        const id = hash2i(cc, rr, S + 77);
        const id2 = hash2i(cc, rr, S + 78);

        // distance to the brick edge, in pixels
        const ex = Math.min(tx, 1 - tx) * brickW;
        const ey = Math.min(ty, 1 - ty) * brickH;
        const e = Math.min(ex, ey) - mPx * (0.55 + rowSag + (id2 - 0.5) * 0.25);
        const isBrick = sstep(0, mPx * 0.85, e);
        const arris = 1 - sstep(mPx * 0.6, mPx * 2.6, e);   // rounded / chipped edge band

        const chipped = sstep(0.60, 0.86, chip[i]) * arris;
        const bubble = sstep(0.66, 0.90, pit[i]) * isBrick;

        const bh = 0.72 + (id - 0.5) * 0.10 + (face[i] - 0.5) * 0.13 + (fine[i] - 0.5) * 0.05;
        const mh = 0.30 + (mortN[i] - 0.5) * 0.22 + (fine[i] - 0.5) * 0.06;
        let h = lerp(mh, bh, isBrick);
        h -= arris * 0.10 * isBrick;
        h -= chipped * 0.22;
        h -= bubble * 0.10;
        H[i] = clamp01(h);

        // ---- colour
        const p = PAL[Math.min(PAL.length - 1, (id * PAL.length) | 0)];
        let R = p[0], G = p[1], B = p[2];
        const fv = (face[i] - 0.5) * 0.13;
        R += fv * 1.15; G += fv * 0.92; B += fv * 0.72;
        const fg = (fine[i] - 0.5) * 0.07;
        R += fg; G += fg * 0.95; B += fg * 0.9;
        // freshly chipped faces are paler and unweathered
        R = lerp(R, R * 1.28 + 0.10, chipped * 0.7);
        G = lerp(G, G * 1.22 + 0.09, chipped * 0.7);
        B = lerp(B, B * 1.18 + 0.08, chipped * 0.7);

        let MR = lerp(MORTD[0], MORT[0], mortN[i]);
        let MG = lerp(MORTD[1], MORT[1], mortN[i]);
        let MB = lerp(MORTD[2], MORT[2], mortN[i]);
        const mg = (fine[i] - 0.5) * 0.09;
        MR += mg; MG += mg; MB += mg;

        R = lerp(MR, R, isBrick); G = lerp(MG, G, isBrick); B = lerp(MB, B, isBrick);

        // large-scale damp / sun-face variation so a wall never reads as one tile
        const bg = (big[i] - 0.5) * 0.13;
        R += bg * 1.05; G += bg * 0.95; B += bg * 0.85;

        cr[i] = R; cg[i] = G; cb[i] = B;

        // ---- roughness
        const rBrick = 0.80 + (face[i] - 0.5) * 0.14 + bubble * 0.06;
        const rMort = 0.95 + (mortN[i] - 0.5) * 0.06;
        rg[i] = clamp01(lerp(rMort, rBrick, isBrick) + chipped * 0.05);
      }
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- SAND: wind ripples, grain, embedded pebbles, damp patches -- */
  _bSand(N, S) {
    const n = N * N;
    const warp = this._grab(N, 'big', S + 1);
    const warp2 = this._grab(N, 'mid', S + 2);
    const grain = this._grab(N, 'grain', S + 3);
    const med = this._grab(N, 'wear', S + 4);
    const damp = this._grab(N, 'bleach', S + 5);
    const fine = this._grab(N, 'fine', S + 6);
    const peb = this._cells(N, Math.max(18, N >> 6) * 4, Math.max(18, N >> 6) * 4, S + 7);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const LIGHT = C(0xd8bf92);
    const MID = C(0xc4a878);
    const DARKW = C(0x8e7350);   // damp / shaded sand
    const PEB1 = C(0x9c9182);
    const PEB2 = C(0x6e6558);

    const TWO_PI = Math.PI * 2;

    for (let y = 0; y < N; y++) {
      const v = y / N;
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        const u = x / N;

        // wind ripples: a sine train marching along v, domain-warped so the
        // crests wander instead of forming stripes
        const ph = (v * 13 + (warp[i] - 0.5) * 2.6 + (warp2[i] - 0.5) * 0.9 + u * 1.1);
        let rip = Math.sin(ph * TWO_PI) * 0.5 + 0.5;
        rip = rip * rip * (1.5 - 0.5 * rip);
        const ripAmt = 0.35 + 0.65 * sstep(0.25, 0.75, warp2[i]);

        const pebble = sstep(0.28, 0.05, peb.d[i]);
        const grit = sstep(0.80, 0.97, fine[i]);

        let h = 0.44 + (rip - 0.4) * 0.26 * ripAmt;
        h += (med[i] - 0.5) * 0.22 + (warp[i] - 0.5) * 0.16;
        h += (grain[i] - 0.5) * 0.07;
        h += pebble * 0.14 + grit * 0.05;
        H[i] = clamp01(h);

        // ---- colour: crests bleached, troughs a shade cooler
        const t = clamp01(rip * 0.55 + med[i] * 0.45);
        let R = lerp(MID[0], LIGHT[0], t), G = lerp(MID[1], LIGHT[1], t), B = lerp(MID[2], LIGHT[2], t);
        const dv = sstep(0.58, 0.92, damp[i]) * 0.55;
        R = lerp(R, DARKW[0], dv); G = lerp(G, DARKW[1], dv); B = lerp(B, DARKW[2], dv);
        const gg = (grain[i] - 0.5) * 0.11;
        R += gg * 1.05; G += gg; B += gg * 0.85;

        if (pebble > 0.01) {
          const dk = peb.id[i] < 0.5;
          const PR = dk ? PEB2[0] : PEB1[0], PG = dk ? PEB2[1] : PEB1[1], PB = dk ? PEB2[2] : PEB1[2];
          const tt = pebble * 0.80;
          R = lerp(R, PR, tt); G = lerp(G, PG, tt); B = lerp(B, PB, tt);
        }
        if (grit > 0.01) {
          const tt = grit * 0.30;
          R = lerp(R, 0.44, tt); G = lerp(G, 0.40, tt); B = lerp(B, 0.33, tt);
        }

        cr[i] = R; cg[i] = G; cb[i] = B;

        let r = 0.94 + (grain[i] - 0.5) * 0.06;
        r -= pebble * 0.24;               // stones are smoother than the sand
        r -= dv * 0.12;                   // damp sand is glossier
        rg[i] = clamp01(r);
      }
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- ASPHALT: exposed aggregate in tar, cracks, patches, oil -- */
  _bAsphalt(N, S) {
    const n = N * N;
    const crk = this._fbm(N, { oct: 5, nx: 4, ny: 4, gain: 0.60, seed: S + 1, ridged: true });
    const agg = this._cells(N, Math.max(28, N >> 5) * 3, Math.max(28, N >> 5) * 3, S + 2);
    const big = this._grab(N, 'big', S + 3);
    const med = this._grab(N, 'mid', S + 4);
    const fine = this._grab(N, 'fine', S + 5);
    const grain = this._grab(N, 'grain', S + 6);
    const oil = this._grab(N, 'grime', S + 7);
    const patch = this._grab(N, 'bleach', S + 8);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const TAR = C(0x413f3d);
    const TARW = C(0x5d5a55);   // sun-oxidised tar
    const A1 = C(0x8f8a80);
    const A2 = C(0x6a655d);
    const A3 = C(0xa39a8b);
    const PATCH = C(0x333231);

    for (let i = 0; i < n; i++) {
      const stone = sstep(0.30, 0.06, agg.d[i]);
      const chip = sstep(0.72, 0.94, grain[i]);
      const crack = sstep(0.900, 0.985, crk[i]);
      const patchM = sstep(0.62, 0.74, patch[i]);
      const oilM = sstep(0.70, 0.93, oil[i]);

      let h = 0.44 + (med[i] - 0.5) * 0.18 + (big[i] - 0.5) * 0.14 + (fine[i] - 0.5) * 0.10;
      h += stone * 0.16 + chip * 0.07;
      h -= crack * 0.55;
      h += patchM * 0.05;
      H[i] = clamp01(h);

      const t = clamp01(big[i] * 0.7 + med[i] * 0.3);
      let R = lerp(TAR[0], TARW[0], t), G = lerp(TAR[1], TARW[1], t), B = lerp(TAR[2], TARW[2], t);
      R = lerp(R, PATCH[0], patchM * 0.55); G = lerp(G, PATCH[1], patchM * 0.55); B = lerp(B, PATCH[2], patchM * 0.55);

      if (stone > 0.01) {
        const r0 = agg.id[i];
        const P = r0 < 0.36 ? A2 : (r0 < 0.78 ? A1 : A3);
        const tt = stone * (0.55 + 0.35 * r0);
        R = lerp(R, P[0], tt); G = lerp(G, P[1], tt); B = lerp(B, P[2], tt);
      }
      if (chip > 0.01) {
        const tt = chip * 0.30;
        R = lerp(R, A1[0], tt); G = lerp(G, A1[1], tt); B = lerp(B, A1[2], tt);
      }
      const fg = (fine[i] - 0.5) * 0.085;
      R += fg; G += fg; B += fg;

      // cracks show the dark unweathered binder (never a true black)
      R = lerp(R, 0.105, crack * 0.85); G = lerp(G, 0.100, crack * 0.85); B = lerp(B, 0.098, crack * 0.85);
      // oil: dark, glossy, biased slightly blue
      R = lerp(R, 0.080, oilM * 0.55); G = lerp(G, 0.080, oilM * 0.55); B = lerp(B, 0.094, oilM * 0.55);

      cr[i] = R; cg[i] = G; cb[i] = B;

      let r = 0.86 + (fine[i] - 0.5) * 0.10 - stone * 0.22 - chip * 0.06;
      r += crack * 0.05;
      r -= oilM * 0.55;
      rg[i] = clamp01(r);
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- METAL: brushed / rolled steel sheet, scratches, dents -- */
  _bMetal(N, S) {
    const n = N * N;
    const brush = this._fbm(N, { oct: 4, nx: 3, ny: N, gain: 0.55, seed: S + 1 });
    const brush2 = this._fbm(N, { oct: 3, nx: 8, ny: N >> 1, gain: 0.5, seed: S + 2 });
    const scr = this._fbm(N, { oct: 3, nx: 5, ny: N, gain: 0.5, seed: S + 3, ridged: true });
    const scrX = this._fbm(N, { oct: 3, nx: N, ny: 6, gain: 0.5, seed: S + 4, ridged: true });
    const dent = this._grab(N, 'grime', S + 5);
    const big = this._grab(N, 'big', S + 6);
    const grime = this._grab(N, 'wear', S + 7);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n), mt = new Float32Array(n);

    const STEEL = C(0x9aa0a4);
    const DARKS = C(0x6d7276);
    const BRIGHT = C(0xc3c8cb);

    for (let i = 0; i < n; i++) {
      const sc = sstep(0.905, 0.995, scr[i]);
      const scx = sstep(0.945, 0.998, scrX[i]) * 0.5;

      let h = 0.55 + (brush[i] - 0.5) * 0.10 + (brush2[i] - 0.5) * 0.07 + (dent[i] - 0.5) * 0.28;
      h -= sc * 0.16 + scx * 0.10;
      H[i] = clamp01(h);

      const t = clamp01(big[i] * 0.55 + dent[i] * 0.45);
      let R = lerp(DARKS[0], STEEL[0], t), G = lerp(DARKS[1], STEEL[1], t), B = lerp(DARKS[2], STEEL[2], t);
      const bv = (brush[i] - 0.5) * 0.10 + (brush2[i] - 0.5) * 0.05;
      R += bv; G += bv; B += bv * 1.05;
      const s = clamp01(sc + scx);
      R = lerp(R, BRIGHT[0], s * 0.65); G = lerp(G, BRIGHT[1], s * 0.65); B = lerp(B, BRIGHT[2], s * 0.65);
      // faint warm grime film so bare steel is never a mirror
      const gm = sstep(0.55, 0.95, grime[i]) * 0.20;
      R = lerp(R, 0.34, gm); G = lerp(G, 0.31, gm); B = lerp(B, 0.27, gm);

      cr[i] = R; cg[i] = G; cb[i] = B;

      // anisotropic brushing lives in the roughness — this is what sells metal
      let r = 0.36 + (brush[i] - 0.5) * 0.26 + (brush2[i] - 0.5) * 0.10;
      r += (dent[i] - 0.5) * 0.08;
      r -= s * 0.14;
      r += gm * 0.9;
      rg[i] = clamp01(r);
      mt[i] = clamp01(0.98 - gm * 1.4);
    }
    return { H, cr, cg, cb, rg, mt };
  }

  /* -- RUSTMETAL: corroded steel, scaly rust blooms over bare plate -- */
  _bRustMetal(N, S) {
    const n = N * N;
    const rustF = this._fbm(N, { oct: 5, nx: 4, ny: 4, gain: 0.57, seed: S + 1 });
    const brush = this._fbm(N, { oct: 3, nx: 4, ny: N >> 1, gain: 0.5, seed: S + 2 });
    const scr = this._fbm(N, { oct: 3, nx: 6, ny: N, gain: 0.5, seed: S + 3, ridged: true });
    const rustD = this._grab(N, 'mid', S + 4);
    const scale = this._grab(N, 'fine', S + 5);
    const dent = this._grab(N, 'grime', S + 6);
    const paint = this._grab(N, 'bleach', S + 7);
    const pit = this._cells(N, Math.max(24, N >> 5) * 3, Math.max(24, N >> 5) * 3, S + 8);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n), mt = new Float32Array(n);

    const STEEL = C(0x8b9095);
    const DARKS = C(0x5c6165);
    const RUST1 = C(0x8a4a26);   // fresh orange rust
    const RUST2 = C(0x5a3320);   // old dark scale
    const RUST3 = C(0xb0703a);   // bright bloom edge
    const PAINT = C(0x6b7a6a);   // faded military green remnants

    for (let i = 0; i < n; i++) {
      const rustM = clamp01(sstep(0.34, 0.68, rustF[i] * 0.82 + rustD[i] * 0.18));
      const bloom = sstep(0.52, 0.66, rustF[i]) * (1 - sstep(0.72, 0.86, rustF[i]));
      const paintM = sstep(0.70, 0.86, paint[i]) * (1 - rustM);
      const pitM = sstep(0.30, 0.07, pit.d[i]) * rustM;
      const sc = sstep(0.92, 0.995, scr[i]);

      let h = 0.52 + (dent[i] - 0.5) * 0.26 + (brush[i] - 0.5) * 0.07;
      h += rustM * (scale[i] - 0.35) * 0.34;             // scaly build-up
      h += pitM * 0.10;
      h -= rustM * sstep(0.55, 0.95, rustD[i]) * 0.18;   // flaked-away pits
      h -= sc * 0.10;
      h += paintM * 0.03;
      H[i] = clamp01(h);

      // bare plate
      const t = clamp01(dent[i] * 0.6 + brush[i] * 0.4);
      let R = lerp(DARKS[0], STEEL[0], t), G = lerp(DARKS[1], STEEL[1], t), B = lerp(DARKS[2], STEEL[2], t);
      R = lerp(R, R * 1.25, sc * 0.7); G = lerp(G, G * 1.25, sc * 0.7); B = lerp(B, B * 1.25, sc * 0.7);
      R = lerp(R, PAINT[0], paintM * 0.85); G = lerp(G, PAINT[1], paintM * 0.85); B = lerp(B, PAINT[2], paintM * 0.85);

      // rust
      const rt = clamp01(rustD[i] * 0.75 + scale[i] * 0.25);
      let RR = lerp(RUST2[0], RUST1[0], rt), RG = lerp(RUST2[1], RUST1[1], rt), RB = lerp(RUST2[2], RUST1[2], rt);
      RR = lerp(RR, RUST3[0], bloom * 0.7); RG = lerp(RG, RUST3[1], bloom * 0.7); RB = lerp(RB, RUST3[2], bloom * 0.7);
      const sg = (scale[i] - 0.5) * 0.10;
      RR += sg * 1.1; RG += sg * 0.8; RB += sg * 0.6;

      R = lerp(R, RR, rustM); G = lerp(G, RG, rustM); B = lerp(B, RB, rustM);

      cr[i] = R; cg[i] = G; cb[i] = B;

      const rSteel = 0.42 + (brush[i] - 0.5) * 0.22 - sc * 0.16 + paintM * 0.24;
      const rRust = 0.93 + (scale[i] - 0.5) * 0.10 + pitM * 0.04;
      rg[i] = clamp01(lerp(rSteel, rRust, rustM));
      mt[i] = clamp01(lerp(0.96 - paintM * 0.9, 0.10, rustM));
    }
    return { H, cr, cg, cb, rg, mt };
  }

  /* -- WOOD: sawn planks, grain, knots, split-out, seams -- */
  _bWood(N, S) {
    const n = N * N;
    const PLANKS = 3;
    const warp = this._fbm(N, { oct: 4, nx: 6, ny: 3, gain: 0.55, seed: S + 1 });
    const warp2 = this._fbm(N, { oct: 3, nx: 20, ny: 4, gain: 0.5, seed: S + 2 });
    const fibre = this._fbm(N, { oct: 3, nx: 6, ny: N >> 1, gain: 0.5, seed: S + 3 });
    const split = this._fbm(N, { oct: 3, nx: 4, ny: N >> 1, gain: 0.5, seed: S + 4, ridged: true });
    const rough = this._grab(N, 'fine', S + 5);
    const big = this._grab(N, 'big', S + 6);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const EARLY = C(0x9c7146);   // pale earlywood
    const LATE = C(0x63421f);    // dark latewood
    const WEATH = C(0x8d7c68);   // silvered, sun-bleached timber
    const KNOT = C(0x3b2512);

    const knots = [];
    for (let p = 0; p < PLANKS; p++) {
      const c = 1 + ((hash1i(p, S + 51) * 2) | 0);
      for (let q = 0; q < c; q++) {
        knots.push({
          x: hash2i(p, q, S + 52),
          y: (p + 0.2 + hash2i(p, q, S + 53) * 0.6) / PLANKS,
          r: 0.020 + hash2i(p, q, S + 54) * 0.030,
        });
      }
    }
    const KN = knots.length;
    const kx = new Float32Array(KN), ky = new Float32Array(KN), kr = new Float32Array(KN);
    for (let q = 0; q < KN; q++) { kx[q] = knots[q].x; ky[q] = knots[q].y; kr[q] = knots[q].r; }

    const rings = 26;
    for (let y = 0; y < N; y++) {
      const v = y / N;
      const pf = v * PLANKS;
      const pi = Math.floor(pf);
      const tp = pf - pi;
      const seam = 1 - sstep(0, 0.028, Math.min(tp, 1 - tp));
      const plankTone = hash1i(pi, S + 61);
      const plankShift = (hash1i(pi, S + 62) - 0.5) * 3.0;

      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        const u = x / N;

        // ring coordinate runs across the plank (v), warped along its length
        let t = tp * rings * 0.55 + plankShift + (warp[i] - 0.5) * 3.2 + (warp2[i] - 0.5) * 1.1;

        // knots bend the grain around them and darken the core
        let knotC = 0, knotRing = 0;
        for (let q = 0; q < KN; q++) {
          let dx = u - kx[q]; dx -= Math.round(dx);
          let dy = v - ky[q]; dy -= Math.round(dy);
          const r0 = kr[q];
          const dd = Math.sqrt(dx * dx * 0.5 + dy * dy);
          const infl = Math.exp(-(dd * dd) / (r0 * r0 * 9));
          t += infl * 6.5;
          const kc = sstep(r0 * 1.15, r0 * 0.35, dd);
          if (kc > knotC) knotC = kc;
          if (infl > knotRing) knotRing = infl;
        }

        let ring = Math.abs(Math.sin(t * Math.PI));
        ring = ring * ring * (ring * 0.6 + 0.4);      // sharpen toward latewood
        const fib = (fibre[i] - 0.5);
        const sp = sstep(0.905, 0.995, split[i]);

        let h = 0.60 - ring * 0.16 + fib * 0.10 + (rough[i] - 0.5) * 0.07;
        h -= sp * 0.28;
        h -= seam * 0.42;
        h -= knotC * 0.14;
        H[i] = clamp01(h);

        let R = lerp(EARLY[0], LATE[0], ring), G = lerp(EARLY[1], LATE[1], ring), B = lerp(EARLY[2], LATE[2], ring);
        const pt = (plankTone - 0.5) * 0.16;
        R += pt * 1.1; G += pt * 0.9; B += pt * 0.7;
        R += fib * 0.075; G += fib * 0.062; B += fib * 0.045;
        // sun-silvered surface on the exposed faces
        const wv = sstep(0.45, 0.95, big[i]) * 0.42;
        R = lerp(R, WEATH[0], wv); G = lerp(G, WEATH[1], wv); B = lerp(B, WEATH[2], wv);
        R = lerp(R, KNOT[0], knotC * 0.85); G = lerp(G, KNOT[1], knotC * 0.85); B = lerp(B, KNOT[2], knotC * 0.85);
        R = lerp(R, R * 0.72, knotRing * 0.30); G = lerp(G, G * 0.72, knotRing * 0.30); B = lerp(B, B * 0.72, knotRing * 0.30);
        R = lerp(R, 0.13, seam * 0.80); G = lerp(G, 0.115, seam * 0.80); B = lerp(B, 0.10, seam * 0.80);
        R = lerp(R, 0.18, sp * 0.60); G = lerp(G, 0.15, sp * 0.60); B = lerp(B, 0.12, sp * 0.60);

        cr[i] = R; cg[i] = G; cb[i] = B;

        let r = 0.80 + ring * 0.10 + (rough[i] - 0.5) * 0.10 + sp * 0.08;
        r -= wv * 0.05;
        rg[i] = clamp01(r);
      }
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- FABRIC: coarse plain-weave canvas / awning -- */
  _bFabric(N, S) {
    return this._weave(N, S, {
      threads: 44, base: C(0x6f6555), alt: C(0x574f43), light: C(0x8a7f6b),
      fuzz: 0.5, rough: 0.93, ripstop: 0,
    });
  }

  /* -- CLOTH_TAN: tactical coyote-tan cordura with a ripstop grid -- */
  _bClothTan(N, S) {
    return this._weave(N, S, {
      threads: 64, base: C(0xb49a72), alt: C(0x947c58), light: C(0xcbb389),
      fuzz: 0.28, rough: 0.86, ripstop: 8,
    });
  }

  _weave(N, S, o) {
    const n = N * N;
    const fuzz = this._grab(N, 'grain', S + 1);
    const big = this._grab(N, 'big', S + 2);
    const med = this._grab(N, 'mid', S + 3);
    const pill = this._grab(N, 'grimeF', S + 4);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const T = o.threads;
    const PI = Math.PI;

    for (let y = 0; y < N; y++) {
      const ty = (y / N) * T;
      const iy = Math.floor(ty), fy = ty - iy;
      const rsY = o.ripstop > 0 && (iy % o.ripstop === 0);
      const by = Math.sin(fy * PI);
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        const tx = (x / N) * T;
        const ix = Math.floor(tx), fx = tx - ix;
        const rsX = o.ripstop > 0 && (ix % o.ripstop === 0);

        const over = ((ix + iy) & 1) === 0;
        // rounded thread cross-sections; the thread on top gets the full bulge
        const bx = Math.sin(fx * PI);
        const topB = over ? bx : by;
        const botB = over ? by : bx;
        const thick = (rsX || rsY) ? 1.28 : 1.0;

        const jitterA = hash2i(ix, iy, S + 11);
        const jitterB = hash2i(ix, iy, S + 12);

        let h = 0.24 + topB * 0.52 * thick + botB * 0.16;
        h += (fuzz[i] - 0.5) * 0.13 * (0.4 + o.fuzz);
        h += (jitterA - 0.5) * 0.05;
        H[i] = clamp01(h);

        const tsel = (over ? 0.0 : 0.55) + (jitterB - 0.5) * 0.35;
        let R = lerp(o.base[0], o.alt[0], tsel);
        let G = lerp(o.base[1], o.alt[1], tsel);
        let B = lerp(o.base[2], o.alt[2], tsel);
        const lit = clamp01(topB) * 0.42;
        R = lerp(R, o.light[0], lit); G = lerp(G, o.light[1], lit); B = lerp(B, o.light[2], lit);
        const bv = (big[i] - 0.5) * 0.14;
        R += bv * 1.1; G += bv; B += bv * 0.8;
        const mv = (med[i] - 0.5) * 0.07;
        R += mv; G += mv; B += mv;
        const pl = sstep(0.72, 0.95, pill[i]) * 0.18 * (0.3 + o.fuzz);
        R = lerp(R, R * 1.30 + 0.06, pl); G = lerp(G, G * 1.28 + 0.06, pl); B = lerp(B, B * 1.26 + 0.06, pl);
        if (rsX || rsY) { R *= 0.94; G *= 0.94; B *= 0.95; }

        cr[i] = R; cg[i] = G; cb[i] = B;
        rg[i] = clamp01(o.rough + (fuzz[i] - 0.5) * 0.08 - clamp01(topB) * 0.06);
      }
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- RUBBER: moulded tread / stair-nosing, matte and dusty -- */
  _bRubber(N, S) {
    const n = N * N;
    const crk = this._fbm(N, { oct: 4, nx: 7, ny: 7, gain: 0.55, seed: S + 1, ridged: true });
    const knurl = this._cells(N, Math.max(14, N >> 6) * 3, Math.max(14, N >> 6) * 3, S + 2);
    const grain = this._grab(N, 'grain', S + 3);
    const big = this._grab(N, 'big', S + 4);
    const scuff = this._grab(N, 'mid', S + 5);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const BASE = C(0x2b2b2d);
    const LIFT = C(0x43433f);
    const DUST = C(0x6e6455);

    for (let i = 0; i < n; i++) {
      const stud = sstep(0.42, 0.24, knurl.d[i]);   // raised moulded studs
      const crack = sstep(0.90, 0.99, crk[i]);      // perished rubber

      let h = 0.34 + stud * 0.44 + (grain[i] - 0.5) * 0.08 + (big[i] - 0.5) * 0.10;
      h -= crack * 0.22;
      H[i] = clamp01(h);

      let R = BASE[0], G = BASE[1], B = BASE[2];
      const lv = clamp01(stud * 0.45 + (big[i] - 0.5) * 0.20);
      R = lerp(R, LIFT[0], lv); G = lerp(G, LIFT[1], lv); B = lerp(B, LIFT[2], lv);
      const gg = (grain[i] - 0.5) * 0.05;
      R += gg; G += gg; B += gg;
      const du = sstep(0.55, 0.95, scuff[i]) * 0.26;
      R = lerp(R, DUST[0], du); G = lerp(G, DUST[1], du); B = lerp(B, DUST[2], du);
      R = lerp(R, R * 1.5 + 0.05, crack * 0.4); G = lerp(G, G * 1.5 + 0.05, crack * 0.4); B = lerp(B, B * 1.5 + 0.05, crack * 0.4);

      cr[i] = R; cg[i] = G; cb[i] = B;
      rg[i] = clamp01(0.88 + (grain[i] - 0.5) * 0.09 - stud * 0.14 + crack * 0.06 + du * 0.05);
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- GLASS: dusty, streaked window pane -- */
  _bGlass(N, S) {
    const n = N * N;
    const smear = this._fbm(N, { oct: 3, nx: 22, ny: 5, gain: 0.5, seed: S + 1 });
    const crk = this._fbm(N, { oct: 4, nx: 5, ny: 5, gain: 0.6, seed: S + 2, ridged: true });
    const spat = this._cells(N, Math.max(20, N >> 5) * 3, Math.max(20, N >> 5) * 3, S + 3);
    const wave = this._grab(N, 'bleach', S + 4);
    const dust = this._grab(N, 'grime', S + 5);
    const dustF = this._grab(N, 'grimeF', S + 6);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const GL = C(0xcdd8dc);
    const GRIME = C(0x8f8674);

    for (let i = 0; i < n; i++) {
      const dm = clamp01(sstep(0.42, 0.90, dust[i]) * (0.55 + 0.75 * dustF[i]));
      const sm = sstep(0.55, 0.95, smear[i]) * 0.7;
      const dot = sstep(0.22, 0.06, spat.d[i]);
      const crack = sstep(0.955, 0.995, crk[i]);

      let h = 0.55 + (wave[i] - 0.5) * 0.10 + dm * 0.05 + dot * 0.06;
      h -= crack * 0.45;
      H[i] = clamp01(h);

      let R = GL[0], G = GL[1], B = GL[2];
      const grime = clamp01(dm * 0.85 + sm * 0.35 + dot * 0.6);
      R = lerp(R, GRIME[0], grime * 0.72); G = lerp(G, GRIME[1], grime * 0.72); B = lerp(B, GRIME[2], grime * 0.72);
      R = lerp(R, 0.92, crack * 0.7); G = lerp(G, 0.94, crack * 0.7); B = lerp(B, 0.97, crack * 0.7);
      cr[i] = R; cg[i] = G; cb[i] = B;

      rg[i] = clamp01(0.045 + grime * 0.62 + crack * 0.25 + (wave[i] - 0.5) * 0.03);
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- GUNMETAL: phosphated / bead-blasted receiver steel with edge wear -- */
  _bGunmetal(N, S) {
    const n = N * N;
    const mach = this._fbm(N, { oct: 3, nx: 6, ny: N >> 1, gain: 0.5, seed: S + 1 });
    const scr = this._fbm(N, { oct: 3, nx: N, ny: 7, gain: 0.5, seed: S + 2, ridged: true });
    const blast = this._cells(N, Math.max(40, N >> 4) * 3, Math.max(40, N >> 4) * 3, S + 3);
    const fine = this._grab(N, 'grain', S + 4);
    const wear = this._grab(N, 'wear', S + 5);
    const oil = this._grab(N, 'mid', S + 6);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n), mt = new Float32Array(n);

    const PHOS = C(0x35383b);
    const PHOS2 = C(0x24262a);
    const BARE = C(0x9ea3a7);

    for (let i = 0; i < n; i++) {
      const bl = sstep(0.50, 0.16, blast.d[i]) * 0.55;
      const sc = sstep(0.93, 0.997, scr[i]);
      const wr = sstep(0.62, 0.93, wear[i]);

      let h = 0.55 + bl * 0.10 + (fine[i] - 0.5) * 0.10 + (mach[i] - 0.5) * 0.09;
      h -= sc * 0.10;
      H[i] = clamp01(h);

      const t = clamp01(fine[i] * 0.5 + mach[i] * 0.5);
      let R = lerp(PHOS2[0], PHOS[0], t), G = lerp(PHOS2[1], PHOS[1], t), B = lerp(PHOS2[2], PHOS[2], t);
      // the finish rubs through to bright steel on the high-wear areas
      const rub = clamp01(wr * 0.85 + sc * 0.9);
      R = lerp(R, BARE[0], rub * 0.80); G = lerp(G, BARE[1], rub * 0.80); B = lerp(B, BARE[2], rub * 0.80);
      const ov = sstep(0.60, 0.95, oil[i]) * 0.20;
      R = lerp(R, R * 0.78, ov); G = lerp(G, G * 0.80, ov); B = lerp(B, B * 0.86, ov);

      cr[i] = R; cg[i] = G; cb[i] = B;

      let r = 0.46 + (fine[i] - 0.5) * 0.10 + bl * 0.16 + (mach[i] - 0.5) * 0.08;
      r -= rub * 0.28;
      r -= ov * 0.35;                       // oiled steel reads glossier
      rg[i] = clamp01(r);
      mt[i] = clamp01(0.86 + rub * 0.14);
    }
    return { H, cr, cg, cb, rg, mt };
  }

  /* -- POLYMER: injection-moulded stipple, mould lines, edge polish -- */
  _bPolymer(N, S) {
    const n = N * N;
    const scr = this._fbm(N, { oct: 3, nx: N, ny: 8, gain: 0.5, seed: S + 1, ridged: true });
    const stip = this._cells(N, Math.max(34, N >> 4) * 3, Math.max(34, N >> 4) * 3, S + 2);
    const fine = this._grab(N, 'grain', S + 3);
    const big = this._grab(N, 'big', S + 4);
    const scuff = this._grab(N, 'mid', S + 5);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const BASE = C(0x2e3032);
    const LIFT = C(0x45484a);

    for (let y = 0; y < N; y++) {
      // mould parting line across the middle of the tile
      const ml = 1 - sstep(0, 0.006, Math.abs(y / N - 0.5));
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        const st = sstep(0.46, 0.22, stip.d[i]);
        const sc = sstep(0.94, 0.998, scr[i]);

        let h = 0.52 + st * 0.20 + (fine[i] - 0.5) * 0.09 + (big[i] - 0.5) * 0.08;
        h += ml * 0.10;
        h -= sc * 0.09;
        H[i] = clamp01(h);

        const t = clamp01(st * 0.6 + big[i] * 0.4);
        let R = lerp(BASE[0], LIFT[0], t), G = lerp(BASE[1], LIFT[1], t), B = lerp(BASE[2], LIFT[2], t);
        const fv = (fine[i] - 0.5) * 0.045;
        R += fv; G += fv; B += fv;
        const sv = sstep(0.60, 0.95, scuff[i]) * 0.20;
        R = lerp(R, R * 1.55 + 0.05, sv); G = lerp(G, G * 1.52 + 0.05, sv); B = lerp(B, B * 1.50 + 0.05, sv);
        R = lerp(R, R * 1.4 + 0.05, sc * 0.6); G = lerp(G, G * 1.4 + 0.05, sc * 0.6); B = lerp(B, B * 1.4 + 0.05, sc * 0.6);

        cr[i] = R; cg[i] = G; cb[i] = B;
        rg[i] = clamp01(0.55 + st * 0.16 + (fine[i] - 0.5) * 0.08 - sv * 0.14 - sc * 0.20 - ml * 0.12);
      }
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- FLESH: skin — pores, fine wrinkles, subdermal colour variation -- */
  _bFlesh(N, S) {
    const n = N * N;
    const wr1 = this._fbm(N, { oct: 3, nx: 26, ny: 7, gain: 0.5, seed: S + 1, ridged: true });
    const wr2 = this._fbm(N, { oct: 3, nx: 8, ny: 30, gain: 0.5, seed: S + 2, ridged: true });
    const pore = this._cells(N, Math.max(30, N >> 4) * 3, Math.max(30, N >> 4) * 3, S + 3);
    const cell = this._cells(N, Math.max(12, N >> 6) * 3, Math.max(12, N >> 6) * 3, S + 4);
    const sub = this._grab(N, 'wear', S + 5);
    const fine = this._grab(N, 'grain', S + 6);
    const oily = this._grab(N, 'mid', S + 7);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const SKIN = C(0xb98866);
    const RED = C(0xa8624a);
    const PALE = C(0xd0a883);
    const SHAD = C(0x7c523c);

    for (let i = 0; i < n; i++) {
      const p = sstep(0.34, 0.10, pore.d[i]);
      const micro = sstep(0.55, 0.25, cell.d[i]) * 0.5;
      const w = clamp01(sstep(0.86, 0.99, wr1[i]) + sstep(0.88, 0.99, wr2[i]) * 0.7);

      let h = 0.58 + micro * 0.10 + (fine[i] - 0.5) * 0.06;
      h -= p * 0.22;
      h -= w * 0.20;
      H[i] = clamp01(h);

      const t = clamp01(sub[i] * 1.1 - 0.05);
      let R = lerp(SHAD[0], PALE[0], t), G = lerp(SHAD[1], PALE[1], t), B = lerp(SHAD[2], PALE[2], t);
      R = lerp(R, SKIN[0], 0.45); G = lerp(G, SKIN[1], 0.45); B = lerp(B, SKIN[2], 0.45);
      // blood tone pools in the creases
      const rd = clamp01(w * 0.7 + p * 0.35);
      R = lerp(R, RED[0], rd * 0.55); G = lerp(G, RED[1], rd * 0.55); B = lerp(B, RED[2], rd * 0.55);
      const fv = (fine[i] - 0.5) * 0.05;
      R += fv * 1.1; G += fv * 0.9; B += fv * 0.8;

      cr[i] = R; cg[i] = G; cb[i] = B;

      let r = 0.62 + (fine[i] - 0.5) * 0.10 + p * 0.10;
      r -= sstep(0.55, 0.95, oily[i]) * 0.24;   // sweat / sebum sheen
      rg[i] = clamp01(r);
    }
    return { H, cr, cg, cb, rg };
  }

  /* -- FOLIAGE: dusty olive leaf mass with veins -- */
  _bFoliage(N, S) {
    const n = N * N;
    const vein = this._fbm(N, { oct: 3, nx: 20, ny: 40, gain: 0.5, seed: S + 1, ridged: true });
    const leaf = this._cells(N, Math.max(10, N >> 6) * 3, Math.max(10, N >> 6) * 3, S + 2);
    const mott = this._grab(N, 'mid', S + 3);
    const big = this._grab(N, 'big', S + 4);
    const fine = this._grab(N, 'grain', S + 5);
    const dry = this._grab(N, 'grime', S + 6);

    const H = new Float32Array(n), cr = new Float32Array(n), cg = new Float32Array(n),
      cb = new Float32Array(n), rg = new Float32Array(n);

    const GREEN = C(0x6d7a41);
    const DEEP = C(0x44522c);
    const DRY = C(0x8f7d45);
    const DUST = C(0xa79877);

    for (let i = 0; i < n; i++) {
      const lf = sstep(0.62, 0.20, leaf.d[i]);
      const vn = sstep(0.88, 0.995, vein[i]);
      const dz = sstep(0.60, 0.92, dry[i]);

      let h = 0.48 + lf * 0.30 + (mott[i] - 0.5) * 0.16 + (fine[i] - 0.5) * 0.06;
      h += vn * 0.08;
      H[i] = clamp01(h);

      const t = clamp01(mott[i] * 0.6 + leaf.id[i] * 0.4);
      let R = lerp(DEEP[0], GREEN[0], t), G = lerp(DEEP[1], GREEN[1], t), B = lerp(DEEP[2], GREEN[2], t);
      R = lerp(R, DRY[0], dz * 0.55); G = lerp(G, DRY[1], dz * 0.55); B = lerp(B, DRY[2], dz * 0.55);
      R = lerp(R, R * 1.28 + 0.03, vn * 0.55); G = lerp(G, G * 1.26 + 0.03, vn * 0.55); B = lerp(B, B * 1.20 + 0.02, vn * 0.55);
      const bv = (big[i] - 0.5) * 0.16;
      R += bv * 0.95; G += bv; B += bv * 0.7;
      // everything in this compound wears a film of dust
      const du = sstep(0.40, 0.90, big[i]) * 0.24;
      R = lerp(R, DUST[0], du); G = lerp(G, DUST[1], du); B = lerp(B, DUST[2], du);

      cr[i] = R; cg[i] = G; cb[i] = B;
      rg[i] = clamp01(0.78 + (fine[i] - 0.5) * 0.10 - lf * 0.14 + dz * 0.10);
    }
    return { H, cr, cg, cb, rg };
  }
}

export default Materials;
