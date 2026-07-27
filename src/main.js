// BLACKSITE — bootstrap & frame loop. Owner: core. Agents must NOT edit this file.
import * as THREE from 'three';
import Config from './core/Config.js';
import Bus from './core/Bus.js';

import { Renderer }   from './render/Renderer.js';
import { PostFX }     from './render/PostFX.js';
import { Sky }        from './render/Sky.js';
import { Materials }  from './art/Materials.js';
import { Level }      from './art/Level.js';
import { Physics }    from './physics/Physics.js';
import { Controller } from './player/Controller.js';
import { Weapon }     from './weapons/Weapon.js';
import { Ballistics } from './weapons/Ballistics.js';
import { Particles }  from './fx/Particles.js';
import { Decals }     from './fx/Decals.js';
import { Enemies }    from './ai/Enemies.js';
import { Audio }      from './audio/Audio.js';
import { HUD }        from './ui/HUD.js';
import { Story }      from './story/Story.js';

const bootbar = document.querySelector('#bar > i');
const bootmsg = document.getElementById('bootmsg');
const progress = (p, m) => { if (bootbar) bootbar.style.width = (p * 100).toFixed(0) + '%'; if (m && bootmsg) bootmsg.textContent = m; };

class Game {
  constructor() {
    this.THREE = THREE;
    this.config = Config;
    this.bus = Bus;
    this.clock = new THREE.Clock();
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(Config.weapon.fovHip, innerWidth / innerHeight, 0.05, 900);
    this.time = 0;
    this.frame = 0;
    this.modules = [];
    this.ready = false;
    // Modules register colliders/targets here so others can query without hard imports.
    this.registry = { colliders: [], enemies: [], lights: [], reflectionProbes: [] };
  }

  add(name, mod) { this[name] = mod; this.modules.push(mod); return mod; }

  async init() {
    const steps = [
      ['renderer',   () => new Renderer(this),   'BOOTING RENDERER'],
      ['materials',  () => new Materials(this),  'GENERATING MATERIALS'],
      ['sky',        () => new Sky(this),        'BUILDING ATMOSPHERE'],
      ['level',      () => new Level(this),      'STREAMING GEOMETRY'],
      ['physics',    () => new Physics(this),    'BUILDING COLLISION'],
      ['decals',     () => new Decals(this),     'DECAL SYSTEM'],
      ['particles',  () => new Particles(this),  'PARTICLE SYSTEM'],
      ['controller', () => new Controller(this), 'PLAYER'],
      ['weapon',     () => new Weapon(this),     'WEAPON'],
      ['ballistics', () => new Ballistics(this), 'BALLISTICS'],
      ['enemies',    () => new Enemies(this),    'AI'],
      ['audio',      () => new Audio(this),      'AUDIO'],
      ['hud',        () => new HUD(this),        'HUD'],
      ['post',       () => new PostFX(this),     'POST PROCESSING'],
      // 매불쇼 에디션 — 마지막에 얹는 내러티브 레이어.
      // 앞선 모듈이 전부 준비된 뒤라 game.weapon 등을 안전하게 참조할 수 있다.
      ['story',      () => new Story(this),      '큐시트 준비 중'],
    ];
    for (let i = 0; i < steps.length; i++) {
      const [name, make, msg] = steps[i];
      progress(i / steps.length, msg);
      const mod = this.add(name, make());
      if (mod.init) await mod.init();
      await new Promise(r => requestAnimationFrame(r));
    }
    progress(1, 'READY');
    addEventListener('resize', () => this.resize());
    this.resize();
    this.ready = true;
    document.getElementById('overlay')?.classList.add('hidden');
    this.clock.start();
    this.renderer.renderer.setAnimationLoop(() => this.tick());
  }

  resize() {
    const w = innerWidth, h = innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const m of this.modules) m.resize?.(w, h);
  }

  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    this.time += dt;
    this.frame++;
    for (const m of this.modules) m.update?.(dt, this.time);
    // PostFX owns the final present; if absent, fall back to a direct render.
    if (this.post?.render) this.post.render(dt);
    else this.renderer.renderer.render(this.scene, this.camera);
  }
}

const game = new Game();
window.__GAME = game;
await game.init();
