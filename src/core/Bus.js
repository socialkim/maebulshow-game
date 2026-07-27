// Tiny synchronous event bus. Owner: core.
// Canonical events (emit/listen — do not invent parallel names):
//   'shot'        {origin, dir, weapon}
//   'impact'      {point, normal, material, surface}   surface: 'concrete'|'metal'|'sand'|'wood'|'flesh'|'glass'
//   'hit'         {enemy, point, damage, headshot}
//   'kill'        {enemy, headshot, distance}
//   'damage'      {amount, fromDir}
//   'reload'      {phase:'start'|'magout'|'magin'|'end'}
//   'ads'         {active}
//   'footstep'    {surface, speed}
//   'explosion'   {point, radius, power}
//   'ammo'        {mag, reserve}
//   'health'      {hp, max}

class EventBus {
  constructor() { this.map = new Map(); }
  on(evt, fn) { (this.map.get(evt) ?? this.map.set(evt, []).get(evt)).push(fn); return () => this.off(evt, fn); }
  off(evt, fn) { const a = this.map.get(evt); if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); } }
  emit(evt, data) {
    const a = this.map.get(evt); if (!a) return;
    for (let i = 0; i < a.length; i++) { try { a[i](data); } catch (e) { console.error(`[bus:${evt}]`, e); } }
  }
}
export const Bus = new EventBus();
export default Bus;
