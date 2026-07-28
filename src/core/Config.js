// Central tunables. Modules READ from here; only Config.js declares defaults.
// Owner: core (do not restructure — add keys under your own namespace only).

export const Config = {
  quality: 'ultra',              // 'low' | 'high' | 'ultra' — set by QualityScaler

  render: {
    exposure: 1.0,
    toneMapping: 'agx',          // 'aces' | 'agx' | 'neutral'
    shadowMapSize: 4096,
    cascades: 4,
    shadowDistance: 140,
    maxAnisotropy: 16,
    pixelRatioCap: 2,
  },

  post: {
    taa: true, smaa: true, ssao: true, ssr: true, bloom: true,
    motionBlur: true, dof: true, chromatic: true, grain: true,
    vignette: true, sharpen: true, lensDirt: true,
  },

  sky: {
    timeOfDay: 16.4,             // hours — late-afternoon golden hour
    turbidity: 6.2,
    rayleigh: 1.4,
    sunIntensity: 3.2,
    godRays: true,
    fogDensity: 0.0072,
  },

  player: {
    height: 1.75, eyeHeight: 1.62, crouchHeight: 1.0, radius: 0.32,
    walkSpeed: 3.4, sprintSpeed: 6.2, crouchSpeed: 1.9, adsSpeedMul: 0.42,
    accel: 48, airAccel: 9, friction: 11, jumpVel: 4.6, gravity: -18.5,
    mouseSensitivity: 0.0022, adsSensMul: 0.62,
  },

  weapon: {
    fovHip: 68, fovAds: 52, viewmodelFov: 55,
    adsTime: 0.17, rpm: 780, magSize: 30, reserve: 210,
    damage: 34, headMul: 2.4, range: 220,
  },

  // 매불쇼 에디션 — 서바이벌 특집 설정. 상대 팀은 원본 엔진의 익명 병사 그대로다.
  // 원본은 14. 스토리 진행이 막히지 않도록 조금 낮춰 10으로 둔다.
  ai: { count: 10, sightRange: 90, fov: 120, reactionTime: 0.28 },

  // 3인칭 아바타 외형 — 'staff'(매불쇼 스태프) | 'operator'(원본)
  avatar: { variant: 'staff' },

  audio: { master: 0.8, sfx: 1.0, music: 0.35 },

  debug: { stats: false, freecam: false },
};

export default Config;
