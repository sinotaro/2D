import { Renderer } from './renderer.js';
import { Rig, DEFAULT_PARAMS } from './rig.js';
import { FaceTracker } from './tracker.js';

const $ = (id) => document.getElementById(id);
const MODEL_DIR = 'assets/model/';

const state = {
  params: { ...DEFAULT_PARAMS },
  target: { ...DEFAULT_PARAMS },
  source: 'idle', // 'idle' | 'camera' | 'mouse'
  lastFace: 0,
  smoothing: 0.5,
};

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`画像を読み込めません: ${src}`));
    img.src = src;
  });
}

function setStatus(text, isError = false) {
  const el = $('status');
  el.textContent = text;
  el.classList.toggle('error', isError);
}

async function main() {
  const renderer = new Renderer($('stage'));
  // 1ファイル版（vtuber.html）ではモデルが埋め込まれている
  const embedded = window.__EMBEDDED_MODEL;
  const model = embedded ? embedded.model : await (await fetch(MODEL_DIR + 'model.json')).json();
  const images = {};
  await Promise.all(model.layers.map(async (L) => {
    images[L.file] = await loadImage(embedded ? embedded.images[L.file] : MODEL_DIR + L.file);
  }));
  const rig = new Rig(renderer, model, images);
  const fb = model.faceBox || [0, 0, model.width, model.height];
  const view = { cx: model.width / 2, cy: model.height / 2, size: model.height };
  const zoomTo = (z) => {
    // 1 = 全身, 大きいほど顔に寄る
    const faceCy = (fb[1] + fb[3]) / 2;
    view.size = model.height / z;
    view.cx = model.width / 2;
    view.cy = Math.max(view.size / 2, lerp(model.height / 2, faceCy + 60, (z - 1) / 1.5));
  };
  const tracker = new FaceTracker();
  window.__avatar = { rig, state, tracker, view };

  // ---- UI ----
  setStatus('「カメラ開始」を押すと、カメラに映ったあなたの動きに合わせて動きます');
  $('start').addEventListener('click', async () => {
    $('start').disabled = true;
    try {
      if (!tracker.landmarker) await tracker.init(setStatus);
      setStatus('カメラを起動中…');
      await tracker.startCamera($('video'), $('camera').value || undefined);
      await listCameras();
      state.source = 'camera';
      setStatus('トラッキング中：正面を向いて「正面リセット」を押すと基準を合わせ直せます');
      $('start').textContent = 'カメラ再起動';
    } catch (e) {
      console.error(e);
      setStatus(`カメラを開始できませんでした：${e.message || e}`, true);
    } finally {
      $('start').disabled = false;
    }
  });
  $('calibrate').addEventListener('click', () => {
    tracker.calibrate();
    setStatus('正面の基準を取り直しました');
  });
  $('mirror').addEventListener('change', (e) => {
    tracker.mirror = e.target.checked;
    $('video').classList.toggle('mirror', tracker.mirror);
    tracker.calibrate();
  });
  $('preview').addEventListener('change', (e) => {
    $('video').hidden = !e.target.checked;
  });
  $('smooth').addEventListener('input', (e) => { state.smoothing = +e.target.value; });
  $('zoom').addEventListener('input', (e) => zoomTo(+e.target.value));
  $('bg').addEventListener('change', (e) => {
    const v = e.target.value;
    document.body.dataset.bg = v;
    renderer.background = {
      transparent: [0, 0, 0, 0],
      green: [0, 1, 0, 1],
      blue: [0, 0, 1, 1],
      white: [1, 1, 1, 1],
    }[v] || [0, 0, 0, 0];
  });
  $('hide-ui').addEventListener('click', () => document.body.classList.toggle('ui-hidden'));
  window.addEventListener('keydown', (e) => {
    if (e.key === 'h' || e.key === 'H') document.body.classList.toggle('ui-hidden');
    if (e.key === 'c' || e.key === 'C') $('calibrate').click();
  });
  $('camera').addEventListener('change', () => {
    if (state.source === 'camera') $('start').click();
  });

  // カメラがない時はマウスで操作できる
  $('stage').addEventListener('pointermove', (e) => {
    if (state.source === 'camera') return;
    state.source = 'mouse';
    const r = e.currentTarget.getBoundingClientRect();
    const mx = (e.clientX - r.left) / r.width - 0.5;
    const my = (e.clientY - r.top) / r.height - 0.5;
    Object.assign(state.target, { yaw: mx * 1.2, pitch: my * 0.9, gazeX: mx * 2, gazeY: my * 1.5, roll: -mx * 0.15 });
  });
  $('stage').addEventListener('pointerdown', () => {
    if (state.source !== 'camera') state.target.mouthOpen = 0.8;
  });
  window.addEventListener('pointerup', () => {
    if (state.source !== 'camera') state.target.mouthOpen = 0;
  });

  async function listCameras() {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    const sel = $('camera');
    const current = sel.value;
    sel.innerHTML = '<option value="">既定のカメラ</option>' +
      devices.map((d, i) => `<option value="${d.deviceId}">${d.label || `カメラ ${i + 1}`}</option>`).join('');
    sel.value = devices.some((d) => d.deviceId === current) ? current : '';
  }

  // ---- 自動まばたき（カメラ未使用 or 顔を見失った時） ----
  let nextBlink = 2, blinkT = -1;
  function idleMotion(t, dt) {
    const T = state.target;
    if (state.source === 'idle') {
      T.yaw = Math.sin(t * 0.5) * 0.15;
      T.pitch = Math.sin(t * 0.37) * 0.06;
      T.roll = Math.sin(t * 0.29) * 0.05;
      T.gazeX = Math.sin(t * 0.5) * 0.3;
    }
    nextBlink -= dt;
    if (nextBlink <= 0) { blinkT = 0; nextBlink = 2 + Math.random() * 4; }
    if (blinkT >= 0) {
      blinkT += dt;
      const o = blinkT < 0.08 ? 1 - blinkT / 0.08 : Math.min(1, (blinkT - 0.08) / 0.12);
      T.eyeL = T.eyeR = o;
      if (blinkT > 0.2) blinkT = -1;
    }
  }

  // ---- メインループ ----
  let prev = performance.now();
  let frames = 0, fpsT = 0;
  function frame(now) {
    const dt = Math.min(0.1, (now - prev) / 1000);
    prev = now;
    const t = now / 1000;

    if (state.source === 'camera') {
      const result = tracker.detect(now);
      if (result) {
        Object.assign(state.target, result);
        state.lastFace = t;
      } else if (result === null && t - state.lastFace > 1) {
        // 顔を見失ったら正面に戻す
        Object.assign(state.target, { yaw: 0, pitch: 0, roll: 0, tx: 0, ty: 0, mouthOpen: 0, smile: 0, brow: 0, gazeX: 0, gazeY: 0 });
        idleMotion(t, dt);
      }
    } else {
      idleMotion(t, dt);
    }

    // 平滑化（フレームレートに依存しない指数平滑）
    const P = state.params, T = state.target;
    const s = state.smoothing;
    for (const k of Object.keys(P)) {
      // まばたき・口は速く追従させる
      const fast = k === 'eyeL' || k === 'eyeR' || k === 'mouthOpen';
      const tau = (fast ? 0.02 : 0.03) + s * (fast ? 0.05 : 0.15);
      P[k] += (T[k] - P[k]) * (1 - Math.exp(-dt / tau));
    }

    rig.update(P, dt);
    renderer.begin(view);
    rig.draw(renderer);

    frames++;
    fpsT += dt;
    if (fpsT > 0.5) {
      $('fps').textContent = `${Math.round(frames / fpsT)} fps`;
      frames = 0;
      fpsT = 0;
    }
    requestAnimationFrame(frame);
  }
  zoomTo(+$('zoom').value);
  requestAnimationFrame(frame);
}

function lerp(a, b, t) { return a + (b - a) * t; }

main().catch((e) => {
  console.error(e);
  setStatus(`読み込みに失敗しました：${e.message || e}（ローカルサーバー経由で開いてください）`, true);
});
