// PSD レイヤーをパラメータ（顔の向き・まばたき・口の開き等）に応じて変形させるリグ。
// 座標はすべて PSD のピクセル座標（モデル座標）。

import { Mesh } from './renderer.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

// レイヤー名ごとの役割。
// depth は顔の向きを変えたときの「奥行き」による追加のずれ（頭の半径に対する割合）。
// 頭全体は1つの丸い面として一緒に変形するので、ここは小さな値にしておく。
// 大きくするとパーツ同士がずれて隙間ができる。
const ROLES = {
  'back hair': { part: 'head', depth: -0.03, swing: 0.4 },
  'neck': { part: 'neck' },
  'topwear': { part: 'body' },
  'neckwear': { part: 'body' },
  'ears': { part: 'head', depth: -0.02 },
  'face': { part: 'head', depth: 0 },
  'nose': { part: 'head', depth: 0.06 },
  'mouth': { part: 'head', depth: 0.035, deform: 'mouth' },
  'eyewhite': { part: 'head', depth: 0.03, deform: 'eyewhite', mode: 'mask' },
  'irides': { part: 'head', depth: 0.032, deform: 'iris', mode: 'masked' },
  'eyelash': { part: 'head', depth: 0.034, deform: 'lash' },
  'eyebrow': { part: 'head', depth: 0.036, deform: 'brow' },
  'front hair': { part: 'head', depth: 0.03, swing: 0.6 },
};

export const DEFAULT_PARAMS = {
  yaw: 0, pitch: 0, roll: 0, tx: 0, ty: 0,
  eyeL: 1, eyeR: 1, gazeX: 0, gazeY: 0,
  mouthOpen: 0, smile: 0, brow: 0,
};

// 口の中のテクスチャを canvas で生成
function mouthTexture() {
  const c = document.createElement('canvas');
  c.width = 128;
  c.height = 96;
  const g = c.getContext('2d');
  g.beginPath();
  g.moveTo(4, 4);
  g.quadraticCurveTo(64, -2, 124, 4);
  g.bezierCurveTo(118, 70, 92, 92, 64, 92);
  g.bezierCurveTo(36, 92, 10, 70, 4, 4);
  g.closePath();
  g.save();
  g.clip();
  g.fillStyle = '#4a1a22';
  g.fillRect(0, 0, 128, 96);
  // 舌
  g.fillStyle = '#c4636b';
  g.beginPath();
  g.ellipse(64, 96, 42, 34, 0, 0, Math.PI * 2);
  g.fill();
  // 上の歯
  g.fillStyle = '#fbf6f4';
  g.beginPath();
  g.moveTo(14, 0);
  g.lineTo(114, 0);
  g.quadraticCurveTo(64, 26, 14, 0);
  g.fill();
  g.restore();
  // 輪郭
  g.lineWidth = 5;
  g.strokeStyle = '#3a1418';
  g.stroke();
  return c;
}

export class Rig {
  constructor(renderer, model, images) {
    this.model = model;
    const fb = model.faceBox || [0, 0, model.width, model.height];
    this.center = { x: (fb[0] + fb[2]) / 2, y: (fb[1] + fb[3]) / 2 };
    // 頭全体（髪・耳を含む）の範囲。向きを変えるときはこの範囲を1つの丸い面として変形する
    const hb = [Infinity, Infinity, -Infinity, -Infinity];
    for (const L of model.layers) {
      if (ROLES[L.name]?.part !== 'head') continue;
      hb[0] = Math.min(hb[0], L.x); hb[1] = Math.min(hb[1], L.y);
      hb[2] = Math.max(hb[2], L.x + L.w); hb[3] = Math.max(hb[3], L.y + L.h);
    }
    if (!isFinite(hb[0])) hb.splice(0, 4, ...fb);
    this.head = {
      cx: (hb[0] + hb[2]) / 2,
      rx: (hb[2] - hb[0]) / 2,
      ry: Math.max(hb[3] - this.center.y, this.center.y - hb[1]),
    };
    // 首の付け根（頭の傾きの回転中心）
    this.pivot = { x: this.center.x, y: fb[3] - 30 };
    this.bodyPivot = { x: this.center.x, y: model.height + 40 };
    this.chinY = fb[3];

    // 目の形状
    this.eyes = {};
    for (const [side, e] of Object.entries(model.eyes || {})) {
      const [x0, y0, x1, y1] = e.white;
      const lash = e.lash || e.white;
      const h = y1 - y0;
      this.eyes[side] = {
        cx: (x0 + x1) / 2,
        hw: (x1 - x0) / 2,
        closeY: y0 + h * 0.8, // まぶたが閉じる位置
        lidY: y0 + 4, // 上まぶたの線
        lashTop: lash[1],
        lashHw: (lash[2] - lash[0]) / 2,
        lashCx: (lash[0] + lash[2]) / 2,
      };
    }
    const m = model.mouth || [this.center.x - 30, this.center.y + 160, this.center.x + 30, this.center.y + 170];
    this.mouth = { cx: (m[0] + m[2]) / 2, y: (m[1] + m[3]) / 2, hw: (m[2] - m[0]) / 2 };

    this.layers = [];
    const gl = renderer.gl;
    for (const L of model.layers) {
      const role = ROLES[L.name] || {
        part: L.y + L.h / 2 < this.pivot.y ? 'head' : 'body', depth: 0.02,
      };
      const fine = role.deform === 'lash' || role.deform === 'eyewhite' || role.deform === 'iris';
      const cols = Math.min(80, Math.max(2, Math.ceil(L.w / (fine ? 6 : 20))));
      const rows = Math.min(64, Math.max(2, Math.ceil(L.h / (fine ? 2 : 20))));
      const tex = renderer.createTexture(images[L.file]);
      const mesh = new Mesh(gl, tex, L.x, L.y, L.w, L.h, cols, rows);
      const layer = { name: L.name, role, mesh, opacity: L.opacity ?? 1, top: L.y, bottom: L.y + L.h };
      if (L.name === 'mouth') {
        // 口の中を口レイヤーの直前に差し込む
        const w = this.mouth.hw * 2 * 0.95;
        const inner = new Mesh(gl, renderer.createTexture(mouthTexture()),
          this.mouth.cx - w / 2, this.mouth.y - 1, w, 1, 8, 6);
        this.layers.push({ name: 'mouth inner', role: { part: 'head', depth: 0.155, deform: 'mouthInner' }, mesh: inner, opacity: 1 });
      }
      this.layers.push(layer);
    }

    // 髪揺れ用のばね
    this.spring = { x: 0, y: 0, vx: 0, vy: 0, px: null, py: null };
    this.time = 0;
  }

  // 頭パーツの一点を、顔の向き＋傾き＋移動で変換
  headTransform(x, y, depth, p, out) {
    const H = this.head;
    const u = clamp((x - H.cx) / H.rx, -1, 1);
    const v = clamp((y - this.center.y) / H.ry, -1, 1);
    // 頭全体を1つの丸い面とみなす：中央ほど大きく動き、外側の輪郭はほとんど動かない。
    // 全レイヤーが同じ変形を共有するので、後ろ髪・顔・前髪の間に隙間ができない。
    const bx = Math.cos(u * Math.PI / 2) * (1 - 0.35 * v * v);
    const by = Math.cos(v * Math.PI / 2) * (1 - 0.35 * u * u);
    let X = x + p.yaw * H.rx * (0.2 * bx + depth + 0.05);
    let Y = y + p.pitch * H.ry * (0.13 * by + depth * 0.6 + 0.03);
    X += p.tx;
    Y += p.ty;
    const dx = X - this.pivot.x, dy = Y - this.pivot.y;
    const c = Math.cos(p.roll), s = Math.sin(p.roll);
    out[0] = this.pivot.x + dx * c - dy * s;
    out[1] = this.pivot.y + dx * s + dy * c;
  }

  bodyTransform(x, y, p, out) {
    const P = this.bodyPivot;
    const breath = Math.sin(this.time * 2 * Math.PI / 3.6) * 0.004;
    let X = x + p.tx * 0.3 + p.yaw * 10;
    let Y = P.y - (P.y - y) * (1 + breath) + p.ty * 0.15;
    const r = p.roll * 0.15;
    const dx = X - P.x, dy = Y - P.y;
    const c = Math.cos(r), s = Math.sin(r);
    out[0] = P.x + dx * c - dy * s;
    out[1] = P.y + dx * s + dy * c;
  }

  eyeOf(x) {
    return x < this.center.x ? 'L' : 'R';
  }

  // レイヤー固有の局所変形（目・口・眉）
  localDeform(kind, x, y, p, out) {
    out[0] = x;
    out[1] = y;
    if (!kind) return;
    if (kind === 'eyewhite' || kind === 'iris' || kind === 'lash') {
      const side = this.eyeOf(x);
      const e = this.eyes[side];
      if (!e) return;
      const open = clamp(side === 'L' ? p.eyeL : p.eyeR, 0, 1);
      if (kind === 'iris') {
        out[0] = x + p.gazeX * e.hw * 0.28;
        y += p.gazeY * 5;
      }
      if (kind === 'eyewhite' || kind === 'iris') {
        out[1] = e.closeY + (y - e.closeY) * open;
        return;
      }
      // まつ毛：上まぶたを closeY まで下ろす（目尻・目頭はあまり動かさない）
      const t = 1 - open;
      const ux = clamp((x - e.lashCx) / e.lashHw, -1, 1);
      const wx = 1 - 0.75 * ux * ux;
      const S = (e.closeY - 2 - e.lidY) * t * wx;
      if (y <= e.lidY) {
        const k = clamp((y - e.lashTop) / Math.max(1, e.lidY - e.lashTop), 0, 1);
        out[1] = y + S * (0.5 + 0.5 * k);
      } else if (y <= e.closeY) {
        out[1] = y + S * (e.closeY - y) / (e.closeY - e.lidY);
      }
      return;
    }
    if (kind === 'brow') {
      const side = this.eyeOf(x);
      const open = side === 'L' ? p.eyeL : p.eyeR;
      out[1] = y - p.brow * 9 + (1 - clamp(open, 0, 1)) * 3;
      return;
    }
    if (kind === 'mouth' || kind === 'mouthInner') {
      const M = this.mouth;
      const widen = 1 + 0.22 * p.smile - 0.1 * p.mouthOpen;
      const u = clamp((x - M.cx) / M.hw, -1.2, 1.2);
      out[0] = M.cx + (x - M.cx) * widen;
      const curve = -p.smile * 5 * u * u;
      if (kind === 'mouth') {
        out[1] = y + curve - p.mouthOpen * 2;
      } else {
        // 上端は口の線、下端は開き具合に応じて下がる
        const depthPx = M.hw * 2 * 0.5 * p.mouthOpen * (1 - 0.35 * u * u);
        const k = (y - (M.y - 1)); // 0..1 (rest 高さ 1)
        out[1] = M.y - 1 - p.mouthOpen * 2 + curve + k * depthPx;
      }
    }
  }

  update(p, dt) {
    this.time += dt;
    // 髪の揺れ：頭の動きに遅れてついてくるばね
    const anchor = [0, 0];
    this.headTransform(this.center.x, this.center.y, 0, p, anchor);
    const sp = this.spring;
    if (sp.px === null) { sp.px = anchor[0]; sp.py = anchor[1]; }
    const k = 120, d = 11, h = Math.min(dt, 1 / 30);
    // ばねの基準点が動いた分だけ慣性で遅れる
    sp.x -= anchor[0] - sp.px;
    sp.y -= anchor[1] - sp.py;
    sp.px = anchor[0];
    sp.py = anchor[1];
    sp.vx += (-k * sp.x - d * sp.vx) * h;
    sp.vy += (-k * sp.y - d * sp.vy) * h;
    sp.x = clamp(sp.x + sp.vx * h, -18, 18);
    sp.y = clamp(sp.y + sp.vy * h, -10, 10);

    const tmp = [0, 0], hp = [0, 0], bp = [0, 0];
    for (const layer of this.layers) {
      const { role, mesh } = layer;
      const base = mesh.base, pos = mesh.pos;
      const n = base.length;
      const span = layer.bottom - layer.top;
      for (let i = 0; i < n; i += 2) {
        const x = base[i], y = base[i + 1];
        if (role.part === 'body') {
          this.bodyTransform(x, y, p, tmp);
        } else if (role.part === 'neck') {
          this.bodyTransform(x, y, p, bp);
          this.headTransform(x, y, 0, p, hp);
          const w = smoothstep(this.chinY + 150, this.chinY - 20, y);
          tmp[0] = lerp(bp[0], hp[0], w);
          tmp[1] = lerp(bp[1], hp[1], w);
        } else {
          this.localDeform(role.deform, x, y, p, tmp);
          let lx = tmp[0], ly = tmp[1];
          if (role.swing) {
            // 根元（上）は固定、毛先（下）ほど揺れる
            const w = Math.pow(clamp((y - layer.top) / span, 0, 1), 2.2) * role.swing;
            lx += sp.x * w;
            ly += sp.y * w * 0.5;
          }
          this.headTransform(lx, ly, role.depth, p, tmp);
        }
        pos[i] = tmp[0];
        pos[i + 1] = tmp[1];
      }
    }
  }

  draw(renderer) {
    for (const layer of this.layers) {
      renderer.draw(layer.mesh, layer.opacity, layer.role.mode || 'normal');
    }
  }
}
