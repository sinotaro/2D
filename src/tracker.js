// MediaPipe Face Landmarker でカメラ映像から顔の向き・まばたき・口の開きなどを取り出す。

const params = new URLSearchParams(location.search);
// ?mp=<URL> で MediaPipe の配置先を差し替えられる（オフライン環境・自前ホスト用）
const MP_BASE = params.get('mp') || 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0';
const MODEL_URL = params.get('model') ||
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export class FaceTracker {
  constructor() {
    this.landmarker = null;
    this.video = null;
    this.lastTime = -1;
    this.mirror = true;
    // 正面を向いたときの基準値（キャリブレーション）
    this.neutral = null;
    this.calibFrames = [];
    // まばたきの左右対応を自動判定するための相関
    this.blinkCorr = 0;
  }

  async init(onStatus = () => {}) {
    onStatus('顔認識モデルを読み込み中…');
    const { FaceLandmarker, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
    });
    try {
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
    } catch (e) {
      console.warn('GPU delegate failed, falling back to CPU', e);
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
    }
  }

  async startCamera(video, deviceId) {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 },
      },
      audio: false,
    });
    video.srcObject = this.stream;
    await video.play();
    this.video = video;
  }

  calibrate() {
    this.neutral = null;
    this.calibFrames = [];
  }

  // 新しいフレームがあれば解析して生のパラメータを返す。顔が無ければ null。
  detect(now) {
    const v = this.video;
    if (!this.landmarker || !v || v.readyState < 2) return undefined;
    if (v.currentTime === this.lastTime) return undefined;
    this.lastTime = v.currentTime;
    const res = this.landmarker.detectForVideo(v, now);
    if (!res.faceLandmarks || !res.faceLandmarks.length) return null;
    const bs = {};
    for (const c of res.faceBlendshapes?.[0]?.categories || []) bs[c.categoryName] = c.score;
    return this.solve(res.faceLandmarks[0], bs, v.videoWidth, v.videoHeight);
  }

  solve(lm, bs, W, H) {
    const mir = this.mirror;
    // 画面に映る向き（ミラー時は左右反転）のピクセル座標
    const P = (i) => ({ x: (mir ? 1 - lm[i].x : lm[i].x) * W, y: lm[i].y * H });

    // 傾き：両目の外側の端を結ぶ線の角度
    let a = P(33), b = P(263);
    if (a.x > b.x) [a, b] = [b, a];
    const roll = Math.atan2(b.y - a.y, b.x - a.x);
    const eyeMid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    // 傾きを打ち消した座標系に直す
    const c = Math.cos(-roll), s = Math.sin(-roll);
    const R = (i) => {
      const q = P(i);
      const dx = q.x - eyeMid.x, dy = q.y - eyeMid.y;
      return { x: dx * c - dy * s, y: dx * s + dy * c };
    };

    const nose = R(1), chin = R(152);
    let l = R(234), r = R(454);
    if (l.x > r.x) [l, r] = [r, l];
    const halfW = (r.x - l.x) / 2;
    const faceH = chin.y; // 目の高さ→あごまでの距離
    const yawRaw = (nose.x - (l.x + r.x) / 2) / halfW;
    const pitchRaw = nose.y / faceH;

    // 顔の位置（画面中央からのずれ、顔の幅基準）
    const center = { x: eyeMid.x / W - 0.5, y: eyeMid.y / H - 0.5 };
    const faceSize = (halfW * 2) / W;

    // 目の開き（ブレンドシェイプ）。どちらが画面左の目かはランドマークとの相関で自動判定する
    const ear = (up, lo, c1, c2) => {
      const u = P(up), d = P(lo), e1 = P(c1), e2 = P(c2);
      return Math.hypot(u.x - d.x, u.y - d.y) / Math.max(1, Math.hypot(e1.x - e2.x, e1.y - e2.y));
    };
    // 被写体の右目 (33,133,159,145) / 左目 (263,362,386,374)
    const earSR = ear(159, 145, 33, 133), earSL = ear(386, 374, 263, 362);
    const subjRightOnScreenLeft = P(33).x < P(263).x;
    const earScreenL = subjRightOnScreenLeft ? earSR : earSL;
    const earScreenR = subjRightOnScreenLeft ? earSL : earSR;
    const bl = bs.eyeBlinkLeft ?? 0, br = bs.eyeBlinkRight ?? 0;
    this.blinkCorr = clamp(this.blinkCorr * 0.995 + (earScreenL - earScreenR) * (bl - br), -1, 1);
    // EAR が小さい（閉じている）目ほど blink が大きいはず → 相関が負なら Left=画面左
    const leftIsScreenLeft = this.blinkCorr !== 0 ? this.blinkCorr < 0 : mir;
    const blinkScreenL = leftIsScreenLeft ? bl : br;
    const blinkScreenR = leftIsScreenLeft ? br : bl;

    // 視線：虹彩の中心が目の端と端のどこにあるか
    const irisX = (iris, c1, c2) => {
      const q = R(iris), e1 = R(c1), e2 = R(c2);
      const mid = (e1.x + e2.x) / 2, half = Math.abs(e2.x - e1.x) / 2;
      return (q.x - mid) / Math.max(1, half);
    };
    let gazeX = 0;
    if (lm.length > 473) gazeX = (irisX(468, 33, 133) + irisX(473, 263, 362)) / 2;
    const lookUp = ((bs.eyeLookUpLeft ?? 0) + (bs.eyeLookUpRight ?? 0)) / 2;
    const lookDown = ((bs.eyeLookDownLeft ?? 0) + (bs.eyeLookDownRight ?? 0)) / 2;

    // 口：上下の唇の内側の距離
    const lipGap = Math.hypot(P(13).x - P(14).x, P(13).y - P(14).y) / Math.max(1, faceH);

    const raw = {
      roll, yawRaw, pitchRaw, cx: center.x, cy: center.y, faceSize,
      blinkL: blinkScreenL, blinkR: blinkScreenR,
      gazeX, gazeY: lookDown - lookUp,
      lipGap, jawOpen: bs.jawOpen ?? 0,
      smile: ((bs.mouthSmileLeft ?? 0) + (bs.mouthSmileRight ?? 0)) / 2,
      frown: ((bs.mouthFrownLeft ?? 0) + (bs.mouthFrownRight ?? 0)) / 2,
      browUp: (bs.browInnerUp ?? 0) * 0.6 + ((bs.browOuterUpLeft ?? 0) + (bs.browOuterUpRight ?? 0)) * 0.2,
      browDown: ((bs.browDownLeft ?? 0) + (bs.browDownRight ?? 0)) / 2,
    };

    this.lastRaw = raw;
    // 最初の数フレームを正面の基準として記録
    if (!this.neutral) {
      this.calibFrames.push(raw);
      if (this.calibFrames.length >= 15) {
        const avg = {};
        for (const k of Object.keys(raw)) {
          avg[k] = this.calibFrames.reduce((s, f) => s + f[k], 0) / this.calibFrames.length;
        }
        this.neutral = avg;
      }
    }
    return this.toParams(raw, this.neutral || raw);
  }

  // 生の値 → リグのパラメータ
  toParams(r, n) {
    const blink = (v, base) => 1 - clamp((v - Math.max(0.12, base + 0.1)) / 0.35, 0, 1);
    let eyeL = blink(r.blinkL, Math.min(n.blinkL, 0.35));
    let eyeR = blink(r.blinkR, Math.min(n.blinkR, 0.35));
    // 左右差が小さいときはそろえる（片目だけ半開きになるのを防ぐ）
    if (Math.abs(eyeL - eyeR) < 0.3) eyeL = eyeR = (eyeL + eyeR) / 2;
    const open = Math.max(
      clamp((r.lipGap - n.lipGap - 0.01) / 0.12, 0, 1),
      clamp((r.jawOpen - n.jawOpen - 0.04) / 0.45, 0, 1),
    );
    const scale = 1 / Math.max(0.05, n.faceSize);
    return {
      yaw: clamp((r.yawRaw - n.yawRaw) * 1.3, -0.75, 0.75),
      pitch: clamp((r.pitchRaw - n.pitchRaw) * 3.2, -0.6, 0.6),
      roll: clamp(r.roll, -0.6, 0.6), // 傾きは目の線の角度そのもの（基準合わせ不要）
      tx: clamp((r.cx - n.cx) * scale * 60, -80, 80),
      // 上下の移動は控えめに（大きいと首が伸び縮みして見える）
      ty: clamp((r.cy - n.cy) * scale * 20, -20, 20),
      eyeL, eyeR,
      gazeX: clamp((r.gazeX - n.gazeX) * 2.2, -1, 1),
      gazeY: clamp((r.gazeY - n.gazeY) * 1.5, -1, 1),
      mouthOpen: open,
      smile: clamp((r.smile - n.smile) * 1.6 - (r.frown - n.frown) * 1.2, -0.6, 1),
      brow: clamp((r.browUp - n.browUp) * 3 - (r.browDown - n.browDown) * 2, -1, 1),
    };
  }
}
