// WebGL でテクスチャ付きメッシュ（グリッド）を描画するだけの小さなレンダラ。
// 各レイヤーは格子状メッシュを持ち、頂点位置を毎フレーム CPU 側で書き換えて変形させる。

const VS = `
attribute vec2 aPos;
attribute vec2 aUv;
uniform vec2 uView;     // モデル座標 → クリップ座標の変換 (scale)
uniform vec2 uOffset;
varying vec2 vUv;
void main() {
  vec2 p = aPos * uView + uOffset;
  gl_Position = vec4(p.x, -p.y, 0.0, 1.0);
  vUv = aUv;
}`;

const FS = `
precision mediump float;
uniform sampler2D uTex;
uniform float uAlpha;
uniform float uDiscard;
varying vec2 vUv;
void main() {
  vec4 c = texture2D(uTex, vUv) * uAlpha;
  if (c.a <= uDiscard) discard;
  gl_FragColor = c;
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export class Mesh {
  // x, y, w, h: モデル座標上の矩形。cols × rows の格子に分割する。
  constructor(gl, texture, x, y, w, h, cols, rows) {
    this.gl = gl;
    this.texture = texture;
    this.cols = cols;
    this.rows = rows;
    const n = (cols + 1) * (rows + 1);
    this.base = new Float32Array(n * 2); // 変形前の頂点位置
    this.pos = new Float32Array(n * 2); // 変形後の頂点位置（毎フレーム更新）
    const uv = new Float32Array(n * 2);
    let k = 0;
    for (let j = 0; j <= rows; j++) {
      for (let i = 0; i <= cols; i++) {
        const u = i / cols, v = j / rows;
        this.base[k] = x + u * w;
        this.base[k + 1] = y + v * h;
        uv[k] = u;
        uv[k + 1] = v;
        k += 2;
      }
    }
    this.pos.set(this.base);
    const idx = new Uint16Array(cols * rows * 6);
    k = 0;
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const a = j * (cols + 1) + i, b = a + 1, c = a + cols + 1, d = c + 1;
        idx.set([a, b, c, b, d, c], k);
        k += 6;
      }
    }
    this.count = idx.length;
    this.posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.pos, gl.DYNAMIC_DRAW);
    this.uvBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    this.idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  }

  upload() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.pos);
  }
}

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl', {
      alpha: true,
      premultipliedAlpha: true,
      stencil: true,
      antialias: true,
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL が使えません');
    this.gl = gl;
    this.canvas = canvas;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.aPos = gl.getAttribLocation(prog, 'aPos');
    this.aUv = gl.getAttribLocation(prog, 'aUv');
    this.uView = gl.getUniformLocation(prog, 'uView');
    this.uOffset = gl.getUniformLocation(prog, 'uOffset');
    this.uAlpha = gl.getUniformLocation(prog, 'uAlpha');
    this.uDiscard = gl.getUniformLocation(prog, 'uDiscard');
    this.uTex = gl.getUniformLocation(prog, 'uTex');
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    this.background = [0, 0, 0, 0];
  }

  createTexture(source) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(this.canvas.clientWidth * dpr);
    const h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  // view: { cx, cy, size } モデル座標でこの正方形が画面の短辺に収まるように表示
  begin(view) {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const [r, g, b, a] = this.background;
    gl.clearColor(r * a, g * a, b * a, a);
    gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
    const aspect = this.canvas.width / this.canvas.height;
    let sx = 2 / view.size, sy = 2 / view.size;
    if (aspect > 1) sx /= aspect;
    else sy *= aspect;
    gl.useProgram(this.prog);
    gl.uniform2f(this.uView, sx, sy);
    gl.uniform2f(this.uOffset, -view.cx * sx, -view.cy * sy);
    gl.uniform1i(this.uTex, 0);
    gl.activeTexture(gl.TEXTURE0);
  }

  // mode: 'normal' | 'mask'（描画しつつステンシルに書き込む） | 'masked'（ステンシル内のみ描画）
  draw(mesh, alpha = 1, mode = 'normal') {
    const gl = this.gl;
    mesh.upload();
    gl.uniform1f(this.uAlpha, alpha);
    gl.bindTexture(gl.TEXTURE_2D, mesh.texture);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.posBuf);
    gl.enableVertexAttribArray(this.aPos);
    gl.vertexAttribPointer(this.aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, mesh.uvBuf);
    gl.enableVertexAttribArray(this.aUv);
    gl.vertexAttribPointer(this.aUv, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.idxBuf);

    if (mode === 'masked') {
      gl.enable(gl.STENCIL_TEST);
      gl.stencilFunc(gl.EQUAL, 1, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    } else {
      gl.disable(gl.STENCIL_TEST);
    }
    gl.uniform1f(this.uDiscard, 0.002);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);

    if (mode === 'mask') {
      // 色は書かず、不透明な部分だけステンシルに 1 を書く
      gl.enable(gl.STENCIL_TEST);
      gl.stencilFunc(gl.ALWAYS, 1, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
      gl.colorMask(false, false, false, false);
      gl.uniform1f(this.uDiscard, 0.35);
      gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
      gl.colorMask(true, true, true, true);
    }
    gl.disable(gl.STENCIL_TEST);
  }
}
