// worker.js — обработка изображений вне главного потока.
// Загружает TF.js, применяет ML-модель для подбора параметров и
// коррекцию через WebGL-шейдер (OffscreenCanvas). Шлёт события прогресса.

import { EnhancementModel } from './model.js';

let tfReady = (async () => {
  const tf = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/+esm');
  self.tf = tf;
  let ok = false;
  try { ok = await tf.setBackend('webgl'); } catch (e) { ok = false; }
  if (!ok) { try { await tf.setBackend('cpu'); } catch (e) {} }
  await tf.ready();
})();

const model = new EnhancementModel();
const tasks = new Map(); // taskId -> { status, progress, cancelled, result }

function emit(taskId, status, progress) {
  const t = tasks.get(taskId);
  if (t) { t.status = status; t.progress = progress; }
  self.postMessage({ type: 'statusChange', taskId, status, progress });
}

// Уменьшенное превью для анализа моделью (макс. сторона 160 px).
function makePreview(bitmap) {
  const maxSide = 160;
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// --- WebGL коррекция ---
const VERT = `
attribute vec2 aPos;
attribute vec2 aTex;
varying vec2 vTex;
void main(){ vTex = aTex; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
varying vec2 vTex;
uniform sampler2D uTex;
uniform vec3  uBlack;     // точка чёрного по каналам
uniform vec3  uWhite;     // точка белого по каналам
uniform float uGamma;     // светлота средних тонов
uniform float uContrast;  // сила S-кривой
uniform float uVibrance;  // цветность

void main(){
  vec4 src = texture2D(uTex, vTex);
  vec3 c = src.rgb;

  // 1. Автоуровни: раскрытие тонального диапазона на полную,
  //    поканальные точки заодно смягчают цветовой сдвиг.
  c = (c - uBlack) / max(uWhite - uBlack, vec3(1e-4));
  c = clamp(c, 0.0, 1.0);

  // 2. Гамма: светлота средних тонов без клиппинга краёв.
  c = pow(c, vec3(uGamma));

  // 3. S-образная кривая контраста: добавляет объём,
  //    но не выбивает света и тени, в отличие от линейной.
  vec3 s = c * c * (3.0 - 2.0 * c);
  c = mix(c, s, uContrast);

  // 4. Vibrance: сильнее поднимает блёклые цвета и щадит уже насыщенные,
  //    поэтому кожа и небо не уходят в кислоту.
  float l  = dot(c, vec3(0.299, 0.587, 0.114));
  float mx = max(c.r, max(c.g, c.b));
  float mn = min(c.r, min(c.g, c.b));
  float sat = mx > 0.0 ? (mx - mn) / mx : 0.0;
  // при подъёме щадим уже насыщенные пиксели, при снижении работаем в полную силу
  float amt = uVibrance * (uVibrance > 0.0 ? (1.0 - sat) : 1.0);
  c = mix(vec3(l), c, 1.0 + amt);

  gl_FragColor = vec4(clamp(c, 0.0, 1.0), src.a);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function applyWebGL(bitmap, p) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const gl = canvas.getContext('webgl', {
    premultipliedAlpha: false,
    preserveDrawingBuffer: true   // иначе буфер очистится до чтения результата
  });
  if (!gl) return applyCPU(bitmap, p);

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return applyCPU(bitmap, p);
  gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 0, 1,  1, -1, 1, 1,  -1, 1, 0, 0,
    -1, 1, 0, 0,   1, -1, 1, 1,   1, 1, 1, 0,
  ]), gl.STATIC_DRAW);

  const aPos = gl.getAttribLocation(prog, 'aPos');
  const aTex = gl.getAttribLocation(prog, 'aTex');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(aTex);
  gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 16, 8);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);

  gl.uniform3f(gl.getUniformLocation(prog, 'uBlack'), p.black[0], p.black[1], p.black[2]);
  gl.uniform3f(gl.getUniformLocation(prog, 'uWhite'), p.white[0], p.white[1], p.white[2]);
  gl.uniform1f(gl.getUniformLocation(prog, 'uGamma'), p.gamma);
  gl.uniform1f(gl.getUniformLocation(prog, 'uContrast'), p.contrast);
  gl.uniform1f(gl.getUniformLocation(prog, 'uVibrance'), p.vibrance);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  return canvas;
}

// Резервный путь на CPU — та же математика, если WebGL недоступен.
function applyCPU(bitmap, p) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const d = img.data;
  const [bR, bG, bB] = p.black, [wR, wG, wB] = p.white;

  // Таблицы уровней+гаммы на 256 значений — считаем один раз, не на каждый пиксель.
  const lut = [0, 1, 2].map(ch => {
    const b = [bR, bG, bB][ch], w = [wR, wG, wB][ch];
    const t = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
      let v = (i / 255 - b) / Math.max(1e-4, w - b);
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      t[i] = Math.pow(v, p.gamma);
    }
    return t;
  });

  for (let i = 0; i < d.length; i += 4) {
    let r = lut[0][d[i]], g = lut[1][d[i + 1]], b = lut[2][d[i + 2]];
    // S-кривая
    const sc = (v) => v * v * (3 - 2 * v);
    r = r + (sc(r) - r) * p.contrast;
    g = g + (sc(g) - g) * p.contrast;
    b = b + (sc(b) - b) * p.contrast;
    // vibrance
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx > 0 ? (mx - mn) / mx : 0;
    const k = 1 + p.vibrance * (p.vibrance > 0 ? (1 - sat) : 1);
    r = l + (r - l) * k; g = l + (g - l) * k; b = l + (b - l) * k;
    d[i]     = r <= 0 ? 0 : r >= 1 ? 255 : r * 255;
    d[i + 1] = g <= 0 ? 0 : g >= 1 ? 255 : g * 255;
    d[i + 2] = b <= 0 ? 0 : b >= 1 ? 255 : b * 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Значения для показа пользователю, в процентах.
function toDisplay(p) {
  const bAvg = (p.black[0] + p.black[1] + p.black[2]) / 3;
  const wAvg = (p.white[0] + p.white[1] + p.white[2]) / 3;
  const stretch = 1 / Math.max(1e-4, wAvg - bAvg);   // вклад автоуровней
  const q = p._quality;
  return {
    brightness: Math.round((Math.pow(0.5, p.gamma) / 0.5 - 1) * 100),
    contrast: Math.round(((stretch - 1) + p.contrast) * 100),
    saturation: Math.round(p.vibrance * 100),
    // объективная оценка качества до/после коррекции
    qBefore: q ? Math.round(q.before.score * 100) : null,
    qAfter: q ? Math.round(q.after.score * 100) : null,
    qGain: q ? Math.round(q.gain * 100) : null
  };
}

async function processTask(taskId, bitmap, format, quality) {
  const t = tasks.get(taskId);
  try {
    emit(taskId, 'processing', 5);
    await tfReady;
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 20);
    const preview = makePreview(bitmap);
    const analysis = model.analyze(preview);
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 45);
    const params = await model.predict(analysis, preview); // ML + контур качества
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 70);
    const canvas = applyWebGL(bitmap, params);      // применение коррекции
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 90);
    const mime = format === 'png' ? 'image/png' : 'image/jpeg';
    const blob = await canvas.convertToBlob({ type: mime, quality: quality ?? 0.92 });
    t.result = { blob, params, display: toDisplay(params) };
    bitmap.close?.();
    emit(taskId, 'done', 100);
  } catch (err) {
    t.error = String(err && err.message || err);
    emit(taskId, 'error', t.progress || 0);
  }
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'submit') {
    const { taskId, bitmap, format, quality } = msg;
    tasks.set(taskId, { status: 'queued', progress: 0, cancelled: false });
    emit(taskId, 'queued', 0);
    processTask(taskId, bitmap, format, quality);
  } else if (msg.type === 'status') {
    const t = tasks.get(msg.taskId);
    self.postMessage({ type: 'statusResult', taskId: msg.taskId,
      status: t?.status ?? 'unknown', progress: t?.progress ?? 0 });
  } else if (msg.type === 'cancel') {
    const t = tasks.get(msg.taskId);
    if (t && t.status !== 'done') { t.cancelled = true; t.status = 'cancelled'; }
    self.postMessage({ type: 'cancelResult', taskId: msg.taskId, success: !!t });
  } else if (msg.type === 'result') {
    const t = tasks.get(msg.taskId);
    if (t?.result) {
      self.postMessage({ type: 'resultData', taskId: msg.taskId,
        blob: t.result.blob, params: t.result.params, display: t.result.display });
    } else {
      self.postMessage({ type: 'resultData', taskId: msg.taskId,
        blob: null, error: t?.error ?? 'not ready' });
    }
  }
};
