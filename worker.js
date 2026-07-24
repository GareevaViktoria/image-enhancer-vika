// worker.js — обработка изображений вне главного потока.
// Загружает TF.js, применяет ML-модель для подбора параметров и
// коррекцию через WebGL-шейдер (OffscreenCanvas). Шлёт события прогресса.

import { EnhancementModel } from './model.js';

// TF.js подключается из CDN внутри воркера (importScripts работает и в module-worker
// через self, но для надёжности грузим через динамический скрипт-текст нельзя —
// поэтому используем importScripts из classic-части ниже недоступен в module).
// Решение: грузим TF.js как ESM.
let tfReady = (async () => {
  const tf = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/+esm');
  self.tf = tf;
  try { await tf.setBackend('webgl'); } catch (e) { await tf.setBackend('cpu'); }
  await tf.ready();
})();

const model = new EnhancementModel();
const tasks = new Map(); // taskId -> { status, progress, cancelled, result }

function emit(taskId, status, progress) {
  const t = tasks.get(taskId);
  if (t) { t.status = status; t.progress = progress; }
  self.postMessage({ type: 'statusChange', taskId, status, progress });
}

// Уменьшенное превью для анализа моделью (макс. сторона 128 px).
function makePreview(bitmap) {
  const maxSide = 128;
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// --- WebGL коррекция: применяет параметры к полному изображению на GPU ---
const VERT = `
attribute vec2 aPos;
attribute vec2 aTex;
varying vec2 vTex;
void main(){ vTex = aTex; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
varying vec2 vTex;
uniform sampler2D uTex;
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;
void main(){
  vec4 c = texture2D(uTex, vTex);
  // яркость
  c.rgb += uBrightness;
  // контраст относительно средней точки 0.5
  c.rgb = (c.rgb - 0.5) * uContrast + 0.5;
  // насыщенность через смешивание с яркостной серой
  float l = dot(c.rgb, vec3(0.299, 0.587, 0.114));
  c.rgb = mix(vec3(l), c.rgb, uSaturation);
  gl_FragColor = vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function applyWebGL(bitmap, params) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const gl = canvas.getContext('webgl', { premultipliedAlpha: false });
  if (!gl) return applyCPU(bitmap, params); // fallback

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog); gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  // два треугольника на весь экран: pos.xy, tex.xy
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

  gl.uniform1f(gl.getUniformLocation(prog, 'uBrightness'), params.brightness);
  gl.uniform1f(gl.getUniformLocation(prog, 'uContrast'), params.contrast);
  gl.uniform1f(gl.getUniformLocation(prog, 'uSaturation'), params.saturation);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  return canvas;
}

// Fallback на CPU, если WebGL недоступен.
function applyCPU(bitmap, params) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const d = img.data;
  const { brightness: br, contrast: co, saturation: sa } = params;
  for (let i = 0; i < d.length; i += 4) {
    let r = d[i] / 255, g = d[i + 1] / 255, b = d[i + 2] / 255;
    r += br; g += br; b += br;
    r = (r - 0.5) * co + 0.5; g = (g - 0.5) * co + 0.5; b = (b - 0.5) * co + 0.5;
    const l = 0.299 * r + 0.587 * g + 0.114 * b;
    r = l + (r - l) * sa; g = l + (g - l) * sa; b = l + (b - l) * sa;
    d[i] = Math.max(0, Math.min(255, r * 255));
    d[i + 1] = Math.max(0, Math.min(255, g * 255));
    d[i + 2] = Math.max(0, Math.min(255, b * 255));
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

async function processTask(taskId, bitmap, format, quality) {
  const t = tasks.get(taskId);
  try {
    emit(taskId, 'processing', 5);
    await tfReady;
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 20);
    const preview = makePreview(bitmap);
    const feats = model.extractFeatures(preview);
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 45);
    const params = await model.predict(feats); // ML подбор параметров
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 70);
    const canvas = applyWebGL(bitmap, params); // применение коррекции
    if (t.cancelled) return emit(taskId, 'cancelled', t.progress);

    emit(taskId, 'processing', 90);
    const mime = format === 'png' ? 'image/png' : 'image/jpeg';
    const blob = await canvas.convertToBlob({ type: mime, quality: quality ?? 0.92 });
    t.result = { blob, params };
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
        blob: t.result.blob, params: t.result.params });
    } else {
      self.postMessage({ type: 'resultData', taskId: msg.taskId,
        blob: null, error: t?.error ?? 'not ready' });
    }
  }
};
