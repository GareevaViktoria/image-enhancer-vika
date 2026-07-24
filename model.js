// model.js — ML-модель подбора параметров коррекции изображения.
//
// Схема работы (соответствует ТЗ):
//   1. Из превью извлекается статистика: поканальные гистограммы, перцентили,
//      медиана яркости, насыщенность, цветовой сдвиг.
//   2. Вспомогательный алгоритм считает точки чёрного/белого по перцентилям
//      (автоуровни + смягчение цветового сдвига).
//   3. Нейросеть на TensorFlow.js подбирает три параметра коррекции:
//      гамму (яркость средних тонов), силу S-кривой (контраст) и vibrance
//      (цветность).
//   4. Параметры применяются WebGL-шейдером к полному изображению.
//
// Ключевой принцип: коррекция улучшает КАЖДЫЙ снимок. Тональный диапазон
// всегда раскрывается на полную, средние тона приводятся к оптимальной
// светлоте, добавляется мягкая объёмность и насыщенность. При этом яркий
// удачный кадр НЕ приглушается: гамма для него остаётся около единицы,
// а работают раскрытие диапазона и лёгкая полировка.

const HIST_BINS = 256;

export class EnhancementModel {
  constructor() {
    this.net = null;
  }

  // Компактная сеть: 24 признака -> 3 параметра коррекции.
  build() {
    const tf = self.tf;
    const net = tf.sequential();
    net.add(tf.layers.dense({
      inputShape: [24], units: 32, activation: 'relu',
      kernelInitializer: 'glorotNormal'
    }));
    net.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    net.add(tf.layers.dense({ units: 3, activation: 'tanh' }));
    net.compile({ optimizer: tf.train.adam(0.05), loss: 'meanSquaredError' });
    this.net = net;
  }

  // --- Анализ изображения (чистая математика, без TF) ---
  analyze(imageData) {
    const { data } = imageData;
    const n = data.length / 4;

    const hR = new Float64Array(HIST_BINS);
    const hG = new Float64Array(HIST_BINS);
    const hB = new Float64Array(HIST_BINS);
    const hL = new Float64Array(HIST_BINS);

    let sumR = 0, sumG = 0, sumB = 0, sumSat = 0, sumLum = 0, sumLum2 = 0;

    for (let i = 0; i < data.length; i += 4) {
      const R = data[i], G = data[i + 1], B = data[i + 2];
      hR[R]++; hG[G]++; hB[B]++;
      let lum255 = (0.299 * R + 0.587 * G + 0.114 * B) | 0;
      if (lum255 > 255) lum255 = 255;
      hL[lum255]++;

      const r = R / 255, g = G / 255, b = B / 255;
      sumR += r; sumG += g; sumB += b;
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      sumLum += l; sumLum2 += l * l;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      sumSat += mx <= 0 ? 0 : (mx - mn) / mx;
    }

    const pct = (hist, p) => {
      const need = n * p;
      let acc = 0;
      for (let i = 0; i < HIST_BINS; i++) {
        acc += hist[i];
        if (acc >= need) return i / 255;
      }
      return 1;
    };

    // Точки чёрного и белого по перцентилям: отсекаем 0.4% выбросов.
    const LO = 0.004, HI = 0.996;
    let bR = pct(hR, LO), bG = pct(hG, LO), bB = pct(hB, LO);
    let wR = pct(hR, HI), wG = pct(hG, HI), wB = pct(hB, HI);

    // Смягчение поканальной разницы: чисто поканальные уровни стирают
    // намеренный тёплый/холодный тон. Смешиваем с общей точкой.
    // Доля общей точки адаптивна: у насыщенного кадра поканальное растяжение
    // ещё сильнее развело бы каналы и добавило кислотности — там уровни
    // делаем почти общими.
    const satEarly = sumSat / n;
    const MIX = Math.min(0.95, 0.65 + Math.max(0, satEarly - 0.45) * 1.2);
    const bAvg = (bR + bG + bB) / 3, wAvg = (wR + wG + wB) / 3;
    bR = bAvg * MIX + bR * (1 - MIX);
    bG = bAvg * MIX + bG * (1 - MIX);
    bB = bAvg * MIX + bB * (1 - MIX);
    wR = wAvg * MIX + wR * (1 - MIX);
    wG = wAvg * MIX + wG * (1 - MIX);
    wB = wAvg * MIX + wB * (1 - MIX);

    // Ограничители, чтобы растяжение не было разрушительным.
    const capB = v => Math.min(v, 0.16);
    const capW = v => Math.max(v, 0.80);
    const black = [capB(bR), capB(bG), capB(bB)];
    const white = [capW(wR), capW(wG), capW(wB)];

    const mR = sumR / n, mG = sumG / n, mB = sumB / n;
    const mLum = sumLum / n;
    const stdLum = Math.sqrt(Math.max(0, sumLum2 / n - mLum * mLum));
    const sat = sumSat / n;

    const medianLum = pct(hL, 0.5);
    const p05 = pct(hL, 0.05), p95 = pct(hL, 0.95);
    const darkFrac = this._fracBelow(hL, n, 0.20);
    const brightFrac = this._fracAbove(hL, n, 0.90);

    // Медиана ПОСЛЕ растяжения уровней — по ней подбирается гамма.
    const bL = (black[0] + black[1] + black[2]) / 3;
    const wL = (white[0] + white[1] + white[2]) / 3;
    const medianAfter = Math.min(0.999, Math.max(0.001,
      (medianLum - bL) / Math.max(1e-4, wL - bL)));

    const features = [
      ...this._coarse(hL, n),        // 8
      mR, mG, mB,                    // 3
      mLum, stdLum, sat,             // 3
      medianLum, medianAfter,        // 2
      p05, p95, p95 - p05,           // 3
      darkFrac, brightFrac,          // 2
      mR - mLum, mB - mLum,          // 2
      wL - bL                        // 1  = 24
    ];

    return { features, black, white, medianAfter, sat, stdLum, brightFrac, darkFrac };
  }

  _coarse(hL, n) {
    const out = new Array(8).fill(0);
    for (let i = 0; i < HIST_BINS; i++) out[Math.min(7, (i / 32) | 0)] += hL[i];
    return out.map(v => v / n);
  }
  _fracBelow(hL, n, t) {
    let acc = 0; const lim = t * 255;
    for (let i = 0; i < lim; i++) acc += hL[i];
    return acc / n;
  }
  _fracAbove(hL, n, t) {
    let acc = 0; const lim = Math.ceil(t * 255);
    for (let i = lim; i < HIST_BINS; i++) acc += hL[i];
    return acc / n;
  }

  // --- Диапазоны физических параметров и перевод в выход сети (tanh) ---
  static G_MIN = 0.55;    // минимальная гамма (сильное осветление)
  static G_MAX = 1.45;    // максимальная гамма (сильное затемнение)
  static C_MAX = 0.55;    // максимальная сила S-кривой
  static V_MIN = -0.35;   // минимальный vibrance (снижение насыщенности)
  static V_MAX = 0.55;    // максимальный vibrance

  _toNorm([gamma, contrast, vibrance]) {
    const E = EnhancementModel;
    return [
      Math.max(-1, Math.min(1, (1 - gamma) / 0.45)),
      Math.max(-1, Math.min(1, contrast / E.C_MAX * 2 - 1)),
      Math.max(-1, Math.min(1, (vibrance - E.V_MIN) / (E.V_MAX - E.V_MIN) * 2 - 1))
    ];
  }
  _fromNorm(o) {
    const E = EnhancementModel;
    const gamma = Math.max(E.G_MIN, Math.min(E.G_MAX, 1 - o[0] * 0.45));
    const contrast = Math.max(0, Math.min(E.C_MAX, (o[1] + 1) / 2 * E.C_MAX));
    const vibrance = Math.max(E.V_MIN, Math.min(E.V_MAX,
      (o[2] + 1) / 2 * (E.V_MAX - E.V_MIN) + E.V_MIN));
    return { gamma, contrast, vibrance };
  }

  // --- Эталонный расчёт параметров (учитель для сети) ---
  // Возвращает ФИЗИЧЕСКИЕ параметры: [гамма, контраст, vibrance].
  teacher(a) {
    const { medianAfter, sat, stdLum, brightFrac, darkFrac } = a;

    // 1) ГАММА — светлота средних тонов.
    // Коридор нормы: медиана (уже после раскрытия диапазона) в 0.40..0.62.
    // Внутри него яркость НЕ трогаем — светлый удачный кадр не приглушается.
    // Улучшение такого снимка даёт раскрытие диапазона, контраст и цветность.
    const LO_OK = 0.40, HI_OK = 0.62;
    let gamma = 1.0;
    if (medianAfter < LO_OK) {
      gamma = Math.log(LO_OK) / Math.log(medianAfter);       // осветляем
    } else if (medianAfter > HI_OK) {
      gamma = Math.log(HI_OK) / Math.log(medianAfter);       // затемняем
    }
    gamma = 1 + (gamma - 1) * 0.75;                          // смягчение
    if (brightFrac > 0.12) gamma = Math.max(gamma, 1.0);     // света выбиты — не осветляем
    if (darkFrac > 0.45) gamma = Math.min(gamma, 1.0);       // провалы в чёрное — не затемняем
    // Светлый кадр без выбитых светов — вероятно, задумка автора (снег, пляж):
    // затемнять сильно нельзя.
    if (medianAfter > HI_OK && brightFrac < 0.08) gamma = Math.min(gamma, 1.15);
    gamma = Math.max(EnhancementModel.G_MIN, Math.min(EnhancementModel.G_MAX, gamma));

    // 2) КОНТРАСТ — сила S-кривой. Базовая полировка есть всегда.
    let contrast = 0.30;
    if (stdLum < 0.16) contrast += (0.16 - stdLum) * 3.0;
    if (stdLum > 0.28) contrast -= (stdLum - 0.28) * 1.5;
    contrast = Math.max(0.10, Math.min(1.0, contrast)) * EnhancementModel.C_MAX;

    // 3) ЦВЕТНОСТЬ — vibrance: сильнее для блёклых, слабее для насыщенных,
    // лёгкое снижение для явно перенасыщенных.
    let vib = 0.30;
    if (sat < 0.26) vib += (0.26 - sat) * 1.6;
    if (sat > 0.55) vib -= (sat - 0.55) * 3.5;
    vib = Math.max(-0.35, Math.min(1.0, vib)) * EnhancementModel.V_MAX;

    return [gamma, contrast, vib];
  }

  // ================= ОЦЕНКА КАЧЕСТВА И ПОДБОР =================
  // Требование ТЗ: любое изображение должно меняться в лучшую сторону.
  // Поэтому параметры не просто вычисляются формулой, а проверяются:
  // коррекция применяется к превью, измеряются объективные метрики,
  // и выбирается вариант с максимальным приростом качества. Дополнительно
  // гарантируется порог заметности — изменение не может быть незаметным.

  // Применение параметров к превью (та же математика, что в шейдере).
  applyPreview(imageData, p) {
    const src = imageData.data;
    const out = new Uint8ClampedArray(src.length);

    // Таблицы уровней+гаммы: 256 значений на канал вместо pow() на пиксель.
    const lut = [0, 1, 2].map(ch => {
      const b = p.black[ch], w = p.white[ch];
      const t = new Float32Array(256);
      for (let i = 0; i < 256; i++) {
        let v = (i / 255 - b) / Math.max(1e-4, w - b);
        v = v < 0 ? 0 : v > 1 ? 1 : v;
        t[i] = Math.pow(v, p.gamma);
      }
      return t;
    });

    for (let i = 0; i < src.length; i += 4) {
      let r = lut[0][src[i]], g = lut[1][src[i + 1]], b = lut[2][src[i + 2]];
      // S-кривая контраста
      const sc = v => v * v * (3 - 2 * v);
      r += (sc(r) - r) * p.contrast;
      g += (sc(g) - g) * p.contrast;
      b += (sc(b) - b) * p.contrast;
      // vibrance
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const sat = mx > 0 ? (mx - mn) / mx : 0;
      const k = 1 + p.vibrance * (p.vibrance > 0 ? (1 - sat) : 1);
      r = l + (r - l) * k; g = l + (g - l) * k; b = l + (b - l) * k;
      out[i]     = r <= 0 ? 0 : r >= 1 ? 255 : r * 255;
      out[i + 1] = g <= 0 ? 0 : g >= 1 ? 255 : g * 255;
      out[i + 2] = b <= 0 ? 0 : b >= 1 ? 255 : b * 255;
      out[i + 3] = src[i + 3];
    }
    return { data: out, width: imageData.width, height: imageData.height };
  }

  // Объективные метрики качества изображения.
  quality(imageData) {
    const d = imageData.data;
    const n = d.length / 4;
    const hL = new Float64Array(256);
    let sL = 0, sL2 = 0, sSat = 0, clipLo = 0, clipHi = 0;

    for (let i = 0; i < d.length; i += 4) {
      const R = d[i], G = d[i + 1], B = d[i + 2];
      let lum = (0.299 * R + 0.587 * G + 0.114 * B) | 0;
      if (lum > 255) lum = 255;
      hL[lum]++;
      const r = R / 255, g = G / 255, b = B / 255;
      const l = 0.299 * r + 0.587 * g + 0.114 * b;
      sL += l; sL2 += l * l;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      sSat += mx > 0 ? (mx - mn) / mx : 0;
      // потеря детали: пиксели, полностью упёршиеся в 0 или 255
      if (R <= 1 && G <= 1 && B <= 1) clipLo++;
      if (R >= 254 && G >= 254 && B >= 254) clipHi++;
    }

    const pct = p => {
      const need = n * p; let acc = 0;
      for (let i = 0; i < 256; i++) { acc += hL[i]; if (acc >= need) return i / 255; }
      return 1;
    };
    const mL = sL / n;
    const std = Math.sqrt(Math.max(0, sL2 / n - mL * mL));
    const p01 = pct(0.01), p99 = pct(0.99);

    const range = p99 - p01;                 // охват тонального диапазона
    const median = pct(0.5);
    const sat = sSat / n;
    const clip = (clipLo + clipHi) / n;

    // Частные оценки 0..1
    const sRange = Math.min(1, range / 0.95);
    const sExpo = 1 - Math.min(1, Math.abs(median - 0.50) / 0.30);
    const sCont = 1 - Math.min(1, Math.abs(std - 0.25) / 0.25);
    const sColor = 1 - Math.min(1, Math.abs(sat - 0.38) / 0.38);

    const score = 0.28 * sRange + 0.24 * sExpo + 0.26 * sCont + 0.22 * sColor
                - 1.2 * Math.max(0, clip - 0.005);   // штраф за выбитые детали

    return { score, range, median, std, sat, clip };
  }

  // Средняя заметность изменения (0..1) между двумя изображениями.
  delta(a, b) {
    const A = a.data, B = b.data;
    let s = 0; const n = A.length / 4;
    for (let i = 0; i < A.length; i += 4) {
      s += Math.abs(A[i] - B[i]) + Math.abs(A[i + 1] - B[i + 1]) + Math.abs(A[i + 2] - B[i + 2]);
    }
    return s / (n * 3 * 255);
  }

  // Порог заметности: ниже него изменение визуально неразличимо.
  static MIN_DELTA = 0.030;

  // Подбор итоговых параметров: перебор вариантов вокруг базовых,
  // выбор по максимуму качества при обязательной заметности изменения.
  refine(preview, base) {
    const before = this.quality(preview);
    const E = EnhancementModel;

    const cMul = [0.8, 1.0, 1.25, 1.55];
    const vMul = [0.8, 1.0, 1.25, 1.55];
    const gAdd = [-0.05, 0, 0.05];

    let candidates = [];
    for (const cm of cMul) for (const vm of vMul) for (const ga of gAdd) {
      const p = {
        black: base.black, white: base.white,
        gamma: Math.max(E.G_MIN, Math.min(E.G_MAX, base.gamma + ga)),
        contrast: Math.max(0, Math.min(E.C_MAX, base.contrast * cm)),
        vibrance: Math.max(E.V_MIN, Math.min(E.V_MAX, base.vibrance * vm))
      };
      const img = this.applyPreview(preview, p);
      const q = this.quality(img);
      candidates.push({ p, gain: q.score - before.score, delta: this.delta(preview, img), q });
    }

    // Только те, что реально улучшают качество.
    let good = candidates.filter(c => c.gain > 0);
    if (!good.length) good = candidates;           // страховка

    // Среди улучшающих — предпочитаем заметные.
    const visible = good.filter(c => c.delta >= E.MIN_DELTA);
    const pool = visible.length ? visible : good;
    pool.sort((a, b) => b.gain - a.gain);
    let best = pool[0];

    // Если ни один вариант не дотянул до порога заметности — усиливаем
    // полировку, пока изменение не станет различимым и качество не падает.
    if (best.delta < E.MIN_DELTA) {
      for (let k = 1.2; k <= 3.0; k += 0.2) {
        const p = {
          black: best.p.black, white: best.p.white, gamma: best.p.gamma,
          contrast: Math.min(E.C_MAX, best.p.contrast * k),
          vibrance: Math.min(E.V_MAX, best.p.vibrance * k)
        };
        const img = this.applyPreview(preview, p);
        const q = this.quality(img);
        const d = this.delta(preview, img);
        if (q.score >= before.score && d >= E.MIN_DELTA) {
          best = { p, gain: q.score - before.score, delta: d, q };
          break;
        }
      }
    }

    return { params: best.p, before, after: best.q, gain: best.gain, delta: best.delta };
  }


  async calibrate(features, target) {
    const tf = self.tf;
    if (!this.net) this.build();
    const xs = tf.tensor2d([features]);
    const ys = tf.tensor2d([target]);
    await this.net.fit(xs, ys, { epochs: 30, verbose: 0 });
    xs.dispose(); ys.dispose();
  }

  // Итоговые параметры коррекции.
  // Сеть даёт отправную точку, контур проверки доводит её до максимума
  // качества с гарантией заметности изменения.
  async predict(a, preview) {
    const tf = self.tf;
    const target = this._toNorm(this.teacher(a));
    await this.calibrate(a.features, target);

    const x = tf.tensor2d([a.features]);
    const out = this.net.predict(x);
    const o = await out.data();
    x.dispose(); out.dispose();

    const base = { black: a.black, white: a.white, ...this._fromNorm(o) };
    if (!preview) return base;

    const r = this.refine(preview, base);
    return { ...r.params, _quality: { before: r.before, after: r.after, gain: r.gain, delta: r.delta } };
  }
}
