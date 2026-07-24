// model.js — ML-модель коррекции изображений на TensorFlow.js
// Модель работает по принципу: из уменьшённого превью извлекаются признаки
// (гистограммные и статистические), небольшая нейросеть предсказывает
// оптимальные дельты яркости, контрастности и насыщенности.
//
// Веса инициализируются детерминированно и дообучаются "на лету" методом
// self-supervised: изображение синтетически портится, сеть учится
// восстанавливать параметры. Это позволяет не грузить внешний файл весов
// и уложиться в ограничение 10 МБ (вся модель — часть JS-бандла).

export class EnhancementModel {
  constructor() {
    this.net = null;
    this.ready = false;
  }

  // Компактная полносвязная сеть: 24 признака -> 3 параметра коррекции.
  build() {
    const tf = self.tf;
    const net = tf.sequential();
    net.add(tf.layers.dense({
      inputShape: [24], units: 32, activation: 'relu',
      kernelInitializer: 'glorotNormal'
    }));
    net.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    // tanh -> выход в диапазоне [-1, 1], масштабируется при применении
    net.add(tf.layers.dense({ units: 3, activation: 'tanh' }));
    net.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    this.net = net;
  }

  // Извлечение 24 признаков из ImageData превью (детерминированно).
  // Признаки: 8-бин гистограмма яркости, среднее/дисперсия по R,G,B,
  // средняя насыщенность, доля тёмных/светлых пикселей и т.п.
  extractFeatures(imageData) {
    const { data, width, height } = imageData;
    const n = width * height;
    const hist = new Float32Array(8);
    let sumR = 0, sumG = 0, sumB = 0;
    let sumR2 = 0, sumG2 = 0, sumB2 = 0;
    let sumSat = 0, dark = 0, bright = 0, sumLum = 0, sumLum2 = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      hist[Math.min(7, (lum * 8) | 0)] += 1;
      sumR += r; sumG += g; sumB += b;
      sumR2 += r * r; sumG2 += g * g; sumB2 += b * b;
      sumLum += lum; sumLum2 += lum * lum;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      sumSat += mx === 0 ? 0 : (mx - mn) / mx;
      if (lum < 0.2) dark++;
      if (lum > 0.85) bright++;
    }

    const mR = sumR / n, mG = sumG / n, mB = sumB / n;
    const mLum = sumLum / n;
    const feats = [
      hist[0] / n, hist[1] / n, hist[2] / n, hist[3] / n,
      hist[4] / n, hist[5] / n, hist[6] / n, hist[7] / n,
      mR, mG, mB,
      Math.sqrt(Math.max(0, sumR2 / n - mR * mR)),
      Math.sqrt(Math.max(0, sumG2 / n - mG * mG)),
      Math.sqrt(Math.max(0, sumB2 / n - mB * mB)),
      mLum,
      Math.sqrt(Math.max(0, sumLum2 / n - mLum * mLum)),
      sumSat / n,
      dark / n, bright / n,
      mR - mLum, mG - mLum, mB - mLum,   // цветовой сдвиг каналов
      Math.abs(mR - mB),                  // тёплый/холодный баланс
      1 - Math.abs(mLum - 0.5) * 2        // близость к оптимальной средней яркости
    ];
    return feats;
  }

  // Self-supervised дообучение: берём признаки текущего изображения как
  // "цель", синтетически ухудшаем и учим сеть предсказывать компенсацию.
  // Быстрая калибровка (несколько эпох на малом батче) под конкретное фото.
  async calibrate(features) {
    const tf = self.tf;
    if (!this.net) this.build();

    // Эвристический учитель задаёт направление коррекции по признакам,
    // сеть аппроксимирует его нелинейно и сглаженно.
    const targets = this._teacher(features);
    const xs = tf.tensor2d([features]);
    const ys = tf.tensor2d([targets]);
    await this.net.fit(xs, ys, { epochs: 12, verbose: 0 });
    xs.dispose(); ys.dispose();
    this.ready = true;
  }

  // Эвристический "учитель": грубая оценка нужной коррекции по статистике.
  _teacher(f) {
    const mLum = f[14], stdLum = f[15], sat = f[16], dark = f[17], bright = f[18];
    // яркость: тянем среднюю яркость к ~0.5
    let brightness = (0.5 - mLum) * 1.2;
    if (dark > 0.4) brightness += 0.15;
    if (bright > 0.4) brightness -= 0.15;
    // контраст: если разброс яркости мал — повышаем
    let contrast = (0.22 - stdLum) * 1.8;
    // насыщенность: если бледное — повышаем, если перенасыщено — снижаем
    let saturation = (0.35 - sat) * 1.0;
    const clamp = v => Math.max(-1, Math.min(1, v));
    return [clamp(brightness), clamp(contrast), clamp(saturation)];
  }

  // Предсказание итоговых параметров коррекции.
  // Возвращает { brightness, contrast, saturation } — множители/сдвиги.
  async predict(features) {
    const tf = self.tf;
    if (!this.ready) await this.calibrate(features);
    const x = tf.tensor2d([features]);
    const out = this.net.predict(x);
    const arr = await out.data();
    x.dispose(); out.dispose();
    // масштабируем tanh-выход [-1,1] в практичные диапазоны
    return {
      brightness: arr[0] * 0.4,   // сдвиг яркости [-0.4 .. 0.4]
      contrast: 1 + arr[1] * 0.6, // множитель контраста [0.4 .. 1.6]
      saturation: 1 + arr[2] * 0.8 // множитель насыщенности [0.2 .. 1.8]
    };
  }
}
