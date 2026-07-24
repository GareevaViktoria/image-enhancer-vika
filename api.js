// api.js — публичный API модуля улучшения изображений.
// Реализует контракт из ТЗ:
//   submitTask(image)  -> Promise<taskId>
//   getStatus(taskId)  -> Promise<{status, progress}>
//   cancelTask(taskId) -> Promise<{success}>
//   getResult(taskId)  -> Promise<{blob, params}>
//   события: addEventListener('statusChange', ({taskId,status,progress}) => ...)
//
// Поддержка форматов JPG, PNG, HEIC, BMP. HEIC декодируется через heic2any
// (WASM libheif), остальные — нативным createImageBitmap.

export class ImageEnhancer extends EventTarget {
  constructor() {
    super();
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this._pending = new Map(); // для request/response методов
    this._seq = 0;

    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'statusChange') {
        this.dispatchEvent(new CustomEvent('statusChange', { detail: {
          taskId: m.taskId, status: m.status, progress: m.progress } }));
      } else if (m.type === 'statusResult') {
        this._resolve('status:' + m.taskId, { status: m.status, progress: m.progress });
      } else if (m.type === 'cancelResult') {
        this._resolve('cancel:' + m.taskId, { success: m.success });
      } else if (m.type === 'resultData') {
        this._resolve('result:' + m.taskId, m.blob
          ? { blob: m.blob, params: m.params }
          : Promise.reject(new Error(m.error || 'no result')));
      }
    };
  }

  _resolve(key, value) {
    const p = this._pending.get(key);
    if (p) { this._pending.delete(key); Promise.resolve(value).then(p.res, p.rej); }
  }
  _await(key) {
    return new Promise((res, rej) => this._pending.set(key, { res, rej }));
  }

  // Декодирование входного файла в ImageBitmap с учётом формата.
  async _decode(file) {
    const name = (file.name || '').toLowerCase();
    const isHeic = /\.hei[cf]$/.test(name) || file.type === 'image/heic' || file.type === 'image/heif';
    if (isHeic) {
      // heic2any подключается глобально из bundled-скрипта
      if (!self.heic2any) throw new Error('HEIC-декодер не загружен');
      const out = await self.heic2any({ blob: file, toType: 'image/png' });
      const blob = Array.isArray(out) ? out[0] : out;
      return await createImageBitmap(blob);
    }
    // JPG, PNG, BMP — нативно
    return await createImageBitmap(file);
  }

  // Метод постановки задачи. Принимает File/Blob, возвращает идентификатор.
  async submitTask(file, opts = {}) {
    const bitmap = await this._decode(file);
    const taskId = 't' + Date.now().toString(36) + (this._seq++);
    const outName = (file.name || '').toLowerCase();
    const format = /\.png$/.test(outName) ? 'png' : 'jpeg';
    // ImageBitmap передаётся через transferable — без копирования
    this.worker.postMessage(
      { type: 'submit', taskId, bitmap, format, quality: opts.quality ?? 0.92 },
      [bitmap]
    );
    return taskId;
  }

  // Метод получения статуса задачи.
  getStatus(taskId) {
    const key = 'status:' + taskId;
    const p = this._await(key);
    this.worker.postMessage({ type: 'status', taskId });
    return p;
  }

  // Метод прерывания задачи.
  cancelTask(taskId) {
    const key = 'cancel:' + taskId;
    const p = this._await(key);
    this.worker.postMessage({ type: 'cancel', taskId });
    return p;
  }

  // Метод получения готового изображения (Blob + применённые параметры).
  getResult(taskId) {
    const key = 'result:' + taskId;
    const p = this._await(key);
    this.worker.postMessage({ type: 'result', taskId });
    return p;
  }
}
