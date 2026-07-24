// app.js — связывает интерфейс с API-модулем ImageEnhancer.
import { ImageEnhancer } from './api.js';

const enhancer = new ImageEnhancer();

const $ = (id) => document.getElementById(id);
const drop = $('drop'), fileInput = $('file');
const prog = $('prog'), barFill = $('barFill'), pct = $('pct'), statusLabel = $('statusLabel');
const viewer = $('viewer'), imgBefore = $('imgBefore'), imgAfter = $('imgAfter');
const params = $('params'), pBright = $('pBright'), pContrast = $('pContrast'), pSat = $('pSat');
const actions = $('actions'), btnDownload = $('download'), btnAgain = $('again'), btnCancel = $('cancel');
const err = $('err'), compare = $('compare'), handle = $('handle');

let currentTask = null, resultBlob = null, srcName = 'image';

const STATUS_RU = {
  queued: 'В очереди', processing: 'Обработка', done: 'Готово',
  cancelled: 'Прервано', error: 'Ошибка', unknown: 'Неизвестно'
};

function setProgress(status, progress) {
  prog.classList.add('on');
  statusLabel.textContent = STATUS_RU[status] || status;
  pct.textContent = Math.round(progress) + '%';
  barFill.style.width = progress + '%';
}

function showError(msg) {
  err.textContent = 'Ошибка: ' + msg;
  err.classList.add('on');
}

// Событие изменения статуса задачи (по контракту ТЗ).
enhancer.addEventListener('statusChange', async (e) => {
  const { taskId, status, progress } = e.detail;
  if (taskId !== currentTask) return;
  setProgress(status, progress);

  if (status === 'processing') btnCancel.style.display = '';
  if (status === 'done') {
    btnCancel.style.display = 'none';
    const { blob, params: p } = await enhancer.getResult(taskId);
    resultBlob = blob;
    imgAfter.src = URL.createObjectURL(blob);
    showParams(p);
    viewer.classList.add('on');
    params.classList.add('on');
    actions.style.display = 'flex';
    setTimeout(() => prog.classList.remove('on'), 600);
  }
  if (status === 'error') {
    btnCancel.style.display = 'none';
    showError('не удалось обработать изображение');
  }
  if (status === 'cancelled') {
    btnCancel.style.display = 'none';
  }
});

function showParams(p) {
  const br = Math.round(p.brightness * 100);
  const co = Math.round((p.contrast - 1) * 100);
  const sa = Math.round((p.saturation - 1) * 100);
  const fmt = (v) => (v > 0 ? '+' : '') + v + '%';
  pBright.textContent = fmt(br);
  pContrast.textContent = fmt(co);
  pSat.textContent = fmt(sa);
  pBright.className = 'v' + (br >= 0 ? ' pos' : '');
  pContrast.className = 'v' + (co >= 0 ? ' pos' : '');
  pSat.className = 'v' + (sa >= 0 ? ' pos' : '');
}

async function handleFile(file) {
  if (!file) return;
  err.classList.remove('on');
  drop.style.display = 'none';
  viewer.classList.remove('on');
  params.classList.remove('on');
  actions.style.display = 'none';
  resultBlob = null;
  srcName = (file.name || 'image').replace(/\.[^.]+$/, '');

  imgBefore.src = URL.createObjectURL(file.type.includes('heic') ? file : file);
  // для HEIC превью оригинала браузер не покажет — заменим после декода result-ом,
  // но "до" всё равно нужно; используем after как fallback ниже при ошибке
  imgBefore.onerror = () => { imgBefore.onerror = null; };

  setProgress('queued', 0);
  try {
    currentTask = await enhancer.submitTask(file);
  } catch (ex) {
    drop.style.display = '';
    showError(ex.message || 'не удалось прочитать файл');
  }
}

// --- Drag & drop + выбор файла ---
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
['dragover', 'dragenter'].forEach(ev =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('hot'); }));
['dragleave', 'drop'].forEach(ev =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('hot'); }));
drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));

// --- Кнопки ---
btnDownload.addEventListener('click', () => {
  if (!resultBlob) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(resultBlob);
  a.download = srcName + '_enhanced.' + (resultBlob.type === 'image/png' ? 'png' : 'jpg');
  a.click();
});
btnAgain.addEventListener('click', () => {
  drop.style.display = '';
  viewer.classList.remove('on');
  params.classList.remove('on');
  prog.classList.remove('on');
  actions.style.display = 'none';
  fileInput.value = '';
});
btnCancel.addEventListener('click', async () => {
  if (currentTask) {
    const { success } = await enhancer.cancelTask(currentTask);
    if (success) setProgress('cancelled', 0);
  }
});

// --- Слайдер сравнения до/после ---
let dragging = false;
function setSplit(clientX) {
  const rect = compare.getBoundingClientRect();
  let x = (clientX - rect.left) / rect.width;
  x = Math.max(0, Math.min(1, x));
  const pctX = x * 100;
  compare.querySelector('.after').style.clipPath = `inset(0 0 0 ${pctX}%)`;
  handle.style.left = pctX + '%';
}
const startDrag = (e) => { dragging = true; move(e); };
const endDrag = () => { dragging = false; };
const move = (e) => {
  if (!dragging) return;
  const cx = e.touches ? e.touches[0].clientX : e.clientX;
  setSplit(cx);
};
handle.addEventListener('mousedown', startDrag);
compare.addEventListener('mousedown', startDrag);
window.addEventListener('mousemove', move);
window.addEventListener('mouseup', endDrag);
handle.addEventListener('touchstart', startDrag, { passive: true });
compare.addEventListener('touchstart', startDrag, { passive: true });
window.addEventListener('touchmove', move, { passive: true });
window.addEventListener('touchend', endDrag);
