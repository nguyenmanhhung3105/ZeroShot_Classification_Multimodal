'use strict';
const $ = id => document.getElementById(id);
const pages = {overview: 'Bài toán', setup: 'Cấu hình suy luận', results: 'Kết quả suy luận', experiments: 'Kết quả thực nghiệm', simulations: 'Mô phỏng phương pháp', history: 'Lịch sử'};
const statuses = {queued: 'Chờ worker', running: 'Đang xử lý', complete: 'Hoàn tất', failed: 'Lỗi', cancelled: 'Đã hủy'};
const modalities = {image: 'Ảnh', title: 'Tiêu đề', image_title: 'Ảnh + tiêu đề'};
const families = {
  zero: {title: 'Zero-shot', description: 'Phân loại bằng mô tả lớp, không huấn luyện bộ phân loại bằng nhãn train.'},
  adaptation: {title: 'Thích nghi không nhãn', description: 'Sử dụng nhóm mẫu không nhãn để điều chỉnh dự đoán; cần có nhóm tham chiếu khi demo.'},
  fewshot: {title: 'Few-shot', description: 'Sử dụng bộ phân loại đã huấn luyện từ tập hỗ trợ có nhãn, tương ứng với backbone đã chọn.'},
  control: {title: 'Đối chứng bổ sung', description: 'Các biến thể dùng để đối chiếu thêm với phương pháp chính.'},
};
const selectedMethods = new Set(['clip_image', 'clip_image_title', 'dclip', 'waffle', 'fewshot_image', 'fewshot_image_title', 'fewshot_flat', 'fewshot_cascade']);
let catalog, activeJob = null, file = null, previewURL = null, polling = false, resultFilter = 'all';
let lastJobSnapshot = '', lastHistorySnapshot = '';
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
function familyOf(method) {
  if (method.startsWith('fewshot_')) return 'fewshot';
  if (method.startsWith('legacy_') || ['waffle_unit', 'calip', 'majority', 'em_image', 'flat_majority', 'clip_stata_image', 'hierarchy_em', 'hierarchy_majority'].includes(method)) return 'control';
  if (['majority', 'em_image', 'flat_stata', 'flat_majority', 'stata_image', 'clip_stata_image', 'clip_stata_image_title', 'hierarchy', 'hierarchy_em', 'hierarchy_majority'].includes(method)) return 'adaptation';
  return 'zero';
}
function openView(name, focus = true) {
  if (!(name in pages)) name = 'overview';
  for (const key of Object.keys(pages)) $('view-' + key).hidden = key !== name;
  document.querySelectorAll('[data-page]').forEach(link => {
    if (link.dataset.page === name) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  $('current-section').textContent = pages[name];
  document.title = `${pages[name]} · VLM Classifier`;
  if (name === 'experiments') window.ExperimentCharts?.load();
  if (name === 'simulations') window.MethodSimulations?.init();
  if (focus) {
    $('view-' + name).querySelector('h1').focus({preventScroll: true});
    window.scrollTo({top: 0, behavior: 'auto'});
  }
}
function navigate(name) {
  if (location.hash !== '#view-' + name) location.hash = 'view-' + name;
  else openView(name);
}
async function api(path, body) {
  const options = {cache: 'no-store', signal: AbortSignal.timeout(8000)};
  if (body !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Demo-Client': 'localhost'}, body: JSON.stringify(body)});
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function error(message, global = false) {
  const area = $(global ? 'app-error' : 'form-error');
  area.textContent = message;
  area.hidden = !message;
}
function applicable(method) {
  if (method.id === 'clip_stata_image_title') return false;
  if (method.modality === 'title') return false;
  if ($('pipeline').value !== 'all' && method.pipeline !== $('pipeline').value) return false;
  if ($('task').value === '2way') return method.pipeline !== 'cascade' && !['fewshot_flat', 'flat_stata', 'flat_majority'].includes(method.id);
  return !['majority', 'stata_image', 'clip_stata_image', 'clip_stata_image_title', 'fewshot_image', 'fewshot_title', 'fewshot_image_title'].includes(method.id);
}
function renderMethods() {
  const advancedOpen = $('methods').querySelector('details')?.open || false;
  $('methods').replaceChildren();
  for (const [key, family] of Object.entries(families)) {
    const methods = catalog.methods.filter(method => applicable(method) && familyOf(method.id) === key);
    if (!methods.length) continue;
    const group = el(key === 'control' ? 'details' : 'section', undefined, `method-group ${key}`);
    if (key === 'control') {
      group.open = advancedOpen;
      group.append(el('summary', `${family.title} · ${methods.length} cấu hình`));
    } else {
      const heading = el('div', undefined, 'method-group-header');
      heading.append(el('h3', family.title), el('p', family.description));
      group.append(heading);
    }
    const choices = el('div', undefined, 'choice-list method-options');
    for (const method of methods) {
      const label = el('label', undefined, 'method-option'), input = el('input');
      input.type = 'checkbox'; input.value = method.id; input.checked = selectedMethods.has(method.id);
      const copy = el('span');
      copy.append(el('strong', method.name), el('small', `${modalities[method.modality]} · ${method.pipeline === 'cascade' ? 'Hai tầng' : 'Một tầng'}`));
      input.addEventListener('change', () => {
        input.checked ? selectedMethods.add(method.id) : selectedMethods.delete(method.id);
        count();
      });
      label.append(input, copy); choices.append(label);
    }
    group.append(choices); $('methods').append(group);
  }
  count();
}
function count() {
  const total = $('methods').querySelectorAll(':checked').length;
  $('method-count').textContent = `· ${total} đã chọn`;
  $('selection-count').textContent = total;
  $('backbone-count').textContent = $('backbones').querySelectorAll(':checked').length;
}
function setFile(next) {
  if (!next) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type) || next.size > 8 * 1024 * 1024) {
    file = null;
    $('image').value = '';
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = null;
    $('preview').removeAttribute('src'); $('preview').hidden = true;
    $('upload-prompt').hidden = false; $('file-name').textContent = 'Chưa chọn tệp';
    error('Định dạng ảnh: JPEG, PNG hoặc WebP; dung lượng tối đa 8 MB.'); return;
  }
  file = next;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $('preview').src = previewURL; $('preview').hidden = false; $('upload-prompt').hidden = true;
  $('file-name').textContent = file.name; error('');
}
function base64File(value) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = () => reject(new Error('Không đọc được tệp ảnh.'));
    reader.readAsDataURL(value);
  });
}
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
function seconds(value) {
  if (!Number.isFinite(value)) return '—';
  return value > 0 && value < .01 ? '< 0,01 s' : `${value.toFixed(2)} s`;
}
function details(row) {
  const area = $('details-body');
  area.replaceChildren(el('h3', row.name), el('p', `${catalog.backbones[row.backbone] || row.backbone} · ${row.pipeline === 'cascade' ? 'Hai tầng' : 'Một tầng'} · ${modalities[row.modality]}`, 'detail-meta'));
  if (row.message) area.append(el('p', row.message, 'detail-notice'));
  if (row.note) area.append(el('p', row.note, 'detail-notice'));
  if (row.predicted) area.append(el('p', `Nhãn dự đoán: ${row.predicted}`));
  if (row.route) area.append(el('p', `Định tuyến: ${row.route}`, 'detail-meta'));
  if (row.gate) area.append(el('p', `Tầng Real/Fake: Real: ${percent(row.gate.Real)} · Fake: ${percent(row.gate.Fake)} · Ngưỡng: ${row.threshold ?? 'chọn điểm cao nhất; nếu bằng nhau chọn Real'}`, 'detail-meta'));
  const scores = Object.entries(row.scores || {});
  if (scores.length) area.append(el('h3', 'Phân phối điểm theo lớp', 'detail-label'));
  for (const [name, score] of scores) {
    const line = el('div', undefined, 'score-row'), meter = el('meter');
    meter.min = 0; meter.max = 1; meter.value = score; meter.setAttribute('aria-label', name);
    line.append(el('span', name), meter, el('span', percent(score))); area.append(line);
  }
  if (row.variants?.length) {
    area.append(el('h3', 'Cấu hình lấy mẫu và seed', 'detail-label'));
    const wrap = el('div', undefined, 'variant-table-wrap'), table = el('table'), head = el('thead'), header = el('tr'), body = el('tbody');
    for (const name of ['Policy', 'Seed', 'Dự đoán', 'Điểm', 'Định tuyến']) { const th = el('th', name); th.scope = 'col'; header.append(th); }
    head.append(header);
    for (const variant of row.variants) {
      const tr = el('tr');
      for (const value of [variant.policy, variant.seed, variant.predicted, percent(variant.score), variant.route || '—']) tr.append(el('td', value));
      body.append(tr);
    }
    table.append(head, body); wrap.append(table); area.append(wrap);
  }
  $('details').showModal();
}
function renderRows() {
  $('results').replaceChildren();
  const all = activeJob?.rows || [];
  const shown = all.filter(row => resultFilter === 'all' || familyOf(row.method) === resultFilter);
  $('row-count').textContent = `${shown.length} / ${all.length} kết quả`;
  $('table-wrap').hidden = !shown.length;
  $('filtered-empty').hidden = !all.length || !!shown.length;
  for (const [key, family] of Object.entries(families)) {
    const rows = shown.filter(row => familyOf(row.method) === key);
    if (!rows.length) continue;
    const groupRow = el('tr', undefined, `family-row ${key}`), groupHeading = el('th', family.title);
    groupHeading.colSpan = 6; groupHeading.scope = 'rowgroup';
    groupHeading.append(el('span', `${rows.length} kết quả`)); groupRow.append(groupHeading); $('results').append(groupRow);
    for (const row of rows) {
      const tr = el('tr'), method = el('td');
      method.append(el('strong', row.name), el('small', `${catalog.backbones[row.backbone] || row.backbone} · ${row.pipeline === 'cascade' ? 'Hai tầng' : 'Một tầng'}`));
      const variant = [];
      if (row.policy) variant.push(row.policy);
      if (Number.isInteger(row.seed)) variant.push(`seed ${row.seed}`);
      if (variant.length) method.append(el('small', variant.join(' · ')));
      tr.append(method, el('td', modalities[row.modality] || '—'));
      if (row.status === 'complete') {
        const prediction = el('td'); prediction.append(el('span', row.predicted, `prediction ${row.predicted === 'Real' ? 'real' : row.predicted === 'Fake' ? 'fake' : ''}`));
        tr.append(prediction, el('td', percent(row.score), 'score'));
      } else {
        const reason = el('td', `${row.status === 'failed' ? 'Lỗi thực thi' : 'Không thực thi'}: ${row.message || 'Không có thông tin'}`, `reason ${row.status === 'failed' ? 'failed' : ''}`);
        reason.colSpan = 2; tr.append(reason);
      }
      tr.append(el('td', seconds(row.seconds), 'score'));
      const cell = el('td'), button = el('button', 'Chi tiết', 'detail-button');
      button.type = 'button'; button.setAttribute('aria-label', `Chi tiết ${row.name} ${row.backbone}`);
      button.onclick = () => details(row); cell.append(button); tr.append(cell); $('results').append(tr);
    }
  }
}
function render(job, force = false) {
  activeJob = job;
  const snapshot = JSON.stringify(job);
  if (snapshot === lastJobSnapshot && !force) return;
  lastJobSnapshot = snapshot;
  const rows = job.rows || [], pending = ['queued', 'running'].includes(job.status);
  $('result-context').hidden = false;
  $('result-subtitle').textContent = `${job.task === '2way' ? 'Nhị phân · Real / Fake' : 'Sáu lớp Fakeddit'} — ${job.title}`;
  $('job-status').textContent = statuses[job.status] || job.status;
  $('job-status').className = `chip ${job.status === 'complete' ? 'success' : job.status === 'failed' ? 'error' : 'neutral'}`;
  $('empty').hidden = true; $('job-progress').hidden = false; $('cancel').hidden = !pending;
  $('job-message').textContent = `${statuses[job.status] || job.status} · ${job.message || ''}`;
  const expected = job.expected || rows.length || 1;
  $('progress').value = pending ? Math.min(95, rows.length / expected * 100) : 100;
  $('completed-count').textContent = `${rows.filter(row => row.status === 'complete').length} / ${job.expected || rows.length || '—'}`;
  $('unavailable-count').textContent = rows.filter(row => row.status !== 'complete').length;
  $('total-time').textContent = seconds(job.elapsed_seconds);
  const encodings = Object.values(job.encoding_seconds || {}).filter(Number.isFinite);
  $('encoding-time').textContent = encodings.length ? seconds(encodings.reduce((sum, value) => sum + value, 0)) : '—';
  $('download').disabled = false;
  renderRows();
}
async function refreshHistory() {
  const jobs = await api('/api/jobs'), snapshot = JSON.stringify(jobs);
  if (lastHistorySnapshot === snapshot) return;
  lastHistorySnapshot = snapshot; $('history').replaceChildren();
  if (!jobs.length) $('history').append(el('p', 'Chưa có yêu cầu.', 'empty-text'));
  for (const job of jobs) {
    const button = el('button', undefined, 'history-item'), text = el('span');
    text.append(el('strong', job.title), el('small', `${job.task === '2way' ? 'Nhị phân' : 'Sáu lớp'} · ${(job.rows || []).length} kết quả`));
    const time = el('time', new Date(job.created * 1000).toLocaleTimeString('vi-VN'));
    time.dateTime = new Date(job.created * 1000).toISOString();
    button.append(text, el('span', statuses[job.status] || job.status, 'chip neutral'), time);
    button.onclick = () => { render(job); navigate('results'); };
    $('history').append(button);
  }
}
function healthState(health) {
  const label = health.online ? ({ready: 'Worker sẵn sàng', loading: 'Worker đang nạp model', busy: 'Worker đang xử lý'}[health.state] || 'Worker đã kết nối') : 'Worker chưa kết nối';
  $('local-state').textContent = label;
  $('local-state').className = 'local-state connected';
  $('gpu-state').textContent = health.online ? ({ready: 'GPU sẵn sàng', loading: 'Đang nạp model', busy: 'GPU đang xử lý'}[health.state] || 'Đã kết nối worker') : 'Chưa kết nối GPU';
  $('gpu-state').closest('.runtime').classList.toggle('online', health.online);
  $('gpu-detail').textContent = health.online ? (health.message || 'Đã nhận tín hiệu từ worker') : 'Chưa nhận được tín hiệu gần đây từ worker';
}
async function tick() {
  if (polling) return;
  polling = true;
  try {
    try { healthState(await api('/api/health')); }
    catch {
      $('local-state').textContent = 'API local không phản hồi'; $('local-state').className = 'local-state disconnected';
      $('gpu-state').textContent = 'Chưa xác định trạng thái worker';
      $('gpu-detail').textContent = 'Không truy cập được API local';
      $('gpu-state').closest('.runtime').classList.remove('online');
      return;
    }
    try {
      if (activeJob && ['queued', 'running'].includes(activeJob.status)) {
        const id = activeJob.id, latest = await api(`/api/jobs/${id}`);
        if (activeJob?.id === id) render(latest);
      }
      await refreshHistory(); error('', true);
    } catch (exc) { error(`Không cập nhật được dữ liệu phiên: ${exc.message}`, true); }
  } finally { polling = false; }
}
function bindInterface() {
  // Prevent native form submission if catalog loading fails.
  $('classify-form').onsubmit = event => event.preventDefault();
  window.addEventListener('hashchange', () => {
    if (location.hash.startsWith('#view-')) openView(location.hash.replace('#view-', ''));
  });
  openView(location.hash.replace('#view-', ''), false);
  document.querySelectorAll('.result-filters [data-family]').forEach(button => {
    button.onclick = () => {
      resultFilter = button.dataset.family;
      document.querySelectorAll('.result-filters [data-family]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      renderRows();
    };
  });
  $('connection-open').onclick = () => $('connection').showModal();
  $('connection-close').onclick = () => $('connection').close();
  $('connection').addEventListener('close', () => { $('connection-token').value = ''; $('token-box').hidden = true; $('connection-message').textContent = ''; });
  $('details-close').onclick = () => $('details').close();
  $('show-token').onclick = async () => {
    try {
      const info = await api('/api/connection', {});
      $('connection-token').value = info.token;
      $('gateway-address').textContent = `http://127.0.0.1:${info.worker_port}`;
      $('token-box').hidden = false;
    } catch (exc) { $('connection-message').textContent = exc.message; }
  };
  $('copy-token').onclick = async () => {
    try { await navigator.clipboard.writeText($('connection-token').value); $('connection-message').textContent = 'Đã sao chép token phiên.'; }
    catch { $('connection-message').textContent = 'Không truy cập được clipboard.'; }
  };
}
async function init() {
  bindInterface();
  catalog = await api('/api/catalog');
  for (const [key, name] of Object.entries(catalog.backbones)) {
    const label = el('label'), input = el('input');
    input.type = 'checkbox'; input.value = key; input.checked = key === 'metaclip2'; input.onchange = count;
    label.append(input, document.createTextNode(name)); $('backbones').append(label);
  }
  renderMethods();
  $('task').onchange = () => {
    if ($('task').value === '2way' && $('pipeline').value === 'cascade') $('pipeline').value = 'single';
    renderMethods();
  };
  $('pipeline').onchange = () => { if ($('pipeline').value === 'cascade') $('task').value = '6way'; renderMethods(); };
  $('select-all').onclick = () => { catalog.methods.filter(applicable).forEach(method => selectedMethods.add(method.id)); renderMethods(); };
  $('select-none').onclick = () => { catalog.methods.filter(applicable).forEach(method => selectedMethods.delete(method.id)); renderMethods(); };
  $('image').onchange = event => setFile(event.target.files[0]);
  $('image-change').onclick = () => $('image').click();
  const zone = $('upload-zone');
  zone.ondragover = event => { event.preventDefault(); zone.classList.add('dragover'); };
  zone.ondragleave = () => zone.classList.remove('dragover');
  zone.ondrop = event => {
    event.preventDefault(); zone.classList.remove('dragover');
    if (event.dataTransfer.files.length) { $('image').files = event.dataTransfer.files; setFile(event.dataTransfer.files[0]); }
  };
  $('classify-form').onsubmit = async event => {
    event.preventDefault(); error('');
    const methods = [...$('methods').querySelectorAll(':checked')].map(input => input.value);
    const backbones = [...$('backbones').querySelectorAll(':checked')].map(input => input.value);
    if (!file || !methods.length || !backbones.length) { error('Chọn ảnh, ít nhất một phương pháp và một backbone để chạy phân loại.'); return; }
    $('run').disabled = true;
    try {
      const job = await api('/api/jobs', {image: await base64File(file), title: $('title').value, task: $('task').value, pipeline: $('pipeline').value, methods, backbones});
      render(job); navigate('results'); await tick();
    } catch (exc) { error(exc.message); }
    finally { $('run').disabled = false; }
  };
  $('cancel').onclick = async () => {
    const id = activeJob?.id; if (!id) return;
    try { const job = await api(`/api/jobs/${id}/cancel`, {}); if (activeJob?.id === id) render(job); await refreshHistory(); }
    catch (exc) { error(exc.message, true); }
  };
  $('download').onclick = () => {
    if (!activeJob) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(activeJob, null, 2)], {type: 'application/json'})), anchor = el('a');
    anchor.href = url; anchor.download = `vlm-demo-${activeJob.id}.json`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  await tick(); setInterval(tick, 2000);
}
init().catch(exc => { error(`Không khởi tạo được giao diện: ${exc.message}`, true); $('local-state').textContent = 'Không tải được cấu hình'; });
