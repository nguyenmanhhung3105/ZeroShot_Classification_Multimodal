/* Read-only charts from verified experiment snapshots. No model or third-party scripts. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ns = 'http://www.w3.org/2000/svg';
  const metricNames = {macro_f1: 'Macro-F1', accuracy: 'Accuracy', balanced_accuracy: 'Balanced Accuracy', top5: 'Top-5 Accuracy'};
  const kindNames = {fakeddit_comparison: 'Zero-shot / Few-shot · cùng cohort', fakeddit_task1: 'Zero-shot và StatA',
    fakeddit_fewshot_support: 'Modality và phân bổ support', fakeddit_fewshot_task1: 'Few-shot · ảnh và ảnh + tiêu đề',
    fakeddit_fewshot_cascade: 'Phẳng và cascade', fewshot_support_thresholds: 'Đánh đổi ngưỡng · audit dev',
    fewshot_task1_train_cv: 'Chọn C trong train', paper_reference: 'Tham chiếu OpenAI CLIP', benchmark: 'Prompt benchmark',
    waffle_semantic_stata: 'Waffle–semantic–StatA', waffle_stata: 'Waffle–StatA', dclip_stata: 'DCLIP–StatA',
    fakeddit_hierarchy: 'Hierarchy · cohort cũ', fakeddit_stata_calip: 'StatA / CALIP · cohort cũ',
    cifar100_em_stata: 'EM / StatA', cifar100_rerank: 'Rerank', cifar100_two_stage: 'Phân loại hai tầng'};
  const datasetNames = {fakeddit_2way: 'Fakeddit · Real/Fake', fakeddit_6way: 'Fakeddit · Sáu lớp',
    dtd: 'DTD', eurosat: 'EuroSAT', imagenet: 'ImageNet', imagenetv2: 'ImageNetV2', cub: 'CUB', food101: 'Food101', pets: 'Oxford Pets', places365: 'Places365', cifar100: 'CIFAR-100'};
  const names = {clip_image: 'CLIP · Ảnh', clip_title: 'CLIP · Tiêu đề', clip_image_title: 'CLIP · Fusion',
    stata_image: 'StatA · Ảnh', clip_stata_image: 'CLIP–StatA · Ảnh', clip_stata_image_title: 'CLIP–StatA · Fusion',
    clip: 'CLIP', dclip: 'DCLIP', waffle: 'WaffleCLIP', waffle_unit: 'WaffleCLIP · unit',
    waffle_semantic: 'Waffle–semantic', waffle_semantic_stata: 'Waffle–semantic–StatA'};
  const palette = ['#2f61c4', '#188276', '#af762c', '#7452a5', '#50667e', '#b14f59'];
  const state = {index: null, run: null, tab: 'summary', request: 0, loaded: false, loading: false,
    method: '', classPage: 0, classSearch: '', classMetric: 'f1', matrixSeed: 'mean', matrixMode: 'percent', curve: 0, probability: 'nll', subtypeGroup: '', subtypeMetric: '', table: []};
  const html = (tag, text, cls) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; };
  function svg(tag, attrs = {}, text) { const node = document.createElementNS(ns, tag); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value)); if (text !== undefined) node.textContent = text; return node; }
  const short = (text, n = 40) => text.length > n ? text.slice(0, n - 1) + '…' : text;
  const pct = value => (value * 100).toFixed(2) + '%';
  const fmt = value => Number.isFinite(value) ? Number(value.toFixed(3)).toLocaleString('vi-VN') : String(value ?? '—');
  const agg = values => { const mean = values.reduce((a, b) => a + b, 0) / values.length; return {mean, std: values.length > 1 ? Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (values.length - 1)) : 0}; };
  function titleOf(method) {
    if (names[method]) return names[method];
    let prefix = '', name = method;
    if (name.startsWith('balanced_binary_')) { prefix = 'B · '; name = name.slice(16); }
    if (name.startsWith('legacy_6way_')) { prefix = 'L · '; name = name.slice(12); }
    const replacements = {image_title: 'Ảnh–tiêu đề', image: 'Ảnh', title: 'Tiêu đề', flat: 'Phẳng 6 lớp', hierarchy_default: 'Cascade mặc định', 'hierarchy_target_0.87': 'Cascade R87'};
    return prefix + (replacements[name] || names[name] || name.replaceAll('_', ' '));
  }
  function metric() { return $('exp-metric').value; }
  function selected() { return state.run.methods.find(row => row.key === state.method) || state.run.methods[0]; }
  function color(row) { return row.regime === 'Few-shot' || row.method.includes('balanced_binary') ? palette[2] : /stata|majority|em_/.test(row.method) ? palette[1] : palette[0]; }
  function plot(width, height, title, description) {
    const node = svg('svg', {viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': title,
      'font-family': 'Arial, sans-serif', 'font-size': 12, fill: '#233954', xmlns: ns});
    node.append(svg('title', {}, title), svg('desc', {}, description), svg('rect', {x: 0, y: 0, width, height, fill: '#fff'}));
    return node;
  }
  function tip(node, text) { node.setAttribute('tabindex', '0'); node.setAttribute('aria-label', text); node.append(svg('title', {}, text)); }
  function mount(node, target = $('exp-chart')) { const wrap = html('div', undefined, 'report-chart'); wrap.append(node); target.append(wrap); }
  function heading(text, parent = $('exp-chart')) { parent.append(html('h3', text, 'chart-subtitle')); }
  function note(text, parent = $('exp-chart')) { parent.append(html('p', text, 'chart-footnote')); }
  function bars(items, title, options = {}, target) {
    const width = 1080, left = 340, right = 105, top = 42, step = 39, height = top + items.length * step + 45;
    const lower = options.diverging ? -Math.max(1, ...items.map(row => Math.abs(row.value) + (row.std || 0))) * 1.15 : 0;
    const upper = options.diverging ? -lower : options.percent ? 100 : Math.max(1, ...items.map(row => row.value + (row.std || 0))) * 1.15;
    const x = value => left + (value - lower) / (upper - lower) * (width - left - right);
    const node = plot(width, height, title, options.note || 'Thanh biểu diễn giá trị; đường mảnh biểu diễn độ lệch chuẩn mẫu nếu có.');
    for (let i = 0; i <= 5; i++) {
      const value = lower + (upper - lower) * i / 5;
      node.append(svg('line', {x1: x(value), x2: x(value), y1: top - 17, y2: height - 35, stroke: '#e5eaf2'}),
        svg('text', {x: x(value), y: 16, 'text-anchor': 'middle', fill: '#526780', 'font-size': 11}, `${Number(value.toFixed(1))}${options.percent ? '%' : ''}`));
    }
    node.append(svg('line', {x1: x(0), x2: x(0), y1: top - 17, y2: height - 35, stroke: '#a5b6ce'}));
    items.forEach((row, i) => {
      const y = top + i * step;
      const label = svg('text', {x: left - 15, y: y + 17, 'text-anchor': 'end', 'font-size': 12}, short(row.label));
      label.append(svg('title', {}, row.label));
      const bar = svg('rect', {x: Math.min(x(0), x(row.value)), y, width: Math.max(1, Math.abs(x(row.value) - x(0))), height: 24, rx: 3, fill: row.color || palette[i % palette.length]});
      tip(bar, `${row.label}: ${fmt(row.value)}${options.percent ? '%' : ''}${row.std ? ' ± ' + fmt(row.std) : ''}`);
      node.append(label, bar);
      if (row.std) {
        const a = x(row.value - row.std), b = x(row.value + row.std);
        node.append(svg('line', {x1: a, x2: b, y1: y + 12, y2: y + 12, stroke: '#172e4f', 'stroke-width': 1.5}),
          svg('line', {x1: a, x2: a, y1: y + 7, y2: y + 17, stroke: '#172e4f'}), svg('line', {x1: b, x2: b, y1: y + 7, y2: y + 17, stroke: '#172e4f'}));
      }
      node.append(svg('text', {x: width - right + 13, y: y + 17, 'font-size': 12}, `${fmt(row.value)}${options.percent ? '%' : ''}`));
    });
    mount(node, target);
  }
  function dataTable(headers, rows) {
    state.table = [headers, ...rows];
    const table = html('table'), head = html('thead'), tr = html('tr'), body = html('tbody');
    headers.forEach(text => { const th = html('th', text); th.scope = 'col'; tr.append(th); }); head.append(tr);
    rows.forEach(values => { const row = html('tr'); values.forEach(value => row.append(html('td', typeof value === 'number' ? fmt(value) : value))); body.append(row); });
    table.append(head, body); $('exp-data-table').replaceChildren(table);
  }
  function control(label, options, value, change) {
    const box = html('div'), caption = html('label', label), select = html('select');
    const id = 'exp-control-' + $('exp-extra-controls').children.length;
    select.id = id; caption.htmlFor = id;
    options.forEach(([key, text]) => { const option = html('option', text); option.value = key; select.append(option); });
    select.value = value; select.onchange = () => change(select.value); box.append(caption, select); $('exp-extra-controls').append(box);
  }
  function summary(rows = state.run.methods) {
    const key = metric(), valid = rows.filter(row => row.metrics[key]);
    bars(valid.map(row => ({label: (row.regime === 'Few-shot' ? 'Few-shot · ' : '') + titleOf(row.method), value: row.metrics[key].mean * 100, std: row.metrics[key].std * 100, color: color(row)})), metricNames[key], {percent: true});
    dataTable(['Phương pháp', 'Nhánh', metricNames[key] + ' (%)', 'Std (điểm %)', 'Số seed', 'Mẫu / seed'], valid.map(row => [titleOf(row.method), row.regime || kindNames[state.run.kind] || state.run.kind, row.metrics[key].mean * 100, row.metrics[key].std * 100, row.seeds, row.samples]));
    note('Mean ± sample std (ddof = 1). Một kết quả tất định có std = 0. Số seed có thể khác giữa các nhánh.');
    if (state.run.support_budget) {
      const budgets = grouped(state.run.support_budget, ['policy']);
      note('Support có nhãn: ' + budgets.map(group => `${group.name === 'legacy_6way' ? 'L' : group.name === 'balanced_binary' ? 'B' : group.name} = ${fmt(agg(group.values.map(v => v.counts[0])).mean)} Real + ${fmt(agg(group.values.map(v => v.counts.slice(1).reduce((a, b) => a + b, 0))).mean)} Fake / seed`).join('; ') + '.');
    }
  }
  function confusion() {
    const row = selected();
    if (!row.confusion) {
      const pairs = row.confusions;
      bars(pairs.slice(0, 20).map(item => ({label: `${item.truth} → ${item.predicted}`, value: item.count, color: palette[0]})), 'Các cặp lớp nhầm lẫn phổ biến');
      dataTable(['Nhãn thật', 'Dự đoán', 'Số lỗi trung bình / seed'], pairs.map(p => [p.truth, p.predicted, p.count]));
      note('Hiển thị các cặp có nhiều lỗi nhất; không dựng ma trận lớn trên một màn hình. Không bao gồm ô dự đoán đúng.');
      return;
    }
    control('Seed', [['mean', 'Trung bình các seed'], ...row.seed_confusions.map(v => [v.seed, v.seed])], state.matrixSeed, value => { state.matrixSeed = value; render(); });
    control('Đơn vị', [['percent', '% theo nhãn thật'], ['count', 'Số mẫu']], state.matrixMode, value => { state.matrixMode = value; render(); });
    const chosen = row.seed_confusions.find(v => v.seed === state.matrixSeed);
    const matrix = chosen ? chosen.matrix : row.confusion, n = row.labels.length, cell = n <= 6 ? 82 : 33, left = n <= 6 ? 240 : 300, top = 155;
    const width = Math.max(760, left + n * cell + 180), height = top + n * cell + 65;
    const node = plot(width, height, 'Ma trận nhầm lẫn', 'Hàng là nhãn thật, cột là nhãn dự đoán.');
    node.append(svg('text', {x: left + n * cell / 2, y: 25, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 600}, 'NHÃN DỰ ĐOÁN'));
    row.labels.forEach((label, i) => {
      const x = left + i * cell + cell / 2;
      node.append(svg('text', {x, y: top - 14, transform: `rotate(-38 ${x} ${top - 14})`, 'text-anchor': 'start', 'font-size': 11}, short(label, 28)),
        svg('text', {x: left - 15, y: top + i * cell + cell / 2 + 4, 'text-anchor': 'end', 'font-size': 12}, short(label, 30)));
    });
    const max = Math.max(1, ...matrix.flat());
    matrix.forEach((values, i) => {
      const support = values.reduce((a, b) => a + b, 0);
      values.forEach((count, j) => {
        const ratio = support ? count / support : 0, intensity = state.matrixMode === 'percent' ? ratio : count / max;
        const shade = Math.round(248 - intensity * 196), fill = `rgb(${Math.round(shade * .72)},${Math.round(shade * .84)},${Math.max(108, shade)})`;
        const rect = svg('rect', {x: left + j * cell, y: top + i * cell, width: cell - 2, height: cell - 2, rx: 3, fill});
        tip(rect, `${row.labels[i]} → ${row.labels[j]}: ${fmt(count)} mẫu; ${pct(ratio)} theo hàng`); node.append(rect);
        if (n <= 10) node.append(svg('text', {x: left + j * cell + cell / 2 - 1, y: top + i * cell + cell / 2 + 4, 'text-anchor': 'middle', fill: intensity > .6 ? '#fff' : '#183354', 'font-size': 13}, state.matrixMode === 'percent' ? (ratio * 100).toFixed(1) + '%' : fmt(count)));
      });
    });
    mount(node);
    note(`${state.matrixSeed === 'mean' ? 'Số mẫu là trung bình theo seed, có thể là số thập phân.' : 'Số mẫu của seed đã chọn.'} Mỗi hàng chuẩn hóa theo support của nhãn thật; không cộng seed thành số mẫu độc lập.`);
    dataTable(['Nhãn thật / Dự đoán', ...row.labels], matrix.map((values, i) => [row.labels[i], ...values]));
  }
  function classes(distribution = false) {
    const row = selected(), key = state.classMetric;
    if (!distribution) control('Chỉ số từng lớp', [['f1', 'F1'], ['precision', 'Precision'], ['recall', 'Recall']], key, value => { state.classMetric = value; state.classPage = 0; render(); });
    const field = html('div'), label = html('label', 'Tìm lớp'), input = html('input');
    input.type = 'search'; input.id = 'exp-class-search'; input.value = state.classSearch; label.htmlFor = input.id;
    input.onchange = () => { state.classSearch = input.value; state.classPage = 0; render(); };
    field.append(label, input); $('exp-extra-controls').append(field);
    const rows = row.per_class.filter(c => c.label.toLowerCase().includes(state.classSearch.toLowerCase())).sort((a, b) => distribution ? b.support - a.support : a[key].mean - b[key].mean);
    const pages = Math.max(1, Math.ceil(rows.length / 20)); state.classPage = Math.min(state.classPage, pages - 1);
    control('Nhóm lớp hiển thị', Array.from({length: pages}, (_, i) => [String(i), `${i * 20 + 1}–${Math.min((i + 1) * 20, rows.length)} / ${rows.length} lớp`]), String(state.classPage), value => { state.classPage = Number(value); render(); });
    const visible = rows.slice(state.classPage * 20, state.classPage * 20 + 20);
    bars(visible.map(c => ({label: c.label, value: distribution ? c.support : c[key].mean * 100, std: distribution ? 0 : c[key].std * 100, color: palette[0]})), distribution ? 'Phân bố nhãn thật' : key.toUpperCase() + ' theo lớp', {percent: !distribution});
    dataTable(['Lớp', 'Support / seed', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'Std F1 (điểm %)'], visible.map(c => [c.label, c.support, c.precision.mean * 100, c.recall.mean * 100, c.f1.mean * 100, c.f1.std * 100]));
    note(distribution ? 'Support lấy từ nhãn thật trong artifact prediction; không phải phân bố nhãn mô hình dự đoán.' : 'Lớp được sắp theo điểm tăng dần. Biểu đồ hiển thị tối đa 20 lớp mỗi trang; đổi nhóm lớp để xem toàn bộ.');
  }
  function seeds() {
    const row = selected(), key = metric(), points = row.seed_metrics.filter(v => Number.isFinite(v[key]));
    const width = 1000, height = 380, left = 75, top = 40, bottom = 310, end = 940;
    const node = plot(width, height, 'Kết quả theo seed', 'Seed hoặc tổ hợp seed là các cấu hình lặp trên cùng tập đánh giá, không phải epoch.');
    const x = i => points.length === 1 ? (left + end) / 2 : left + i * (end - left) / (points.length - 1);
    const y = value => bottom - value * (bottom - top);
    for (let i = 0; i <= 5; i++) { const value = i / 5; node.append(svg('line', {x1: left, x2: end, y1: y(value), y2: y(value), stroke: '#e1e8f2'}), svg('text', {x: left - 12, y: y(value) + 4, 'text-anchor': 'end'}, (value * 100) + '%')); }
    const mean = row.metrics[key]?.mean;
    if (Number.isFinite(mean)) node.append(svg('line', {x1: left, x2: end, y1: y(mean), y2: y(mean), stroke: palette[2], 'stroke-dasharray': '6 4', 'stroke-width': 2}));
    points.forEach((point, i) => {
      const dot = svg('circle', {cx: x(i), cy: y(point[key]), r: 6, fill: palette[0]}); tip(dot, `${point.seed}: ${pct(point[key])}`);
      node.append(dot, svg('text', {x: x(i), y: bottom + 26, 'text-anchor': 'middle', 'font-size': 11}, short(point.seed, 14)));
    });
    mount(node); note(`Đường nét đứt: trung bình. ${row.seed_type || 'Các điểm là seed hoặc tổ hợp seed của run; không phải đường huấn luyện.'}`);
    dataTable(['Seed / cấu hình lặp', metricNames[key] + ' (%)'], points.map(p => [p.seed, p[key] * 100]));
  }
  function grouped(rows, keys) {
    const groups = new Map();
    rows.forEach(row => { const key = keys.map(k => row[k]).join(' · '); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(row); });
    return [...groups].map(([name, values]) => ({name, values}));
  }
  function cascade() {
    const routing = grouped(state.run.artifacts['routing.tsv'], ['policy', 'decision']);
    const width = 1080, left = 285, right = 90, step = 57, top = 45, height = top + routing.length * step + 75;
    const fields = [['real_routed_to_fake', 'Real → Fake'], ['fake_routed_to_real', 'Fake → Real'], ['subtype_errors_after_correct_fake_gate', 'Sai subtype sau gate đúng']];
    const sums = routing.map(group => fields.reduce((sum, [key]) => sum + agg(group.values.map(v => v[key])).mean, 0));
    const max = Math.max(...sums) * 1.12, scale = value => value / max * (width - left - right);
    const node = plot(width, height, 'Phân rã lỗi cascade', 'Ba nhóm lỗi tách rời; mỗi giá trị là số lỗi trung bình theo seed.');
    const tableRows = [];
    routing.forEach((group, i) => {
      const y = top + i * step; let offset = left; const counts = fields.map(([key]) => agg(group.values.map(v => v[key])).mean);
      node.append(svg('text', {x: left - 15, y: y + 20, 'text-anchor': 'end', 'font-size': 11}, short(group.name)));
      fields.forEach(([key, label], j) => { const rect = svg('rect', {x: offset, y, width: scale(counts[j]), height: 30, fill: palette[j]}); tip(rect, `${group.name} · ${label}: ${fmt(counts[j])}`); node.append(rect); offset += scale(counts[j]); });
      node.append(svg('text', {x: offset + 10, y: y + 20}, fmt(sums[i])));
      tableRows.push([group.name, ...counts, sums[i], (counts[0] + counts[1]) / sums[i] * 100]);
    });
    fields.forEach(([, label], i) => { node.append(svg('rect', {x: 25 + i * 335, y: height - 32, width: 12, height: 12, fill: palette[i]}), svg('text', {x: 43 + i * 335, y: height - 22, 'font-size': 11}, label)); });
    mount(node);
    dataTable(['Policy / gate', ...fields.map(v => v[1]), 'Tổng lỗi / seed', 'Lỗi gate / tổng lỗi (%)'], tableRows);
    const gates = state.run.artifacts['gate_metrics.tsv'];
    if (gates) { heading('Recall tại gate Real/Fake'); const items = [];
      grouped(gates, ['policy', 'decision']).forEach(group => ['recall_real', 'recall_fake'].forEach((field, i) => { const value = agg(group.values.map(v => v[field])); items.push({label: `${group.name} · ${field === 'recall_real' ? 'Real' : 'Fake'}`, value: value.mean * 100, std: value.std * 100, color: palette[i]}); }));
      bars(items, 'Recall gate', {percent: true});
    }
    heading('Head con: toàn bộ Fake và điều kiện qua gate');
    const wrap = html('div', undefined, 'table-wrap'), table = html('table'), head = html('thead'), tr = html('tr'), body = html('tbody');
    ['Policy / gate', 'Accuracy trên mọi Fake thật', 'Accuracy trên Fake qua gate đúng'].forEach(name => tr.append(html('th', name))); head.append(tr);
    routing.forEach(group => { const row = html('tr'); row.append(html('td', group.name), html('td', pct(agg(group.values.map(v => v.child_accuracy_all_true_fake)).mean)), html('td', pct(agg(group.values.map(v => v.child_accuracy_given_correct_fake_gate)).mean))); body.append(row); }); table.append(head, body); wrap.append(table); $('exp-chart').append(wrap);
    note('Hai cột đo trên tập con khác nhau. Conditional accuracy không thay thế chất lượng head con trên toàn bộ Fake thật.');
    const oracle = state.run.diagnostics['oracle_diagnostic.json'];
    if (oracle) {
      const details = html('details', undefined, 'oracle-diagnostic'); details.append(html('summary', 'Chẩn đoán oracle gate — sử dụng nhãn thật, không phải phương pháp triển khai'));
      const entries = Object.entries(oracle.metrics || oracle).filter(([, value]) => value && typeof value === 'object' && Number.isFinite(value.accuracy));
      const items = grouped(entries.map(([name, value]) => ({policy: name.replace(/_s\d+$/, ''), accuracy: value.accuracy})), ['policy']);
      items.forEach(item => details.append(html('p', `${item.name}: Accuracy ${pct(agg(item.values.map(v => v.accuracy)).mean)}`)));
      details.append(html('p', 'Gate dùng nhãn nhị phân thật để phân tích ảnh hưởng định tuyến khi giữ head con cố định. Kết quả này không nằm trong bảng xếp hạng phương pháp.'));
      $('exp-chart').append(details);
    }
  }
  function linePlot(points, xKey, series, title, target) {
    const width = 940, height = 400, left = 72, top = 35, bottom = 320, right = 900;
    const node = plot(width, height, title, 'Các điểm lấy từ artifact xác suất hoặc bảng precision–recall đã lưu.');
    const x = value => left + value * (right - left), y = value => bottom - value * (bottom - top);
    for (let i = 0; i <= 5; i++) { const v = i / 5; node.append(svg('line', {x1: left, x2: right, y1: y(v), y2: y(v), stroke: '#e2e8f1'}), svg('text', {x: left - 10, y: y(v) + 4, 'text-anchor': 'end'}, (v * 100) + '%'), svg('text', {x: x(v), y: bottom + 22, 'text-anchor': 'middle'}, v.toFixed(1))); }
    for (const [key, label, c] of series) {
      const ordered = [...points].filter(p => Number.isFinite(p[xKey]) && Number.isFinite(p[key])).sort((a, b) => a[xKey] - b[xKey]);
      node.append(svg('polyline', {points: ordered.map(p => `${x(p[xKey])},${y(p[key])}`).join(' '), fill: 'none', stroke: c, 'stroke-width': 2}));
      ordered.forEach((point, i) => { if (i % 10 && i !== ordered.length - 1) return; const dot = svg('circle', {cx: x(point[xKey]), cy: y(point[key]), r: 3, fill: c}); tip(dot, `${label}: ${pct(point[key])}; ${xKey}: ${fmt(point[xKey])}`); node.append(dot); });
    }
    series.forEach(([, label, c], i) => node.append(svg('text', {x: left + i * 270, y: height - 12, fill: c, 'font-weight': 600}, label)));
    node.append(svg('text', {x: (left + right) / 2, y: height - 42, 'text-anchor': 'middle'}, xKey === 'threshold' ? 'Ngưỡng p(Fake)' : xKey === 'fpr' ? 'False Positive Rate · lớp dương Fake' : 'Recall Fake'));
    mount(node, target);
  }
  function curves() {
    const items = state.run.artifacts['precision_recall_selection.tsv'];
    state.curve = Math.min(state.curve, items.length - 1);
    control('Policy / modality / seed', items.map((v, i) => [String(i), `${v.policy} · ${v.modality} · seed ${v.seed}`]), String(state.curve), value => { state.curve = Number(value); render(); });
    const curve = items[state.curve];
    linePlot(curve.points, 'recall_fake', [['precision_fake', 'Precision Fake', palette[0]]], 'Precision–Recall trên selection dev');
    heading('Đánh đổi recall theo ngưỡng');
    linePlot(curve.points, 'threshold', [['recall_fake', 'Recall Fake', palette[0]], ['recall_real', 'Recall Real', palette[1]]], 'Recall Real/Fake theo ngưỡng');
    note(`Đường selection dev, không phải đường trên test độc lập. Hiển thị ${curve.points.length}/${curve.source_points} điểm theo mẫu chỉ số đều; không tính AUC từ đường rút gọn. Mục tiêu recall 0,87 không có nghĩa ngưỡng xác suất = 0,87.`);
    const thresholds = (state.run.artifacts['thresholds.tsv'] || []).filter(row => row.policy === curve.policy && row.modality === curve.modality && row.seed === curve.seed);
    if (thresholds.length) {
      heading('Ngưỡng đã chọn trên selection dev');
      thresholds.forEach(row => note(`Mục tiêu Recall Fake: ${pct(row.target_recall_fake)} · Ngưỡng: ${fmt(row.threshold)} · Recall Fake selection: ${pct(row.selection_recall_fake)} · Recall Real selection: ${pct(row.selection_recall_real)}`));
    }
    dataTable(['Ngưỡng', 'Precision Fake (%)', 'Recall Fake (%)', 'Recall Real (%)'], curve.points.map(p => [p.threshold, p.precision_fake * 100, p.recall_fake * 100, p.recall_real * 100]));
  }
  function probability() {
    const key = state.probability, titles = {nll: 'NLL đánh giá đầu ra', mean_confidence: 'Điểm dự đoán cực đại trung bình', mean_normalized_entropy: 'Entropy chuẩn hóa trung bình'};
    control('Chẩn đoán', Object.entries(titles), key, value => { state.probability = value; render(); });
    const groups = grouped(state.run.diagnostics['branch_diagnostics.json'], ['method']);
    const items = groups.map(group => { const s = agg(group.values.map(v => v[key])); return {label: titleOf(group.name), value: s.mean, std: s.std, color: palette[0]}; });
    bars(items, titles[key]);
    dataTable(['Phương pháp', titles[key], 'Sample std', 'Số cấu hình lặp'], items.map((item, i) => [item.label, item.value, item.std, groups[i].values.length]));
    note('Đây là chẩn đoán xác suất đầu ra trên tập đánh giá, không phải train loss hoặc loss theo epoch. Điểm cực đại lớn không đồng nghĩa Accuracy cao hoặc đã hiệu chuẩn.');
  }
  function roc() {
    const row = selected(), records = row.roc.curves;
    if (!records.some(r => r.seed === state.rocSeed)) state.rocSeed = records[0].seed;
    control('Seed / cấu hình lặp', records.map(r => [r.seed, r.seed]), state.rocSeed, value => { state.rocSeed = value; render(); });
    const curve = records.find(r => r.seed === state.rocSeed);
    linePlot(curve.points, 'fpr', [['tpr', 'True Positive Rate · Fake', palette[0]]], 'ROC trên tập đánh giá của run');
    heading('Precision–Recall trên cùng tập đánh giá');
    linePlot(curve.points, 'recall', [['precision', 'Precision Fake', palette[1]]], 'Precision–Recall đánh giá');
    note(`Lớp dương: Fake. Biểu đồ của seed ${curve.seed}: ROC-AUC ${fmt(curve.auc)}, AP ${fmt(curve.average_precision)}. AUC/AP được tính từ toàn bộ ngưỡng có gộp các score bằng nhau; biểu đồ hiển thị ${curve.points.length}/${curve.source_points} điểm. Đây là kết quả đánh giá đầu ra, không phải train loss.`);
    dataTable(['Seed', 'ROC-AUC · Fake', 'Average Precision · Fake'], records.map(r => [r.seed, r.auc, r.average_precision]));
  }
  function delta() {
    const key = 'delta_' + metric() + '_pp', records = state.run.artifacts['comparisons_summary.tsv'];
    const rows = records.filter(row => Number.isFinite(row[key + '_mean']));
    bars(rows.map(row => ({label: [row.policy, row.modality, row.decision].filter(Boolean).join(' · '), value: row[key + '_mean'], std: row[key + '_std'], color: row[key + '_mean'] >= 0 ? palette[1] : palette[5]})), 'Chênh lệch ghép cặp', {diverging: true});
    const direction = state.run.kind === 'fakeddit_fewshot_cascade' ? 'Cascade − Phẳng, cùng policy/seed.' : 'Support B − Support L, cùng modality/decision/seed.';
    note(direction + ' Đơn vị: điểm phần trăm. Giá trị âm và dương đều được giữ; std của chênh lệch ghép cặp, không suy ra bằng cách trừ hai std.');
    dataTable(['Đối chiếu', 'Δ ' + metricNames[metric()] + ' (điểm %)', 'Sample std'], rows.map(row => [[row.policy, row.modality, row.decision].filter(Boolean).join(' · '), row[key + '_mean'], row[key + '_std']]));
  }
  function subtypes() {
    const rows = state.run.artifacts['subtype_summary.tsv'];
    const groups = grouped(rows, ['policy', 'decision']);
    if (!groups.some(group => group.name === state.subtypeGroup)) state.subtypeGroup = groups[0].name;
    control('Policy / gate', groups.map(group => [group.name, group.name]), state.subtypeGroup, value => { state.subtypeGroup = value; render(); });
    const group = groups.find(v => v.name === state.subtypeGroup);
    const fields = Object.keys(group.values[0]).filter(key => key.endsWith('_mean') && key.includes('recall'));
    if (!fields.includes(state.subtypeMetric)) state.subtypeMetric = fields[0];
    const field = state.subtypeMetric;
    const labels = {binary_detection_recall_mean: 'Recall quyết định nhị phân theo subtype', binary_gate_recall_mean: 'Recall gate', end_to_end_recall_mean: 'Recall đầu cuối sáu lớp'};
    control('Chỉ số subtype', fields.map(key => [key, labels[key] || key]), field, value => { state.subtypeMetric = value; render(); });
    const six = ['Real', 'Satire/Parody', 'False Connection', 'Imposter Content', 'Manipulated Content', 'Misleading Content'];
    const label = row => row.subtype || six[row.class_6way] || String(row.class_6way);
    bars(group.values.map(row => ({label: label(row), value: row[field] * 100, std: (row[field.replace('_mean', '_std')] || 0) * 100, color: palette[0]})), labels[field] || field, {percent: true});
    dataTable(['Subtype thật', labels[field] || field, 'Sample std (điểm %)'], group.values.map(row => [label(row), row[field] * 100, (row[field.replace('_mean', '_std')] || 0) * 100]));
    note(field.includes('binary') ? 'Đây là chất lượng quyết định Real/Fake trong từng nhóm subtype thật, không phải Accuracy phân loại sáu lớp.' : 'Recall đầu cuối chịu tác động của cả gate và head con.');
  }
  function support() {
    const groups = grouped(state.run.support_budget, ['policy']);
    const items = [];
    groups.forEach((group, g) => group.values[0].labels.forEach((label, i) => { const value = agg(group.values.map(row => row.counts[i])); items.push({label: `${group.name} · ${label}`, value: value.mean, std: value.std, color: palette[g]}); }));
    bars(items, 'Ngân sách support có nhãn');
    dataTable(['Policy', 'Seed', ...groups[0].values[0].labels, 'Tổng support'], state.run.support_budget.map(row => [row.policy, row.seed, ...row.counts, row.counts.reduce((a, b) => a + b, 0)]));
    note('Đếm nhãn từ support artifact, không phải đường học theo số shot. Các head dùng chung ngân sách; head con của cascade chỉ dùng phần Fake trong support đó.');
  }
  function oracle() {
    const value = state.run.diagnostics['oracle_diagnostic.json'];
    const entries = Object.entries(value.metrics || value).filter(([, row]) => row && typeof row === 'object' && Number.isFinite(row.accuracy));
    const rows = grouped(entries.map(([key, row]) => ({name: key.replace(/_s\d+$/, ''), ...row})), ['name']);
    bars(rows.map(group => { const v = agg(group.values.map(row => row[metric()] ?? row.accuracy)); return {label: group.name, value: v.mean * 100, std: v.std * 100, color: palette[4]}; }), 'Oracle gate · chẩn đoán', {percent: true});
    dataTable(['Cấu hình oracle', 'Accuracy (%)', 'Macro-F1 (%)', 'Balanced accuracy (%)'], rows.map(group => [group.name, ...['accuracy', 'macro_f1', 'balanced_accuracy'].map(key => agg(group.values.map(row => row[key])).mean * 100)]));
    note('Sử dụng nhãn nhị phân thật ở gate. Không phải phương pháp triển khai, không nằm trong bảng đối chiếu chính và không phải cận trên cho mọi chỉ số.');
  }
  const panels = {
    summary: ['So sánh phương pháp', 'Mean và độ lệch chuẩn trên cùng bộ kết quả. Không xếp hạng từ điểm của một ảnh.', () => summary()],
    modality: ['Đóng góp ảnh–tiêu đề', 'Các modality và policy support được giữ riêng; không cộng gộp thành một phương pháp.', () => summary(state.run.methods.filter(row => /(?:image|title)$/.test(row.method)))],
    confusion: ['Ma trận và cặp lớp nhầm lẫn', 'Nhãn thật và dự đoán lấy từ artifact prediction, không suy diễn từ Accuracy.', confusion],
    classes: ['Chỉ số theo lớp', 'Precision, Recall và F1 trung bình theo seed. Support là số mẫu của mỗi lớp.', () => classes()],
    distribution: ['Phân bố lớp đánh giá', 'Đếm nhãn thật của cohort đã dùng trong run, không đọc dataset gốc.', () => classes(true)],
    seeds: ['Biến thiên giữa các seed', 'Các lần lặp trên cùng cohort; không phải các epoch huấn luyện.', seeds],
    cascade: ['Gate và head con', 'Tách lỗi định tuyến, lỗi subtype và chất lượng head con.', cascade],
    delta: ['Chênh lệch ghép cặp', 'Đối chiếu các cấu hình trên cùng seed, theo bảng phân tích đã lưu.', delta],
    curves: ['Precision–Recall và ngưỡng', 'Ngưỡng được chọn trên selection dev; bộ audit dev không phải test độc lập.', curves],
    probability: ['Chẩn đoán xác suất', 'NLL và entropy từ bảng diagnostic đã lưu; không phải train loss.', probability],
    subtypes: ['Recall theo subtype', 'Phân biệt chất lượng gate nhị phân và recall đầu cuối theo nhãn thật.', subtypes],
    support: ['Ngân sách support', 'Phân bổ mẫu train có nhãn giữa các policy và seed.', support],
    oracle: ['Chẩn đoán oracle gate', 'Dùng nhãn gate thật; tách biệt với các phương pháp suy luận có thể triển khai.', oracle],
    roc: ['ROC / PR đánh giá', 'Tính lại từ p(Fake) đã lưu; chỉ hiển thị ở phương pháp có score, trên đúng tập đánh giá của run.', roc],
  };
  function available() {
    const tabs = ['summary'];
    if (state.run.methods.some(r => r.method.includes('image_title')) && state.run.methods.some(r => /(?:^|_)image$/.test(r.method))) tabs.push('modality');
    tabs.push('confusion', 'classes', 'distribution', 'seeds');
    if (state.run.artifacts['routing.tsv']) tabs.push('cascade');
    if (state.run.artifacts['comparisons_summary.tsv']) tabs.push('delta');
    if (state.run.artifacts['precision_recall_selection.tsv']) tabs.push('curves');
    if (Array.isArray(state.run.diagnostics['branch_diagnostics.json'])) tabs.push('probability');
    if (state.run.artifacts['subtype_summary.tsv']) tabs.push('subtypes');
    if (state.run.support_budget) tabs.push('support');
    if (state.run.diagnostics['oracle_diagnostic.json']) tabs.push('oracle');
    if (state.run.methods.some(row => row.roc)) tabs.push('roc');
    return tabs;
  }
  function render() {
    if (!state.run) return;
    const tabs = available(); if (!tabs.includes(state.tab)) state.tab = 'summary';
    if (state.tab === 'roc' && !selected().roc) {
      state.method = state.run.methods.find(row => row.roc).key;
      $('exp-method').value = state.method;
    }
    const [title, description, draw] = panels[state.tab];
    if (state.tab === 'oracle' && metric() === 'top5') $('exp-metric').value = 'accuracy';
    $('exp-chart-title').textContent = title; $('exp-chart-note').textContent = description;
    $('exp-chart').replaceChildren(); $('exp-extra-controls').replaceChildren();
    $('exp-method-control').hidden = !['confusion', 'classes', 'distribution', 'seeds', 'roc'].includes(state.tab);
    $('exp-metric').disabled = !['summary', 'modality', 'seeds', 'delta', 'oracle'].includes(state.tab);
    $('exp-tabs').replaceChildren();
    tabs.forEach(key => { const button = html('button', panels[key][0]); button.type = 'button'; button.setAttribute('aria-pressed', String(key === state.tab)); button.onclick = () => { state.tab = key; render(); $('exp-tabs').querySelector('[aria-pressed="true"]').focus({preventScroll: true}); }; $('exp-tabs').append(button); });
    draw();
    $('exp-download-svg').disabled = !$('exp-chart').querySelector('svg');
    $('exp-download-csv').disabled = !state.table.length;
  }
  function provenance() {
    const run = state.run; $('exp-meta').replaceChildren();
    for (const [name, value] of [['Dataset', datasetNames[run.dataset] || run.dataset], ['Split', run.split], ['Mẫu / seed', run.samples.toLocaleString('vi-VN')], ['Backbone', run.model === 'metaclip2_vith14' ? 'MetaCLIP2 ViT-H/14' : 'OpenAI CLIP ViT-L/14'], ['Cohort', run.cohort.slice(0, 12)]]) {
      const item = html('div'); item.append(html('span', name), html('strong', value)); $('exp-meta').append(item);
    }
    $('exp-source').replaceChildren(html('p', run.scope || 'Kết quả full theo metadata run; không suy ra đánh giá độc lập chỉ từ tên split.'));
    for (const source of run.sources) { const item = html('p'); item.append(html('code', source.run), html('br'), html('span', 'SHA256 run: ' + source.sha256)); $('exp-source').append(item); }
    $('exp-source').append(html('p', `Audit nguồn: ${state.index.audit}. Các artifact dùng để dựng biểu đồ đã được đối chiếu checksum khi xuất snapshot.`));
  }
  async function fetchJSON(path) { const response = await fetch(path, {cache: 'no-store', signal: AbortSignal.timeout(15000)}); if (!response.ok) throw new Error('HTTP ' + response.status); return response.json(); }
  async function chooseRun(id) {
    const request = ++state.request; state.run = null; $('exp-status').hidden = false; $('exp-status').textContent = 'Đang tải bộ kết quả…';
    delete $('view-experiments').dataset.runId;
    for (const name of ['exp-chart', 'exp-meta', 'exp-data-table', 'exp-source', 'exp-tabs', 'exp-extra-controls']) $(name).replaceChildren();
    state.table = [];
    $('exp-chart-title').textContent = 'Đang tải kết quả'; $('exp-chart-note').textContent = '';
    $('exp-method-control').hidden = true;
    $('exp-download-json').disabled = true; $('exp-download-svg').disabled = true; $('exp-download-csv').disabled = true;
    try {
      const run = await fetchJSON('/api/experiments/' + encodeURIComponent(id)); if (request !== state.request) return;
      // Filter presentation only; source snapshots and research results stay intact.
      run.methods = run.methods.filter(row => !/(?:^|_)title$/.test(row.method) || row.method.endsWith('image_title'));
      for (const records of Object.values(run.artifacts || {})) {
        if (Array.isArray(records)) {
          for (let i = records.length - 1; i >= 0; i--) {
            const row = records[i];
            if (row.modality === 'title' || (row.method && /(?:^|_)title$/.test(row.method) && !row.method.endsWith('image_title'))) records.splice(i, 1);
          }
        }
      }
      state.run = run; state.classPage = 0; state.classSearch = ''; state.matrixSeed = 'mean'; state.curve = 0;
      const select = $('exp-method'); select.replaceChildren();
      run.methods.forEach(row => { const option = html('option', titleOf(row.method)); option.value = row.key; select.append(option); }); state.method = select.value;
      const metricOptions = [...$('exp-metric').options]; metricOptions.forEach(option => { option.hidden = !run.methods.some(row => row.metrics[option.value]); });
      if (!run.methods.some(row => row.metrics[metric()])) $('exp-metric').value = 'macro_f1';
      provenance(); render(); $('view-experiments').dataset.runId = run.id;
      $('exp-status').textContent = run.split === 'dev' ? 'Đánh giá phát triển (dev). Không phải test độc lập.' : 'Full cohort của thí nghiệm; phạm vi diễn giải theo metadata nguồn.';
      $('exp-status').className = 'notice experiment-scope'; $('exp-download-json').disabled = false;
    } catch (error) {
      if (request !== state.request) return;
      $('exp-status').textContent = 'Không tải được snapshot thực nghiệm: ' + error.message;
      $('exp-status').className = 'notice danger';
    }
  }
  function download(content, filename, type) { const url = URL.createObjectURL(new Blob([content], {type})), anchor = html('a'); anchor.href = url; anchor.download = filename; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
  async function load() {
    if (state.loaded || state.loading) return;
    state.loading = true;
    try {
      state.index = await fetchJSON('/api/experiments');
      const select = $('exp-run'); select.replaceChildren();
      const groups = new Map();
      for (const run of state.index.runs) {
        if (!groups.has(run.dataset)) { const group = html('optgroup'); group.label = datasetNames[run.dataset] || run.dataset; groups.set(run.dataset, group); select.append(group); }
        const option = html('option', `${kindNames[run.kind] || run.kind} · ${run.samples} mẫu · ${run.model.includes('metaclip') ? 'MetaCLIP2' : 'OpenAI'} · ${run.cohort.slice(0, 8)}`); option.value = run.id; groups.get(run.dataset).append(option);
      }
      select.disabled = false; select.value = state.index.default; select.onchange = () => chooseRun(select.value);
      $('exp-metric').onchange = render;
      $('exp-method').onchange = () => { state.method = $('exp-method').value; state.classPage = 0; state.matrixSeed = 'mean'; render(); };
      $('exp-download-json').onclick = () => { if (state.run) download(JSON.stringify(state.run, null, 2), `experiment-${state.run.id}.json`, 'application/json'); };
      $('exp-download-csv').onclick = () => download('\ufeff' + state.table.map(row => row.map(value => '"' + String(value ?? '').replaceAll('"', '""') + '"').join(',')).join('\r\n'), `experiment-${state.run.id}-${state.tab}.csv`, 'text/csv;charset=utf-8');
      $('exp-download-svg').onclick = () => { const node = $('exp-chart').querySelector('svg'); if (node) download(new XMLSerializer().serializeToString(node), `experiment-${state.run.id}-${state.tab}.svg`, 'image/svg+xml'); };
      state.loaded = true; await chooseRun(select.value);
    } catch (error) { $('exp-status').textContent = 'Không tải được danh mục: ' + error.message; $('exp-status').className = 'notice danger'; }
    finally { state.loading = false; }
  }
  window.ExperimentCharts = {load};
})();
