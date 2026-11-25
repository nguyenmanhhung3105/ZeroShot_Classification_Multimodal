/** Chart logic regression in a minimal DOM harness. Does not claim browser/layout QA. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/vlm_project/demo_web');
const registry = new Map();
class Element {
  constructor(tag) { this.tagName = tag; this.attributes = {}; this.children = []; this._value = undefined; this._text = ''; }
  set id(value) { this.attributes.id = value; registry.set(value, this); }
  get id() { return this.attributes.id; }
  setAttribute(key, value) { if (key === 'id') this.id = value; else this.attributes[key] = String(value); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent || '').join(''); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this._text = ''; if (this.tagName === 'select') this._value = undefined; }
  get options() { return this.children.flatMap(child => child.tagName === 'option' ? [child] : child.options || []); }
  set value(value) { this._value = String(value); }
  get value() { return this._value ?? (this.tagName === 'select' ? this.options[0]?.value || '' : this._text); }
  querySelectorAll(selector) { return this.children.flatMap(child => [child, ...child.querySelectorAll(selector)]).filter(child => selector === '[aria-pressed="true"]' ? child.attributes['aria-pressed'] === 'true' : child.tagName === selector); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  focus() {}
  click() { this.onclick?.(); }
  get outerHTML() { return `<${this.tagName} ${Object.entries(this.attributes).map(([k,v]) => `${k}="${v}"`).join(' ')}>${this._text}${this.children.map(child => child.outerHTML).join('')}</${this.tagName}>`; }
}
const sourceHTML = await fs.readFile(path.join(root, 'index.html'), 'utf8');
for (const match of sourceHTML.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) { const node = new Element(match[1]); node.id = match[2]; }
for (const key of ['macro_f1', 'accuracy', 'balanced_accuracy', 'top5']) { const option = new Element('option'); option.value = key; registry.get('exp-metric').append(option); }
registry.get('view-experiments').dataset = {};
const document = {getElementById: id => registry.get(id), createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag)};
const index = JSON.parse(await fs.readFile(path.join(root, 'experiment-results.json')));
let failNext = false;
const fetch = async url => {
  if (failNext) { failNext = false; return {ok: false, status: 503}; }
  const filename = url === '/api/experiments' ? 'experiment-results.json' : `experiments/${url.split('/').at(-1)}.json`;
  return {ok: true, status: 200, json: async () => JSON.parse(await fs.readFile(path.join(root, filename)))};
};
const window = {};
const context = vm.createContext({window, document, fetch, console, AbortSignal, Blob, URL, setTimeout, XMLSerializer: class { serializeToString(node) { return node.outerHTML; } }});
vm.runInContext(await fs.readFile(path.join(root, 'analytics.js'), 'utf8'), context, {filename: 'analytics.js'});
await window.ExperimentCharts.load();
let charts = 0;
for (const entry of index.runs) {
  const select = registry.get('exp-run'); select.value = entry.id; await select.onchange();
  assert.equal(registry.get('view-experiments').dataset.runId, entry.id, registry.get('exp-status').textContent);
  for (const option of registry.get('exp-method').options) {
    assert.ok(!/(?:CLIP|[LB]) · Tiêu đề/.test(option.textContent), 'Title-only methods must be excluded from the main presentation');
  }
  const titles = registry.get('exp-tabs').children.map(button => button.textContent);
  for (const title of titles) {
    registry.get('exp-tabs').children.find(button => button.textContent === title).click();
    const svg = registry.get('exp-chart').querySelector('svg');
    assert.ok(svg, `${entry.kind}: ${title} missing chart`);
    assert.ok(!/NaN|Infinity|undefined/.test(svg.outerHTML), `${entry.kind}: ${title} invalid SVG values`);
    assert.ok(registry.get('exp-data-table').querySelectorAll('th').length > 1);
    charts++;
  }
}
const select = registry.get('exp-run'); select.value = 'fakeddit-binary-comparison'; await select.onchange();
registry.get('exp-tabs').children.find(button => button.textContent === 'Ma trận và cặp lớp nhầm lẫn').click();
registry.get('exp-control-1').value = 'count'; registry.get('exp-control-1').onchange();
assert.ok(registry.get('exp-data-table').textContent.includes('Real'));
failNext = true;
await select.onchange();
assert.equal(registry.get('exp-data-table').children.length, 0, 'Failed load must not retain old-run data');
assert.equal(registry.get('exp-source').children.length, 0);
assert.equal(registry.get('exp-chart').children.length, 0);
assert.equal(registry.get('exp-download-json').disabled, true);
console.log(`PASS: ${index.runs.length} exported experiment views, ${charts} charts, tables and matrix mode; no invalid SVG numbers. DOM logic only, not layout QA.`);
