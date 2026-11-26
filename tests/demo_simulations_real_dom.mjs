/** Full HTML DOM integration. Optional dependency is supplied outside the repo. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const packagePath=process.argv[2];
if(!packagePath)throw new Error('Usage: node tests/demo_simulations_real_dom.mjs /path/to/happy-dom');
const {Window}=await import(pathToFileURL(path.join(packagePath,'lib/index.js')));
const window=new Window({url:'http://localhost',settings:{disableJavaScriptFileLoading:true,disableCSSFileLoading:true}});
window.document.write(await fs.readFile('src/vlm_project/demo_web/index.html','utf8'));
window.ResizeObserver=class{observe(){}};
const visual=window.document.getElementById('sim-visual');
let width=1000;
visual.getBoundingClientRect=()=>({width});
for(const file of ['simulations-recipes.js','simulations-stage.js','simulations.js'])window.eval(await fs.readFile('src/vlm_project/demo_web/'+file,'utf8'));
window.MethodSimulations.init();
const $=id=>window.document.getElementById(id);
const sim=window.MethodSimulations;
assert.equal(Object.keys(sim.methods).length,26);
assert.equal($('sim-families').children.length,4);
function end(){for(let i=0;i<10&&!$('sim-next').disabled;i++)$('sim-next').click();}
await fs.mkdir('/private/tmp/vlm-simulation-v3',{recursive:true});
const image=await fs.readFile('src/vlm_project/demo_web/simulations-sample.jpg');
for(const viewport of [1000,360]){
 width=viewport;
 for(const [key,method] of Object.entries(sim.methods)){
  sim.selectMethod(key);$('sim-restart').click();
  assert.ok($('sim-method-title').textContent.includes(method.name));
  assert.ok($('sim-input-preview').querySelector('img').src.endsWith('/simulations-sample.jpg'));
  for(let i=0;i<method.steps.length;i++){
   assert.ok($('sim-calculation').textContent.length>30);
   $('sim-next').click();
  }
  const svg=visual.querySelector('svg');assert.ok(svg);
  assert.ok(!/NaN|Infinity|undefined/.test(svg.outerHTML));
  assert.ok($('sim-output').querySelector('.sim-output-label'));
  if(['stata','flat','cascade','r87','dclip','waffle','clip_image_title','calip','majority'].includes(key))await fs.writeFile(`/private/tmp/vlm-simulation-v3/${key}-${viewport}.svg`,svg.outerHTML.replaceAll('/simulations-sample.jpg','data:image/jpeg;base64,'+image.toString('base64')));
  if(key==='stata')assert.equal(svg.querySelectorAll('[data-image]').length,41);
 }
}
sim.selectMethod('r87');$('sim-case').value='borderline';$('sim-case').dispatchEvent(new window.Event('change'));end();assert.match($('sim-output').textContent,/Báo nhầm/);
$('sim-restart').click();$('sim-play').click();assert.match($('sim-play').textContent,/Tạm dừng/);
sim.selectMethod('stata');assert.match($('sim-play').textContent,/Chạy mô phỏng/);
assert.equal($('sim-details').open,false);
window.happyDOM.abort();
await window.happyDOM.close();
console.log('PASS: 26 methods at both widths, real input image, title and mathematics;  real HTML DOM, all four diagrams at 1000/360px, SVG snapshots, child gating, R87 error case, autoplay cancellation and collapsed explanations. Not browser layout QA.');
