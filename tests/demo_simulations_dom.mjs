/** Catalog coverage and synthetic arithmetic; no model or dataset access. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const registry=new Map();
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.attributes={};this.style={};this.dataset={};this._text='';}
  set textContent(v){this._text=String(v);this.children=[];}
  get textContent(){return this._text+this.children.map(e=>e.textContent).join('');}
  setAttribute(k,v){this.attributes[k]=String(v);}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;this._text='';}
  getBoundingClientRect(){return {width:1000};}
  click(){if(!this.disabled)this.onclick?.();}
  focus(){}
}
for(const id of ['view-simulations','sim-tabs','sim-families','sim-case','sim-stage','sim-method-title','sim-input-preview','sim-calculation','sim-visual','sim-explanation','sim-output','sim-prev','sim-next','sim-restart','sim-method-note','sim-play','sim-timeline'])registry.set(id,new Element());
let timer=null;
const context=vm.createContext({window:{},document:{getElementById:id=>registry.get(id),createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)},ResizeObserver:class{observe(){}},setInterval:fn=>{timer=fn;return 1;},clearInterval:()=>{timer=null;}});
for(const file of ['simulations-recipes.js','simulations-stage.js','simulations.js'])vm.runInContext(fs.readFileSync('src/vlm_project/demo_web/'+file,'utf8'),context,{filename:file});
const sim=context.window.MethodSimulations,recipes=context.window.SimulationRecipes;
sim.init();
const catalog=[...fs.readFileSync('src/vlm_project/demo_service.py','utf8').matchAll(/^    '(\w+)': \('/gm)].map(m=>m[1]);
assert.equal(catalog.length,27);
assert.deepEqual(Object.values(sim.methods).map(m=>m.catalogId).sort(),catalog.filter(id=>id!=='clip_stata_image_title').sort());
assert.equal(registry.get('sim-families').children.length,4);
assert.equal(sim.argmax([.4,.08,.1,.04,.3,.08]),0);
assert.equal(sim.route(.5,[.1,.2,.1,.2,.4]),0);
assert.equal(sim.route(.36,[.1,.2,.1,.2,.4],.36),5);
assert.equal(sim.chooseThreshold([.9,.8,.7,.6,.5,.4,.3,.2,.1,.05],.87),.1);
const tau=sim.chooseThreshold(sim.selectionFake,.87);
assert.ok(Math.abs(tau-.36)<1e-12);
assert.equal(sim.selectionFake.filter(x=>x>=tau).length,87);
assert.ok(Math.abs(recipes.softmax([41.2,40.8]).reduce((a,b)=>a+b)-1)<1e-12);
function end(){for(let i=0;i<10&&!registry.get('sim-next').disabled;i++)registry.get('sim-next').click();}
for(const [key,method] of Object.entries(sim.methods)){
  sim.selectMethod(key);
  registry.get('sim-restart').click();
  assert.match(registry.get('sim-method-title').textContent,new RegExp(method.name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')));
  assert.equal(registry.get('sim-input-preview').children[0].src,'/simulations-sample.jpg');
  for(let i=0;i<method.steps.length;i++){
    assert.ok(registry.get('sim-calculation').textContent.length>50);
    assert.ok(!/NaN|Infinity|undefined/.test(registry.get('sim-visual').textContent+registry.get('sim-calculation').textContent));
    registry.get('sim-next').click();
  }
  assert.equal(registry.get('sim-next').disabled,true);
  assert.ok(registry.get('sim-output').children.some(x=>x.className==='sim-output-label'));
}
sim.selectMethod('stata');registry.get('sim-restart').click();
const coords=()=>registry.get('sim-visual').children[0].children.filter(e=>e.attributes['data-image']).map(e=>[e.attributes.cx,e.attributes.cy]);
const initial=coords();assert.equal(initial.length,41);end();assert.deepEqual(coords(),initial);
for(const [key,label] of [['flat','Real'],['cascade','Real'],['r87','Misleading']]){
  sim.selectMethod(key);registry.get('sim-case').value='subtle';registry.get('sim-case').onchange();end();
  assert.equal(registry.get('sim-output').children.find(x=>x.className==='sim-output-label').textContent,label);
}
registry.get('sim-case').value='borderline';registry.get('sim-case').onchange();end();assert.match(registry.get('sim-output').textContent,/Báo nhầm/);
registry.get('sim-restart').click();registry.get('sim-play').click();assert.equal(typeof timer,'function');timer();assert.equal(registry.get('sim-timeline').children[1].attributes['aria-pressed'],'true');
sim.selectMethod('clip_image');assert.equal(timer,null);
console.log('PASS: 26 displayed catalog methods, separate states, input asset, every math step, normalized scores, fixed StatA points, R87 boundary/false positive and autoplay cancellation. DOM logic, not browser layout.');
