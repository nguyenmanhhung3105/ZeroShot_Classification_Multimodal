/* Educational fixtures only. No uploads, model calls, query labels or GPU jobs. */
(() => {
  'use strict';
  const names = ['Real', 'Satire/Parody', 'False Connection', 'Imposter', 'Manipulated', 'Misleading'];
  const methods = {
    stata: {name: 'StatA · ảnh', note: 'Thích nghi từ nhóm ảnh không nhãn. Tiêu đề bài đăng không tham gia nhánh StatA ảnh; prompt mô tả lớp vẫn được dùng.', steps: ['Nhận nhóm ảnh', 'Mã hóa ảnh', 'Điểm ban đầu', 'Ba láng giềng', 'Cập nhật điểm', 'Cập nhật thống kê', 'Dự đoán cuối']},
    flat: {name: 'Few-shot · phẳng 6 lớp', note: 'Một bộ phân loại Logistic Regression được huấn luyện bằng tập hỗ trợ có nhãn từ train. Bộ phân loại chấm điểm trực tiếp sáu lớp trên đặc trưng ảnh và tiêu đề.', steps: ['Nhận bài đăng', 'Mã hóa hai nguồn', 'Head đã học', 'Chấm sáu lớp', 'Chọn một nhãn']},
    cascade: {name: 'Few-shot · cascade', note: 'Tầng đầu phân loại Real/Fake; tầng sau phân biệt năm nhóm Fake. Nếu tầng đầu chọn Real, kết quả của bộ phân loại con không tham gia quyết định cuối.', steps: ['Nhận bài đăng', 'Mã hóa hai nguồn', 'Hai head đã học', 'Gate Real/Fake', 'Định tuyến', 'Chọn nhãn cuối']},
    r87: {name: 'Few-shot · cascade R87', note: 'Giữ nguyên gate và child của Cascade; đổi ngưỡng gate theo mục tiêu Recall Fake ≥ 87% trên selection dev. R87 không phải ngưỡng xác suất 0,87.', steps: ['Nhận bài đăng', 'Mã hóa hai nguồn', 'Chọn ngưỡng trước', 'So điểm với ngưỡng', 'Định tuyến', 'Chọn nhãn cuối']}
  };
  window.SimulationRecipes.enrich(methods);
  const cases = {
    subtle: {name: 'Fake khó · sai ngữ cảnh', title: 'Trung tâm thương mại vừa khai trương hôm nay', context: 'Tình huống giả lập dùng ảnh quảng cáo để khẳng định một sự kiện mới.', truth: 5, gate: .43, child: [.07,.18,.05,.13,.57], flat: [.40,.08,.10,.04,.30,.08]},
    edited: {name: 'Fake rõ · nội dung thao túng', title: 'Ảnh quảng cáo có thêm đối tượng được ghép', context: 'Tình huống điểm giả lập về thao túng; ảnh train gốc không bị sửa.', truth: 4, gate: .82, child: [.05,.05,.04,.80,.06], flat: [.08,.05,.07,.04,.70,.06]},
    genuine: {name: 'Real · thông tin phù hợp', title: 'Các cửa hàng trên quảng cáo đều mang tên phông chữ', context: 'Ảnh và thông tin phù hợp trong tình huống giả lập.', truth: 0, gate: .18, child: [.1,.2,.1,.2,.4], flat: [.70,.05,.08,.02,.06,.09]},
    borderline: {name: 'Real khó · nguy cơ báo nhầm', title: 'Tên cửa hàng trong quảng cáo là tên các phông chữ', context: 'Nội dung được quy ước Real để minh họa chi phí của việc hạ ngưỡng.', truth: 0, gate: .40, child: [.06,.12,.04,.08,.70], flat: [.50,.05,.09,.03,.13,.20]}
  };
  function argmax(scores) { return scores.reduce((best, value, i) => value > scores[best] ? i : best, 0); }
  function route(fake, child, threshold = null) { return (threshold === null ? fake > .5 : fake >= threshold) ? 1 + argmax(child) : 0; }
  function chooseThreshold(fakeScores, target = .87) { return [...fakeScores].sort((a,b) => b-a)[Math.ceil(target * fakeScores.length)-1]; }
  cases.dataset={name:'Mẫu Fakeddit train · d89ibd',title:'the stores on this mall ad are all font names',context:'Tiêu đề gốc của mẫu train d89ibd.',truth:0,gate:.18,child:[.1,.2,.1,.2,.4],flat:[.70,.05,.08,.02,.06,.09]};
  for(const c of Object.values(cases))c.image='/simulations-sample.jpg';
  const selection = Array.from({length: 100}, (_,i) => i <= 86 ? .98-i*(.62/86) : .34-(i-87)*.02);
  const threshold = chooseThreshold(selection);
  const state = Object.fromEntries(Object.keys(methods).map(key => [key, {step: 0, example: 'dataset'}]));
  let active = 'clip_image', ready = false;
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => { const n=document.createElement(tag); if (text !== undefined) n.textContent=text; if (cls) n.className=cls; return n; };
  const fmt = value => `${Math.round(value*100)}%`;
  let timer=null;
  const captions={
    stata:[
      ['41 ảnh, chưa có nhãn', 'Theo dõi ảnh A được khoanh tròn. Các ảnh khác cung cấp ngữ cảnh cho StatA.'],
      ['Ảnh trở thành các điểm đặc trưng', 'Mỗi chấm là một ảnh; tọa độ của các chấm sẽ giữ nguyên trong toàn bộ quá trình.'],
      ['So với mô tả Real và Fake', 'Hình thoi là prototype từ prompt. A ban đầu nghiêng Real: 52% so với Fake 48%.'],
      ['Đọc thêm thông tin từ ảnh giống A', 'Ba đường nối minh họa láng giềng. Thuật toán dùng điểm mềm và độ tương đồng, không bỏ phiếu theo màu.'],
      ['Ba nguồn cùng điều chỉnh điểm A', 'Prompt, phân bố lớp và láng giềng cùng góp vào điểm mới. A đổi màu, nhưng không đổi vị trí.'],
      ['Tâm và độ phân tán được cập nhật', 'Dấu cộng là tâm ước lượng mới. Vùng mờ thay đổi theo thống kê; prototype hình thoi vẫn đứng yên.'],
      ['Chọn lớp có điểm cuối cao hơn', 'Ví dụ kết thúc với Fake 65%, Real 35%. Không phải chỉ vì chấm A ở gần tâm Fake hơn.']
    ],
    flat:[
      ['Một bài đăng đi vào hệ thống', 'Ảnh và tiêu đề cùng mô tả nội dung cần phân loại.'],
      ['Mã hóa ảnh và tiêu đề', 'Hai vector được nối thành h. MetaCLIP2 giữ nguyên trọng số.'],
      ['Một head đã học từ support train', 'Logistic Regression sáu lớp dùng cùng ngân sách 384 mẫu có nhãn mỗi seed.'],
      ['Sáu lớp cùng nhận điểm', 'Head trả điểm riêng cho từng lớp. Những đường nhánh minh họa các lựa chọn cùng cạnh tranh.'],
      ['Chỉ chọn một lớp cao nhất', 'Không cộng năm lớp Fake trước khi chọn; đó sẽ là một quy tắc khác.']
    ],
    cascade:[
      ['Một bài đăng đi vào hệ thống', 'Theo dõi đường đi của cùng một bài đăng qua hai tầng.'],
      ['Mã hóa thành đặc trưng chung', 'Gate và child dùng cùng đặc trưng ảnh–tiêu đề từ backbone đóng băng.'],
      ['Hai bộ phân loại cho hai tầng', 'Tầng đầu học phân loại Real/Fake. Bộ phân loại con học năm nhóm Fake từ các mẫu Fake trong cùng tập hỗ trợ.'],
      ['Gate đưa ra quyết định đầu tiên', 'Mặc định: điểm Fake > 50% thì vào nhánh Fake; nếu bằng nhau chọn Real.'],
      ['Định tuyến theo kết quả Real/Fake', 'Đường sáng biểu diễn nhánh được chọn. Khi tầng đầu chọn Real, bộ phân loại con không thay đổi quyết định đó.'],
      ['Xác định nhãn cuối', 'Nếu tầng đầu chọn Fake, bộ phân loại con chọn nhãn cuối trong năm nhóm Fake.']
    ],
    r87:[
      ['Cùng đầu vào như Cascade', 'Chỉ quy tắc ngưỡng gate thay đổi; backbone và các head vẫn giữ nguyên.'],
      ['Cùng vector, cùng điểm model', 'R87 không huấn luyện lại gate hay child để xử lý bài đăng này.'],
      ['Hạ ngưỡng trong ví dụ này', 'Selection giả lập có 100 Fake. Chọn ngưỡng cao nhất nhận ra ít nhất 87 mẫu: khoảng 36%, không phải 87%.'],
      ['So điểm Fake với ngưỡng đã khóa', 'Điểm lớn hơn hoặc bằng ngưỡng thì vào nhánh Fake. Ngưỡng không được chọn lại từ đáp án của mẫu đang xem.'],
      ['Ngưỡng khác có thể chọn nhánh khác', 'Trong tình huống “Real khó”, hạ ngưỡng có thể giảm bỏ sót Fake nhưng tăng số mẫu Real bị báo nhầm.'],
      ['Đọc nhãn cuối và đánh đổi', '87% là mục tiêu recall trên selection, không bảo đảm accuracy, subtype recall hay recall trên test bằng 87%.']
    ]
  };
  const points=Array.from({length:40},(_,i)=>{
    const k=i<20?0:1,j=i%20,a=j*2.399963,r=Math.sqrt((j+.5)/20);
    return {x:(k?.69:.29)+Math.cos(a)*r*.19,y:(k?.54:.46)+Math.sin(a)*r*.24,k};
  });
  points.push({x:.51,y:.47,k:1});
  const near=points.slice(0,40).map((p,i)=>({i,d:(p.x-.51)**2+(p.y-.47)**2})).sort((a,b)=>a.d-b.d).slice(0,3);
  function stop(){if(timer!==null){clearInterval(timer);timer=null;}if($('sim-play'))$('sim-play').textContent='▶ Chạy mô phỏng';}
  function move(delta){stop();state[active].step=Math.max(0,Math.min(methods[active].steps.length-1,state[active].step+delta));render();}
  function render(){
    const current=state[active],method=methods[active],c=cases[current.example],step=current.step,last=step===method.steps.length-1;
    const caption=captions[active]?.[step] || [method.steps[step],method.note];
    const calculation=window.SimulationRecipes.calculation(method,c,step,threshold);
    $('sim-method-title').textContent='Đang mô phỏng: '+method.name;
    const preview=$('sim-input-preview');preview.replaceChildren();preview.dataset.modality=method.modality;
    const img=el('img');img.src=c.image;img.alt='Ảnh Fakeddit train d89ibd: quảng cáo trung tâm thương mại';
    const copy=el('div');copy.append(el('strong','Đầu vào · ảnh mẫu train'),el('p',current.example==='dataset'?c.title:'Tiêu đề tình huống: '+c.title),el('span','Ảnh train d89ibd · phép tính và điểm là minh họa','sim-input-origin'));preview.append(img,copy);
    const math=$('sim-calculation');math.replaceChildren(el('strong','Phép tính đang thực hiện'),el('code',calculation.formula),el('p',calculation.numbers),el('output',calculation.result));
    $('sim-method-note').textContent=method.note;
    $('sim-case').value=current.example;
    $('sim-case').disabled=active==='stata';
    $('sim-stage').textContent=`Bước ${step+1}/${method.steps.length} · ${caption[0]}`;
    $('sim-prev').disabled=step===0;$('sim-next').disabled=last;
    const visual=$('sim-visual');
    visual.replaceChildren(window.MethodStage.draw({host:visual,method:method.kind,recipe:method,calculation,step,c,threshold,example:current.example,points,near}));
    const timeline=$('sim-timeline');timeline.replaceChildren();
    method.steps.forEach((label,i)=>{
      const b=el('button',`${String(i+1).padStart(2,'0')} ${label}`);b.type='button';
      b.setAttribute('aria-pressed',String(i===step));b.onclick=()=>{stop();current.step=i;render();};timeline.append(b);
    });
    const explanation=$('sim-explanation');
    explanation.replaceChildren(el('p',caption[1]));
    if(active==='stata')explanation.append(el('p','Điểm mới kết hợp phân bố ban đầu, độ phù hợp với phân bố lớp có xét phương sai và thông tin láng giềng. Thuật toán cập nhật xen kẽ điểm gán mềm, tâm và phương sai. Vector ảnh giữ nguyên; nhãn thật của nhóm mẫu không được sử dụng.'));
    if(active==='r87')explanation.append(el('p','Ngưỡng 0,36 trong ví dụ được tính từ 100 điểm Fake giả lập trên tập chọn ngưỡng. Ngưỡng của thí nghiệm thực tế được chọn riêng cho từng cách lấy mẫu hỗ trợ và seed.'));
    const output=$('sim-output');output.replaceChildren();
    output.append(el('span',`${step+1} / ${method.steps.length}`,'sim-step-number'),el('p',caption[1]));
    if(last){
      const pred=method.kind==='stata'?1:method.kind==='flat'?argmax(c.flat):['cascade','r87','hierarchy'].includes(method.kind)?route(c.gate,c.child,method.kind==='r87'?threshold:null):method.kind==='vote6'?4:argmax(calculation.p);
      output.append(el('strong',method.kind==='stata'?'Fake':calculation.p.length===2&&!['cascade','r87','hierarchy'].includes(method.kind)?['Real','Fake'][pred]:names[pred],'sim-output-label'));
      if(active!=='stata'){const binary=calculation.p.length===2&&!['cascade','r87','hierarchy'].includes(method.kind);const matches=binary?pred===Number(c.truth!==0):pred===c.truth;output.append(el('span',matches?'Khớp nhãn minh họa':'Không khớp nhãn minh họa','sim-outcome'));}
      if(active==='r87'&&pred!==c.truth&&c.truth===0)output.append(el('span','Báo nhầm Real thành Fake','sim-outcome'));
    }
  }
  function select(key) {
    stop();active=key;
    const family=methods[active].family;
    for(const button of $('sim-families').children)button.setAttribute('aria-pressed',String(button.dataset.family===family));
    $('sim-tabs').replaceChildren();
    for(const [key,method] of Object.entries(methods).filter(([,m])=>m.family===family)){
      const button=el('button',method.name);button.type='button';button.id=`sim-tab-${key}`;button.dataset.method=key;button.setAttribute('role','tab');button.setAttribute('aria-controls','sim-workspace');button.setAttribute('aria-selected',String(key===active));button.tabIndex=key===active?0:-1;button.onclick=()=>select(key);
      button.onkeydown=event=>{const keys=Object.keys(methods).filter(k=>methods[k].family===family),i=keys.indexOf(active);let next;if(event.key==='ArrowRight')next=keys[(i+1)%keys.length];if(event.key==='ArrowLeft')next=keys[(i+keys.length-1)%keys.length];if(event.key==='Home')next=keys[0];if(event.key==='End')next=keys.at(-1);if(next){event.preventDefault();select(next);document.getElementById(`sim-tab-${next}`)?.focus();}};
      $('sim-tabs').append(button);
    }
    $('sim-workspace')?.setAttribute('aria-labelledby',`sim-tab-${active}`);
    render();
  }
  function init() {
    if(ready) { render(); return; }
    ready=true;
    for(const [key,name] of Object.entries(window.SimulationRecipes.groups)){const b=el('button',name);b.type='button';b.dataset.family=key;b.onclick=()=>select(Object.keys(methods).find(k=>methods[k].family===key));$('sim-families').append(b);}
    for(const [key,c] of Object.entries(cases)) { const option=el('option',c.name);option.value=key;$('sim-case').append(option); }
    $('sim-case').onchange=()=> {stop();state[active].example=$('sim-case').value;state[active].step=0;render();};
    $('sim-prev').onclick=()=>move(-1);
    $('sim-next').onclick=()=>move(1);
    $('sim-restart').onclick=()=>{stop();state[active].step=0;render();};
    $('sim-play').onclick=()=>{
      if(timer!==null){stop();return;}
      if(state[active].step===methods[active].steps.length-1)state[active].step=0;
      render();$('sim-play').textContent='Ⅱ Tạm dừng';
      timer=setInterval(()=>{state[active].step++;render();if(state[active].step===methods[active].steps.length-1)stop();},2400);
    };
    let lastWidth=0;
    if(typeof ResizeObserver!=='undefined')new ResizeObserver(()=>{const width=$('sim-visual').getBoundingClientRect().width;if(Math.abs(width-lastWidth)>1){lastWidth=width;render();}}).observe($('sim-visual'));
    window.addEventListener?.('hashchange',stop);
    document.addEventListener?.('visibilitychange',()=>{if(document.hidden)stop();});
    select(active);
  }
  window.MethodSimulations={init,argmax,route,chooseThreshold,selectionFake:[...selection],methods,selectMethod:select};
})();
