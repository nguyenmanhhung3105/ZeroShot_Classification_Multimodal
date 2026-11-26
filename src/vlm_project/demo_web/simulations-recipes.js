/* Catalog-aligned teaching recipes and worked arithmetic, never model inference. */
(() => {
  'use strict';
  const entries=[
    ['clip_image','CLIP · ảnh','zero','prompt','image','Tên lớp → text encoder'],
    ['clip_title','CLIP · tiêu đề','zero','prompt','title','Tiêu đề so với mô tả lớp'],
    ['clip_image_title','CLIP · fusion','zero','blend','image_title','0,75 ảnh + 0,25 tiêu đề'],
    ['dclip','DCLIP','zero','descriptor','image','Nhiều mô tả đặc trưng mỗi lớp'],
    ['waffle','WaffleCLIP','zero','waffle','image','Tên lớp + hậu tố ngẫu nhiên'],
    ['fewshot_image','Few-shot · ảnh','fewshot','linear','image','Logistic Regression nhị phân'],
    ['fewshot_title','Few-shot · tiêu đề','fewshot','linear','title','Logistic Regression nhị phân'],
    ['fewshot_image_title','Few-shot · ảnh–tiêu đề','fewshot','linear','image_title','Logistic Regression nhị phân'],
    ['legacy_image','Baseline legacy · ảnh','control','legacy','image','Không chuẩn hóa lại prototype sau khi lấy trung bình'],
    ['legacy_title','Baseline legacy · tiêu đề','control','legacy','title','Tiêu đề so với prototype ảnh legacy'],
    ['legacy_fusion','Baseline legacy · fusion','control','blend','image_title','Hai nhánh dùng prototype legacy'],
    ['majority','Majority · ảnh–tiêu đề','control','vote','image_title','CLIP fusion + EM + StatA'],
    ['waffle_unit','WaffleCLIP · unit','control','waffle','image','Chuẩn hóa prototype sau khi lấy trung bình'],
    ['calip','CALIP','control','calip','image','Attention giữa patch ảnh và lớp'],
    ['em_image','EM-Dirichlet','control','em','image','Ước lượng cụm trên vector xác suất'],
    ['flat_stata','StatA · phẳng 6 lớp','control','stata6','image','StatA trên cả sáu lớp'],
    ['flat_majority','Majority · phẳng 6 lớp','control','vote6','image','Ba bộ dự đoán sáu lớp bỏ phiếu'],
    ['clip_stata_image','CLIP–StatA · ảnh','control','blend','image','0,5 CLIP + 0,5 StatA'],
    ['hierarchy','Zero-shot · hierarchy StatA','control','hierarchy','image','Gate StatA → giới hạn lớp con'],
    ['hierarchy_clip','Zero-shot · hierarchy CLIP','control','hierarchy','image','Gate CLIP → giới hạn lớp con'],
    ['hierarchy_em','Zero-shot · hierarchy EM','control','hierarchy','image','Gate EM → giới hạn lớp con'],
    ['hierarchy_majority','Zero-shot · hierarchy Majority','control','hierarchy','image','Gate Majority → giới hạn lớp con']
  ];
  const softmax=values=>{const high=Math.max(...values),e=values.map(x=>Math.exp(x-high)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);};
  const unit=values=>{const norm=Math.hypot(...values);return values.map(x=>x/norm);};
  const round=x=>Number(x.toFixed(4));
  const percent=x=>`${(x*100).toFixed(1)}%`;
  const groups={zero:'Zero-shot',adaptation:'Thích nghi không nhãn',fewshot:'Few-shot',control:'Đối chứng bổ sung'};
  function enrich(methods){
    for(const [key,catalogId,family,kind,modality] of [
      ['stata','stata_image','adaptation','stata','image'],['flat','fewshot_flat','fewshot','flat','image_title'],
      ['cascade','fewshot_cascade','fewshot','cascade','image_title'],['r87','fewshot_cascade_r87','fewshot','r87','image_title']
    ])Object.assign(methods[key],{catalogId,family,kind,modality});
    for(const [key,name,family,kind,modality,note] of entries)methods[key]={name,family,kind,modality,note,signature:note,catalogId:key,steps:kind==='hierarchy'?['Đầu vào','Embedding','Gate không nhãn','Quyết định nhóm','Giới hạn lớp con','Nhãn cuối']:['Đầu vào','Embedding','Biểu diễn lớp','Tính điểm','Tổng hợp','Nhãn cuối']};
    const steps={
      linear:['Đầu vào','Embedding','Head đã học','Logit wᵀh+b','Sigmoid','Nhãn cuối'],
      blend:['Đầu vào','Embedding','Các nhánh riêng','Xác suất từng nhánh','Gộp trọng số','Nhãn cuối'],
      descriptor:['Đầu vào','Embedding','Mô tả đặc trưng','Gộp prototype','Cosine / softmax','Nhãn cuối'],
      waffle:['Đầu vào','Embedding','Prompt ngẫu nhiên','Gộp prototype','Điểm ảnh–lớp','Nhãn cuối'],
      em:['Đầu vào','Embedding','Vector xác suất','EM-Dirichlet','Ghép cụm–lớp','Nhãn cuối'],
      calip:['Đầu vào','Global + patch','Attention','Ba nhánh điểm','Gộp điểm','Nhãn cuối'],
      vote:['Đầu vào','Embedding','Ba bộ dự đoán','Ba lá phiếu','Đếm phiếu','Nhãn cuối'],
      vote6:['Đầu vào','Embedding','Ba bộ sáu lớp','Ba lá phiếu','Đếm phiếu','Nhãn cuối']
    };
    for(const method of Object.values(methods))if(steps[method.kind])method.steps=[...steps[method.kind]];
    return methods;
  }
  function calculation(recipe,c,step,threshold){
    const kind=recipe.kind,id=recipe.catalogId;
    const clip=softmax([41.2,40.8]),title=[.28,.72],stata=[.35,.65];
    let p=clip,formula='s = 100 × uᵀt; p = softmax(s)',numbers='s = [41,2; 40,8]',result=`Real ${percent(clip[0])} · Fake ${percent(clip[1])}`;
    if(recipe.modality==='title') {p=title;numbers='s = [ln(0,28); ln(0,72)]';result='Real 28,0% · Fake 72,0%';}
    if(kind==='linear'||kind==='cascade'||kind==='r87') {
      const logit=Math.log(c.gate/(1-c.gate));p=[1-c.gate,c.gate];formula='a = wᵀh + b; p(Fake) = 1 / (1 + e⁻ᵃ)';
      numbers=`a = ${round(logit)}; sigmoid(${round(logit)}) = ${round(c.gate)}`;result=`Real ${percent(1-c.gate)} · Fake ${percent(c.gate)}`;
    }
    if(kind==='flat'||kind==='stata6'||kind==='vote6'){p=c.flat;formula=kind==='flat'?'s = Wh + b; p = softmax(s)':'Chấm điểm / gán mềm trên cả 6 lớp';numbers='s minh họa = ln(p), nên softmax(s) = p';result=p.map(percent).join(' · ');}
    if(kind==='blend') {
      let weights=[.75,.25],banks=[clip,title],labels=['CLIP ảnh','Tiêu đề'];
      if(id==='clip_stata_image'){weights=[.5,.5];banks=[clip,stata];labels=['CLIP','StatA'];}
      p=[0,1].map(i=>weights.reduce((v,w,j)=>v+w*banks[j][i],0));formula=weights.map((w,j)=>`${w} × ${labels[j]}`).join(' + ');
      numbers=`p(Fake) = ${weights.map((w,j)=>`${w} × ${round(banks[j][1])}`).join(' + ')} = ${round(p[1])}`;result=`Real ${percent(p[0])} · Fake ${percent(p[1])}`;
    }
    if(kind==='descriptor'){formula='t = mean(text_encoder(mô tả lớp)); s = 100 × uᵀt';numbers='“bị cắt ghép”, “đối tượng bị thay đổi”… → nhiều vector → trung bình';}
    if(kind==='waffle'){formula=id==='waffle_unit'?'t = unit(mean(unit(e₁), …, unit(eₙ)))':'t = mean(unit(e₁), …, unit(eₙ))';numbers='e₁ = tên lớp + từ ngẫu nhiên; e₂ = tên lớp + ký tự ngẫu nhiên';}
    if(kind==='legacy'){formula='q = mean(unit(eᵢ)); s = scale × featureᵀq';numbers='Không unit(q) lại; scale ảnh 100, tiêu đề 25; q theo prompt legacy';}
    if(kind==='calip'){p=softmax([1,1.592]);formula='s = s_global + 1,12 × s_text_attention + 0,02 × s_visual_attention';numbers='s minh họa = [1,000; 1,592] → softmax';result=`Real ${percent(p[0])} · Fake ${percent(p[1])}`;}
    if(kind==='em'){p=[.35,.65];formula='zᵢₖ ∝ Dirichlet(p_CLIP,i ; αₖ) × hệ số phân cụm';numbers='Cập nhật α và gán mềm; ghép cụm với lớp bằng Hungarian trên điểm CLIP';result='Ví dụ cụm A được ghép với Fake';}
    if(kind==='vote'||kind==='vote6'){p=kind==='vote6'?[0,0,0,0,1,0]:[0,1];formula='ŷ = majority(ŷ_CLIP, ŷ_EM, ŷ_StatA)';numbers=kind==='vote6'?'Ví dụ: [Manipulated, Real, Manipulated] → 2 phiếu Manipulated':'Ví dụ: [Fake, Real, Fake] → 2 phiếu Fake';result=(kind==='vote6'?'Manipulated':'Fake')+' · số phiếu, không phải xác suất dự đoán';}
    if(kind==='hierarchy'){formula='Gate Real → lớp 0; Gate Fake → argmax score lớp 1…5';numbers=`Điểm gate Fake minh họa ${percent(c.gate)}; không học Logistic Regression ở nhánh này`;p=[1-c.gate,c.gate];result=c.gate>.5?'Fake → chọn subtype':'Real';}
    if(kind==='stata'){
      const final=step>=6,fake=step>=4?(final?.65:.58):.48,factor=(fake*.52)/(.48*(1-fake)),rawFake=.48*factor;
      p=[1-fake,fake];formula='z ∝ prior × exp((L_phân_bố + G_láng_giềng) / 50)';
      numbers=step>=4?`Ví dụ hệ số gộp [1; ${round(factor)}] → [0,52; ${round(rawFake)}] / ${round(.52+rawFake)}`:'prior = softmax(100 × uᵀt) = [0,52; 0,48]';result=`Real ${percent(p[0])} · Fake ${percent(p[1])}`;
      if(step===5){formula='μ = unit(β × μ_query + (1−β) × t); β = n / (1+n)';numbers='Ví dụ n = 4 → β = 0,8: 80% tâm query + 20% neo prompt';result='Cập nhật μ và phương sai; giữ nguyên u';}
    }
    if(step===0){formula=recipe.modality==='image'?'Ảnh I → bộ tiền xử lý ảnh':recipe.modality==='title'?'Tiêu đề t → tokenizer':'(Ảnh I, tiêu đề t) → hai encoder';numbers='Ảnh mẫu train d89ibd; các điểm số bên dưới là minh họa';result=recipe.modality==='image'?'Chỉ ảnh tham gia nhánh này':recipe.modality==='title'?'Chỉ tiêu đề tham gia nhánh này':'Ảnh và tiêu đề cùng tham gia';}
    if(step===1){formula=recipe.modality==='image_title'?(recipe.family==='fewshot'?'u = unit(fI(I)); v = unit(fT(t)); h = [u; v] / √2':'u = unit(fI(I)); v = unit(fT(t)); chấm hai nhánh riêng'):recipe.modality==='image'?'u = unit(fI(I))':'v = unit(fT(t))';numbers='Ví dụ unit([3; 4]) = [0,6; 0,8]; norm = 1';result='Backbone đóng băng; không cập nhật trọng số encoder';}
    if(kind==='r87'&&step===2){formula='k = ceil(0,87 × N_Fake); τ = score_Fake xếp thứ k';numbers='N = 100 → k = 87 → τ ≈ 0,36';result='Khóa ngưỡng trước khi dự đoán query';}
    if((kind==='r87'||kind==='cascade')&&step>=3){formula=kind==='r87'?'Fake nếu p(Fake) ≥ τ; còn lại Real':'Fake nếu p(Fake) > 0,5; bằng nhau chọn Real';numbers=`${round(c.gate)} ${kind==='r87'?(c.gate>=threshold?'≥':'<'):(c.gate>.5?'>':'≤')} ${kind==='r87'?round(threshold):.5}`;result=(kind==='r87'?c.gate>=threshold:c.gate>.5)?'→ Fake, dùng child':'→ Real, không dùng child';if(step>=4&&(kind==='r87'?c.gate>=threshold:c.gate>.5)){numbers=`child = [${c.child.map(round).join('; ')}]`;result='Chọn subtype có điểm cao nhất';}}
    return {formula,numbers,result,p,clip,title,stata};
  }
  window.SimulationRecipes={enrich,calculation,softmax,unit,groups};
})();
