/* Original educational drawings; inspired by interactive ML explainers, not their code. */
(() => {
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const C={ink:'#192e48',muted:'#63758c',line:'#dce5ee',blue:'#3564df',teal:'#137d72',orange:'#c06532',paper:'#ffffff',soft:'#f3f6fb'};
  const classes=['Real','Satire/Parody','False Connection','Imposter','Manipulated','Misleading'];
  function draw({host,method,recipe,calculation,step,c,threshold,example,points,near}) {
    const W=Math.max(280,Math.round(host.getBoundingClientRect().width || 960));
    const mobile=W<680, H=method==='stata'?(mobile?760:565):(mobile?(calculation?.p?.length===6?1040:960):560);
    const root=n('svg',{viewBox:`0 0 ${W} ${H}`,class:'sim-stage-svg',role:'img','aria-label':method==='stata'?'StatA: ảnh giữ nguyên vị trí; ba nguồn thông tin cập nhật điểm dự đoán.':`${method}: theo đường xử lý sáng màu từ ảnh và tiêu đề đến lớp dự đoán.`});
    root.setAttribute('xmlns',NS);root.setAttribute('font-family','Inter, Arial, sans-serif');
    const motion=!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    function n(tag,attrs={},text) { const e=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e; }
    function add(tag,attrs,text){const e=n(tag,attrs,text);root.append(e);return e;}
    function text(x,y,label,size=14,color=C.ink,anchor='start',weight=400){return add('text',{x,y,fill:color,'font-size':size,'text-anchor':anchor,'font-weight':weight},label);}
    function wrap(x,y,label,maxWidth,size=14,color=C.ink,anchor='start') {
      const words=label.split(' ');let line='',row=0;
      for(const word of words){if((line+' '+word).length>maxWidth/(size*.54)&&line){text(x,y+row*20,line,size,color,anchor);row++;line=word;}else line+=(line?' ':'')+word;}
      if(line)text(x,y+row*20,line,size,color,anchor);
      return (row+1)*20;
    }
    function rect(x,y,w,h,fill=C.paper,stroke=C.line,r=12){return add('rect',{x,y,width:w,height:h,rx:r,fill,stroke,'stroke-width':1});}
    function path(d,color=C.line,active=false,token=false){
      add('path',{d,fill:'none',stroke:color,'stroke-width':active?2.5:1.5,'stroke-linecap':'round'});
      if(token&&motion){const dot=add('circle',{r:5,fill:color,class:'sim-travel-dot'});dot.append(n('animateMotion',{path:d,dur:'1.1s',repeatCount:1,fill:'freeze'}));}
    }
    function arrow(x1,y1,x2,y2,on=false,token=false,color=C.blue){
      const mx=(x1+x2)/2, d=Math.abs(x1-x2)<2?`M${x1} ${y1}L${x2} ${y2}`:`M${x1} ${y1}C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`;
      path(d,on?color:C.line,on,token);
      const down=Math.abs(x1-x2)<2;
      add('path',{d:down?`M${x2-4} ${y2-6}L${x2} ${y2}L${x2+4} ${y2-6}`:`M${x2-6} ${y2-4}L${x2} ${y2}L${x2-6} ${y2+4}`,fill:'none',stroke:on?color:C.line,'stroke-width':1.5});
    }
    function node(x,y,w,h,title,subtitle,on=false,color=C.blue){
      rect(x,y,w,h,on?'#f0f4ff':C.paper,on?color:C.line);
      text(x+w/2,y+29,title,w<130?12:16,on?color:C.muted,'middle',500);
      if(subtitle)wrap(x+w/2,y+51,subtitle,w-16,12,C.muted,'middle');
    }
    function tinyPhoto(x,y,w,h){
      rect(x,y,w,h,'#eaf2f7','none',6);
      add('image',{x,y,width:w,height:h,href:c.image,preserveAspectRatio:'xMidYMid meet',role:'img','aria-label':'Ảnh Fakeddit train d89ibd',opacity:recipe?.modality==='title'?.25:1});
    }
    function post(x,y,w,on=true,showTitle=true){
      rect(x,y,w,showTitle?172:128,C.paper,on?C.blue:C.line);
      tinyPhoto(x+10,y+10,w-20,90,example==='edited');
      text(x+12,y+119,recipe?.modality==='title'?'Chỉ tiêu đề':showTitle?'Ảnh + tiêu đề':'Chỉ ảnh',13,C.ink,'start',500);
      if(showTitle)wrap(x+12,y+141,c.title,w-24,11,C.muted);
    }
    function vector(x,y,w,on){
      node(x,y,w,78,'Embedding','',on);
      for(let i=0;i<10;i++)add('rect',{x:x+12+i*(w-24)/10,y:y+56-((i*7)%13),width:(w-24)/10-4,height:6+((i*7)%13),rx:1,fill:on?C.blue:C.line,opacity:.5});
    }
    function classNode(x,y,w,label,p,show,selected=false,muted=false){
      rect(x,y,w,46,selected?'#eef8f5':C.paper,selected?C.teal:C.line,9);
      text(x+12,y+21,label,mobile?13:14,muted?C.muted:C.ink,'start',selected?500:400);
      text(x+w-12,y+21,show?`${Math.round(p*100)}%`:'—',15,selected?C.teal:C.muted,'end',500);
      if(show){add('rect',{x:x+12,y:y+31,width:w-24,height:4,rx:2,fill:C.soft});add('rect',{x:x+12,y:y+31,width:(w-24)*p,height:4,rx:2,fill:selected?C.teal:C.blue,opacity:muted?.3:1});}
    }
    if(method==='stata') {
      const sx=mobile?20:W*.04, sy=mobile?175:75, sw=mobile?W-40:W*.61, sh=mobile?280:315;
      text(mobile?W/2:sx,28,'41 ảnh → một nhóm không nhãn',16,C.ink,mobile?'middle':'start',500);
      if(!mobile){tinyPhoto(W-190,18,160,84);text(W-110,122,'Theo dõi ảnh A',12,C.muted,'middle');}
      if(mobile){tinyPhoto(W/2-65,48,130,80);text(W/2,150,'Ảnh A trong nhóm',12,C.muted,'middle');}
      const X=v=>sx+v*sw,Y=v=>sy+(1-v)*sh;
      for(let i=1;i<5;i++){add('line',{x1:sx,y1:sy+i*sh/5,x2:sx+sw,y2:sy+i*sh/5,stroke:'#e9eef5'});add('line',{x1:sx+i*sw/5,y1:sy,x2:sx+i*sw/5,y2:sy+sh,stroke:'#e9eef5'});}
      if(step>=2)for(let k=0;k<2;k++){
        const start=k?[.78,.35]:[.22,.65], end=step>=5?(k?[.69,.54]:[.29,.46]):start;
        const ellipse=add('ellipse',{cx:X(end[0]),cy:Y(end[1]),rx:sw*(step>=5?.22:.19),ry:sh*.24,fill:k?C.orange:C.teal,opacity:.08,stroke:k?C.orange:C.teal,'stroke-width':1});
        if(step>=5&&motion)ellipse.append(n('animate',{attributeName:'cx',from:X(start[0]),to:X(end[0]),dur:'.8s',repeatCount:1} ),n('animate',{attributeName:'cy',from:Y(start[1]),to:Y(end[1]),dur:'.8s',repeatCount:1}));
      }
      if(step>=3)near.forEach(({i})=>path(`M${X(.51)} ${Y(.47)}L${X(points[i].x)} ${Y(points[i].y)}`,C.muted,false,step===3));
      points.forEach((p,i)=>{
        const fake=i===40?step>=4:p.k===1;
        const fill=step<2?'#a9b6c6':fake?C.orange:C.teal;
        add('circle',{cx:X(p.x),cy:Y(p.y),r:i===40?8:5.5,fill,'data-image':i+1,opacity:i===40?1:.76});
      });
      add('circle',{cx:X(.51),cy:Y(.47),r:15,fill:'none',stroke:C.ink,'stroke-width':1.5});text(X(.51)+19,Y(.47)-14,'A',17,C.ink,'start',500);
      if(step>=2)for(let k=0;k<2;k++){
        const px=X(k?.78:.22),py=Y(k?.35:.65),color=k?C.orange:C.teal;
        add('path',{d:`M${px} ${py-9}l9 9-9 9-9-9Z`,fill:C.paper,stroke:color,'stroke-width':2});
        text(px,py+(k?28:-19),k?'Prompt Fake':'Prompt Real',12,color,'middle',500);
        if(step>=5){const mx=X(k?.69:.29),my=Y(k?.54:.46);path(`M${px} ${py}L${mx} ${my}`,color);text(mx,my+7,'+',28,color,'middle');}
      }
      const fake=step>=6?.65:step>=4?.58:.48, bx=mobile?20:W*.72, by=mobile?500:170, bw=mobile?W-40:W*.24;
      text(bx,by,step<2?'Chưa có dự đoán':'Điểm của ảnh A',14,C.ink,'start',500);
      ['Real','Fake'].forEach((label,i)=>{
        const yy=by+34+i*55,v=i?fake:1-fake;
        text(bx,yy,label,14,i?C.orange:C.teal);text(bx+bw,yy,step>=2?`${Math.round(v*100)}%`:'—',20,C.ink,'end',500);
        add('rect',{x:bx,y:yy+10,width:bw,height:7,rx:3,fill:C.line});
        if(step>=2)add('rect',{x:bx,y:yy+10,width:bw*v,height:7,rx:3,fill:i?C.orange:C.teal});
      });
      if(step>=6)text(bx,by+170,'→ Fake',26,C.orange,'start',500);
      const ey=mobile?685:450, gap=12, ew=(W-40-gap*2)/3;
      ['Prompt lớp','Phân bố lớp','Láng giềng'].forEach((label,i)=>node(20+i*(ew+gap),ey,ew,65,label,step>=4?(mobile?'Cùng góp':'Cùng góp vào điểm A'):'',step>=[2,2,3][i]));
      text(W/2,mobile?ey-18:ey-28,'Vector ảnh giữ nguyên; điểm và thống kê được cập nhật',mobile?11:13,C.muted,'middle');
      return root;
    }
    if(!['flat','cascade','r87','hierarchy'].includes(method)){
      const iw=mobile?Math.min(W-40,250):W*.19,ix=mobile?(W-iw)/2:20,iy=mobile?20:160;
      post(ix,iy,iw,true,recipe.modality!=='image');
      const vx=mobile?W/2-100:W*.25,vy=mobile?245:207,vw=mobile?200:W*.15;
      vector(vx,vy,vw,step>=1);
      arrow(mobile?W/2:ix+iw,mobile?iy+172:iy+80,mobile?W/2:vx,mobile?vy:vy+39,step>=1,step===1);
      const ex=mobile?20:W*.45,ey=mobile?365:100,ew=mobile?W-40:W*.28;
      rect(ex,ey,ew,260,step>=2?'#f0f4ff':C.paper,step>=2?C.blue:C.line);
      wrap(ex+ew/2,ey+27,recipe.name,ew-20,15,C.blue,'middle');
      let rows=['Prompt Real','Prompt Fake','Text encoder → prototype'];
      if(recipe.kind==='descriptor')rows=['Real: các mô tả lớp Real','→ Gộp riêng thành t_Real','Fake: các mô tả lớp Fake','→ Gộp riêng thành t_Fake'];
      if(recipe.kind==='waffle')rows=['Real + qx7 nzt…','Fake + kp9 vbr…','Nhiều hậu tố ngẫu nhiên',recipe.catalogId==='waffle_unit'?'unit(mean(prompt))':'mean(prompt)'];
      if(recipe.kind==='legacy')rows=['Prompt legacy của từng lớp','Mean riêng cho mỗi lớp','Không unit lại mean'];
      if(recipe.kind==='linear')rows=['h → trọng số w đã học','a = wᵀh + b',`a minh họa = ${Math.log(c.gate/(1-c.gate)).toFixed(3)}`,'sigmoid(a) → Real / Fake'];
      if(recipe.kind==='blend')rows=[recipe.signature,'Tính riêng từng nhánh','Gộp xác suất cùng thứ tự lớp'];
      if(recipe.kind==='em')rows=['Cosine → vector xác suất','Ước lượng α Dirichlet','Cập nhật gán cụm mềm','Hungarian: cụm → lớp'];
      if(recipe.kind==='calip')rows=['Ảnh toàn cục + patch','Attention patch ↔ prompt','3 nhánh điểm bổ sung','s = s₀ + 1,12s₁ + 0,02s₂'];
      if(recipe.kind==='stata6')rows=['6 prototype lớp','Nhóm query không nhãn','Gán mềm + thống kê có neo','Argmax cả 6 lớp'];
      if(recipe.kind==='vote'||recipe.kind==='vote6')rows=recipe.kind==='vote'?['CLIP fusion → Fake','EM → Real','StatA → Fake','2 / 3 phiếu → Fake']:['CLIP → Manipulated','EM → Real','StatA → Manipulated','2 / 3 → Manipulated'];
      rows.forEach((label,i)=>{rect(ex+12,ey+55+i*44,ew-24,35,C.paper,C.line,6);wrap(ex+ew/2,ey+77+i*44,label,ew-35,mobile?12:11,step>=2?C.ink:C.muted,'middle');});
      arrow(mobile?W/2:vx+vw,mobile?vy+78:vy+39,mobile?W/2:ex,mobile?ey:ey+130,step>=2,step===2);
      const p=calculation.p,labels=p.length===6?classes:['Real','Fake'],best=p.indexOf(Math.max(...p));
      const ox=mobile?20:W*.79,oy=mobile?655:p.length===6?45:175,ow=mobile?W-40:W*.19;
      p.forEach((value,i)=>{
        const yy=oy+i*(mobile?42:51);
        if(!mobile)arrow(ex+ew,ey+130,ox,yy+23,step>=3,step===3);
        if(recipe.kind==='vote'||recipe.kind==='vote6'){
          rect(ox,yy,ow,38,i===best&&step>=4?'#eef8f5':C.paper,C.line,7);
          text(ox+10,yy+24,labels[i],12,C.ink);text(ox+ow-10,yy+24,step>=3?`${i===best?2:i===0?1:0} phiếu`:'—',12,C.muted,'end');
        }else classNode(ox,yy,ow,labels[i],value,step>=3,step>=5&&i===best);
      });
      if(step>=5)text(mobile?W/2:ox+ow/2,mobile?940:oy+p.length*51+35,`→ ${labels[best]}`,22,C.teal,'middle',500);
      return root;
    }
    const isFlat=method==='flat',isR87=method==='r87',cutoff=isR87?threshold:.5;
    const prediction=isFlat?c.flat.indexOf(Math.max(...c.flat)):(isR87?c.gate>=cutoff:c.gate>.5)?1+c.child.indexOf(Math.max(...c.child)):0;
    const routedFake=prediction!==0;
    const inputW=mobile?Math.min(W-40,260):W*.19;
    const ix=mobile?(W-inputW)/2:22,iy=mobile?20:165;
    post(ix,iy,inputW,step>=0,recipe?.modality!=='image');
    const vx=mobile?W/2-100:W*.26,vy=mobile?245:212,vw=mobile?200:W*.15;
    vector(vx,vy,vw,step>=1);
    arrow(mobile?W/2:ix+inputW,mobile?iy+172:iy+85,mobile?W/2:vx,mobile?vy:vy+39,step>=1,step===1);
    if(isFlat){
      const cx=mobile?W/2-95:W*.47,cy=mobile?375:215,cw=mobile?190:W*.16;
      node(cx,cy,cw,75,'Head 6 lớp','Logistic Regression',step>=2);
      arrow(mobile?W/2:vx+vw,mobile?vy+78:vy+39,mobile?W/2:cx,mobile?cy:cy+37,step>=2,step===2);
      const ox=mobile?20:W*.71,oy=mobile?510:75,ow=mobile?W-40:W*.26;
      if(mobile)arrow(W/2,cy+75,W/2,oy-10,step>=3,step===3);
      for(let i=0;i<6;i++){
        const yy=oy+i*55;
        if(!mobile)arrow(cx+cw,cy+37,ox,yy+23,step>=3&&(step<4||i===prediction),step===3||(step===4&&i===prediction),i===prediction&&step>=4?C.teal:C.blue);
        classNode(ox,yy,ow,classes[i],c.flat[i],step>=3,step>=4&&prediction===i);
      }
      text(mobile?W/2:cx+cw/2,mobile?485:cy+117,step>=4?'Chọn lớp có điểm cao nhất':'Chấm điểm trực tiếp cả 6 lớp',12,C.muted,'middle');
      if(step>=4)text(mobile?W/2:ox+ow/2,mobile?875:475,`→ ${classes[prediction]}`,24,C.teal,'middle',500);
      return root;
    }
    const gx=mobile?W/2-105:W*.46,gy=mobile?385:190,gw=mobile?210:W*.18;
    node(gx,gy,gw,106,recipe?.kind==='hierarchy'?recipe.signature.replace('Gate ','').slice(0,22):'Gate Real / Fake',step>=3?`Fake ${Math.round(c.gate*100)}%`:recipe?.kind==='hierarchy'?'Không học head train':'Head đã học',step>=2);
    if(step>=3)text(gx+gw/2,gy+83,`Ngưỡng ${Math.round(cutoff*100)}%`,13,C.ink,'middle',500);
    arrow(mobile?W/2:vx+vw,mobile?vy+78:vy+39,mobile?W/2:gx,mobile?gy:gy+53,step>=2,step===2);
    const rx=mobile?15:W*.76,ry=mobile?555:65,rw=mobile?(W-45)/2:W*.21;
    const fx=mobile?W/2+8:W*.76,fy=mobile?555:295,fw=mobile?(W-45)/2:W*.21;
    arrow(mobile?gx+30:gx+gw,mobile?gy+106:gy+30,mobile?rx+rw/2:rx,mobile?ry:ry+32,step>=4&&!routedFake,step===4&&!routedFake,C.teal);
    arrow(mobile?gx+gw-30:gx+gw,mobile?gy+106:gy+78,mobile?fx+fw/2:fx,mobile?fy:fy+32,step>=4&&routedFake,step===4&&routedFake,C.orange);
    node(rx,ry,rw,65,'Real',step>=4&&!routedFake?'Trả kết quả':'',step>=4&&!routedFake,C.teal);
    node(fx,fy,fw,65,'Fake',step>=4&&routedFake?'Chuyển sang bộ phân loại con':'',step>=4&&routedFake,C.orange);
    const childY=mobile?665:385, childX=mobile?15:W*.70,childW=mobile?W-30:W*.27;
    rect(childX,childY,childW,mobile?160:140,C.paper,step>=4&&routedFake?C.orange:C.line);
    text(childX+12,childY+25,recipe?.kind==='hierarchy'?'Chọn trong 5 score Fake':'Head con · 5 loại Fake',14,routedFake&&step>=4?C.ink:C.muted,'start',500);
    if(step>=4&&routedFake){
      const best=c.child.indexOf(Math.max(...c.child));
      c.child.forEach((v,i)=>{
        const yy=childY+47+i*(mobile?22:18);
        text(childX+12,yy,classes[i+1],mobile?12:11,i===best?C.orange:C.muted);
        const start=childX+childW*.58,barW=childW*.22;
        add('rect',{x:start,y:yy-8,width:barW*v,height:6,rx:2,fill:i===best?C.orange:C.line});
        text(childX+childW-10,yy,`${Math.round(v*100)}%`,11,C.muted,'end');
      });
      if(mobile)arrow(fx+fw/2,fy+65,childX+childW*.75,childY,true,step===4,C.orange);
      else arrow(fx+fw/2,fy+65,fx+fw/2,childY,true,step===4,C.orange);
    }else wrap(childX+childW/2,childY+65,step>=4?'Không đi vào nhánh này':'Chờ kết quả Real/Fake',childW-28,13,C.muted,'middle');
    if(isR87&&step>=2){
      const tx=mobile?20:25,ty=mobile?872:415,tw=mobile?W-40:W*.58;
      text(tx,ty,'R87 · đổi ngưỡng, giữ nguyên model',13,C.ink,'start',500);
      if(step===2)text(tx,ty-26,'Mục tiêu: nhận ra 87 / 100 Fake',17,C.blue,'start',500);
      const y=ty+22;add('rect',{x:tx,y,width:tw,height:8,rx:4,fill:'#e4eaf3'});
      const scoreX=tx+c.gate*tw,newX=tx+threshold*tw,oldX=tx+.5*tw;
      add('line',{x1:oldX,y1:y-7,x2:oldX,y2:y+16,stroke:C.muted,'stroke-width':1.5,'stroke-dasharray':'3 2'});
      const line=add('line',{x1:newX,y1:y-7,x2:newX,y2:y+16,stroke:C.blue,'stroke-width':3});
      if(step===2&&motion){line.append(n('animate',{attributeName:'x1',from:oldX,to:newX,dur:'.8s',repeatCount:1}),n('animate',{attributeName:'x2',from:oldX,to:newX,dur:'.8s',repeatCount:1}));}
      if(step>=3)add('circle',{cx:scoreX,cy:y+4,r:7,fill:C.orange});
      text(newX,y+35,'36%',12,C.blue,'middle');text(oldX,y+35,'50%',12,C.muted,'middle');
      if(step>=3)text(tx,ty-23,`${Math.round(c.gate*100)}% ${c.gate>=threshold?'≥':'<'} 36% → ${c.gate>=threshold?'Fake':'Real'}`,18,C.orange,'start',500);
    }
    return root;
  }
  window.MethodStage={draw};
})();
