(() => {
  const COLORS = ["#18b76a","#a946d2","#ef633a","#4778e8","#d69b25","#299da4"];
  const DEFAULT_NAMES = ["Class 1","Class 2","Class 3"];
  const MAX_CLASSES = 6;
  const MAX_SAMPLES_PER_CLASS = 250;
  const FEATURE_SIZE = 34;

  let classes = DEFAULT_NAMES.map((name,i)=>({name,color:COLORS[i],samples:[],thumbs:[]}));
  let detector = null, classifier = null, stream = null;
  let cameraOn = false, showSkeleton = true, predictionPaused = false, modelTrained = false;
  let currentPose = null, currentFeature = null, detecting = false;
  let captureTimer = null, activeCapture = -1, training = false;

  const $ = s => document.querySelector(s);
  const learningStack = $("#learningStack"), predictions = $("#predictions");
  const video = $("#video"), canvas = $("#poseCanvas"), ctx = canvas.getContext("2d");
  const camera = $("#camera"), placeholder = $("#placeholder");
  const cameraBtn = $("#cameraBtn"), skeletonBtn = $("#skeletonBtn");
  const detectorStatus = $("#detectorStatus"), poseStatus = $("#poseStatus");
  const trainBtn = $("#trainBtn"), trainProgress = $("#trainProgress"), progressBar = $("#progressBar");
  const epochText = $("#epochText"), lossText = $("#lossText"), learningNote = $("#learningNote");
  const resultDisc = $("#resultDisc"), resultName = $("#resultName"), resultConfidence = $("#resultConfidence");
  const exportModelBtn = $("#exportModelBtn"), pauseBtn = $("#pauseBtn"), toast = $("#toast");

  const bones = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[1,3],[0,2],[2,4]];

  function notify(msg){
    toast.textContent = msg; toast.classList.add("show");
    clearTimeout(toast._t); toast._t = setTimeout(()=>toast.classList.remove("show"),2200);
  }

  function esc(s){ return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c])); }

  function render(){
    learningStack.innerHTML = "";
    predictions.innerHTML = "";
    classes.forEach((c,i)=>{
      const card = document.createElement("div");
      card.className = "class-card";
      card.innerHTML = `
        <div class="class-top">
          <span class="count"><strong id="count-${i}">${c.samples.length}</strong> examples</span>
          <button class="class-name" data-rename="${i}" title="Đổi tên class">${esc(c.name)}</button>
        </div>
        <div class="sample-strip" id="thumbs-${i}"></div>
        <div class="class-bottom">
          <div class="confbox">
            <div class="confhead"><span>Confidence</span><strong id="pct-${i}">0%</strong></div>
            <div class="meter"><div id="fill-${i}" style="background:${c.color}"></div></div>
          </div>
          <button class="capture-btn" data-capture="${i}" style="background:${c.color}">Hold to record</button>
        </div>`;
      learningStack.appendChild(card);
      renderThumbs(i);

      const p = document.createElement("div");
      p.className = "pred";
      p.innerHTML = `<span class="dot" style="background:${c.color}"></span><b>${esc(c.name)}</b><span id="pred-${i}">0%</span>`;
      predictions.appendChild(p);
    });
    bindClassControls();
    $("#addClassBtn").disabled = classes.length >= MAX_CLASSES;
    updateTrainState();
  }

  function renderThumbs(i){
    const host = $("#thumbs-"+i); if(!host) return;
    host.innerHTML = "";
    for(let n=0;n<8;n++){
      const cell = document.createElement("div");
      cell.className = "sample " + (classes[i].thumbs[n] ? "" : "empty");
      if(classes[i].thumbs[n]){
        const cn = document.createElement("canvas"); cn.width=70; cn.height=60; cell.appendChild(cn);
        drawMiniPose(cn.getContext("2d"), classes[i].thumbs[n], 70, 60, classes[i].color);
      }
      host.appendChild(cell);
    }
  }

  function bindClassControls(){
    document.querySelectorAll("[data-rename]").forEach(btn=>{
      btn.addEventListener("click",()=>{
        const i = +btn.dataset.rename;
        const v = prompt("Tên class/tư thế:",classes[i].name);
        if(v && v.trim()){ classes[i].name=v.trim().slice(0,40); render(); }
      });
    });
    document.querySelectorAll("[data-capture]").forEach(btn=>{
      const start = e => { e.preventDefault(); startCapture(+btn.dataset.capture,btn); };
      const stop = e => { if(e)e.preventDefault(); stopCapture(); };
      btn.addEventListener("pointerdown",start);
      btn.addEventListener("pointerup",stop);
      btn.addEventListener("pointercancel",stop);
      btn.addEventListener("pointerleave",stop);
    });
  }

  async function ensureDetector(){
    if(detector) return;
    detectorStatus.textContent = "Loading...";
    await tf.setBackend("webgl").catch(()=>{});
    await tf.ready();
    const model = poseDetection.SupportedModels.MoveNet;
    const config = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING, enableSmoothing: true };
    detector = await poseDetection.createDetector(model, config);
    detectorStatus.textContent = "MoveNet ready";
  }

  async function startCamera(){
    if(cameraOn){ stopCamera(); return; }
    try{
      cameraBtn.disabled=true; cameraBtn.textContent="Starting...";
      if(!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable");
      stream = await navigator.mediaDevices.getUserMedia({ video:{facingMode:"user",width:{ideal:640},height:{ideal:480}}, audio:false });
      video.srcObject = stream; await video.play();
      resizeCanvas(); cameraOn=true; placeholder.style.display="none";
      cameraBtn.textContent="Stop camera"; skeletonBtn.disabled=false;
      await ensureDetector();
      detectionLoop();
    }catch(err){
      console.error(err); notify("Không mở được camera. Hãy cấp quyền camera và dùng HTTPS hoặc localhost.");
      cameraBtn.textContent="Start camera";
    }finally{cameraBtn.disabled=false}
  }

  function stopCamera(){
    stopCapture();
    if(stream) stream.getTracks().forEach(t=>t.stop());
    stream=null; cameraOn=false; video.srcObject=null; currentPose=null; currentFeature=null;
    placeholder.style.display=""; cameraBtn.textContent="Start camera"; skeletonBtn.disabled=true;
    poseStatus.textContent="Not detected"; ctx.clearRect(0,0,canvas.width,canvas.height);
  }

  function resizeCanvas(){
    const r = camera.getBoundingClientRect(), d = Math.min(window.devicePixelRatio||1,2);
    canvas.width = Math.round(r.width*d); canvas.height = Math.round(r.height*d);
    ctx.setTransform(d,0,0,d,0,0);
  }

  function videoPoint(k){
    const vw=video.videoWidth||640, vh=video.videoHeight||480, sw=camera.clientWidth, sh=camera.clientHeight;
    const scale=Math.max(sw/vw,sh/vh), dw=vw*scale, dh=vh*scale, ox=(sw-dw)/2, oy=(sh-dh)/2;
    return {x:ox+k.x*scale,y:oy+k.y*scale};
  }

  function drawPose(pose){
    ctx.clearRect(0,0,camera.clientWidth,camera.clientHeight);
    if(!showSkeleton || !pose?.keypoints) return;
    ctx.lineWidth=3; ctx.strokeStyle="rgba(255,255,255,.92)";
    bones.forEach(([a,b])=>{
      const A=pose.keypoints[a],B=pose.keypoints[b];
      if((A.score||0)>.4 && (B.score||0)>.4){
        const p=videoPoint(A),q=videoPoint(B); ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();
      }
    });
    pose.keypoints.forEach(k=>{
      if((k.score||0)>.4){const p=videoPoint(k);ctx.beginPath();ctx.fillStyle="#39e58d";ctx.arc(p.x,p.y,4.5,0,Math.PI*2);ctx.fill()}
    });
  }

  function drawMiniPose(g,pose,w,h,color){
    g.clearRect(0,0,w,h);g.fillStyle="#f5f3f6";g.fillRect(0,0,w,h);
    const good=pose.keypoints.filter(k=>(k.score||0)>.25); if(!good.length)return;
    const xs=good.map(k=>k.x), ys=good.map(k=>k.y);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const bw=Math.max(1,maxX-minX), bh=Math.max(1,maxY-minY), s=Math.min((w-8)/bw,(h-8)/bh);
    const ox=(w-bw*s)/2-minX*s, oy=(h-bh*s)/2-minY*s, mp=k=>({x:k.x*s+ox,y:k.y*s+oy});
    g.strokeStyle=color;g.lineWidth=2;
    bones.forEach(([a,b])=>{
      const A=pose.keypoints[a],B=pose.keypoints[b];
      if((A.score||0)>.3&&(B.score||0)>.3){const p=mp(A),q=mp(B);g.beginPath();g.moveTo(p.x,p.y);g.lineTo(q.x,q.y);g.stroke()}
    });
    g.fillStyle=color;pose.keypoints.forEach(k=>{if((k.score||0)>.3){const p=mp(k);g.beginPath();g.arc(p.x,p.y,1.5,0,Math.PI*2);g.fill()}});
  }

  function poseToFeature(pose){
    if(!pose?.keypoints || pose.keypoints.length<17) return null;
    const valid=pose.keypoints.filter(k=>(k.score||0)>.25);
    if(valid.length<8) return null;

    const leftShoulder=pose.keypoints[5], rightShoulder=pose.keypoints[6];
    const leftHip=pose.keypoints[11], rightHip=pose.keypoints[12];
    const centers=[leftShoulder,rightShoulder,leftHip,rightHip].filter(k=>(k.score||0)>.25);
    let cx,cy;
    if(centers.length>=2){
      cx=centers.reduce((s,k)=>s+k.x,0)/centers.length;
      cy=centers.reduce((s,k)=>s+k.y,0)/centers.length;
    }else{
      cx=valid.reduce((s,k)=>s+k.x,0)/valid.length;
      cy=valid.reduce((s,k)=>s+k.y,0)/valid.length;
    }

    let scale=0;
    if((leftShoulder.score||0)>.25 && (rightShoulder.score||0)>.25){
      scale=Math.hypot(leftShoulder.x-rightShoulder.x,leftShoulder.y-rightShoulder.y);
    }
    if(scale<20){
      const xs=valid.map(k=>k.x),ys=valid.map(k=>k.y);
      scale=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),60);
    }
    scale=Math.max(scale,30);

    const feat=[];
    pose.keypoints.forEach(k=>{
      const visible=(k.score||0)>.2;
      feat.push(visible?(k.x-cx)/scale:0);
      feat.push(visible?(k.y-cy)/scale:0);
    });
    return feat;
  }

  async function detectionLoop(){
    if(!cameraOn || !detector) return;
    if(!detecting){
      detecting=true;
      try{
        const poses=await detector.estimatePoses(video,{flipHorizontal:true});
        const pose=poses[0]||null;
        currentPose=pose; currentFeature=poseToFeature(pose);
        poseStatus.textContent=currentFeature?"Detected":"Move into frame";
        drawPose(pose);
        if(modelTrained && classifier && currentFeature && !predictionPaused && !training){
          await predict(currentFeature);
        }
      }catch(e){console.error(e)}
      detecting=false;
    }
    requestAnimationFrame(detectionLoop);
  }

  function startCapture(i,btn){
    if(training){notify("Đang huấn luyện, hãy chờ hoàn tất.");return}
    if(!cameraOn || !currentFeature || !currentPose){notify("Bật camera và đứng rõ trong khung hình trước.");return}
    stopCapture(); activeCapture=i; btn.classList.add("active"); addExample();
    captureTimer=setInterval(addExample,150);
  }

  function addExample(){
    if(activeCapture<0 || !currentFeature || !currentPose)return;
    const c=classes[activeCapture];
    if(c.samples.length>=MAX_SAMPLES_PER_CLASS){notify("Đã đạt giới hạn 250 mẫu/class.");stopCapture();return}
    c.samples.push([...currentFeature]);
    if(c.thumbs.length<8) c.thumbs.push(JSON.parse(JSON.stringify(currentPose)));
    else if(c.samples.length%15===0) c.thumbs[(Math.floor(c.samples.length/15)-1)%8]=JSON.parse(JSON.stringify(currentPose));
    $("#count-"+activeCapture).textContent=c.samples.length;
    renderThumbs(activeCapture);
    invalidateModel();
    updateTrainState();
  }

  function stopCapture(){
    if(captureTimer)clearInterval(captureTimer);
    captureTimer=null; activeCapture=-1;
    document.querySelectorAll(".capture-btn.active").forEach(b=>b.classList.remove("active"));
  }

  function invalidateModel(){
    if(modelTrained){
      modelTrained=false; exportModelBtn.disabled=true;
      resultName.textContent="Cần Train lại";resultConfidence.textContent="Dữ liệu đã thay đổi";
      classes.forEach((_,i)=>setConfidence(i,0));
    }
  }

  function updateTrainState(){
    const everyClassReady = classes.length>=2 && classes.every(c=>c.samples.length>=5);
    trainBtn.disabled=training || !everyClassReady;
    if(training) trainBtn.textContent="Training...";
    else trainBtn.textContent=modelTrained?"Train Again":"Train Model";
    learningNote.textContent=!everyClassReady
      ?"Mỗi lớp cần ít nhất 5 mẫu để Train; khuyến nghị 30–60 mẫu/lớp."
      :"Đã đủ để huấn luyện thử. Dữ liệu càng đa dạng, model càng có cơ hội khái quát tốt.";
  }

  function buildClassifier(numClasses){
    const model=tf.sequential();
    model.add(tf.layers.dense({inputShape:[FEATURE_SIZE],units:64,activation:"relu"}));
    model.add(tf.layers.dropout({rate:.15}));
    model.add(tf.layers.dense({units:32,activation:"relu"}));
    model.add(tf.layers.dense({units:numClasses,activation:"softmax"}));
    model.compile({optimizer:tf.train.adam(0.003),loss:"categoricalCrossentropy",metrics:["accuracy"]});
    return model;
  }

  async function trainModel(){
    if(trainBtn.disabled || training)return;
    stopCapture(); training=true; updateTrainState(); trainProgress.style.display="block"; progressBar.style.width="0%";
    try{
      const xs=[],ys=[];
      classes.forEach((c,ci)=>c.samples.forEach(sample=>{xs.push(sample);ys.push(ci)}));
      if(xs.length<10) throw new Error("Not enough samples");

      if(classifier)classifier.dispose();
      classifier=buildClassifier(classes.length);

      const xTensor=tf.tensor2d(xs,[xs.length,FEATURE_SIZE]);
      const yIndex=tf.tensor1d(ys,"int32");
      const yTensor=tf.oneHot(yIndex,classes.length);
      const epochs=Math.min(45,Math.max(18,Math.round(1200/xs.length)+18));

      await classifier.fit(xTensor,yTensor,{
        epochs,
        batchSize:Math.min(32,Math.max(4,Math.floor(xs.length/4))),
        shuffle:true,
        validationSplit:xs.length>=30?0.15:0,
        callbacks:{
          onEpochEnd:async(epoch,logs)=>{
            const pct=Math.round(((epoch+1)/epochs)*100);
            progressBar.style.width=pct+"%";
            epochText.textContent=`Epoch ${epoch+1}/${epochs}`;
            const acc=logs.acc ?? logs.accuracy;
            lossText.textContent=`Loss ${Number(logs.loss).toFixed(3)}${acc!=null?" · Acc "+Math.round(acc*100)+"%":""}`;
            await tf.nextFrame();
          }
        }
      });
      xTensor.dispose();yIndex.dispose();yTensor.dispose();

      modelTrained=true; exportModelBtn.disabled=false; resetOutput();
      notify("Huấn luyện xong. Bây giờ hãy thử tư thế mới.");
    }catch(e){
      console.error(e);notify("Huấn luyện không thành công. Hãy thu thêm dữ liệu rồi thử lại.");
    }finally{
      training=false;updateTrainState();
    }
  }

  async function predict(feature){
    const x=tf.tensor2d([feature],[1,FEATURE_SIZE]);
    const out=classifier.predict(x);
    const probs=Array.from(await out.data());
    x.dispose();out.dispose();
    let best=0;probs.forEach((p,i)=>{setConfidence(i,p);if(p>probs[best])best=i});
    const p=probs[best]||0,c=classes[best];
    resultDisc.style.background=c.color; resultDisc.classList.toggle("live",p>.65);
    resultName.textContent=c.name; resultConfidence.textContent="Confidence "+Math.round(p*100)+"%";
  }

  function setConfidence(i,p){
    const pct=Math.round((p||0)*100);
    const fill=$("#fill-"+i),pt=$("#pct-"+i),pred=$("#pred-"+i);
    if(fill)fill.style.width=pct+"%";if(pt)pt.textContent=pct+"%";if(pred)pred.textContent=pct+"%";
  }

  function resetOutput(){
    resultDisc.style.background="#dedbe2";resultDisc.classList.remove("live");
    resultName.textContent=modelTrained?"Sẵn sàng kiểm thử":"Chưa huấn luyện";
    resultConfidence.textContent="Confidence 0%";classes.forEach((_,i)=>setConfidence(i,0));
  }

  function addClass(){
    if(classes.length>=MAX_CLASSES)return;
    classes.push({name:"Class "+(classes.length+1),color:COLORS[classes.length%COLORS.length],samples:[],thumbs:[]});
    invalidateModel();render();resetOutput();
  }

  function newProject(){
    if(!confirm("Tạo dự án mới và xóa toàn bộ dữ liệu hiện tại?"))return;
    if(classifier){classifier.dispose();classifier=null}
    classes=DEFAULT_NAMES.map((name,i)=>({name,color:COLORS[i],samples:[],thumbs:[]}));
    modelTrained=false;training=false;render();resetOutput();notify("Đã tạo dự án mới.");
  }

  function applyPreset(names){
    if(names.length>MAX_CLASSES)return;
    if(classes.some(c=>c.samples.length) && !confirm("Áp dụng bộ mới sẽ xóa dữ liệu đang thu. Tiếp tục?"))return;
    if(classifier){classifier.dispose();classifier=null}
    classes=names.map((name,i)=>({name,color:COLORS[i],samples:[],thumbs:[]}));
    modelTrained=false;render();resetOutput();$("#ideasDialog").close();notify("Đã tạo các class. HS bắt đầu tự thu dữ liệu.");
  }

  function downloadData(){
    const payload={
      app:"Pose Teachable Lab",version:1,feature:"MoveNet normalized keypoints",
      classes:classes.map(c=>({name:c.name,samples:c.samples}))
    };
    const blob=new Blob([JSON.stringify(payload)],{type:"application/json"});
    const url=URL.createObjectURL(blob),a=document.createElement("a");
    a.href=url;a.download="pose-training-data.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
  }

  async function exportModel(){
    if(!classifier || !modelTrained){notify("Hãy Train Model trước.");return}
    try{
      await classifier.save("downloads://pose-classifier");
      const meta={app:"Pose Teachable Lab",labels:classes.map(c=>c.name),featureSize:FEATURE_SIZE,poseExtractor:"MoveNet SinglePose Lightning"};
      const blob=new Blob([JSON.stringify(meta,null,2)],{type:"application/json"});
      const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="pose-classifier-metadata.json";a.click();
      setTimeout(()=>URL.revokeObjectURL(url),500);
      notify("Đã xuất model và metadata.");
    }catch(e){console.error(e);notify("Trình duyệt không cho tải model.")}
  }

  const presets=[
    {title:"Bộ 1 · Cơ bản",desc:"Dễ quan sát sự khác nhau giữa các class.",names:["Đứng thẳng","Giơ hai tay","Dang hai tay"]},
    {title:"Bộ 2 · Trái / phải",desc:"Phù hợp để thảo luận camera lật và dữ liệu.",names:["Giơ tay trái","Giơ tay phải","Giơ hai tay"]},
    {title:"Bộ 3 · Toàn thân",desc:"Có thay đổi chân và chiều cao cơ thể.",names:["Đứng","Squat","Tạo chữ T"]},
    {title:"Bộ 4 · Dễ gây nhầm",desc:"Tốt để HS cải tiến dataset sau lần train đầu.",names:["Tay trái ngang","Hai tay ngang","Tay phải ngang"]}
  ];
  function renderPresets(){
    const host=$("#presetList");host.innerHTML="";
    presets.forEach((p,i)=>{
      const el=document.createElement("div");el.className="preset";
      el.innerHTML=`<h3>${esc(p.title)}</h3><p>${esc(p.desc)}</p><div class="pills">${p.names.map(n=>`<span class="pill">${esc(n)}</span>`).join("")}</div><button data-preset="${i}">Tạo các class này</button>`;
      host.appendChild(el);
    });
    host.querySelectorAll("[data-preset]").forEach(b=>b.addEventListener("click",()=>applyPreset(presets[+b.dataset.preset].names)));
  }

  cameraBtn.addEventListener("click",startCamera);
  skeletonBtn.addEventListener("click",()=>{showSkeleton=!showSkeleton;skeletonBtn.textContent=showSkeleton?"Hide skeleton":"Show skeleton";if(!showSkeleton)ctx.clearRect(0,0,camera.clientWidth,camera.clientHeight)});
  $("#addClassBtn").addEventListener("click",addClass);
  trainBtn.addEventListener("click",trainModel);
  $("#newProjectBtn").addEventListener("click",newProject);
  $("#downloadDataBtn").addEventListener("click",downloadData);
  exportModelBtn.addEventListener("click",exportModel);
  pauseBtn.addEventListener("click",()=>{predictionPaused=!predictionPaused;pauseBtn.textContent=predictionPaused?"Resume prediction":"Pause prediction"});
  $("#ideasBtn").addEventListener("click",()=>$("#ideasDialog").showModal());
  $("#ideasBtn2").addEventListener("click",()=>$("#ideasDialog").showModal());
  document.querySelectorAll("[data-close]").forEach(b=>b.addEventListener("click",()=>$("#"+b.dataset.close).close()));
  window.addEventListener("resize",()=>cameraOn&&resizeCanvas());
  window.addEventListener("pointerup",stopCapture);
  window.addEventListener("beforeunload",()=>{if(stream)stream.getTracks().forEach(t=>t.stop());if(detector)detector.dispose?.();if(classifier)classifier.dispose()});

  render();renderPresets();resetOutput();
})();
