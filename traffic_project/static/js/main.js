// main.js - chart helpers + prediction form logic
const COLORS={Low:'#16a34a',Medium:'#eab308',High:'#f97316',Severe:'#dc2626'};
const charts={};
function mk(id,cfg){if(charts[id])charts[id].destroy();charts[id]=new Chart(id,cfg);}
function barChart(id,labels,data,label,horizontal=false){
  const colors=labels.map(l=>COLORS[l]||'#3b82f6');
  mk(id,{type:'bar',data:{labels,datasets:[{label,data,backgroundColor:colors}]},
    options:{indexAxis:horizontal?'y':'x',plugins:{legend:{display:false}}}});
}
function lineChart(id,labels,data,label){
  mk(id,{type:'line',data:{labels,datasets:[{label,data,borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,.15)',fill:true,tension:.3}]}});
}
function setupRoads(areaId,roadId,roads){
  const a=document.getElementById(areaId),r=document.getElementById(roadId);
  const fill=()=>{r.innerHTML=roads[a.value].map(x=>`<option>${x}</option>`).join('');};
  a.onchange=fill;fill();
}
async function showResult(fd){
  const body=Object.fromEntries(fd.entries());
  const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await res.json();
  if(d.error){alert(d.error);return;}
  document.getElementById('out').classList.remove('hidden');
  const lvl=document.getElementById('lvl');lvl.textContent=d.level;lvl.className='badge big '+d.level;
  const al=document.getElementById('alert');al.className='alert '+d.alert.type;
  al.textContent=`${d.alert.message} (model confidence ${d.alert.confidence}%)`;
  if(document.getElementById('p')&&document.getElementById('p').offsetParent!==null)
    barChart('p',Object.keys(d.probabilities),Object.values(d.probabilities),'%');
  if(document.getElementById('fx')&&document.getElementById('fx').offsetParent!==null){
    mk('fx',{type:'bar',data:{labels:d.factors.map(f=>f.factor),datasets:[{label:'Effect on congestion risk',
      data:d.factors.map(f=>f.effect),backgroundColor:d.factors.map(f=>f.effect>=0?'#dc2626':'#16a34a')}]},
      options:{indexAxis:'y',plugins:{legend:{display:false}}}});}
  const rb=document.getElementById('routeBox');
  if(d.routes.length){
    rb.innerHTML='<h3>Alternative route suggestions</h3><small>Model/system suggestions based on historical patterns for the same time and weather. Not real-time navigation. Alternatives assume no road-specific incident or roadwork.</small>'+
      d.routes.map(r=>`<div class="route"><div><b>${r.road}</b><br><small>${r.area}${r.same_area?' (same area)':''} · ~${r.improvement}% lower risk</small></div><span class="badge ${r.level}">${r.level}</span></div>`).join('');
  } else if(d.level==='High'||d.level==='Severe'){
    rb.innerHTML='<h3>Alternative route suggestions</h3><p>No clearly better road found in the dataset for these conditions.</p>';
  } else {rb.innerHTML='<h3>Route recommendation</h3><p>Not needed: congestion is '+d.level+'. Alternatives are shown only for High or Severe.</p>';}
}
