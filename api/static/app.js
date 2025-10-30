const $ = (id)=>document.getElementById(id);
const file = $("file"), send = $("send"), drop = $("drop");
const preview = $("preview"), resBox = $("result"), errBox = $("error");
const badge = $("badge"), score = $("score"), thr = $("threshold"), lat = $("latency"), ver = $("version");

let currentFile = null;

function setError(msg){
  errBox.textContent = msg ?? "";
  errBox.classList.toggle("hidden", !msg);
}
function setPreview(fileObj){
  if(!fileObj) { preview.style.display="none"; preview.src=""; return; }
  const url = URL.createObjectURL(fileObj);
  preview.src = url; preview.style.display = "block";
}

file.addEventListener("change", (e)=>{
  const f = e.target.files?.[0];
  currentFile = f || null;
  setError(null);
  setPreview(currentFile);
  send.disabled = !currentFile;
});

["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev, (e)=>{e.preventDefault(); drop.classList.add("drag");}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev, (e)=>{e.preventDefault(); drop.classList.remove("drag");}));
drop.addEventListener("drop", (e)=>{
  const f = e.dataTransfer.files?.[0];
  if(f){ file.files = e.dataTransfer.files; file.dispatchEvent(new Event("change")); }
});

send.addEventListener("click", async ()=>{
  if(!currentFile) return;
  setError(null); resBox.classList.add("hidden"); send.disabled = true; send.textContent = "Verificando...";
  try{
    const fd = new FormData(); fd.append("image", currentFile, currentFile.name);
    const t0 = performance.now();
    const r = await fetch("/verify",{ method:"POST", body:fd });
    const txt = await r.text();
    let data;
    try { data = JSON.parse(txt); } catch{ throw new Error(txt); }
    if(!r.ok && data?.error) throw new Error(data.error);
    const t1 = performance.now();

    score.textContent = (data.score ?? 0).toFixed(4);
    thr.textContent = (data.threshold ?? 0).toString();
    lat.textContent = (data.timing_ms ?? (t1 - t0)).toFixed(2) + " ms";
    ver.textContent = data.model_version ?? "n/a";
    badge.className = "badge " + (data.is_me ? "ok" : "bad");
    badge.textContent = data.is_me ? "✅ ERES TÚ" : "⛔ NO ERES TÚ";
    resBox.classList.remove("hidden");
  }catch(err){
    setError(err.message || String(err));
  }finally{
    send.disabled = false; send.textContent = "Verificar";
  }
});

// precargar healthz para mostrar threshold y versión por defecto
(async ()=>{
  try{
    const r = await fetch("/healthz"); const d = await r.json();
    thr.textContent = d.threshold?.toString() ?? "—";
    ver.textContent = d.model_version ?? "—";
  }catch{}
})();
