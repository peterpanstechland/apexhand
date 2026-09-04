const $ = (id) => document.getElementById(id);
let SCHEMA = null;
let currentJob = null;
let chart = null;
let lastCode = "";
let pollTimer = null;
let sandboxTimer = null;
let sandboxPaused = false;
let sandboxReady = false;
let sandboxCmdTimer = null;
let sandboxTargets = {};
let sandboxAppliedDefault = false;
let sandboxCam = { yaw: 0, pitch: 0.6, distance: 0.16, target: [-0.17, 0.0, 0.488] };
let sandboxCamReady = false;

function fillSelect(el, items, labelKey = "label", idKey = "id") {
  el.innerHTML = items.map((it) => `<option value="${it[idKey]}">${it[labelKey]}</option>`).join("");
}

function paramVisible(p, task) {
  if (!p.tasks || p.tasks.length === 0) return true;
  return p.tasks.includes(task);
}

function renderParams() {
  const task = $("task").value;
  const root = $("param-groups");
  root.innerHTML = "";
  for (const group of SCHEMA.groups) {
    const params = SCHEMA.params.filter((p) => p.group === group && paramVisible(p, task));
    if (!params.length) continue;
    const box = document.createElement("div");
    box.className = "group";
    box.innerHTML = `<h3>${group}</h3><div class="body"></div>`;
    const body = box.querySelector(".body");
    for (const p of params) {
      body.appendChild(renderParam(p));
    }
    box.querySelector("h3").onclick = () => {
      body.style.display = body.style.display === "none" ? "" : "none";
    };
    root.appendChild(box);
  }
}

function renderParam(p) {
  const wrap = document.createElement("div");
  wrap.className = "param";
  wrap.dataset.id = p.id;
  wrap.dataset.path = p.path;
  wrap.dataset.kind = p.kind;
  wrap.dataset.ptype = p.ptype;
  let control = "";
  const def = p.default;
  if (p.ptype === "select") {
    control = `<select data-id="${p.id}">${p.options.map((o) => `<option ${o === def ? "selected" : ""}>${o}</option>`).join("")}</select>`;
  } else if (p.ptype === "bool") {
    control = `<input type="checkbox" data-id="${p.id}" ${def ? "checked" : ""} />`;
  } else if (p.ptype === "float_pair" || p.ptype === "int_list") {
    control = `<input type="text" data-id="${p.id}" value="${Array.isArray(def) ? def.join(", ") : def}" />`;
  } else if (p.ptype === "str") {
    control = `<input type="text" data-id="${p.id}" value="${def ?? ""}" />`;
  } else {
    const min = p.min ?? 0;
    const max = p.max ?? 1;
    const step = p.step ?? 1;
    control = `<input type="range" data-id="${p.id}" min="${min}" max="${max}" step="${step}" value="${def}" /><span class="val">${def}</span>`;
  }
  wrap.innerHTML = `<div class="head"><span class="name">${p.label}</span><span class="info" title="解释">解释</span></div>
    ${control}
    <div class="explain">
      <div>${p.summary}</div>
      <div><b>调大：</b>${p.bigger}</div>
      <div><b>调小：</b>${p.smaller}</div>
      <div><b>新手：</b>${p.novice}</div>
      <div><code>${p.path}</code></div>
    </div>`;
  const info = wrap.querySelector(".info");
  const expl = wrap.querySelector(".explain");
  info.onclick = () => expl.classList.toggle("open");
  const slider = wrap.querySelector("input[type=range]");
  if (slider) {
    const val = wrap.querySelector(".val");
    slider.oninput = () => {
      val.textContent = slider.value;
    };
  }
  return wrap;
}

function readParam(wrap) {
  const ptype = wrap.dataset.ptype;
  const input = wrap.querySelector("input, select");
  if (!input) return null;
  if (ptype === "bool") return input.checked;
  if (ptype === "int") return parseInt(input.value, 10);
  if (ptype === "float") return parseFloat(input.value);
  if (ptype === "float_pair" || ptype === "int_list") {
    return input.value.split(/[,\s]+/).filter(Boolean).map((x) => (ptype === "int_list" ? parseInt(x, 10) : parseFloat(x)));
  }
  return input.value;
}

function writeParam(id, value) {
  const wrap = document.querySelector(`.param[data-id="${id}"]`);
  if (!wrap) return;
  const input = wrap.querySelector("input, select");
  if (!input) return;
  if (input.type === "checkbox") input.checked = !!value;
  else if (Array.isArray(value)) input.value = value.join(", ");
  else input.value = value;
  const val = wrap.querySelector(".val");
  if (val) val.textContent = input.value;
}

function collectRecipe() {
  const taskSel = $("task");
  const task = taskSel.value;
  const meta = SCHEMA.tasks.find((t) => t.id === task) || {};
  const cli = {};
  const overrides = {};
  for (const wrap of document.querySelectorAll(".param")) {
    const value = readParam(wrap);
    if (value === null || Number.isNaN(value)) continue;
    if (wrap.dataset.kind === "cli") {
      const key = wrap.dataset.path.replace(/^--/, "");
      cli[key] = value;
    } else {
      overrides[wrap.dataset.path] = value;
    }
  }
  return {
    name: undefined,
    task,
    play_task: meta.play_id,
    mode: "train",
    physics: $("physics").value,
    object: $("object").value,
    hand_pose: $("hand_pose").value,
    cameras: $("cameras").value,
    cli,
    overrides,
  };
}

function applyRecipe(r) {
  if (r.task) $("task").value = r.task;
  if (r.physics) $("physics").value = r.physics;
  if (r.object) $("object").value = r.object;
  if (r.hand_pose) $("hand_pose").value = r.hand_pose;
  if (r.cameras) $("cameras").value = r.cameras;
  renderParams();
  const cli = r.cli || {};
  if (cli.num_envs != null) writeParam("num_envs", cli.num_envs);
  if (cli.max_iterations != null) writeParam("max_iterations", cli.max_iterations);
  if (cli.seed != null) writeParam("seed", cli.seed);
  for (const [path, value] of Object.entries(r.overrides || {})) {
    const wrap = document.querySelector(`.param[data-path="${path}"]`);
    if (wrap) writeParam(wrap.dataset.id, value);
  }
}

function showAlerts(alerts, where = "alerts") {
  $(where).innerHTML = (alerts || [])
    .map((a) => `<div class="alert ${a.level}">${a.message}</div>`)
    .join("");
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw data.detail || data || res.statusText;
  return data;
}

async function refreshVram() {
  const rec = collectRecipe();
  const width = rec.overrides["env.scene.wrist_camera.width"] || 128;
  const data = await api("/api/vram", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      num_envs: rec.cli.num_envs || 512,
      cameras: rec.cameras,
      width,
      height: width,
    }),
  });
  $("vram-hint").textContent = data.message;
  $("vram-hint").style.color = data.ok ? "var(--ok)" : "var(--danger)";
}

async function startMode(mode) {
  const recipe = collectRecipe();
  recipe.mode = mode;
  if (mode === "play") recipe.cli.video = true;
  if (mode === "export_mjcf") recipe.cli.mjcf_out = "assets/mjcf/apex_hand.xml";
  if (mode === "preview") recipe.cli.preview_out = "logs/webui/preview/scene.png";
  if (mode === "export_onnx") {
    $("export-out").textContent = "正在导出最新 checkpoint → policy.onnx … 日志见底部。";
  }
  if (mode === "sandbox") {
    sandboxReady = false;
    $("sandbox-status").textContent = "正在启动 Kit（大约 30–90 秒），日志在底部。";
    switchTab("playhand");
  }
  const data = await api("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(recipe),
  });
  currentJob = data.job.id;
  showAlerts(data.alerts);
  $("run-meta").textContent = `${data.job.mode} · ${data.job.id} · ${recipe.task}`;
  $("log-status").textContent = `${data.job.mode} · 启动中`;
  await refreshRunList();
  $("log-run").value = currentJob;
  startPolling();
}

async function refreshLog() {
  if (!currentJob) return;
  const job = await api(`/api/runs/${currentJob}`);
  const log = await api(`/api/runs/${currentJob}/log?tail=400`);
  const box = $("log");
  const follow = $("log-follow").checked;
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.textContent = log.text || "(还没有输出)";
  if (follow || atBottom) box.scrollTop = box.scrollHeight;
  const line = `${job.mode || ""} · ${job.status} · pid ${job.pid || "-"}`;
  $("run-meta").textContent = line;
  $("log-status").textContent = line;
  return job;
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    if (!currentJob) return;
    try {
      const job = await refreshLog();
      if (job.status === "running") {
        const m = await api(`/api/runs/${currentJob}/metrics`);
        drawChart(m.series);
        showAlerts(m.alerts, "sentry");
        renderVideos(m.videos);
      }
      if (job.status === "succeeded" && (job.mode === "export_onnx" || job.mode === "export_mjcf")) {
        refreshExports();
      }
      if (job.mode === "sandbox" && (job.status === "failed" || job.status === "stopped" || job.status === "orphaned")) {
        sandboxReady = false;
        $("sandbox-status").textContent = `仿真已结束（${job.status}）。看底部日志。`;
      }
    } catch (err) {
      console.warn(err);
    }
  }, 1500);
}

async function refreshRunList() {
  const data = await api("/api/runs");
  const sel = $("log-run");
  const prev = sel.value;
  sel.innerHTML = (data.runs || [])
    .map((r) => `<option value="${r.id}">${r.id} · ${r.mode || ""} · ${r.status || ""}</option>`)
    .join("");
  if (currentJob) sel.value = currentJob;
  else if (prev) sel.value = prev;
}

async function refreshExports() {
  try {
    const data = await api("/api/exports");
    if (!data.exports || !data.exports.length) {
      $("export-out").textContent = "还没有 policy.onnx。训完后点顶栏「导出 ONNX」。";
      return;
    }
    $("export-out").innerHTML = data.exports
      .slice(0, 5)
      .map((e) => `ONNX <code>${e.onnx}</code>${e.map ? ` · map <code>${e.map}</code>` : ""}`)
      .join("<br/>");
  } catch (err) {
    console.warn(err);
  }
}

function pickSeries(series) {
  const keys = Object.keys(series);
  const prefer = ["mean_reward", "Episode_Reward", "success", "entropy"];
  const chosen = [];
  for (const p of prefer) {
    const hit = keys.find((k) => k.toLowerCase().includes(p.toLowerCase()));
    if (hit) chosen.push(hit);
  }
  return chosen.slice(0, 4);
}

function drawChart(series) {
  if (!window.Chart || !series) return;
  const names = pickSeries(series);
  $("chart-legend").textContent = names.join("  ·  ");
  const datasets = names.map((name, i) => ({
    label: name,
    data: (series[name] || []).map((p) => ({ x: p.x, y: p.y })),
    borderColor: ["#d4a24c", "#6ea8fe", "#6fbf7a", "#d4655a"][i % 4],
    tension: 0.2,
    pointRadius: 0,
  }));
  if (chart) chart.destroy();
  chart = new Chart($("chart"), {
    type: "line",
    data: { datasets },
    options: {
      parsing: false,
      responsive: true,
      scales: { x: { type: "linear", title: { display: true, text: "iter" } } },
      plugins: { legend: { labels: { color: "#9aa0ad" } } },
    },
  });
}

function renderVideos(videos) {
  if (!videos || !videos.length) return;
  $("video-list").innerHTML = videos
    .map((v) => `<a href="#" data-rel="${v.rel}">${v.rel}</a>`)
    .join("<br/>");
  $("video-list").querySelectorAll("a").forEach((a) => {
    a.onclick = (ev) => {
      ev.preventDefault();
      $("player").src = `/api/runs/${currentJob}/video?rel=${encodeURIComponent(a.dataset.rel)}`;
    };
  });
}

async function loadRecipes() {
  const data = await api("/api/recipes");
  fillSelect($("recipe-list"), data.recipes.map((r) => ({ id: r.name, label: r.name })));
}

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.add("hidden"));
  const pane = $(`tab-${name}`);
  if (pane) pane.classList.remove("hidden");
}

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.onclick = () => switchTab(btn.dataset.tab);
});

function renderSandbox() {
  const box = SCHEMA.sandbox || {};
  $("sandbox-presets").innerHTML = (box.presets || [])
    .map((p) => `<button class="ghost" data-preset="${p.id}" title="${p.hint || ""}">${p.label}</button>`)
    .join("");
  $("sandbox-presets").querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => applySandboxPreset(btn.dataset.preset);
  });
  const grid = $("sandbox-joints");
  grid.innerHTML = "";
  for (const finger of box.fingers || []) {
    const card = document.createElement("div");
    card.className = "finger-card";
    card.innerHTML = `<h4>${finger.label}</h4>`;
    for (const joint of finger.joints) {
      const row = document.createElement("div");
      row.className = "joint-row";
      const mid = (joint.min + joint.max) / 2;
      sandboxTargets[joint.id] = 0;
      row.innerHTML = `<span>${joint.label}</span>
        <input type="range" min="${joint.min}" max="${joint.max}" step="${joint.step}" value="0" data-joint="${joint.id}" />
        <span class="deg">0°</span>`;
      const slider = row.querySelector("input");
      const deg = row.querySelector(".deg");
      slider.oninput = () => {
        const value = parseFloat(slider.value);
        sandboxTargets[joint.id] = value;
        deg.textContent = `${value.toFixed(1)}°`;
        queueSandboxCommand({ targets_deg: { ...sandboxTargets } });
      };
      void mid;
      card.appendChild(row);
    }
    grid.appendChild(card);
  }
}

function setSandboxSliders(targets) {
  for (const [id, value] of Object.entries(targets || {})) {
    sandboxTargets[id] = value;
    const slider = document.querySelector(`#sandbox-joints input[data-joint="${id}"]`);
    if (!slider) continue;
    slider.value = value;
    const deg = slider.parentElement.querySelector(".deg");
    if (deg) deg.textContent = `${Number(value).toFixed(1)}°`;
  }
}

function sandboxEye() {
  const cp = Math.cos(sandboxCam.pitch);
  const sp = Math.sin(sandboxCam.pitch);
  const cy = Math.cos(sandboxCam.yaw);
  const sy = Math.sin(sandboxCam.yaw);
  return [
    sandboxCam.target[0] + sandboxCam.distance * cp * cy,
    sandboxCam.target[1] + sandboxCam.distance * cp * sy,
    sandboxCam.target[2] + sandboxCam.distance * sp,
  ];
}

function sendSandboxCamera() {
  $("sandbox-view").value = "viewport";
  queueSandboxCommand({
    view: "viewport",
    camera: {
      yaw: sandboxCam.yaw,
      pitch: sandboxCam.pitch,
      distance: sandboxCam.distance,
      target: sandboxCam.target,
      eye: sandboxEye(),
    },
  });
}

function adoptSandboxCamera(cam) {
  if (!cam) return;
  if (cam.yaw != null) sandboxCam.yaw = cam.yaw;
  if (cam.pitch != null) sandboxCam.pitch = cam.pitch;
  if (cam.distance != null) sandboxCam.distance = cam.distance;
  if (Array.isArray(cam.target) && cam.target.length === 3) sandboxCam.target = cam.target.slice();
  sandboxCamReady = true;
}

function bindSandboxOrbit() {
  const stage = $("sandbox-stage");
  if (!stage || stage.dataset.bound) return;
  stage.dataset.bound = "1";
  let dragging = false;
  let mode = "orbit";
  let lastX = 0;
  let lastY = 0;
  stage.addEventListener("pointerdown", (ev) => {
    if (!sandboxReady || !sandboxCamReady) return;
    if (ev.button === 2 || ev.shiftKey) mode = "pan";
    else if (ev.button === 0) mode = "orbit";
    else return;
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    stage.classList.add("dragging");
    stage.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  });
  stage.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - lastX;
    const dy = ev.clientY - lastY;
    lastX = ev.clientX;
    lastY = ev.clientY;
    if (mode === "orbit") {
      sandboxCam.yaw += dx * 0.008;
      sandboxCam.pitch = Math.max(-1.35, Math.min(1.35, sandboxCam.pitch - dy * 0.008));
    } else {
      const right = [-Math.sin(sandboxCam.yaw), Math.cos(sandboxCam.yaw), 0];
      const up = [
        -Math.cos(sandboxCam.yaw) * Math.sin(sandboxCam.pitch),
        -Math.sin(sandboxCam.yaw) * Math.sin(sandboxCam.pitch),
        Math.cos(sandboxCam.pitch),
      ];
      const scale = sandboxCam.distance * 0.0022;
      sandboxCam.target = [
        sandboxCam.target[0] - right[0] * dx * scale + up[0] * dy * scale,
        sandboxCam.target[1] - right[1] * dx * scale + up[1] * dy * scale,
        sandboxCam.target[2] - right[2] * dx * scale + up[2] * dy * scale,
      ];
    }
    sendSandboxCamera();
  });
  const endDrag = () => {
    dragging = false;
    stage.classList.remove("dragging");
  };
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);
  stage.addEventListener(
    "wheel",
    (ev) => {
      if (!sandboxReady || !sandboxCamReady) return;
      ev.preventDefault();
      sandboxCam.distance = Math.max(0.04, Math.min(0.9, sandboxCam.distance * (ev.deltaY > 0 ? 1.08 : 0.92)));
      sendSandboxCamera();
    },
    { passive: false },
  );
  stage.addEventListener("contextmenu", (ev) => ev.preventDefault());
  stage.addEventListener("dblclick", () => {
    if (!sandboxReady) return;
    sandboxCamReady = false;
    api("/api/sandbox/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset_view: true, view: "viewport" }),
    }).catch((err) => console.warn(err));
  });
}

function queueSandboxCommand(payload) {
  if (sandboxCmdTimer) clearTimeout(sandboxCmdTimer);
  sandboxCmdTimer = setTimeout(() => {
    api("/api/sandbox/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch((err) => console.warn(err));
  }, 80);
}

async function applySandboxPreset(id) {
  const preset = (SCHEMA.sandbox.presets || []).find((p) => p.id === id);
  if (!preset) return;
  if (id === "reset") {
    await api("/api/sandbox/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset: true, targets_deg: {} }),
    });
    return;
  }
  if (preset.targets) {
    setSandboxSliders(preset.targets);
    await api("/api/sandbox/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targets_deg: preset.targets }),
    });
  }
}

function startSandboxPolling() {
  if (sandboxTimer) clearInterval(sandboxTimer);
  sandboxTimer = setInterval(async () => {
    try {
      const state = await api("/api/sandbox/state");
      if (state.views && state.views.length) {
        const sel = $("sandbox-view");
        const labels = Object.fromEntries((SCHEMA.sandbox.views || []).map((v) => [v.id, v.label]));
        const prev = sel.value;
        sel.innerHTML = state.views.map((id) => `<option value="${id}">${labels[id] || id}</option>`).join("");
        if (state.views.includes(prev)) sel.value = prev;
        else if (state.view) sel.value = state.view;
      }
      if (state.ready) {
        sandboxReady = true;
        const obj = state.object_xyz ? ` · 物体 (${state.object_xyz.join(", ")})` : "";
        $("sandbox-status").textContent = `${state.message || "运行中"} · ${state.fps || 0} fps${obj}`;
        const frame = $("sandbox-frame");
        frame.onload = () => {
          frame.style.display = "block";
        };
        frame.onerror = () => {
          frame.style.display = "none";
        };
        frame.src = `/api/sandbox/frame?t=${Date.now()}`;
        if (state.camera && !sandboxCamReady) adoptSandboxCamera(state.camera);
        if (state.default_deg && !sandboxAppliedDefault) {
          setSandboxSliders(state.default_deg);
          sandboxAppliedDefault = true;
        }
      } else {
        $("sandbox-status").textContent = state.message || "等待仿真…";
        $("sandbox-frame").style.display = "none";
      }
    } catch (err) {
      console.warn(err);
    }
  }, 400);
}

$("task").onchange = () => {
  renderParams();
  refreshVram();
};
$("cameras").onchange = refreshVram;

$("btn-start").onclick = () => startMode("train").catch((e) => showAlerts([{ level: "error", message: JSON.stringify(e) }]));
$("btn-play").onclick = () => startMode("play").catch((e) => showAlerts([{ level: "error", message: JSON.stringify(e) }]));
$("btn-preview").onclick = async () => {
  await startMode("preview");
  setTimeout(async () => {
    $("preview-img").src = `/api/preview/image?t=${Date.now()}`;
  }, 8000);
};
$("btn-sandbox").onclick = () => startSandbox();
$("btn-sandbox-start").onclick = () => startSandbox();
$("btn-sandbox-reset").onclick = async () => {
  sandboxAppliedDefault = false;
  await applySandboxPreset("reset");
};
$("btn-sandbox-pause").onclick = async () => {
  sandboxPaused = !sandboxPaused;
  $("btn-sandbox-pause").textContent = sandboxPaused ? "继续" : "暂停";
  await api("/api/sandbox/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pause: sandboxPaused }),
  });
};
$("btn-sandbox-stop").onclick = async () => {
  await api("/api/sandbox/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stop: true }),
  }).catch(() => {});
  if (currentJob) await api(`/api/runs/${currentJob}`, { method: "DELETE" }).catch(() => {});
  sandboxReady = false;
  $("sandbox-status").textContent = "已关闭仿真。";
};
$("sandbox-view").onchange = () => {
  queueSandboxCommand({ view: $("sandbox-view").value });
};
$("btn-sandbox-view-reset").onclick = async () => {
  sandboxCamReady = false;
  await api("/api/sandbox/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reset_view: true, view: "viewport" }),
  });
};
async function startSandbox() {
  sandboxAppliedDefault = false;
  sandboxCamReady = false;
  sandboxPaused = false;
  $("btn-sandbox-pause").textContent = "暂停";
  startSandboxPolling();
  try {
    await startMode("sandbox");
  } catch (e) {
    showAlerts([{ level: "error", message: typeof e === "string" ? e : JSON.stringify(e) }]);
    $("sandbox-status").textContent = typeof e === "string" ? e : JSON.stringify(e);
  }
}
$("btn-probe").onclick = () => startMode("probe");
$("btn-sim2sim").onclick = () => startMode("sim2sim");
$("btn-mjcf").onclick = () => startMode("export_mjcf");
$("btn-export").onclick = () => startMode("export_onnx").catch((e) => showAlerts([{ level: "error", message: JSON.stringify(e) }]));
$("btn-export-tools").onclick = () => $("btn-export").click();
$("btn-log-clear").onclick = () => {
  $("log").textContent = "";
};
$("log-run").onchange = () => {
  currentJob = $("log-run").value;
  refreshLog().catch(() => {});
};
$("btn-stop").onclick = async () => {
  if (currentJob) await api(`/api/runs/${currentJob}`, { method: "DELETE" });
};
$("btn-preflight").onclick = async () => {
  const data = await api("/api/preflight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectRecipe()),
  });
  showAlerts(data.alerts);
};
$("btn-save-recipe").onclick = async () => {
  const name = prompt("配方名字", "my_recipe");
  if (!name) return;
  const rec = collectRecipe();
  rec.name = name;
  await api("/api/recipes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rec),
  });
  loadRecipes();
};
$("btn-load-recipe").onclick = async () => {
  const data = await api("/api/recipes");
  const found = data.recipes.find((r) => r.name === $("recipe-list").value);
  if (found) applyRecipe(found);
};

$("btn-llm-save").onclick = async () => {
  await api("/api/llm/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      base_url: $("llm-url").value,
      model: $("llm-model").value,
      api_key: $("llm-key").value,
    }),
  });
};
$("btn-generate").onclick = async () => {
  $("reward-diff").textContent = "生成中…";
  try {
    const data = await api("/api/llm/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: $("reward-prompt").value, task: $("task").value }),
    });
    lastCode = data.code;
    const extra = data.static && !data.static.ok ? `\n静态检查:\n${data.static.errors.join("\n")}\n` : "";
    $("reward-diff").textContent = extra + (data.diff || data.code);
  } catch (err) {
    $("reward-diff").textContent = typeof err === "string" ? err : JSON.stringify(err);
  }
};
$("btn-apply").onclick = async () => {
  if (!lastCode) return;
  const data = await api("/api/llm/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: lastCode, task: "PAN-CoinHold-Apex-Play-v0", live_check: true }),
  });
  if (data.live_job) {
    currentJob = data.live_job.id;
    startPolling();
  }
  $("reward-diff").textContent = "已写入 user_rewards.py。自检任务已启动，权重仍是 0，用「用户自定义奖励权重」打开。";
};

(async function init() {
  SCHEMA = await api("/api/schema");
  fillSelect($("task"), SCHEMA.tasks);
  fillSelect($("physics"), SCHEMA.physics);
  fillSelect($("hand_pose"), SCHEMA.hand_poses);
  fillSelect($("cameras"), SCHEMA.cameras);
  fillSelect($("object"), SCHEMA.objects);
  renderParams();
  renderSandbox();
  bindSandboxOrbit();
  startSandboxPolling();
  await loadRecipes();
  await refreshVram();
  const cfg = await api("/api/llm/config");
  $("llm-url").value = cfg.base_url || "";
  $("llm-model").value = cfg.model || "";
  await refreshRunList();
  await refreshExports();
  const runs = await api("/api/runs");
  if (runs.runs && runs.runs[0]) {
    currentJob = runs.runs[0].id;
    $("log-run").value = currentJob;
    startPolling();
  }
})();
