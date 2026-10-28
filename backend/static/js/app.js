/**
 * CrimeVision AI - Frontend Controller & 3D WebGL Visualization Engine
 * Orchestrates RT-DETR visual canvas, Whisper audio playback, Three.js 3D scene,
 * Chronological master timeline, Groq AI evidence chat, and PDF reporting.
 */

// Global State
const state = {
  currentCaseId: "CASE-2026-0842",
  caseData: null,
  activeDetections: [],
  selectedDetectionId: null,
  bboxFilters: { person: true, weapon: true, forensic_object: true, metadata: true },
  audioPlaying: false,
  audioDuration: 24.0,
  audioCurrentTime: 0.0,
  chatHistory: [],
  scene3D: {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    markers: [],
    animId: null
  }
};

// DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initTabs();
  initCaseSelector();
  initAudioPlayer();
  initChat();
  initUpload();
  loadCase(state.currentCaseId);
});

// HUD Clock
function initClock() {
  const clockEl = document.getElementById("hud-clock");
  if (!clockEl) return;
  setInterval(() => {
    const now = new Date();
    clockEl.textContent = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
  }, 1000);
}

// Tab Switching
function initTabs() {
  const tabBtns = document.querySelectorAll(".nav-tab-item");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
        if (pane.id === `tab-${targetTab}`) {
          pane.classList.add("active");
        }
      });

      // Resize 3D scene if opening 3D tab
      if (targetTab === "3d-reconstruction") {
        setTimeout(handle3DResize, 100);
      }
      // Redraw canvas if opening studio
      if (targetTab === "multimodal-studio") {
        setTimeout(drawBBoxes, 100);
      }
    });
  });
}

// Case Selector
function initCaseSelector() {
  const selector = document.getElementById("case-select");
  if (!selector) return;

  fetch("/api/cases")
    .then(r => r.json())
    .then(data => {
      selector.innerHTML = "";
      data.cases.forEach(c => {
        const opt = document.createElement("option");
        opt.value = c.case_id;
        opt.textContent = `${c.case_id} - ${c.title.substring(0, 32)}...`;
        selector.appendChild(opt);
      });
      selector.value = state.currentCaseId;
    })
    .catch(err => console.error("Error loading case list:", err));

  selector.addEventListener("change", (e) => {
    state.currentCaseId = e.target.value;
    loadCase(state.currentCaseId);
  });
}

// Load Case Data & AI Engine Results
async function loadCase(caseId) {
  showToast(`Loading forensic evidence graph for ${caseId}...`);
  try {
    const res = await fetch(`/api/cases/${caseId}`);
    if (!res.ok) throw new Error("Failed to load case data");
    const data = await res.json();
    state.caseData = data;
    state.activeDetections = data.rt_detr.detections;

    renderCaseOverview(data.case, data.reconstruction);
    renderStudioMedia(data);
    renderDetectionsList(data.rt_detr);
    renderClipBlip(data.vision_language);
    renderAudioForensics(data.whisper);
    renderTimeline(data.fusion.timeline);
    renderReconstruction(data.reconstruction);
    renderScene3D(data.scene_3d);

    showToast(`Case ${caseId} analysis active (RT-DETR + Whisper + Groq).`, "success");
  } catch (err) {
    console.error("Load case error:", err);
    showToast("Error loading case evidence: " + err.message, "danger");
  }
}

// Render Overview Banner
function renderCaseOverview(c, recon) {
  document.getElementById("banner-case-id").textContent = c.case_id;
  document.getElementById("banner-title").textContent = c.title;
  document.getElementById("banner-incident-time").textContent = c.incident_date;
  document.getElementById("banner-location").textContent = c.location;
  document.getElementById("banner-investigator").textContent = c.lead_investigator || "Special Agent R. Vance";
  document.getElementById("banner-sha256").textContent = c.sha256_hash ? c.sha256_hash.substring(0, 16) + "..." : "e3b0c442...";

  const threatBadge = document.getElementById("banner-threat-badge");
  const crimeCls = recon?.crime_classification;
  if (threatBadge && crimeCls) {
    threatBadge.textContent = crimeCls.statutory_charge || c.incident_type;
  }
}

// Studio Media Viewport & RT-DETR Bounding Boxes
function renderStudioMedia(data) {
  const imgEl = document.getElementById("evidence-img");
  const cctvLabel = document.getElementById("cctv-label");

  const imgEvid = data.case.evidence.find(e => e.type === "image");
  if (imgEvid && imgEl) {
    imgEl.src = imgEvid.file_path;
    cctvLabel.textContent = imgEvid.label;
    imgEl.onload = () => drawBBoxes();
  }

  // Set up bounding box toggle buttons
  const chips = document.querySelectorAll(".toggle-chip");
  chips.forEach(chip => {
    chip.onclick = () => {
      const category = chip.getAttribute("data-cat");
      chip.classList.toggle("active");
      state.bboxFilters[category] = chip.classList.contains("active");
      drawBBoxes();
    };
  });
}

function drawBBoxes() {
  const canvas = document.getElementById("bbox-canvas");
  const img = document.getElementById("evidence-img");
  if (!canvas || !img || !img.complete || img.naturalWidth === 0) return;

  const rect = img.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  state.activeDetections.forEach(det => {
    // Check if category is enabled in filters
    if (state.bboxFilters[det.category] === false) return;

    const [ymin, xmin, ymax, xmax] = det.bbox_norm;
    const x = xmin * canvas.width;
    const y = ymin * canvas.height;
    const w = (xmax - xmin) * canvas.width;
    const h = (ymax - ymin) * canvas.height;

    const isSelected = det.detection_id === state.selectedDetectionId;
    const boxColor = det.color || (det.category === "weapon" ? "#dc2626" : "#06b6d4");

    // Box stroke
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeStyle = boxColor;
    ctx.strokeRect(x, y, w, h);

    // Glowing fill on hover/select
    if (isSelected) {
      ctx.fillStyle = "rgba(6, 182, 212, 0.25)";
      ctx.fillRect(x, y, w, h);
    } else {
      ctx.fillStyle = "rgba(0, 0, 0, 0.15)";
      ctx.fillRect(x, y, w, h);
    }

    // Corner HUD Brackets
    const blen = Math.min(12, w / 4, h / 4);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#ffffff";
    // Top-left
    ctx.beginPath(); ctx.moveTo(x, y + blen); ctx.lineTo(x, y); ctx.lineTo(x + blen, y); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(x + w - blen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + blen); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(x, y + h - blen); ctx.lineTo(x, y + h); ctx.lineTo(x + blen, y + h); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(x + w - blen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - blen); ctx.stroke();

    // Label banner
    const labelText = `${det.label} (${Math.round(det.confidence * 100)}%)`;
    ctx.font = "bold 11px Inter, sans-serif";
    const textWidth = ctx.measureText(labelText).width;

    ctx.fillStyle = boxColor;
    ctx.fillRect(x, Math.max(0, y - 18), textWidth + 8, 18);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(labelText, x + 4, Math.max(13, y - 5));
  });
}

// Window resize updates canvas size
window.addEventListener("resize", () => {
  drawBBoxes();
  handle3DResize();
});

// Render Detections List
function renderDetectionsList(rtDetr) {
  const container = document.getElementById("detections-list-container");
  if (!container) return;

  container.innerHTML = "";
  rtDetr.detections.forEach(det => {
    const item = document.createElement("div");
    item.className = "detection-item";
    if (det.detection_id === state.selectedDetectionId) item.classList.add("selected");

    const badgeColor = det.category === "weapon" ? "badge-crimson" :
                       det.category === "person" ? "badge-cyan" : "badge-emerald";

    item.innerHTML = `
      <div class="detection-item-header">
        <div class="detection-label">
          <span class="badge-tag ${badgeColor}">${det.category}</span>
          <span>${det.label}</span>
        </div>
        <div class="detection-conf">${Math.round(det.confidence * 100)}% Conf</div>
      </div>
      <div class="detection-attr">${Object.values(det.attributes || {})[0] || ""}</div>
    `;

    item.onmouseenter = () => {
      state.selectedDetectionId = det.detection_id;
      drawBBoxes();
    };
    item.onmouseleave = () => {
      state.selectedDetectionId = null;
      drawBBoxes();
    };

    container.appendChild(item);
  });
}

// Render CLIP & BLIP/BLIP-2
function renderClipBlip(vl) {
  // CLIP probabilities
  const clipList = document.getElementById("clip-prob-list");
  if (clipList) {
    clipList.innerHTML = "";
    vl.clip_engine.probability_distribution.forEach(item => {
      const row = document.createElement("div");
      row.style.marginBottom = "0.5rem";
      row.innerHTML = `
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:2px;">
          <span style="color:var(--text-main); font-weight:600;">${item.category}</span>
          <span style="font-family:var(--font-mono); color:var(--cyan-primary);">${Math.round(item.score * 100)}%</span>
        </div>
        <div style="width:100%; height:4px; background:rgba(255,255,255,0.08); border-radius:2px; overflow:hidden;">
          <div style="width:${item.score * 100}%; height:100%; background:linear-gradient(90deg, var(--cyan-primary), #6366f1);"></div>
        </div>
      `;
      clipList.appendChild(row);
    });
  }

  // BLIP narrative
  const blipNarrative = document.getElementById("blip-narrative-text");
  if (blipNarrative) {
    blipNarrative.textContent = vl.blip_engine.detailed_narrative;
  }
}

// Audio Player & Waveform
function initAudioPlayer() {
  const audio = document.getElementById("forensic-audio");
  const playBtn = document.getElementById("audio-play-btn");
  const timeLabel = document.getElementById("audio-time-label");
  const waveformWrap = document.getElementById("waveform-container");

  if (!audio || !playBtn) return;

  playBtn.onclick = () => {
    if (audio.paused) {
      audio.play();
      state.audioPlaying = true;
      playBtn.innerHTML = `<span>⏸</span> Pause Audio`;
    } else {
      audio.pause();
      state.audioPlaying = false;
      playBtn.innerHTML = `<span>▶</span> Play Evidence Audio`;
    }
  };

  audio.ontimeupdate = () => {
    state.audioCurrentTime = audio.currentTime;
    const dur = audio.duration || state.audioDuration;
    const pct = (audio.currentTime / dur) * 100;

    const scrubber = document.getElementById("waveform-scrubber");
    if (scrubber) scrubber.style.left = `${pct}%`;

    const m = Math.floor(audio.currentTime / 60);
    const s = Math.floor(audio.currentTime % 60);
    const ms = Math.floor((audio.currentTime % 1) * 100);
    timeLabel.textContent = `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(2,'0')} / 00:24.00`;
  };

  waveformWrap.onclick = (e) => {
    const rect = waveformWrap.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = clickX / rect.width;
    const dur = audio.duration || state.audioDuration;
    audio.currentTime = pct * dur;
  };
}

function renderAudioForensics(whisper) {
  const audio = document.getElementById("forensic-audio");
  const audEvid = state.caseData?.case?.evidence?.find(e => e.type === "audio");
  if (audEvid && audio) {
    audio.src = audEvid.file_path;
  }

  // Draw waveform
  drawWaveform(whisper.waveform_metrics.waveform_points);

  // Acoustic Pins
  const pinContainer = document.getElementById("acoustic-pins-container");
  if (pinContainer) {
    pinContainer.innerHTML = "";
    whisper.acoustic_forensics.events.forEach(ev => {
      const pin = document.createElement("div");
      pin.className = "acoustic-pin";
      const pct = (ev.time_sec / 24.0) * 100;
      pin.style.left = `${pct}%`;
      pin.style.color = ev.type === "GUNSHOT_SIGNATURE" ? "#ef4444" : "#f59e0b";
      pin.title = `${ev.timestamp} - ${ev.label} (${ev.peak_db} dB)`;
      pin.onclick = (e) => {
        e.stopPropagation();
        if (audio) {
          audio.currentTime = ev.time_sec;
          audio.play();
        }
      };
      pinContainer.appendChild(pin);
    });
  }

  // Render Transcript segments
  const transContainer = document.getElementById("transcript-segments-list");
  if (transContainer) {
    transContainer.innerHTML = "";
    whisper.transcription.segments.forEach(seg => {
      const row = document.createElement("div");
      row.style.padding = "0.5rem 0";
      row.style.borderBottom = "1px solid var(--border-dim)";
      row.style.fontSize = "0.78rem";
      row.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
          <span style="font-weight:700; color:var(--cyan-primary);">${seg.speaker}</span>
          <span style="font-family:var(--font-mono); color:var(--text-dim);">00:${seg.start.toFixed(2)}</span>
        </div>
        <div style="color:var(--text-main); font-style:italic;">"${seg.text}"</div>
      `;
      transContainer.appendChild(row);
    });
  }
}

function drawWaveform(points) {
  const canvas = document.getElementById("waveform-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.offsetWidth;
  const h = canvas.height = canvas.offsetHeight;

  ctx.clearRect(0, 0, w, h);
  if (!points || points.length === 0) return;

  const barWidth = w / points.length;
  points.forEach((val, i) => {
    const barHeight = val * (h * 0.9);
    const x = i * barWidth;
    const y = (h - barHeight) / 2;

    // Highlight gunshot burst around 6.8s (points ~ 34)
    if (i >= 32 && i <= 38) {
      ctx.fillStyle = "#ef4444";
    } else {
      ctx.fillStyle = "rgba(6, 182, 212, 0.75)";
    }
    ctx.fillRect(x, y, barWidth - 1, barHeight);
  });
}

// Master Chronological Timeline
function renderTimeline(timelineItems) {
  const container = document.getElementById("timeline-flow-container");
  if (!container) return;

  container.innerHTML = "";
  timelineItems.forEach(item => {
    const node = document.createElement("div");
    node.className = "timeline-node";
    if (item.corroboration) node.classList.add("corroborated");

    const corrobHtml = item.corroboration ? `
      <div class="corroboration-badge">
        <span>⚡ MULTI-MODAL CORROBORATED</span>
        <span>(${item.corroboration.courtroom_admissibility})</span>
      </div>
    ` : "";

    node.innerHTML = `
      <div class="timeline-node-pin"></div>
      <div class="timeline-header">
        <span class="timeline-timecode">${item.timestamp_str}</span>
        <span class="badge-tag ${item.source_modality === 'video_vision' ? 'badge-cyan' : item.source_modality === 'audio_acoustic' ? 'badge-amber' : 'badge-crimson'}">${item.category}</span>
      </div>
      <div class="timeline-title">${item.title}</div>
      <div class="timeline-desc">${item.description}</div>
      ${corrobHtml}
    `;

    container.appendChild(node);
  });

  // Timeline Filters
  const filterBtns = document.querySelectorAll(".timeline-filter-btn");
  filterBtns.forEach(btn => {
    btn.onclick = () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const f = btn.getAttribute("data-filter");

      document.querySelectorAll(".timeline-node").forEach(node => {
        if (f === "all") {
          node.style.display = "block";
        } else if (f === "corroborated") {
          node.style.display = node.classList.contains("corroborated") ? "block" : "none";
        }
      });
    };
  });
}

// AI Forensic Reconstruction
function renderReconstruction(recon) {
  const scen = recon?.scene_reconstruction || {};
  document.getElementById("recon-primary-hyp").textContent = scen.primary_hypothesis || "Analyzing evidentiary matrix...";
  document.getElementById("recon-secondary-hyp").textContent = scen.secondary_hypothesis || "Secondary hypothesis formulation underway...";
  document.getElementById("recon-discrepancies").textContent = scen.discrepancies || "Zero critical discrepancies detected.";

  const leadsList = document.getElementById("recon-leads-list");
  if (leadsList && recon.investigative_leads) {
    leadsList.innerHTML = "";
    recon.investigative_leads.forEach((lead, i) => {
      const li = document.createElement("li");
      li.style.marginBottom = "0.45rem";
      li.innerHTML = `<strong>Lead #${i+1}:</strong> ${lead}`;
      leadsList.appendChild(li);
    });
  }
}

// ==========================================================================
// 3D Crime Scene Reconstruction Viewport (Three.js WebGL)
// ==========================================================================

function renderScene3D(sceneData) {
  const container = document.getElementById("three-canvas-container");
  if (!container || !window.THREE) return;

  // Clean up previous instance
  if (state.scene3D.renderer) {
    cancelAnimationFrame(state.scene3D.animId);
    container.innerHTML = "";
  }

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  // 1. Scene & Camera
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060911);
  scene.fog = new THREE.FogExp2(0x060911, 0.04);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
  camera.position.set(-5, 4.5, -5);

  // 2. Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // 3. Controls (OrbitControls)
  let controls = null;
  if (window.THREE.OrbitControls) {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.05; // don't go below floor
    controls.target.set(0, 1.0, 0);
  }

  // 4. Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
  scene.add(ambientLight);

  const ceilingSpot = new THREE.SpotLight(0x38bdf8, 2.5);
  ceilingSpot.position.set(0, 3.2, 0);
  ceilingSpot.angle = Math.PI / 3;
  ceilingSpot.penumbra = 0.6;
  ceilingSpot.castShadow = true;
  scene.add(ceilingSpot);

  const crimePointLight = new THREE.PointLight(0xef4444, 1.8, 8);
  crimePointLight.position.set(-0.8, 1.4, -0.4);
  scene.add(crimePointLight);

  // 5. Build Room Architectural Elements
  const gridHelper = new THREE.GridHelper(16, 16, 0x06b6d4, 0x1e293b);
  gridHelper.position.y = 0.01;
  scene.add(gridHelper);

  sceneData.room_elements.forEach(elem => {
    const [w, h, d] = elem.size;
    const [x, y, z] = elem.pos;
    const geom = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({
      color: elem.color ? parseInt(elem.color.replace("#", "0x")) : 0x334155,
      roughness: 0.4,
      metalness: 0.2
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
  });

  // 6. 3D Evidence Markers with Glowing Halos
  const markerMeshes = [];
  sceneData.evidence_markers.forEach(mkr => {
    const [x, y, z] = mkr.pos;
    const mkrColor = parseInt(mkr.color.replace("#", "0x"));

    // Core sphere
    const sphereGeom = new THREE.SphereGeometry(0.12, 16, 16);
    const sphereMat = new THREE.MeshBasicMaterial({ color: mkrColor });
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.position.set(x, y, z);
    sphere.userData = mkr;

    // Glowing wireframe ring
    const ringGeom = new THREE.RingGeometry(0.18, 0.22, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: mkrColor, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(x, y, z);

    scene.add(sphere);
    scene.add(ring);
    markerMeshes.push(sphere);
  });

  // 7. Suspect Trajectory Path Spline
  if (sceneData.suspect_trajectory && sceneData.suspect_trajectory.waypoints) {
    const points = sceneData.suspect_trajectory.waypoints.map(wp => new THREE.Vector3(wp.pos[0], wp.pos[1] + 0.08, wp.pos[2]));
    const curve = new THREE.CatmullRomCurve3(points);
    const tubeGeom = new THREE.TubeGeometry(curve, 64, 0.04, 8, false);
    const tubeMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const tube = new THREE.Mesh(tubeGeom, tubeMat);
    scene.add(tube);
  }

  // 8. Ballistic Trajectory Line (Laser Vector)
  if (sceneData.ballistic_trajectory) {
    const o = sceneData.ballistic_trajectory.origin;
    const d = sceneData.ballistic_trajectory.destination;
    const lineMat = new THREE.LineDashedMaterial({
      color: 0xf43f5e,
      linewidth: 3,
      scale: 1,
      dashSize: 0.2,
      gapSize: 0.1
    });
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(o[0], o[1], o[2]),
      new THREE.Vector3(d[0], d[1], d[2])
    ]);
    const laserLine = new THREE.Line(lineGeom, lineMat);
    laserLine.computeLineDistances();
    scene.add(laserLine);
  }

  state.scene3D = { scene, camera, renderer, controls, markers: markerMeshes, animId: null };

  // Animation Loop
  function animate() {
    state.scene3D.animId = requestAnimationFrame(animate);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Setup Camera Switcher buttons
  init3DCameraButtons(sceneData.camera_presets);
}

function handle3DResize() {
  const container = document.getElementById("three-canvas-container");
  const s = state.scene3D;
  if (!container || !s.renderer || !s.camera) return;
  const w = container.clientWidth;
  const h = container.clientHeight;
  s.camera.aspect = w / h;
  s.camera.updateProjectionMatrix();
  s.renderer.setSize(w, h);
}

function init3DCameraButtons(presets) {
  const toolbar = document.getElementById("scene-camera-toolbar");
  if (!toolbar || !presets) return;

  toolbar.innerHTML = "";
  presets.forEach(p => {
    const btn = document.createElement("button");
    btn.className = "btn btn-secondary";
    btn.style.fontSize = "0.72rem";
    btn.style.padding = "0.25rem 0.65rem";
    btn.textContent = p.label;
    btn.onclick = () => {
      const cam = state.scene3D.camera;
      const ctrl = state.scene3D.controls;
      if (cam) {
        cam.position.set(p.pos[0], p.pos[1], p.pos[2]);
        if (ctrl && p.look_at) {
          ctrl.target.set(p.look_at[0], p.look_at[1], p.look_at[2]);
          ctrl.update();
        }
      }
    };
    toolbar.appendChild(btn);
  });
}

// ==========================================================================
// Investigator AI Evidence Chat
// ==========================================================================

function initChat() {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send-btn");
  const chipBtns = document.querySelectorAll(".query-chip");

  if (!input || !sendBtn) return;

  const handleSend = () => {
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    sendChatMessage(q);
  };

  sendBtn.onclick = handleSend;
  input.onkeydown = (e) => {
    if (e.key === "Enter") handleSend();
  };

  chipBtns.forEach(chip => {
    chip.onclick = () => {
      input.value = chip.textContent;
      handleSend();
    };
  });
}

async function sendChatMessage(query) {
  const msgContainer = document.getElementById("chat-messages-container");
  if (!msgContainer) return;

  // Append user bubble
  appendChatBubble("user", query);

  // Append typing indicator
  const typingBubble = appendChatBubble("assistant", "Analyzing forensic evidence graph and timestamps...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        case_id: state.currentCaseId,
        history: state.chatHistory
      })
    });
    const data = await res.json();

    // Replace typing bubble
    typingBubble.innerHTML = formatMarkdown(data.answer);

    // Citations
    if (data.citations && data.citations.length > 0) {
      const citWrap = document.createElement("div");
      citWrap.style.marginTop = "0.5rem";
      data.citations.forEach(c => {
        const pill = document.createElement("span");
        pill.className = "chat-citation-pill";
        pill.textContent = `📍 [${c.id}] ${c.timestamp}`;
        citWrap.appendChild(pill);
      });
      typingBubble.appendChild(citWrap);
    }

    state.chatHistory.push({ role: "user", content: query });
    state.chatHistory.push({ role: "assistant", content: data.answer });
    msgContainer.scrollTop = msgContainer.scrollHeight;
  } catch (err) {
    typingBubble.textContent = "Error communicating with forensic AI: " + err.message;
  }
}

function appendChatBubble(role, text) {
  const container = document.getElementById("chat-messages-container");
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.innerHTML = role === "user" ? escapeHtml(text) : formatMarkdown(text);
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

// Markdown formatter helper
function formatMarkdown(str) {
  if (!str) return "";
  let s = escapeHtml(str);
  s = s.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*(.*?)\*/g, "<em>$1</em>");
  s = s.replace(/`([^`]+)`/g, "<code style='background:rgba(255,255,255,0.1); padding:1px 4px; border-radius:3px; font-family:var(--font-mono); font-size:0.8em;'>$1</code>");
  s = s.replace(/\n/g, "<br/>");
  return s;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}

// ==========================================================================
// Courtroom PDF Report Generation
// ==========================================================================

async function generatePDFReport() {
  showToast("Compiling Courtroom-Grade Forensic PDF Report...", "info");
  try {
    const res = await fetch(`/api/reports/generate/${state.currentCaseId}`, { method: "POST" });
    if (!res.ok) throw new Error("PDF generation failed");
    const data = await res.json();

    showToast("PDF Report compiled! Downloading...", "success");
    window.open(data.download_url, "_blank");
  } catch (err) {
    showToast("Report generation error: " + err.message, "danger");
  }
}

// ==========================================================================
// Evidence File Upload & Pipeline Ingest
// ==========================================================================

function initUpload() {
  const dropZone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("evidence-file-input");
  const typeSelect = document.getElementById("evidence-type-select");

  if (!dropZone || !fileInput) return;

  dropZone.onclick = () => fileInput.click();

  dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--cyan-primary)";
  };
  dropZone.ondragleave = () => {
    dropZone.style.borderColor = "var(--border-dim)";
  };
  dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--border-dim)";
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileUpload();
    }
  };

  fileInput.onchange = () => handleFileUpload();
}

async function handleFileUpload() {
  const fileInput = document.getElementById("evidence-file-input");
  const typeSelect = document.getElementById("evidence-type-select");
  if (!fileInput.files || fileInput.files.length === 0) return;

  const file = fileInput.files[0];
  const type = typeSelect ? typeSelect.value : "image";

  showToast(`Uploading and calculating SHA-256 for ${file.name}...`, "info");

  const formData = new FormData();
  formData.append("file", file);
  formData.append("case_id", state.currentCaseId);
  formData.append("evidence_type", type);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Upload failed");
    const data = await res.json();

    showToast(`Uploaded ${file.name}. SHA-256: ${data.sha256.substring(0, 12)}...`, "success");
    // Reload active case to refresh pipeline
    loadCase(state.currentCaseId);
  } catch (err) {
    showToast("Upload error: " + err.message, "danger");
  }
}

// ==========================================================================
// Project Epics & Tasks Modal
// ==========================================================================

function openProjectStatsModal() {
  const modal = document.getElementById("project-stats-modal");
  if (modal) modal.classList.add("active");
}

function closeProjectStatsModal() {
  const modal = document.getElementById("project-stats-modal");
  if (modal) modal.classList.remove("active");
}

// Groq Key Settings Modal
function openGroqModal() {
  const modal = document.getElementById("groq-key-modal");
  if (modal) modal.classList.add("active");
}

function closeGroqModal() {
  const modal = document.getElementById("groq-key-modal");
  if (modal) modal.classList.remove("active");
}

async function saveGroqKey() {
  const input = document.getElementById("groq-api-key-input");
  if (!input) return;
  const key = input.value.trim();

  try {
    const res = await fetch("/api/settings/groq-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key })
    });
    const data = await res.json();
    showToast(data.message, data.status === "success" ? "success" : "warning");
    closeGroqModal();
  } catch (err) {
    showToast("Failed to save Groq Key: " + err.message, "danger");
  }
}

// UI Notification Toast
function showToast(msg, type = "info") {
  let toast = document.getElementById("hud-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "hud-toast";
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      padding: 0.65rem 1.15rem;
      border-radius: var(--radius-md);
      font-size: 0.82rem;
      font-weight: 600;
      z-index: 999;
      box-shadow: var(--shadow-card);
      transition: var(--transition);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    `;
    document.body.appendChild(toast);
  }

  if (type === "success") {
    toast.style.background = "#064e3b";
    toast.style.color = "#34d399";
    toast.style.border = "1px solid #10b981";
  } else if (type === "danger") {
    toast.style.background = "#7f1d1d";
    toast.style.color = "#f87171";
    toast.style.border = "1px solid #ef4444";
  } else {
    toast.style.background = "#0c1e3a";
    toast.style.color = "#38bdf8";
    toast.style.border = "1px solid #06b6d4";
  }

  toast.textContent = msg;
  toast.style.opacity = "1";
  toast.style.transform = "translateY(0)";

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(8px)";
  }, 4000);
}
