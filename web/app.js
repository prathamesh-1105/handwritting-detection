/* ==========================================================================
   NEUROSCRIBE LSTM - CLIENT WORKSTATION LOGIC
   Deterministic Canvas Input, Inference Client, and Sequence Visualizer
   ========================================================================== */

// Canvas State
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 20;
let undoStack = [];
const MAX_UNDO = 15;

const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const brushSlider = document.getElementById("brush-slider");
const brushVal = document.getElementById("brush-val");

function initCanvas() {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  saveCanvasState();

  // Mouse Listeners
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("mouseleave", onMouseUp);

  // Touch Listeners
  canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = (touch.clientX - rect.left) * (canvas.width / rect.width);
    const y = (touch.clientY - rect.top) * (canvas.height / rect.height);
    isDrawing = true;
    lastX = x;
    lastY = y;
    drawPoint(x, y);
  });

  canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = (touch.clientX - rect.left) * (canvas.width / rect.width);
    const y = (touch.clientY - rect.top) * (canvas.height / rect.height);
    drawLine(lastX, lastY, x, y);
    lastX = x;
    lastY = y;
  });

  canvas.addEventListener("touchend", (e) => {
    e.preventDefault();
    if (isDrawing) {
      isDrawing = false;
      saveCanvasState();
    }
  });

  // Slider
  brushSlider.addEventListener("input", (e) => {
    brushSize = parseInt(e.target.value, 10);
    brushVal.textContent = `${brushSize}px`;
  });

  initWaveform();
}

function getCoords(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function onMouseDown(e) {
  isDrawing = true;
  const { x, y } = getCoords(e);
  lastX = x;
  lastY = y;
  drawPoint(x, y);
}

function onMouseMove(e) {
  if (!isDrawing) return;
  const { x, y } = getCoords(e);
  drawLine(lastX, lastY, x, y);
  lastX = x;
  lastY = y;
}

function onMouseUp() {
  if (isDrawing) {
    isDrawing = false;
    saveCanvasState();
  }
}

function drawPoint(x, y) {
  ctx.beginPath();
  ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
}

function drawLine(x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = brushSize;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
}

function saveCanvasState() {
  if (undoStack.length >= MAX_UNDO) {
    undoStack.shift();
  }
  undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
}

function clearCanvas() {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  saveCanvasState();
  resetOutputs();
}

function undoCanvas() {
  if (undoStack.length > 1) {
    undoStack.pop();
    const prev = undoStack[undoStack.length - 1];
    ctx.putImageData(prev, 0, 0);
  } else if (undoStack.length === 1) {
    clearCanvas();
  }
}

function resetOutputs() {
  document.getElementById("pred-char-output").textContent = "-";
  document.getElementById("pred-class-idx").textContent = "-- / 46";
  document.getElementById("pred-conf-pct").textContent = "-- %";
  document.getElementById("pred-conf-bar").style.width = "0%";
  document.getElementById("pred-status-text").textContent = "Idle";

  const ind = document.getElementById("confidence-indicator");
  ind.textContent = "Awaiting Input";
  ind.style.borderColor = "var(--border-subtle)";

  document.getElementById("alert-ambiguity").classList.add("hidden");
  document.getElementById("top5-table-body").innerHTML = `
    <tr>
      <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 18px;">
        Draw a character or select a preset to compute the softmax ranking.
      </td>
    </tr>
  `;
}

// Preset vectors for fast verification
function drawPreset(char) {
  clearCanvas();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const w = canvas.width;
  const h = canvas.height;

  ctx.beginPath();
  if (char === "A") {
    ctx.moveTo(w * 0.22, h * 0.82);
    ctx.lineTo(w * 0.50, h * 0.18);
    ctx.lineTo(w * 0.78, h * 0.82);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(w * 0.33, h * 0.58);
    ctx.lineTo(w * 0.67, h * 0.58);
    ctx.stroke();
  } else if (char === "3") {
    ctx.arc(w * 0.5, h * 0.36, w * 0.22, Math.PI * 1.2, Math.PI * 0.5);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.66, w * 0.24, -Math.PI * 0.5, Math.PI * 0.8);
    ctx.stroke();
  } else if (char === "W") {
    ctx.moveTo(w * 0.20, h * 0.22);
    ctx.lineTo(w * 0.35, h * 0.80);
    ctx.lineTo(w * 0.50, h * 0.40);
    ctx.lineTo(w * 0.65, h * 0.80);
    ctx.lineTo(w * 0.80, h * 0.22);
    ctx.stroke();
  } else if (char === "7") {
    ctx.moveTo(w * 0.25, h * 0.24);
    ctx.lineTo(w * 0.75, h * 0.24);
    ctx.lineTo(w * 0.40, h * 0.84);
    ctx.stroke();
  } else if (char === "L") {
    ctx.moveTo(w * 0.35, h * 0.18);
    ctx.lineTo(w * 0.35, h * 0.80);
    ctx.lineTo(w * 0.75, h * 0.80);
    ctx.stroke();
  } else if (char === "0") {
    ctx.ellipse(w * 0.5, h * 0.5, w * 0.26, h * 0.35, 0, 0, Math.PI * 2);
    ctx.stroke();
  } else if (char === "q") {
    ctx.arc(w * 0.52, h * 0.42, w * 0.20, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(w * 0.72, h * 0.25);
    ctx.lineTo(w * 0.72, h * 0.85);
    ctx.stroke();
  }
  saveCanvasState();
  predictCanvas();
}

// POST to /api/predict
async function predictCanvas() {
  const dataUrl = canvas.toDataURL("image/png");
  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.textContent = "Computing...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });

    const data = await res.json();
    if (data.is_blank) {
      alert("Canvas frame is empty. Please draw a character stroke.");
      return;
    }
    if (data.error) {
      alert(`Inference error: ${data.error}`);
      return;
    }

    renderInferenceResult(data);
  } catch (err) {
    console.error("Inference request failed:", err);
    alert("Connection error to localhost:8000. Verify the Python backend is running.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Execute Inference";
  }
}

function renderInferenceResult(data) {
  // Main outputs
  document.getElementById("pred-char-output").textContent = data.predicted_character;
  document.getElementById("pred-class-idx").textContent = `${data.class_index} / 46`;
  document.getElementById("pred-conf-pct").textContent = `${data.confidence.toFixed(2)}%`;
  document.getElementById("pred-conf-bar").style.width = `${Math.min(100, data.confidence)}%`;

  const ind = document.getElementById("confidence-indicator");
  const statusText = document.getElementById("pred-status-text");

  if (data.confidence >= 80) {
    ind.textContent = "High Certainty";
    ind.style.borderColor = "var(--accent-green)";
    statusText.textContent = "High Confidence";
  } else if (data.confidence >= 50) {
    ind.textContent = "Moderate Certainty";
    ind.style.borderColor = "var(--accent-amber)";
    statusText.textContent = "Moderate Confidence";
  } else {
    ind.textContent = "Ambiguous Classification";
    ind.style.borderColor = "var(--accent-red)";
    statusText.textContent = "Low / Split Distribution";
  }

  // Ambiguity notification
  const alertBox = document.getElementById("alert-ambiguity");
  if (data.ambiguity_note) {
    document.getElementById("alert-ambiguity-text").textContent = data.ambiguity_note;
    alertBox.classList.remove("hidden");
  } else {
    alertBox.classList.add("hidden");
  }

  // Render Top 5 table
  renderTop5Table("top5-table-body", data.top_5);

  // 28x28 normalized image
  if (data.preview_image) {
    document.getElementById("matrix-28x28-img").src = data.preview_image;
  }

  // Waveform
  if (data.scanline_intensities) {
    updateWaveform(data.scanline_intensities);
  }
}

function renderTop5Table(tbodyId, top5List) {
  const tbody = document.getElementById(tbodyId);
  if (!top5List || top5List.length === 0) return;

  let html = "";
  top5List.forEach((item, idx) => {
    const pct = (item.confidence * 100).toFixed(2);
    html += `
      <tr>
        <td class="rank-idx">${idx + 1}</td>
        <td class="char-cell">${item.character}</td>
        <td class="prob-cell">
          <div class="progress-bar-bg" style="margin: 0;">
            <div class="progress-bar-fill" style="width: ${pct}%;"></div>
          </div>
        </td>
        <td class="prob-pct">${pct}%</td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

// 28 Time Steps Waveform
function initWaveform() {
  const container = document.getElementById("waveform-display");
  container.innerHTML = "";
  for (let i = 0; i < 28; i++) {
    const bar = document.createElement("div");
    bar.className = "waveform-col";
    bar.id = `wave-bar-${i}`;
    bar.style.height = "2px";
    bar.title = `Step t=${i + 1} (Row ${i})`;
    container.appendChild(bar);
  }
}

function updateWaveform(intensities) {
  const maxVal = Math.max(...intensities, 0.01);
  for (let i = 0; i < 28; i++) {
    const bar = document.getElementById(`wave-bar-${i}`);
    if (bar && intensities[i] !== undefined) {
      const h = Math.max(2, Math.min(100, (intensities[i] / maxVal) * 100));
      bar.style.height = `${h}%`;
    }
  }
}

// Section Switching
function switchSection(name) {
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
  document.querySelectorAll(".tab-section").forEach((sec) => sec.classList.remove("active"));

  if (name === "canvas") {
    document.getElementById("tab-canvas-btn").classList.add("active");
    document.getElementById("section-canvas").classList.add("active");
  } else if (name === "explorer") {
    document.getElementById("tab-explorer-btn").classList.add("active");
    document.getElementById("section-explorer").classList.add("active");
    loadRandomTestSample();
  } else if (name === "analytics") {
    document.getElementById("tab-analytics-btn").classList.add("active");
    document.getElementById("section-analytics").classList.add("active");
  }
}

// Test Explorer Methods
async function loadRandomTestSample() {
  const btn = document.getElementById("btn-random-sample");
  btn.disabled = true;
  btn.textContent = "Fetching Sample...";

  try {
    const res = await fetch("/api/random_test_sample");
    const data = await res.json();
    renderTestExplorerSample(data);
  } catch (err) {
    console.error("Failed to load random test sample:", err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Load Random Test Sample";
  }
}

async function loadSpecificTestSample() {
  const input = document.getElementById("input-test-index");
  const idx = parseInt(input.value, 10);
  if (isNaN(idx) || idx < 0 || idx >= 18800) {
    alert("Please provide an index integer between 0 and 18799.");
    return;
  }

  try {
    const res = await fetch(`/api/test_sample?index=${idx}`);
    const data = await res.json();
    renderTestExplorerSample(data);
  } catch (err) {
    console.error("Failed to load sample by index:", err);
  }
}

function renderTestExplorerSample(data) {
  document.getElementById("test-view-img").src = data.image_data;
  document.getElementById("test-idx-tag").textContent = `Index: #${data.sample_index}`;
  document.getElementById("test-cls-tag").textContent = `Class: ${data.ground_truth_index}`;
  document.getElementById("input-test-index").value = data.sample_index;

  document.getElementById("test-gt-char").textContent = data.ground_truth_character;
  const predElem = document.getElementById("test-pred-char");
  predElem.textContent = data.predicted_character;

  const pill = document.getElementById("test-status-pill");
  if (data.is_correct) {
    pill.textContent = "CORRECT MATCH";
    pill.className = "comp-status-badge badge-match";
    predElem.style.color = "var(--accent-green)";
  } else {
    pill.textContent = "MISCLASSIFICATION";
    pill.className = "comp-status-badge badge-diff";
    predElem.style.color = "var(--accent-red)";
  }

  renderTop5Table("test-top5-body", data.top_5);
}

// Lifecycle Init
window.addEventListener("DOMContentLoaded", () => {
  initCanvas();
});
