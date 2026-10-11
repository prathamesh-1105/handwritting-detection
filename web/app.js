/* ==========================================================================
   NEUROSCRIBE LSTM - CLIENT JAVASCRIPT LOGIC
   Interactive Canvas, Real-time REST API, and Sequence Visualizer
   ========================================================================== */

// State
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 20;
let undoStack = [];
const MAX_UNDO = 15;

// DOM Elements
const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const brushSlider = document.getElementById("brush-size-slider");
const brushLabel = document.getElementById("brush-size-label");

// Initialize Canvas
function initCanvas() {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  saveState();

  // Mouse Listeners
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseleave", stopDrawing);

  // Touch Listeners (Mobile & Tablet)
  canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent("mousedown", {
      clientX: touch.clientX,
      clientY: touch.clientY,
    });
    canvas.dispatchEvent(mouseEvent);
  });

  canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent("mousemove", {
      clientX: touch.clientX,
      clientY: touch.clientY,
    });
    canvas.dispatchEvent(mouseEvent);
  });

  canvas.addEventListener("touchend", (e) => {
    e.preventDefault();
    const mouseEvent = new MouseEvent("mouseup", {});
    canvas.dispatchEvent(mouseEvent);
  });

  // Slider
  brushSlider.addEventListener("input", (e) => {
    brushSize = parseInt(e.target.value, 10);
    brushLabel.textContent = `${brushSize}px`;
  });

  // Initialize Timeline graph
  initTimelineGraph();
}

function getCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.x) * scaleX,
    y: (e.clientY - rect.y) * scaleY,
  };
}

function startDrawing(e) {
  isDrawing = true;
  const coords = getCanvasCoords(e);
  lastX = coords.x;
  lastY = coords.y;

  // Draw initial dot
  ctx.beginPath();
  ctx.arc(lastX, lastY, brushSize / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
}

function draw(e) {
  if (!isDrawing) return;
  const coords = getCanvasCoords(e);

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(coords.x, coords.y);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = brushSize;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();

  lastX = coords.x;
  lastY = coords.y;
}

function stopDrawing() {
  if (isDrawing) {
    isDrawing = false;
    saveState();
  }
}

function saveState() {
  if (undoStack.length >= MAX_UNDO) {
    undoStack.shift();
  }
  undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
}

function clearCanvas() {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  saveState();
  resetPredictionDisplay();
}

function undoCanvas() {
  if (undoStack.length > 1) {
    undoStack.pop(); // Remove current state
    const prevState = undoStack[undoStack.length - 1];
    ctx.putImageData(prevState, 0, 0);
  } else if (undoStack.length === 1) {
    clearCanvas();
  }
}

function resetPredictionDisplay() {
  document.getElementById("hero-pred-char").textContent = "?";
  document.getElementById("hero-pred-label").textContent = "Draw & Classify";
  document.getElementById("stat-confidence").textContent = "-- %";
  document.getElementById("stat-confidence-bar").style.width = "0%";
  document.getElementById("stat-class-idx").textContent = "-- / 46";

  const badge = document.getElementById("confidence-badge");
  badge.textContent = "Awaiting Input";
  badge.className = "badge badge-idle";

  document.getElementById("ambiguity-alert").classList.add("hidden");
  document.getElementById("top5-container").innerHTML = `
    <div class="empty-state-text">Draw on the canvas and click "Classify with LSTM" to inspect the top 5 hypotheses.</div>
  `;
}

// Draw quick presets for fast demonstration
function drawPresetChar(char) {
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
  saveState();
  predictCanvas();
}

// Call Classification API
async function predictCanvas() {
  const dataUrl = canvas.toDataURL("image/png");
  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.innerHTML = `<span>Classifying...</span>`;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });

    const data = await res.json();
    if (data.is_blank) {
      alert("Canvas is blank! Please draw a character before classifying.");
      return;
    }

    if (data.error) {
      alert(`Classification error: ${data.error}`);
      return;
    }

    renderPredictionResults(data);
  } catch (err) {
    console.error("Prediction request failed:", err);
    alert("Server error during classification. Check localhost console.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Classify with LSTM</span>
    `;
  }
}

function renderPredictionResults(data) {
  // Hero section
  const heroChar = document.getElementById("hero-pred-char");
  heroChar.textContent = data.predicted_character;
  document.getElementById("hero-pred-label").textContent = `Class ${data.class_index}`;
  document.getElementById("stat-confidence").textContent = `${data.confidence.toFixed(1)}%`;
  document.getElementById("stat-confidence-bar").style.width = `${Math.min(100, data.confidence)}%`;
  document.getElementById("stat-class-idx").textContent = `${data.class_index} / 46`;

  // Badge tier
  const badge = document.getElementById("confidence-badge");
  if (data.confidence >= 80) {
    badge.textContent = "High Confidence";
    badge.className = "badge badge-high";
  } else if (data.confidence >= 50) {
    badge.textContent = "Moderate Confidence";
    badge.className = "badge badge-medium";
  } else {
    badge.textContent = "Low / Ambiguous";
    badge.className = "badge badge-low";
  }

  // Ambiguity alert
  const alertBox = document.getElementById("ambiguity-alert");
  if (data.ambiguity_note) {
    document.getElementById("ambiguity-alert-msg").textContent = data.ambiguity_note;
    alertBox.classList.remove("hidden");
  } else {
    alertBox.classList.add("hidden");
  }

  // Top-5 list
  renderTop5List("top5-container", data.top_5);

  // 28x28 Preprocessed Preview
  if (data.preview_image) {
    document.getElementById("processed-28x28-img").src = data.preview_image;
  }

  // Update 28-step timeline bars
  if (data.scanline_intensities) {
    updateTimelineGraph(data.scanline_intensities);
  }
}

function renderTop5List(containerId, top5) {
  const container = document.getElementById(containerId);
  if (!top5 || top5.length === 0) return;

  let html = "";
  top5.forEach((item, idx) => {
    const pct = (item.confidence * 100).toFixed(1);
    html += `
      <div class="top5-row">
        <span class="top5-rank">#${idx + 1}</span>
        <span class="top5-char">${item.character}</span>
        <div class="top5-bar-wrap">
          <div class="top5-bar-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="top5-pct">${pct}%</span>
      </div>
    `;
  });
  container.innerHTML = html;
}

// 28 Time Steps Timeline Visualization
function initTimelineGraph() {
  const container = document.getElementById("timeline-bars");
  container.innerHTML = "";
  for (let i = 0; i < 28; i++) {
    const bar = document.createElement("div");
    bar.className = "timeline-bar-col";
    bar.id = `timeline-col-${i}`;
    bar.style.height = "4px";
    bar.title = `Step t=${i + 1} (Row ${i})`;
    container.appendChild(bar);
  }
}

function updateTimelineGraph(intensities) {
  const maxVal = Math.max(...intensities, 0.01);
  for (let i = 0; i < 28; i++) {
    const col = document.getElementById(`timeline-col-${i}`);
    if (col && intensities[i] !== undefined) {
      const heightPct = Math.max(4, Math.min(100, (intensities[i] / maxVal) * 100));
      col.style.height = `${heightPct}%`;
    }
  }
}

// Tab Switching
function switchTab(tabName) {
  document.querySelectorAll(".tab-button").forEach((btn) => btn.classList.remove("active"));
  document.querySelectorAll(".tab-view").forEach((view) => view.classList.remove("active"));

  if (tabName === "canvas") {
    document.getElementById("tab-canvas-btn").classList.add("active");
    document.getElementById("view-canvas").classList.add("active");
  } else if (tabName === "explorer") {
    document.getElementById("tab-explorer-btn").classList.add("active");
    document.getElementById("view-explorer").classList.add("active");
    loadRandomTestSample();
  } else if (tabName === "analytics") {
    document.getElementById("tab-analytics-btn").classList.add("active");
    document.getElementById("view-analytics").classList.add("active");
  }
}

// ==========================================================================
// TAB 2: TEST EXPLORER
// ==========================================================================
async function loadRandomTestSample() {
  const btn = document.getElementById("btn-random-test");
  btn.disabled = true;

  try {
    const res = await fetch("/api/random_test_sample");
    const data = await res.json();
    renderTestSample(data);
  } catch (err) {
    console.error("Failed to load random test sample:", err);
  } finally {
    btn.disabled = false;
  }
}

async function loadSpecificTestSample() {
  const input = document.getElementById("input-sample-idx");
  const idx = parseInt(input.value, 10);
  if (isNaN(idx) || idx < 0 || idx >= 18800) {
    alert("Please enter a valid sample index between 0 and 18799.");
    return;
  }

  try {
    const res = await fetch(`/api/test_sample?index=${idx}`);
    const data = await res.json();
    renderTestSample(data);
  } catch (err) {
    console.error("Failed to load test sample:", err);
  }
}

function renderTestSample(data) {
  document.getElementById("test-sample-img").src = data.image_data;
  document.getElementById("test-sample-idx-badge").textContent = `Index: #${data.sample_index}`;
  document.getElementById("test-sample-class-badge").textContent = `Class ${data.ground_truth_index}`;
  document.getElementById("input-sample-idx").value = data.sample_index;

  document.getElementById("test-gt-char").textContent = data.ground_truth_character;
  const predElem = document.getElementById("test-pred-char");
  predElem.textContent = data.predicted_character;

  const pill = document.getElementById("test-outcome-pill");
  if (data.is_correct) {
    pill.textContent = "[CORRECT MATCH]";
    pill.className = "outcome-pill pill-correct";
    predElem.className = "comparison-char text-success";
  } else {
    pill.textContent = "[MISCLASSIFICATION]";
    pill.className = "outcome-pill pill-mismatch";
    predElem.className = "comparison-char text-accent";
  }

  renderTop5List("test-top5-container", data.top_5);
}

// On Page Load
window.addEventListener("DOMContentLoaded", () => {
  initCanvas();
  // Fetch metrics to verify server
  fetch("/api/metrics")
    .then((res) => res.json())
    .then((data) => {
      console.log("Connected to NeuroScribe LSTM Backend:", data);
    })
    .catch((err) => console.warn("Backend not yet responsive:", err));
});
