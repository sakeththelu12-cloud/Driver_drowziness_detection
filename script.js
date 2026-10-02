/* ==========================================================
   ALERTNESS — Driver Monitoring System
   Client-side inference with TensorFlow.js
   ========================================================== */

// ---- Configuration ----------------------------------------------------
const MODE = "browser";           // "browser" = run TF.js model locally, "server" = call a backend API
const MODEL_URL = "model/model.json"; // path to converted TF.js model (browser mode)
const API_URL = "http://localhost:5000/predict"; // used only if MODE === "server"

const IMG_SIZE = 224;
const SMOOTHING_WINDOW = 8;       // frames averaged to reduce flicker
const DROWSY_THRESHOLD = 0.5;     // sigmoid output above this = drowsy
const ALARM_SUSTAIN_SECONDS = 2;  // how long "drowsy" must persist before alarm fires
const INFER_INTERVAL_MS = 250;    // ~4 inferences/sec is plenty for drowsiness

// ---- DOM references -----------------------------------------------------
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const prompt = document.getElementById("prompt");
const stateBadge = document.getElementById("stateBadge");
const stateLabel = document.getElementById("stateLabel");
const scoreValue = document.getElementById("scoreValue");
const statusText = document.getElementById("statusText");
const confidenceText = document.getElementById("confidenceText");
const fpsText = document.getElementById("fpsText");
const sustainedText = document.getElementById("sustainedText");
const gaugeFill = document.getElementById("gaugeFill");
const gaugeNeedle = document.getElementById("gaugeNeedle");
const logList = document.getElementById("logList");
const clockEl = document.getElementById("clock");
const clusterEl = document.querySelector(".cluster");

// ---- State ----------------------------------------------------------
let model = null;
let recentScores = [];
let drowsySince = null;
let lastLoggedState = null;
let audioCtx = null;
let alarmOscillator = null;

// ---- Clock ----------------------------------------------------------
function tickClock() {
  clockEl.textContent = new Date().toLocaleTimeString("en-GB");
}
setInterval(tickClock, 1000);
tickClock();

// ---- Model loading ----------------------------------------------------
async function loadModel() {
  if (MODE !== "browser") return;
  try {
    model = await tf.loadLayersModel(MODEL_URL);
    // warm up
    const dummy = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]);
    model.predict(dummy).dispose();
    dummy.dispose();
    stateLabel.textContent = "MODEL READY";
  } catch (err) {
    console.error("Could not load model:", err);
    stateLabel.textContent = "MODEL LOAD FAILED";
    logEvent("Model failed to load — check frontend/model/model.json exists.", false);
  }
}

// ---- Webcam ----------------------------------------------------------
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    prompt.classList.add("hidden");
    logEvent("Monitoring started.", false);
    requestAnimationFrame(inferenceLoop);
  } catch (err) {
    console.error(err);
    stateLabel.textContent = "CAMERA DENIED";
    logEvent("Camera access denied or unavailable.", false);
  }
}

// ---- Inference loop ----------------------------------------------------
let lastInferTime = 0;
let lastFrameTime = performance.now();

async function inferenceLoop(now) {
  requestAnimationFrame(inferenceLoop);

  const dt = now - lastFrameTime;
  lastFrameTime = now;
  if (dt > 0) fpsText.textContent = `${(1000 / dt).toFixed(0)} fps`;

  if (now - lastInferTime < INFER_INTERVAL_MS) return;
  lastInferTime = now;

  if (video.readyState < 2) return;

  let prob;
  try {
    prob = MODE === "browser" ? await predictBrowser() : await predictServer();
  } catch (err) {
    console.error("Inference error:", err);
    return;
  }
  if (prob === null || prob === undefined) return;

  recentScores.push(prob);
  if (recentScores.length > SMOOTHING_WINDOW) recentScores.shift();
  const smoothed = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;

  updateUI(smoothed, prob);
}

async function predictBrowser() {
  if (!model) return null;
  return tf.tidy(() => {
    let img = tf.browser.fromPixels(video);
    img = tf.image.resizeBilinear(img, [IMG_SIZE, IMG_SIZE]);
    img = img.toFloat().expandDims(0); // [1, 224, 224, 3], values 0-255 (matches training: no manual /255)
    const pred = model.predict(img);
    return pred.dataSync()[0];
  });
}

async function predictServer() {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  canvas.getContext("2d").drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.85);

  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  });
  const json = await res.json();
  return json.probability;
}

// ---- UI updates ----------------------------------------------------
function updateUI(smoothedProb, rawProb) {
  const pct = Math.round(smoothedProb * 100);
  scoreValue.textContent = pct;
  confidenceText.textContent = `${(rawProb * 100).toFixed(1)}%`;

  // Gauge: semicircle, dasharray 283 = full arc. 0 => full offset, 1 => 0 offset
  const offset = 283 - smoothedProb * 283;
  gaugeFill.style.strokeDashoffset = offset.toFixed(1);
  gaugeFill.style.stroke = smoothedProb > DROWSY_THRESHOLD ? "var(--red)" : "var(--cyan)";

  // Needle: -90deg (fully alert) to +90deg (fully drowsy)
  const angle = -90 + smoothedProb * 180;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;

  const isDrowsy = smoothedProb > DROWSY_THRESHOLD;

  if (isDrowsy) {
    if (drowsySince === null) drowsySince = performance.now();
  } else {
    drowsySince = null;
    stopAlarm();
  }

  const sustained = drowsySince ? (performance.now() - drowsySince) / 1000 : 0;
  sustainedText.textContent = `${sustained.toFixed(1)}s`;

  const state = isDrowsy ? "drowsy" : "alert";
  if (state !== lastLoggedState) {
    logEvent(isDrowsy ? "Drowsiness detected." : "Driver alert.", isDrowsy);
    lastLoggedState = state;
  }

  stateBadge.classList.toggle("state-alert", !isDrowsy);
  stateBadge.classList.toggle("state-drowsy", isDrowsy);
  stateLabel.textContent = isDrowsy ? "DROWSY" : "ALERT";
  statusText.textContent = isDrowsy ? "Drowsiness detected" : "Driver alert";
  clusterEl.classList.toggle("is-drowsy", isDrowsy);

  if (isDrowsy && sustained >= ALARM_SUSTAIN_SECONDS) {
    startAlarm();
  }
}

function logEvent(message, isDrowsy) {
  const li = document.createElement("li");
  const time = new Date().toLocaleTimeString("en-GB");
  li.textContent = `${time}  ${message}`;
  if (isDrowsy) li.classList.add("is-drowsy");
  logList.prepend(li);
  while (logList.children.length > 20) logList.removeChild(logList.lastChild);
}

// ---- Alarm (generated tone, no audio file needed) ----------------------
function startAlarm() {
  if (alarmOscillator) return;
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "square";
  osc.frequency.value = 880;
  gain.gain.value = 0.08;
  osc.connect(gain).connect(audioCtx.destination);
  osc.start();
  alarmOscillator = { osc, gain };

  // pulse the tone for urgency
  alarmOscillator.pulseInterval = setInterval(() => {
    if (!alarmOscillator) return;
    const t = audioCtx.currentTime;
    gain.gain.cancelScheduledValues(t);
    gain.gain.setValueAtTime(0.09, t);
    gain.gain.linearRampToValueAtTime(0.0, t + 0.25);
  }, 400);
}

function stopAlarm() {
  if (!alarmOscillator) return;
  clearInterval(alarmOscillator.pulseInterval);
  alarmOscillator.osc.stop();
  alarmOscillator.osc.disconnect();
  alarmOscillator = null;
}

// ---- Init ----------------------------------------------------------
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Starting…";
  await loadModel();
  await startCamera();
});

loadModel(); // preload model in the background while user reads the prompt
