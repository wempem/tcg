import { TEMPLATE_CROPS, TCG_NAMES } from "./cropRegions.js";
import { loadIndex, getCachedIndex } from "./cardIndex.js";
import { fuzzyMatch } from "./fuzzyMatch.js";

const BORDER_NAMES = { 0: "black", 1: "white" };  // matches main.cpp BORDER_*
const SYMBOL_LABELS = { mtg: null, pokemon: null };  // lazy-loaded per TCG

async function loadSymbolLabels(tcgName) {
  if (SYMBOL_LABELS[tcgName] !== null) return SYMBOL_LABELS[tcgName];
  if (tcgName === "yugioh") return SYMBOL_LABELS[tcgName] = {};
  const res = await fetch(`static/labels/symbol_${tcgName}_labels.json`);
  if (!res.ok) return SYMBOL_LABELS[tcgName] = {};
  const data = await res.json();
  const idx_to = {};
  for (const [name, idx] of Object.entries(data.label_to_idx)) idx_to[idx] = name;
  SYMBOL_LABELS[tcgName] = idx_to;
  return idx_to;
}

// Detection slot layout — must match struct Detection in wasm_ncnn/src/main.cpp.
// 256 bytes per slot, 8 slots. Phase 1 only reads conf + bbox; later phases
// (template_id, OCR strings, symbol_id, border_id) will land in the same slots.
const SLOT_SIZE = 256;
const MAX_DETECTIONS = 8;
const RESULT_BUFFER_SIZE = SLOT_SIZE * MAX_DETECTIONS;

// Offsets within one Detection slot.
const O_CONF = 0;
const O_BBOX_X = 4;
const O_BBOX_Y = 8;
const O_BBOX_W = 12;
const O_BBOX_H = 16;
const O_TEMPLATE_ID = 20;
const O_SYMBOL_ID = 24;
const O_BORDER_ID = 28;
const O_TEMPLATE_CONF = 32;
const O_SYMBOL_CONF = 36;
const O_NAME = 40;            // 64 bytes
const O_COLLECTOR_ID = 104;   // 32 bytes
const O_SET_TEXT = 136;       // 32 bytes

const TEXT_DECODER = new TextDecoder();

export function createDetector({ confidenceThreshold = 0.25 } = {}) {
  let imageBuf;     // RGBA frame bytes pushed into WASM heap
  let resultBuf;    // 2048 bytes; struct Detection[8]
  let tcgId = 0;    // 0=mtg, 1=pokemon, 2=yugioh

  function initMemory() {
    // 640×640 RGBA = 1.6 MB. Larger than the YOLO model input (320×320) — C++
    // resizes for YOLO but keeps the full 640 source for downstream crops.
    imageBuf = _malloc(640 * 640 * 4);
    resultBuf = _malloc(RESULT_BUFFER_SIZE);
  }

  initMemory();

  function setTcg(id) {
    tcgId = id;
    const tcgName = TCG_NAMES[id];
    renderEmpty();
    const card = document.getElementById("cardCropCanvas");
    if (card) card.getContext("2d").clearRect(0, 0, card.width, card.height);
    const previews = document.getElementById("regionPreviews");
    if (previews) previews.innerHTML = "";
    loadSymbolLabels(tcgName);  // warm symbol-label cache
    loadIndex(tcgName).catch(err => console.error("index load failed:", err));
  }

  // The most recent raw frame ImageData — captured pre-overlay so the
  // locked snapshot shows the exact pixels the pipeline processed.
  let lastFrame = null;
  function processFrame(ctx, canvas) {
    lastFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    HEAPU8.set(lastFrame.data, imageBuf);

    _process_frame(
      imageBuf,
      canvas.width,
      canvas.height,
      tcgId,
      confidenceThreshold,
      resultBuf,
    );

    renderDetections(ctx);
  }

  function decodeSlot(slotIndex) {
    const base = resultBuf + slotIndex * SLOT_SIZE;
    const view = new DataView(HEAPU8.buffer, base, SLOT_SIZE);
    const slot = HEAPU8.subarray(base, base + SLOT_SIZE);
    const conf = view.getFloat32(O_CONF, true);
    if (conf <= 0) return null;
    return {
      conf,
      x: view.getFloat32(O_BBOX_X, true),
      y: view.getFloat32(O_BBOX_Y, true),
      w: view.getFloat32(O_BBOX_W, true),
      h: view.getFloat32(O_BBOX_H, true),
      templateId: view.getInt32(O_TEMPLATE_ID, true),
      symbolId: view.getInt32(O_SYMBOL_ID, true),
      borderId: view.getInt32(O_BORDER_ID, true),
      templateConf: view.getFloat32(O_TEMPLATE_CONF, true),
      symbolConf: view.getFloat32(O_SYMBOL_CONF, true),
      name: decodeCStr(slot, O_NAME, 64),
      collectorId: decodeCStr(slot, O_COLLECTOR_ID, 32),
      setText: decodeCStr(slot, O_SET_TEXT, 32),
    };
  }

  function renderDetections(ctx) {
    let topDetection = null;
    for (let i = 0; i < MAX_DETECTIONS; i++) {
      const d = decodeSlot(i);
      if (!d) continue;
      drawBox(ctx, d);
      if (!topDetection || d.conf > topDetection.conf) topDetection = d;
    }
    if (!topDetection) {
      renderEmpty();
      return;
    }
    // Surface the detection even when OCR is empty — this way you can see
    // "card detected, template predicted, just OCR didn't read anything"
    // rather than the silent "Hold a card…" state.
    updateMatches(topDetection);
  }

  function renderEmpty() {
    const panel = document.getElementById("matches");
    if (panel) panel.innerHTML = `<div class="text-slate-500 text-sm">Hold a card in front of the camera…</div>`;
  }

  function updateMatches(d) {
    const panel = document.getElementById("matches");
    if (!panel) return;
    const tcgName = TCG_NAMES[tcgId];
    const idx = getCachedIndex(tcgName);
    if (!idx) {
      panel.innerHTML = `<div class="text-slate-500 text-sm">Loading ${tcgName} index…</div>`;
      return;
    }
    const tmpl = TEMPLATE_CROPS[tcgName]?.[d.templateId];
    const predictedSet = SYMBOL_LABELS[tcgName]?.[d.symbolId] || null;
    const predictedFamily = tmpl?.family
      ? (tcgName === "mtg" ? mtgFamilyFromTemplate(tmpl.family) : null)
      : null;
    const sampledBorder = BORDER_NAMES[d.borderId] || null;
    const top = fuzzyMatch({
      tcg: tcgName,
      indexData: idx,
      ocrName: d.name,
      ocrId: d.collectorId,
      ocrSetText: d.setText,
      predictedSet,
      predictedSetConf: d.symbolConf || 0,
      predictedFamily,
      sampledBorder,
      topK: 5,
    });
    // Debug: log every ~30 frames so the console doesn't drown.
    if ((updateMatches._n = (updateMatches._n || 0) + 1) % 30 === 0) {
      console.log("[match]", {
        ocr: { name: d.name, id: d.collectorId, setText: d.setText },
        signals: { predictedSet, predictedSetConf: d.symbolConf, predictedFamily, sampledBorder },
        top5: top.map(r => `${r.entry.n} [${r.entry.set}] ${r.combined.toFixed(1)}`),
      });
    }
    const html = top.length
      ? top.map(r => renderMatchRow(r)).join("")
      : `<div class="text-slate-500 text-sm">No matches found.</div>`;
    panel.innerHTML = html;
    snapshotLockedFrame(d);
  }

  // Render what the pipeline actually saw: the YOLO-cropped card, followed
  // by each populated sub-region with its OCR or symbol-classifier output.
  // Driven off `lastFrame` (the exact pre-overlay ImageData fed to WASM) so
  // every pixel shown is bit-identical to what was processed.
  function snapshotLockedFrame(d) {
    if (!lastFrame) return;
    drawCardCrop(d);
    drawRegionPreviews(d);
  }

  function drawCardCrop(d) {
    const cv = document.getElementById("cardCropCanvas");
    if (!cv) return;
    const aspect = d.h > 0 ? d.w / d.h : 1;
    const targetW = 200;
    const targetH = Math.max(40, Math.round(targetW / aspect));
    cv.width = targetW; cv.height = targetH;
    blitFrameRegion(cv, d.x, d.y, d.w, d.h);

    // Overlay the four percent-based sub-region rectangles on the card crop
    // itself — the colored boxes here should land on the same features as
    // the colored canvases below. Misalignment is visible at a glance.
    const tcgName = TCG_NAMES[tcgId];
    const tmpl = TEMPLATE_CROPS[tcgName]?.[d.templateId];
    if (!tmpl) return;
    const ctx = cv.getContext("2d");
    ctx.lineWidth = 1.5;
    for (const info of REGION_INFO) {
      const r = tmpl[info.key];
      if (!r) continue;
      const [rx, ry, rw, rh] = r;
      ctx.strokeStyle = info.color;
      ctx.strokeRect(rx * targetW, ry * targetH, rw * targetW, rh * targetH);
    }
  }

  // Color → human label, matches drawCropOverlays' palette.
  const REGION_INFO = [
    { key: "name",         color: "rgba(34,197,94,0.95)",  title: "name (OCR)" },
    { key: "collector_id", color: "rgba(239,68,68,0.95)",  title: "id (OCR)" },
    { key: "set_text",     color: "rgba(249,115,22,0.95)", title: "set text (OCR)" },
    { key: "set_symbol",   color: "rgba(59,130,246,0.95)", title: "set symbol (classifier)" },
  ];

  function drawRegionPreviews(d) {
    const host = document.getElementById("regionPreviews");
    if (!host) return;
    host.innerHTML = "";
    const tcgName = TCG_NAMES[tcgId];
    const tmpl = TEMPLATE_CROPS[tcgName]?.[d.templateId];
    if (!tmpl) {
      host.innerHTML = `<div class="text-xs text-slate-500">No template — nothing cropped.</div>`;
      return;
    }
    const tplHeader = document.createElement("div");
    tplHeader.className = "text-xs text-slate-400";
    tplHeader.innerHTML = `template: <span class="text-slate-200 font-mono">${escapeHtml(tmpl.family || "?")}</span> <span class="text-slate-500">(${(d.templateConf * 100).toFixed(0)}%)</span>`;
    host.appendChild(tplHeader);
    const sym = SYMBOL_LABELS[tcgName]?.[d.symbolId];
    const outputs = {
      name: d.name || "(empty)",
      collector_id: d.collectorId || "(empty)",
      set_text: d.setText || "(empty)",
      set_symbol: d.symbolId >= 0
        ? `${sym || `class ${d.symbolId}`}  (${(d.symbolConf * 100).toFixed(0)}%)`
        : "(skipped)",
    };
    for (const info of REGION_INFO) {
      const r = tmpl[info.key];
      if (!r) continue;
      const [rx, ry, rw, rh] = r;
      const sx = d.x + rx * d.w;
      const sy = d.y + ry * d.h;
      const sw = rw * d.w;
      const sh = rh * d.h;
      const row = document.createElement("div");
      row.className = "flex items-center gap-3";
      const cv = document.createElement("canvas");
      // Pick a reasonable display height; preserve source aspect.
      const aspect = sh > 0 ? sw / sh : 1;
      const dispH = 36;
      const dispW = Math.max(20, Math.min(160, Math.round(dispH * aspect)));
      cv.width = dispW; cv.height = dispH;
      cv.style.border = `1.5px solid ${info.color}`;
      cv.className = "rounded bg-slate-900/60 shrink-0";
      row.appendChild(cv);
      blitFrameRegion(cv, sx, sy, sw, sh);
      const label = document.createElement("div");
      label.className = "min-w-0 flex-1";
      label.innerHTML =
        `<div class="text-[10px] uppercase tracking-wide text-slate-400">${info.title}</div>` +
        `<div class="text-sm font-mono text-slate-200 truncate">${escapeHtml(outputs[info.key])}</div>`;
      row.appendChild(label);
      host.appendChild(row);
    }
  }

  // Paint a rectangular region of `lastFrame` onto `cv`, scaled to fit.
  function blitFrameRegion(cv, sx, sy, sw, sh) {
    sx = Math.max(0, sx); sy = Math.max(0, sy);
    sw = Math.min(lastFrame.width - sx, sw);
    sh = Math.min(lastFrame.height - sy, sh);
    if (sw < 1 || sh < 1) return;
    // Use a scratch canvas to host the full frame so drawImage can scale.
    if (!blitFrameRegion._scratch) {
      blitFrameRegion._scratch = document.createElement("canvas");
    }
    const scratch = blitFrameRegion._scratch;
    if (scratch.width !== lastFrame.width || scratch.height !== lastFrame.height) {
      scratch.width = lastFrame.width;
      scratch.height = lastFrame.height;
    }
    scratch.getContext("2d").putImageData(lastFrame, 0, 0);
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(scratch, sx, sy, sw, sh, 0, 0, cv.width, cv.height);
  }

  function renderMatchRow(r) {
    const e = r.entry;
    const badge = e.set ? `<span class="text-cyan-400 ml-2">${e.set}</span>` : "";
    const id = e.id ? `<span class="text-slate-400 ml-2">${e.id}</span>` : "";
    const score = r.combined.toFixed(1);
    return `<div class="py-2 border-b border-white/5 flex items-baseline justify-between gap-3">
      <div class="truncate"><span class="font-medium">${escapeHtml(e.n || "(unnamed)")}</span>${badge}${id}</div>
      <div class="text-xs text-slate-500 tabular-nums">${score}</div>
    </div>`;
  }

  function mtgFamilyFromTemplate(family) {
    // family is e.g. "mtg_2015"; strip the mtg_ prefix and modern→2003 mapping.
    if (!family || !family.startsWith("mtg_")) return null;
    const era = family.split("_")[1];
    if (era === "exclude") return null;
    if (era === "modern") return "2003";
    return era;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g,
      c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c]));
  }

  // Live camera overlay: just the YOLO box. Sub-rectangles + readouts move
  // to the locked-frame canvas so the user can compare apples-to-apples.
  function drawBox(ctx, d) {
    const color = confColor(d.conf);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    ctx.restore();
  }

  // 4 sub-region colors matching PIPELINE.md (green/red/orange/blue).
  const REGION_COLORS = {
    name:         "rgba(34,197,94,0.9)",   // green
    collector_id: "rgba(239,68,68,0.9)",   // red
    set_text:     "rgba(249,115,22,0.9)",  // orange
    set_symbol:   "rgba(59,130,246,0.9)",  // blue
  };

  function drawCropOverlays(ctx, d) {
    if (d.templateId < 0) return;
    const tcgName = TCG_NAMES[tcgId];
    const tmpl = TEMPLATE_CROPS[tcgName]?.[d.templateId];
    if (!tmpl) return;
    ctx.save();
    ctx.lineWidth = 1.5;
    for (const key of ["name", "collector_id", "set_text", "set_symbol"]) {
      const r = tmpl[key];
      if (!r) continue;
      const [rx, ry, rw, rh] = r;
      ctx.strokeStyle = REGION_COLORS[key];
      ctx.strokeRect(d.x + rx * d.w, d.y + ry * d.h, rw * d.w, rh * d.h);
    }
    ctx.restore();
  }

  function drawLabel(ctx, x, y, text, color) {
    ctx.save();
    ctx.font = "600 14px Inter, sans-serif";
    const padX = 8, padY = 4;
    const tw = ctx.measureText(text).width;
    const lw = tw + padX * 2;
    const lh = 22;
    const lx = x;
    const ly = Math.max(0, y - lh - 4);
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(lx, ly, lw, lh);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, lx + padX, ly + lh / 2);
    ctx.restore();
  }

  function confColor(c) {
    if (c >= 0.6) return "rgba(34,197,94,1)";
    if (c >= 0.4) return "rgba(255,165,0,1)";
    return "rgba(239,68,68,1)";
  }

  return { processFrame, setTcg };
}

function decodeCStr(slot, offset, maxLen) {
  let end = offset;
  const limit = offset + maxLen;
  while (end < limit && slot[end] !== 0) end++;
  if (end === offset) return "";
  return TEXT_DECODER.decode(slot.subarray(offset, end));
}
