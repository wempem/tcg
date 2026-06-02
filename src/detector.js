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
  let loBuf;        // small RGBA frame for Stage A (YOLO + template + border)
  let cropBuf;      // one high-res region crop for Stage B (OCR / symbol)
  let ocrBuf;       // OCR output C-string
  let symBuf;       // symbol_region output: [int32 id, float32 conf]
  let resultBuf;    // 2048 bytes; struct Detection[8]
  let tcgId = 0;    // 0=mtg, 1=pokemon, 2=yugioh

  // Stage A runs on a downscale (YOLO/template only need ~320). Stage B crops
  // name/id/symbol regions from the FULL-res capture frame, so OCR/symbol see
  // real detail. We never copy the full-res frame into the wasm heap — only
  // the small low-res frame and the even smaller region crops cross the line.
  const LO_EDGE = 512;             // Stage-A frame long edge
  const CROP_MAX_PX = 1024 * 768;  // cap one region crop's pixel count
  const loCanvas = document.createElement("canvas");
  const regionCanvas = document.createElement("canvas");
  const frameScratch = document.createElement("canvas");  // hosts the full-res frame
  // Per-signal locking. A single sharp frame is enough for the name, but the
  // id/symbol may still be soft on that frame — so we do NOT latch all signals
  // at once. Instead each region accumulates independently: it's read whenever
  // ITS OWN crop is sharp, and locked once the read is trustworthy (OCR: the
  // same text from two sharp crops; symbol: classifier confidence). Unlocked
  // signals keep trying while locked ones stay frozen. Scoped to the card.
  const REGION_FOCUS_MIN = 40;  // per-region variance-of-Laplacian to trust a read (tunable)
  const OCR_AGREE = 2;          // identical sharp OCR reads needed to lock a text signal
  const SYMBOL_LOCK = 0.6;      // symbol classifier softmax needed to lock the symbol
  const STABLE_IOU = 0.85;      // bbox overlap vs previous frame == "not moving"
  const LATCH_IOU = 0.5;        // overlap to treat a detection as the same card
  let lastBox = null;           // previous frame's bbox (movement check)
  let track = null;             // per-card signal accumulator (see freshTrack)
  let lastRenderKey = "";       // dedupe panel re-renders (keeps the UI static)
  let frameNo = 0;

  // The YOLO box runs loose AROUND the card (margin on every side), not just
  // at the top. Percent regions are calibrated against a tight box (card ==
  // box), so against a loose box the top regions (name) ride high and the
  // bottom regions (id/set_text) ride low while the middle stays put —
  // error ~ INSET*(2*r - 1). Fix: treat the true card as the box inset by
  // INSET per side and place the calibrated percentages inside it. inset()
  // maps a normalized region coord into that inset box; used on BOTH axes.
  const BBOX_INSET = 0.025;  // fractional margin per side (tunable)
  const inset = (r) => BBOX_INSET + r * (1 - 2 * BBOX_INSET);

  function initMemory() {
    loBuf = _malloc(LO_EDGE * LO_EDGE * 4);
    cropBuf = _malloc(CROP_MAX_PX * 4);
    ocrBuf = _malloc(64);
    symBuf = _malloc(8);
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
  // locked snapshot shows the exact pixels the pipeline processed. This is now
  // the FULL-resolution capture frame (Stage B crops + previews read from it).
  let lastFrame = null;

  function processFrame(ctx, canvas) {
    const HW = canvas.width, HH = canvas.height;
    lastFrame = ctx.getImageData(0, 0, HW, HH);
    // Host the full-res frame on a scratch canvas so we can crop sub-regions
    // (the live canvas gets the overlay drawn on it below).
    if (frameScratch.width !== HW || frameScratch.height !== HH) {
      frameScratch.width = HW; frameScratch.height = HH;
    }
    frameScratch.getContext("2d", { willReadFrequently: true }).putImageData(lastFrame, 0, 0);

    // ---- Stage A: downscale -> YOLO + template + border ----
    const k = Math.min(1, LO_EDGE / Math.max(HW, HH));
    const lw = Math.max(1, Math.round(HW * k));
    const lh = Math.max(1, Math.round(HH * k));
    if (loCanvas.width !== lw || loCanvas.height !== lh) { loCanvas.width = lw; loCanvas.height = lh; }
    const lctx = loCanvas.getContext("2d", { willReadFrequently: true });
    lctx.drawImage(frameScratch, 0, 0, lw, lh);
    const loFrame = lctx.getImageData(0, 0, lw, lh);
    HEAPU8.set(loFrame.data, loBuf);
    _analyze_frame(loBuf, lw, lh, tcgId, confidenceThreshold, resultBuf);

    // Decode slots (bbox in lo-res coords), scale to full-res, draw boxes,
    // and pick the top detection for Stage B.
    let top = null;
    const inv = 1 / k;
    for (let i = 0; i < MAX_DETECTIONS; i++) {
      const d = decodeSlot(i);
      if (!d) continue;
      d.x *= inv; d.y *= inv; d.w *= inv; d.h *= inv;  // lo-res -> full-res
      drawBox(ctx, d);
      if (!top || d.conf > top.conf) top = d;
    }
    if (!top) {
      lastBox = null; track = null;
      if (lastRenderKey !== "") { renderEmpty(); lastRenderKey = ""; }
      return;
    }

    // Same card as the running accumulator, or a new one?
    if (!track || iou(top, track.box) < LATCH_IOU) {
      track = freshTrack(top);
    } else {
      track.box = { x: top.x, y: top.y, w: top.w, h: top.h };
      track.borderId = top.borderId;
      // Keep the highest-confidence template we've seen for this card.
      if (top.templateConf > track.templateConf) {
        track.templateId = top.templateId;
        track.templateConf = top.templateConf;
      }
    }

    // Only accumulate when the card isn't moving (motion blurs every region).
    const stable = lastBox ? iou(top, lastBox) >= STABLE_IOU : false;
    lastBox = { x: top.x, y: top.y, w: top.w, h: top.h };
    if (stable && track.templateId >= 0) accumulate(track);

    renderTrack(track);
  }

  function freshTrack(d) {
    return {
      box: { x: d.x, y: d.y, w: d.w, h: d.h },
      templateId: d.templateId, templateConf: d.templateConf, borderId: d.borderId,
      sig: {
        name:         { v: "", agree: 0, lock: false },
        collector_id: { v: "", agree: 0, lock: false },
        set_text:     { v: "", agree: 0, lock: false },
        set_symbol:   { id: -1, conf: 0, lock: false },
      },
    };
  }

  // Read each not-yet-locked region; lock it once its OWN crop is sharp and the
  // read is trustworthy. Different signals can lock on different frames.
  function accumulate(track) {
    const tmpl = TEMPLATE_CROPS[TCG_NAMES[tcgId]]?.[track.templateId];
    if (!tmpl) return;
    for (const key of ["name", "collector_id", "set_text"]) {
      const s = track.sig[key];
      if (s.lock || !tmpl[key]) continue;
      const r = readOcr(track.box, tmpl[key]);
      if (!r || r.sharp < REGION_FOCUS_MIN || !r.text) continue;
      if (r.text === s.v) {
        if (++s.agree >= OCR_AGREE) s.lock = true;   // two sharp frames agree → lock
      } else {
        s.v = r.text; s.agree = 1;                   // adopt the newer/sharper read
      }
    }
    const ss = track.sig.set_symbol;
    if (!ss.lock && tmpl.set_symbol) {
      const r = readSymbol(track.box, tmpl.set_symbol);
      if (r && r.sharp >= REGION_FOCUS_MIN && r.conf > ss.conf) {
        ss.id = r.id; ss.conf = r.conf;
        if (r.conf >= SYMBOL_LOCK) ss.lock = true;
      }
    }
    if ((++frameNo) % 30 === 0) {
      const g = track.sig;
      console.log(`[signals] name="${g.name.v}"${g.name.lock ? "🔒" : ""} `
        + `id="${g.collector_id.v}"${g.collector_id.lock ? "🔒" : ""} `
        + `set="${g.set_text.v}"${g.set_text.lock ? "🔒" : ""} `
        + `sym=${g.set_symbol.id}@${g.set_symbol.conf.toFixed(2)}${g.set_symbol.lock ? "🔒" : ""}`);
    }
  }

  // Re-render only when a displayed signal actually changes (keeps it static).
  function renderTrack(track) {
    const g = track.sig;
    const key = `${track.templateId}|${g.name.v}|${g.collector_id.v}|${g.set_text.v}|${g.set_symbol.id}`;
    if (key === lastRenderKey) return;
    lastRenderKey = key;
    if (!g.name.v && !g.collector_id.v && !g.set_text.v && g.set_symbol.id < 0) {
      renderFocusing();
      return;
    }
    updateMatches({
      x: track.box.x, y: track.box.y, w: track.box.w, h: track.box.h,
      templateId: track.templateId, templateConf: track.templateConf, borderId: track.borderId,
      name: g.name.v, collectorId: g.collector_id.v, setText: g.set_text.v,
      symbolId: g.set_symbol.id, symbolConf: g.set_symbol.conf,
    });
  }

  // Variance-of-Laplacian over RGBA pixels — high == sharp, low == blurry.
  function laplacianVar(data, w, h) {
    const g = new Float32Array(w * h);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    }
    let sum = 0, sum2 = 0, n = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const lap = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
        sum += lap; sum2 += lap * lap; n++;
      }
    }
    if (n === 0) return 0;
    const mean = sum / n;
    return sum2 / n - mean * mean;
  }

  function iou(a, b) {
    const ix0 = Math.max(a.x, b.x), iy0 = Math.max(a.y, b.y);
    const ix1 = Math.min(a.x + a.w, b.x + b.w), iy1 = Math.min(a.y + a.h, b.y + b.h);
    const iw = Math.max(0, ix1 - ix0), ih = Math.max(0, iy1 - iy0);
    const inter = iw * ih;
    const uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0;
  }

  function renderFocusing() {
    const panel = document.getElementById("matches");
    if (panel) panel.innerHTML = `<div class="text-amber-400/80 text-sm">Hold steady — focusing…</div>`;
  }

  // Crop a normalized-within-bbox region from the full-res frame into cropBuf.
  // Returns {w, h, sharp} for the crop placed in cropBuf, or null if too small.
  function cropRegionToBuf(d, rect) {
    let sx = Math.round(d.x + inset(rect[0]) * d.w);
    let sy = Math.round(d.y + inset(rect[1]) * d.h);
    let sw = Math.round(rect[2] * d.w * (1 - 2 * BBOX_INSET));
    let sh = Math.round(rect[3] * d.h * (1 - 2 * BBOX_INSET));
    sx = Math.max(0, Math.min(frameScratch.width - 1, sx));
    sy = Math.max(0, Math.min(frameScratch.height - 1, sy));
    sw = Math.min(frameScratch.width - sx, sw);
    sh = Math.min(frameScratch.height - sy, sh);
    if (sw < 4 || sh < 4) return null;
    // Cap to the crop buffer, preserving aspect.
    let dw = sw, dh = sh;
    if (dw * dh > CROP_MAX_PX) {
      const s = Math.sqrt(CROP_MAX_PX / (dw * dh));
      dw = Math.max(4, Math.floor(dw * s));
      dh = Math.max(4, Math.floor(dh * s));
    }
    if (regionCanvas.width !== dw || regionCanvas.height !== dh) {
      regionCanvas.width = dw; regionCanvas.height = dh;
    }
    const rctx = regionCanvas.getContext("2d", { willReadFrequently: true });
    rctx.drawImage(frameScratch, sx, sy, sw, sh, 0, 0, dw, dh);
    const img = rctx.getImageData(0, 0, dw, dh);
    HEAPU8.set(img.data, cropBuf);
    return { w: dw, h: dh, sharp: laplacianVar(img.data, dw, dh) };
  }

  // OCR one region from the full-res frame. Returns {text, sharp} or null.
  function readOcr(box, rect) {
    const c = cropRegionToBuf(box, rect);
    if (!c) return null;
    _ocr_region(cropBuf, c.w, c.h, ocrBuf, 64);
    return { text: decodeCStr(HEAPU8, ocrBuf, 64), sharp: c.sharp };
  }

  // Symbol-classify one region. Returns {id, conf, sharp} or null.
  function readSymbol(box, rect) {
    const c = cropRegionToBuf(box, rect);
    if (!c) return null;
    _symbol_region(tcgId, cropBuf, c.w, c.h, symBuf);
    const dv = new DataView(HEAPU8.buffer, symBuf, 8);
    return { id: dv.getInt32(0, true), conf: dv.getFloat32(4, true), sharp: c.sharp };
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
      const sc = 1 - 2 * BBOX_INSET;
      ctx.strokeStyle = info.color;
      ctx.strokeRect(inset(rx) * targetW, inset(ry) * targetH, rw * sc * targetW, rh * sc * targetH);
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
      const sc = 1 - 2 * BBOX_INSET;
      const sx = d.x + inset(rx) * d.w;
      const sy = d.y + inset(ry) * d.h;
      const sw = rw * sc * d.w;
      const sh = rh * sc * d.h;
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
