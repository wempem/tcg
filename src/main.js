import { initWasm } from "./wasmLoader.js";
import { initCamera } from "./camera.js";
import { createDetector } from "./detector.js";
import { detectWasmFeatures } from "./wasmFeatureDetect.js";

const CONF_THRESHOLD = 0.25;

const TCG_IDS = { mtg: 0, pokemon: 1, yugioh: 2 };

let detector;

document.addEventListener("DOMContentLoaded", async () => {
  const features = await detectWasmFeatures();
  console.log("WASM Features:", features);

  await initWasm();

  detector = createDetector({ confidenceThreshold: CONF_THRESHOLD });
  detector.setTcg(0);  // MTG default; also kicks off the index fetch
  wireTcgSelector(detector);

  await initCamera((ctx, canvas) => {
    detector.processFrame(ctx, canvas);
  });
});

function wireTcgSelector(det) {
  const buttons = document.querySelectorAll("[data-tcg]");
  for (const btn of buttons) {
    btn.addEventListener("click", () => {
      const tcg = btn.dataset.tcg;
      det.setTcg(TCG_IDS[tcg]);
      for (const b of buttons) b.classList.toggle("active", b === btn);
    });
  }
}
