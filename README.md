# Card Browser Demo

Live trading-card identifier that runs **entirely in the browser** — no server, no cloud. Point your camera at an MTG / Pokemon / Yugioh card and watch the pipeline detect, classify, OCR, and match it against ~150,000 known cards.

[**Live demo →**](https://wempem.github.io/tcg/) *(takes ~30s to download the model bundle on first load)*

![pipeline overview](./docs/pipeline-overview.png) <!-- optional; add later -->

## What it does

```
camera → YOLO card detector → template classifier → percent-region crops
       → PaddleOCR text recognition (name + collector_id + set_text)
       → set-symbol CNN classifier
       → border-color sampler (pure pixel math)
       → fuzzy match against per-TCG card index (~30k MTG / ~30k Pokemon / ~45k Yu-Gi-Oh entries)
       → top-5 candidates
```

Five neural networks plus a fuzzy matcher, ~28 MB of compiled assets, all on-device via [ncnn](https://github.com/Tencent/ncnn) + WebAssembly.

## Running locally

```bash
python3 -m http.server 8080
# open http://localhost:8080
```

Camera permission required (HTTPS or localhost). The page asks for a rear-facing camera (`facingMode: "environment"`) but falls back to any available device.

## Architecture

The HTML page loads a tiny ES-module entry point (`src/main.js`) which initializes the WASM module, wires the TCG selector, and starts the camera capture loop. Each frame:

1. `src/camera.js` draws the video onto a 320×320 canvas and hands the `ImageData` to the detector.
2. `src/detector.js` copies the pixels into the WASM heap and calls `_process_frame(buf, w, h, tcg_id, conf_thresh, result_buf)`.
3. The C++ module (`wasm_ncnn/src/main.cpp`) runs YOLO → per-TCG template classifier → percent-region crops → OCR over text crops + symbol classifier over the symbol crop + edge-pixel border sampler, and writes 8 × 256-byte detection slots into the result buffer.
4. JS decodes the slots, draws the bounding box on the canvas, and (when a card is detected) renders the "What the pipeline saw" debug panel + runs the fuzzy matcher against the prebuilt per-TCG index.

The JS ↔ C++ contract is a fixed 256-byte `Detection` slot. All compute happens client-side; only the static asset bundle and per-TCG card indexes are fetched.

## Repo layout

```
card_browser_demo/
├── index.html                  # entry page
├── src/                        # ES module sources
│   ├── main.js                 # bootstrap (camera + WASM + TCG selector)
│   ├── camera.js               # getUserMedia + canvas capture loop
│   ├── detector.js             # WASM bridge, Detection-slot decoder, UI renderer
│   ├── wasmLoader.js           # loads cards.{wasm,js,data} from /static/model/
│   ├── wasmFeatureDetect.js    # SIMD / threads feature detection
│   ├── cardIndex.js            # lazy per-TCG card index loader
│   ├── fuzzyMatch.js           # rapidfuzz-style scoring over OCR + signals
│   └── cropRegions.js          # auto-generated; mirrors crop_regions.h
├── static/
│   ├── model/                  # built WASM bundle (cards.wasm + .js + .data)
│   └── index/                  # prebuilt per-TCG card indexes (JSON)
├── wasm_ncnn/
│   ├── CMakeLists.txt          # emcmake target
│   ├── include/                # stb_image headers + auto-gen crop_regions.h
│   └── src/main.cpp            # the C++ pipeline
├── codegen_crops.py            # generates crop_regions.h + cropRegions.js
├── build_index.py              # walks generate_*_set/dataset and emits static/index/*.json
└── README.md
```

## Building from source

Requires [emscripten](https://emscripten.org/) and a prebuilt ncnn-wasm checkout (`simd` variant). On the original dev box those are at:

```
~/projects/ml_training/emsdk
~/projects/ml_training/ncnn-20260113-webassembly/simd
```

```bash
# one-time
ln -sfn /path/to/ncnn-20260113-webassembly/simd wasm_ncnn/ncnn

# WASM build
source /path/to/emsdk/emsdk_env.sh
cd wasm_ncnn && mkdir -p build && cd build
emcmake cmake ..
emmake make -j$(nproc)
cp cards.wasm cards.js cards.data ../../static/model/
```

Model assets live under `wasm_ncnn/assets/` and get bundled into `cards.data` via emscripten's `--preload-file`. They aren't checked into git (regenerable from the training repos elsewhere); see `wasm_ncnn/assets/` in the source tree for the file list.

## Credits

- [ncnn](https://github.com/Tencent/ncnn) — neural network inference framework
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — text recognition
- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- Card scans from [Scryfall](https://scryfall.com/) (MTG), [PokemonTCG API](https://pokemontcg.io/) (Pokemon), [Yugipedia](https://yugipedia.com/) (Yu-Gi-Oh)

License: TBD.
