"""Stage trained set-symbol classifiers into the browser demo — single source of truth.

WHY THIS EXISTS (2026-06-03): the MTG symbol model was retrained 311->167 classes,
re-indexing every class. The ncnn model in cards.data was refreshed but the JS-side
label map `static/labels/symbol_mtg_labels.json` was NOT, so detector.js translated
the model's correct index 94 (mbs) through a stale 311-class map where 94 == hop.
The live demo showed `hop` on a correctly-classified card for days.

Root cause: staging was a hand-run list of `cp` commands (see AGENT_HANDOFF.md) and
the `static/labels/` copy was simply forgotten. There are TWO label destinations in
TWO formats and they MUST agree:

  - static/labels/symbol_{tcg}_labels.json   {tcg, label_to_idx, distribution}
        ^ THE load-bearing one. detector.js loadSymbolLabels() fetches it at runtime
          and inverts label_to_idx to turn the model's argmax index into a set name.
  - wasm_ncnn/assets/symbol_{tcg}_labels.json {idx_to_label, input_size}
        ^ bundled into cards.data, but main.cpp NEVER reads it (it returns the raw
          argmax index). Kept only for parity/debugging.

This script derives BOTH from the one source the trainer writes
(symbol_classifier/weights/{tcg}/labels.json), and copies the ncnn .param/.bin, so
the model and its label map can never again be staged out of sync. Run it after every
`export_ncnn.py`, before rebuilding the WASM.

Usage:
    python3 stage_symbol_models.py            # all tcgs
    python3 stage_symbol_models.py --tcg mtg  # one
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ML_TRAINING_DIR = DEMO_DIR.parent
WEIGHTS_ROOT = ML_TRAINING_DIR / "symbol_classifier" / "weights"
ASSETS_DIR = DEMO_DIR / "wasm_ncnn" / "assets"
STATIC_LABELS_DIR = DEMO_DIR / "static" / "labels"

# C++ main.cpp loads symbol classifiers at this fixed input size (SYMBOL_INPUT).
SYMBOL_INPUT = 320

# TCGs with a symbol classifier (yugioh has none).
TCGS = ("mtg", "pokemon")


def stage_one(tcg: str) -> None:
    src = WEIGHTS_ROOT / tcg
    labels_path = src / "labels.json"          # written by symbol_classifier/fit.py
    param_path = src / "classifier.ncnn.param"
    bin_path = src / "classifier.ncnn.bin"
    for p in (labels_path, param_path, bin_path):
        if not p.exists():
            raise SystemExit(f"[{tcg}] missing {p} — train + export_ncnn first")

    labels = json.loads(labels_path.read_text())
    label_to_idx = labels["label_to_idx"]
    n = len(label_to_idx)

    # Invariant: indices must be a contiguous 0..n-1 permutation. A stale or
    # half-merged weights dir would violate this and silently mistranslate.
    idxs = sorted(label_to_idx.values())
    if idxs != list(range(n)):
        raise SystemExit(f"[{tcg}] label_to_idx indices are not 0..{n-1} contiguous — "
                         f"refusing to stage a corrupt map ({labels_path})")
    idx_to_label = {str(i): name for name, i in label_to_idx.items()}

    # 1. ncnn model -> assets (what cards.data actually runs)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(param_path, ASSETS_DIR / f"symbol_{tcg}.param")
    shutil.copy2(bin_path, ASSETS_DIR / f"symbol_{tcg}.bin")

    # 2. JS label map -> static/labels  (THE load-bearing file; copy as-is)
    STATIC_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, STATIC_LABELS_DIR / f"symbol_{tcg}_labels.json")

    # 3. bundled label map -> assets (parity/debug only; idx_to_label + input_size)
    (ASSETS_DIR / f"symbol_{tcg}_labels.json").write_text(
        json.dumps({"idx_to_label": idx_to_label, "input_size": SYMBOL_INPUT}, indent=2)
    )

    sample = idx_to_label.get("94") or idx_to_label.get("0")
    print(f"[{tcg}] staged {n} classes  (idx94={idx_to_label.get('94')!r})")
    print(f"        -> {STATIC_LABELS_DIR / f'symbol_{tcg}_labels.json'}")
    print(f"        -> {ASSETS_DIR / f'symbol_{tcg}.{{param,bin}}'} + labels")
    _ = sample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tcg", choices=TCGS, help="stage one tcg (default: all)")
    args = p.parse_args()
    targets = [args.tcg] if args.tcg else list(TCGS)
    for tcg in targets:
        stage_one(tcg)
    print("\nDone. Now rebuild the WASM (emmake make) and copy cards.* to static/model/,"
          "\nthen commit static/labels/ + static/model/ together.")


if __name__ == "__main__":
    main()
