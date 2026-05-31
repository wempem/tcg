"""Generate wasm_ncnn/include/crop_regions.h from per-TCG crop configs.

Joins each TCG's classifier label JSON (idx_to_label) with its set_config
template definitions, producing a `lookup_template(tcg_id, template_id)`
function that returns the four crop sub-regions used downstream:

  name         — OCR'd text card name
  collector_id — MTG `set_id_region` ∪ Pokemon/Yugioh `collector_id_region`
  set_text     — MTG only (`set_text_region`)
  set_symbol   — MTG + Pokemon (`set_symbol_region`; Yugioh has none)

Region rectangles are normalized 0..1 — the C++ side multiplies by the YOLO
bbox dimensions to land them on the cropped card.

Wired into CMakeLists.txt via add_custom_command so the header regenerates
whenever a config or label file changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # card_browser_demo/
ASSETS_DIR = REPO_ROOT / "wasm_ncnn" / "assets"
OUTPUT_HEADER = REPO_ROOT / "wasm_ncnn" / "include" / "crop_regions.h"
OUTPUT_JS = REPO_ROOT / "src" / "cropRegions.js"
CONFIGS_ROOT = REPO_ROOT.parent / "identify_bounding_box"

# Same canonical (tcg_id, region-source-field) mapping the inference C++ uses.
TCGS = [
    ("mtg", 0),
    ("pokemon", 1),
    ("yugioh", 2),
]


def load_label_map(tcg: str) -> dict[int, str]:
    """idx_to_label as int-keyed map (the JSON has string keys)."""
    path = ASSETS_DIR / f"template_{tcg}_labels.json"
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["idx_to_label"].items()}


def load_set_config(tcg: str) -> dict:
    path = CONFIGS_ROOT / tcg / f"{tcg}_set_config.json"
    with open(path) as f:
        return json.load(f)


def pct_to_norm(region: dict | None, margin: float = 0.0) -> tuple[float, float, float, float] | None:
    """Convert {x_pct, y_pct, width_pct, height_pct} (0..100) → 0..1 tuple.

    `margin` (also 0..1) expands the region by that fraction of the bbox on
    every side — used to absorb minor camera-side perspective drift so the
    OCR / symbol classifier still see their target even when the YOLO bbox
    isn't perfectly card-tight. Clamped to [0, 1].
    """
    if region is None:
        return None
    x = region["x_pct"] / 100.0 - margin
    y = region["y_pct"] / 100.0 - margin
    w = region["width_pct"] / 100.0 + 2 * margin
    h = region["height_pct"] / 100.0 + 2 * margin
    # Clamp into the unit bbox.
    if x < 0: w += x; x = 0
    if y < 0: h += y; y = 0
    if x + w > 1: w = 1 - x
    if y + h > 1: h = 1 - y
    return (x, y, w, h)


def resolve_region(template: dict, *aliases: str) -> dict | None:
    """Return the first present region under any of `aliases`."""
    for a in aliases:
        if a in template and template[a] is not None:
            return template[a]
    return None


def template_regions(tcg: str, template: dict, margin: float = 0.0) -> dict:
    """Return the 4 unified regions for one template, or None where absent.

    - name: name_region (universal)
    - collector_id: MTG → set_id_region; Pokemon/Yugioh → collector_id_region
    - set_text: MTG → set_text_region; others → None
    - set_symbol: MTG/Pokemon → set_symbol_region; Yugioh → None

    `margin` is applied to every region via pct_to_norm.
    """
    name = resolve_region(template, "name_region")
    if tcg == "mtg":
        collector_id = resolve_region(template, "set_id_region")
        set_text = resolve_region(template, "set_text_region")
        set_symbol = resolve_region(template, "set_symbol_region")
    elif tcg == "pokemon":
        collector_id = resolve_region(template, "collector_id_region")
        set_text = None
        set_symbol = resolve_region(template, "set_symbol_region")
    elif tcg == "yugioh":
        collector_id = resolve_region(template, "collector_id_region")
        set_text = None
        set_symbol = None
    else:
        raise ValueError(f"unknown tcg {tcg}")
    return {
        "name": pct_to_norm(name, margin),
        "collector_id": pct_to_norm(collector_id, margin),
        "set_text": pct_to_norm(set_text, margin),
        "set_symbol": pct_to_norm(set_symbol, margin),
    }


def format_region(r: tuple[float, float, float, float] | None) -> str:
    if r is None:
        return "{0.f, 0.f, 0.f, 0.f}"
    return "{%.4ff, %.4ff, %.4ff, %.4ff}" % r


def emit_template_entry(tcg: str, idx: int, label: str, regions: dict) -> str:
    has = {k: r is not None for k, r in regions.items()}
    return (
        f"    {{ /* {tcg}[{idx}] {label} */\n"
        f"      .name        = {format_region(regions['name'])},\n"
        f"      .collector_id= {format_region(regions['collector_id'])},\n"
        f"      .set_text    = {format_region(regions['set_text'])},\n"
        f"      .set_symbol  = {format_region(regions['set_symbol'])},\n"
        f"      .has_name        = {int(has['name'])},\n"
        f"      .has_collector_id= {int(has['collector_id'])},\n"
        f"      .has_set_text    = {int(has['set_text'])},\n"
        f"      .has_set_symbol  = {int(has['set_symbol'])},\n"
        f"      .family = \"{label}\",\n"
        f"    }},\n"
    )


def emit_header(per_tcg: dict[str, list[tuple[int, str, dict]]]) -> str:
    out = [
        "// AUTO-GENERATED by card_browser_demo/codegen_crops.py — do not edit.\n",
        "// Joins template_{tcg}_labels.json with identify_bounding_box/{tcg}/{tcg}_set_config.json.\n",
        "//\n",
        "// Region rects are normalized 0..1; multiply by the YOLO bbox W/H to land them.\n",
        "#pragma once\n",
        "\n",
        "#include <cstddef>\n",
        "\n",
        "struct CropRegion {\n",
        "    float x, y, w, h;\n",
        "};\n",
        "\n",
        "struct TemplateCrops {\n",
        "    CropRegion name;\n",
        "    CropRegion collector_id;\n",
        "    CropRegion set_text;\n",
        "    CropRegion set_symbol;\n",
        "    int has_name;\n",
        "    int has_collector_id;\n",
        "    int has_set_text;\n",
        "    int has_set_symbol;\n",
        "    const char* family;\n",
        "};\n",
        "\n",
    ]

    for tcg, _ in TCGS:
        entries = per_tcg[tcg]
        out.append(f"static const TemplateCrops k_templates_{tcg}[] = {{\n")
        for idx, label, regions in entries:
            out.append(emit_template_entry(tcg, idx, label, regions))
        out.append("};\n")
        out.append(f"static const size_t k_n_templates_{tcg} = "
                   f"sizeof(k_templates_{tcg}) / sizeof(k_templates_{tcg}[0]);\n\n")

    out.append("inline const TemplateCrops* lookup_template(int tcg_id, int template_id) {\n")
    out.append("    if (template_id < 0) return nullptr;\n")
    for tcg, tcg_id in TCGS:
        out.append(
            f"    if (tcg_id == {tcg_id}) {{\n"
            f"        if ((size_t)template_id >= k_n_templates_{tcg}) return nullptr;\n"
            f"        return &k_templates_{tcg}[template_id];\n"
            f"    }}\n"
        )
    out.append("    return nullptr;\n")
    out.append("}\n")
    return "".join(out)


def emit_js(per_tcg: dict[str, list[tuple[int, str, dict]]]) -> str:
    """Same table the C++ header carries, exposed for the JS overlay drawer.

    JS does its own per-detection crop-rect lookup so it can draw the four
    sub-regions inside the YOLO bbox without round-tripping pixel data.
    """
    out = ["// AUTO-GENERATED by card_browser_demo/codegen_crops.py — do not edit.\n",
           "// Mirrors wasm_ncnn/include/crop_regions.h for JS-side overlay drawing.\n",
           "// Region rects are 0..1 normalized relative to the YOLO bbox.\n",
           "\n",
           "export const TEMPLATE_CROPS = {\n"]
    for tcg, _ in TCGS:
        out.append(f"  {tcg}: [\n")
        for idx, label, regions in per_tcg[tcg]:
            r = regions
            out.append(
                f"    {{idx: {idx}, family: {label!r}, "
                f"name: {js_region(r['name'])}, "
                f"collector_id: {js_region(r['collector_id'])}, "
                f"set_text: {js_region(r['set_text'])}, "
                f"set_symbol: {js_region(r['set_symbol'])}}},\n"
            )
        out.append("  ],\n")
    out.append("};\n\n")
    out.append("export const TCG_IDS = { mtg: 0, pokemon: 1, yugioh: 2 };\n")
    out.append("export const TCG_NAMES = ['mtg', 'pokemon', 'yugioh'];\n")
    return "".join(out)


def js_region(r: tuple[float, float, float, float] | None) -> str:
    if r is None:
        return "null"
    return "[%.4f,%.4f,%.4f,%.4f]" % r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-header", type=Path, default=OUTPUT_HEADER)
    p.add_argument("--output-js", type=Path, default=OUTPUT_JS)
    p.add_argument("--margin", type=float, default=0.0,
                   help="Per-side expansion (0..1) applied to every crop "
                        "region. Default 0.0 - emits raw configured regions.")
    args = p.parse_args()

    per_tcg: dict[str, list[tuple[int, str, dict]]] = {}
    for tcg, _ in TCGS:
        labels = load_label_map(tcg)
        cfg = load_set_config(tcg)
        templates_by_name = cfg["templates"]

        entries = []
        for idx in sorted(labels):
            label = labels[idx]
            tmpl = templates_by_name.get(label)
            if tmpl is None:
                # Classifier knows about a template the config doesn't define
                # (e.g. legacy class still in the model). Emit a no-op row so
                # the index still aligns.
                regions = {"name": None, "collector_id": None,
                           "set_text": None, "set_symbol": None}
            else:
                regions = template_regions(tcg, tmpl, margin=args.margin)
            entries.append((idx, label, regions))
        per_tcg[tcg] = entries
        print(f"  {tcg}: {len(entries)} templates")

    text = emit_header(per_tcg)
    args.output_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_header.write_text(text)
    print(f"wrote {args.output_header} ({len(text)} bytes)")

    js_text = emit_js(per_tcg)
    args.output_js.parent.mkdir(parents=True, exist_ok=True)
    args.output_js.write_text(js_text)
    print(f"wrote {args.output_js} ({len(js_text)} bytes)")


if __name__ == "__main__":
    main()
