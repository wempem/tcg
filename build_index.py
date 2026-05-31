"""Build per-TCG card index JSONs for the browser fuzzy-match step.

Walks each TCG's dataset directories under generate_{tcg}_set/ and emits a
compact JSON file under static/index/{tcg}.json:

    {
      "tcg": "mtg",
      "entries": [
        {"n": "Lightning Bolt", "id": "161", "set": "lea", "tpl": "1993",
         "key": "<oracle-uuid>", "border": "black", "foil": false}, ...
      ],
      "by_set": {"lea": [0, 7, 12], ...}
    }

Schema notes
------------
- `n` / `id`: printed name + collector identifier as the OCR would see them
  (derived via card_metadata.canonical_*).
- `set`: lowercased set code — used to match against the symbol classifier's
  predicted set and against OCR'd set_text.
- `tpl`: MTG frame-era family (1993 / 1997 / 2003 / 2015 / future). Null for
  Pokemon/Yugioh.
- `key`: card-identity key (MTG oracle_id, Yugioh card-name hash, Pokemon
  card_id). Used to dedupe top-K — two entries with the same key are the
  same card (different printings).
- `border`: "black" / "white" / null (MTG only).
- `foil`: true iff the printing's finishes set is {"foil"} only (MTG only).

Mirrors create_paddle_dataset/test_pipeline.py:build_search_index but
without the heavy torch/paddle imports — runs under uv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ML = REPO_ROOT.parent

# Reuse the existing per-TCG metadata extractors + cropper routing.
sys.path.insert(0, str(ML / "create_paddle_dataset"))
sys.path.insert(0, str(ML / "identify_bounding_box"))

from crop_mapper import CropMapper  # noqa: E402
from crop_and_label import DATASET_DIRS, load_metadata  # noqa: E402
from card_metadata import canonical_identifier, canonical_name  # noqa: E402

TCG_SOURCES = {
    "pokemon": ["pokemon"],
    "mtg": ["mtg"],
    # ygoprodeck = digital renders without printed set codes, so we drop them
    # for the search index (same call test_pipeline.py makes).
    "yugioh": ["yugioh"],
}


def is_inference_eligible(tcg: str, metadata: dict) -> bool:
    """Port of test_pipeline.is_inference_eligible — keep in sync."""
    if tcg == "mtg":
        if (metadata.get("lang") or "en") != "en":
            return False
        if metadata.get("image_status") in ("placeholder", "missing"):
            return False
        if metadata.get("digital"):
            return False
        if metadata.get("oversized"):
            return False
        if metadata.get("layout") in (
            "token", "emblem", "double_faced_token", "art_series",
        ):
            return False
    return True


def mtg_template_family(template_name: str | None) -> str | None:
    """Port of test_pipeline.mtg_template_family — frame-era family for MTG."""
    if not template_name:
        return None
    parts = template_name.split("_")
    if len(parts) < 2 or parts[0] != "mtg":
        return None
    era = parts[1]
    if era == "exclude":
        return None
    if era == "modern":
        return "2003"
    return era


def _identity_key(source: str, card_id: str, metadata: dict) -> str:
    if source == "mtg":
        return metadata.get("oracle_id") or card_id
    if source in ("yugioh", "yugioh_ygoprodeck"):
        nm = (metadata.get("card_name")
              or (metadata.get("card") or {}).get("name")
              or metadata.get("name") or "")
        return f"name:{nm.strip().lower()}" if nm.strip() else card_id
    return card_id  # Pokemon: each printing is its own identity


def build_for_tcg(tcg: str, mapper: CropMapper) -> dict:
    entries = []
    by_set: dict[str, list[int]] = {}
    for source in TCG_SOURCES[tcg]:
        d = DATASET_DIRS[source]
        if not d.exists():
            print(f"  warning: {d} does not exist, skipping", file=sys.stderr)
            continue
        for card_dir in sorted(d.iterdir()):
            if not card_dir.is_dir():
                continue
            card_id = card_dir.name
            metadata = load_metadata(source, card_id)
            if metadata is None:
                continue
            if not is_inference_eligible(source, metadata):
                continue
            crop_game = "yugioh" if source == "yugioh_ygoprodeck" else source
            gt_template = mapper.resolve_template(crop_game, metadata)
            name = canonical_name(source, metadata, gt_template or "")
            ident = canonical_identifier(source, metadata, gt_template or "")
            if not name and not ident:
                continue
            if source == "mtg":
                set_code = (metadata.get("set") or "").strip().lower() or None
            elif source == "pokemon":
                set_code = (metadata.get("_set", {}).get("id") or "").strip().lower() or None
            else:
                set_code = None
            tpl = mtg_template_family(gt_template) if source == "mtg" else None
            entry = {
                "n": name or "",
                "id": ident or "",
                "set": set_code,
                "tpl": tpl,
                "key": _identity_key(source, card_id, metadata),
            }
            if source == "mtg":
                entry["border"] = (metadata.get("border_color") or "").lower() or None
                finishes = set(metadata.get("finishes") or [])
                entry["foil"] = finishes == {"foil"}
            idx = len(entries)
            entries.append(entry)
            if set_code:
                by_set.setdefault(set_code, []).append(idx)
    return {"tcg": tcg, "entries": entries, "by_set": by_set}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tcg", choices=["mtg", "pokemon", "yugioh", "all"],
                   default="all")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "static" / "index")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapper = CropMapper()  # default configs (per-TCG set_config.json)

    tcgs = ["mtg", "pokemon", "yugioh"] if args.tcg == "all" else [args.tcg]
    for tcg in tcgs:
        print(f"=== {tcg} ===")
        data = build_for_tcg(tcg, mapper)
        out = args.output_dir / f"{tcg}.json"
        with open(out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        sets = len(data["by_set"])
        print(f"  {len(data['entries'])} entries across {sets} sets -> {out} "
              f"({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
