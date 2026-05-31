// Fuzzy match a card detection's OCR signals against a TCG index, applying
// the Python pipeline's per-TCG signal bonuses (symbol/border/family/foil).
//
// Faithful in structure to create_paddle_dataset/test_pipeline.py:fuzzy_match
// but simplified in the scorer:
//   * Python uses rapidfuzz WRatio (a max-blend of ratio/partial_ratio/
//     token_set/token_sort). We approximate with token-set Jaccard +
//     Levenshtein ratio, which catches the same dominant cases for card
//     names + short collector IDs.
//   * print_year and copyright_region signals are deferred (the index doesn't
//     carry released_at to keep size small).
//
// Returns top-K [{entry, name, id, combined}, ...] sorted by combined desc.

import { normalize } from "./cardIndex.js";

const MTG_BASIC_LANDS = new Set([
  "plains", "island", "swamp", "mountain", "forest", "wastes",
]);

// Per-TCG signal weights — mirrors _TCG_WEIGHTS in test_pipeline.py.
const TCG_WEIGHTS = {
  mtg: {
    name_hi: 0.7, name_lo: 0.3,
    // symbol classifier is unreliable on in-the-wild camera crops — keep the
    // bonus for confidently-correct hits, but raise the threshold and lower
    // the penalty so a confidently-wrong prediction barely hurts.
    symbol_bonus: 15, symbol_penalty: 3, symbol_penalty_conf: 0.95,
    set_text_bonus: 12, set_text_penalty: 8,
    border_bonus: 15,
    family_bonus: 3,
    foil_bonus: 4,
    basic_land_set_mult: 3.0,
  },
  pokemon: {
    name_hi: 0.7, name_lo: 0.3,
    symbol_bonus: 12, symbol_penalty: 2, symbol_penalty_conf: 0.95,
    set_text_bonus: 0, set_text_penalty: 0,
    border_bonus: 0,
    family_bonus: 0,
    foil_bonus: 0,
    basic_land_set_mult: 1.0,
  },
  yugioh: {
    name_hi: 0.7, name_lo: 0.3,
    symbol_bonus: 0, symbol_penalty: 0, symbol_penalty_conf: 1,
    set_text_bonus: 0, set_text_penalty: 0,
    border_bonus: 0,
    family_bonus: 0,
    foil_bonus: 0,
    basic_land_set_mult: 1.0,
  },
};

export function fuzzyMatch({
  tcg, indexData, ocrName, ocrId, ocrSetText,
  predictedSet,       // string set code or null
  predictedSetConf,   // 0..1
  predictedFamily,    // MTG template family ("1993", "2003", ...) or null
  sampledBorder,      // "black" / "white" / null
  topK = 5,
}) {
  if (!indexData) return [];
  const entries = indexData.entries;
  const w = TCG_WEIGHTS[tcg] || TCG_WEIGHTS.mtg;

  const nQuery = normalize(ocrName);
  let iQuery = normalize(ocrId);
  // MTG ID frequently leaks rarity letters / "Illus." fragments — extract
  // the canonical "n" or "n/total" digit run.
  if (tcg === "mtg" && iQuery) {
    const m = iQuery.match(/(\d+)(?:\s*\/\s*(\d+))?/);
    if (m) iQuery = m[2] ? `${m[1]}/${m[2]}` : m[1];
  }
  const hasName = !!nQuery;
  // Drop degenerate id reads (single char, all-same digit) — they otherwise
  // drag the combined score down via the lo-blend even when name matches
  // perfectly. Real collector numbers are 2+ chars and not "00" / "11" /
  // "999" / "aaa".
  const hasId = !!iQuery && iQuery.length >= 2 && !/^(.)\1+$/.test(iQuery);

  // Candidate pool: take top 200 by name AND top 200 by id (union). Cheap
  // prelim using substring contains; cheap and good enough for v1.
  // For MTG + Pokemon also pull in every card whose set matches a signalled set.
  const pool = new Set();
  if (hasName) addPrelim(pool, entries, nQuery, "_n", 200);
  if (hasId) addPrelim(pool, entries, iQuery, "_id", 200);

  if ((tcg === "mtg" || tcg === "pokemon") && indexData.by_set) {
    const signalled = new Set();
    if (ocrSetText) signalled.add(normalize(ocrSetText));
    if (predictedSet && predictedSetConf >= 0.5) signalled.add(predictedSet);
    for (const code of signalled) {
      const ids = indexData.by_set[code];
      if (ids) for (const i of ids) pool.add(i);
    }
  }

  if (pool.size === 0) for (let i = 0; i < entries.length; i++) pool.add(i);

  const isBasicLand = tcg === "mtg" && MTG_BASIC_LANDS.has(nQuery);
  const setMult = isBasicLand ? w.basic_land_set_mult : 1.0;
  const setTextCode = ocrSetText ? normalize(ocrSetText) : null;

  const scored = [];
  for (const i of pool) {
    const e = entries[i];
    const nScore = hasName && e._n ? wRatio(nQuery, e._n) : 0;
    const iScore = hasId && e._id ? wRatio(iQuery, e._id) : 0;

    let combined;
    if (hasName && hasId && e._n && e._id) {
      const hi = Math.max(nScore, iScore), lo = Math.min(nScore, iScore);
      combined = w.name_hi * hi + w.name_lo * lo;
    } else if (e._n && hasName) {
      combined = nScore;
    } else if (e._id && hasId) {
      combined = iScore;
    } else {
      combined = 0;
    }

    if (predictedSet && e.set) {
      if (e.set === predictedSet) {
        combined += w.symbol_bonus * predictedSetConf * setMult;
      } else if (predictedSetConf >= w.symbol_penalty_conf) {
        combined -= w.symbol_penalty * predictedSetConf;
      }
    }
    if (setTextCode && e.set) {
      if (e.set === setTextCode) combined += w.set_text_bonus * setMult;
      else                       combined -= w.set_text_penalty;
    }
    if (tcg === "mtg" && sampledBorder && e.border
        && ["1993", "1997", "2003"].includes(predictedFamily)) {
      if (e.border === sampledBorder) combined += w.border_bonus;
    }
    if (tcg === "mtg" && predictedFamily && e.tpl
        && predictedFamily === e.tpl) {
      combined += w.family_bonus;
    }

    scored.push({ entry: e, name: nScore, id: iScore, combined });
  }

  scored.sort((a, b) => b.combined - a.combined);
  return scored.slice(0, topK);
}

function addPrelim(pool, entries, query, field, limit) {
  // Two-pass cheap prelim: first any entry whose field contains the query
  // (or vice versa), then if we still need more, take any entry sharing the
  // query's first token.
  const firstTok = query.split(" ")[0];
  let added = 0;
  for (let i = 0; i < entries.length && added < limit; i++) {
    const v = entries[i][field];
    if (!v) continue;
    if (v.includes(query) || query.includes(v)) {
      if (!pool.has(i)) { pool.add(i); added++; }
    }
  }
  if (added >= limit || !firstTok) return;
  for (let i = 0; i < entries.length && added < limit; i++) {
    const v = entries[i][field];
    if (!v) continue;
    if (v.startsWith(firstTok)) {
      if (!pool.has(i)) { pool.add(i); added++; }
    }
  }
}

// Approximation of rapidfuzz fuzz.WRatio. Blends ratio + token-set ratio;
// returns a 0..100 score. Not bit-identical to Python but rank-correlated
// for the cases that matter (real card names vs misreads).
function wRatio(a, b) {
  if (!a || !b) return 0;
  const r = ratio(a, b);
  const t = tokenSetRatio(a, b);
  return Math.max(r, t * 0.95);
}

function ratio(a, b) {
  if (a === b) return 100;
  const d = levenshtein(a, b);
  const m = Math.max(a.length, b.length);
  return m ? 100 * (1 - d / m) : 100;
}

function tokenSetRatio(a, b) {
  const setA = new Set(a.split(" ").filter(Boolean));
  const setB = new Set(b.split(" ").filter(Boolean));
  const inter = [...setA].filter(x => setB.has(x));
  const onlyA = [...setA].filter(x => !setB.has(x));
  const onlyB = [...setB].filter(x => !setA.has(x));
  const s = inter.sort().join(" ");
  const sa = [...inter.sort(), ...onlyA.sort()].join(" ");
  const sb = [...inter.sort(), ...onlyB.sort()].join(" ");
  // rapidfuzz takes max of three ratios over these reorderings.
  return Math.max(ratio(s, sa), ratio(s, sb), ratio(sa, sb));
}

// Levenshtein distance — O(a.length * b.length) with single-row rolling.
function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const prev = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    let prevDiag = prev[0];
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j];
      const cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
      prev[j] = Math.min(
        prev[j] + 1,        // deletion
        prev[j - 1] + 1,    // insertion
        prevDiag + cost,    // substitution
      );
      prevDiag = tmp;
    }
  }
  return prev[b.length];
}
