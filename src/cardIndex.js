// Per-TCG card index loader. Fetches static/index/{tcg}.json on first
// selection and caches it. Each entry is the slim record produced by
// build_index.py — see that file's docstring for the schema.

const CACHE = { mtg: null, pokemon: null, yugioh: null };
const PENDING = { mtg: null, pokemon: null, yugioh: null };

export async function loadIndex(tcgName) {
  if (CACHE[tcgName]) return CACHE[tcgName];
  if (PENDING[tcgName]) return PENDING[tcgName];
  PENDING[tcgName] = (async () => {
    const res = await fetch(`static/index/${tcgName}.json`);
    if (!res.ok) throw new Error(`failed to load ${tcgName} index: ${res.status}`);
    const data = await res.json();
    // Precompute lowercased norm fields for fuzzy match.
    for (const e of data.entries) {
      e._n = normalize(e.n);
      e._id = normalize(e.id);
    }
    CACHE[tcgName] = data;
    PENDING[tcgName] = null;
    return data;
  })();
  return PENDING[tcgName];
}

export function getCachedIndex(tcgName) {
  return CACHE[tcgName];
}

// Card-name OCR drops spaces inconsistently (kerning, low res, font weight) —
// so we strip *all* non-alphanumerics including spaces. Cards basically never
// rely on tokenization for identity ("Wurm Massacre" isn't a card), so we
// trade tokenSetRatio power for robust prelim filtering.
export function normalize(s) {
  if (!s) return "";
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, "");
}
