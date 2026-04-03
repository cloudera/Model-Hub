#!/usr/bin/env python3
"""
Fetches Artificial Analysis intelligence scores for models in the Model Hub.
Run manually or in CI to refresh the data:
    python3 fetch-aa-scores.py
"""
import json, os, re, sys, glob, urllib.request
import yaml

AA_KEY   = os.environ.get("AA_API_KEY")
if not AA_KEY:
    sys.exit("AA_API_KEY environment variable is required")
AA_URL   = "https://artificialanalysis.ai/api/v2/data/llms/models"
OUT      = os.path.join(os.path.dirname(__file__), "aa-scores.json")
YAML_DIR = os.path.join(os.path.dirname(__file__), "models", "public")


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def tokens(s):
    return [t for t in norm(s).split() if len(t) > 1]


def token_overlap(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    shared = len(set(a_tokens) & set(b_tokens))
    return shared / max(len(a_tokens), len(b_tokens))


def hub_identifiers():
    """Return (display_names, model_ids) sets from all YAML files."""
    display_names, model_ids = set(), set()
    for path in glob.glob(os.path.join(YAML_DIR, "*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        for m in data.get("models", []):
            raw_variants = m.get("modelVariants") or []
            multi = len(raw_variants) > 1
            for v in raw_variants:
                name = (v.get("variantId") if multi else None) or m.get("displayName") or m.get("name") or ""
                if name:
                    display_names.add(name)
                for p in v.get("optimizationProfiles") or []:
                    for meta in (p.get("ngcMetadata") or {}).values():
                        mid = (meta or {}).get("model", "")
                        if mid:
                            model_ids.add(mid)
    return display_names, model_ids


def matches_hub(aa_creator, aa_slug, aa_name, display_names, model_ids):
    aa_slug_norm = aa_slug.replace(".", "-")
    aa_toks = set(tokens(aa_slug_norm))

    # Creator/slug token overlap
    for mid in model_ids:
        mid_norm = mid.lower().replace(".", "-")
        slash = mid_norm.find("/")
        hub_creator = mid_norm[:slash] if slash >= 0 else ""
        hub_slug = mid_norm[slash + 1:] if slash >= 0 else mid_norm
        hub_toks = tokens(hub_slug)
        if not hub_toks or not aa_toks:
            continue
        sc = len(aa_toks & set(hub_toks)) / max(len(aa_toks), len(hub_toks))
        if aa_creator == hub_creator and sc >= 0.6:
            return True
        if sc >= 0.8:  # Very high overlap allows cross-creator (e.g., no-slash modelIds)
            return True

    # Display name fallback
    aa_name_toks = tokens(aa_name)
    for dn in display_names:
        dn_toks = tokens(dn)
        if not dn_toks:
            continue
        shared = set(aa_name_toks) & set(dn_toks)
        sc = len(shared) / max(len(aa_name_toks), len(dn_toks)) if aa_name_toks else 0
        # Short hub names (1 distinctive token): match if that token appears in the AA slug
        if len(dn_toks) == 1 and dn_toks[0] in aa_toks:
            return True
        # Longer names: require >=2 shared tokens and a strong ratio
        if len(shared) >= 2 and sc >= 0.6:
            return True

    return False


display_names, model_ids = hub_identifiers()
print(f"Hub models: {len(display_names)} display names, {len(model_ids)} model IDs")

req = urllib.request.Request(AA_URL, headers={"x-api-key": AA_KEY})
try:
    with urllib.request.urlopen(req) as r:
        payload = json.loads(r.read())
except Exception as e:
    sys.exit(f"Fetch failed: {e}")

scores = {}
skipped = 0
for m in payload.get("data", []):
    score = (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
    if score is None:
        continue
    slug         = m.get("slug", "")
    creator_slug = (m.get("model_creator") or {}).get("slug", "")
    creator_name = (m.get("model_creator") or {}).get("name", "")

    if not matches_hub(creator_slug, slug, m["name"], display_names, model_ids):
        skipped += 1
        continue

    evaluations = {k: v for k, v in (m.get("evaluations") or {}).items() if v is not None}
    entry = {
        "score":        score,
        "name":         m["name"],
        "slug":         slug,
        "creator":      creator_slug,
        "creator_name": creator_name,
        "release_date": m.get("release_date", ""),
        "evaluations":  evaluations,
    }
    if creator_slug and slug:
        scores[f"{creator_slug}/{slug}"] = entry
    scores[m["name"]] = entry

with open(OUT, "w") as f:
    json.dump(scores, f, indent=2)

print(f"Wrote {len(scores)} entries ({skipped} AA models skipped) to {OUT}")
