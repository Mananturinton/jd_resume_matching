"""
prepare_jd_entities.py
=======================

Bridges a JD classification file (like combined_classification.json, whose
"extracted_skills" is a flat list of strings with no tool/concept tagging)
into the {target, relation_type, category} shape jd_indexer.py expects.

How the tagging works: each skill name is resolved against the SAME dataset
jd_indexer.py uses. Whatever entity_type comes back decides the category:
entity_type == "Concept" -> "concept", anything else -> "tool". No guessing
by keyword, no separate rulebook - one source of truth for what counts as
a concept vs a tool.

Also handles comma-packed entries some extractors produce, e.g.
"NestJS, Fastify" or "Jest, Mocha, Chai" -> split into separate skills
before resolution.

Usage:
    python prepare_jd_entities.py \
        --dataset updated_dataset \
        --classification combined_classification.json \
        --out jd_entities.json

Then feed jd_entities.json straight into jd_indexer.py's --jd-entities.
"""

from __future__ import annotations

import argparse
import json

from jd_indexer import build_name_index, load_dataset, resolve_name


def split_skill_entry(raw: str) -> list[str]:
    """"NestJS, Fastify" -> ["NestJS", "Fastify"]. Leaves single entries as-is."""
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def main():
    p = argparse.ArgumentParser(description="Convert a flat JD skills list into tagged {target, category} items.")
    p.add_argument("--dataset", required=True, help="Path to dataset dir or .zip")
    p.add_argument("--classification", required=True, help="Path to the JD classification json (e.g. combined_classification.json)")
    p.add_argument("--out", required=True, help="Where to write the {target, relation_type, category} list")
    p.add_argument("--skills-key", default="extracted_skills", help="Which key in the classification json holds the flat skills list")
    args = p.parse_args()

    entities_by_id = load_dataset(args.dataset)
    name_index = build_name_index(entities_by_id)

    with open(args.classification, encoding="utf-8") as f:
        data = json.load(f)

    raw_skills = data.get(args.skills_key)
    if raw_skills is None:
        print(f"WARNING: key '{args.skills_key}' not found. Available keys: {list(data.keys())}")
        raw_skills = []

    flat_names: list[str] = []
    for entry in raw_skills:
        if isinstance(entry, str):
            flat_names.extend(split_skill_entry(entry))
        elif isinstance(entry, dict) and "name" in entry:
            flat_names.append(entry["name"])

    items = []
    unresolved = []
    seen = set()

    for name in flat_names:
        key = name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)

        eid = resolve_name(name, name_index, entities_by_id)
        if eid is None:
            unresolved.append(name)
            continue

        entity = entities_by_id[eid]
        category = "concept" if entity.get("entity_type") == "Concept" else "tool"
        items.append({
            "target": entity["name"],  # canonical dataset name, not the raw JD string
            "relation_type": None,     # not available from a flat skills list
            "category": category,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Resolved {len(items)} skill(s) -> {args.out}")
    if unresolved:
        print(f"Unresolved ({len(unresolved)}), not in the dataset or no matching alias:")
        for n in unresolved:
            print(f"  - {n}")


if __name__ == "__main__":
    main()