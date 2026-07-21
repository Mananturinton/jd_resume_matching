"""
jd_indexer.py
=============

Turns a JD's extracted entities (tools + concepts, already tagged by your
NER/extraction step) into a fully expanded index pulled from the skills
dataset.

Flow implemented (per your spec):
  - item.category == "tool"    -> resolve it in the dataset, pull its full
                                   detail + relationships directly.
  - item.category == "concept" -> resolve it in the dataset, look at its
                                   relationship edges, keep only the edges
                                   that point at *tool-like* entities
                                   (entity_type != "Concept") -> those are
                                   "the tools under that concept" -> pull
                                   full detail + relationships for each.

Input format expected for each extracted item (matches what you showed):
    {
        "target": "SQL",
        "relation_type": "USES_LANGUAGE",   # optional, just carried through
        "category": "tool"                  # "tool" | "concept"
    }

Dataset assumptions (validated against updated_dataset.zip):
  - Every category file (tools.json, languages.json, frameworks.json, ...)
    is a JSON list of entity records with a globally-unique "id".
  - relationships.json is a dict keyed by that same "id" (as a string):
        {
          "165": {
            "source_name": "Microservices Architecture",
            "relationships": [
              {"target_gid": 195, "target_name": "API Gateway",
               "relation": "USES", "weight": 0.7},
              ...
            ]
          },
          ...
        }
    NOTE: despite the key name, "target_gid" is actually the target's
    "id" field (not its 9-digit "gid"). This module resolves against "id".
  - "Concept" entities can live in ANY file (concepts.json,
    components_subsystems.json, methodologies_processes.json,
    metrics_measurement_systems.json, networking.json, security.json, ...).
    So "is this a concept" is decided by entity_type == "Concept", not by
    which file it came from.
"""

from __future__ import annotations

import glob
import json
import os
import zipfile
from difflib import get_close_matches
from typing import Any


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_dataset(dataset_path: str) -> dict[int, dict]:
    """Load every entity from every category file into {id: record}.

    dataset_path can be a directory of *.json files OR a .zip archive
    containing them (as in updated_dataset.zip).
    """
    entities_by_id: dict[int, dict] = {}

    def _ingest(name: str, raw_text: str):
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return
        if not isinstance(data, list):
            return
        for e in data:
            if not isinstance(e, dict) or "id" not in e:
                continue
            e = dict(e)  # shallow copy so we don't mutate caller data
            e["_source_file"] = os.path.basename(name)
            entities_by_id[e["id"]] = e

    if dataset_path.endswith(".zip"):
        with zipfile.ZipFile(dataset_path) as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    _ingest(name, zf.read(name).decode("utf-8"))
    else:
        for path in glob.glob(os.path.join(dataset_path, "*.json")):
            with open(path, encoding="utf-8") as f:
                _ingest(path, f.read())

    return entities_by_id


def build_name_index(entities_by_id: dict[int, dict]) -> dict[str, list[int]]:
    """lowercase(name or alias) -> [entity ids] (collisions kept as a list)."""
    name_index: dict[str, list[int]] = {}
    for eid, e in entities_by_id.items():
        candidates = [e.get("name")] + list(e.get("aliases") or [])
        for n in candidates:
            if not n:
                continue
            key = n.strip().lower()
            name_index.setdefault(key, []).append(eid)
    return name_index


def load_relationships(relationships_path: str) -> dict[int, dict]:
    with open(relationships_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def resolve_name(
    target: str,
    name_index: dict[str, list[int]],
    entities_by_id: dict[int, dict],
    category_hint: str | None = None,
    fuzzy_cutoff: float = 0.85,
) -> int | None:
    """Resolve a free-text target name to an entity id.

    If multiple entities share a name/alias, prefer the one whose
    entity_type matches the category_hint ("concept" -> entity_type
    == "Concept", "tool" -> anything else).
    """
    key = target.strip().lower()

    ids = name_index.get(key)
    if not ids:
        close = get_close_matches(key, name_index.keys(), n=1, cutoff=fuzzy_cutoff)
        ids = name_index.get(close[0]) if close else None
    if not ids:
        return None
    if len(ids) == 1 or not category_hint:
        return ids[0]

    for i in ids:
        is_concept = entities_by_id[i].get("entity_type") == "Concept"
        if (category_hint == "concept") == is_concept:
            return i
    return ids[0]


# --------------------------------------------------------------------------
# Detail assembly
# --------------------------------------------------------------------------

def build_detail(
    entity_id: int,
    entities_by_id: dict[int, dict],
    relationships: dict[int, dict],
) -> dict[str, Any] | None:
    e = entities_by_id.get(entity_id)
    if e is None:
        return None
    rel = relationships.get(entity_id, {})
    return {
        "id": e["id"],
        "gid": e.get("gid"),
        "name": e.get("name"),
        "entity_type": e.get("entity_type"),
        "entity_subtype": e.get("entity_subtype"),
        "hierarchy": e.get("hierarchy"),
        "aliases": e.get("aliases", []),
        "sub_concepts": e.get("concepts", []),       # nested breakdown, e.g. Redux -> Store, Reducers...
        "prerequisites": e.get("prerequisites", []),
        "index_id": e.get("index_id"),
        "source_file": e.get("_source_file"),
        "relationships": rel.get("relationships", []),
    }


def tools_under_concept(
    concept_id: int,
    entities_by_id: dict[int, dict],
    relationships: dict[int, dict],
) -> list[dict]:
    """Channel 1 - graph edges: walk a concept's relationship edges and keep
    only edges pointing at non-Concept (i.e. tool-like) entities."""
    edges = relationships.get(concept_id, {}).get("relationships", [])
    tools = []
    for edge in edges:
        tid = edge.get("target_gid")  # actually an "id", see module docstring
        target = entities_by_id.get(tid)
        if target is not None and target.get("entity_type") != "Concept":
            tools.append({
                "id": tid,
                "name": target.get("name"),
                "source": "relationship_edge",
                "relation": edge.get("relation"),
                "weight": edge.get("weight"),
            })
    return tools


def tools_via_subconcept_match(
    concept_id: int,
    entities_by_id: dict[int, dict],
) -> list[dict]:
    """Channel 2 - reverse sub-concept lookup: a tool's own record carries a
    "concepts" field (its internal breakdown, e.g. Redux -> Store, Reducers,
    ...). Some tools list a *top-level* concept name/alias in there too
    (e.g. Spring Boot's concepts include "Microservices Architecture",
    Express.js's include "RESTful API Design"). This scans every non-Concept
    entity's own concepts list for a name/alias match against the given
    concept, independent of whether a relationships.json edge exists."""
    concept = entities_by_id.get(concept_id)
    if concept is None:
        return []

    names_to_match = {concept["name"].strip().lower()}
    names_to_match.update(a.strip().lower() for a in concept.get("aliases", []) if a)

    tools = []
    for eid, e in entities_by_id.items():
        if eid == concept_id or e.get("entity_type") == "Concept":
            continue
        sub_names = {
            (sc.get("name") or "").strip().lower()
            for sc in e.get("concepts", [])
        }
        hit = names_to_match & sub_names
        if hit:
            # carry the weight the tool itself assigned to that sub-concept
            weight = next(
                (sc.get("weight") for sc in e.get("concepts", [])
                 if (sc.get("name") or "").strip().lower() in hit),
                None,
            )
            tools.append({
                "id": eid,
                "name": e.get("name"),
                "source": "subconcept_match",
                "matched_via": sorted(hit),
                "weight": weight,
            })
    return tools


def tools_related_to_concept(
    concept_id: int,
    entities_by_id: dict[int, dict],
    relationships: dict[int, dict],
) -> list[dict]:
    """Union of both channels, deduped by tool id. If a tool is found by
    both channels, both provenance records are kept under "sources"."""
    by_id: dict[int, dict] = {}

    for t in tools_under_concept(concept_id, entities_by_id, relationships):
        by_id.setdefault(t["id"], {"id": t["id"], "name": t["name"], "sources": []})
        by_id[t["id"]]["sources"].append({
            "channel": "relationship_edge",
            "relation": t["relation"],
            "weight": t["weight"],
        })

    for t in tools_via_subconcept_match(concept_id, entities_by_id):
        by_id.setdefault(t["id"], {"id": t["id"], "name": t["name"], "sources": []})
        by_id[t["id"]]["sources"].append({
            "channel": "subconcept_match",
            "matched_via": t["matched_via"],
            "weight": t["weight"],
        })

    return list(by_id.values())


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def index_jd_entities(
    extracted_items: list[dict],
    entities_by_id: dict[int, dict],
    name_index: dict[str, list[int]],
    relationships: dict[int, dict],
) -> dict[str, Any]:
    """
    extracted_items: list of {"target": str, "relation_type": str?, "category": "tool"|"concept"}

    Returns:
        {
          "resolved_tools": {id: detail, ...},       # every tool, direct or via a concept
          "resolved_concepts": {id: detail_with_related_tools, ...},
          "unresolved": [item, ...],                 # couldn't match to the dataset
        }
    """
    resolved_tools: dict[int, dict] = {}
    resolved_concepts: dict[int, dict] = {}
    unresolved: list[dict] = []

    for item in extracted_items:
        target = (item.get("target") or "").strip()
        category = (item.get("category") or "").strip().lower()
        if not target or category not in ("tool", "concept"):
            unresolved.append(item)
            continue

        eid = resolve_name(target, name_index, entities_by_id, category_hint=category)
        if eid is None:
            unresolved.append(item)
            continue

        if category == "tool":
            if eid not in resolved_tools:
                resolved_tools[eid] = build_detail(eid, entities_by_id, relationships)
                resolved_tools[eid]["matched_from"] = {
                    "target": target,
                    "relation_type": item.get("relation_type"),
                    "found_directly": True,
                }

        else:  # concept
            if eid not in resolved_concepts:
                detail = build_detail(eid, entities_by_id, relationships)
                related = tools_related_to_concept(eid, entities_by_id, relationships)
                detail["related_tools"] = related
                detail["matched_from"] = {
                    "target": target,
                    "relation_type": item.get("relation_type"),
                }
                resolved_concepts[eid] = detail

                for r in related:
                    tid = r["id"]
                    if tid not in resolved_tools:
                        tool_detail = build_detail(tid, entities_by_id, relationships)
                        tool_detail["matched_from"] = {
                            "found_directly": False,
                            "via_concept": detail["name"],
                            "via_concept_id": eid,
                            "sources": r["sources"],  # relationship_edge and/or subconcept_match
                        }
                        resolved_tools[tid] = tool_detail

    return {
        "resolved_tools": resolved_tools,
        "resolved_concepts": resolved_concepts,
        "unresolved": unresolved,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description="Expand JD-extracted tools/concepts into a full dataset index.")
    p.add_argument("--dataset", required=True, help="Path to dataset dir or .zip")
    p.add_argument("--relationships", required=True, help="Path to relationships.json")
    p.add_argument("--jd-entities", required=True, help="Path to JSON file: a list of {target, relation_type, category}")
    p.add_argument("--out", required=True, help="Where to write the resulting index JSON")
    args = p.parse_args()

    entities_by_id = load_dataset(args.dataset)
    name_index = build_name_index(entities_by_id)
    relationships = load_relationships(args.relationships)

    with open(args.jd_entities, encoding="utf-8") as f:
        extracted_items = json.load(f)

    result = index_jd_entities(extracted_items, entities_by_id, name_index, relationships)

    # JSON needs string keys
    out = {
        "resolved_tools": {str(k): v for k, v in result["resolved_tools"].items()},
        "resolved_concepts": {str(k): v for k, v in result["resolved_concepts"].items()},
        "unresolved": result["unresolved"],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Resolved tools: {len(result['resolved_tools'])}")
    print(f"Resolved concepts: {len(result['resolved_concepts'])}")
    print(f"Unresolved: {len(result['unresolved'])}")


if __name__ == "__main__":
    main()