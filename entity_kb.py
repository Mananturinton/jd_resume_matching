"""
entity_kb.py — Entity Knowledge Base for JD matching.

Loads the "*_knowledge_graph.json" shard files (02_frameworks... through
19_components_subsystems..., plus index.json and relationships.json),
merges them into one canonical entity table, and exposes lookup +
derived-relationship helpers so that, given a technology/skill name
pulled out of a JD (e.g. by jd.py's ToolTaxonomy), you can pull back
*everything* the dataset knows about it: hierarchy, aliases, concepts,
role-importance weights, prerequisites, and relationships to other
entities in the same dataset.

WHY A MERGE STEP IS NEEDED (read before changing the loader)
--------------------------------------------------------------
Inspecting the 19 uploaded files shows they are NOT 19 independent
category taxonomies. They are 19 overlapping *exports* of one 221-entity
master list:

  - Every entity has a stable `gid`. Across all 19 files there are only
    221 distinct gids, but 337 total rows — because an entity that
    belongs to more than one category (e.g. "React" tagged both
    "Frameworks" and "Concepts") is exported once per category into a
    separate file.
  - For a given gid, `aliases` / `concepts` / `roles` / `prerequisites` /
    `hierarchy` are byte-identical across every file it appears in. The
    only field that differs is an optional `categories` list (present
    on the file that happens to carry it), which is just bookkeeping
    for which category bucket(s) an entity was filed under.
  - The file names don't reliably describe their contents either (e.g.
    "03_tools_knowledge_graph.json" is mostly Programming Languages).
    Don't rely on file names for classification — rely on the data.

So this loader does NOT trust any single file to be "the" master. It
merges by `gid` across every shard: list-valued fields are unioned
(order-preserving, de-duplicated), `roles` are merged by role name
(keeping the max importance seen and the union of that role's
concepts), and `categories` is the union of every categories tag seen
for that gid anywhere. This is correct today (where one file happens to
already contain the full union) and stays correct if a future data
refresh splits things up differently.

WHAT "RELATIONSHIPS" MEANS IN THIS DATASET
--------------------------------------------------------------
There is no explicit edge list (no "related_tools" field like
merged_tools.json has). Two relationship types are derivable from what
IS here, and both are pre-computed at load time:

  1. Prerequisite edges: an entity's `prerequisites` are free-text
     labels ("Docker", "Programming Fundamentals", ...). Where a
     prerequisite string matches another entity's name/alias in this
     same dataset, that's a real REQUIRES edge (and its inverse,
     PREREQUISITE_OF). Most prerequisite strings are generic skills
     with no matching entity ("Programming Fundamentals") — those are
     kept as plain text, just not resolved into a graph edge.
  2. Shared-concept edges: two entities that both list the same
     concept (e.g. Django and FastAPI both touch "REST APIs") are
     related in the sense that matters for JD matching — they draw on
     overlapping skill vocabulary. These are computed on demand via
     `related_by_concepts()`, ranked by overlap size.

USAGE
--------------------------------------------------------------
    from entity_kb import EntityKB

    kb = EntityKB()  # loads every *_knowledge_graph.json / index.json /
                      # relationships.json next to this file

    kb.lookup("React")               # -> full detail dict, or None
    kb.lookup("reactjs")             # alias match, case-insensitive
    kb.detail("react")["roles"]      # role importance list, sorted desc

    # Batch-enrich a list of tool names (e.g. jd.py's extracted_tools)
    kb.enrich_names(["React", "PostgreSQL", "Docker", "SomeUnknownTool"])

    # Attach KB detail onto every tool node in a jd.py graph in-place
    kb.enrich_jd_tree(jd_result["tree"])
"""

import glob
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------
# Loading + merging
# --------------------------------------------------------------------------

DEFAULT_GLOB_PATTERNS = ("*.json",)


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace, for case/spacing-insensitive
    name and alias matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _merge_role(existing: Optional[dict], new: dict) -> dict:
    if existing is None:
        return {
            "role": new.get("role"),
            "importance": new.get("importance"),
            "concepts": list(dict.fromkeys(new.get("concepts") or [])),
        }
    if (new.get("importance") or 0) > (existing.get("importance") or 0):
        existing["importance"] = new.get("importance")
    for c in new.get("concepts") or []:
        if c not in existing["concepts"]:
            existing["concepts"].append(c)
    return existing


def _merge_entity(acc: Optional[dict], row: dict, source_file: str) -> dict:
    """Fold one raw JSON row into the accumulated canonical record for
    its gid. Scalar fields (name/entity_type/entity_subtype/hierarchy)
    are taken from the first row seen and assumed stable across
    duplicates (verified true for this dataset); list fields are
    unioned; roles are merged by role name."""
    if acc is None:
        acc = {
            "gid": row.get("gid"),
            "id": row.get("id"),
            "name": row.get("name"),
            "entity_type": row.get("entity_type"),
            "entity_subtype": row.get("entity_subtype"),
            "hierarchy": row.get("hierarchy") or {},
            "aliases": [],
            "concepts": [],
            "roles_by_name": {},
            "prerequisites": [],
            "categories": [],
            "source_files": [],
        }

    for a in row.get("aliases") or []:
        if a not in acc["aliases"]:
            acc["aliases"].append(a)
    for c in row.get("concepts") or []:
        if c not in acc["concepts"]:
            acc["concepts"].append(c)
    for p in row.get("prerequisites") or []:
        if p not in acc["prerequisites"]:
            acc["prerequisites"].append(p)
    for r in row.get("roles") or []:
        name = r.get("role")
        if not name:
            continue
        acc["roles_by_name"][name] = _merge_role(acc["roles_by_name"].get(name), r)
    for cat in row.get("categories") or []:
        if cat not in acc["categories"]:
            acc["categories"].append(cat)
    if source_file not in acc["source_files"]:
        acc["source_files"].append(source_file)

    return acc


class EntityKB:
    def __init__(self, data_dir: Optional[str] = None, paths: Optional[List[str]] = None):
        """
        data_dir: directory to glob for every *.json file in (default:
            the directory this file lives in). Every JSON file found is
            inspected — files that aren't a list of entity records (i.e.
            don't look like {"gid": ..., "name": ..., ...} rows) are
            skipped with a printed warning rather than crashing the load,
            so pointing data_dir at a folder that also happens to contain
            an unrelated JSON file (e.g. merged_tools.json, or a
            classifier output) is safe.
        paths: explicit list of file paths to load instead of globbing.
        """
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.paths = paths or self._discover_paths(self.data_dir)
        if not self.paths:
            raise FileNotFoundError(
                f"No .json files found in {self.data_dir}. "
                f"Pass data_dir=... or paths=[...] pointing at the folder "
                f"containing your knowledge-graph JSON files."
            )

        self.entities: Dict[int, dict] = {}       # gid -> canonical record
        self._name_index: Dict[str, int] = {}      # normalized name -> gid
        self._alias_index: Dict[str, List[int]] = defaultdict(list)  # normalized alias -> [gid,...]
        self._concept_index: Dict[str, Set[int]] = defaultdict(set)  # normalized concept -> {gid,...}

        self._load()
        self._build_indexes()

    @staticmethod
    def _discover_paths(data_dir: str) -> List[str]:
        seen = set()
        paths = []
        for pattern in DEFAULT_GLOB_PATTERNS:
            for p in sorted(glob.glob(os.path.join(data_dir, pattern))):
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
        return paths

    def _load(self) -> None:
        merged: Dict[int, dict] = {}
        used_paths: List[str] = []
        for path in self.paths:
            fname = os.path.basename(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[entity_kb] skipping {fname} — not valid JSON ({e})")
                continue
            if not isinstance(rows, list) or not rows:
                print(f"[entity_kb] skipping {fname} — not a non-empty JSON list of entity records")
                continue
            if not all(isinstance(r, dict) and "gid" in r for r in rows):
                print(f"[entity_kb] skipping {fname} — doesn't look like entity records (no \"gid\" field)")
                continue

            used_paths.append(path)
            for row in rows:
                gid = row.get("gid")
                if gid is None:
                    continue
                merged[gid] = _merge_entity(merged.get(gid), row, fname)

        self.paths = used_paths
        for gid, acc in merged.items():
            roles = sorted(
                acc["roles_by_name"].values(),
                key=lambda r: -(r.get("importance") or 0),
            )
            acc["roles"] = roles
            del acc["roles_by_name"]
            self.entities[gid] = acc

    def _build_indexes(self) -> None:
        for gid, e in self.entities.items():
            self._name_index[_norm(e["name"])] = gid
            for a in e["aliases"]:
                self._alias_index[_norm(a)].append(gid)
            for c in e["concepts"]:
                self._concept_index[_norm(c)].add(gid)

    # ----------------------------------------------------------------
    # Lookup
    # ----------------------------------------------------------------

    def lookup_gid(self, name: str) -> Optional[int]:
        """Resolve a free-text name to a single gid, or None if it
        doesn't match anything (or matches more than one entity
        ambiguously, e.g. the alias "TF" -> TensorFlow/Terraform)."""
        key = _norm(name)
        if key in self._name_index:
            return self._name_index[key]
        candidates = self._alias_index.get(key)
        if not candidates:
            return None
        distinct = set(candidates)
        if len(distinct) == 1:
            return next(iter(distinct))
        return None  # ambiguous alias — caller should use lookup_all()

    def lookup_all(self, name: str) -> List[int]:
        """Like lookup_gid, but returns every gid an ambiguous alias
        could refer to instead of giving up."""
        key = _norm(name)
        if key in self._name_index:
            return [self._name_index[key]]
        return list(dict.fromkeys(self._alias_index.get(key, [])))

    def lookup(self, name: str) -> Optional[dict]:
        gid = self.lookup_gid(name)
        return self.detail(gid) if gid is not None else None

    # ----------------------------------------------------------------
    # Derived relationships
    # ----------------------------------------------------------------

    def _resolve_prerequisites(self, gid: int) -> List[dict]:
        e = self.entities[gid]
        resolved = []
        for p in e["prerequisites"]:
            pgid = self.lookup_gid(p)
            entry = {"label": p, "resolved": False}
            if pgid is not None and pgid != gid:
                target = self.entities[pgid]
                entry.update({
                    "resolved": True,
                    "gid": pgid,
                    "name": target["name"],
                    "entity_type": target["entity_type"],
                })
            resolved.append(entry)
        return resolved

    def prerequisite_of(self, gid: int) -> List[dict]:
        """Reverse edge: entities that list `gid` as a prerequisite —
        i.e. things this entity is a foundation for."""
        name = self.entities[gid]["name"]
        aliases = {_norm(a) for a in self.entities[gid]["aliases"]}
        aliases.add(_norm(name))
        out = []
        for other_gid, e in self.entities.items():
            if other_gid == gid:
                continue
            for p in e["prerequisites"]:
                if _norm(p) in aliases:
                    out.append({"gid": other_gid, "name": e["name"], "entity_type": e["entity_type"]})
                    break
        return out

    def related_by_concepts(self, gid: int, top_n: int = 10) -> List[dict]:
        """Other entities ranked by number of shared concepts."""
        e = self.entities[gid]
        overlap: Dict[int, Set[str]] = defaultdict(set)
        for c in e["concepts"]:
            for other_gid in self._concept_index.get(_norm(c), ()):
                if other_gid != gid:
                    overlap[other_gid].add(c)
        ranked = sorted(overlap.items(), key=lambda kv: -len(kv[1]))[:top_n]
        return [
            {
                "gid": g,
                "name": self.entities[g]["name"],
                "entity_type": self.entities[g]["entity_type"],
                "shared_concepts": sorted(shared),
                "overlap_count": len(shared),
            }
            for g, shared in ranked
        ]

    # ----------------------------------------------------------------
    # Detail assembly (the "give me everything about this entity" call)
    # ----------------------------------------------------------------

    def detail(self, gid: int, include_related: bool = True, related_top_n: int = 8) -> dict:
        e = self.entities[gid]
        out = {
            "gid": gid,
            "name": e["name"],
            "entity_type": e["entity_type"],
            "entity_subtype": e["entity_subtype"],
            "hierarchy": e["hierarchy"],
            "categories": e["categories"],
            "aliases": e["aliases"],
            "concepts": e["concepts"],
            "roles": e["roles"],
            "prerequisites": self._resolve_prerequisites(gid),
        }
        if include_related:
            out["related"] = {
                "prerequisite_of": self.prerequisite_of(gid),
                "shared_concept_neighbors": self.related_by_concepts(gid, top_n=related_top_n),
            }
        return out

    # ----------------------------------------------------------------
    # Graph-fragment export (Neo4j-shaped nodes/edges, for merging into
    # another tool's graph output — e.g. jd.py's `graph["nodes"]` /
    # `graph["relationships"]` — without either module depending on the
    # other's internals).
    # ----------------------------------------------------------------

    def _light_props(self, gid: int) -> dict:
        """Minimal identifying properties for a neighbor node (a
        prerequisite or a shared-concept relative) — full detail is only
        attached to the entity that was actually asked for, so the
        fragment doesn't recursively explode into the whole dataset."""
        e = self.entities[gid]
        return {
            "name": e["name"],
            "entity_type": e["entity_type"],
            "entity_subtype": e["entity_subtype"],
            "hierarchy": e["hierarchy"],
        }

    def to_graph_fragment(
        self, gid: int, related_top_n: int = 6
    ) -> Tuple[List[dict], List[dict]]:
        """Everything this KB knows about entity `gid`, as
        (nodes, relationships) in the same {"id","label","properties"} /
        {"source","target","type"} shape jd.py's `_to_graph` uses:

          KBEntity  (this entity, full detail incl. roles/aliases/concepts)
          Category  (one per entry in this entity's `categories`)
          Role      (one per role in `roles`, edge carries importance)
          KBEntity  (light stub, one per resolved prerequisite — REQUIRES)
          KBEntity  (light stub, one per shared-concept neighbor —
                     RELATED_BY_CONCEPT, edge carries shared_concepts)

        Node ids are namespaced `kb::<gid>`, `kb_role::<role name>`,
        `kb_category::<category name>` so they can't collide with a
        caller's own `tool::<name>` / `concept::<name>` ids, and role
        nodes are shared across every entity that lists that role so a
        merged graph doesn't grow one Role node per tool.
        """
        primary_id = f"kb::{gid}"
        nodes = [{"id": primary_id, "label": "KBEntity", "properties": self.detail(gid, include_related=False)}]
        edges: List[dict] = []

        for cat in self.entities[gid]["categories"]:
            cat_id = f"kb_category::{cat}"
            nodes.append({"id": cat_id, "label": "Category", "properties": {"name": cat}})
            edges.append({"source": primary_id, "target": cat_id, "type": "IN_CATEGORY"})

        for r in self.entities[gid]["roles"]:
            role_id = f"kb_role::{r['role']}"
            nodes.append({"id": role_id, "label": "Role", "properties": {"name": r["role"]}})
            edges.append({
                "source": primary_id, "target": role_id, "type": "RELEVANT_TO_ROLE",
                "properties": {"importance": r.get("importance")},
            })

        for p in self._resolve_prerequisites(gid):
            if p["resolved"] and p["gid"] != gid:
                target_id = f"kb::{p['gid']}"
                nodes.append({"id": target_id, "label": "KBEntity", "properties": self._light_props(p["gid"])})
                edges.append({"source": primary_id, "target": target_id, "type": "REQUIRES"})

        for rel in self.related_by_concepts(gid, top_n=related_top_n):
            target_id = f"kb::{rel['gid']}"
            nodes.append({"id": target_id, "label": "KBEntity", "properties": self._light_props(rel["gid"])})
            edges.append({
                "source": primary_id, "target": target_id, "type": "RELATED_BY_CONCEPT",
                "properties": {"shared_concepts": rel["shared_concepts"], "overlap_count": rel["overlap_count"]},
            })

        return nodes, edges

    # ----------------------------------------------------------------
    # Batch / JD-integration helpers
    # ----------------------------------------------------------------

    def enrich_names(self, names: List[str], include_related: bool = True) -> Dict[str, Optional[dict]]:
        """Given a list of extracted skill/tool names (e.g. jd.py's
        `extracted_tools`), return {name: full_detail_or_None}."""
        out = {}
        for name in names:
            gid = self.lookup_gid(name)
            if gid is None:
                candidates = self.lookup_all(name)
                out[name] = None if not candidates else {
                    "ambiguous": True,
                    "candidates": [self.entities[g]["name"] for g in candidates],
                }
            else:
                out[name] = self.detail(gid, include_related=include_related)
        return out

    def enrich_jd_tree(self, tree: List[dict], include_related: bool = True) -> List[dict]:
        """Walk a jd.py-shaped tree (list of root capability nodes, each
        with subdomains[*].tools[*]) and attach a `kb_detail` key to
        every tool node whose name resolves in this KB. Mutates and
        returns `tree`. Tools that don't resolve are left untouched
        (jd.py's own taxonomy covers ~6.4K tools; this KB covers 221 —
        not everything jd.py finds will have a match here, and that's
        expected, not an error)."""
        for root in tree:
            for sub in root.get("subdomains", []):
                for tool in sub.get("tools", []):
                    gid = self.lookup_gid(tool["name"])
                    if gid is not None:
                        tool["kb_detail"] = self.detail(gid, include_related=include_related)
        return tree

    # ----------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "files_loaded": self.paths,
            "entity_count": len(self.entities),
            "entity_types": sorted({e["entity_type"] for e in self.entities.values()}),
        }


if __name__ == "__main__":
    import sys

    kb = EntityKB(data_dir=sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps(kb.stats(), indent=2))
    for test_name in ["React", "reactjs", "PostgreSQL", "tf", "Docker", "NotARealTool"]:
        gid = kb.lookup_gid(test_name)
        print(f"\n{test_name!r} -> gid={gid} name={kb.entities[gid]['name'] if gid else None}")