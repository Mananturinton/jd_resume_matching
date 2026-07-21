"""
jd_kb_bridge.py — bridges a structured JD (combined_classification.json)
to entity_kb.py's EntityKB, and produces ONE merged graph carrying:

  - every JD-matched entity's FULL detail (entity_kb.detail(): merged
    categories, aliases, concepts, role-importance list, resolved
    prerequisites) — not a trimmed subset.
  - every relation entity_kb.py itself derives: REQUIRES (resolved
    prerequisites), its reverse PREREQUISITE_OF, RELATED_BY_CONCEPT
    (shared-vocabulary neighbors), IN_CATEGORY, RELEVANT_TO_ROLE.
  - every EXPLICIT typed edge from relationships.json (ALTERNATIVE_TO,
    COMPETES_WITH, RUNS_ON, ORCHESTRATES, IMPLEMENTS, TESTS, ...) —
    entity_kb.py's own loader skips this file entirely (it isn't a list
    of entity rows), so without this step those edges are invisible.

Nodes reached only as a neighbor (a prerequisite, a concept-neighbor, an
explicit-relation target) that the JD itself never asked for are kept as
"light" stubs and flagged jd_matched: false, so the output distinguishes
"what this JD requires" from "what the dataset says is connected to
that" without losing either.
"""
import json
import os
from collections import defaultdict

from entity_kb import EntityKB

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "kb")
CLASSIFICATION_PATH = os.path.join(HERE, "combined_classification.json")

TIER_RANK = {"CRITICAL": 3, "IMPORTANT": 2, "GENERIC": 1}


def match_jd_terms(classification, kb: EntityKB):
    """Every sub_requirement's match_keywords -> KB gid, with full JD
    provenance (tier / importance / role attribution / which sentence
    and sub_requirement it came from)."""
    matched = {}   # gid -> provenance dict
    ambiguous = {} # term -> [candidate names]
    unmatched = set()

    def record(gid, sentence, sub, term):
        p = matched.setdefault(gid, {
            "matched_via_terms": set(), "sentence_ids": set(), "sub_ids": set(),
            "tiers": set(), "role_attribution": defaultdict(float), "dominant_roles": set(),
        })
        p["matched_via_terms"].add(term)
        p["sentence_ids"].add(sentence["id"])
        p["sub_ids"].add(sub["sub_id"])
        p["tiers"].add(sentence["tier"])
        p["dominant_roles"].add(sentence["dominant_role"])
        for role, pct in (sentence.get("role_attribution") or {}).items():
            p["role_attribution"][role] = max(p["role_attribution"][role], pct)

    for sentence in classification["processed_sentences"]:
        for sub in sentence.get("sub_requirements") or []:
            if sub.get("matchable") is False:
                continue
            for term in sub.get("match_keywords") or []:
                gid = kb.lookup_gid(term)
                if gid is not None:
                    record(gid, sentence, sub, term)
                    continue
                candidates = kb.lookup_all(term)
                if len(candidates) > 1:
                    ambiguous[term] = [kb.entities[g]["name"] for g in candidates]
                else:
                    unmatched.add(term)

    return matched, ambiguous, unmatched


def load_explicit_relationships(kb_dir=None):
    with open(os.path.join(kb_dir or KB_DIR, "relationships.json"), encoding="utf-8") as f:
        return json.load(f)


def build_enriched_graph_from_classification(classification, kb: EntityKB, related_top_n=6):
    matched, ambiguous, unmatched = match_jd_terms(classification, kb)
    explicit_rel = load_explicit_relationships(kb.data_dir)

    nodes = {}     # node id -> node dict (mutated in place to upgrade light->full)
    edges = []     # list of edge dicts, de-duplicated via `seen`
    seen_edges = set()

    def add_node(node_id, label, properties):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "properties": properties}
        elif label == "KBEntity" and "roles" in properties and "roles" not in nodes[node_id]["properties"]:
            # upgrade a previously-added light stub to full detail
            nodes[node_id]["properties"] = properties

    def add_edge(source, target, etype, properties=None):
        key = (source, target, etype)
        if key in seen_edges:
            return
        seen_edges.add(key)
        e = {"source": source, "target": target, "type": etype}
        if properties:
            e["properties"] = properties
        edges.append(e)

    # ---- 1. every JD-matched entity's full fragment (entity_kb.py's own
    #         derived relations: IN_CATEGORY, RELEVANT_TO_ROLE, REQUIRES,
    #         RELATED_BY_CONCEPT) ------------------------------------------
    for gid, prov in matched.items():
        frag_nodes, frag_edges = kb.to_graph_fragment(gid, related_top_n=related_top_n)
        for n in frag_nodes:
            add_node(n["id"], n["label"], n["properties"])
        for e in frag_edges:
            add_edge(e["source"], e["target"], e["type"], e.get("properties"))

        primary_id = f"kb::{gid}"
        nodes[primary_id]["jd_matched"] = True
        nodes[primary_id]["jd"] = {
            "tier": sorted(prov["tiers"], key=lambda t: -TIER_RANK.get(t, 0))[0],
            "role_attribution": dict(prov["role_attribution"]),
            "dominant_roles": sorted(prov["dominant_roles"]),
            "found_in_sentences": sorted(prov["sentence_ids"]),
            "found_in_sub_requirements": sorted(prov["sub_ids"]),
            "matched_via_terms": sorted(prov["matched_via_terms"]),
        }

    # ---- 2. reverse PREREQUISITE_OF edges (entity_kb has the method but
    #         to_graph_fragment doesn't call it — add explicitly) ---------
    for gid in list(matched.keys()):
        primary_id = f"kb::{gid}"
        for rev in kb.prerequisite_of(gid):
            target_id = f"kb::{rev['gid']}"
            add_node(target_id, "KBEntity", kb._light_props(rev["gid"]))
            add_edge(target_id, primary_id, "PREREQUISITE_OF")

    # ---- 3. explicit typed edges from relationships.json — entity_kb.py
    #         never loads this file, so this is the only source for
    #         ALTERNATIVE_TO / COMPETES_WITH / RUNS_ON / IMPLEMENTS / etc.
    #         Pulled for every node currently in the graph (matched AND
    #         stub neighbors), both directions. --------------------------
    all_gids_seen = {int(nid.split("::")[1]) for nid in list(nodes) if nid.startswith("kb::")}
    outgoing_by_gid = defaultdict(list)
    incoming_by_gid = defaultdict(list)
    for src_str, block in explicit_rel.items():
        src_gid = int(src_str)
        for rel in block["relationships"]:
            outgoing_by_gid[src_gid].append(rel)
            incoming_by_gid[rel["target_gid"]].append({"source_gid": src_gid, **rel})

    for gid in list(all_gids_seen):
        src_id = f"kb::{gid}"
        for rel in outgoing_by_gid.get(gid, []):
            tgt_gid = rel["target_gid"]
            tgt_id = f"kb::{tgt_gid}"
            if tgt_gid in kb.entities:
                add_node(tgt_id, "KBEntity", kb._light_props(tgt_gid))
                add_edge(src_id, tgt_id, rel["relation"], {"weight": rel["weight"], "source": "explicit"})
        for rel in incoming_by_gid.get(gid, []):
            src_gid2 = rel["source_gid"]
            src_id2 = f"kb::{src_gid2}"
            if src_gid2 in kb.entities:
                add_node(src_id2, "KBEntity", kb._light_props(src_gid2))
                add_edge(src_id2, src_id, rel["relation"], {"weight": rel["weight"], "source": "explicit"})

    for nid, n in nodes.items():
        if n["label"] == "KBEntity" and "jd_matched" not in n:
            n["jd_matched"] = False

    tree = [{"name": r["role"], "gid": None} for r in
            (classification.get("testlify_analysis") or {}).get("roles_ranked", [])]

    return {
        "job_roles": classification.get("job_roles") or classification.get("chatgpt_roles"),
        "job_role_domain": classification.get("job_role_domain"),
        "role_anchor": classification.get("role_anchor"),
        "job_logistics": classification.get("job_logistics"),
        "kb_coverage": {
            "total_kb_entities": len(kb.entities),
            "jd_matched_count": len(matched),
            "jd_matched_entities": sorted(nodes[f"kb::{g}"]["properties"]["name"] for g in matched),
            "ambiguous_terms": ambiguous,
            "unmatched_terms": sorted(unmatched),
            "total_nodes_in_graph": len(nodes),
            "total_edges_in_graph": len(edges),
        },
        "tree": tree,
        "graph": {"nodes": list(nodes.values()), "relationships": edges},
    }


def main():
    with open(CLASSIFICATION_PATH, encoding="utf-8") as f:
        classification = json.load(f)
    kb = EntityKB(data_dir=KB_DIR)
    result = build_enriched_graph_from_classification(classification, kb)

    with open(os.path.join(HERE, "jd_knowledge_graph_full.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    cov = result["kb_coverage"]
    print(f"JD-matched entities: {cov['jd_matched_count']} / {cov['total_kb_entities']}")
    print(f"Total graph nodes (matched + neighbor stubs + categories + roles): {cov['total_nodes_in_graph']}")
    print(f"Total graph edges (all relation types): {cov['total_edges_in_graph']}")
    print(f"Unmatched extracted terms: {cov['unmatched_terms']}")
    if cov["ambiguous_terms"]:
        print(f"Ambiguous terms: {cov['ambiguous_terms']}")

    edge_types = defaultdict(int)
    for e in result["graph"]["relationships"]:
        edge_types[e["type"]] += 1
    print("Edge type breakdown:")
    for t, c in sorted(edge_types.items(), key=lambda kv: -kv[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()