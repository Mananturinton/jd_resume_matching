"""
build_knowledge_graph.py
=========================

Stage 2 of the pipeline: transforms the already-processed semantic JSON
(combined_classification.json, produced by chatgptapi_new.py) into a
complete weighted knowledge graph.

NO NLP. NO GPT CALL. NO RE-EXTRACTION. Every node/edge/weight below is
read directly out of fields chatgptapi_new.py already computed:
  - processed_sentences[].entities[]     -> per-sentence entity weights
                                             (importance_percentage, relation,
                                             role_weight — already verb- and
                                             tier-weighted, see chatgptapi_new.py
                                             distribute_entity_importance_locally)
  - processed_sentences[].governing_verb -> verb -> relation type mapping
  - entity_importance_summary            -> JD-wide normalized entity score,
                                             dataset_detail, related_tools
                                             (already-fetched complete dataset
                                             records — reused here as-is)
  - responsibility_actions               -> action/object nodes
  - job_roles / role_attribution         -> role nodes + edges
  - sub_requirements                     -> practice / requirement nodes
  - soft_skill sentences                 -> soft skill nodes

NODE-CENTRIC, NOT SENTENCE-CENTRIC: Sentence nodes are provenance only — the
origin a node was extracted from — and carry NO importance/weight. Every
score in this graph is either read straight from an already node-centric
Stage-1 field (entity_importance_summary — itself computed from tier +
governing verb + dataset role-type, never from an LLM-assigned sentence
percentage) or derived bottom-up in this file: Action priority (from
verbs_priority_engg.json), Object importance (aggregated from its connected
actions/tools/concepts), and Requirement importance (aggregated from its
Action/Object/Tool/Concept nodes) — see _compute_object_and_requirement_scores.

This module ONLY restructures + propagates weight through that data. If a
spec field has no source in the semantic JSON (e.g. "Metric" nodes — this
pipeline never extracted measurable-outcome metrics, "ontology_version" —
the dataset carries no version field), it is left null/empty rather than
invented. Deterministic: same input always produces the same graph.
"""

from __future__ import annotations

import json
import os
import sys

from verb_priority_graph import load_verb_lookup, match_verb

# ---------------------------------------------------------------------------
# STEP 5 — verb -> relation type mapping (the only "rule table" in this file;
# everything else is a pure read-through of already-computed fields).
# ---------------------------------------------------------------------------
VERB_RELATION_MAP = {
    "design": "CREATES",
    "architect": "CREATES",
    "build": "CREATES",
    "create": "CREATES",
    "develop": "IMPLEMENTS",
    "implement": "IMPLEMENTS",
    "code": "IMPLEMENTS",
    "integrate": "CONNECTS_TO",
    "connect": "CONNECTS_TO",
    "optimize": "IMPROVES",
    "improve": "IMPROVES",
    "enhance": "IMPROVES",
    "refactor": "IMPROVES",
    "write": "PRODUCES",
    "document": "PRODUCES",
    "ensure": "ENFORCES",
    "enforce": "ENFORCES",
    "secure": "ENFORCES",
    "deploy": "DEPLOYS",
    "release": "DEPLOYS",
    "monitor": "OBSERVES",
    "observe": "OBSERVES",
    "log": "OBSERVES",
    "test": "VALIDATES",
    "validate": "VALIDATES",
    "verify": "VALIDATES",
    "maintain": "MAINTAINS",
    "support": "MAINTAINS",
    "manage": "MAINTAINS",
    "use": "USES",
    "utilize": "USES",
    "learn": "LEARNED",
}
DEFAULT_VERB_RELATION = "PERFORMS"  # unmapped verb — never invents a specific relation

VERBS_PRIORITY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verbs_priority_engg.json")

# Same fixed lookup chatgptapi_new.py uses — a deterministic classification,
# not an LLM-assigned score. Used only for the graph_summary tier counts.
TIER_WEIGHTS = {"CRITICAL": 1.0, "IMPORTANT": 0.7, "GENERIC": 0.3, "NON_MATCHABLE": 0.0}


class IdFactory:
    """Deterministic, sequential ids per node type: TOOL_001, SENT_002, ..."""

    def __init__(self):
        self._counters: dict = {}
        self._assigned: dict = {}  # (type, canonical_key) -> id, so re-requests return the same id

    def get(self, prefix: str, canonical_key: str) -> str:
        key = (prefix, canonical_key)
        if key in self._assigned:
            return self._assigned[key]
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        node_id = f"{prefix}_{n:03d}"
        self._assigned[key] = node_id
        return node_id


def _canon(name: str) -> str:
    """Canonicalization key: lowercase, strip whitespace and the handful of
    suffix variants that show up verbatim in this dataset/JD text (React.js /
    ReactJS -> react; PWA is handled via alias lists on the node itself, not
    guessed here)."""
    if not name:
        return ""
    key = name.strip().lower()
    for suffix in (".js", " js", "js"):
        if key.endswith(suffix) and len(key) > len(suffix) + 1:
            key = key[: -len(suffix)]
            break
    return key.strip()


def _entity_aliases(entity_name: str, dataset_detail: dict | None) -> list:
    """Aliases come ONLY from the dataset record already fetched for this
    entity (dataset_detail.aliases) — never guessed. Falls back to just the
    entity's own name if unresolved."""
    if dataset_detail and dataset_detail.get("aliases"):
        return sorted(set(dataset_detail["aliases"]) | {entity_name})
    return [entity_name]


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------

def _base_node(node_id, name, ntype, depth, **extra):
    node = {
        "id": node_id,
        "name": name,
        "type": ntype,
        # CHANGE 11 — explicit (mentioned directly in the JD) vs knowledge_base
        # (added automatically, e.g. a parent_relations target never itself
        # mentioned in any sentence). Defaults to explicit; callers adding a
        # KB-inferred node must pass source="knowledge_base" explicitly.
        "source": extra.pop("source", "explicit"),
        "aliases": extra.pop("aliases", [name]),
        "source_sentence": extra.pop("source_sentence", None),
        "governing_action": extra.pop("governing_action", None),
        "sentence_weight": extra.pop("sentence_weight", None),
        "entity_importance": extra.pop("entity_importance", None),
        "importance": extra.pop("importance", None),                    # CHANGE 8
        "importance_reason": extra.pop("importance_reason", None),      # CHANGE 8
        "matching_priority": extra.pop("matching_priority", None),      # CHANGE 12
        "propagated_weight": extra.pop("propagated_weight", None),
        "normalized_weight": extra.pop("normalized_weight", None),
        "confidence": extra.pop("confidence", None),
        "role": extra.pop("role", None),
        "role_attribution": extra.pop("role_attribution", None),
        "tier": extra.pop("tier", None),
        "mentions": extra.pop("mentions", []),          # CHANGE 1 — sentence ids this node was seen in
        "mention_count": extra.pop("mention_count", 0),  # CHANGE 1
        "actions": extra.pop("actions", []),             # CHANGE 2 — action NAMES this entity participated in
        "objects": extra.pop("objects", []),             # CHANGE 3 — object NAMES this entity was used for
        "related_concepts": extra.pop("related_concepts", []),
        "depth": depth,
        "parent_nodes": extra.pop("parent_nodes", []),
        "child_nodes": extra.pop("child_nodes", []),  # ALL outgoing-edge targets (graph adjacency)
        "children": extra.pop("children", []),        # CHANGE 5 — explicit hierarchy children only (reverse of parent_relations)
        "incoming_edges": [],
        "outgoing_edges": [],
        "ontology_id": extra.pop("ontology_id", None),
        "ontology_version": None,  # not present anywhere in updated_dataset — left null, not invented
        "graph_metrics": None,     # CHANGE 9 — filled in by _compute_graph_metrics
    }
    node.update(extra)
    return node


def _add_unique(lst: list, value) -> None:
    if value and value not in lst:
        lst.append(value)


# ---------------------------------------------------------------------------
# NODE IMPORTANCE -> REQUIREMENT IMPORTANCE (bottom-up, run after the graph
# is built): Object importance = mean of its connected Action/Tool/Concept
# scores; Requirement/Practice importance = mean of its parent sentence's
# Action/Object/Tool/Concept scores. Nothing here reads sentence text or an
# LLM-assigned percentage — every input is a node score already computed
# (Action.priority from verbs_priority_engg.json, Tool/Concept.normalized_
# weight from chatgptapi_new.py's entity_importance_summary).
# ---------------------------------------------------------------------------

ACTION_SCORE_SCALE = 10  # Action priority is 1-10; scale to the same 0-100 range as Tool/Concept normalized_weight


def _node_score(nodes: dict, node_id: str):
    n = nodes.get(node_id)
    if not n:
        return None
    if n["type"] == "Action":
        p = n.get("priority")
        return p * ACTION_SCORE_SCALE if p is not None else None
    return n.get("normalized_weight")


def _compute_object_and_requirement_scores(nodes: dict, edges: list, sentence_subgraphs: list, requirement_sentence_map: dict) -> None:
    edges_by_target: dict = {}
    edges_by_source: dict = {}
    for e in edges:
        edges_by_target.setdefault(e["target"], []).append(e)
        edges_by_source.setdefault(e["source"], []).append(e)

    # -- Object importance: mean of connected Action (ACTS_ON, reversed) and
    # Tool (USED_FOR) scores, plus any Concept (BELONGS_TO) scores. ---------
    for node_id, n in nodes.items():
        if n["type"] != "Object":
            continue
        connected = set()
        for e in edges_by_target.get(node_id, []):
            src = nodes.get(e["source"])
            if src and src["type"] in ("Action", "Tool"):
                connected.add(e["source"])
        for e in edges_by_source.get(node_id, []):
            tgt = nodes.get(e["target"])
            if tgt and tgt["type"] == "Concept":
                connected.add(e["target"])
        scores = [s for s in (_node_score(nodes, cid) for cid in connected) if s is not None]
        importance = round(sum(scores) / len(scores), 3) if scores else None
        n["normalized_weight"] = importance
        n["propagated_weight"] = importance

    # -- Requirement / Practice importance: mean of the parent sentence's
    # Action + Object + Tool/Concept node scores (all already computed above
    # or in Stage 1) — the aggregate the spec calls for, computed AFTER node
    # scoring, never an input to it. -----------------------------------------
    node_chain_by_sentence = {sg["sentence_id"]: sg["node_chain"] for sg in sentence_subgraphs}
    requirement_score_by_sentence = {}
    for sid, chain in node_chain_by_sentence.items():
        candidate_ids = list(chain.get("actions", [])) + list(chain.get("objects", [])) + list(chain.get("entities", []))
        scores = [s for s in (_node_score(nodes, cid) for cid in candidate_ids) if s is not None]
        requirement_score_by_sentence[sid] = round(sum(scores) / len(scores), 3) if scores else None

    for req_id, sid in requirement_sentence_map.items():
        agg = requirement_score_by_sentence.get(sid)
        if req_id in nodes:
            nodes[req_id]["normalized_weight"] = agg
            nodes[req_id]["propagated_weight"] = agg

    for sg in sentence_subgraphs:
        sg["requirement_importance"] = requirement_score_by_sentence.get(sg["sentence_id"])


# Relation types that represent genuine hierarchy (child depends on/extends
# parent) — used for CHANGE 5's reverse "children" list. RELATED_TO and
# ALTERNATIVE_TO are deliberately excluded (not a parent/child relationship).
HIERARCHY_RELATION_TYPES = {
    "USES_LANGUAGE", "BUILT_ON", "EXTENDS", "PART_OF",
    "IMPLEMENTS", "RUNS_ON", "USES_PROTOCOL", "REQUIRES", "USES_TECHNOLOGY",
}

TIER_RANK = {"CRITICAL": 3, "IMPORTANT": 2, "GENERIC": 1, "NON_MATCHABLE": 0}


def _compute_children_from_parents(nodes: dict, edges: list) -> None:
    """CHANGE 5 — reverse of the existing parent_relations edges (entity ->
    target). Purely a pass over already-built edges; no new relationships
    invented. Node.js -[USES_LANGUAGE]-> JavaScript means JavaScript's
    "children" gains "Node.js"."""
    for e in edges:
        if e["relation_type"] not in HIERARCHY_RELATION_TYPES:
            continue
        src, tgt = nodes.get(e["source"]), nodes.get(e["target"])
        if not src or not tgt:
            continue
        if src["type"] in ("Tool", "Concept") and tgt["type"] in ("Tool", "Concept"):
            _add_unique(tgt["children"], src["name"])


def _pagerank(node_ids: list, out_adj: dict, damping: float = 0.85, iterations: int = 40) -> dict:
    """CHANGE 9 — plain power-iteration PageRank, no external graph library."""
    n = len(node_ids)
    if n == 0:
        return {}
    rank = {nid: 1.0 / n for nid in node_ids}
    for _ in range(iterations):
        dangling_sum = sum(rank[nid] for nid in node_ids if not out_adj.get(nid))
        new_rank = {nid: (1 - damping) / n + damping * dangling_sum / n for nid in node_ids}
        for nid in node_ids:
            outs = out_adj.get(nid) or []
            if not outs:
                continue
            share = damping * rank[nid] / len(outs)
            for tgt in outs:
                if tgt in new_rank:
                    new_rank[tgt] += share
        rank = new_rank
    return rank


def _betweenness_centrality(node_ids: list, adjacency: dict) -> dict:
    """CHANGE 9 — Brandes' algorithm (unweighted, undirected), pure Python.
    O(V*E), fine at this graph's scale (a few hundred nodes/edges)."""
    betweenness = {nid: 0.0 for nid in node_ids}
    for s in node_ids:
        pred = {nid: [] for nid in node_ids}
        sigma = {nid: 0.0 for nid in node_ids}
        sigma[s] = 1.0
        dist = {nid: -1 for nid in node_ids}
        dist[s] = 0
        queue = [s]
        stack = []
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            stack.append(v)
            for w in adjacency.get(v, ()):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {nid: 0.0 for nid in node_ids}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    n = len(node_ids)
    norm = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    return {nid: round(v / 2.0 * norm, 6) for nid, v in betweenness.items()}


def _compute_graph_metrics(nodes: dict, edges: list) -> None:
    """CHANGE 9 — degree/in-degree/out-degree/neighbor_count/connected_
    objects/connected_actions/dependency_count/pagerank/betweenness,
    computed algorithmically from the already-built edge list only."""
    node_ids = list(nodes.keys())
    adjacency = {nid: set() for nid in node_ids}
    out_adj = {nid: [] for nid in node_ids}
    in_count = {nid: 0 for nid in node_ids}
    out_count = {nid: 0 for nid in node_ids}
    connected_objects = {nid: set() for nid in node_ids}
    connected_actions = {nid: set() for nid in node_ids}

    for e in edges:
        s, t = e["source"], e["target"]
        if s not in nodes or t not in nodes:
            continue
        adjacency[s].add(t)
        adjacency[t].add(s)
        out_adj[s].append(t)
        out_count[s] += 1
        in_count[t] += 1
        if nodes[t]["type"] == "Object":
            connected_objects[s].add(t)
        if nodes[s]["type"] == "Object":
            connected_objects[t].add(s)
        if nodes[t]["type"] == "Action":
            connected_actions[s].add(t)
        if nodes[s]["type"] == "Action":
            connected_actions[t].add(s)

    pagerank = _pagerank(node_ids, out_adj)
    betweenness = _betweenness_centrality(node_ids, adjacency)

    for nid, n in nodes.items():
        degree = len(adjacency[nid])
        n["graph_metrics"] = {
            "degree": degree,
            "in_degree": in_count[nid],
            "out_degree": out_count[nid],
            "neighbor_count": degree,
            "connected_objects": len(connected_objects[nid]),
            "connected_actions": len(connected_actions[nid]),
            "dependency_count": len(n.get("children") or []) + len(n.get("parent_nodes") or []),
            "pagerank": round(pagerank.get(nid, 0.0), 6),
            "betweenness": betweenness.get(nid, 0.0),
        }


def _compute_global_importance(nodes: dict) -> None:
    """CHANGE 8/7 — Tool/Concept "importance" (0-100, absolute per-node
    score — distinct from Stage 1's entity_importance_summary "percentage",
    which is a JD-wide share that sums to 100 across ALL entities). Combines
    the Stage-1 tier/verb/role/frequency signal (entity_importance, already
    computed without any sentence-text score) with this graph's own
    connectivity (pagerank) — never re-reads sentence text or LLM output.
    importance_reason is assembled from the same computed signals, not
    generated by an LLM."""
    candidates = [n for n in nodes.values() if n["type"] in ("Tool", "Concept")]
    if not candidates:
        return

    max_base = max((n.get("entity_importance") or 0) for n in candidates) or 1
    max_pr = max((n["graph_metrics"]["pagerank"] for n in candidates), default=0) or 1e-9

    for n in candidates:
        base_component = (n.get("entity_importance") or 0) / max_base * 100
        connectivity_component = n["graph_metrics"]["pagerank"] / max_pr * 100
        importance = round(0.7 * base_component + 0.3 * connectivity_component, 1)
        n["importance"] = max(0.0, min(100.0, importance))

        reasons = []
        if n.get("tier") == "CRITICAL":
            reasons.append("appears in a CRITICAL-tier requirement")
        elif n.get("tier") == "IMPORTANT":
            reasons.append("appears in an IMPORTANT-tier requirement")
        if n.get("mention_count", 0) > 1:
            reasons.append(f"mentioned in {n['mention_count']} sentences")
        avg_role_weight = (n["_role_weight_sum"] / n["_role_weight_n"]) if n.get("_role_weight_n") else None
        if avg_role_weight is not None and avg_role_weight >= 1.1:
            reasons.append("classified as a core technology in the dataset")
        elif avg_role_weight is not None and avg_role_weight <= 0.9:
            reasons.append("classified as a supporting/utility technology in the dataset")
        if n["graph_metrics"]["pagerank"] >= max_pr * 0.5:
            reasons.append("high graph connectivity")
        if not n.get("resolved_in_dataset"):
            reasons.append("not resolved against the technology dataset")
        n["importance_reason"] = "; ".join(reasons) if reasons else "single, peripheral mention"


def _compute_matching_priority(nodes: dict) -> None:
    """CHANGE 12 — how much this node should weigh in Resume<->JD matching.
    NOT the same as graph "importance": scaled 1-10 from the dataset's own
    TechnologyRoleWeight (avg role_weight this entity carried across its
    mentions — core languages/frameworks score higher than generic tools),
    adjusted for explicit-vs-knowledge_base provenance and tier."""
    for n in nodes.values():
        if n["type"] not in ("Tool", "Concept"):
            continue
        avg_role_weight = (n["_role_weight_sum"] / n["_role_weight_n"]) if n.get("_role_weight_n") else 1.0
        scaled = avg_role_weight * (10 / 1.2)
        if n["source"] == "knowledge_base":
            scaled *= 0.6  # never explicitly mentioned in the JD — contributes less to matching
        if n.get("tier") == "CRITICAL":
            scaled += 1
        elif n.get("tier") == "GENERIC":
            scaled -= 1
        n["matching_priority"] = int(round(max(1, min(10, scaled))))


def build_knowledge_graph(classification: dict) -> dict:
    ids = IdFactory()
    nodes: dict = {}   # node_id -> node
    edges: list = []
    edge_seen: set = set()  # (source, target, relation_type) -> dedup (STEP 8)
    sentence_subgraphs: list = []

    def add_node(node):
        if node["id"] not in nodes:
            nodes[node["id"]] = node
        return nodes[node["id"]]

    def add_edge(source, target, relation_type, weight=None, depth=None, confidence=None):
        key = (source, target, relation_type)
        if key in edge_seen:
            return None
        edge_seen.add(key)
        edge = {
            "edge_id": f"EDGE_{len(edges) + 1:04d}",
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "weight": weight,
            "depth": depth,
            "confidence": confidence,
        }
        edges.append(edge)
        if source in nodes:
            nodes[source]["outgoing_edges"].append(edge["edge_id"])
            if target not in nodes[source]["child_nodes"]:
                nodes[source]["child_nodes"].append(target)
        if target in nodes:
            nodes[target]["incoming_edges"].append(edge["edge_id"])
            if source not in nodes[target]["parent_nodes"]:
                nodes[target]["parent_nodes"].append(source)
        return edge

    # -- JD root (STEP 10: JD always starts at 100%) -------------------------
    jd_id = "JD_001"
    add_node(_base_node(jd_id, classification.get("predicted_role") or "Job Description", "JobDescription",
                         depth=0, normalized_weight=100.0, propagated_weight=100.0, confidence=1.0))

    # -- Role nodes -----------------------------------------------------------
    role_pct_by_name = {}
    for role_entry in (classification.get("testlify_analysis") or {}).get("roles_ranked", []) or []:
        role_pct_by_name[role_entry["role"]] = role_entry.get("role_importance_pct")

    role_ids = {}
    for role_name in classification.get("chatgpt_roles", []) or []:
        rid = ids.get("ROLE", _canon(role_name))
        role_ids[role_name] = rid
        add_node(_base_node(rid, role_name, "Role", depth=1,
                             normalized_weight=role_pct_by_name.get(role_name),
                             propagated_weight=role_pct_by_name.get(role_name),
                             parent_nodes=[jd_id]))
        add_edge(jd_id, rid, "HAS_ROLE", weight=role_pct_by_name.get(role_name), depth=1, confidence=1.0)

    # NOTE: sentence_importance_analysis is deliberately NOT read here.
    # Sentences are provenance only ("the origin of extraction," never a
    # scored entity) — see module docstring. Tier is still available per
    # sentence (a fixed CRITICAL/IMPORTANT/GENERIC classification, not an
    # LLM-assigned percentage) and is used below only for the summary's
    # critical_weight/important_weight counts, never to weight a node.

    # responsibility_actions indexed by sentence_id (already extracted, not re-parsed)
    actions_by_sentence: dict = {}
    for act in classification.get("responsibility_actions", []) or []:
        actions_by_sentence.setdefault(act.get("sentence_id"), []).append(act)

    # Entity-level dataset detail, keyed by canonical entity name, already
    # fetched in Stage 1 (entity_importance_summary) — reused, not refetched.
    detail_by_entity = {}
    jd_wide_pct_by_entity = {}
    for e in classification.get("entity_importance_summary", []) or []:
        key = _canon(e["entity"])
        detail_by_entity[key] = e
        jd_wide_pct_by_entity[key] = e.get("percentage")

    tool_concept_ids: dict = {}  # canonical key -> node_id, so every mention merges into ONE node (STEP 1/8)

    def get_or_create_entity_node(entity_name, category, confidence):
        key = _canon(entity_name)
        summary_entry = detail_by_entity.get(key)
        dataset_detail = summary_entry.get("dataset_detail") if summary_entry else None
        ntype = "Concept" if category == "concept" else "Tool"
        prefix = "CONCEPT" if ntype == "Concept" else "TOOL"

        if key in tool_concept_ids:
            node_id = tool_concept_ids[key]
        else:
            node_id = ids.get(prefix, key)
            tool_concept_ids[key] = node_id
            mentions = (summary_entry.get("sentences") if summary_entry else None) or []
            node = add_node(_base_node(
                node_id, entity_name, ntype, depth=4,
                source="explicit",  # CHANGE 11 — extracted directly from a sentence's own entities[]
                aliases=_entity_aliases(entity_name, dataset_detail),
                entity_importance=jd_wide_pct_by_entity.get(key),
                normalized_weight=jd_wide_pct_by_entity.get(key),
                confidence=confidence,
                mentions=mentions,                 # CHANGE 1
                mention_count=len(mentions),        # CHANGE 1
                ontology_id=(dataset_detail or {}).get("index_id"),
                dataset_entity_type=(dataset_detail or {}).get("entity_type"),
                resolved_in_dataset=summary_entry.get("resolved_in_dataset") if summary_entry else False,
            ))
            # Private accumulators for CHANGE 8/12's role-weight averaging —
            # stripped from the node before the graph is returned.
            node["_role_weight_sum"] = 0.0
            node["_role_weight_n"] = 0
        return node_id

    verb_lookup = load_verb_lookup(VERBS_PRIORITY_PATH)
    requirement_sentence_map: dict = {}  # req_id -> sentence_id, for the post-pass below

    # -- Walk every processed_sentence: build its subgraph --------------------
    for sentence in classification.get("processed_sentences", []) or []:
        sid = sentence.get("id")
        confidence = sentence.get("specificity_weight")
        tier = sentence.get("tier")

        # Sentence node: PROVENANCE ONLY — no weight/importance fields.
        # It is where nodes were extracted FROM, not a scored entity.
        sent_node_id = ids.get("SENT", f"s{sid}")
        add_node(_base_node(
            sent_node_id, sentence.get("text", ""), "Sentence", depth=1,
            source_sentence=sid, tier=tier,
            role=sentence.get("dominant_role"), role_attribution=sentence.get("role_attribution"),
        ))
        add_edge(jd_id, sent_node_id, "HAS_SENTENCE", depth=1, confidence=confidence)
        dominant_role = sentence.get("dominant_role")
        if dominant_role in role_ids:
            add_edge(sent_node_id, role_ids[dominant_role], "ATTRIBUTED_TO",
                      weight=(sentence.get("role_attribution") or {}).get(dominant_role), depth=1)

        subgraph_edges = []

        # STEP 5 — Action + Object nodes from responsibility_actions. Action
        # priority/category come from verbs_priority_engg.json (the same
        # deterministic table chatgptapi_new.py uses for VerbWeight) — NOT
        # from sentence importance.
        governing_verb = sentence.get("governing_verb")
        action_node_ids = []
        object_node_ids = []
        for act in actions_by_sentence.get(sid, []):
            target = (act.get("target") or "").strip()
            object_id = None
            if target:
                object_id = ids.get("OBJECT", _canon(target))
                if object_id not in nodes:
                    add_node(_base_node(object_id, target, "Object", depth=3, source_sentence=sid))
                object_node_ids.append(object_id)

            for raw_verb in act.get("actions", []) or []:
                verb_key = raw_verb.strip().lower()
                relation = VERB_RELATION_MAP.get(verb_key, DEFAULT_VERB_RELATION)
                verb_entry = match_verb(raw_verb, verb_lookup)
                priority = verb_entry["priority"] if verb_entry else None
                category = verb_entry["category"] if verb_entry else None
                action_id = ids.get("ACTION", verb_key)
                if action_id not in nodes:
                    add_node(_base_node(action_id, raw_verb, "Action", depth=2, source_sentence=sid,
                                         mapped_relation=relation, priority=priority,
                                         category=category))
                action_node_ids.append(action_id)

                e1 = add_edge(sent_node_id, action_id, "HAS_ACTION", depth=2, confidence=confidence)
                if e1:
                    subgraph_edges.append(e1["edge_id"])
                if object_id:
                    e2 = add_edge(action_id, object_id, relation, depth=3, confidence=confidence)
                    if e2:
                        subgraph_edges.append(e2["edge_id"])
                    # CHANGE 4 — the generic, always-present Action-ACTS_ON->Object
                    # edge, alongside the more specific verb-mapped relation above
                    # (e.g. CREATES) — additive, doesn't replace it.
                    e2b = add_edge(action_id, object_id, "ACTS_ON", depth=3, confidence=confidence)
                    if e2b:
                        subgraph_edges.append(e2b["edge_id"])

        # STEP 4/1 — Tool/Concept nodes, canonicalized, weight = already-computed
        # per-sentence contribution (verb-weight + role-weight already baked in
        # by chatgptapi_new.py's distribute_entity_importance_locally — NOT
        # recomputed here).
        entity_node_ids_this_sentence = []
        or_group_members = []  # CHANGE 6 — node_ids in THIS sentence tagged relation="OR"
        for ent in sentence.get("entities", []) or []:
            node_id = get_or_create_entity_node(ent["entity"], ent.get("category"), confidence)
            entity_node_ids_this_sentence.append(node_id)
            node = nodes[node_id]
            # Already node-centric: tier-weight x verb-weight x role-weight,
            # computed in chatgptapi_new.py's distribute_entity_importance_locally
            # — not sentence-text importance.
            contribution_weight = ent.get("importance_percentage")
            ntype = node["type"]
            if ent.get("relation") == "OR":
                or_group_members.append(node_id)

            # CHANGE 8/12 bookkeeping: role_weight per occurrence, and the
            # strongest tier this entity has been seen under (CRITICAL beats
            # IMPORTANT beats GENERIC) — both used only in the post-pass.
            rw = ent.get("role_weight")
            if rw is not None and "_role_weight_sum" in node:
                node["_role_weight_sum"] += rw
                node["_role_weight_n"] += 1
            if TIER_RANK.get(tier, 0) >= TIER_RANK.get(node.get("tier"), -1):
                node["tier"] = tier

            e3 = add_edge(sent_node_id, node_id, "CONTAINS", weight=contribution_weight, depth=4, confidence=confidence)
            if e3:
                subgraph_edges.append(e3["edge_id"])
                e3["relation_flag"] = ent.get("relation")  # AND/OR, as already computed
            for action_id in action_node_ids:
                e4 = add_edge(action_id, node_id, "USES", weight=contribution_weight, depth=4, confidence=confidence)
                if e4:
                    subgraph_edges.append(e4["edge_id"])
                _add_unique(node["actions"], nodes[action_id]["name"])  # CHANGE 2

            # GRAPH EDGES — Tool USED_FOR Object / Object BELONGS_TO Concept
            # (existing), plus the spec's literal Object-USES_TOOL->Tool edge
            # (additive) — linking every tool/concept in this sentence to
            # this sentence's object(s), from responsibility_actions.
            for object_id in object_node_ids:
                if ntype == "Tool":
                    e8 = add_edge(node_id, object_id, "USED_FOR", depth=4, confidence=confidence)
                    e8b = add_edge(object_id, node_id, "USES_TOOL", depth=4, confidence=confidence)  # CHANGE 4
                    if e8b:
                        subgraph_edges.append(e8b["edge_id"])
                else:
                    e8 = add_edge(object_id, node_id, "BELONGS_TO", depth=4, confidence=confidence)
                if e8:
                    subgraph_edges.append(e8["edge_id"])
                _add_unique(node["objects"], nodes[object_id]["name"])  # CHANGE 3

            # STEP 4 — parent_relations, already GPT-extracted verbatim on the
            # entity (BUILT_ON / USES_LANGUAGE / USES_PROTOCOL / IMPLEMENTS /
            # PART_OF / EXTENDS / RELATED_TO / ALTERNATIVE_TO / REQUIRES /
            # RUNS_ON / USES_TECHNOLOGY) — never re-derived.
            for rel in ent.get("parent_relations", []) or []:
                target_key = _canon(rel["target"])
                target_ntype = "Concept" if rel.get("category") == "concept" else "Tool"
                target_prefix = "CONCEPT" if target_ntype == "Concept" else "TOOL"
                if target_key in tool_concept_ids:
                    target_id = tool_concept_ids[target_key]
                else:
                    target_id = ids.get(target_prefix, target_key)
                    tool_concept_ids[target_key] = target_id
                    # CHANGE 11 — a parent_relations target never itself seen
                    # as a top-level extracted entity is knowledge-base-added,
                    # not explicit.
                    add_node(_base_node(target_id, rel["target"], target_ntype, depth=5, source="knowledge_base"))
                e5 = add_edge(node_id, target_id, rel.get("relation_type", "RELATED_TO"), depth=5, confidence=confidence)
                if e5:
                    subgraph_edges.append(e5["edge_id"])

        # CHANGE 6 — OR alternatives in this sentence become one explicit
        # ANY_OF group node (deduped globally by its exact member set, so
        # the same "AWS, Azure, or GCP" trio mentioned in two sentences
        # collapses into a single group).
        if len(or_group_members) >= 2:
            group_key = frozenset(or_group_members)
            group_id = ids.get("GROUP", "|".join(sorted(group_key)))
            if group_id not in nodes:
                member_names = [nodes[m]["name"] for m in or_group_members]
                add_node(_base_node(
                    group_id, " / ".join(member_names), "Group", depth=4,
                    group_relation="ANY_OF", members=list(or_group_members),
                ))
                for member_id in or_group_members:
                    add_edge(group_id, member_id, "ANY_OF", depth=4)
            e9 = add_edge(sent_node_id, group_id, "HAS_GROUP", depth=4, confidence=confidence)
            if e9:
                subgraph_edges.append(e9["edge_id"])

        # sub_requirements -> Practice / Requirement nodes. Importance is NOT
        # read from GPT (sentence_importance_analysis is unused throughout
        # this module) — it's computed AFTER this loop, bottom-up from the
        # parent sentence's Action/Object/Tool/Concept node scores (see
        # _compute_object_and_requirement_scores below).
        for sub in sentence.get("sub_requirements") or []:
            sub_type = sub.get("type")
            ntype = "Practice" if sub_type == "practice" else "Requirement"
            prefix = "PRACTICE" if ntype == "Practice" else "REQUIREMENT"
            req_id = ids.get(prefix, sub["sub_id"])
            add_node(_base_node(req_id, sub["text"], ntype, depth=3, source_sentence=sid,
                                 confidence=confidence, sub_type=sub_type,
                                 match_keywords=sub.get("match_keywords")))
            e6 = add_edge(sent_node_id, req_id, "HAS_REQUIREMENT", depth=3, confidence=confidence)
            if e6:
                subgraph_edges.append(e6["edge_id"])
            requirement_sentence_map[req_id] = sid

        # is_soft_skill -> Soft Skill node
        if sentence.get("is_soft_skill"):
            ss_key = sentence.get("soft_skill_type") or "unspecified"
            ss_id = ids.get("SOFTSKILL", ss_key)
            if ss_id not in nodes:
                add_node(_base_node(ss_id, ss_key.replace("_", " ").title(), "SoftSkill", depth=2))
            e7 = add_edge(sent_node_id, ss_id, "EXPRESSES", depth=2, confidence=confidence)
            if e7:
                subgraph_edges.append(e7["edge_id"])

        sentence_subgraphs.append({
            "sentence_id": sid,
            "sentence_node": sent_node_id,
            "text": sentence.get("text", ""),  # provenance only — not scored
            "governing_verb": governing_verb,
            "node_chain": {
                "sentence": sent_node_id,
                "actions": action_node_ids,
                "objects": object_node_ids,
                "entities": entity_node_ids_this_sentence,
            },
            "edges": subgraph_edges,
            "requirement_importance": None,  # filled in by _compute_object_and_requirement_scores below
        })

    _compute_object_and_requirement_scores(nodes, edges, sentence_subgraphs, requirement_sentence_map)
    _compute_children_from_parents(nodes, edges)       # CHANGE 5
    _compute_graph_metrics(nodes, edges)                # CHANGE 9 — must run before importance (uses pagerank)
    _compute_global_importance(nodes)                   # CHANGE 8/7
    _compute_matching_priority(nodes)                   # CHANGE 12

    # Strip the private role-weight accumulators used only to compute
    # importance/matching_priority above — not part of the public schema.
    for n in nodes.values():
        n.pop("_role_weight_sum", None)
        n.pop("_role_weight_n", None)

    # -- STEP 7 — Ontology expansion (from already-fetched dataset_detail /
    # related_tools on entity_importance_summary — no new lookups, no
    # invented relations) --------------------------------------------------
    ontology_expansion = []
    for e in classification.get("entity_importance_summary", []) or []:
        key = _canon(e["entity"])
        node_id = tool_concept_ids.get(key)
        if node_id is None:
            continue
        detail = e.get("dataset_detail")
        entry = {
            "entity": e["entity"],
            "node_id": node_id,
            "resolved_in_dataset": e.get("resolved_in_dataset"),
            "ontology_id": (detail or {}).get("index_id"),
            "entity_type": (detail or {}).get("entity_type"),
            "aliases": (detail or {}).get("aliases", []),
            "parents": [
                {"name": r.get("target_name"), "relation": r.get("relation"), "weight": r.get("weight")}
                for r in (detail or {}).get("relationships", []) or []
            ],
            "children": (detail or {}).get("sub_concepts", []),
            "prerequisites": (detail or {}).get("prerequisites", []),
            "related_tools": [
                {"name": t.get("name"), "entity_type": t.get("entity_type"), "matched_via": t.get("matched_via")}
                for t in (e.get("related_tools") or [])
            ],
        }
        ontology_expansion.append(entry)

    # -- STEP 10/11 — weight validation (already-normalized inputs; this just
    # verifies + reports, never re-creates weight). No sentence_weight_sum
    # here — sentences carry no weight in this design; the JD's 100% pool is
    # validated at the entity (node) level only. -----------------------------
    total_entity_weight = round(sum(
        e.get("percentage", 0) for e in classification.get("entity_importance_summary", []) or []
    ), 3)

    weight_distribution = {
        "jd_total": 100.0,
        "entity_weight_sum": total_entity_weight,
        "entity_weight_valid": abs(total_entity_weight - 100.0) < 0.5,
    }

    propagated_scores = [
        {
            "node_id": tool_concept_ids.get(_canon(e["entity"])),
            "entity": e["entity"],
            "raw_score": e.get("raw_score"),
            "normalized_weight": e.get("percentage"),
            "frequency": e.get("frequency"),
            "contributions": e.get("contributions"),
        }
        for e in classification.get("entity_importance_summary", []) or []
        if tool_concept_ids.get(_canon(e["entity"])) is not None
    ]

    # -- STEP 14 — graph summary ------------------------------------------
    depths = [n["depth"] for n in nodes.values() if n.get("depth") is not None]
    type_counts: dict = {}
    for n in nodes.values():
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

    # Weighted mass per tier = count of sentences at that tier x its fixed
    # TIER_WEIGHT — a composition metric, not a claim about % of the JD
    # (sentences aren't scored, so there is no per-sentence % to sum here).
    tier_weight_totals: dict = {}
    for s in classification.get("processed_sentences", []) or []:
        tier = s.get("tier")
        tier_weight_totals[tier] = round(tier_weight_totals.get(tier, 0.0) + TIER_WEIGHTS.get(tier, 0.0), 3)

    graph_summary = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "max_depth": max(depths) if depths else 0,
        "avg_depth": round(sum(depths) / len(depths), 3) if depths else 0,
        "tool_count": type_counts.get("Tool", 0),
        "concept_count": type_counts.get("Concept", 0),
        "action_count": type_counts.get("Action", 0),
        "object_count": type_counts.get("Object", 0),
        "role_count": type_counts.get("Role", 0),
        "sentence_count": type_counts.get("Sentence", 0),
        "practice_count": type_counts.get("Practice", 0),
        "requirement_count": type_counts.get("Requirement", 0),
        "soft_skill_count": type_counts.get("SoftSkill", 0),
        "or_group_count": type_counts.get("Group", 0),
        "explicit_node_count": sum(1 for n in nodes.values() if n.get("source") == "explicit"),
        "knowledge_base_node_count": sum(1 for n in nodes.values() if n.get("source") == "knowledge_base"),
        "critical_weight": tier_weight_totals.get("CRITICAL", 0.0),
        "important_weight": tier_weight_totals.get("IMPORTANT", 0.0),
        "weight_validation": weight_distribution,
        "duplicate_nodes_removed": None,  # dedup happens inline via IdFactory/tool_concept_ids; no separate pass to count
        "duplicate_edges_removed": None,  # dedup happens inline via edge_seen; no separate pass to count
    }

    return {
        "metadata": {
            "source": "combined_classification.json",
            "predicted_role": classification.get("predicted_role"),
            "job_role_domain": classification.get("job_role_domain"),
            "roles": classification.get("chatgpt_roles"),
        },
        "graph_summary": graph_summary,
        "nodes": list(nodes.values()),
        "edges": edges,
        "sentence_subgraphs": sentence_subgraphs,
        "ontology_expansion": ontology_expansion,
        "weight_distribution": weight_distribution,
        "propagated_scores": propagated_scores,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "combined_classification.json")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "knowledge_graph.json")

    with open(in_path, encoding="utf-8") as f:
        classification = json.load(f)

    graph = build_knowledge_graph(classification)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(json.dumps(graph["graph_summary"], indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
