#!/usr/bin/env python3
"""
match_resume_jd_optimized.py

Same matching logic/behavior as match_resume_jd.py, but with the hot paths
memoized so the same tool/skill comparison is never computed twice in a
single run.

WHY THIS WAS SLOW BEFORE
-------------------------
`tool_is_matched()` (nested alias x resume-skill loop with SequenceMatcher
fuzzy matching) is the single most expensive routine in the file, and the
original code called it *independently, from scratch* for the same JD tool
in half a dozen different places:
  - build_report()                         (flat-schema path)
  - _score_single_tool()                   (rich-schema tool bridging)
  - compute_requirement_and_action_scores()  (once per anchor, per sub_requirement)
  - compute_relation_score()               (once per tool_relations pair)
  - compute_tool_score() -> _score_single_tool() again, per role

For an N-tool JD scored across R roles, that's roughly O(N x R) full
re-scans of the resume skill set instead of O(N).

WHAT CHANGED
------------
1. `tool_is_matched`, `_score_single_tool`, and `_tool_concept_coverage`
   are now thin wrappers that build a hashable cache key (sorted tuples)
   and delegate to an `lru_cache`-wrapped implementation. Since the resume
   and JD tool catalog are constant within a single report run, each tool
   is now scored exactly once no matter how many components/roles ask for
   it.
2. `word_boundary_in` no longer compiles a fresh regex on every single
   alias/resume-skill pair; the compiled pattern is cached per alias
   (aliases repeat across many resume skills and many sentences).
3. `fuzzy_hit` and `is_meaningful` are memoized too — both are pure
   functions called repeatedly with the same short strings (aliases,
   resume phrases).
4. `build_action_strength_map` now shares a single memo dict across *all*
   starting nodes instead of only deduping within one node's own DFS. The
   resume graph is a DAG in practice (provenance edges point backwards
   from tool/object to action), so a node's best-action result doesn't
   depend on where the DFS started, and can be reused freely.
5. Added `clear_caches()` for long-running / batch processes (e.g. scoring
   many resumes against the same JD in a loop) so memory doesn't grow
   unbounded across resumes.

Nothing about the *scoring logic itself* changed — every formula, weight,
tier rule, and edge case comment from the original file is preserved
as-is. This is a performance refactor, not a behavior change.

Usage (same as before):
    python match_resume_jd_optimized.py
    python match_resume_jd_optimized.py resume.json jd.json
    python match_resume_jd_optimized.py resume.json jd.json --output report.json
"""

import json
import argparse
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache

# Words too generic to serve as sole evidence that a resume phrase refers to
# a *specific* tool when it merely appears inside one of that tool's longer,
# compound alias phrases (e.g. resume 'microservices' should not match the
# 'Java' tool just because 'java microservices' is in Java's alias list).
GENERIC_TERMS = {
    "microservices", "microservice", "architecture", "api", "apis", "service",
    "services", "platform", "platforms", "cloud", "application", "applications",
    "backend", "frontend", "development", "developer", "programming", "language",
    "framework", "design", "system", "systems", "data", "management", "integration",
    "security", "monitoring", "testing", "deployment", "deployments", "automation",
    "scalable", "distributed", "orchestration", "container", "containers",
    "database", "databases", "server", "servers", "client", "web", "software",
    "engineering", "tool", "tools", "solution", "solutions", "pipeline",
    "pipelines", "script", "scripts", "schema", "schemas", "project", "projects",
    "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "using",
}

VERBS_PRIORITY_FILENAME = "verbs_priority_engg.json"
# Unrecognized verb: treated as a mild, unremarkable action rather than
# assuming the best or the worst — roughly where hedge verbs like
# "support"/"assist" (priority 3) sit on the file's 1-10 scale.
DEFAULT_VERB_PRIORITY = 4
MAX_VERB_PRIORITY = 10

_VERB_PRIORITY_CACHE = None


def load_verb_priorities(path=VERBS_PRIORITY_FILENAME):
    """
    Loads verbs_priority_engg.json — a curated verb -> (priority 1-10,
    category, inflected/multi-word variants) table — and is now the single
    source of truth for every verb-strength judgment in this file. It
    replaces three things that used to live here as separate, coarser,
    hand-maintained tables: STRONG/MODERATE/WEAK_ACTIONS (flat-schema
    action strength, a 3-bucket scale), ACTION_ONTOLOGY/VERB_LEVEL
    (rich-schema action level, a 5-bucket scale), and a generic
    suffix-stripping heuristic (_normalize_verb_candidate) that was the
    only way those two ever handled inflected forms — and broke on
    irregulars ("led" -> "lead", "drove"/"driven" -> "drive", "oversaw" ->
    "oversee" all failed to normalize under simple suffix stripping).
    verbs_priority_engg.json instead lists every real inflected/multi-word
    surface form explicitly, so lookup is now exact rather than guessed.

    Returns (variant_to_verb, verb_priority, phrase_variants):
      - variant_to_verb: every inflected/multi-word surface form (lowercased)
        -> (canonical_verb, priority, category), e.g. 'led' -> ('lead', 10, 'leadership')
      - verb_priority: canonical verb -> priority, for looking up a verb
        that's already been resolved to its canonical form
      - phrase_variants: the subset of variants containing a space (e.g.
        'load test', 'roll back'), longest-first, so phrase scanning can
        check multi-word forms before falling back to single tokens
    """
    global _VERB_PRIORITY_CACHE
    if _VERB_PRIORITY_CACHE is not None:
        return _VERB_PRIORITY_CACHE
    try:
        data = load_json(path)
    except FileNotFoundError:
        _VERB_PRIORITY_CACHE = ({}, {}, [])
        return _VERB_PRIORITY_CACHE

    variant_to_verb, verb_priority, phrase_variants = {}, {}, []
    for entry in data.get("verb_priorities", []):
        verb = entry.get("verb", "").strip().lower()
        if not verb:
            continue
        priority = entry.get("priority", DEFAULT_VERB_PRIORITY)
        category = entry.get("category")
        verb_priority[verb] = priority
        for variant in entry.get("variants") or [verb]:
            v = variant.strip().lower()
            if v:
                variant_to_verb[v] = (verb, priority, category)
                if " " in v:
                    phrase_variants.append(v)
    phrase_variants = sorted(set(phrase_variants), key=len, reverse=True)
    _VERB_PRIORITY_CACHE = (variant_to_verb, verb_priority, phrase_variants)
    return _VERB_PRIORITY_CACHE

_WORD_PATTERN = re.compile(r"[a-z0-9+#.]+")
_VERB_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")


# ---------------------------------------------------------------------------
# Loading & extraction
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_resume_skills(resume_json):
    skills = set()
    for node in resume_json.get("nodes", []):
        if node.get("type") in ("tool", "object"):
            label = node.get("label", "").strip().lower()
            if label:
                skills.add(label)
    return skills


def extract_resume_activities(resume_json):
    """
    Action->object pairs straight from the resume graph's edges, e.g.
    ('prepare', 'test cases') or ('execute', 'manual test cases').

    Used only as a fallback for JD sub_requirements that name no tool at
    all (see _text_fallback_match): most manual-QA duties ("prepare test
    cases", "track bugs", "perform regression testing") never mention a
    tool, so _find_anchor_tools never anchors them and they'd otherwise be
    silently dropped into 'unverifiable' and excluded from scoring even
    when the resume clearly demonstrates the same activity.
    """
    nodes = {n["id"]: n for n in resume_json.get("nodes", [])}
    activities = []
    for e in resume_json.get("edges", []):
        src, tgt = nodes.get(e["source"]), nodes.get(e["target"])
        if not src or not tgt:
            continue
        if src.get("type") == "action" and tgt.get("type") in ("object", "tool"):
            verb = src.get("label", "").strip().lower()
            obj = tgt.get("label", "").strip().lower()
            if verb and obj:
                activities.append({"verb": verb, "object": obj})
    return activities


def build_action_strength_map(resume_json):
    """
    Walks the resume graph backward from each tool/object node to the
    action node(s) that produced it. Returns
    {skill_label_lower: {"strength": float, "verb": str}}.

    OPTIMIZATION: `memo` is shared across every starting node's DFS (not
    just within one node's own traversal, as before). A node's best-action
    result is intrinsic to the graph, not to which node we started from,
    so once any DFS resolves node X, every later DFS that reaches X reuses
    that answer instead of re-walking its whole upstream subgraph. This
    turns what was effectively O(nodes^2) in dense provenance graphs into
    O(nodes + edges).
    """
    nodes = {n["id"]: n for n in resume_json.get("nodes", [])}
    reverse_adj = {}
    for e in resume_json.get("edges", []):
        reverse_adj.setdefault(e["target"], []).append(e["source"])

    memo = {}

    def best_action(node_id, seen):
        if node_id in memo:
            return memo[node_id]
        if node_id in seen:
            return None  # cycle guard; not memoized since result is path-dependent here
        node = nodes.get(node_id)
        if node is None:
            return None
        if node["type"] == "action":
            verb = node["label"].strip().lower()
            variant_to_verb, _, _ = load_verb_priorities()
            hit = variant_to_verb.get(verb)
            if hit:
                canonical, priority, _ = hit
            else:
                canonical, priority = verb, DEFAULT_VERB_PRIORITY
            result = (priority / MAX_VERB_PRIORITY, canonical)
            memo[node_id] = result
            return result
        seen = seen | {node_id}
        best = None
        for src_id in reverse_adj.get(node_id, []):
            candidate = best_action(src_id, seen)
            if candidate and (best is None or candidate[0] > best[0]):
                best = candidate
        memo[node_id] = best
        return best

    strengths = {}
    for node in nodes.values():
        if node["type"] not in ("tool", "object"):
            continue
        label = node["label"].strip().lower()
        if not label:
            continue
        result = best_action(node["id"], frozenset())
        strength, verb = result if result else (DEFAULT_VERB_PRIORITY / MAX_VERB_PRIORITY, None)
        existing = strengths.get(label)
        if existing is None or strength > existing["strength"]:
            strengths[label] = {"strength": strength, "verb": verb}
    return strengths


def extract_jd_tools(output_json):
    _, graph_aliases_of = load_tool_graph()
    jd_tools = {}
    for t in output_json.get("tools", []):
        name = t.get("name", "").strip()
        if not name:
            continue
        weight = t.get("importance_percentage")
        tier = t.get("tier")
        aliases = [a.lower() for a in t.get("aliases", [])]
        aliases.append(name.lower())
        # know_graph.json's own alias list for this tool (e.g. "Kong" ->
        # "kong gateway") — catches resume phrasing the JD extraction's
        # own aliases list never anticipated.
        aliases.extend(graph_aliases_of.get(name.lower(), ()))
        jd_tools[name] = {
            "weight": weight if weight is not None else 0,
            "tier": tier if tier else "UNSCORED",
            "aliases": set(aliases),
        }
    return jd_tools


# ---------------------------------------------------------------------------
# Matching (memoized hot path)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=200_000)
def _boundary_pattern(needle):
    """Compiled once per distinct alias/needle, reused across every resume
    skill it's checked against (aliases repeat far more than they change)."""
    return re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")


def word_boundary_in(needle, haystack):
    """True if `needle` appears in `haystack` as a whole phrase, not a mid-word fragment."""
    return _boundary_pattern(needle).search(haystack) is not None


@lru_cache(maxsize=200_000)
def fuzzy_hit(a, b, threshold=0.82):
    """Loose similarity check for near-matches (typos, minor variants)."""
    return SequenceMatcher(None, a, b).ratio() >= threshold


@lru_cache(maxsize=200_000)
def is_meaningful(phrase):
    """False if every word in `phrase` is a generic filler term (see GENERIC_TERMS)."""
    words = set(_WORD_PATTERN.findall(phrase))
    return bool(words - GENERIC_TERMS)


# NOTE: there is deliberately no 'resume phrase contained inside a longer
# alias' direction. It was tried (gated by is_meaningful/GENERIC_TERMS) and
# repeatedly produced false positives no blocklist could keep up with: any
# compound alias of the form '<tool name> <unrelated-but-real qualifier>'
# (e.g. Django's alias 'django jwt', Java's alias 'java microservices')
# credits the tool just because the resume mentions the qualifier alone
# ('jwt', 'microservices') — those are real technologies, not filler, so no
# generic-word gate can catch them. Structurally unreliable; removed rather
# than patched.
MATCH_RANK = {"exact": 3, "alias-in-resume": 2, "fuzzy": 0}


@lru_cache(maxsize=None)
def _tool_is_matched_cached(aliases, resume_skills):
    """
    Core matcher, keyed on hashable (sorted-tuple) forms of the alias set
    and resume skill set. Because both are constant for the lifetime of a
    single report run, every distinct JD tool gets scored against the
    resume exactly once, regardless of how many downstream components
    (role scoring, relation scoring, tool bridging...) ask for it.
    """
    best = None  # (rank, rskill, match_type)
    for alias in aliases:
        for rskill in resume_skills:
            match_type = None
            if alias == rskill:
                match_type = "exact"
            elif word_boundary_in(alias, rskill):
                match_type = "alias-in-resume"
            elif is_meaningful(rskill) and fuzzy_hit(alias, rskill):
                match_type = "fuzzy"

            if match_type is None:
                continue
            rank = MATCH_RANK[match_type]
            if rank == MATCH_RANK["exact"]:
                return rskill, match_type
            if best is None or rank > best[0]:
                best = (rank, rskill, match_type)
    return (best[1], best[2]) if best else (None, None)


def tool_is_matched(tool_info, resume_skills):
    """
    Returns the STRONGEST evidence linking any resume skill to this tool.
    Thin wrapper: builds the cache key and delegates to
    `_tool_is_matched_cached`.
    """
    aliases_key = tuple(sorted(tool_info["aliases"]))
    resume_key = tuple(sorted(resume_skills))
    return _tool_is_matched_cached(aliases_key, resume_key)


# ---------------------------------------------------------------------------
# Semantic matching (catches paraphrases lexical/fuzzy matching can't see)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def semantic_matches(unmatched_tools, resume_skills, threshold=0.72):
    """
    For tools that lexical/fuzzy matching missed, fall back to embedding
    cosine similarity between resume skill phrases and each tool's aliases.
    Already batches all encodes in two calls (one per side) rather than
    per-pair, which was the right design in the original — left unchanged.
    """
    if not unmatched_tools or not resume_skills:
        return {}
    try:
        model = _get_embed_model()
    except ImportError:
        return {}

    import numpy as np

    resume_list = sorted(s for s in resume_skills if is_meaningful(s))
    if not resume_list:
        return {}
    resume_vecs = model.encode(resume_list, normalize_embeddings=True)

    flat_aliases, tool_bounds = [], []
    for name, info in unmatched_tools.items():
        aliases = sorted(info["aliases"])
        start = len(flat_aliases)
        flat_aliases.extend(aliases)
        tool_bounds.append((name, aliases, start, len(flat_aliases)))

    alias_vecs = model.encode(flat_aliases, normalize_embeddings=True)
    sims = alias_vecs @ resume_vecs.T

    results = {}
    for name, aliases, start, end in tool_bounds:
        sub = sims[start:end]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        score = float(sub[i, j])
        if score >= threshold:
            results[name] = {
                "resume_term": resume_list[j],
                "alias": aliases[i],
                "score": round(score, 3),
            }
    return results


def build_report(resume_json, jd_json, use_semantic=False, semantic_threshold=0.72):
    resume_skills = extract_resume_skills(resume_json)
    action_strengths = build_action_strength_map(resume_json)
    jd_tools = extract_jd_tools(jd_json)

    def strength_for(evidence):
        info = action_strengths.get(evidence, {"strength": DEFAULT_VERB_PRIORITY / MAX_VERB_PRIORITY, "verb": None})
        return info["strength"], info["verb"]

    matched, missing = [], []
    unresolved = {}
    for name, info in jd_tools.items():
        evidence, match_type = tool_is_matched(info, resume_skills)
        entry = {
            "tool": name,
            "tier": info["tier"],
            "weight_pct": info["weight"],
            "matched_resume_term": evidence,
            "match_type": match_type,
        }
        if evidence:
            strength, verb = strength_for(evidence)
            entry["action_strength"] = strength
            entry["action_verb"] = verb
            entry["credited_weight_pct"] = round(info["weight"] * strength, 3)
            matched.append(entry)
        else:
            missing.append(entry)
            unresolved[name] = info

    if use_semantic:
        sem_hits = semantic_matches(unresolved, resume_skills, threshold=semantic_threshold)
        if sem_hits:
            missing = [m for m in missing if m["tool"] not in sem_hits]
            for name, hit in sem_hits.items():
                info = jd_tools[name]
                strength, verb = strength_for(hit["resume_term"])
                matched.append({
                    "tool": name,
                    "tier": info["tier"],
                    "weight_pct": info["weight"],
                    "matched_resume_term": hit["resume_term"],
                    "match_type": f"semantic ({hit['score']:.2f} vs '{hit['alias']}')",
                    "action_strength": strength,
                    "action_verb": verb,
                    "credited_weight_pct": round(info["weight"] * strength, 3),
                })

    raw_total_weight = sum(t["weight"] for t in jd_tools.values())
    matched_weight = sum(m["credited_weight_pct"] for m in matched)
    overall_score = round(100 * matched_weight / raw_total_weight, 1) if raw_total_weight else None

    def tier_score(tier_name):
        tier_total = sum(t["weight"] for t in jd_tools.values() if t["tier"] == tier_name)
        tier_matched = sum(m["credited_weight_pct"] for m in matched if m["tier"] == tier_name)
        if tier_total == 0:
            return None
        return round(100 * tier_matched / tier_total, 1)

    critical_score = tier_score("CRITICAL")
    important_score = tier_score("IMPORTANT")

    matched.sort(key=lambda x: -x["credited_weight_pct"])
    missing.sort(key=lambda x: -x["weight_pct"])

    total_tools = len(matched) + len(missing)
    raw_coverage_pct = round(100 * len(matched) / total_tools, 1) if total_tools else None

    report = {
        "overall_weighted_score_pct": overall_score,
        "critical_tier_score_pct": critical_score,
        "important_tier_score_pct": important_score,
        "raw_coverage_pct": raw_coverage_pct,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "matched_tools": matched,
        "missing_tools": missing,
        "recommendation": recommend(overall_score, critical_score, missing, raw_coverage_pct),
    }
    return report


def recommend(overall, critical, missing, raw_coverage=None):
    top_missing_critical = [m["tool"] for m in missing if m["tier"] == "CRITICAL"][:3]
    if overall is None:
        base = (
            f"No weighted requirements in the JD analysis (all tools UNSCORED) — "
            f"cannot compute a weighted score. Raw tool coverage: {raw_coverage}%"
        )
    elif critical is not None and critical < 40:
        base = "Weak match: resume is missing multiple high-weight CRITICAL requirements"
    elif overall >= 70:
        base = "Strong match"
    elif overall >= 40:
        base = "Partial match: some important gaps"
    else:
        base = "Weak overall match"
    if top_missing_critical:
        base += f". Top missing CRITICAL items: {', '.join(top_missing_critical)}"
    return base


# ---------------------------------------------------------------------------
# Rich-schema scoring
# ---------------------------------------------------------------------------

RICH_COMPONENT_WEIGHTS = {"requirements": 35, "tools": 15, "concepts": 15, "actions": 5, "relations": 5}

def _verb_level(text):
    """Best (highest-priority) verb recognized via verbs_priority_engg.json
    anywhere in `text`. Multi-word variants ('load test', 'roll back',
    'penetration testing') are checked as whole phrases first so they
    aren't mis-split by the single-token fallback pass."""
    variant_to_verb, _, phrase_variants = load_verb_priorities()
    low = text.lower()
    best = None
    for phrase in phrase_variants:
        if word_boundary_in(phrase, low):
            verb, priority, _ = variant_to_verb[phrase]
            if best is None or priority > best[1]:
                best = (verb, priority)
    for word in _VERB_TOKEN_PATTERN.findall(low):
        hit = variant_to_verb.get(word)
        if hit and (best is None or hit[1] > best[1]):
            best = (hit[0], hit[1])
    return best if best else (None, None)


def _verb_level_from_action_entry(action_entry, sr_text=""):
    variant_to_verb, _, _ = load_verb_priorities()
    sr_words = set(_VERB_TOKEN_PATTERN.findall(sr_text.lower()))
    candidates = []
    for v in action_entry.get("actions") or []:
        hit = variant_to_verb.get(v.strip().lower())
        if hit:
            verb, priority, _ = hit
            candidates.append((verb, priority, v.lower() in sr_words))
    if not candidates:
        return None, None
    pool = [c for c in candidates if c[2]] or candidates
    verb, level, _ = max(pool, key=lambda c: c[1])
    return verb, level


def _best_action_for_subreq(sr_text, actions):
    if not actions:
        return None
    if len(actions) == 1:
        return actions[0]
    sr_words = set(re.findall(r"[a-z0-9]+", sr_text.lower()))
    best, best_overlap = None, 0
    for act in actions:
        target_words = set(re.findall(r"[a-z0-9]+", (act.get("target") or "").lower()))
        overlap = len(sr_words & target_words)
        if overlap > best_overlap:
            best, best_overlap = act, overlap
    return best if best_overlap > 0 else None


def action_score(jd_level, resume_level):
    if resume_level >= jd_level:
        return 1.0
    return round(resume_level / jd_level, 3)


TOOL_GRAPH_FILENAME = "know_graph.json"
GRAPH_MAX_HOPS = 2
# Compounded weight lost per hop beyond the first, so a 2-hop bridge (e.g.
# Kong -related-> Envoy -related-> Istio) can never outscore an actual
# direct relation between two tools.
GRAPH_HOP_DECAY = 0.85

_TOOL_GRAPH_CACHE = None


def load_tool_graph(path=TOOL_GRAPH_FILENAME):
    """
    Parses know_graph.json's category tree (Technology -> category ->
    tool/service/library/framework/... leaf nodes, each carrying aliases
    + weighted related_tools) into a flat graph: every leaf becomes a node
    keyed by its lowercased name, with weighted edges to its related_tools
    and the set of aliases it's known by.

    Returns (edges, aliases_of):
      - edges: name -> {related_name: weight}, symmetrized (relatedness
        is treated as mutual even where the tree only lists it from one
        side, e.g. Apigee -> Azure API Management but not vice versa).
      - aliases_of: name -> set of alias strings (includes the node's own
        name), used in extract_jd_tools() to recognize alternate spellings
        the JD extraction's own alias list didn't anticipate.
    """
    global _TOOL_GRAPH_CACHE
    if _TOOL_GRAPH_CACHE is not None:
        return _TOOL_GRAPH_CACHE
    try:
        root = load_json(path)
    except FileNotFoundError:
        _TOOL_GRAPH_CACHE = ({}, {})
        return _TOOL_GRAPH_CACHE

    edges = {}
    aliases_of = {}

    def add_edge(a, b, weight):
        if a == b:
            return
        d = edges.setdefault(a, {})
        if weight > d.get(b, 0):
            d[b] = weight

    def walk(node):
        name = (node.get("name") or "").strip().lower()
        if name and node.get("type") not in ("root", "category"):
            names = aliases_of.setdefault(name, set())
            names.add(name)
            for a in node.get("aliases") or []:
                a = a.strip().lower()
                if a:
                    names.add(a)
            for rt in node.get("related_tools") or []:
                rname = (rt.get("name") or "").strip().lower()
                weight = rt.get("weight", 0)
                if rname:
                    add_edge(name, rname, weight)
                    add_edge(rname, name, weight)
        for c in node.get("children") or []:
            walk(c)

    walk(root)
    _TOOL_GRAPH_CACHE = (edges, aliases_of)
    return _TOOL_GRAPH_CACHE


def _expand_graph_related(edges, max_hops=GRAPH_MAX_HOPS, decay=GRAPH_HOP_DECAY):
    """
    Best reachable relation weight from every graph node to every other
    node within `max_hops`, via weighted BFS. Hop 1 is a node's real edges,
    taken at face value; each hop beyond that multiplies in one extra
    `decay` factor on top of the compounded edge weights, so an indirect
    bridge (tool A unrelated to C directly, but both related to B) never
    outscores an actual direct A<->C relation.
    """
    expanded = {}
    for start in edges:
        best = {}
        frontier = {start: 1.0}
        for hop in range(max_hops):
            next_frontier = {}
            for node, path_weight in frontier.items():
                for neighbor, w in edges.get(node, {}).items():
                    if neighbor == start:
                        continue
                    combined = path_weight * w * (1.0 if hop == 0 else decay)
                    if combined > best.get(neighbor, 0):
                        best[neighbor] = combined
                        next_frontier[neighbor] = combined
            frontier = next_frontier
            if not frontier:
                break
        expanded[start] = best
    return expanded


_TAXONOMY_CACHE = None


def is_rich_schema(jd_json):
    return "sentences" in jd_json and isinstance(jd_json.get("tools"), list)


def extract_resume_tools(resume_json):
    return {n["label"].strip().lower() for n in resume_json.get("nodes", []) if n.get("type") == "tool" and n.get("label", "").strip()}


def load_taxonomy(path="merged_tools.json"):
    """
    concepts_by_tool comes solely from merged_tools.json (know_graph.json
    carries no per-tool concept lists). related_by_tool starts from
    merged_tools.json's flat, 1-hop-only related_tools lists, then has
    know_graph.json's richer graph merged on top (see _expand_graph_related)
    — its 2-hop expansion surfaces bridges merged_tools.json's per-subdomain
    lists can't (e.g. Kong -> Envoy -> Istio), and wherever both sources
    cover the same pair, the higher-confidence weight wins.
    """
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE
    try:
        entries = load_json(path)
    except FileNotFoundError:
        entries = []

    concepts_by_tool, related_by_tool = {}, {}
    for entry in entries:
        name = entry.get("tool", "").strip().lower()
        if not name:
            continue
        concept_set = concepts_by_tool.setdefault(name, set())
        related_map = related_by_tool.setdefault(name, {})
        for data in entry.get("subdomain_data", {}).values():
            concept_set.update(c.strip().lower() for c in data.get("concepts", []) if c.strip())
            for rt in data.get("related_tools", []):
                rt_name = rt.get("tool", "").strip().lower()
                rt_weight = rt.get("weight", 0)
                if rt_name and rt_weight > related_map.get(rt_name, 0):
                    related_map[rt_name] = rt_weight

    graph_edges, _ = load_tool_graph()
    for name, related in _expand_graph_related(graph_edges).items():
        dest = related_by_tool.setdefault(name, {})
        for other, weight in related.items():
            if weight > dest.get(other, 0):
                dest[other] = weight

    _TAXONOMY_CACHE = (concepts_by_tool, related_by_tool)
    return _TAXONOMY_CACHE


def find_resume_node_by_label(resume_json, label):
    label = label.strip().lower()
    for n in resume_json.get("nodes", []):
        if n.get("type") == "tool" and n.get("label", "").strip().lower() == label:
            return n
    return None


@lru_cache(maxsize=None)
def _score_single_tool_cached(name, aliases_key, jd_related, jd_concepts_key, resume_tools_key, resume_skills_key):
    """
    Memoized core of tool bridging. Keyed on hashable tuples so the same
    JD tool scored from multiple roles (Requirements/Tools/Concepts
    components each ask for it independently) is computed once per run.
    """
    evidence, match_type = (
        _tool_is_matched_cached(aliases_key, resume_skills_key) if aliases_key else (None, None)
    )
    if evidence:
        return {"score": 1.0, "via": f"direct ({match_type})", "evidence": evidence}

    concepts_by_tool, related_by_tool = load_taxonomy()
    jd_related_set = set(jd_related)
    jd_concepts_set = set(jd_concepts_key)
    best_score, best_via, best_evidence = 0.0, None, None
    for rtool in resume_tools_key:
        if rtool in jd_related_set:
            score, via = 0.85, "JD related_tools"
        elif name.lower() in related_by_tool.get(rtool, {}):
            score, via = related_by_tool[rtool][name.lower()], "taxonomy related_tools"
        else:
            r_concepts = concepts_by_tool.get(rtool, set())
            if jd_concepts_set and r_concepts:
                overlap = jd_concepts_set & r_concepts
                score = 0.7 * (len(overlap) / len(jd_concepts_set)) if overlap else 0.0
                via = f"concept overlap ({', '.join(sorted(overlap))})" if overlap else None
            else:
                score, via = 0.0, None
        if score > best_score:
            best_score, best_via, best_evidence = score, via, rtool
    return {"score": round(best_score, 3), "via": best_via, "evidence": best_evidence}


def _score_single_tool(jd_tool_entry, resume_tools, resume_skills, jd_tools):
    """
    1.0 if directly matched; otherwise bridge through the tool taxonomy.
    Thin wrapper around the memoized implementation.
    """
    name = jd_tool_entry.get("name", "").strip()
    info = jd_tools.get(name)
    aliases_key = tuple(sorted(info["aliases"])) if info else ()
    related_key = tuple(sorted(r.strip().lower() for r in jd_tool_entry.get("related_tools", [])))
    concepts_key = tuple(sorted(c.strip().lower() for c in jd_tool_entry.get("concepts", [])))
    resume_tools_key = tuple(sorted(resume_tools))
    resume_skills_key = tuple(sorted(resume_skills))
    return _score_single_tool_cached(
        name, aliases_key, related_key, concepts_key, resume_tools_key, resume_skills_key
    )


@lru_cache(maxsize=None)
def _tool_concept_coverage_cached(jd_concepts_key, resume_tools_key):
    if not jd_concepts_key:
        return None
    concepts_by_tool, _ = load_taxonomy()
    resume_concepts = set()
    for rtool in resume_tools_key:
        resume_concepts.update(concepts_by_tool.get(rtool, set()))
    return round(len(set(jd_concepts_key) & resume_concepts) / len(jd_concepts_key), 3)


def _tool_concept_coverage(jd_tool_entry, resume_tools):
    """Fraction of this one JD tool's concepts covered by any resume tool's taxonomy concepts."""
    concepts_key = tuple(sorted(c.strip().lower() for c in jd_tool_entry.get("concepts", []) if c.strip()))
    resume_tools_key = tuple(sorted(resume_tools))
    return _tool_concept_coverage_cached(concepts_key, resume_tools_key)


def compute_tool_score(jd_tools_raw, resume_tools, resume_skills, jd_tools):
    per_tool = []
    for t in jd_tools_raw:
        if not t.get("name", "").strip():
            continue
        result = _score_single_tool(t, resume_tools, resume_skills, jd_tools)
        per_tool.append({"tool": t["name"], **result})
    avg = round(sum(p["score"] for p in per_tool) / len(per_tool), 3) if per_tool else None
    return avg, per_tool


def compute_concept_score(jd_tools_raw, resume_tools):
    concepts_by_tool, _ = load_taxonomy()
    concept_weight = {}
    for t in jd_tools_raw:
        for c in t.get("concepts", []):
            c = c.strip().lower()
            if c:
                concept_weight[c] = concept_weight.get(c, 0) + 1
    if not concept_weight:
        return None, [], []

    resume_concepts = set()
    for rtool in resume_tools:
        resume_concepts.update(concepts_by_tool.get(rtool, set()))

    matched = sorted(c for c in concept_weight if c in resume_concepts)
    missing = sorted(c for c in concept_weight if c not in resume_concepts)
    total_weight = sum(concept_weight.values())
    matched_weight = sum(concept_weight[c] for c in matched)
    score = round(matched_weight / total_weight, 3)
    return score, matched, missing


def _find_anchor_tools(text, jd_tools, explicit_tools=None):
    """
    Which JD tool names a requirement/sentence is 'about'.

    FIX: the JD extraction already tells us this directly for
    sub_requirements — sub_requirement["tools"] (e.g. sub_id "9a":
    text="API testing knowledge", tools=["REST","Postman"]) — but the
    requirement's own *prose* frequently never repeats the tool name
    ("API testing knowledge" doesn't contain "REST" or "Postman"). Scanning
    only the text for alias mentions silently anchors nothing in that case,
    which turned nearly every sub_requirement into 'unverifiable' and
    zeroed out the Requirements component (40%/35% of the score) even
    though the JD literally told us the answer. `explicit_tools` is now
    checked first; text-scanning remains as a fallback/supplement for
    tools named in passing prose but not captured in an explicit list.
    """
    hits = []
    if explicit_tools:
        hits.extend(name for name in explicit_tools if name in jd_tools and name not in hits)
    text_lower = text.lower()
    for name, info in jd_tools.items():
        if name in hits:
            continue
        for alias in info["aliases"]:
            if word_boundary_in(alias, text_lower):
                hits.append(name)
                break
    return hits


TEXT_MATCH_THRESHOLD = 0.5


def _target_phrase_for_subreq(sr_text, action_entry):
    if action_entry and action_entry.get("target"):
        return action_entry["target"]
    return sr_text


def _meaningful_words(phrase):
    """
    Tokenize + drop GENERIC_TERMS. `_WORD_PATTERN` deliberately allows
    embedded periods (so 'node.js' stays one token), which also means a
    sentence-ending period glues onto the last word ('testing.') and stops
    it from matching its own bare form ('testing') in GENERIC_TERMS —
    same trap jd.py's tokenizer documents and strips for the same reason.
    Only ever bites here now that full sentence text (which ends in '.'),
    not just short requirement fragments, gets tokenized for matching.
    """
    words = set()
    for w in _WORD_PATTERN.findall(phrase.lower()):
        if len(w) > 1 and w.endswith("."):
            w = w[:-1]
        words.add(w)
    return words - GENERIC_TERMS


def _best_phrase_overlap(target_phrase, candidate_phrases, threshold=TEXT_MATCH_THRESHOLD):
    """
    Best word-overlap ratio between `target_phrase` and any phrase in
    `candidate_phrases`, measured against the target side so a longer,
    more-qualified candidate ('manual test cases') still fully covers a
    plainer target ('test cases'). Returns (score, matched_phrase), or
    (0.0, None) if nothing clears `threshold`.
    """
    target_words = _meaningful_words(target_phrase)
    if not target_words:
        return 0.0, None
    best_score, best_phrase = 0.0, None
    for phrase in candidate_phrases:
        phrase_words = _meaningful_words(phrase)
        if not phrase_words:
            continue
        overlap = len(target_words & phrase_words) / len(target_words)
        if overlap > best_score:
            best_score, best_phrase = overlap, phrase
    if best_score >= threshold:
        return round(best_score, 3), best_phrase
    return 0.0, None


def _text_fallback_match(target_phrase, resume_activities, action_strengths, threshold=TEXT_MATCH_THRESHOLD):
    """
    Best-effort match of a tool-less JD sub_requirement against the resume's
    action->object pairs, by word overlap between the requirement's target
    phrase (e.g. 'test cases') and each resume object label (e.g. 'manual
    test cases'). Returns (object_score, resume_verb, evidence) or
    (0.0, None, None) if nothing clears the threshold.

    The matched object's verb is read from `action_strengths` (built by
    build_action_strength_map) rather than picked arbitrarily out of
    resume_activities — an object like 'manual test cases' can be the
    target of several verbs ('prepare', 'maintain', 'execute'), and
    action_strengths already knows which of those is the strongest by
    verbs_priority_engg.json priority. Picking the wrong one used to be
    harmless when most verbs were unrecognized anyway; it stopped being
    harmless once the priority table could actually tell them apart.
    """
    score, matched_obj = _best_phrase_overlap(
        target_phrase, [act["object"] for act in resume_activities], threshold
    )
    if score <= 0.0:
        return 0.0, None, None
    verb = (action_strengths.get(matched_obj) or {}).get("verb")
    return score, verb, matched_obj


def compute_role_concept_score(role_entry, resume_skills):
    """
    Score a role's own concept vocabulary (e.g. testlify_analysis's "Manual
    testing" 95%, "Defect tracking" 80%) against the resume, by word overlap
    against every resume object/tool label. This is often a richer signal
    than the merged_tools.json taxonomy concepts (which key off recognized
    *tool* names): most QA/analyst-style JD concepts are activities and
    domains, not named tools, so they'd otherwise never be scored at all.
    Returns (score_fraction, matched_concept_names, missing_concept_names) —
    same contract as compute_concept_score — or (None, [], []) if this role
    carries no concept list.
    """
    concepts = (role_entry or {}).get("concepts") or []
    total_weight = sum(c.get("importance_pct", 0) or 0 for c in concepts)
    if not concepts or total_weight <= 0:
        return None, [], []
    matched, missing = [], []
    credited = 0.0
    for c in concepts:
        name = c.get("concept", "")
        weight = c.get("importance_pct", 0) or 0
        score, _ = _best_phrase_overlap(name, resume_skills)
        credited += weight * score
        (matched if score > 0 else missing).append(name)
    return round(credited / total_weight, 3), sorted(matched), sorted(missing)


def compute_requirement_and_action_scores(jd_json, jd_tools, jd_tools_by_name, resume_skills, resume_tools, action_strengths, resume_activities=(), role=None):
    matched, missing, unverifiable = [], [], []
    action_detail = []
    MATCH_DISPLAY_THRESHOLD = 0.6

    for sentence in jd_json.get("sentences", []):
        if role is not None and sentence.get("dominant_role") != role:
            continue
        subreqs = sentence.get("sub_requirements") or []
        if not subreqs and sentence.get("importance_percentage"):
            # Some sentences (e.g. "Experience in Manual Testing.", "Familiarity
            # with bug-tracking tools such as Jira, Trello...") carry real
            # weight and tier but never got broken into sub_requirements by
            # the JD extraction step. Without this, they're invisible to this
            # whole scoring loop (which only ever iterates sub_requirements)
            # and their weight silently vanishes from the denominator instead
            # of being scored — treat the sentence itself as its one and only
            # sub_requirement so its importance_percentage still counts.
            subreqs = [{
                "sub_id": f"s{sentence['id']}",
                "text": sentence["text"],
                "importance_percentage": sentence["importance_percentage"],
            }]
        for sr in subreqs:
            anchors = _find_anchor_tools(sr["text"], jd_tools, explicit_tools=sr.get("tools"))
            weight = sr.get("importance_percentage", 0) or 0
            entry = {"sub_id": sr["sub_id"], "text": sr["text"], "weight_pct": weight, "anchors": anchors}
            if not anchors:
                action_entry = _best_action_for_subreq(sr["text"], sentence.get("actions") or [])
                target_phrase = _target_phrase_for_subreq(sr["text"], action_entry)
                obj_score, resume_verb, resume_evidence = _text_fallback_match(target_phrase, resume_activities, action_strengths)
                if obj_score <= 0.0:
                    unverifiable.append(entry)
                    continue

                jd_verb, jd_level = _verb_level_from_action_entry(action_entry, sr["text"]) if action_entry else (None, None)
                if jd_level is None:
                    jd_verb, jd_level = _verb_level(sr["text"])
                if jd_level is None or resume_verb is None:
                    act_component = 1.0
                else:
                    _, verb_priority, _ = load_verb_priorities()
                    resume_level = verb_priority.get(resume_verb, DEFAULT_VERB_PRIORITY)
                    act_component = action_score(jd_level, resume_level)
                    action_detail.append({
                        "sub_id": sr["sub_id"], "anchor": "<text-match>", "jd_verb": jd_verb, "jd_level": jd_level,
                        "resume_verb": resume_verb, "resume_level": resume_level, "score": act_component,
                    })

                item_score = round(obj_score * act_component, 3)
                entry["item_score"] = item_score
                entry["per_anchor"] = [{
                    "anchor": "<text-match>", "match_type": "text", "tool_score": obj_score,
                    "action_score": act_component, "concept_score": 1.0, "evidence": resume_evidence,
                }]
                entry["jd_verb"] = jd_verb
                (matched if item_score >= MATCH_DISPLAY_THRESHOLD else missing).append(entry)
                continue

            action_entry = _best_action_for_subreq(sr["text"], sentence.get("actions") or [])
            jd_verb, jd_level = _verb_level_from_action_entry(action_entry, sr["text"]) if action_entry else (None, None)
            if jd_level is None:
                jd_verb, jd_level = _verb_level(sr["text"])
            item_scores, per_anchor = [], []
            for a in anchors:
                evidence, match_type = tool_is_matched(jd_tools[a], resume_skills)
                jd_tool_entry = jd_tools_by_name.get(a, {})
                if evidence:
                    tool_component = 1.0
                else:
                    result = _score_single_tool(jd_tool_entry, resume_tools, resume_skills, jd_tools)
                    tool_component, evidence = result["score"], result["evidence"]
                concept_component = _tool_concept_coverage(jd_tool_entry, resume_tools)
                concept_component = 1.0 if concept_component is None else concept_component

                if jd_level is None:
                    act_component, resume_verb, resume_level = 1.0, None, None
                else:
                    strength_info = action_strengths.get(evidence) if evidence else None
                    resume_verb = strength_info["verb"] if strength_info else None
                    _, verb_priority, _ = load_verb_priorities()
                    resume_level = verb_priority.get(resume_verb, DEFAULT_VERB_PRIORITY) if resume_verb else DEFAULT_VERB_PRIORITY
                    act_component = action_score(jd_level, resume_level)
                    action_detail.append({
                        "sub_id": sr["sub_id"], "anchor": a, "jd_verb": jd_verb, "jd_level": jd_level,
                        "resume_verb": resume_verb, "resume_level": resume_level, "score": act_component,
                    })

                item_scores.append(tool_component * act_component * concept_component)
                per_anchor.append({
                    "anchor": a, "tool_score": tool_component, "action_score": act_component,
                    "concept_score": concept_component, "evidence": evidence,
                })

            item_score = round(sum(item_scores) / len(item_scores), 3)
            entry["item_score"] = item_score
            entry["per_anchor"] = per_anchor
            entry["jd_verb"] = jd_verb
            (matched if item_score >= MATCH_DISPLAY_THRESHOLD else missing).append(entry)

    total_weight = sum(e["weight_pct"] for e in matched) + sum(e["weight_pct"] for e in missing)
    credited = sum(e["weight_pct"] * e["item_score"] for e in matched) + sum(e["weight_pct"] * e["item_score"] for e in missing)
    req_score = round(100 * credited / total_weight, 1) if total_weight else None
    action_avg = round(sum(d["score"] for d in action_detail) / len(action_detail), 3) if action_detail else None

    return {
        "score_pct": req_score,
        "matched": matched,
        "missing": missing,
        "unverifiable": unverifiable,
    }, {
        "score": action_avg,
        "n_scored": len(action_detail),
        "detail": action_detail,
    }


def compute_relation_score(jd_json, resume_json, jd_tools, resume_skills, role_filter=None, tool_roles=None):
    raw_pairs = jd_json.get("tool_relations", [])
    unique_pairs = {frozenset((p["source"], p["target"])) for p in raw_pairs if p.get("source") != p.get("target")}
    if role_filter is not None and tool_roles is not None:
        unique_pairs = {
            pair for pair in unique_pairs
            if (tool_roles.get(tuple(pair)[0]) or tool_roles.get(tuple(pair)[1])) == role_filter
        }
    if not unique_pairs:
        return None, []

    results = []
    for pair in unique_pairs:
        a, b = tuple(pair)
        ev_a, mt_a = tool_is_matched(jd_tools[a], resume_skills) if a in jd_tools else (None, None)
        ev_b, mt_b = tool_is_matched(jd_tools[b], resume_skills) if b in jd_tools else (None, None)
        direct_a = ev_a is not None and mt_a in ("exact", "alias-in-resume")
        direct_b = ev_b is not None and mt_b in ("exact", "alias-in-resume")
        if not (direct_a and direct_b):
            results.append({"pair": [a, b], "score": 0.0, "status": "one or both tools not directly matched"})
            continue
        node_a = find_resume_node_by_label(resume_json, a) or find_resume_node_by_label(resume_json, ev_a)
        node_b = find_resume_node_by_label(resume_json, b) or find_resume_node_by_label(resume_json, ev_b)
        if node_a and node_b and set(node_a.get("sentence_ids", [])) & set(node_b.get("sentence_ids", [])):
            results.append({"pair": [a, b], "score": 1.0, "status": "co-occur in resume"})
        else:
            results.append({"pair": [a, b], "score": 0.5, "status": "both present, not evidenced together"})

    score = round(sum(r["score"] for r in results) / len(results), 3)
    return score, results


def assign_tool_roles(jd_json, jd_tools):
    """
    Which role 'owns' each JD tool.

    FIX: the JD extraction's own tools[] catalog already carries a
    per-tool `dominant_role` (e.g. "REST": dominant_role="QA Engineer") —
    that's checked FIRST and used whenever present. Sentence-text voting
    (unreliable: most tools are named only in a sub_requirement's explicit
    `tools` list, never repeated in the sentence's own prose — see
    _find_anchor_tools) is now only a fallback for tools the JD extraction
    itself left role-less (dominant_role: null, e.g. generic catch-alls
    like 'Jira' that show up as an example rather than a scored
    requirement). Sentence voting also now checks each sentence's
    sub_requirement-level explicit tools, not just literal prose mentions.
    """
    declared_role = {
        t.get("name", ""): t.get("dominant_role")
        for t in jd_json.get("tools", [])
    }

    votes = {name: {} for name in jd_tools}
    for sentence in jd_json.get("sentences", []):
        role = sentence.get("dominant_role")
        if not role:
            continue
        sentence_tools = set()
        for sr in sentence.get("sub_requirements", []) or []:
            sentence_tools.update(sr.get("tools") or [])
        for name in _find_anchor_tools(sentence.get("text", ""), jd_tools, explicit_tools=sentence_tools):
            votes[name][role] = votes[name].get(role, 0) + 1

    result = {}
    for name in jd_tools:
        declared = declared_role.get(name)
        if declared:
            result[name] = declared
        else:
            v = votes.get(name) or {}
            result[name] = max(v, key=v.get) if v else None
    return result


ROLE_COMPONENT_WEIGHTS = {"requirements": 40, "concepts": 25, "tools": 20, "relations": 10}


def compute_role_scores(resume_json, jd_json):
    resume_skills = extract_resume_skills(resume_json)
    resume_tools = extract_resume_tools(resume_json)
    action_strengths = build_action_strength_map(resume_json)
    resume_activities = extract_resume_activities(resume_json)
    jd_tools = extract_jd_tools(jd_json)
    jd_tools_raw = jd_json.get("tools", [])
    jd_tools_by_name = {t.get("name", ""): t for t in jd_tools_raw}
    tool_roles = assign_tool_roles(jd_json, jd_tools)

    role_results = []
    for r in jd_json.get("roles", []):
        role_name = r["name"]
        role_weight = r.get("importance_pct", 0)

        req, action = compute_requirement_and_action_scores(
            jd_json, jd_tools, jd_tools_by_name, resume_skills, resume_tools, action_strengths,
            resume_activities=resume_activities, role=role_name
        )
        role_tools_raw = [t for t in jd_tools_raw if tool_roles.get(t.get("name", "")) == role_name]
        tool_score, tool_detail = compute_tool_score(role_tools_raw, resume_tools, resume_skills, jd_tools)
        # Prefer the role's own concept vocabulary (testlify_analysis, e.g.
        # "Manual testing" 95%) when the JD carries one — richer than the
        # merged_tools.json taxonomy path below, which only knows concepts
        # tied to a recognized tool name and is blind to activity/domain
        # concepts (most of what a QA/analyst-style JD actually lists).
        # Fall back to the taxonomy path for JDs with no role-concept data.
        concept_score, c_matched, c_missing = compute_role_concept_score(r, resume_skills)
        if concept_score is None:
            concept_score, c_matched, c_missing = compute_concept_score(role_tools_raw, resume_tools)
        relation_score, relation_detail = compute_relation_score(
            jd_json, resume_json, jd_tools, resume_skills, role_filter=role_name, tool_roles=tool_roles
        )

        components = {
            "requirements": req["score_pct"],
            "concepts": concept_score * 100 if concept_score is not None else None,
            "tools": tool_score * 100 if tool_score is not None else None,
            "relations": relation_score * 100 if relation_score is not None else None,
        }
        available = {k: v for k, v in components.items() if v is not None}
        if available:
            weight_sum = sum(ROLE_COMPONENT_WEIGHTS[k] for k in available)
            role_score = round(sum(v * ROLE_COMPONENT_WEIGHTS[k] for k, v in available.items()) / weight_sum, 1)
        else:
            role_score = None

        role_results.append({
            "role": role_name,
            "jd_weight_pct": role_weight,
            "score_pct": role_score,
            "component_scores_pct": {k: (round(v, 1) if v is not None else None) for k, v in components.items()},
            "action_score_pct": round(action["score"] * 100, 1) if action["score"] is not None else None,
            "tools_owned_by_role": sorted(t.get("name", "") for t in role_tools_raw),
            "requirements_detail": req,
            "tools_detail": tool_detail,
            "concepts_detail": {"matched": c_matched, "missing": c_missing},
            "relations_detail": relation_detail,
        })

    scored = [r for r in role_results if r["score_pct"] is not None]
    if scored:
        weight_sum = sum(r["jd_weight_pct"] for r in scored)
        overall = round(sum(r["score_pct"] * r["jd_weight_pct"] for r in scored) / weight_sum, 1) if weight_sum else None
    else:
        overall = None

    unassigned_tools = sorted(name for name, role in tool_roles.items() if role is None)

    return {
        "overall_match_pct": overall,
        "role_scores": role_results,
        "unassigned_tools": unassigned_tools,
        "note": "Action Score is shown per role for reference but not part of the weighted role score (matches the "
                "stated Requirements/Concepts/Tools/Relations/Evidence formula). Evidence/Metrics Score omitted: "
                "no captured sentence text/metrics in resume.json to score it from.",
    }


def print_role_report(report):
    overall = report["overall_match_pct"]
    print(f"\nOverall match score (role-weighted): {'N/A' if overall is None else f'{overall}%'}")
    for r in report["role_scores"]:
        score = "N/A" if r["score_pct"] is None else f"{r['score_pct']}%"
        print(f"\n=== {r['role']}  (JD weight {r['jd_weight_pct']}%, score {score}) ===")
        for name, pct in r["component_scores_pct"].items():
            print(f"  {name:<14} {'N/A' if pct is None else f'{pct}%'}")
        if r["action_score_pct"] is not None:
            print(f"  {'actions (ref)':<14} {r['action_score_pct']}%  [not in weighted score]")
        print(f"  tools owned by this role: {', '.join(r['tools_owned_by_role']) or '(none)'}")
        req = r["requirements_detail"]
        if req["missing"]:
            print(f"  weak/missing requirements (item_score < 0.6):")
            for m in req["missing"]:
                print(f"    [{m['weight_pct']}%, score {m['item_score']}] {m['text']}")
        if r["concepts_detail"]["missing"]:
            print(f"  missing concepts: {', '.join(r['concepts_detail']['missing'][:8])}")
        if r["relations_detail"]:
            for rel in r["relations_detail"]:
                print(f"  relation {' <-> '.join(rel['pair']):<28} {rel['score']}  ({rel['status']})")

    if report["unassigned_tools"]:
        print(f"\nTools never named in any JD sentence text (excluded from all roles): {', '.join(report['unassigned_tools'])}")
    print(f"\nNote: {report['note']}\n")


def build_rich_report(resume_json, jd_json):
    resume_skills = extract_resume_skills(resume_json)
    resume_tools = extract_resume_tools(resume_json)
    action_strengths = build_action_strength_map(resume_json)
    resume_activities = extract_resume_activities(resume_json)
    jd_tools = extract_jd_tools(jd_json)
    jd_tools_raw = jd_json.get("tools", [])
    jd_tools_by_name = {t.get("name", ""): t for t in jd_tools_raw}

    req, action = compute_requirement_and_action_scores(
        jd_json, jd_tools, jd_tools_by_name, resume_skills, resume_tools, action_strengths,
        resume_activities=resume_activities
    )
    tool_score, tool_detail = compute_tool_score(jd_tools_raw, resume_tools, resume_skills, jd_tools)
    concept_score, concept_matched, concept_missing = compute_concept_score(jd_tools_raw, resume_tools)
    relation_score, relation_detail = compute_relation_score(jd_json, resume_json, jd_tools, resume_skills)

    components = {
        "requirements": req["score_pct"],
        "tools": tool_score * 100 if tool_score is not None else None,
        "concepts": concept_score * 100 if concept_score is not None else None,
        "actions": action["score"] * 100 if action["score"] is not None else None,
        "relations": relation_score * 100 if relation_score is not None else None,
    }
    available = {k: v for k, v in components.items() if v is not None}
    if available:
        weight_sum = sum(RICH_COMPONENT_WEIGHTS[k] for k in available)
        overall = round(sum(v * RICH_COMPONENT_WEIGHTS[k] for k, v in available.items()) / weight_sum, 1)
    else:
        overall = None

    return {
        "overall_match_pct": overall,
        "component_scores_pct": {k: (round(v, 1) if v is not None else None) for k, v in components.items()},
        "component_weights_used": {k: RICH_COMPONENT_WEIGHTS[k] for k in available},
        "requirements_detail": req,
        "tools_detail": tool_detail,
        "concepts_detail": {"matched": concept_matched, "missing": concept_missing},
        "actions_detail": action,
        "relations_detail": relation_detail,
        "note": "Role Score and Evidence/Metrics Score omitted: resume.json has no comparable role-attribution "
                "or captured-sentence-text/metrics data to score them from.",
    }


def print_rich_report(report):
    overall = report["overall_match_pct"]
    print(f"\nOverall match score: {'N/A' if overall is None else f'{overall}%'}  (components: {', '.join(report['component_weights_used'])})")
    for name, pct in report["component_scores_pct"].items():
        label = f"{pct}%" if pct is not None else "N/A"
        print(f"  {name:<14} {label}")

    req = report["requirements_detail"]
    print(f"\nRequirements matched ({len(req['matched'])}) / weak-or-missing ({len(req['missing'])}) / unverifiable-no-anchor ({len(req['unverifiable'])}):")
    for r in req["missing"]:
        print(f"  [{r['weight_pct']}%, score {r['item_score']}] {r['text']}")

    print(f"\nTool scores:")
    for t in report["tools_detail"]:
        via = f" <- {t['via']} ({t['evidence']})" if t["via"] else ""
        print(f"  {t['tool']:<20} {t['score']}{via}")

    print(f"\nConcept coverage: {len(report['concepts_detail']['matched'])} matched, {len(report['concepts_detail']['missing'])} missing")
    if report["concepts_detail"]["missing"]:
        print(f"  Missing: {', '.join(report['concepts_detail']['missing'][:10])}")

    if report["relations_detail"]:
        print(f"\nTool relations:")
        for r in report["relations_detail"]:
            print(f"  {' <-> '.join(r['pair']):<30} {r['score']}  ({r['status']})")

    print(f"\nNote: {report['note']}\n")


# ---------------------------------------------------------------------------
# Cache management (for batch use: many resumes vs. one JD, or vice versa)
# ---------------------------------------------------------------------------

def clear_caches():
    """
    All the hot-path caches are keyed on the *content* of a run (sorted
    tuples of skills/aliases), so they stay correct across different
    resumes/JDs automatically — call this between runs only if you're
    processing a large batch and want to bound memory growth, or if you've
    mutated GENERIC_TERMS/verbs_priority_engg.json/etc. at runtime.
    """
    global _VERB_PRIORITY_CACHE, _TAXONOMY_CACHE, _TOOL_GRAPH_CACHE
    _VERB_PRIORITY_CACHE = None
    _TAXONOMY_CACHE = None
    _TOOL_GRAPH_CACHE = None
    for fn in (
        _boundary_pattern, fuzzy_hit, is_meaningful, _tool_is_matched_cached,
        _score_single_tool_cached, _tool_concept_coverage_cached,
    ):
        fn.cache_clear()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _looks_like_resume_graph(d):
    return isinstance(d, dict) and "nodes" in d and "edges" in d


def _looks_like_jd_analysis(d):
    return isinstance(d, dict) and "sentences" in d and isinstance(d.get("tools"), list)


def main():
    parser = argparse.ArgumentParser(description="Match resume graph JSON against JD analysis JSON.")
    parser.add_argument("resume", nargs="?", default="resume.json", help="Path to resume knowledge-graph JSON (default: resume.json)")
    parser.add_argument("jd", nargs="?", default="output.json", help="Path to JD analysis JSON (default: output.json)")
    parser.add_argument("--output", "-o", help="Optional path to write full JSON report")
    parser.add_argument(
        "--semantic", action="store_true",
        help="EXPERIMENTAL: also try embedding-based semantic matching for tools lexical/fuzzy "
             "matching missed. Off by default — review any semantic hits manually.",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.72, help="Cosine similarity cutoff for semantic matches (default: 0.72)")
    parser.add_argument(
        "--no-role-split", action="store_true",
        help="For rich-schema JDs with role data, use the single global score instead of per-role scoring.",
    )
    args = parser.parse_args()

    resume_json = load_json(args.resume)
    jd_json = load_json(args.jd)

    if _looks_like_jd_analysis(resume_json) and _looks_like_resume_graph(jd_json):
        print(
            f"Note: '{args.resume}' looks like a JD analysis and '{args.jd}' looks like a "
            f"resume graph — arguments appear swapped, correcting automatically.",
            file=sys.stderr,
        )
        resume_json, jd_json = jd_json, resume_json

    if is_rich_schema(jd_json):
        if jd_json.get("roles") and not args.no_role_split:
            report = compute_role_scores(resume_json, jd_json)
            print_role_report(report)
        else:
            report = build_rich_report(resume_json, jd_json)
            print_rich_report(report)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Full report written to {args.output}")
        return

    report = build_report(resume_json, jd_json, use_semantic=args.semantic, semantic_threshold=args.semantic_threshold)

    score = report["overall_weighted_score_pct"]
    print(f"\nOverall weighted match score: {'N/A (no weighted tools in JD)' if score is None else f'{score}%'}")
    print(f"Raw tool coverage:             {report['raw_coverage_pct']}%  ({report['matched_count']}/{report['matched_count'] + report['missing_count']} tools present)")
    if report["critical_tier_score_pct"] is not None:
        print(f"CRITICAL-tier coverage:       {report['critical_tier_score_pct']}%")
    if report["important_tier_score_pct"] is not None:
        print(f"IMPORTANT-tier coverage:      {report['important_tier_score_pct']}%")
    print(f"\nMatched ({report['matched_count']}):")
    for m in report["matched_tools"]:
        verb = m.get("action_verb") or "?"
        credit = f"{m['weight_pct']}%->{m['credited_weight_pct']}%" if m["weight_pct"] else f"{m['weight_pct']}%"
        print(f"  [{m['tier']:<10}] {m['tool']:<20} ({credit})  <- resume: '{m['matched_resume_term']}' (verb: {verb}, x{m['action_strength']}) [{m['match_type']}]")
    print(f"\nMissing ({report['missing_count']}):")
    for m in report["missing_tools"]:
        print(f"  [{m['tier']:<10}] {m['tool']:<20} ({m['weight_pct']}%)")
    print(f"\nRecommendation: {report['recommendation']}\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Full report written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())