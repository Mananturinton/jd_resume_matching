"""
verb_priority_graph.py
=======================

Turns "responsibility_actions" (from combined_classification.json) into a
verb-priority network graph, scored via verbs_priority_engg.json.

DESIGN vs. the existing "action" nodes in generate_dependency_trees.py's
build_unified_graph_dot(): that function draws ONE ellipse per sentence
bundling all its verbs together (e.g. "Build, maintain -> data pipelines").
This module instead gives EACH verb its own node, deduped GLOBALLY across
the whole JD - so if "Build" appears in 3 different responsibility lines,
that's one "build" node with 3 outgoing edges, not 3 separate nodes. That's
what makes this a genuine priority network instead of disconnected trees.

Node styling:
  - fill/line color by the verb's "category" from verbs_priority_engg.json
    (core_engineering, leadership, devops, security, ...)
  - penwidth scaled by "priority" (1-10) - higher priority verbs draw
    thicker/heavier boxes, same visual language as the existing pct-based
    penwidth() helper for tools/concepts.
  - label shows the verb, its priority, and its category.

Edges:
  - sentence -> verb (dotted, same style as the existing action edges, for
    traceability back to combined_classification.json)
  - verb -> target (the direct object it acts on, e.g. "data pipelines"),
    penwidth scaled by that verb's priority, colored by its category.

Verbs not found under any variant in verbs_priority_engg.json still get a
node (grey, "(unmatched)") rather than silently vanishing - check the
`unmatched` list this module returns/prints so gaps in the verb dataset
(e.g. British spellings like "optimise" vs. the dataset's "optimize") don't
go unnoticed.

Usage (standalone):
    python verb_priority_graph.py \
        --classification combined_classification.json \
        --verbs verbs_priority_engg.json \
        --out dependency_tree_verb_priority.png

Or import build_verb_priority_dot()/extract_verb_graph() directly if you
want to fold this in as another layer of generate_dependency_trees.py's
build_unified_graph_dot() later.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


# --------------------------------------------------------------------------
# small helpers (kept local/duplicated from generate_dependency_trees.py's
# conventions so this file has zero import dependency on your existing
# scripts and can run completely standalone)
# --------------------------------------------------------------------------

def node_id(prefix: str, name: str) -> str:
    return prefix + "_" + "".join(c if c.isalnum() else "_" for c in str(name))


def dot_escape(s: Any) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def render(dot_source: str, out_png: str) -> None:
    proc = subprocess.run(
        ["dot", "-Tpng", "-o", out_png],
        input=dot_source.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8"))


LEAF_FILL, LEAF_LINE, LEAF_TEXT = "#f6f8fa", "#8996a1", "#57646f"

# One color pair per category found in verbs_priority_engg.json. Deliberately
# a DIFFERENT palette from generate_dependency_trees.py's TIER_COLOR /
# ROLE_COLORS / CATEGORY_COLOR (tool vs concept) - verb category is a third,
# unrelated taxonomy, so it gets its own visual language rather than
# colliding with either existing one.
VERB_CATEGORY_COLOR = {
    "core_engineering": {"fill": "#e4ebf2", "line": "#2c4a6e"},  # blue
    "leadership":       {"fill": "#f0e9f5", "line": "#6b4a8a"},  # plum
    "devops":           {"fill": "#e2efec", "line": "#3c6e63"},  # teal
    "performance":      {"fill": "#fbeee0", "line": "#8a5a2c"},  # amber
    "security":         {"fill": "#f7e6dd", "line": "#a8471f"},  # brick
    "qa_testing":       {"fill": "#eaf1fb", "line": "#31577a"},  # steel blue
    "data":             {"fill": "#eef7f0", "line": "#2f6b45"},  # green
    "cloud_infra":      {"fill": "#f3ecf7", "line": "#5c3d7a"},  # purple
    "release_ops":      {"fill": "#fdf0e6", "line": "#a5651f"},  # orange
    "observability":    {"fill": "#e9f4f4", "line": "#276e6e"},  # teal-cyan
    "documentation":    {"fill": "#f5f0e6", "line": "#7a651f"},  # tan
    "growth":           {"fill": LEAF_FILL, "line": LEAF_LINE},  # neutral
    "unclassified":     {"fill": LEAF_FILL, "line": LEAF_LINE},  # neutral
}


def _penwidth_for_priority(priority: int | None) -> float:
    """1.0 (priority 1 or unmatched) .. 2.8 (priority 10), same scale
    philosophy as generate_dependency_trees.py's penwidth(pct)."""
    if priority is None:
        return 1.0
    return round(1.0 + (priority / 10) * 1.8, 2)


# --------------------------------------------------------------------------
# lookup + matching
# --------------------------------------------------------------------------

def load_verb_lookup(verbs_priority_path: str) -> dict[str, dict]:
    """variant text (lowercase) -> the full verb_priorities entry it belongs to."""
    with open(verbs_priority_path, encoding="utf-8") as f:
        data = json.load(f)

    lookup: dict[str, dict] = {}
    for entry in data.get("verb_priorities", []):
        for variant in entry.get("variants", []):
            lookup[variant.strip().lower()] = entry
        # also index the canonical verb key itself (with underscores turned
        # back into spaces) in case it's ever referenced directly and isn't
        # already covered by variants, e.g. "scale_up" -> "scale up"
        canonical_as_text = entry["verb"].replace("_", " ").strip().lower()
        lookup.setdefault(canonical_as_text, entry)
    return lookup


def match_verb(raw_verb: str, lookup: dict[str, dict]) -> dict | None:
    """Resolve a raw verb string from responsibility_actions (e.g. 'Building',
    'led', 'optimise') to its verbs_priority_engg.json entry. Tries an exact
    match on the lowercased text first (handles multi-word variants like
    'load test'), then a light suffix-stripping fallback for inflections not
    explicitly listed as a variant. Does NOT attempt British/American
    spelling normalization (e.g. 'optimise' won't match 'optimize') - that
    shows up as unmatched, which is worth knowing rather than silently
    guessing.
    """
    key = raw_verb.strip().lower()
    if key in lookup:
        return lookup[key]
    for suffix in ("ing", "ed", "es", "s"):
        if key.endswith(suffix) and key[: -len(suffix)] in lookup:
            return lookup[key[: -len(suffix)]]
    return None


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def extract_verb_graph(
    classification: dict,
    lookup: dict[str, dict],
) -> tuple[dict[str, dict], list[dict], list[str]]:
    """
    Reads classification["responsibility_actions"] (list of
    {"sentence_id", "actions": [...], "target": "..."}).

    Returns:
      verb_nodes: {canonical_verb: {"verb", "priority", "category", "matched"}}
                  deduped globally - one entry per distinct verb in the JD.
      edges: [{"verb", "raw_verb", "sentence_id", "target", "priority", "category"}, ...]
             one per (verb, target) occurrence - NOT deduped, since the same
             verb acting on different targets in different sentences is
             exactly the network structure we want to see.
      unmatched: sorted list of raw verb strings that had no match in
                 verbs_priority_engg.json under any variant - worth a glance,
                 these silently become priority=None / category=unclassified
                 nodes rather than errors.
    """
    verb_nodes: dict[str, dict] = {}
    edges: list[dict] = []
    unmatched: set[str] = set()

    for act in classification.get("responsibility_actions", []):
        sentence_id = act.get("sentence_id")
        target = (act.get("target") or "").strip()

        for raw_verb in act.get("actions", []):
            entry = match_verb(raw_verb, lookup)
            if entry:
                canonical = entry["verb"]
                priority = entry["priority"]
                category = entry["category"]
            else:
                canonical = raw_verb.strip().lower().replace(" ", "_")
                priority = None
                category = "unclassified"
                unmatched.add(raw_verb)

            if canonical not in verb_nodes:
                verb_nodes[canonical] = {
                    "verb": canonical,
                    "priority": priority,
                    "category": category,
                    "matched": entry is not None,
                }

            edges.append({
                "verb": canonical,
                "raw_verb": raw_verb,
                "sentence_id": sentence_id,
                "target": target,
                "priority": priority,
                "category": category,
            })

    return verb_nodes, edges, sorted(unmatched)


def verb_graph_to_json(classification: dict, lookup: dict[str, dict]) -> dict:
    """
    Flat shape that mirrors the PNG directly - one entry per verb (deduped
    globally, same as a node in the picture) with its outgoing connections
    nested right under it, instead of a generic nodes/relationships/
    properties graph schema:

      {
        "verbs": [
          {
            "verb": "build",
            "priority": 10,
            "category": "core_engineering",
            "matched": true,
            "sentences": [4, 12],
            "connections": [
              {"target": "data pipelines", "sentence_id": 4},
              {"target": "RESTful APIs", "sentence_id": 12}
            ]
          },
          ...
        ],
        "unmatched_verbs": ["optimise"]
      }

    "connections" is that verb's list of (target, sentence_id) pairs - one
    per occurrence, so a verb used in 3 sentences has 3 entries there, same
    as its 3 outgoing edges in the PNG. priority/category live once at the
    verb level (they don't change per-connection), so they're not repeated
    on every connection.
    """
    verb_nodes, edges, unmatched = extract_verb_graph(classification, lookup)

    connections_by_verb: dict[str, list[dict]] = {}
    sentences_by_verb: dict[str, list[int]] = {}

    for e in edges:
        if e["target"]:
            connections_by_verb.setdefault(e["verb"], []).append({
                "target": e["target"],
                "sentence_id": e["sentence_id"],
            })
        if e["sentence_id"] is not None:
            bucket = sentences_by_verb.setdefault(e["verb"], [])
            if e["sentence_id"] not in bucket:
                bucket.append(e["sentence_id"])

    verbs_out = []
    for verb, info in sorted(
        verb_nodes.items(), key=lambda kv: (kv[1]["priority"] is None, -(kv[1]["priority"] or 0))
    ):
        verbs_out.append({
            "verb": verb,
            "priority": info["priority"],
            "category": info["category"],
            "matched": info["matched"],
            "sentences": sentences_by_verb.get(verb, []),
            "connections": connections_by_verb.get(verb, []),
        })

    return {"verbs": verbs_out, "unmatched_verbs": unmatched}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def build_verb_priority_dot(
    classification: dict,
    lookup: dict[str, dict],
    title_suffix: str = "",
) -> tuple[str, list[str]]:
    """Returns (dot_source, unmatched_verbs)."""
    verb_nodes, edges, unmatched = extract_verb_graph(classification, lookup)

    lines = ["digraph VerbPriorityGraph {"]
    lines.append("  rankdir=LR;")
    lines.append('  bgcolor="white";')
    lines.append("  pad=0.35; nodesep=0.3; ranksep=0.9;")
    lines.append('  fontname="Helvetica";')
    lines.append(
        f'  label="Action-verb priority network — from responsibility_actions, '
        f'scored via verbs_priority_engg.json{dot_escape(title_suffix)}\\n'
        f'node border weight = priority (1-10) · color = category · dotted = which sentence it came from";'
    )
    lines.append('  labelloc="t"; fontsize=14; fontcolor="#121a22";')
    lines.append('  node [fontname="Helvetica", style="rounded,filled", fontsize=11];')
    lines.append('  edge [fontname="Helvetica", fontsize=9, color="#aab6bf", fontcolor="#57646f", arrowsize=0.7];')

    # verb nodes, sorted by priority descending so the DOT source itself
    # reads high-to-low (doesn't affect layout, just easier to skim/diff)
    for verb, info in sorted(
        verb_nodes.items(), key=lambda kv: (kv[1]["priority"] is None, -(kv[1]["priority"] or 0))
    ):
        color = VERB_CATEGORY_COLOR.get(info["category"], VERB_CATEGORY_COLOR["unclassified"])
        v_node = node_id("verb", verb)
        label = dot_escape(verb.replace("_", " "))
        if info["priority"] is not None:
            label += f"\\np{info['priority']} · {dot_escape(info['category'])}"
        else:
            label += "\\n(unmatched)"
        lines.append(
            f'  {v_node} [label="{label}", shape=box, fillcolor="{color["fill"]}", '
            f'color="{color["line"]}", fontcolor="{color["line"]}", '
            f'penwidth={_penwidth_for_priority(info["priority"])}, margin="0.16,0.1"];'
        )

    # sentence stand-in nodes (lightweight - just "s<id>") + sentence->verb
    # dotted edges, and verb->target priority-weighted edges
    seen_sentence_nodes: set[str] = set()
    seen_sentence_edges: set[tuple[str, str]] = set()
    seen_targets: set[str] = set()

    for e in edges:
        v_node = node_id("verb", e["verb"])

        if e["sentence_id"] is not None:
            s_key = f's{e["sentence_id"]}'
            s_node = node_id("sent", s_key)
            if s_node not in seen_sentence_nodes:
                seen_sentence_nodes.add(s_node)
                lines.append(
                    f'  {s_node} [label="Sentence {dot_escape(e["sentence_id"])}", shape=ellipse, '
                    f'fillcolor="{LEAF_FILL}", color="{LEAF_LINE}", fontcolor="{LEAF_TEXT}", '
                    f'fontsize=9, margin="0.1,0.04"];'
                )
            if (s_node, v_node) not in seen_sentence_edges:
                seen_sentence_edges.add((s_node, v_node))
                lines.append(f'  {s_node} -> {v_node} [style=dotted, constraint=false];')

        if e["target"]:
            t_node = node_id("vtarget", e["target"])
            if t_node not in seen_targets:
                seen_targets.add(t_node)
                lines.append(
                    f'  {t_node} [label="{dot_escape(e["target"])}", shape=box, '
                    f'fillcolor="{LEAF_FILL}", color="{LEAF_LINE}", fontcolor="{LEAF_TEXT}", '
                    f'fontsize=10, margin="0.14,0.06"];'
                )
            edge_color = VERB_CATEGORY_COLOR.get(e["category"], VERB_CATEGORY_COLOR["unclassified"])["line"]
            pen = _penwidth_for_priority(e["priority"])
            edge_label = f'p{e["priority"]}' if e["priority"] is not None else "?"
            lines.append(
                f'  {v_node} -> {t_node} [label="{edge_label}", penwidth={pen}, color="{edge_color}"];'
            )

    lines.append("}")
    return "\n".join(lines), unmatched


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description="Build a verb-priority network graph from responsibility_actions.")
    p.add_argument("--classification", required=True, help="Path to combined_classification.json")
    p.add_argument("--verbs", required=True, help="Path to verbs_priority_engg.json")
    p.add_argument("--json-out", help="Write the graph as JSON to this path (nodes/relationships shape)")
    p.add_argument("--out", help="Also render a PNG to this path (requires graphviz's 'dot' on PATH)")
    args = p.parse_args()

    if not args.json_out and not args.out:
        p.error("specify at least one of --json-out or --out")

    with open(args.classification, encoding="utf-8") as f:
        classification = json.load(f)

    lookup = load_verb_lookup(args.verbs)

    if args.json_out:
        result = verb_graph_to_json(classification, lookup)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")
        print(f"  verbs: {len(result['verbs'])}")
        if result["unmatched_verbs"]:
            print(f"  unmatched verbs: {result['unmatched_verbs']}")

    if args.out:
        dot_source, unmatched = build_verb_priority_dot(classification, lookup)
        render(dot_source, args.out)
        print(f"wrote {args.out}")
        if unmatched:
            print(f"  unmatched verbs: {unmatched}")


if __name__ == "__main__":
    main()