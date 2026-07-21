"""
generate_dependency_trees.py — regenerates dependency-tree PNGs straight from
combined_classification.json.

Kinds of PNG that come out of this, all derived from the *current* contents
of combined_classification.json (nothing hardcoded to a specific JD):

  - dependency_tree.png
        The RELATED_TO graph between extracted tools, built via jd.py's own
        build_jd_graph_from_classification().
  - dependency_graph_unified.png
        role -> sentence -> sub_requirement -> tool, plus taxonomy relations,
        PLUS each sentence's GPT-inferred "entities" and their
        "parent_relations" (populated at classification time by
        job_description_classifier.py — NOT from the KB dataset). E.g. for
        "Build backend services in Node.js" this same file also draws
        Node.js -> JavaScript (USES_LANGUAGE), Node.js -> V8 Engine (BUILT_ON),
        etc., right alongside that sentence's sub_requirement/tool nodes.
        Everything sentence-level lives in this ONE file — there is no
        separate PNG per sentence.
  - dependency_tree_kb_enriched.png (only if entity_kb.py/jd_kb_bridge.py
    are present and something in the JD matches the 221-entity KB dataset)
        Each JD tool matched against the KB dataset, one hop into its
        REQUIRES / RELATED_BY_CONCEPT / RELEVANT_TO_ROLE detail.
  - dependency_tree_<role_slug>.png (one per role)
        One tree per role in testlify_analysis.roles_ranked: role -> its
        concepts (with importance_pct) -> the concrete match_keywords each
        concept's sub_requirement carries in processed_sentences. Concepts
        are matched back to a sub_requirement by keyword overlap rather than
        a hand-typed table, so this stays correct across totally different
        JDs/roles.

A manifest (.dependency_trees_manifest.json) tracks which PNGs were
generated on the last run, so renamed/removed roles don't leave stale PNGs
behind (this is what caused confusion the first time the JD was swapped).

Run directly, or via watch_dependency_trees.py for automatic regeneration
on save.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_PATH = os.path.join(HERE, "combined_classification.json")
MANIFEST_PATH = os.path.join(HERE, ".dependency_trees_manifest.json")

sys.path.insert(0, HERE)
import jd  # noqa: E402

# entity_kb.py / jd_kb_bridge.py are optional — the KB-enriched PNG is only
# generated when both are present alongside this script and a KB data
# directory (containing the 19 knowledge-graph shard files) exists. Their
# absence doesn't break the other PNGs this script produces — in particular
# the per-sentence entity graph below is independent of the KB and works
# with or without entity_kb.py, since it's built from GPT's own inferred
# "entities"/"parent_relations" rather than a KB lookup.
try:
    from entity_kb import EntityKB
    import jd_kb_bridge
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False

TIER_COLOR = {
    "CRITICAL": {"fill": "#f7e6dd", "line": "#a8471f"},
    "IMPORTANT": {"fill": "#e2efec", "line": "#3c6e63"},
    "GENERIC": {"fill": "#f6f8fa", "line": "#8996a1"},
}
DEFAULT_TIER_COLOR = {"fill": "#f6f8fa", "line": "#8996a1"}

ROLE_COLORS = [
    {"fill": "#e4ebf2", "line": "#2c4a6e"},   # ink-blue
    {"fill": "#e2efec", "line": "#3c6e63"},   # teal
    {"fill": "#f7e6dd", "line": "#a8471f"},   # brick
    {"fill": "#f0e9f5", "line": "#6b4a8a"},   # plum
    {"fill": "#fbeee0", "line": "#8a5a2c"},   # amber
]
LEAF_FILL = "#f6f8fa"
LEAF_LINE = "#8996a1"
LEAF_TEXT = "#57646f"

# Colors for the entity/concept layer in the unified graph, keyed by the
# "category" field GPT assigns in job_description_classifier.py — exactly
# two buckets now: "tool" (concrete, installable technology) vs "concept"
# (abstract idea/pattern/practice). Parent-relation targets carry their own
# "category" too, so they're colored the same way rather than a flat grey.
CATEGORY_COLOR = {
    "tool": {"fill": "#e4ebf2", "line": "#2c4a6e"},      # blue
    "concept": {"fill": "#f0e9f5", "line": "#6b4a8a"},   # plum
}
DEFAULT_CATEGORY_COLOR = {"fill": "#f6f8fa", "line": "#8996a1"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def node_id(prefix, name):
    return prefix + "_" + "".join(c if c.isalnum() else "_" for c in name)


def dot_escape(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def render(dot_source, out_png):
    # Write to a fresh temp file and rename it onto out_png rather than having
    # `dot` open out_png directly: out_png lives in a OneDrive-synced folder,
    # and OneDrive's cloud-file reparse point occasionally makes third-party
    # apps get ERROR_ACCESS_DENIED opening it for writing mid-sync. A rename
    # is an OS-level operation that doesn't hit that path.
    tmp_png = out_png + ".tmp"
    proc = subprocess.run(
        ["dot", "-Tpng", "-o", tmp_png], input=dot_source.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8"))
    os.replace(tmp_png, out_png)


# --------------------------------------------------------------------------
# 1. RELATED_TO relationship graph
# --------------------------------------------------------------------------

def build_related_to_dot(classification, result):
    nodes = {n["id"]: n for n in result["graph"]["nodes"]}
    rels = result["graph"]["relationships"]
    related = []
    seen = set()
    for r in rels:
        if r["type"] != "RELATED_TO":
            continue
        key = (r["source"], r["target"])
        if key in seen:
            continue
        seen.add(key)
        related.append(r)

    tool_ids_in_edges = {r["source"] for r in related} | {r["target"] for r in related}
    all_tool_ids = [nid for nid in nodes if nid.startswith("tool::")]

    roles = classification.get("chatgpt_roles") or []
    title = " / ".join(roles) if roles else classification.get("predicted_role", "")

    lines = ["digraph JDDependencyTree {"]
    lines.append("  rankdir=TB;")
    lines.append('  bgcolor="white";')
    lines.append("  pad=0.3; nodesep=0.45; ranksep=0.6;")
    lines.append('  fontname="Helvetica";')
    lines.append(
        f'  label="JD Dependency Tree — RELATED_TO edges between extracted tools\\n'
        f'({dot_escape(title)} — combined_classification.json)";'
    )
    lines.append('  labelloc="t"; fontsize=14; fontcolor="#121a22";')
    lines.append(
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=13, penwidth=1.4, margin="0.22,0.12"];'
    )
    lines.append(
        '  edge [color="#8996a1", fontname="Helvetica", fontsize=10, '
        'fontcolor="#57646f", penwidth=1.1, arrowsize=0.8];'
    )

    ids_by_tool = {}
    for tid in all_tool_ids:
        props = nodes[tid]["properties"]
        name = props["name"]
        gid = node_id("tool", name)
        ids_by_tool[tid] = gid
        tier = props.get("tier")
        pct = props.get("importance_percentage")
        color = TIER_COLOR.get(tier, DEFAULT_TIER_COLOR)
        label = dot_escape(name)
        if tier and pct is not None:
            label += f"\\n{dot_escape(tier)} · {pct}%"
        lines.append(
            f'  {gid} [label="{label}", fillcolor="{color["fill"]}", '
            f'color="{color["line"]}", fontcolor="{color["line"]}"];'
        )

    for r in related:
        if r["source"] in ids_by_tool and r["target"] in ids_by_tool:
            lines.append(f'  {ids_by_tool[r["source"]]} -> {ids_by_tool[r["target"]]} [label="related_to"];')

    isolated = [ids_by_tool[t] for t in all_tool_ids if t not in tool_ids_in_edges]
    if isolated:
        lines.append("  { rank=same; " + "; ".join(isolated) + " }")

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 2. Unified graph: Role -> Sentence -> Sub-requirement -> Tool, plus
#    taxonomy RELATED_TO edges between tools, PLUS Sentence -> GPT-inferred
#    Entity -> Parent node (see "entities"/"parent_relations" on each
#    processed_sentence, populated by job_description_classifier.py from
#    GPT's own knowledge — not the KB dataset). Every node traces back to a
#    specific piece of combined_classification.json (no node is asserted
#    without a visible source), and nothing is left an isolated island: a
#    tool with no taxonomy relations still hangs off its sub_requirement,
#    which hangs off its sentence, which hangs off its role(s); an entity
#    with no parent_relations still hangs off its sentence the same way.
# --------------------------------------------------------------------------

def _wrap(text, width=34):
    """Word-wrap, then dot_escape — in that order. dot_escape must run
    *after* the "\\n" line breaks are joined in, otherwise it doubles the
    backslash in "\\n" and Graphviz prints a literal backslash-n instead
    of breaking the line."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + 1 + len(w) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return dot_escape("\n".join(lines)).replace("\n", "\\n")


def build_unified_graph_dot(classification, result):
    """Renders jd.unified_graph_json()'s data as a dot graph — this is the
    single source of truth for the role/sentence/sub_requirement/tool
    structure; this function only turns it into pixels. Keeping the JSON
    (jd.py's default CLI output) and the PNG built from the exact same
    function call means they can never drift apart."""
    graph = jd.unified_graph_json(classification, result)

    role_color = {r["name"]: ROLE_COLORS[i % len(ROLE_COLORS)] for i, r in enumerate(graph["roles"])}
    title = " / ".join(r["name"] for r in graph["roles"]) or classification.get("predicted_role", "")

    lines = ["digraph JDUnifiedGraph {"]
    lines.append("  rankdir=LR;")
    lines.append('  bgcolor="white";')
    lines.append("  pad=0.35; nodesep=0.22; ranksep=1.0;")
    lines.append('  fontname="Helvetica";')
    lines.append(
        f'  label="{dot_escape(title)} — full traceability graph from combined_classification.json\\n'
        f'role → sentence → sub-requirement → tool (solid), plus taxonomy relations between tools\\n'
        f'and sentence → GPT-inferred entity → parent relation (dotted/dashed)";'
    )
    lines.append('  labelloc="t"; fontsize=15; fontcolor="#121a22";')
    lines.append('  node [fontname="Helvetica", style="rounded,filled", fontsize=11];')
    lines.append('  edge [color="#aab6bf", fontname="Helvetica", fontsize=9, fontcolor="#57646f", arrowsize=0.7];')

    for role in graph["roles"]:
        color = role_color.get(role["name"], DEFAULT_TIER_COLOR)
        pct = role["importance_pct"]
        label = dot_escape(role["name"]) + (f"\\n{pct}% of JD" if pct is not None else "")
        lines.append(
            f'  {node_id("role", role["name"])} [label="{label}", shape=box, '
            f'fillcolor="{color["line"]}", fontcolor="white", fontsize=13, '
            f'penwidth=2, margin="0.22,0.14"];'
        )

    # Entities/parent_relations live on classification["processed_sentences"],
    # not on jd.unified_graph_json()'s "graph" object — read them straight
    # from the classification dict (keyed by sentence id) so this works
    # regardless of whether jd.py forwards that field itself.
    entities_by_sentence_id = {
        s.get("id"): (s.get("entities") or [])
        for s in classification.get("processed_sentences", [])
    }
    seen_parents = set()

    seen_tools = set()
    for sentence in graph["sentences"]:
        sid = f"s{sentence['id']}"
        sent_node = node_id("sent", sid)
        tier = sentence.get("tier")
        color = TIER_COLOR.get(tier, DEFAULT_TIER_COLOR)
        label = f"{_wrap(sentence['text'])}\\n[{dot_escape(tier)}]"
        lines.append(
            f'  {sent_node} [label="{label}", shape=box, '
            f'fillcolor="{color["fill"]}", color="{color["line"]}", fontcolor="{color["line"]}", '
            f'margin="0.16,0.1"];'
        )

        dominant = sentence.get("dominant_role")
        for role_name, pct in (sentence.get("role_attribution") or {}).items():
            if role_name not in role_color:
                continue
            is_dominant = role_name == dominant
            style = "bold" if is_dominant else "dashed"
            penwidth = 1.6 if is_dominant else 1.0
            lines.append(
                f'  {node_id("role", role_name)} -> {sent_node} '
                f'[label="{pct}%", style="{style}", penwidth={penwidth}];'
            )

        for act in sentence.get("actions") or []:
            action_node = node_id("action", act["id"])
            verbs = ", ".join(act["actions"])
            label = f'{_wrap(verbs)}\\n→ {_wrap(act["target"] or "")}'
            lines.append(
                f'  {action_node} [label="{label}", shape=ellipse, '
                f'fillcolor="#fbeee0", color="#8a5a2c", fontcolor="#8a5a2c", '
                f'fontsize=9, margin="0.12,0.05"];'
            )
            lines.append(
                f'  {sent_node} -> {action_node} [style=dotted, color="#8a5a2c", constraint=false];'
            )

        for sub in sentence["sub_requirements"]:
            sub_node = node_id("sub", sub["sub_id"])
            sub_pct = sub.get("importance_percentage")
            sub_label = _wrap(sub["text"]) + (f"\\n{sub_pct}%" if sub_pct is not None else "")
            lines.append(
                f'  {sub_node} [label="{sub_label}", shape=box, '
                f'fillcolor="{LEAF_FILL}", color="{LEAF_LINE}", fontcolor="{LEAF_TEXT}", '
                f'fontsize=10, margin="0.14,0.08"];'
            )
            lines.append(f"  {sent_node} -> {sub_node};")

            for tool in sub["tools"]:
                tool_node = node_id("tool", tool)
                seen_tools.add(tool)
                lines.append(f"  {sub_node} -> {tool_node};")

        # GPT-inferred entities for this sentence, plus each entity's
        # "parent_relations" — a separate visual layer (dotted from sentence
        # to entity, dashed from entity to parent) so it reads distinctly
        # from the solid sub_requirement -> tool chain above, even though
        # both can originate from the same sentence node. Every entity and
        # every parent target carries its own "category" ("tool" or
        # "concept"), so both ends of each edge get colored by what they
        # actually are, not by a flat neutral grey.
        for ent in entities_by_sentence_id.get(sentence["id"], []):
            ename = ent.get("entity")
            if not ename:
                continue
            ecategory = ent.get("category")
            ecolor = CATEGORY_COLOR.get(ecategory, DEFAULT_CATEGORY_COLOR)
            e_node = node_id("ent", ename)
            e_label = dot_escape(ename) + (f"\\n[{dot_escape(ecategory)}]" if ecategory else "")
            lines.append(
                f'  {e_node} [label="{e_label}", shape=box, fillcolor="{ecolor["fill"]}", '
                f'color="{ecolor["line"]}", fontcolor="{ecolor["line"]}", fontsize=10, '
                f'penwidth=1.4, margin="0.14,0.08"];'
            )
            lines.append(
                f'  {sent_node} -> {e_node} [style=dotted, color="#8996a1", constraint=false];'
            )

            for rel in ent.get("parent_relations") or []:
                target = rel.get("target")
                rtype = rel.get("relation_type", "RELATED_TO")
                if not target:
                    continue
                pcategory = rel.get("category")
                pcolor = CATEGORY_COLOR.get(pcategory, DEFAULT_CATEGORY_COLOR)
                p_node = node_id("parent", target)
                if p_node not in seen_parents:
                    seen_parents.add(p_node)
                    p_label = dot_escape(target) + (f"\\n[{dot_escape(pcategory)}]" if pcategory else "")
                    lines.append(
                        f'  {p_node} [label="{p_label}", shape=box, '
                        f'fillcolor="{pcolor["fill"]}", color="{pcolor["line"]}", fontcolor="{pcolor["line"]}", '
                        f'fontsize=9, penwidth=1, margin="0.12,0.05"];'
                    )
                lines.append(
                    f'  {e_node} -> {p_node} [label="{dot_escape(rtype.lower())}", '
                    f'style="dashed", color="{pcolor["line"]}", fontsize=8, constraint=false];'
                )

    tool_props = {t["name"]: t for t in graph["tools"]}
    for name, tool_node_props in tool_props.items():
        tool_node = node_id("tool", name)
        tier2 = tool_node_props.get("tier")
        pct2 = tool_node_props.get("importance_percentage")
        role2 = tool_node_props.get("dominant_role")
        color2 = TIER_COLOR.get(tier2, DEFAULT_TIER_COLOR)
        t_label = dot_escape(name)
        if tier2 and pct2 is not None:
            t_label += f"\\n{dot_escape(tier2)} · {pct2}%"
        if role2:
            t_label += f"\\n{dot_escape(role2)}"
        lines.append(
            f'  {tool_node} [label="{t_label}", shape=box, '
            f'fillcolor="{color2["fill"]}", color="{color2["line"]}", '
            f'fontcolor="{color2["line"]}", penwidth=1.6, margin="0.16,0.1"];'
        )

    for rel in graph["tool_relations"]:
        lines.append(
            f'  {node_id("tool", rel["source"])} -> {node_id("tool", rel["target"])} '
            f'[label="related_to", style="dashed", color="#8996a1", constraint=false];'
        )

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 2b. KB-enriched entity detail graph: every tool that resolves in
#     entity_kb.py's 221-entity dataset, expanded one hop into its full
#     detail — roles it's relevant to (with importance), prerequisites it
#     resolves to, and entities it shares concepts with. This is built
#     straight off jd_kb_bridge's enriched graph (KBEntity/Role/Category
#     nodes, REQUIRES/RELATED_BY_CONCEPT/RELEVANT_TO_ROLE/MATCHES_KB_ENTITY
#     edges) — same "read the graph, don't recompute it" principle as
#     build_unified_graph_dot above.
# --------------------------------------------------------------------------

KB_EDGE_STYLE = {
    "RELEVANT_TO_ROLE": {"color": "#2c4a6e", "style": "solid"},
    "REQUIRES": {"color": "#a8471f", "style": "bold"},
    "RELATED_BY_CONCEPT": {"color": "#6b4a8a", "style": "dashed"},
}
KB_ENTITY_FILL = "#eef3f8"
KB_ENTITY_LINE = "#2c4a6e"
ROLE_NODE_FILL = "#e4ebf2"
PREREQ_FILL = "#f7e6dd"
RELATED_FILL = "#f0e9f5"


def build_kb_enriched_dot(enriched_result, max_roles_per_entity=4, max_related_per_entity=4):
    """`enriched_result` is jd_kb_bridge.build_enriched_graph[_from_classification]'s
    output — a jd.py result whose graph already has KBEntity/Role/Category
    nodes merged in. Returns None if nothing in this JD matched the KB (no
    KBEntity nodes to draw)."""
    nodes = {n["id"]: n for n in enriched_result["graph"]["nodes"]}
    rels = enriched_result["graph"]["relationships"]

    if not any(n["label"] == "KBEntity" for n in nodes.values()):
        return None

    matches_edges = [r for r in rels if r["type"] == "MATCHES_KB_ENTITY"]
    lines = ["digraph KBEnrichedGraph {"]
    lines.append("  rankdir=LR;")
    lines.append('  bgcolor="white";')
    lines.append("  pad=0.35; nodesep=0.25; ranksep=1.1;")
    lines.append('  fontname="Helvetica";')
    cov = enriched_result.get("kb_coverage", {})
    lines.append(
        f'  label="Entity detail graph — each JD tool matched against the 221-entity dataset\\n'
        f'({len(cov.get("matched", []))} matched, {len(cov.get("unmatched", []))} not in this dataset)";'
    )
    lines.append('  labelloc="t"; fontsize=15; fontcolor="#121a22";')
    lines.append('  node [fontname="Helvetica", style="rounded,filled", fontsize=10];')
    lines.append('  edge [fontname="Helvetica", fontsize=8, arrowsize=0.7];')

    drawn_nodes = set()

    def draw_node(nid, label, fill, line, fontsize=10, penwidth=1.2):
        if nid in drawn_nodes:
            return
        drawn_nodes.add(nid)
        lines.append(
            f'  {node_id("n", nid)} [label="{label}", fillcolor="{fill}", '
            f'color="{line}", fontcolor="{line}", fontsize={fontsize}, penwidth={penwidth}];'
        )

    for m in matches_edges:
        tool_node = nodes.get(m["source"])
        kb_node = nodes.get(m["target"])
        if not tool_node or not kb_node:
            continue

        tool_props = tool_node["properties"]
        tier = tool_props.get("tier")
        pct = tool_props.get("importance_percentage")
        tcolor = TIER_COLOR.get(tier, DEFAULT_TIER_COLOR)
        tool_label = dot_escape(tool_props["name"])
        if tier and pct is not None:
            tool_label += f"\\n{dot_escape(tier)} · {pct}%"
        draw_node(m["source"], tool_label, tcolor["fill"], tcolor["line"], fontsize=13, penwidth=1.8)

        kb_props = kb_node["properties"]
        kb_label = f"{dot_escape(kb_props['name'])}\\n[{dot_escape(kb_props.get('entity_type', ''))}]"
        draw_node(m["target"], kb_label, KB_ENTITY_FILL, KB_ENTITY_LINE, fontsize=12, penwidth=1.8)
        lines.append(
            f'  {node_id("n", m["source"])} -> {node_id("n", m["target"])} '
            f'[label="matches", color="#8996a1", style="dotted"];'
        )

        role_edges = sorted(
            (r for r in rels if r["type"] == "RELEVANT_TO_ROLE" and r["source"] == m["target"]),
            key=lambda r: -(r.get("properties", {}).get("importance") or 0),
        )[:max_roles_per_entity]
        for r in role_edges:
            role_node = nodes[r["target"]]
            imp = r.get("properties", {}).get("importance")
            label = dot_escape(role_node["properties"]["name"]) + (f"\\n{imp}" if imp is not None else "")
            draw_node(r["target"], label, ROLE_NODE_FILL, "#2c4a6e")
            style = KB_EDGE_STYLE["RELEVANT_TO_ROLE"]
            lines.append(
                f'  {node_id("n", m["target"])} -> {node_id("n", r["target"])} '
                f'[color="{style["color"]}", style="{style["style"]}"];'
            )

        req_edges = [r for r in rels if r["type"] == "REQUIRES" and r["source"] == m["target"]]
        for r in req_edges:
            req_node = nodes[r["target"]]
            label = dot_escape(req_node["properties"]["name"])
            draw_node(r["target"], label, PREREQ_FILL, "#a8471f")
            style = KB_EDGE_STYLE["REQUIRES"]
            lines.append(
                f'  {node_id("n", r["target"])} -> {node_id("n", m["target"])} '
                f'[label="required for", color="{style["color"]}", style="{style["style"]}"];'
            )

        rel_edges = sorted(
            (r for r in rels if r["type"] == "RELATED_BY_CONCEPT" and r["source"] == m["target"]),
            key=lambda r: -(r.get("properties", {}).get("overlap_count") or 0),
        )[:max_related_per_entity]
        for r in rel_edges:
            rel_node = nodes[r["target"]]
            n_shared = r.get("properties", {}).get("overlap_count")
            label = dot_escape(rel_node["properties"]["name"])
            draw_node(r["target"], label, RELATED_FILL, "#6b4a8a")
            style = KB_EDGE_STYLE["RELATED_BY_CONCEPT"]
            lines.append(
                f'  {node_id("n", m["target"])} -> {node_id("n", r["target"])} '
                f'[label="{n_shared} shared concept(s)", color="{style["color"]}", style="{style["style"]}"];'
            )

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 3. Per-role concept/keyword trees
# --------------------------------------------------------------------------

def best_subid_for_concept(concept_name, keywords_by_subid):
    concept_l = concept_name.lower()
    best_subid, best_score = None, 0
    for subid, kws in keywords_by_subid.items():
        score = 0
        for kw in kws:
            kwl = kw.lower()
            if len(kwl) <= 2:
                if re.search(rf"\b{re.escape(kwl)}\b", concept_l):
                    score += 1
            elif kwl in concept_l or concept_l in kwl:
                score += 1
        if score > best_score:
            best_score = score
            best_subid = subid
    return best_subid


def penwidth(pct):
    if pct is None:
        return 1.0
    if pct >= 80:
        return 2.4
    if pct >= 50:
        return 1.6
    return 1.0


def build_role_dot(role, role_idx, keywords_by_subid):
    role_name = role["role"]
    style = ROLE_COLORS[role_idx % len(ROLE_COLORS)]
    lines = ["digraph RoleTree {"]
    lines.append("  rankdir=LR;")
    lines.append('  bgcolor="white";')
    lines.append("  pad=0.35; nodesep=0.28; ranksep=0.9;")
    lines.append('  fontname="Helvetica";')
    lines.append(
        f'  label="{dot_escape(role_name)} — dependency tree from combined_classification.json\\n'
        f'(role_importance_pct: {role.get("role_importance_pct")}%)";'
    )
    lines.append('  labelloc="t"; fontsize=15; fontcolor="#121a22";')
    lines.append('  node [fontname="Helvetica", style="rounded,filled"];')
    lines.append('  edge [color="#aab6bf", arrowsize=0.7];')

    root_id = node_id("root", role_name)
    lines.append(
        f'  {root_id} [label="{dot_escape(role_name)}\\n{role.get("role_importance_pct")}% of JD", '
        f'shape=box, fillcolor="{style["line"]}", fontcolor="white", '
        f'fontsize=15, penwidth=2, margin="0.25,0.16"];'
    )

    for concept in role.get("concepts", []):
        c_name = concept["concept"]
        c_pct = concept.get("importance_pct")
        c_id = node_id("concept", c_name)
        label = dot_escape(c_name) + (f"\\n{c_pct}%" if c_pct is not None else "")
        lines.append(
            f'  {c_id} [label="{label}", shape=box, '
            f'fillcolor="{style["fill"]}", color="{style["line"]}", fontcolor="{style["line"]}", '
            f'fontsize=12, penwidth={penwidth(c_pct)}, margin="0.18,0.1"];'
        )
        lines.append(f"  {root_id} -> {c_id};")

        sub_id = best_subid_for_concept(c_name, keywords_by_subid)
        for kw in keywords_by_subid.get(sub_id, []):
            k_id = node_id("kw_" + c_id, kw)
            lines.append(
                f'  {k_id} [label="{dot_escape(kw)}", shape=box, fillcolor="{LEAF_FILL}", '
                f'color="{LEAF_LINE}", fontcolor="{LEAF_TEXT}", fontsize=10, '
                f'penwidth=1, margin="0.14,0.06"];'
            )
            lines.append(f"  {c_id} -> {k_id};")

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    with open(CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
        classification = json.load(f)

    result = jd.build_jd_graph_from_classification(classification)

    generated = []

    related_dot = build_related_to_dot(classification, result)
    related_png = os.path.join(HERE, "dependency_tree.png")
    render(related_dot, related_png)
    generated.append(os.path.basename(related_png))
    print(f"wrote {related_png}")

    unified_dot = build_unified_graph_dot(classification, result)
    unified_png = os.path.join(HERE, "dependency_graph_unified.png")
    render(unified_dot, unified_png)
    generated.append(os.path.basename(unified_png))
    print(f"wrote {unified_png}")

    if _KB_AVAILABLE:
        try:
            kb = EntityKB()  # loads the 19 shard files from next to this script
            enriched = jd_kb_bridge.build_enriched_graph_from_classification(classification, kb)
            kb_dot = build_kb_enriched_dot(enriched)
            if kb_dot:
                kb_png = os.path.join(HERE, "dependency_tree_kb_enriched.png")
                render(kb_dot, kb_png)
                generated.append(os.path.basename(kb_png))
                print(f"wrote {kb_png}")
            else:
                print("(no tools in this JD matched the 221-entity KB dataset — skipped KB-enriched PNG)")
        except FileNotFoundError as e:
            print(f"(KB-enriched PNG skipped — {e})")
    else:
        print("(entity_kb.py / jd_kb_bridge.py not found next to this script — skipped KB-enriched PNG)")

    keywords_by_subid = {}
    for sentence in classification.get("processed_sentences", []):
        for sub in sentence.get("sub_requirements") or []:
            keywords_by_subid[sub["sub_id"]] = sub.get("match_keywords", [])

    roles_ranked = classification.get("testlify_analysis", {}).get("roles_ranked", [])
    for idx, role in enumerate(roles_ranked):
        role_dot = build_role_dot(role, idx, keywords_by_subid)
        role_png = os.path.join(HERE, f"dependency_tree_{slug(role['role'])}.png")
        render(role_dot, role_png)
        generated.append(os.path.basename(role_png))
        print(f"wrote {role_png}")

    # Clean up PNGs generated by a previous run for roles/sentences that no
    # longer exist (e.g. the JD's roles changed, or re-classification
    # produced different sentence ids) so stale trees don't linger.
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            previous = json.load(f).get("generated", [])
        for stale in set(previous) - set(generated):
            stale_path = os.path.join(HERE, stale)
            if os.path.exists(stale_path):
                os.remove(stale_path)
                print(f"removed stale {stale_path}")

    with open(MANIFEST_PATH, "w") as f:
        json.dump({"generated": generated}, f, indent=2)


if __name__ == "__main__":
    main()