#!/usr/bin/env python3
"""
run_all.py — single entry point for the whole jd_resume_matching pipeline.

    raw JD text
        │
        ▼
    chatgptapi_new.py   (OpenAI classify)      → classification_result.json,
        │                                        combined_classification.json
        ▼
    jd.py                (build tool graph      → jd.json
        │                 from merged_tools.json
        │                 taxonomy)
        ▼
    jd_resume_match.py   (score resume.json      → report.json
        │                 against jd.json —
        │                 INCLUDES the verb-
        │                 priority scoring via
        │                 verbs_priority_engg.json)
        ▼
    generate_dependency_trees.py   (optional — PNGs from combined_classification.json)

Usage
-------------------------------------------------------------------
    python run_all.py --jd job_description.txt --resume resume.json

    # skip PNG generation (needs graphviz's `dot` installed)
    python run_all.py --jd job_description.txt --resume resume.json --no-trees

    # use text inline instead of a file
    python run_all.py --text "We are hiring a Backend Engineer..." --resume resume.json

Requires (same folder as this script):
    chatgptapi_new.py, jd.py, jd_resume_match.py, generate_dependency_trees.py (optional)
    merged_tools.json      — jd.py's tool taxonomy (~6.4K tools)
    know_graph.json        — jd_resume_match.py's related-tools graph (optional)
    verbs_priority_engg.json — the verb-strength table (THIS is the "verb" part)
    .env with OPENAI_API_KEY

NOTE ON THE jd.py -> jd_resume_match.py HANDOFF
-------------------------------------------------------------------
jd.py's own output is a {"tree": [...], "graph": {"nodes": [...], ...}}
capability graph. jd_resume_match.py's extract_jd_tools() instead expects
a flat {"tools": [{"name", "importance_percentage", "tier", "aliases"}]}
shape. I haven't seen jd_indexer.py / prepare_jd_entities.py (the scripts
in your repo that likely do this conversion already), so `_flatten_to_jd_tools()`
below is MY OWN reconstruction of that conversion, built from reading both
files' actual code. If your existing jd_indexer.py / prepare_jd_entities.py
do something different (e.g. richer per-tool concepts/relations for the
"rich schema" scoring path in jd_resume_match.py), swap this function out
for a call to those instead — the rest of this script doesn't care how
jd.json gets built, only that it ends up in that shape.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from chatgptapi_new import (          # noqa: E402
    JobDescriptionClassifier,
    ResultAnalyzer,
    build_sentence_role_matrix_locally,
)
import jd as jd_module                 # noqa: E402
import jd_resume_match as scorer       # noqa: E402


# --------------------------------------------------------------------------- #
# Step 1 — classify
# --------------------------------------------------------------------------- #
def build_combined(result: dict) -> dict:
    """Same shape chatgptapi_new.py's own main() writes to
    combined_classification.json."""
    job_roles = result.get("job_roles", [])
    sentence_matrix = build_sentence_role_matrix_locally(
        result.get("processed_sentences", []), job_roles
    )
    return {
        "chatgpt_roles": job_roles,
        "processed_sentences": result.get("processed_sentences", []),
        "structural_sentences": result.get("structural_sentences", []),
        "non_matchable_sentences": result.get("non_matchable_sentences", []),
        "tier_weights": result.get("tier_weights", {}),
        "extracted_skills": result.get("extracted_skills", []),
        "responsibility_actions": result.get("responsibility_actions", []),
        "experience_requirements": result.get("experience_requirements", {"min": 0, "max": 0}),
        "depth_qualifier_map": result.get("depth_qualifier_map", {}),
        "predicted_role": job_roles[0] if job_roles else None,
        "job_role_domain": result.get("job_role_domain"),
        "role_similarity_matrix": result.get("role_similarity_matrix"),
        "sentence_role_matrix": {
            "roles": sentence_matrix["roles"],
            "sentence_breakdown": sentence_matrix["sentence_breakdown"],
        },
        "summary": result.get("summary", {}),
        "testlify_analysis": result.get("testlify_analysis"),
        "sentence_importance_analysis": result.get("sentence_importance_analysis"),
        "job_logistics": result.get("job_logistics"),
    }


def step1_classify(jd_text: str, model: str, out_dir: str) -> dict:
    print("=" * 80)
    print("STEP 1 — Classifying job description (OpenAI API call)")
    print("=" * 80)
    classifier = JobDescriptionClassifier(model=model)
    result = classifier.classify(jd_text, temperature=0.3)
    ResultAnalyzer.print_summary(result)

    ResultAnalyzer.export_to_json(result, os.path.join(out_dir, "classification_result.json"))
    combined = build_combined(result)
    combined_path = os.path.join(out_dir, "combined_classification.json")
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)
    print(f"✓ {combined_path}")
    return combined


# --------------------------------------------------------------------------- #
# Step 2 — build jd.json (tool graph -> flat tool list jd_resume_match.py wants)
# --------------------------------------------------------------------------- #
def _flatten_to_jd_tools(graph_result: dict, taxonomy: "jd_module.ToolTaxonomy") -> dict:
    """See module docstring's NOTE. Converts jd.py's Tool graph nodes into
    the {"tools": [...]} shape extract_jd_tools() in jd_resume_match.py reads."""
    tools = []
    for node in graph_result["graph"]["nodes"]:
        if node["label"] != "Tool":
            continue
        props = node["properties"]
        name = props["name"]
        entry = taxonomy.entries.get(name.lower())
        aliases = list(entry.get("aliases") or []) if entry else []
        tools.append({
            "name": name,
            "importance_percentage": props.get("importance_percentage"),
            "tier": props.get("tier", "UNSCORED"),
            "aliases": aliases,
        })
    return {"tools": tools, "source_metadata": graph_result.get("source_metadata", {})}


def step2_build_jd_json(combined: dict, out_dir: str) -> dict:
    print("\n" + "=" * 80)
    print("STEP 2 — Building JD tool graph (jd.py, using merged_tools.json)")
    print("=" * 80)
    taxonomy = jd_module.ToolTaxonomy()  # loads merged_tools.json next to this script
    graph_result = jd_module.build_jd_graph_from_classification(combined, taxonomy)
    print(jd_module.render_tree(graph_result["tree"]))

    jd_flat = _flatten_to_jd_tools(graph_result, taxonomy)
    jd_json_path = os.path.join(out_dir, "jd.json")
    with open(jd_json_path, "w", encoding="utf-8") as fh:
        json.dump(jd_flat, fh, indent=2, ensure_ascii=False)
    print(f"✓ {jd_json_path}  ({len(jd_flat['tools'])} tools recognized)")
    return jd_flat


# --------------------------------------------------------------------------- #
# Step 3 — score resume against jd.json (INCLUDES verb-priority scoring)
# --------------------------------------------------------------------------- #
def step3_score(resume_path: str, jd_json: dict, out_dir: str, use_semantic: bool) -> dict:
    print("\n" + "=" * 80)
    print("STEP 3 — Scoring resume against JD (tool match + VERB-PRIORITY strength)")
    print("=" * 80)
    resume_json = scorer.load_json(resume_path)

    # build_report() is the flat-schema path: it calls tool_is_matched() for
    # each JD tool, then strength_for() -> action_strengths (built by
    # build_action_strength_map(), which reads verbs_priority_engg.json via
    # load_verb_priorities()) to weight each match by how strong the verb
    # attached to it was ("architected X" counts more than "assisted with X").
    # This IS the verb-priority ("verb wala") part you asked to include.
    report = scorer.build_report(resume_json, jd_json, use_semantic=use_semantic)

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\nOverall weighted score : {report['overall_weighted_score_pct']}%")
    print(f"CRITICAL tier score    : {report['critical_tier_score_pct']}%")
    print(f"IMPORTANT tier score   : {report['important_tier_score_pct']}%")
    print(f"Raw tool coverage      : {report['raw_coverage_pct']}%")
    print(f"Recommendation         : {report['recommendation']}")
    print(f"✓ {report_path}")
    return report


# --------------------------------------------------------------------------- #
# Step 4 (optional) — dependency tree PNGs
# --------------------------------------------------------------------------- #
def step4_trees(out_dir: str) -> None:
    print("\n" + "=" * 80)
    print("STEP 4 — Generating dependency-tree PNGs (optional)")
    print("=" * 80)
    script = os.path.join(HERE, "generate_dependency_trees.py")
    if not os.path.isfile(script):
        print("(generate_dependency_trees.py not found — skipping)")
        return
    proc = subprocess.run([sys.executable, script], cwd=out_dir, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("⚠️  PNG generation failed (is graphviz's `dot` installed?):")
        print(proc.stderr)
    else:
        print("✓ PNGs written")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full JD -> resume scoring pipeline in one shot.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jd", help="Path to a text file with the raw job description.")
    src.add_argument("--text", help="Raw job description text, inline.")
    ap.add_argument("--resume", required=True, help="Path to resume.json (nodes/edges graph schema).")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--semantic", action="store_true",
                     help="Enable embedding-based fallback matching (needs sentence-transformers).")
    ap.add_argument("--no-trees", action="store_true", help="Skip PNG generation (step 4).")
    args = ap.parse_args()

    jd_text = args.text if args.text else open(args.jd, "r", encoding="utf-8").read()

    combined = step1_classify(jd_text, args.model, args.out_dir)
    jd_json = step2_build_jd_json(combined, args.out_dir)
    step3_score(args.resume, jd_json, args.out_dir, use_semantic=args.semantic)
    if not args.no_trees:
        step4_trees(args.out_dir)

    print("\n" + "=" * 80)
    print("DONE — see classification_result.json, combined_classification.json,")
    print("       jd.json, and report.json in", args.out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()