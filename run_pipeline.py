"""
run_pipeline.py — single entry point: raw job description in, fully
KB-enriched graph (+ optional PNGs) out.

Chains together everything discussed:

    raw JD text
        -> classifier.py       (GPT call -> combined_classification.json)
        -> jd.py                (combined_classification.json -> tool graph,
                                  via merged_tools.json)
        -> jd_kb_bridge.py       (tool graph -> KB-enriched graph, via
                                  entity_kb.py's 221-entity dataset)
        -> generate_dependency_trees.py  (enriched graph -> PNGs, optional)

Nothing in classifier.py, jd.py, entity_kb.py, or jd_kb_bridge.py needed to
change for this to work — this file just calls each of them in order and
passes the right file paths around. That's deliberate: each of those stays
independently usable/testable, and this is the only file that knows about
all of them at once.

FILE LAYOUT THIS EXPECTS
--------------------------------------------------------------
    your_project/
      classifier.py
      jd.py
      entity_kb.py
      jd_kb_bridge.py
      generate_dependency_trees.py   (optional, only needed for PNGs)
      merged_tools.json               (jd.py's ~6.4K-tool taxonomy)
      02_frameworks_knowledge_graph.json   \
      03_tools_knowledge_graph.json         \
      ...                                    >  entity_kb.py's 19 shard files
      index.json                            /
      relationships.json                   /
      run_pipeline.py                 (this file)

Everything defaults to "look next to this script" — pass explicit paths
if your layout differs.

USAGE
--------------------------------------------------------------
CLI:
    python3 run_pipeline.py --jd-file my_job_description.txt
    python3 run_pipeline.py "React and Django developer, PostgreSQL, Docker"
    python3 run_pipeline.py --classification combined_classification.json   # skip the GPT call, reuse an existing one

Programmatic:
    from run_pipeline import run_from_jd_text, run_from_classification

    result = run_from_jd_text(open("jd.txt").read(), openai_api_key="sk-...")
    # or, if you already have a combined_classification.json:
    result = run_from_classification(json.load(open("combined_classification.json")))
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _default(path_arg, filename):
    return path_arg or os.path.join(HERE, filename)


def run_from_classification(
    classification: dict,
    kb_dir: str = None,
    tools_path: str = None,
    out_dir: str = None,
    make_pngs: bool = True,
) -> dict:
    """Stage 2 onward: you already have a combined_classification.json
    (from classifier.py, or from a previous run) — build the KB-enriched
    graph from it, optionally render PNGs, and write both to disk."""
    import jd as jd_module
    from entity_kb import EntityKB
    import jd_kb_bridge

    out_dir = out_dir or HERE
    os.makedirs(out_dir, exist_ok=True)

    kb = EntityKB(data_dir=kb_dir) if kb_dir else EntityKB()
    print(f"[run_pipeline] KB loaded: {kb.stats()['entity_count']} entities")

    result = jd_kb_bridge.build_enriched_graph_from_classification(classification, kb)

    enriched_path = os.path.join(out_dir, "enriched_output.json")
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[run_pipeline] wrote {enriched_path}")

    if result["tree"]:
        print()
        print("Ranked roles:", ", ".join(r["name"] for r in result["tree"]))
    cov = result.get("kb_coverage", {})
    print(f"\n[run_pipeline] KB matches: {', '.join(cov.get('jd_matched_entities', [])) or 'none'}")
    if cov.get("unmatched_terms"):
        print(f"[run_pipeline] no KB match: {', '.join(cov['unmatched_terms'])}")

    if make_pngs:
        _render_pngs(classification, out_dir, kb_dir, tools_path)

    return result


def run_from_jd_text(
    jd_text: str,
    openai_api_key: str = None,
    model: str = "gpt-5.2",
    kb_dir: str = None,
    tools_path: str = None,
    out_dir: str = None,
    classifier_module: str = "classifier",
) -> dict:
    """Full pipeline, starting from raw JD text: calls your GPT classifier
    (one API request), saves combined_classification.json exactly as its
    own main() does, then hands off to run_from_classification().

    classifier_module: the importable module name (filename minus ".py")
    of your GPT classifier script — e.g. "classifier" if the file is
    classifier.py, or "chatgptapi_new" if it's chatgptapi_new.py. It must
    expose a JobDescriptionClassifier class with the same interface used
    below (api_key=, model=, .classify(text), .get_role_names_only(result)).
    """
    import importlib
    try:
        clf_mod = importlib.import_module(classifier_module)
    except ImportError as e:
        raise ImportError(
            f"Could not import classifier module '{classifier_module}' "
            f"(looked for '{classifier_module}.py' on sys.path, which includes "
            f"{HERE}). If your GPT classifier script has a different filename, "
            f"pass classifier_module=\"<filename-without-.py>\" to run_from_jd_text() "
            f"or --classifier-module on the CLI."
        ) from e
    JobDescriptionClassifier = clf_mod.JobDescriptionClassifier

    out_dir = out_dir or HERE
    os.makedirs(out_dir, exist_ok=True)

    print(f"[run_pipeline] classifying JD via GPT (using {classifier_module}.py)...")
    classifier = JobDescriptionClassifier(api_key=openai_api_key, model=model)
    result = classifier.classify(jd_text)
    job_roles = classifier.get_role_names_only(result)

    # Same shape classifier.py's own main() writes to combined_classification.json —
    # jd.py's build_jd_graph_from_classification() expects exactly this.
    combined = {
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
        "job_role_domain": result.get("job_role_domain", None),
        "role_similarity_matrix": result.get("role_similarity_matrix", None),
        "summary": result.get("summary", {}),
        "testlify_analysis": result.get("testlify_analysis", None),
        "sentence_importance_analysis": result.get("sentence_importance_analysis", None),
        "job_logistics": result.get("job_logistics", None),
    }
    combined_path = os.path.join(out_dir, "combined_classification.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"[run_pipeline] wrote {combined_path}")

    return run_from_classification(combined, kb_dir=kb_dir, tools_path=tools_path, out_dir=out_dir)


def _render_pngs(classification: dict, out_dir: str, kb_dir: str, tools_path: str) -> None:
    """Best-effort PNG rendering via generate_dependency_trees.py's own
    functions — skipped (with a printed reason) if `dot` or the script
    itself isn't available, since PNGs are a bonus on top of the JSON,
    not something the rest of the pipeline depends on."""
    try:
        import generate_dependency_trees as gdt
        import jd as jd_module
    except ImportError as e:
        print(f"[run_pipeline] skipped PNGs — {e}")
        return

    taxonomy = jd_module.ToolTaxonomy(path=tools_path) if tools_path else None
    result = jd_module.build_jd_graph_from_classification(classification, taxonomy)
    try:
        gdt.render(gdt.build_related_to_dot(classification, result), os.path.join(out_dir, "dependency_tree.png"))
        gdt.render(gdt.build_unified_graph_dot(classification, result), os.path.join(out_dir, "dependency_graph_unified.png"))
        print("[run_pipeline] wrote dependency_tree.png, dependency_graph_unified.png")
    except Exception as e:
        print(f"[run_pipeline] skipped standard PNGs — {e}")
        return

    if gdt._KB_AVAILABLE:
        try:
            from entity_kb import EntityKB
            import jd_kb_bridge
            kb = EntityKB(data_dir=kb_dir) if kb_dir else EntityKB()
            enriched = jd_kb_bridge.build_enriched_graph_from_classification(classification, kb)
            kb_dot = gdt.build_kb_enriched_dot(enriched)
            if kb_dot:
                gdt.render(kb_dot, os.path.join(out_dir, "dependency_tree_kb_enriched.png"))
                print("[run_pipeline] wrote dependency_tree_kb_enriched.png")
        except Exception as e:
            print(f"[run_pipeline] skipped KB-enriched PNG — {e}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full JD pipeline: classify -> tool graph -> KB-enriched graph -> PNGs.")
    parser.add_argument("text", nargs="?", help="Raw JD text")
    parser.add_argument("--jd-file", help="Read raw JD text from a file")
    parser.add_argument("--classification", "-c", help="Skip the GPT call — use an existing combined_classification.json")
    parser.add_argument("--openai-api-key", help="Defaults to OPENAI_API_KEY env var (only needed without --classification)")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument(
        "--classifier-module", default="classifier",
        help="Filename (without .py) of your GPT classifier script, if not classifier.py "
             "— e.g. --classifier-module chatgptapi_new",
    )
    parser.add_argument("--kb-dir", help="Directory with the 19 KB shard files (default: next to this script)")
    parser.add_argument("--tools", help="Path to merged_tools.json (default: next to this script)")
    parser.add_argument("--out-dir", help="Where to write output files (default: next to this script)")
    parser.add_argument("--no-pngs", action="store_true", help="Skip PNG rendering, write JSON only")
    args = parser.parse_args()

    if args.classification:
        with open(args.classification, "r", encoding="utf-8") as f:
            classification = json.load(f)
        run_from_classification(
            classification, kb_dir=args.kb_dir, tools_path=args.tools,
            out_dir=args.out_dir, make_pngs=not args.no_pngs,
        )
        return

    if args.jd_file:
        with open(args.jd_file, "r", encoding="utf-8") as f:
            jd_text = f.read()
    elif args.text:
        jd_text = args.text
    elif not sys.stdin.isatty():
        jd_text = sys.stdin.read()
    else:
        parser.print_help()
        return

    run_from_jd_text(
        jd_text, openai_api_key=args.openai_api_key, model=args.model,
        kb_dir=args.kb_dir, tools_path=args.tools, out_dir=args.out_dir,
        classifier_module=args.classifier_module,
    )


if __name__ == "__main__":
    main()