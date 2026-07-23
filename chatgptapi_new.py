"""
Job Description Classifier using OpenAI GPT API

This script processes job descriptions to:
1. Extract technically relevant content (processed_job_description)
2. Infer job roles dynamically with coverage percentages (job_roles)


ALL LOGIC IS DYNAMIC - No predefined role lists or keyword sets.
Everything is inferred by GPT based on the job description content.
"""

import json
import os
import re
import sys
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from jd_indexer import load_dataset, build_name_index, resolve_name, load_relationships, tools_related_to_concept, build_detail
from verb_priority_graph import load_verb_lookup, match_verb

# Load environment variables from .env file
load_dotenv()

# Ensure UTF-8 encoding for console output



# =============================================================================
# HARDCODED CONSTANTS - Previously wasted GPT tokens echoing these back
# =============================================================================

TIER_WEIGHTS = {"CRITICAL": 1.0, "IMPORTANT": 0.7, "GENERIC": 0.3, "NON_MATCHABLE": 0.0}

DEPTH_QUALIFIER_MAP = {
    "SURFACE": ["familiar", "exposure", "awareness", "help with", "assist", "participate", "work with"],
    "INTERMEDIATE": ["experience", "good", "knowledge", "ability", "responsible for", "create", "develop", "build"],
    "DEEP": ["strong", "proficiency", "expert", "mastery", "drive", "lead", "architect", "mentor"]
}

# =============================================================================
# SOFT SKILL CATEGORIES
# =============================================================================

SOFT_SKILL_CATEGORIES = [
    "communication",
    "collaboration_teamwork",
    "leadership",
    "adaptability",
    "problem_solving",
    "time_management",
    "emotional_intelligence",
]

# Map GPT soft_skill_type values → canonical categories
SOFT_SKILL_TYPE_MAP: dict = {
    "communication": "communication",
    "problem_solving": "problem_solving",
    "collaboration": "collaboration_teamwork",
    "collaboration_teamwork": "collaboration_teamwork",
    "leadership": "leadership",
    "adaptability": "adaptability",
    "time_management": "time_management",
    "emotional_intelligence": "emotional_intelligence",
    "culture_fit": "adaptability",   # closest canonical match
}

# =============================================================================
# ENTITY / PARENT-RELATION VOCABULARIES
# =============================================================================
# Reference vocabularies for the GPT-inferred "entities" / "parent_relations"
# fields on each processed_sentence (see SYSTEM_PROMPT). These are NOT
# enforced in code — GPT is instructed to pick from these lists in the
# prompt — they're kept here purely as documentation / for anything
# downstream that wants to validate or color-code by category.

ENTITY_CATEGORIES = ["tool", "concept"]

PARENT_RELATION_TYPES = [
    "BUILT_ON", "USES_LANGUAGE", "USES_PROTOCOL", "USES_TECHNOLOGY", "REQUIRES",
    "RUNS_ON", "IMPLEMENTS", "EXTENDS", "PART_OF", "ALTERNATIVE_TO", "RELATED_TO",
]


def compute_soft_skills_locally(
    processed_sentences: list,
    non_matchable_sentences: list,
    structural_sentences: list,
) -> dict:
    """
    Aggregate soft-skill sentences into 7 canonical categories.
    Returns a dict keyed by category containing:
      sentences   – list of matching sentence texts
      count       – number of matching sentences
      percentage  – share of total soft-skill sentences (rounded to 1 dp, sums to 100)
    Only categories with at least one sentence are included.
    """
    buckets: dict = {cat: [] for cat in SOFT_SKILL_CATEGORIES}

    for s in non_matchable_sentences:
        if s.get("type") == "soft_skill":
            mapped = SOFT_SKILL_TYPE_MAP.get(s.get("soft_skill_type") or "")
            if mapped:
                buckets[mapped].append(s.get("text", "").strip())

    for s in processed_sentences:
        if s.get("is_soft_skill"):
            mapped = SOFT_SKILL_TYPE_MAP.get(s.get("soft_skill_type") or "")
            if mapped:
                buckets[mapped].append(s.get("text", "").strip())

    total_soft = sum(len(sents) for sents in buckets.values()) or 1

    return {
        cat: {
            "sentences": sents,
            "count": len(sents),
            "percentage": round(len(sents) / total_soft * 100, 1),
        }
        for cat, sents in buckets.items()
        if sents
    }


def compute_summary_locally(processed_sentences: list, structural_sentences: list, non_matchable_sentences: list) -> dict:
    """
    Compute summary statistics in Python instead of wasting GPT output tokens.
    Replicates the exact summary structure that GPT was previously asked to generate.
    """
    tier_dist = {}
    depth_dist = {}
    compound_types = {}
    matchable_simple = 0
    matchable_compound = 0
    total_sub = 0
    soft_skill_count = 0
    non_matchable_count = 0

    for s in processed_sentences:
        # Tier distribution
        tier = s.get("tier", "GENERIC")
        tier_dist[tier] = tier_dist.get(tier, 0) + 1

        # Depth distribution
        depth = s.get("expected_depth") if s.get("expected_depth") is not None else "null"
        depth_dist[depth] = depth_dist.get(depth, 0) + 1

        # Compound vs simple
        if s.get("is_compound"):
            matchable_compound += 1
            ct = s.get("compound_type", "ALL_REQUIRED")
            compound_types[ct] = compound_types.get(ct, 0) + 1
            total_sub += len(s.get("sub_requirements") or [])
        elif s.get("matchable") is not False:
            matchable_simple += 1

        if s.get("is_soft_skill"):
            soft_skill_count += 1
        if s.get("matchable") is False:
            non_matchable_count += 1

    # Compute effective weight shares
    total_weight = 0.0
    weight_parts = {}
    for tier_name, weight in TIER_WEIGHTS.items():
        count = tier_dist.get(tier_name, 0)
        contribution = count * weight
        total_weight += contribution
        weight_parts[tier_name] = (count, weight, contribution)

    effective_weight = {}
    for tier_name in ["CRITICAL", "IMPORTANT", "GENERIC", "NON_MATCHABLE"]:
        if tier_name == "NON_MATCHABLE":
            effective_weight[f"{tier_name}_share"] = "excluded"
        else:
            count, weight, contribution = weight_parts.get(tier_name, (0, 0, 0))
            pct = (contribution / total_weight * 100) if total_weight > 0 else 0
            effective_weight[f"{tier_name}_share"] = f"{count} stmts × {weight} = {contribution:.1f} ({pct:.0f}%)"

    return {
        "total_sentences": len(processed_sentences) + len(structural_sentences),
        "structural": len(structural_sentences),
        "matchable_simple": matchable_simple,
        "matchable_compound": matchable_compound,
        "total_sub_requirements": total_sub,
        "soft_skill_sentences": soft_skill_count,
        "non_matchable": non_matchable_count + len(non_matchable_sentences),
        "compound_types": compound_types,
        "depth_distribution": depth_dist,
        "tier_distribution": tier_dist,
        "effective_weight": effective_weight
    }

def normalize_sentence_importance_locally(result: dict) -> None:
    """
    Rescales GPT's raw importance_percentage values so they sum to EXACTLY 100%,
    using largest-remainder rounding. Also rescales each sentence's nested
    sub_requirement_importance so it sums to that sentence's FINAL percentage.
    Mutates result in place.
    """
    sia = result.get("sentence_importance_analysis")
    if not sia or not sia.get("sentences"):
        return

    sentences = sia["sentences"]
    raw_total = sum(s.get("importance_percentage", 0) for s in sentences)
    if raw_total <= 0:
        return

    scaled = [(s.get("importance_percentage", 0) / raw_total) * 100 for s in sentences]
    floors = [int(x) for x in scaled]
    remainder = 100 - sum(floors)

    order = sorted(range(len(scaled)), key=lambda i: (scaled[i] - floors[i]), reverse=True)
    for i in range(remainder):
        floors[order[i]] += 1

    for s, final_pct in zip(sentences, floors):
        s["importance_percentage"] = f"{final_pct}%"
        if final_pct >= 8:
            s["importance_level"] = "HIGH"
        elif final_pct >= 3:
            s["importance_level"] = "MEDIUM"
        else:
            s["importance_level"] = "LOW"

        # Rescale nested sub-requirement breakdown to sum to this sentence's final_pct
        sub_items = s.get("sub_requirement_importance")
        if sub_items:
            sub_raw_total = sum(item.get("importance_percentage", 0) for item in sub_items)
            if sub_raw_total > 0:
                sub_scaled = [(item.get("importance_percentage", 0) / sub_raw_total) * final_pct for item in sub_items]
                sub_floors = [int(x) for x in sub_scaled]
                sub_remainder = final_pct - sum(sub_floors)
                sub_order = sorted(range(len(sub_scaled)), key=lambda i: (sub_scaled[i] - sub_floors[i]), reverse=True)
                for i in range(sub_remainder):
                    sub_floors[sub_order[i % len(sub_floors)]] += 1
                for item, sub_final in zip(sub_items, sub_floors):
                    item["importance_percentage"] = sub_final

    sia["total_percentage"] = f"{sum(floors)}%"


# =============================================================================
# ENTITY (TOOL/CONCEPT) IMPORTANCE SCORING
# =============================================================================
# Distributes each sentence's already-normalized importance_percentage across
# the tools/concepts it mentions, using updated_dataset (entity_type) as the
# source of truth for TechnologyRoleWeight instead of a hardcoded guess table
# or a dependency parser.

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "updated_dataset")

# entity_type (from updated_dataset) -> role weight. Core languages/frameworks/
# databases score highest, infra/platform/protocol are neutral, generic tools
# score slightly lower, and Concepts get a boost (transferable knowledge).
ENTITY_TYPE_ROLE_WEIGHT = {
    # core technology
    "Programming Language": 1.15,
    "Query Language": 1.1,
    "Scripting Language": 1.1,
    "Markup Language": 1.0,
    "Style Sheet Language": 1.0,
    "Framework": 1.1,
    "ML Framework": 1.15,
    "AI Framework": 1.15,
    "Testing Framework": 1.05,
    "Database": 1.1,
    "Database Extension": 0.95,
    # infrastructure / platform
    "Cloud Platform": 1.0,
    "Platform": 1.0,
    "Operating System": 0.95,
    "DevOps Tool": 0.95,
    "Container Technology": 1.0,
    "Message Broker": 0.95,
    "Infrastructure Component": 0.9,
    "Protocol": 0.9,
    "Protocol Suite": 0.9,
    "Standard": 0.85,
    "Specification": 0.85,
    "Hardware": 0.85,
    # utility / tool
    "Security Tool": 0.95,
    "Tool": 0.9,
    "Material": 0.8,
    "Material / Component": 0.8,
    "Material Category": 0.75,
    "Role": 0.8,
    # concepts — transferable knowledge (rule: concepts > implementation tools)
    "Concept": 1.2,
    "Methodology": 1.15,
}
DEFAULT_ROLE_WEIGHT = 1.0  # unresolved entities — neutral, never silently zeroed

VERBS_PRIORITY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verbs_priority_engg.json")

# A sentence's governing verb (from responsibility_actions, priority 1-10 via
# verbs_priority_engg.json) scales how much weight flows to that sentence's
# entities — "Designed" (high priority/ownership) pushes its tools/concepts
# higher than "Used" (low priority) would, for the SAME sentence importance.
# Range chosen so an unmatched/absent verb (multiplier 1.0) sits in the middle.
VERB_WEIGHT_MIN = 0.7   # priority 1
VERB_WEIGHT_MAX = 1.3   # priority 10

RELATIONSHIPS_PATH = os.path.join(DATASET_PATH, "relationships.json")

_dataset_cache: dict = {}
_verb_lookup_cache: dict = {}
_relationships_cache: dict = {}


def _get_dataset_index(dataset_path: str = DATASET_PATH):
    """Loads updated_dataset once per process and caches the name index."""
    if dataset_path not in _dataset_cache:
        try:
            entities_by_id = load_dataset(dataset_path)
            name_index = build_name_index(entities_by_id)
        except Exception as e:
            print(f"⚠️  Could not load dataset at {dataset_path} for entity role-weighting: {e}")
            entities_by_id, name_index = {}, {}
        _dataset_cache[dataset_path] = (entities_by_id, name_index)
    return _dataset_cache[dataset_path]


def _get_relationships(relationships_path: str = RELATIONSHIPS_PATH) -> dict:
    """Loads relationships.json once per process and caches it."""
    if relationships_path not in _relationships_cache:
        try:
            relationships = load_relationships(relationships_path)
        except Exception as e:
            print(f"⚠️  Could not load relationships at {relationships_path} for graph expansion: {e}")
            relationships = {}
        _relationships_cache[relationships_path] = relationships
    return _relationships_cache[relationships_path]


def _get_verb_lookup(verbs_path: str = VERBS_PRIORITY_PATH) -> dict:
    """Loads verbs_priority_engg.json once per process."""
    if verbs_path not in _verb_lookup_cache:
        try:
            lookup = load_verb_lookup(verbs_path)
        except Exception as e:
            print(f"⚠️  Could not load verb priorities at {verbs_path} for VerbWeight: {e}")
            lookup = {}
        _verb_lookup_cache[verbs_path] = lookup
    return _verb_lookup_cache[verbs_path]


def _sentence_governing_verb(sentence_id, actions_by_sentence: dict, verb_lookup: dict) -> Optional[dict]:
    """Picks the STRONGEST-ownership verb governing this sentence — the one
    with the highest priority among all responsibility_actions verbs tied to
    it (e.g. "Design and maintain" -> "Design" governs, per the ownership
    hierarchy Designed > Built > Implemented > ... > Used > Learned).
    Returns None if the sentence has no responsibility_actions or none of its
    verbs matched verbs_priority_engg.json — never guesses a priority.
    """
    best = None
    for act in actions_by_sentence.get(sentence_id, []):
        for raw_verb in act.get("actions", []):
            entry = match_verb(raw_verb, verb_lookup)
            if entry and (best is None or entry["priority"] > best["priority"]):
                best = {"verb": entry["verb"], "priority": entry["priority"], "category": entry["category"], "raw_verb": raw_verb}
    return best


def _verb_weight(governing_verb: Optional[dict]) -> float:
    if not governing_verb or governing_verb.get("priority") is None:
        return 1.0  # no matched governing verb — neutral, not a penalty
    priority = governing_verb["priority"]
    return round(VERB_WEIGHT_MIN + (priority / 10) * (VERB_WEIGHT_MAX - VERB_WEIGHT_MIN), 3)


def _hunt_dataset_detail(
    eid: Optional[int],
    entities_by_id: dict,
    relationships: dict,
) -> dict:
    """
    Full, uncapped dataset lookup for one resolved entity, via the id->record
    and id->relationships indexes (both O(1), see build_name_index /
    load_relationships) — no truncation, no synthetic scoring:
      - Tool    -> its COMPLETE dataset record: entity_type, entity_subtype,
                   hierarchy, aliases, every sub_concept, every prerequisite,
                   every relationships.json edge (jd_indexer.build_detail).
      - Concept -> its own complete record, PLUS every tool that references
                   this concept (jd_indexer.tools_related_to_concept — the
                   two indexed channels: relationship edges + reverse
                   sub-concept match), each of THOSE tools also expanded to
                   its own complete record.

    Returns {"resolved_in_dataset": bool, "dataset_detail": {...} | None,
             "related_tools": [...] | None}. related_tools is only populated
             for concepts; None for tools/unresolved entities.
    """
    if eid is None or eid not in entities_by_id:
        return {"resolved_in_dataset": False, "dataset_detail": None, "related_tools": None}

    detail = build_detail(eid, entities_by_id, relationships)
    related_tools = None

    if detail.get("entity_type") == "Concept":
        related_tools = []
        for t in tools_related_to_concept(eid, entities_by_id, relationships):
            tool_detail = build_detail(t["id"], entities_by_id, relationships)
            if tool_detail is not None:
                tool_detail["matched_via"] = t["sources"]  # provenance: relationship_edge and/or subconcept_match
                related_tools.append(tool_detail)

    return {"resolved_in_dataset": True, "dataset_detail": detail, "related_tools": related_tools}


def _largest_remainder_round(values: list, total: int = 100) -> list:
    """Scales `values` (any non-negative floats) to integers summing to
    exactly `total`, preserving relative order via largest-remainder rounding
    — same technique normalize_sentence_importance_locally uses for
    sentence percentages, applied here to the JD-wide entity pool."""
    raw_total = sum(values)
    if raw_total <= 0:
        return [0] * len(values)
    scaled = [(v / raw_total) * total for v in values]
    floors = [int(x) for x in scaled]
    remainder = total - sum(floors)
    order = sorted(range(len(scaled)), key=lambda i: (scaled[i] - floors[i]), reverse=True)
    for i in range(remainder):
        floors[order[i % len(floors)]] += 1
    return floors


_OR_PATTERN = re.compile(r"\bor\b", re.IGNORECASE)


def _sentence_relation(sentence: dict) -> str:
    """AND (divide weight) or OR (equal share, each usable as an alternative).
    Prefers the GPT-assigned compound_type; falls back to a plain 'or'/'/'
    connector scan of the sentence text when the sentence isn't compound.
    """
    compound_type = sentence.get("compound_type")
    if compound_type == "ANY_ONE":
        return "OR"
    if compound_type == "ALL_REQUIRED":
        return "AND"
    text = sentence.get("text", "")
    if _OR_PATTERN.search(text) or "/" in text:
        return "OR"
    return "AND"


def distribute_entity_importance_locally(
    result: dict,
    dataset_path: str = DATASET_PATH,
    verbs_path: str = VERBS_PRIORITY_PATH,
) -> None:
    """
    Full JD-wide scoring pass for tools/concepts. For each processed_sentence:
      1. Look up its governing verb (highest-priority verb from
         responsibility_actions tied to this sentence_id, via
         verbs_priority_engg.json) and turn it into a VerbWeight multiplier
         (rule: "Designed" outranks "Used" — see _verb_weight). Logged onto
         the sentence as "governing_verb" for transparency.
      2. effective_weight = TierWeight (CRITICAL/IMPORTANT/GENERIC, fixed
         lookup — not an LLM-assigned score) x VerbWeight.
      3. Split effective_weight across the sentence's entities:
           - AND (default): divided proportional to each entity's
             TechnologyRoleWeight (resolved against updated_dataset's
             entity_type — Concepts > core languages/frameworks/DBs >
             infra/platform > utility).
           - OR (compound_type == ANY_ONE, or an "or"/"/" connector): same
             proportional split, but tagged relation="OR" so a matching
             engine can treat these as alternatives (max, not sum).

    Then aggregates across the WHOLE JD: contributions per unique entity are
    summed and divided by sqrt(frequency) (diminishing returns), and finally
    ALL entities (tools + concepts together, one combined pool) are
    normalized via largest-remainder rounding so their percentages sum to
    EXACTLY 100% for the JD as a whole — not per sentence.

    Mutates result in place:
      - each processed_sentence gets "governing_verb" (or null)
      - each entity gets "importance_percentage" (raw per-sentence
        contribution, post-verb-weight), "relation", "role_weight"
      - new top-level "entity_importance_summary": ranked list with the
        JD-wide normalized "percentage" (ints, sums to 100), plus
        "raw_score" and full "contributions" trace for auditability.

    NOTE — node-centric scoring, not sentence-centric: the base weight per
    sentence is its TIER_WEIGHT (a fixed, deterministic lookup —
    CRITICAL=1.0/IMPORTANT=0.7/GENERIC=0.3 — NOT the GPT-assigned free-form
    sentence_importance_analysis percentage). Tools/concepts inherit weight
    from tier + governing verb + dataset role-type, never from an LLM's
    opinion of "how much of the JD is this sentence." sentence_importance_
    analysis is still computed (other consumers may want it) but is NOT
    read here.
    """
    entities_by_id, name_index = _get_dataset_index(dataset_path)
    verb_lookup = _get_verb_lookup(verbs_path)

    actions_by_sentence: dict = {}
    for act in result.get("responsibility_actions", []):
        actions_by_sentence.setdefault(act.get("sentence_id"), []).append(act)

    aggregate: dict = {}  # canonical lowercase name -> accumulator

    for sentence in result.get("processed_sentences", []):
        sid = sentence.get("id")

        governing_verb = _sentence_governing_verb(sid, actions_by_sentence, verb_lookup)
        verb_weight = _verb_weight(governing_verb)
        sentence["governing_verb"] = governing_verb

        entities = sentence.get("entities") or []
        if not entities:
            continue

        tier_weight = TIER_WEIGHTS.get(sentence.get("tier"), 0.0) * 100
        effective_pct = tier_weight * verb_weight
        relation = _sentence_relation(sentence)

        eids = []
        weights = []
        for ent in entities:
            eid = None
            if name_index:
                eid = resolve_name(ent["entity"], name_index, entities_by_id, category_hint=ent.get("category"))
            entity_type = entities_by_id[eid].get("entity_type") if eid is not None else None
            eids.append(eid)
            weights.append(ENTITY_TYPE_ROLE_WEIGHT.get(entity_type, DEFAULT_ROLE_WEIGHT))

        total_weight = sum(weights) or len(entities) or 1

        for ent, eid, w in zip(entities, eids, weights):
            share = round(effective_pct * (w / total_weight), 3) if effective_pct else 0.0
            ent["importance_percentage"] = share
            ent["relation"] = relation
            ent["role_weight"] = w

            key = ent["entity"].strip().lower()
            acc = aggregate.setdefault(key, {
                "entity": ent["entity"],
                "category": ent.get("category"),
                "eid": eid,
                "contributions": [],
            })
            if acc["eid"] is None and eid is not None:
                acc["eid"] = eid
            acc["contributions"].append({
                "sentence_id": sid,
                "importance_percentage": share,
                "relation": relation,
                "verb_weight": verb_weight,
            })

    raw_scores = []
    for acc in aggregate.values():
        contributions = acc["contributions"]
        raw_sum = sum(c["importance_percentage"] for c in contributions)
        frequency = len(contributions)
        raw_score = raw_sum / (frequency ** 0.5) if frequency else 0.0
        acc["raw_score"] = raw_score
        acc["frequency"] = frequency
        raw_scores.append(raw_score)

    normalized_pcts = _largest_remainder_round(raw_scores, total=100)
    relationships = _get_relationships()

    summary = []
    for acc, pct in zip(aggregate.values(), normalized_pcts):
        contributions = acc["contributions"]
        hunt = _hunt_dataset_detail(acc["eid"], entities_by_id, relationships)

        summary.append({
            "entity": acc["entity"],
            "category": acc["category"],
            "percentage": pct,
            "raw_score": round(acc["raw_score"], 3),
            "frequency": acc["frequency"],
            "sentences": [c["sentence_id"] for c in contributions],
            "contributions": contributions,
            "resolved_in_dataset": hunt["resolved_in_dataset"],
            "dataset_detail": hunt["dataset_detail"],
            "related_tools": hunt["related_tools"],
        })

    summary.sort(key=lambda x: x["percentage"], reverse=True)
    result["entity_importance_summary"] = summary


def ensure_entities_field_locally(result: dict) -> None:
    """
    Defensive normalization for the GPT-inferred "entities" field on each
    processed_sentence. Guarantees every processed_sentence has an
    "entities" list, and every entity — and every parent_relation target —
    has a well-formed "category" ("tool" or "concept") and a valid
    "relation_type". Even if GPT omitted a field, returned a malformed
    entry, or dropped it on a retry, this fills in a safe default so the
    graph-rendering code downstream (see generate_dependency_trees.py's
    build_unified_graph_dot) never has to guard against missing keys.
    Mutates result in place.
    """
    for s in result.get("processed_sentences", []):
        entities = s.get("entities")
        if not isinstance(entities, list):
            entities = []

        cleaned = []
        for ent in entities:
            if not isinstance(ent, dict) or not ent.get("entity"):
                continue

            category = ent.get("category")
            if category not in ENTITY_CATEGORIES:
                category = "concept"  # safe default for anything unclassifiable

            rels = ent.get("parent_relations")
            if not isinstance(rels, list):
                rels = []

            cleaned_rels = []
            for r in rels:
                if not isinstance(r, dict) or not r.get("target"):
                    continue
                rel_type = r.get("relation_type") or "RELATED_TO"
                if rel_type not in PARENT_RELATION_TYPES:
                    rel_type = "RELATED_TO"
                rel_category = r.get("category")
                if rel_category not in ENTITY_CATEGORIES:
                    rel_category = "concept"
                cleaned_rels.append({
                    "target": r["target"],
                    "relation_type": rel_type,
                    "category": rel_category,
                })

            cleaned.append({
                "entity": ent["entity"],
                "category": category,
                "parent_relations": cleaned_rels,
            })

        s["entities"] = cleaned


def build_sentence_role_matrix_locally(processed_sentences: list, job_roles: list) -> dict:
    """
    Build the sentence→role attribution matrix from processed_sentences data.
    Previously this required a separate GPT API call — now each processed_sentence
    already contains dominant_role and role_attribution fields from the main call.
    """
    sentence_breakdown = []
    matrix = []

    for i, s in enumerate(processed_sentences):
        attribution = s.get("role_attribution", {})
        dominant_role = s.get("dominant_role", "")
        dominant_pct = attribution.get(dominant_role, 0) if attribution else 0

        sentence_breakdown.append({
            "sentence_index": i + 1,
            "sentence": s.get("text", ""),
            "role_attribution": attribution,
            "dominant_role": dominant_role,
            "dominant_pct": dominant_pct
        })
        matrix.append([attribution.get(role, 0) for role in job_roles])

    return {
        "roles": job_roles,
        "sentence_breakdown": sentence_breakdown,
        "matrix": matrix
    }


# =============================================================================
# SYSTEM PROMPT - Core Classification Logic (All Dynamic)
# =============================================================================

SYSTEM_PROMPT = """



You are a classification engine. Your job is to read a raw job description and produce three precise JSON outputs:
"processed_job_description": a single string containing only the technically relevant sections from the original job description — keep the technical wording exactly as in the original, preserve bullets and order, and remove all non-technical sections (About the company, About us, application instructions, benefits, general marketing text, HR/admin details, etc.). Only include sections that are essential for role identification and technical fit: responsibilities, required technical skills, preferred technical skills, examples of work, measurable targets, stack & infra, data sources, performance/SLAs, and any technical constraints. Do not invent technical details or add examples.

"extracted_skills": an array of all specific skills, tools, technologies, frameworks, platforms, methodologies, and domain knowledge areas explicitly mentioned or strongly implied in the job description. Each entry should be a concise string (e.g., "Python", "Terraform", "CI/CD", "Azure", "Microservices", "IaC", "SOC 2 Compliance"). Rules for extraction:

Include programming languages, libraries, frameworks, cloud platforms, DevOps tools, databases, protocols, architectural patterns, certifications, and compliance standards.
Do not include soft skills (e.g., "communication", "problem-solving") or generic qualifiers (e.g., "experience", "proficiency").
Do not duplicate entries. Each skill should appear only once.
Keep each entry concise — prefer "Terraform" over "experience with Terraform".
If a skill is mentioned multiple times, include it only once.
Order: list hard technical skills first (languages, tools, platforms), then architectural/design patterns, then compliance/standards.

CRITICAL SKILL GROUPING RULES — FOLLOW THESE STEPS FOR EVERY SKILL YOU EXTRACT:

STEP 1: Find the exact sentence/phrase in the JD where the skill is mentioned.
STEP 2: Look at the connector word between the skills in that sentence.
STEP 3: Apply the rule for that connector word:

  IF the connector is "or" or "/" → ALL skills in that group become ONE single comma-separated string entry.
    - "Tableau or Power BI" → ONE entry: "Tableau, Power BI"
    - "Python or R" → ONE entry: "Python, R"
    - "AWS, Azure, or Google Cloud" → ONE entry: "AWS, Azure, Google Cloud"
    - "native-cloud platforms such as AWS, Azure, or Google Cloud" → ONE entry: "AWS, Azure, Google Cloud"

  IF the connector is "and" or "&" → EACH skill becomes its OWN separate entry.
    - "Scikit-learn and Pytorch" → TWO entries: "Scikit-learn", "Pytorch"
    - "Terraform, Pulumi, and CloudFormation" → THREE entries: "Terraform", "Pulumi", "CloudFormation"

  IF there is NO connector (comma-only list, "e.g." list, "/" is absent, standalone mention) → EACH skill becomes its OWN separate entry.
    - "e.g. SQL, NoSQL, Graph Databases" → THREE entries: "SQL", "NoSQL", "Graph Databases"
    - "e.g. Scikit-learn, Pytorch" → TWO entries: "Scikit-learn", "Pytorch"
    - "e.g. Vertex AI, LangChain, Hugging Face" → THREE entries: "Vertex AI", "LangChain", "Hugging Face"
    - "e.g. Spark, Hive, Presto" → THREE entries: "Spark", "Hive", "Presto"
    - "e.g. AWS Sagemaker, EC2, Tableau Server" → THREE entries: "AWS Sagemaker", "EC2", "Tableau Server"

STEP 4: Before finalizing, scan your extracted_skills array and for EACH grouped entry (containing a comma), go back to the JD and confirm the word "or" or "/" appears between those skills. If "or"/"/" is NOT there, split them into separate entries immediately.

ABSOLUTE RULES — no exceptions:
  ✅ "or" or "/" between skills → ONE grouped entry
  ✅ "and" or "&" between skills → SEPARATE entries
  ✅ Comma-only list or "e.g." list with no "or" → SEPARATE entries
  ✅ Skill mentioned alone → SEPARATE entry

  ❌ NEVER group skills that are connected by "and", "&", or comma-only
  ❌ NEVER be misled by "and" appearing in a section heading or surrounding text — only the connector word DIRECTLY BETWEEN the skills matters. Example: "Data Science and Model Development (e.g. Python, R)" — the "and" here is between "Data Science" and "Model Development", NOT between "Python" and "R". Python and R are a comma-only e.g. list → TWO separate entries: "Python", "Pytorch"
  ❌ NEVER split skills that are connected by "or" or "/"
  ❌ NEVER guess — always go back to the exact JD sentence to check the connector word

This rule applies to ALL skill types without exception — tech, non-tech, tools, languages, platforms, methodologies, compliance standards, soft skills, and everything else. The connector word alone determines grouping, never the skill type.

"responsibility_actions": an array of objects extracted from every responsibility line in the entire job description. For each responsibility line, extract the action verbs and map them to the thing they are acting on (the direct object/target).


Rules:
- "actions": the verb(s) that describe what the person does in that line (e.g., "Build", "maintain", "Manage", "optimise")
- "target": the specific thing those actions are performed on (e.g., "data pipelines", "GTM systems", "dashboards")
- If one line has multiple distinct action→target pairs, create a separate object for each pair
- Do not include tool names, qualifiers, or purpose clauses in "target" — keep it concise

Each object must include:
- "sentence_id": the "id" integer of the matching sentence from processed_sentences

Examples:
"Build and maintain data pipelines for segmentation, scoring and enrichment"
→ {"sentence_id": 4, "actions": ["Build", "maintain"], "target": "data pipelines"}

"Manage and optimise core GTM systems (Salesforce, Pardot, Demandbase)"
→ {"sentence_id": 7, "actions": ["Manage", "optimise"], "target": "GTM systems"}

"Create and manage dashboards and performance reporting"
→ {"sentence_id": 9, "actions": ["Create", "manage"], "target": "dashboards and performance reporting"}



"experience_requirements": an object extracted from the job description capturing the minimum and maximum years of experience required. Rules:
- Scan the entire job description for any mention of years of experience (e.g., "2+ years", "3 to 5 years", "at least 4 years", "minimum 2 years", "2-4 years").
- For each mention found, extract the numeric values.
- "min": the lowest minimum years mentioned across all experience requirements. Always a number. If a "X+" or "at least X" or "minimum X" pattern is found, min = X.
- "max": the highest maximum years mentioned. If a "X+" or "X or more" pattern is found with no upper bound, max = 0. If a range like "2 to 4" or "2-4" is found, max = 4.
- If multiple experience requirements exist (e.g., "2+ years Python", "3+ years SQL"), take the HIGHEST min value and the HIGHEST max value across all mentions.
- If no experience is mentioned at all, output: {"min": 0, "max": 0}.
- Output must always be a JSON object with exactly two numeric keys: "min" and "max".
- "source_sentence_id": the "id" value of the sentence in processed_sentences where the primary experience requirement was found. If multiple sentences mention experience, use the id of the sentence with the highest min value.


EXAMPLES:
- "2+ years of experience" → {"min": 2, "max": 0}
- "3 to 5 years of experience" → {"min": 3, "max": 5}
- "2-4 years" → {"min": 2, "max": 4}
- "at least 3 years" → {"min": 3, "max": 0}
- "minimum 2 years Python, 3+ years SQL" → {"min": 3, "max": 0}  ← highest min wins
- "2 to 4 years Python, 3 to 6 years SQL" → {"min": 3, "max": 6}  ← highest of each
- No mention → {"min": 0, "max": 0}

"job_roles": an array of role names in ranked order (most-likely first), representing only the roles that are a PRIMARY and DIRECT fit for this specific job description. There must be no pre-defined role list inside your prompt logic — infer roles solely from the processed job description.
Each role should be a simple string (just the role name).
Example: ["AI Engineer", "Machine Learning Engineer", "Software Engineer", "Data Scientist"]
Rules for role inference:

CRITICAL — PRECISION OVER COUNT: Extract only the roles that are a strong, direct match to the core function of this JD. You MUST always return a MINIMUM of 2 roles — a single-role output is NEVER acceptable. If the JD appears to map to only 1 role, re-examine it carefully and split by distinct functional dimensions (e.g. "Sales" vs "Account Management", or "Engineering" vs "Architecture"). Beyond the minimum of 2, return exactly as many as are genuinely distinct and non-overlapping — do NOT pad the list with loosely related roles, and do NOT artificially limit it if more distinct roles truly fit.

ANTI-OVERLAP RULE — MANDATORY STEP BEFORE FINALIZING:
Before outputting job_roles, compare every pair of roles and ask: "Do these two roles describe the same core function, just worded differently?" If yes, merge them into one — keep whichever title is more precise or more commonly used in the industry.
MINIMUM ROLES RULE: You must always return a minimum of 2 roles. If after all merging and collapsing you are left with only 1 role, re-examine the JD more carefully — split the single role by its distinct functional dimensions (e.g. "Sales" vs "Account Management", or "Business Development" vs "Sales Management") and represent each as a separate role. A single-role output is never acceptable.

Merging rules:
- If two roles differ only in wording but mean the same thing → merge into one
  e.g. "ML Engineer" + "Machine Learning Engineer" → one entry only
- If one role is a seniority variant of another → merge, drop the seniority
  e.g. "Senior Account Manager" + "Account Manager" → "Account Manager"
- If one role is a domain-specific child of a broader role already in the list → merge into the one that best fits the JD
  e.g. "Account Manager" + "Enterprise Account Manager" → pick the one that fits the JD better, not both
  e.g. "Solutions Consultant" + "Cloud Solutions Consultant" + "Security Solutions Consultant" → collapse into one
- If two roles have genuinely different core functions (different day-to-day work, different skills required) → keep both as separate entries

The final list must satisfy this test: for every role in the list, no other role in the list should already cover its core function. If that test fails for any pair, merge them before outputting.


Focus on BASE role titles WITHOUT seniority labels. Remove prefixes like "Senior", "Junior", "Mid-level", "Staff", "Lead", "Principal", "Chief", "Entry-level" from role names.
✅ CORRECT: "Software Engineer", "Data Analyst", "Product Manager"
❌ WRONG: "Senior Software Engineer", "Junior Data Analyst", "Lead Product Manager"
Avoid overly specific child roles or niche sub-specializations. Stick to broad, commonly-used role titles that represent the core function, not narrow specializations.
✅ CORRECT: "Cybersecurity Specialist", "Cybersecurity Analyst", "Security Engineer"
❌ WRONG: "Cybersecurity Policy Analyst", "Cybersecurity Compliance Specialist", "Threat Intelligence Analyst"
✅ CORRECT: "Data Analyst", "Business Intelligence Analyst", "Data Scientist"
❌ WRONG: "Healthcare Data Analyst", "Marketing Analytics Specialist", "Customer Insights Analyst"
✅ CORRECT: "Software Engineer", "Backend Developer", "Full Stack Developer"
❌ WRONG: "React Developer", "Node.js Engineer", "API Developer", "Microservices Architect"
Think broadly about different role types (e.g., Engineer vs Developer vs Analyst vs Specialist) but keep them at the same conceptual level.
If the JD describes a Data Engineer doing AI-specific tasks (fine-tuning, RAG, vector DBs, LLM integration), yield a role name that explicitly contains AI (e.g., "AI Data Engineer" or "Data Engineer (AI)"). In other words, do not return plain "Data Engineer" when the work is AI-focused — the role label must reflect that.
Examples of multi-role identification:
Example 1 - AI/ML Job:
✅ INCLUDE: "AI Engineer", "Machine Learning Engineer", "Software Engineer", "MLOps Engineer"
Note: "Data Scientist" would only be included if the JD explicitly mentions modeling/research work, not just engineering. "AI Architect" only if the JD mentions system design/architecture responsibilities.
❌ EXCLUDE: "Senior AI Engineer", "Lead ML Engineer", "Principal Data Scientist" (no seniority), "Prompt Engineer", "LLM Specialist" (too narrow)
Example 2 - Cybersecurity Job:
✅ INCLUDE: "Cybersecurity Specialist", "Cybersecurity Analyst", "Security Engineer", "Information Security Analyst", "Security Architect"
❌ EXCLUDE: "Senior Security Analyst" (no seniority), "Cybersecurity Policy Analyst", "Security Compliance Specialist", "Penetration Tester" (too narrow unless explicitly in JD)
Example 3 - Data Analysis Job:
✅ INCLUDE: "Data Analyst", "Business Intelligence Analyst", "Data Scientist", "Analytics Engineer", "BI Developer"
❌ EXCLUDE: "Senior Data Analyst" (no seniority), "Healthcare Data Analyst", "Marketing Analyst", "SQL Developer" (too specific)
Example 4 - Full Stack Development Job:
✅ INCLUDE: "Full Stack Developer", "Software Engineer", "Web Developer", "Backend Developer", "Frontend Developer"
❌ EXCLUDE: "Senior Full Stack Developer" (no seniority), "React Developer", "Node.js Developer", "API Developer" (too technology-specific)
Example 5 - QA/Testing Job:
✅ INCLUDE: "QA Engineer", "Test Engineer", "Quality Assurance Analyst", "Automation Engineer", "Software Tester"
❌ EXCLUDE: "Senior QA Engineer" (no seniority), "Mobile QA Specialist", "API Test Engineer", "Performance Test Engineer" (too narrow)
For a "Data Engineer" JD with ML pipelines, also identify: "ML Data Engineer", "Analytics Engineer", "Senior Data Engineer", "Data Platform Engineer", "AI Data Engineer"
For an "AI Research Engineer" JD, also identify: "Research Scientist", "Applied AI Scientist", "ML Research Engineer", "Senior Research Engineer"
What to AVOID (too specific or has seniority):
❌ Seniority labels - Remove these prefixes/suffixes:
Senior, Junior, Mid-level, Entry-level
Staff, Lead, Principal, Chief, Head of
I, II, III (level numbers)
Associate, Assistant
Example: "Senior Software Engineer" → "Software Engineer"
❌ Technology-specific roles (unless that technology is the main focus):
"React Developer" → "Frontend Developer"
"AWS Engineer" → "Cloud Engineer"
"PostgreSQL DBA" → "Database Administrator"
❌ Industry-specific roles (unless industry is core to the role):
"Healthcare Data Analyst" → "Data Analyst"
"Financial Software Engineer" → "Software Engineer"
❌ Process/methodology-specific roles:
"Agile Coach" (unless explicitly in JD)
"Scrum Master" (unless explicitly in JD)
❌ Overly narrow sub-specializations:
"Cybersecurity Policy Analyst" → "Cybersecurity Analyst"
"API Developer" → "Backend Developer"
"Test Automation Engineer" → "QA Engineer" or "Automation Engineer"
❌ Compound roles with 3+ qualifiers:
"Senior Cloud Security DevOps Engineer" → "Cloud Engineer" or "Security Engineer" or "DevOps Engineer"
"job_role_domain": a single string representing the primary domain of the job role. Choose exactly one from the following four options: "Tech", "Commerce", "Law", "Medical".
Rules for domain classification:
Tech: Software engineering, data science, AI/ML, cybersecurity, cloud, DevOps, IT, hardware, telecommunications, and any role primarily requiring technical/engineering skills.
Commerce: Business, finance, sales, marketing, accounting, e-commerce, supply chain, operations, HR, consulting, and any role primarily driven by business or trade functions.
Law: Legal counsel, compliance, contracts, regulatory affairs, paralegal, litigation, intellectual property, and any role primarily requiring legal expertise.
Medical: Healthcare, clinical roles, pharmacy, nursing, medical research, public health, biotechnology, and any role primarily requiring medical or clinical expertise.
When a role spans multiple domains (e.g., a Healthcare Data Analyst or a Legal Tech Engineer), assign the domain that represents the primary function of the role, not its industry context. A data engineer working in a hospital is still Tech; a compliance officer at a tech firm is still Law.


"role_similarity_matrix": an object containing pairwise similarity percentages between all extracted job roles. This must be computed based on your deep understanding of what each role actually does day-to-day — their responsibilities, required skills, and core functions.

Rules:
- Compare every unique pair of roles in job_roles
- For each pair, assign a similarity percentage (0–100) based on how much the two roles overlap in terms of daily work, required skills, and core responsibilities
- 100% = identical roles, 0% = completely unrelated roles
- Be precise — use your knowledge of industry role definitions, not just title similarity
- The diagonal (role compared to itself) is always 100%

Output format:
{
  "role_similarity_matrix": {
    "roles": ["Role A", "Role B", "Role C"],
    "matrix": [
      [100, 72, 45],
      [72, 100, 68],
      [45, 68, 100]
    ],
    "pairs": [
      {"role_1": "Role A", "role_2": "Role B", "similarity_pct": 72},
      {"role_1": "Role B", "role_2": "Role C", "similarity_pct": 68},
      {"role_1": "Role A", "role_2": "Role C", "similarity_pct": 45}
    ]
  }
}

Rules for the pairs array:
- Include only unique pairs (upper triangle — no duplicates, no self-pairs)
- Sort pairs by similarity_pct descending
- roles, matrix, and pairs must all be consistent with each other


"role_anchor": scan the raw job description for any sentence that explicitly states what role is being hired for. Look for phrases like:
- "We are seeking a [Role]"
- "We are looking for a [Role]"
- "We're hiring a [Role]"
- "This role is for a [Role]"
- "You will be joining us as a [Role]"
- "We need a [Role]"

If such a phrase is found, extract ONLY the role title (e.g., "Frontend Developer", "Data Scientist") — do not include the surrounding phrase.
If no such explicit hiring statement exists in the job description, return null.

OUTPUT FORMAT:
Return a single JSON object with these exact keys: "processed_job_description", "processed_sentences", "structural_sentences", "non_matchable_sentences", "extracted_skills", "responsibility_actions", "experience_requirements", "job_roles", "job_role_domain", "role_similarity_matrix","testlify_analysis", "sentence_importance_analysis", "job_logistics".
Do NOT include "role_anchor", "uncovered_sentences", "tier_weights", "depth_qualifier_map", or "summary" — these are computed locally.
"processed_sentences": an array of objects, one per sentence extracted from processed_job_description, in the same order. Exclude structural/header sentences (those go into structural_sentences instead).


Sentence splitting rules: split processed_job_description by newline and by punctuation that ends sentences (., :, ;), but preserve meaningful bullet lines as separate sentences. Exclude lines shorter than 5 words UNLESS they contain clear technical tokens.

For each sentence object, include these fields:
MANDATORY PRE-CHECK BEFORE ADDING ANY SENTENCE TO processed_sentences — apply ALL FOUR checks in order:
1. Is this line a section header, label, or heading (e.g. "Job Responsibilities", "Required Qualifications", "Who You Are", any short title-case line introducing a section)? → structural_sentences ONLY. Never give it an "id", never score it.
2. Is this sentence's ENTIRE content a soft skill, mindset, personality trait, or culture-fit statement with NO separately verifiable technical requirement (e.g. "Risk and control mindset...", "You are deeply curious", "comfortable with ambiguity")? → non_matchable_sentences (type="soft_skill"), NEVER processed_sentences.
3. Is this sentence's PRIMARY subject an education/degree requirement or a bare years-of-experience threshold? → non_matchable_sentences (type="education_requirement" or "years_of_experience_requirement"), NEVER processed_sentences, even if it mentions a technical domain in passing.
4. Is this sentence GENERIC boilerplate with no specific tool, technology, platform, or differentiating detail (e.g. "you are expected to be competent", "design and develop systems")? → non_matchable_sentences (type="generic_responsibility"), NEVER processed_sentences.
ONLY sentences that pass ALL FOUR checks — meaning they describe a SPECIFIC, differentiating, verifiable technical task, skill, or requirement (what the candidate must DO or KNOW) — belong in processed_sentences. When in doubt, ask: "Does this sentence name a specific tool, technology, technique, domain, or concrete deliverable?" If no, it does not belong in processed_sentences.
- "id": integer, zero-indexed, sequential (0, 1, 2, ...)
- "text": the exact sentence text
- "section": one of "requirements" or "responsibilities" — infer from context
- "is_compound": true if the sentence contains multiple distinct requirements or responsibilities that can be independently evaluated; false otherwise
- "compound_type": if is_compound is true, one of:
    - "ALL_REQUIRED" — all sub-parts must be satisfied (connected by "and", "&", comma lists)
    - "ANY_ONE" — at least one sub-part must be satisfied (connected by "or", "/", "at least one")
  If is_compound is false, set to null.
- "depth_qualifier": the specific word or phrase in the sentence that signals expected depth of experience. Examples: "experience", "good experience", "strong", "proficiency", "ability", "knowledge", "responsible for", "drive", "mentor", "familiar", "assist", "participate". If none is present, set to null.
- "tier": classify each sentence into exactly one of four tiers:
    - "CRITICAL" — specific, verifiable, differentiating requirement (e.g. named language, specific domain, tool). specificity_weight = 1.0
    - "IMPORTANT" — leadership/seniority signal or moderately specific requirement. specificity_weight = 0.7
    - "GENERIC" — boilerplate that almost every candidate in this domain would claim, with no specific technology, tool, or differentiating detail (e.g. "design and develop systems", "you are expected to be competent"). specificity_weight = 0.3. These sentences must NOT appear in processed_sentences — move them to non_matchable_sentences (type="generic_responsibility") instead, same as NON_MATCHABLE sentences.
    IMPORTANT TIERING RULE: When a sentence contains BOTH a years-of-experience 
    claim AND a specific technical requirement, the sentence-level tier must 
    reflect the technical requirement — NOT the years claim. The years claim is 
    handled at the sub_requirement level with matchable: false. 

Example: "3-6 years of experience working with scalable backends"
→ tier: "CRITICAL" (because scalable backends is a specific verifiable domain)
→ sub_requirement for years → matchable: false
→ sub_requirement for scalable backends → matchable: true
    - "NON_MATCHABLE" — pure soft skill, culture fit, education/degree requirement, bare years-of-experience threshold, or completely generic responsibility that cannot be verified from resume text. specificity_weight = 0.0. These sentences must NOT appear in processed_sentences — move them to non_matchable_sentences instead.

    MANDATORY EXCLUSION RULE — EDUCATION & YEARS OF EXPERIENCE:
    If a sentence's PRIMARY subject is an education/degree requirement (e.g. "Bachelors or Masters degree in...") OR a bare years-of-experience threshold (e.g. "Minimum 2 years of...", "4+ years of..."), that ENTIRE sentence must be tiered NON_MATCHABLE and routed to non_matchable_sentences — even if it also mentions a technical domain in passing (e.g. "computer programming", "full stack application development"). These sentences are captured exclusively in job_logistics (years_of_experience / education_qualification fields) and must NEVER appear in processed_sentences, sub_requirements, extracted_skills, or sentence_importance_analysis.
    This exclusion does NOT apply to sentences where years-of-experience is merely mentioned ALONGSIDE a separately-stated, specific technical requirement in the same sentence (e.g. "3-6 years of experience working with scalable backends") — in that case, per the tiering rule below, the sentence stays in processed_sentences tiered by its technical content, and only the years sub-part is marked non-matchable at the sub_requirement level.
- "specificity_weight": the float weight corresponding to the tier (1.0 / 0.7 / 0.3 / 0.0)
- "tier_reason": a concise one-line explanation of why this tier was assigned
- "expected_depth": infer from the depth_qualifier using these rules:
    - "SURFACE" → qualifiers like: "familiar", "exposure", "awareness", "help with", "assist", "participate", "work with"
    - "INTERMEDIATE" → qualifiers like: "experience", "good", "knowledge", "ability", "responsible for", "create", "develop", "build"
    - "DEEP" → qualifiers like: "strong", "proficiency", "expert", "mastery", "drive", "lead", "architect", "mentor"
  If depth_qualifier is null, set expected_depth to null.
- "sub_requirements": if is_compound is true, an array of sub-objects, one per distinct sub-requirement. Each sub-object contains:
    - "sub_id": string, format "Xa" where X is the parent sentence id (e.g., "2a", "2b")
    - "text": a SHORT, context-bearing phrase optimized for embedding-based semantic matching against resume text — not a full grammatical sentence.
      CRITICAL TEXT RULE — applies to EVERY sub_requirement across ALL sentences:
      The text must always answer "What skill/domain/task does this require?" while staying compact.
      It must NEVER be a naked fragment, a single word, or a bare concept stripped of its context.
      The parent sentence's subject/domain (e.g., "databases", "Kubernetes", "cloud infrastructure")
      MUST be inherited into the sub_requirement text wherever the fragment alone would be
      too generic or meaningless without it.

      COMPACTNESS RULE: strip filler words, connector phrases, and redundant verbs that add no
      matching signal (e.g. "to help manage", "in delivered solutions", "as part of", "by emphasizing").
      Prefer a short noun-phrase or short-verb-phrase over a full sentence. Target roughly 2-6 words —
      only exceed this if the technical meaning genuinely cannot be preserved in fewer words. Never sacrifice
      the domain/technology context to save words — compactness and context-preservation are BOTH required.

      LIST-GROUPING RULE FOR ENUMERATED EXAMPLES: When a sentence lists multiple named tools/products as PARENTHETICAL EXAMPLES of a single broader category (e.g. "Google Cloud Platform services (Vertex AI, Cloud Run, BigQuery preferred)", "vector database (Pinecone, Weaviate, PGVector)"), do NOT split each named example into its own singleton sub_requirement. Instead, create ONE sub_requirement for the category with the named examples preserved as a comma-separated list in the text (e.g. "Google Cloud Platform services: Vertex AI, Cloud Run, BigQuery"). Only split into separate sub_requirements when the sentence structure genuinely requires each item to be independently satisfied (e.g. connected by "and" as separate mandatory requirements, not "such as"/parenthetical examples of one requirement).

      ❌ WRONG — naked fragments with no context:
        "performance", "security", "high availability", "monitoring", "scalability"
      ❌ WRONG — verbose full sentence (correct context, but not compact):
        "Developing tools for MRGR Analytics to improve efficiency and reduce duplication of effort"
      ✅ CORRECT — compact AND context-bearing:
        "Database performance tuning", "Database security practices",
        "Database monitoring and alerting", "MRGR Analytics tooling development",
        "Athena framework (Python) development", "ML model integration, enterprise apps"

      ❌ WRONG — strips the action from its target:
        "provisioning", "backup", "scaling"
      ✅ CORRECT — action + domain preserved, compact:
        "Database provisioning automation", "Database backup automation",
        "Database scaling in Kubernetes"

      ❌ WRONG — drops the technology from its qualifier:
        "custom controllers", "operators"
      ✅ CORRECT — technology context retained, compact:
        "Custom Kubernetes controllers", "Kubernetes operators in Go"

      This rule applies universally — to responsibilities, technical skills, technical domains,
      practices, and every other sub_requirement type without exception.
    - "type": one of "technical_skill", "programming_language", "technical_domain", "years_of_experience", "responsibility", "practice", "soft_skill"
    - "domain": (optional) the technical domain if type is "technical_domain" or "technical_skill"
    - "matchable": true if this sub-requirement can be verified against a resume; false if it cannot (e.g., years of experience cannot be verified from resume text alone)
    - "reason": (only if matchable is false) a brief explanation of why it cannot be matched
    - "match_keywords": (only if matchable is true and type is technical_domain or practice) an array of keywords that would indicate a match
  If is_compound is false, set sub_requirements to null.
- "entities": an array of the specific tools, technologies, frameworks, languages, protocols, or technical concepts explicitly named in THIS sentence's own text (not the whole JD — only what this sentence itself mentions). If the sentence names no specific technology/tool/concept, set this to an empty array [].
  For each entity found, include:
    - "entity": the concise, canonical name of the tool/technology/concept (e.g. "Node.js", "PostgreSQL", "Microservices Architecture"). Normalize casual wording to the commonly recognized proper name (e.g. "node" → "Node.js", "postgres" → "PostgreSQL").
    - "category": one of exactly two values —
        - "tool" — a concrete, named, installable/usable piece of technology: a language, framework, runtime, database, cloud platform, protocol, library, or software product (e.g. "Node.js", "PostgreSQL", "AWS", "OAuth 2.0").
        - "concept" — an abstract idea, architecture pattern, methodology, or practice that is not itself something you install or run (e.g. "Microservices Architecture", "Event-Driven Design", "CI/CD", "Zero Trust Architecture").
      Every entity must be classified as EXACTLY one of these two — there is no third option and no null.
    - "parent_relations": an array of 0-4 relationships describing broader technologies, foundations, or dependencies this entity connects to — drawn from YOUR OWN general knowledge of the technology ecosystem, NOT limited to what else is mentioned in this JD. Each relation is an object:
        - "target": the name of the related entity (e.g. "JavaScript" as a parent of "Node.js")
        - "relation_type": one of "BUILT_ON", "USES_LANGUAGE", "USES_PROTOCOL", "USES_TECHNOLOGY", "REQUIRES", "RUNS_ON", "IMPLEMENTS", "EXTENDS", "PART_OF", "ALTERNATIVE_TO", "RELATED_TO" — pick the single most accurate type.
        - "category": the SAME two-value classification ("tool" or "concept") applied to the TARGET of this relation, not the parent entity itself (e.g. "JavaScript" → "tool", "Service-Oriented Architecture" → "concept"). Every parent_relation target must also be classified — never omit this.
      Prioritize hierarchical/foundational relations (BUILT_ON, USES_LANGUAGE, REQUIRES, RUNS_ON, PART_OF, IMPLEMENTS) over ALTERNATIVE_TO/RELATED_TO. If an entity is itself foundational with no meaningful parent (e.g. "TCP/IP", "JavaScript"), return an empty array for parent_relations rather than forcing a weak relation.

  Example — sentence "Build backend services and microservices using Node.js":
  "entities": [
    {
      "entity": "Node.js",
      "category": "tool",
      "parent_relations": [
        {"target": "JavaScript", "relation_type": "USES_LANGUAGE", "category": "tool"},
        {"target": "V8 Engine", "relation_type": "BUILT_ON", "category": "tool"}
      ]
    },
    {
      "entity": "Microservices Architecture",
      "category": "concept",
      "parent_relations": [
        {"target": "Service-Oriented Architecture", "relation_type": "PART_OF", "category": "concept"}
      ]
    }
  ]
- "is_soft_skill": (optional, include only if true) set to true if this sentence contains a soft skill or culture trait component — even if it also contains technical or verifiable content. Do NOT move these to non_matchable_sentences; sentences that are PURELY soft-skill (no verifiable technical content at all) go to non_matchable_sentences. Mixed sentences referencing leadership, communication, collaboration, problem-solving, adaptability, etc. alongside technical or role-specific content stay in processed_sentences with is_soft_skill: true.
- "soft_skill_type": (required when is_soft_skill is true) one of: "communication", "collaboration_teamwork", "leadership", "adaptability", "problem_solving", "time_management", "emotional_intelligence" — pick the single best-fitting category
- "matchable": (optional, include only if false) set to false if the entire sentence cannot be matched against a resume
- "reason": (only if matchable is false at sentence level) brief explanation
- "dominant_role": the single job role (from job_roles) that this sentence is MOST relevant to. Must be an exact string match to one of the roles in job_roles.
- "role_attribution": an object mapping EACH role in job_roles to an integer percentage (0–100) indicating how much this sentence relates to that role. The percentages across all roles MUST sum to exactly 100. If a sentence is equally relevant to all roles, distribute evenly. If only relevant to one role, assign 100 to that role and 0 to others. Use your understanding of what each role does day-to-day.
"structural_sentences": an array of section headers, labels, or non-content lines found in the processed job description that do NOT belong in processed_sentences. For each structural sentence, include:
- "id": string, format "sN" (e.g., "s0", "s1", ...)
- "text": the exact text of the structural line
- "type": one of "section_header", "role_title", "company_label"
- "matchable": always false

Examples of structural sentences: "What will you do?", "Requirements", "Role Highlights", "Responsibilities:"
Sentence splitting rules: split processed_job_description by newline and by punctuation that ends sentences (., :, ;, or parentheses), but preserve meaningful bullet lines as separate sentences. Exclude lines shorter than 5 words except if they contain clear technical tokens (like "Python", "LLM", "PostgreSQL").
Precision and strict JSON are critical. Do not include any extra keys or narrative text. Respond only with the JSON object.

"non_matchable_sentences": an array of sentences found anywhere in the RAW job description 
(not just processed_job_description) that are pure soft skills, culture fit, or completely 
generic responsibilities that cannot be verified from resume text. 

IMPORTANT: Scan the full original job description for these — they may have been excluded 
from processed_job_description, but they must still be captured here.

Each object contains:
- "id": string, format "nmN" (e.g., "nm0", "nm1", ...)
- "text": the exact sentence text as it appears in the raw job description
- "type": one of "soft_skill", "culture_fit", "generic_responsibility", "education_requirement", "years_of_experience_requirement"
- "soft_skill_type": (required when type is "soft_skill") one of: "communication", "collaboration_teamwork", "leadership", "adaptability", "problem_solving", "time_management", "emotional_intelligence" — choose the single best-fitting category
- "tier": always "NON_MATCHABLE"
- "reason": a concise one-line explanation of why it cannot be matched

"testlify_analysis": an object built specifically for test curation. 
Derive it from the job description and your sentence-level analysis 
above. Judge each concept's importance from how the JD ITSELF 
emphasizes it — NOT from the extracted_skills list. You may surface 
a concept even if it was consolidated or simplified in extracted_skills.

For each role in job_roles, produce:
- "role": the exact role name (must match job_roles exactly)
- "role_importance_pct": an integer (0-100) for how much this entire 
  JD relies on this role. Judge from how many responsibilities and 
  requirements centre on this role and how central they are. The 
  role_importance_pct across ALL roles MUST sum to exactly 100.
- "concepts": an array of the key testable concepts for this role. 
  For each concept:
    - "concept": a SHORT, clean 2-4 word label suitable as a search 
  query. NOT a sentence or description. Examples of correct format:
  ✅ "CI/CD pipelines"
  ✅ "Terraform IaC"  
  ✅ "GCP infrastructure"
  ✅ "Docker Kubernetes"
  ✅ "RERA compliance"
  ✅ "Title due diligence"
  ❌ "CI/CD pipeline automation and management"
  ❌ "GCP core services (Compute Engine, Cloud Functions, VPC)"
  ❌ "Federal security compliance (FedRAMP, NIST) and ATO processes"
  The concept name is used as a direct search query — keep it tight 
  and specific, Testlify's semantic search handles the rest.
    - "importance_pct": an integer (0-100) judged from the JD itself:
        * stated as core requirement, strong language ("strong 
          expertise", "in-depth knowledge"), repeated across sections, 
          named specifically, or given a dedicated responsibility → 85-100
        * clearly required but routine → 60-85
        * mentioned once or in passing → 30-60

Rules for concepts:
- Include between 2 and 6 concepts per role (never more than 6, never 
  fewer than 2)
- Each importance_pct is scored INDEPENDENTLY — they do NOT sum to 100
- Order concepts by importance_pct descending
- Only include concepts with importance_pct >= 30
- Focus on testable technical/domain concepts, not soft skills

Output format:
{
  "testlify_analysis": {
    "roles_ranked": [
      {
        "role": "Backend Developer",
        "role_importance_pct": 70,
        "concepts": [
          {"concept": "AWS", "importance_pct": 90},
          {"concept": "REST APIs", "importance_pct": 75},
          {"concept": "Microservices", "importance_pct": 60}
        ]
      },
      {
        "role": "Java Developer",
        "role_importance_pct": 30,
        "concepts": [
          {"concept": "Java", "importance_pct": 80},
          {"concept": "Spring Boot", "importance_pct": 55}
        ]
      }
    ]
  }
}
"sentence_importance_analysis": an object that distributes a FIXED TOTAL of exactly 100 PERCENT across every content sentence in the job description, representing each sentence's relative importance/reliance within the JD.

Rules:
- Score EVERY sentence in processed_sentences — one entry per sentence, referenced by its "id". Do not skip any, do not invent extra ones.
- "importance_percentage": an integer from 0 to 100 representing that sentence's PERCENTAGE SHARE of the JD's total importance. This is NOT an absolute rating — it is this sentence's slice of a 100% pie shared across all sentences in the JD.
- Judge relative importance honestly — highly specific, differentiating, core sentences (named tools, named platforms, named compliance standards, core responsibilities, hard gating requirements) should claim a LARGER percentage; generic, boilerplate, restated-summary, or peripheral sentences should claim a MINIMAL percentage — as low as 0%.
- Do NOT distribute evenly. A wide spread across sentences is expected and correct.
- If a sentence is a general summary/intro that merely restates responsibilities or skills covered more specifically elsewhere in the JD, give it a LOWER percentage than the specific sentences it summarizes.
- Be consistent with the sentence "tier" you already assigned: CRITICAL sentences should collectively claim the majority of the 100% pool; IMPORTANT sentences should claim a moderate share; GENERIC sentences should claim very little.
- The importance_percentage values across ALL sentences do not need to sum to exactly 100 in your raw output — final normalization to exactly 100% is handled separately. Just be honest and proportional in your relative judgments.
- "importance_level": one of "HIGH", "MEDIUM", "LOW", assigned relative to this JD's distribution:
    * "HIGH"   — among the sentences claiming the largest percentage shares; core/differentiating content
    * "MEDIUM" — a moderate, mid-range percentage share
    * "LOW"    — a small percentage share; generic, boilerplate, or peripheral content
- "sub_requirement_importance": if the matching sentence in processed_sentences has is_compound = true (i.e. it has sub_requirements), this is an array breaking that sentence's importance_percentage down across its sub-parts. If is_compound is false, set this to null.
    * Each entry: {"sub_id": matching the parent sub_requirement's sub_id, "text": the SAME compact text used in that sub_requirement, "importance_percentage": an integer}
    * The importance_percentage values of all sub-parts MUST sum to EXACTLY the parent sentence's own importance_percentage — this is a breakdown of the sentence's already-allocated share, NOT additional points from the 100% pool.
    * Judge each sub-part's share based on how differentiating/specific it is relative to its siblings within the same sentence.
- Order the entries by "sentence_id" ascending.

Output format:
{
  "sentence_importance_analysis": {
    "sentences": [
      {
        "sentence_id": 0,
        "text": "exact sentence text",
        "importance_percentage": 8,
        "importance_level": "HIGH",
        "sub_requirement_importance": [
          {"sub_id": "0a", "text": "MRGR Analytics tooling development", "importance_percentage": 3},
          {"sub_id": "0b", "text": "Resource management tools", "importance_percentage": 3},
          {"sub_id": "0c", "text": "Cost optimization tools", "importance_percentage": 2}
        ]
      },
      {
        "sentence_id": 1,
        "text": "exact sentence text",
        "importance_percentage": 5,
        "importance_level": "MEDIUM",
        "sub_requirement_importance": null
      }
    ],
    "total_percentage": "100%"
  }
}
"job_logistics": an object capturing pre-requirement/administrative facts about the job, scanned from the FULL RAW job description (not just processed_job_description — these often appear in header lines, labels, or standalone statements that may otherwise be treated as structural).

Extract the following fields. If a field is not explicitly mentioned anywhere in the JD, set its value to null — do not guess or infer.

- "years_of_experience": a string summarizing the stated experience requirement exactly as phrased (e.g. "4+ years in cloud engineering, with at least 2 years on GCP", "Minimum 2 years of full stack application development"). If multiple experience mentions exist, include all of them concisely in one string. If none, null.
- "education_qualification": a string summarizing the stated education/degree requirement exactly as phrased (e.g. "Bachelors or Masters degree in a Science, Technology, Engineering, or Mathematics discipline"). If none, null.
- "location": the stated work location, city/region, or address if mentioned (e.g. "Washington, D.C. 20036"). If none, null.
- "work_mode": one of "On-site", "Hybrid", "Remote", or a direct quote if phrased with specific detail (e.g. "Hybrid, On-site 4 days/week and 1 day Telework"). If none mentioned, null.
- "visa_work_authorization": the stated citizenship, visa, or work-authorization requirement if mentioned (e.g. "U.S. citizenship required due to federal contract requirements", "Must be authorized to work in the U.S. without sponsorship"). If none, null.
- "employment_type": one of "Full Time", "Part Time", "Contract", "Contract-to-Hire", "Internship", or a direct quote if phrased differently. If none mentioned, null.
- "security_clearance": the stated clearance requirement if mentioned (e.g. "Willing to obtain a Top Secret security clearance"). If none, null.

Rules:
- Scan the ENTIRE raw job description, including header/label lines (e.g. "Location:", "Job Type:", "Clearance Requirement:") even if those lines were excluded from processed_job_description or classified as structural_sentences elsewhere.
- Extract values as close to the JD's original wording as possible — do not paraphrase away specific details (e.g. keep "4 days/week" if stated).
- Do not fabricate a value for any field not explicitly present in the JD.
- This section is purely extractive/informational — it does not affect tiering, scoring, or role inference elsewhere in the output.

Output format:
{
  "job_logistics": {
    "years_of_experience": "4+ years in cloud engineering, with at least 2 years on GCP",
    "education_qualification": null,
    "location": "Washington, D.C. 20036",
    "work_mode": "Hybrid, On-site 4 days/week and 1 day Telework",
    "visa_work_authorization": "U.S. citizenship required due to federal contract requirements",
    "employment_type": "Full Time",
    "security_clearance": "Willing to obtain a Top Secret security clearance"
  }
}


"""


# =============================================================================
# JOB DESCRIPTION CLASSIFIER CLASS
# =============================================================================

class JobDescriptionClassifier:
    """
    Classifies job descriptions using OpenAI GPT API.
    
    ALL CLASSIFICATION IS DYNAMIC:
    - No predefined role lists
    - No hardcoded keyword sets
    - Everything inferred by GPT from the job description
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.2"):
        """
        Initialize the classifier.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use (default: gpt-5.2 for best results)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        print(f"✓ JobDescriptionClassifier initialized with model: {self.model}")
    
    def classify(self, job_description: str, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Classify a job description and return structured analysis.
        
        ALL LOGIC IS HANDLED BY GPT:
        - Role inference is dynamic (no predefined list)
        - Soft skill categories are created dynamically
        - Word-level coverage computed by GPT
        
        Args:
            job_description: Raw job description text
            temperature: Model temperature (0.3 for balanced creativity in role identification)
            
        Returns:
            Dictionary containing processed JD, roles, soft skills, etc.
        """
        print("\n" + "=" * 80)
        print("PROCESSING JOB DESCRIPTION")
        print("=" * 80)
        
        # Prepare the user message
        user_message = f"""Analyze the following job description and produce the JSON output as specified:

<JOB_DESCRIPTION>
{job_description}
</JOB_DESCRIPTION>

CRITICAL REQUIREMENTS:

1. Extract ONLY technically relevant content for processed_job_description
2. Infer job roles dynamically — do NOT use a predefined list
3. **MINIMUM 2 ROLES REQUIRED: You MUST always return at least 2 distinct, non-overlapping roles. A single-role output is NEVER acceptable. If only 1 role seems to fit, split by distinct functional dimensions (e.g. engineering vs architecture, development vs operations, analysis vs strategy).**
4. **ANTI-OVERLAP IS MANDATORY: Before finalizing, check every pair of roles. If two roles cover the same core function, merge them into one. Do not output roles that overlap with each other.**
5. **PARENT-CHILD COLLAPSE: Never include both a general role and a domain-specific variant of it. Pick the one that best fits the JD.**
6. **Use BASE role titles WITHOUT seniority labels** — no "Senior", "Junior", "Lead", "Principal", etc.
7. If the role involves AI/ML tasks, the role name MUST contain "AI" or "ML"
8. Do NOT output "tier_weights", "depth_qualifier_map", or "summary" — these are computed locally.
9. **SOFT SKILL EXTRACTION IS MANDATORY**: Actively scan ALL sentences — processed, structural, and non-matchable — for soft skill signals (communication, leadership, collaboration, adaptability, problem-solving, time management, emotional intelligence). Pure soft-skill sentences go into non_matchable_sentences with type="soft_skill" AND soft_skill_type. Sentences that combine a soft skill with a SEPARATE, SPECIFIC, verifiable technical requirement (e.g. named tool, named domain, concrete deliverable) stay in processed_sentences with is_soft_skill=true AND soft_skill_type — but only if the technical portion alone would independently qualify as CRITICAL or IMPORTANT per rule 11. If the "technical" portion is itself vague or generic, the whole sentence is pure soft-skill and belongs in non_matchable_sentences instead. Never omit soft_skill_type when a sentence is identified as a soft skill.
10. **TESTLIFY ANALYSIS**: Produce the testlify_analysis object. Rank every role with role_importance_pct (all roles summing to 100). Under each role list 2 to 6 important concepts, each with an independent importance_pct (0-100) judged from how strongly the JD emphasizes that concept. Order concepts by importance descending.
11. **processed_sentences MUST ONLY contain CRITICAL or IMPORTANT tier sentences** — specific, verifiable, technical content describing what the candidate must DO or KNOW. GENERIC boilerplate, pure soft-skill/mindset statements, education/degree requirements, and bare years-of-experience thresholds must ALL be routed to non_matchable_sentences instead, never left in processed_sentences.
12. **ENTITY + PARENT RELATIONSHIP EXTRACTION IS MANDATORY**: For every sentence in processed_sentences, populate the "entities" array with every specific tool/technology/concept explicitly named in that sentence's own text, classify EACH entity with "category": "tool" or "category": "concept" (never omit this — no third option, no null), and for each entity include up to 4 "parent_relations" drawn from your own general technology knowledge (not just what's stated elsewhere in this JD — e.g. "Node.js" → parent relation to "JavaScript" via USES_LANGUAGE). Every parent_relation target must ALSO carry its own "category" ("tool" or "concept"). Use an empty array [] where nothing applies — never omit the "entities" key on any processed_sentence.

Return ONLY valid JSON, no additional text"""



        print(f"\nSending request to {self.model}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_completion_tokens=32000,
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            # Check finish reason for truncation
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise ValueError(
                    "Response was truncated (finish_reason='length') — output exceeded "
                    "max_completion_tokens before the JSON was complete. Increase "
                    "max_completion_tokens and retry."
                )
            elif finish_reason != "stop":
                print(f"⚠️  WARNING: Unexpected finish_reason='{finish_reason}'")
            
            # Extract the response content
            response_text = response.choices[0].message.content
            
            # Handle empty or None response
            if not response_text or not response_text.strip():
                usage = getattr(response, 'usage', None)
                print(f"✗ Empty response received from API")
                print(f"   finish_reason: {finish_reason}")
                if usage:
                    print(f"   prompt_tokens: {usage.prompt_tokens}")
                    print(f"   completion_tokens: {usage.completion_tokens}")
                    print(f"   total_tokens: {usage.total_tokens}")
                raise ValueError(
                    f"API returned empty response (finish_reason='{finish_reason}'). "
                    f"The prompt + output may exceed the model's context window, "
                    f"or the output was truncated. Try increasing max_completion_tokens."
                )
                                                                                                             
            print("✓ Response received from API")
            
            # Ensure response_text is a str for static type checkers and parse JSON response
            if response_text is None:
                raise ValueError("API returned no response content (None).")
            result = self._parse_response(response_text)
            
            # Inject hardcoded constants that were removed from GPT output
            result['tier_weights'] = TIER_WEIGHTS
            result['depth_qualifier_map'] = DEPTH_QUALIFIER_MAP

            # Defensive normalization of GPT-inferred entities/parent_relations
            ensure_entities_field_locally(result)
            
            # Compute summary locally instead of wasting GPT tokens
            result['summary'] = compute_summary_locally(
                result.get('processed_sentences', []),
                result.get('structural_sentences', []),
                result.get('non_matchable_sentences', [])
            )

            # Aggregate soft skills into canonical categories
            result['soft_skills'] = compute_soft_skills_locally(
                result.get('processed_sentences', []),
                result.get('non_matchable_sentences', []),
                result.get('structural_sentences', [])
                
            )
            normalize_sentence_importance_locally(result)

            # Distribute each sentence's importance across its tools/concepts
            distribute_entity_importance_locally(result)

            # Log role count (retry removed — prompt now enforces minimum 2)
            num_roles = len(result.get("job_roles", []))
            if num_roles < 2:
                print(f"\n⚠️  WARNING: Only {num_roles} role(s) returned despite minimum-2 instruction.")
            
            # Validate the result structure
            self._validate_result(result)
            
            print("✓ Response validated successfully")
            
            return result
            
        except Exception as e:
            print(f"✗ Error during classification: {str(e)}")
            raise
    
    def compute_sentence_role_matrix(
    self,
    sentences: List[str],
    job_roles: List[str],
    temperature: float = 0.1
) -> Dict[str, Any]:
        """
        Uses GPT to classify each sentence against all job roles
        and assign percentage attribution.

        Args:
            sentences: List of processed sentences from the JD
            job_roles: List of job roles extracted by GPT
            temperature: Low temperature for precise classification

        Returns:
            Dictionary containing roles, sentence_breakdown, and matrix
        """
        print("\n" + "=" * 80)
        print("COMPUTING SENTENCE → ROLE MATRIX VIA GPT")
        print("=" * 80)

        roles_str = "\n".join([f"- {role}" for role in job_roles])
        sentences_str = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])

        prompt = f"""You are a job description analyst. You will be given a list of sentences from a job description and a list of job roles.

    Your task is to analyze each sentence and determine how much it relates to each job role, expressed as a percentage.

    RULES:
    - For each sentence, assign a percentage to each role indicating how much that sentence relates to that role
    - The percentages for each sentence MUST sum to exactly 100
    - If a sentence is equally relevant to all roles, distribute evenly
    - If a sentence is only relevant to one role, assign 100% to that role and 0% to others
    - If a sentence is not relevant to any role (e.g. generic statements), distribute evenly
    - Be precise and use your deep understanding of what each role does day-to-day
    - The dominant_role must be the role with the highest percentage for that sentence

    JOB ROLES:
    {roles_str}

    SENTENCES:
    {sentences_str}

    Return a JSON object with this exact structure:
    {{
        "roles": {json.dumps(job_roles)},
        "sentence_breakdown": [
            {{
                "sentence_index": 1,
                "sentence": "exact sentence text",
                "role_attribution": {{
                    "Role Name 1": 70,
                    "Role Name 2": 20,
                    "Role Name 3": 10
                }},
                "dominant_role": "Role Name 1",
                "dominant_pct": 70
            }}
        ]
    }}

    CRITICAL:
    - Include ALL {len(sentences)} sentences in sentence_breakdown
    - Percentages per sentence MUST sum to exactly 100
    - Use the EXACT role names as provided
    - Return ONLY valid JSON, no additional text
    """

        print(f"Sending {len(sentences)} sentences and {len(job_roles)} roles to GPT...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise job description analyst. You classify job description sentences into job roles with percentage attribution. Always    return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_completion_tokens=32000,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)

            # Build matrix from sentence_breakdown
            sentence_breakdown = result.get("sentence_breakdown", [])
            matrix = []
            for item in sentence_breakdown:
                attribution = item.get("role_attribution", {})
                matrix.append([attribution.get(role, 0) for role in job_roles])

            result["matrix"] = matrix

            print(f"✓ GPT classified {len(sentence_breakdown)} sentences across {len(job_roles)} roles")

            return result

        except Exception as e:
            print(f"✗ Error computing sentence-role matrix: {str(e)}")
            raise

    # -------------------------------------------------------------------------

    def classify_with_forced_roles(
        self,
        job_description: str,
        forced_roles: List[str],
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Re-classify a job description using a user-supplied list of roles.

        GPT is told to use EXACTLY the provided roles — it will NOT infer new
        ones.  All other classification logic (sentence processing, skill
        extraction, soft-skill tagging, entity/parent-relation extraction,
        etc.) runs as normal.

        Args:
            job_description : Raw JD text (the same one originally submitted)
            forced_roles    : Exact list of role names to enforce
            temperature     : Model temperature (default 0.3)

        Returns:
            Same dict structure as classify() but with job_roles == forced_roles
            and role_attribution computed for those specific roles.
        """
        print("\n" + "=" * 80)
        print("RE-CLASSIFYING WITH FORCED ROLES")
        print(f"Roles: {forced_roles}")
        print("=" * 80)

        roles_json = json.dumps(forced_roles)

        user_message = f"""Analyze the following job description and produce the JSON output as specified.

<JOB_DESCRIPTION>
{job_description}
</JOB_DESCRIPTION>

FORCED ROLES — USE EXACTLY THESE, DO NOT INFER NEW ONES:
{roles_json}

CRITICAL REQUIREMENTS:
1. job_roles MUST equal exactly {roles_json} — no additions, no removals, no renaming.
2. All role_attribution objects in processed_sentences MUST use only the role names above.
3. All role_attribution percentages per sentence MUST sum to exactly 100.
4. Every other field (processed_job_description, processed_sentences, extracted_skills,
   soft_skills, responsibility_actions, experience_requirements, role_similarity_matrix,
   structural_sentences, non_matchable_sentences, job_role_domain) follows the same rules
   as a normal classification.
5. Do NOT output "tier_weights", "depth_qualifier_map", or "summary".
6. Every processed_sentence MUST still include the "entities" array, each entity classified with "category" ("tool" or "concept") and its "parent_relations" (each target also carrying its own "category"), drawn from your own general technology knowledge, exactly per the schema in the system prompt.

Return ONLY valid JSON, no additional text."""

        print(f"\nSending forced-role request to {self.model}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_completion_tokens=32000,
                response_format={"type": "json_object"},
            )

            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise ValueError(
                "Response was truncated (finish_reason='length') — output exceeded "
                "max_completion_tokens before the JSON was complete. Increase "
                "max_completion_tokens and retry."
            )

            response_text = response.choices[0].message.content
            if not response_text or not response_text.strip():
                raise ValueError(
                    f"API returned empty response (finish_reason='{finish_reason}')."
                )

            print("✓ Response received from API")
            result = self._parse_response(response_text)

            # Override job_roles to guarantee the forced list is preserved
            result["job_roles"] = forced_roles

            # Inject constants
            result["tier_weights"] = TIER_WEIGHTS
            result["depth_qualifier_map"] = DEPTH_QUALIFIER_MAP

            # Defensive normalization of GPT-inferred entities/parent_relations
            ensure_entities_field_locally(result)

            # Compute derived fields locally
            result["summary"] = compute_summary_locally(
                result.get("processed_sentences", []),
                result.get("structural_sentences", []),
                result.get("non_matchable_sentences", []),
            )
            result["soft_skills"] = compute_soft_skills_locally(
                result.get("processed_sentences", []),
                result.get("non_matchable_sentences", []),
                result.get("structural_sentences", []),
            )
            normalize_sentence_importance_locally(result)
            distribute_entity_importance_locally(result)

            self._validate_result(result)
            print("✓ Forced-role classification validated successfully")
            return result

        except Exception as e:
            print(f"✗ Error during forced-role classification: {str(e)}")
            raise

    # -------------------------------------------------------------------------

    def get_role_names_only(self, result: Dict[str, Any]) -> List[str]:
        """Extract just the role names from classification result."""
        # job_roles is already a list of strings
        return result.get("job_roles", [])
    
    def get_processed_description(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract processed job description sentences from classification result.
        
        Returns the technically-relevant job description as a list of sentences,
        which can be used directly as the job_description parameter in jobbert_JD.py.
        
        Args:
            result: Classification result from classify()
            
        Returns:
            List of processed job description sentences (technical content only)
        """
        sentences = result.get("processed_sentences", [])
# New format returns objects — extract text for downstream string consumers
        if sentences and isinstance(sentences[0], dict):
            return [s.get("text", "") for s in sentences]
        return sentences

    def get_sentence_entities(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convenience accessor: flatten every sentence's GPT-inferred "entities"
        (with their "parent_relations") into a single list, tagged with the
        sentence id/text they came from. Useful for anything downstream that
        wants a flat entity list rather than walking processed_sentences.

        Args:
            result: Classification result from classify()

        Returns:
            List of {"sentence_id", "sentence_text", "entity", "category",
            "parent_relations"} dicts, one per entity (not per sentence).
            "category" is "tool" or "concept"; each parent_relations item
            also carries its own "category" for its target.
        """
        flat = []
        for s in result.get("processed_sentences", []):
            for ent in s.get("entities", []) or []:
                flat.append({
                    "sentence_id": s.get("id"),
                    "sentence_text": s.get("text"),
                    "entity": ent.get("entity"),
                    "category": ent.get("category"),
                    "parent_relations": ent.get("parent_relations", []),
                })
        return flat
    
    def classify_for_jobbert(self, job_description: str, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Classify job description and return a JobBERT-compatible format.
        
        This is a convenience method that combines classify() with format conversion.
        
        Args:
            job_description: Raw job description text
            temperature: Model temperature (default 0.3 for balanced role diversity)
            
        Returns:
            Dictionary with ONLY:
            - 'job_roles': List[str] - Simple list of role names
            - 'job_description': str - Processed technical description
        """
        # Get full classification result
        result = self.classify(job_description, temperature=temperature)
        
        # Convert to a simple, minimal format (ONLY the essentials)
        return {
            'job_roles': self.get_role_names_only(result),
            'job_description': self.get_processed_description(result)
        }
    
    def _parse_response(self, response_text: Optional[str]) -> Dict[str, Any]:
        """
        Parse the JSON response from the API.
        
        Args:
            response_text: Raw response text from API (may be None)
            
        Returns:
            Parsed dictionary
        """
        if response_text is None:
            raise ValueError("API returned no response content (None).")

        try:
            # Clean up potential markdown code blocks
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            return json.loads(cleaned.strip())
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON parsing error: {str(e)}")
            preview = response_text[:500] if isinstance(response_text, str) else "<None>"
            print(f"Response text preview: {preview}...")
            raise ValueError(f"Failed to parse API response as JSON: {str(e)}")
    
    def _validate_result(self, result: Dict[str, Any]) -> None:
        """
        Validate the classification result against expected schema.

        Args:
            result: Parsed result dictionary

        Raises:
            ValueError: If validation fails
        """
        required_keys = [
            "processed_job_description",
            "processed_sentences", 
            "job_roles"
        ]

        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key in response: {key}")

        # Validate job_roles structure (should be list of strings)
        if not isinstance(result.get("job_roles"), list):
            raise ValueError("job_roles must be a list")

        for i, role in enumerate(result["job_roles"]):
            if not isinstance(role, str):
                raise ValueError(f"job_roles[{i}] must be a string (role name), got {type(role)}")

            if not role.strip():
                raise ValueError(f"job_roles[{i}] is empty")

        # Ensure uncovered_sentences exists (even if empty)
        if "uncovered_sentences" not in result:
            result["uncovered_sentences"] = []

        # Validate role_similarity_matrix
        if "role_similarity_matrix" not in result:
            # Build a default empty matrix if GPT missed it
            roles = result.get("job_roles", [])
            matrix = [[100 if i == j else 0 for j in range(len(roles))] for i in range(len(roles))]
            pairs = []
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    pairs.append({"role_1": roles[i], "role_2": roles[j], "similarity_pct": 0})
            result["role_similarity_matrix"] = {
                "roles": roles,
                "matrix": matrix,
                "pairs": pairs
            }
            print("⚠️  role_similarity_matrix missing from response — default built.")
        else:
            print("✓ role_similarity_matrix present in response")

        # Role count feedback
        num_roles = len(result["job_roles"])
        if num_roles == 0:
            print(f"\n⚠️  WARNING: No roles identified. Check if the JD was processed correctly.")
        elif num_roles == 1:
            print(f"\n⚠️  WARNING: Only 1 role identified — prompt should have enforced minimum 2.")
        else:
            print(f"\n✅ {num_roles} role(s) identified.")
    
    def classify_batch(self, job_descriptions: List[str], temperature: float = 0.3) -> List[Dict[str, Any]]:
        """
        Classify multiple job descriptions.
        
        Args:
            job_descriptions: List of raw job description texts
            temperature: Model temperature
            
        Returns:
            List of classification results
        """
        results = []
        total = len(job_descriptions)
        
        for i, jd in enumerate(job_descriptions, 1):
            print(f"\n{'=' * 80}")
            print(f"Processing job description {i}/{total}")
            print(f"{'=' * 80}")
            
            try:
                result = self.classify(jd, temperature=temperature)
                results.append(result)
            except Exception as e:
                print(f"✗ Failed to classify JD {i}: {str(e)}")
                results.append({"error": str(e), "job_description": jd[:200] + "..."})
        
        return results


# =============================================================================
# RESULT ANALYZER (Display & Export Only - No Hardcoded Logic)
# =============================================================================

class ResultAnalyzer:
    """
    Utility class for analyzing and displaying classification results.
    
    NOTE: This class only displays/exports results from GPT.
    It does NOT perform any classification or use any hardcoded keywords.
    """
    
    @staticmethod
    def print_summary(result: Dict[str, Any]) -> None:
        """Print a formatted summary of the classification result."""
        print("\n" + "=" * 80)
        print("CLASSIFICATION SUMMARY")
        print("=" * 80)
        
        # Processed sentences count
        sentences = result.get("processed_sentences", [])
        print(f"\n📝 Processed Sentences: {len(sentences)}")
        
        # Job roles
        print(f"\n🎯 Inferred Job Roles ({len(result.get('job_roles', []))} total):")
        print("-" * 80)
        
        print(f"\n🎯 Job Roles Identified ({len(result.get('job_roles', []))} total):")
        print("=" * 80)
        
        for i, role in enumerate(result.get("job_roles", []), 1):
            print(f"  {i}. {role}")
        

        
        # Uncovered sentences
        uncovered = result.get("uncovered_sentences", [])
        if uncovered:
            print(f"\n⚠️  Uncovered Sentences ({len(uncovered)}):")
            print("-" * 80)
            for item in uncovered:
                if isinstance(item, dict):
                    print(f"  [{item.get('index', '?')}] {item.get('sentence', 'N/A')}")
                else:
                    print(f"  {item}")
        else:
            print("\n✓ All sentences covered by inferred roles")
        
        print("\n" + "=" * 80)

    
    
    
    @staticmethod
    def export_to_json(result: Dict[str, Any], filepath: str) -> None:
        """Export classification result to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Result exported to: {filepath}")
    
    
    @staticmethod
    def print_sentence_role_matrix(sentence_matrix: Dict[str, Any]) -> None:
        """
        Print the sentence-role matrix in a readable format.

        Args:
            sentence_matrix: Result from build_sentence_role_matrix_locally()
        """
        if not sentence_matrix:
            print("⚠️  No sentence-role matrix to display.")
            return

        print("\n" + "=" * 80)
        print("SENTENCE → ROLE ATTRIBUTION MATRIX")
        print("=" * 80)

        roles = sentence_matrix["roles"]
        breakdown = sentence_matrix["sentence_breakdown"]

        for item in breakdown:
            sentence = item["sentence"]
            attribution = item["role_attribution"]
            dominant = item["dominant_role"]
            dominant_pct = item["dominant_pct"]

            # Truncate long sentences for display
            display_sentence = sentence[:80] + "..." if len(sentence) > 80 else sentence

            print(f"\n[{item.get('sentence_index', '?')}] {display_sentence}")
            print(f"     Dominant: {dominant} ({dominant_pct}%)")
            print(f"     Breakdown:")

            for role in roles:
                pct = attribution.get(role, 0)
                bar_length = int(pct / 5)  # max 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"       {role:<35} [{bar}] {pct}%")

        print("\n" + "=" * 80)
        print("ROLE DOMINANCE SUMMARY")
        print("=" * 80)

        # Count how many sentences each role dominates
        dominance_count = {role: 0 for role in roles}
        for item in breakdown:
            dominant = item.get("dominant_role")
            if dominant in dominance_count:
                dominance_count[dominant] += 1

        total_sentences = len(breakdown)
        print(f"\nTotal sentences: {total_sentences}")
        for role in roles:
            count = dominance_count[role]
            pct = round((count / total_sentences) * 100, 2) if total_sentences > 0 else 0
            bar_length = int(pct / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {role:<35} [{bar}] {count} sentences ({pct}%)")

        print("\n" + "=" * 80)
    
    
    @staticmethod
    def print_role_similarity_matrix(similarity_result: Dict[str, Any]) -> None:
        """
        Print the role similarity matrix in a readable format.

        Args:
            similarity_result: role_similarity_matrix from classify()
        """
        print("\n" + "=" * 80)
        print("ROLE SIMILARITY MATRIX")
        print("=" * 80)

        roles = similarity_result["roles"]
        matrix = similarity_result["matrix"]
        pairs = similarity_result["pairs"]

        # Print matrix header
        col_width = 28
        header = f"{'':>{col_width}}"
        for role in roles:
            truncated = role[:14] + ".." if len(role) > 14 else role
            header += f"  {truncated:>16}"
        print(header)
        print("-" * (col_width + (18 * len(roles))))

        # Print matrix rows
        for i, role in enumerate(roles):
            truncated = role[:26] + ".." if len(role) > 26 else role
            row = f"{truncated:>{col_width}}"
            for j in range(len(roles)):
                if i == j:
                    row += f"  {'100.00%':>16}"
                else:
                    row += f"  {str(matrix[i][j]) + '%':>16}"
            print(row)

        # Print sorted pairs
        print("\n" + "=" * 80)
        print("PAIRWISE SIMILARITY (Sorted by Similarity)")
        print("=" * 80)
        for pair in pairs:
            bar_length = int(pair["similarity_pct"] / 5)  # max 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"\n  {pair['role_1']}  ↔  {pair['role_2']}")
            print(f"  [{bar}] {pair['similarity_pct']}%")

        print("\n" + "=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating the classifier usage."""
    
    # Sample job description
    job_description = """
Job Title: Backend Engineer (Node.js & Cloud) Role Overview We are looking for a skilled Backend Engineer to design, scale, and maintain our core server-side logic and APIs. In this role, you will transition legacy components into microservices, optimize data workflows, and ensure our cloud infrastructure is secure and resilient. You will work closely with frontend, DevOps, and product teams to deliver high-quality, production-ready software. Core Responsibilities API & Service Development: Design, build, and maintain highly scalable backend services and robust RESTful/GraphQL APIs using Node.js. Architecture Evolution: Develop microservices and event-driven applications to support high-throughput, low-latency business logic. Data Management: Write optimized queries, design schemas, and manage data consistency across both relational (SQL) and non-relational (NoSQL) databases. Cloud & Security: Deploy and maintain applications in cloud environments, implementing secure authentication/authorization protocols (e.g., OAuth, JWT) and data encryption. Engineering Excellence: Drive application performance tuning, conduct rigorous code reviews, write automated tests, and support CI/CD pipeline automation. Sourcing & Screening Guide for TA1. Must-Have Technical Skills (Screening Criteria) Language/Framework: Deep production experience with Node.js (and frameworks like Express, NestJS, or Fastify). Database Proficiency: Hands-on experience with both relational (PostgreSQL, MySQL) and non-relational (MongoDB, DynamoDB, Redis) databases. Cloud Infrastructure: Proven experience deploying and monitoring applications in a major cloud environment (AWS, Azure, or GCP). Architectural Patterns: Clear understanding of microservices, event-driven architecture, and message brokers (e.g., Kafka, RabbitMQ, or AWS SQS). Testing & Quality: Experience writing unit/integration tests (using Jest, Mocha, or Chai) and participating in CI/CD workflows. 2. Nice-to-Have Skills (Sourcing Differentiators) Experience with TypeScript in a backend environment. Familiarity with containerization tools like Docker and orchestration via Kubernetes. Understanding of Infrastructure as Code (IaC) tools like Terraform."""
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n" + "=" * 80)
        print("OPENAI API KEY REQUIRED")
        print("=" * 80)
        print("\nPlease set your OpenAI API key using one of these methods:")
        print("  1. Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("  2. Pass it directly when creating the classifier")
        print("\nExample:")
        print("  classifier = JobDescriptionClassifier(api_key='sk-...')")
        print("=" * 80)
        
        api_key = input("\nEnter your OpenAI API key (or press Enter to exit): ").strip()
        if not api_key:
            print("Exiting...")
            return
    
    try:
        # Initialize the classifier
        classifier = JobDescriptionClassifier(api_key=api_key, model="gpt-5.2")
        
        # =====================================================================
        # STEP 1: ChatGPT Classification (CALLED ONLY ONCE)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 1: ChatGPT Role Identification & Preprocessing")
        print("=" * 80)
        
        result = classifier.classify(job_description, temperature=0.3)
        
        # Print ChatGPT summary
        ResultAnalyzer.print_summary(result)

        # =====================================================================
        # STEP 2: Extract roles + processed description (No additional API call)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 2: Roles + Processed Description (Extracted from Step 1)")
        print("=" * 80)
        
        # Extract from already-computed result (no new API call)
        job_roles = classifier.get_role_names_only(result)
        processed_desc = classifier.get_processed_description(result)
        
        print("\n✅ JOB ROLES:")
        print("=" * 80)
        print("self.job_roles = [")
        for role in job_roles:
            print(f'    "{role}",')
        print("]")
        print(f"\nTotal roles: {len(job_roles)}")
        
        print("\n✅ PROCESSED JOB DESCRIPTION:")
        print("=" * 80)
        if isinstance(processed_desc, list):
            print(f"Format: List of {len(processed_desc)} sentences")
            print("\nFirst 5 sentences:")
            for i, sentence in enumerate(processed_desc[:5], 1):
                print(f"  {i}. {sentence}")
            if len(processed_desc) > 5:
                print(f"  ... and {len(processed_desc) - 5} more sentences")
            print(f"\nTotal sentences: {len(processed_desc)}")

        # =====================================================================
        # STEP 2b: Flattened entity + parent-relation list (No additional API call)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 2b: Sentence Entities + Parent Relations (Extracted from Step 1)")
        print("=" * 80)

        sentence_entities = classifier.get_sentence_entities(result)
        print(f"\nTotal entities extracted across all sentences: {len(sentence_entities)}")
        for item in sentence_entities[:10]:
            rels = ", ".join(
                f'{r["relation_type"]}→{r["target"]} ({r["category"]})' for r in item["parent_relations"]
            ) or "(no parent relations)"
            print(f'  [{item["sentence_id"]}] {item["entity"]} ({item["category"]}): {rels}')
        if len(sentence_entities) > 10:
            print(f"  ... and {len(sentence_entities) - 10} more")
        
        # =====================================================================
        # STEP 3: Role Similarity Matrix (GPT-computed, no additional API call)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 3: Role Similarity Matrix (GPT-computed)")
        print("=" * 80)

        similarity_result = result.get("role_similarity_matrix", None)
        if similarity_result:
            ResultAnalyzer.print_role_similarity_matrix(similarity_result)
        else:
            print("\n⚠️  No similarity matrix found in GPT result.")

        # =====================================================================
        # STEP 4: Sentence → Role Attribution Matrix (Built locally from Step 1)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 4: Sentence → Role Attribution Matrix (Built locally from Step 1)")
        print("=" * 80)

        sentence_matrix = build_sentence_role_matrix_locally(
            result.get('processed_sentences', []),
            job_roles
        )

        if sentence_matrix:
            ResultAnalyzer.print_sentence_role_matrix(sentence_matrix)
        else:
            sentence_matrix = None
            print("\n⚠️  Could not build sentence-role matrix.")

        # =====================================================================
        # STEP 5: Soft Skill Summary (already computed in Step 1)
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 5: Soft Skill Extraction (Using Step 1 Results)")
        print("=" * 80)

        soft_skills = result.get('soft_skills', {})

        print("\nSOFT SKILLS EXTRACTION RESULTS")
        print("=" * 80)
        if soft_skills:
            print(f"\nSoft skill categories found: {len(soft_skills)}")
            for category, data in sorted(soft_skills.items()):
                print(f"\n[{category.upper().replace('_', ' ')}] ({data['count']} sentence{'s' if data['count'] != 1 else ''}, {data['percentage']}%)")
                print("-" * 80)
                for i, sentence in enumerate(data['sentences'], 1):
                    print(f"  {i}. {sentence}")
        else:
            print("\n⚠️  No soft skills found in the job description.")
        print("\n" + "=" * 80)

        # =====================================================================
        # SINGLE OUTPUT FILE — everything the pipeline produces, one file.
        # =====================================================================
        output_path = r"combined_classification.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json_result = {
                'chatgpt_roles': job_roles,
                'job_role_domain': result.get('job_role_domain', None),
                'predicted_role': job_roles[0] if job_roles else None,
                'processed_sentences': result.get('processed_sentences', []),
                'structural_sentences': result.get('structural_sentences', []),
                'non_matchable_sentences': result.get('non_matchable_sentences', []),
                'tier_weights': result.get('tier_weights', {}),
                'extracted_skills': result.get('extracted_skills', []),
                'responsibility_actions': result.get('responsibility_actions', []),
                'experience_requirements': result.get('experience_requirements', {'min': 0, 'max': 0}),
                'depth_qualifier_map': result.get('depth_qualifier_map', {}),
                'role_similarity_matrix': similarity_result if similarity_result else None,
                'sentence_role_matrix': {
                    'roles': sentence_matrix['roles'],
                    'sentence_breakdown': sentence_matrix['sentence_breakdown']
                } if sentence_matrix else None,
                'summary': result.get('summary', {}),
                'soft_skills': soft_skills,
                'testlify_analysis': result.get('testlify_analysis', None),
                'sentence_importance_analysis': result.get('sentence_importance_analysis', None),
                # JD-wide, normalized-to-100% ranking of every tool/concept —
                # the headline output of the scoring engine. Verb weighting is
                # already folded in per-sentence (see governing_verb on each
                # processed_sentence above) rather than listed separately here.
                'entity_importance_summary': result.get('entity_importance_summary', []),
                'job_logistics': result.get('job_logistics', None),
                # NOTE: per-sentence "entities" (scores + "governing_verb") are
                # already nested inside 'processed_sentences' above — no
                # separate export needed.
            }
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ All results exported to: {output_path}")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n" + "=" * 80)
        print("COMPLETE PIPELINE SUMMARY")
        print("=" * 80)
        print("\n✅ ChatGPT API called: 1 time")
        print(f"✅ Roles identified: {len(job_roles)}")
        print(f"✅ Sentences processed: {len(processed_desc)}")
        print(f"✅ Entities extracted (with parent relations): {len(sentence_entities)}")
        print(f"✅ Output file: {output_path}")

        top_entities = result.get('entity_importance_summary', [])[:10]
        if top_entities:
            print("\n🏆 TOP TOOLS/CONCEPTS FOR THIS JD (normalized to 100% JD-wide):")
            print("-" * 80)
            for e in top_entities:
                print(f"  {e['percentage']:>3}%  {e['entity']:<35} ({e['category']}, seen in {e['frequency']} sentence(s))")
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()