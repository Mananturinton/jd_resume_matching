"""
jd.py — Job Description Knowledge Graph (JD Graph) builder.

Builds a *semantic capability graph* from a job description: not a parse
tree of the sentence, but a graph that answers "what role/capability does
this set of technologies represent?"

    Domain -> Role/Capability (root) -> Subdomain -> Tool -> Concept

Technology extraction is dictionary-based against merged_tools.json (a
~6.4K-tool taxonomy), so every matched tool already carries real-world
metadata: which engineering subdomains it shows up in, which concepts
it's associated with there, and which other tools it's related to. That
metadata is what capability inference is built on, rather than a small
hardcoded map of "framework -> role". This module is fully self-contained
— it has no dependency on any other script, only on merged_tools.json.

Design notes (read before extending the classification tables below):

  - Root inference (Full Stack vs. Frontend vs. Backend vs. a standalone
    discipline like Machine Learning Engineering) is triggered ONLY by a
    small curated set of unambiguous framework/tool names
    (FRONTEND_FRAMEWORKS, BACKEND_FRAMEWORKS, ML_TOOLS, ...). Taxonomy
    subdomain tags are deliberately NOT used for root-level triggering:
    tags like "Application Developer" and "Backend Engineering" are wide
    enough that they show up on TypeScript, React, and Python alike, and
    would flip a pure-frontend JD into "Full Stack" on tag noise alone.
    Curated sets stay small and unambiguous by construction.
  - Once roots are decided, taxonomy tags DO drive bucket placement and
    cross-listing for the long tail of tools that aren't in any curated
    set — that's where the 6.4K-tool taxonomy earns its keep.
  - A tool can legitimately appear under more than one root (PostgreSQL
    under both "Full Stack Development" and "Data Engineering" if both
    are active) per the "technologies may belong to multiple
    capabilities" rule — see _facet_set / cross-listing below.
  - The reference examples in the spec collapse "Subdomain" and
    "Technology Category" into one layer, and one example (pure
    Frontend) renders fully flat. This implementation always keeps one
    Subdomain layer under every root for schema consistency (useful for
    Neo4j / traversal / matching), so a pure-frontend JD comes out as
    Frontend/Language buckets rather than one flat list — a deliberate,
    explainable deviation from that illustration, not an oversight.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # Windows consoles default to a legacy codepage (e.g. cp1252) that
    # can't encode the box-drawing characters render_tree() prints
    # (└── / ├──), crashing before the JSON output is ever written.
    sys.stdout.reconfigure(encoding="utf-8")
from typing import Any, Dict, List, Optional, Set, Tuple

TOOLS_FILENAME = "merged_tools.json"

# --------------------------------------------------------------------------
# Curated trigger/category tables.
#
# These are the unambiguous, hand-vetted signals used for (a) deciding
# which root(s) a JD belongs under and (b) giving well-known tools a clean
# category without depending on taxonomy tag noise. Everything NOT listed
# here still gets extracted (via the taxonomy lexicon) and categorized —
# just via the taxonomy-tag fallback in _fallback_facets() instead.
# --------------------------------------------------------------------------

LANGUAGES = {
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "c", "ruby", "php", "kotlin", "swift", "scala", "html", "css", "sql",
    "r", "dart", "elixir", "haskell", "lua", "matlab", "perl", "groovy",
    "objective-c", "solidity",
}
FRONTEND_LANGS = {"typescript", "javascript", "html", "css"}
# General-purpose languages whose overwhelming real-world use is
# server-side/application backend work — the symmetric counterpart to the
# bare HTML/CSS -> Frontend trigger below. Domain-specific languages
# (R, MATLAB, Swift, Dart, Objective-C, Solidity, ...) are deliberately
# excluded: they already have better-fitting standalone root categories
# elsewhere, and including them here would make this "unambiguous backend
# signal" set neither small nor unambiguous. Without this, a JD built out
# of bare languages (Java, C#, Python, ...) with no named framework either
# way, but a passing mention of HTML/CSS, silently collapsed to "Frontend
# Development" — the Frontend trigger fired on markup alone while nothing
# symmetric could ever fire "Backend".
BACKEND_LANGS = {
    "python", "java", "c#", "go", "rust", "c++", "c", "ruby", "php",
    "kotlin", "scala",
}

DATABASES = {
    "postgresql", "mysql", "mongodb", "sqlite", "dynamodb", "oracle database",
    "microsoft sql server", "mariadb", "cockroachdb", "elasticsearch",
    "neo4j", "cassandra", "snowflake", "databricks",
}
MESSAGE_BROKERS = {
    "kafka", "rabbitmq", "amazon sqs", "activemq", "google pub/sub", "nats",
    "amazon sns",
}
CACHES = {"redis", "memcached", "hazelcast", "varnish"}
CLOUD_PROVIDERS = {
    "aws", "microsoft azure", "azure", "google cloud platform", "gcp",
    "digitalocean", "heroku", "vercel", "netlify", "cloudflare",
}
DEVOPS_TOOLS = {
    "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "github actions", "gitlab ci", "circleci", "puppet", "chef", "helm",
    "argo cd", "prometheus", "grafana", "nginx", "docker compose",
}
TESTING_TOOLS = {
    "jest", "cypress", "selenium", "pytest", "playwright", "junit", "mocha",
    "testng", "cucumber", "robot framework",
}
FRONTEND_FRAMEWORKS = {
    "react", "redux", "next.js", "vue.js", "angular", "svelte", "jquery",
    "tailwind css", "bootstrap", "material ui", "webpack", "vite",
}
BACKEND_FRAMEWORKS = {
    "express.js", "nestjs", "django", "flask", "fastapi", "spring boot",
    "node.js", "ruby on rails", "laravel", "asp.net core", "gin",
}
ML_TOOLS = {
    "tensorflow", "pytorch", "scikit-learn", "keras", "mlflow",
    "hugging face", "pandas", "numpy", "jupyter",
}
DATA_ENG_TOOLS = {
    "apache spark", "airflow", "apache airflow", "dbt",
}
SECURITY_TOOLS = {
    "nmap", "wireshark", "metasploit", "burp suite", "owasp zap", "splunk",
    "okta", "vault", "hashicorp vault", "crowdstrike", "wazuh",
}
NETWORKING_TOOLS = {"bgp"}
GAME_DEV_TOOLS = {"unity", "unreal engine", "godot"}
EMBEDDED_TOOLS = {"arduino", "raspberry pi", "freertos"}
BLOCKCHAIN_TOOLS = {"ethereum", "web3.js", "hardhat", "truffle"}

CURATED: Dict[str, str] = {}
for _name in LANGUAGES:
    CURATED[_name] = "Language"
for _name in DATABASES:
    CURATED[_name] = "Database"
for _name in MESSAGE_BROKERS:
    CURATED[_name] = "Messaging"
for _name in CACHES:
    CURATED[_name] = "Caching"
for _name in CLOUD_PROVIDERS:
    CURATED[_name] = "Cloud"
for _name in DEVOPS_TOOLS:
    CURATED[_name] = "DevOps"
for _name in TESTING_TOOLS:
    CURATED[_name] = "Testing"
for _name in FRONTEND_FRAMEWORKS:
    CURATED[_name] = "Frontend"
for _name in BACKEND_FRAMEWORKS:
    CURATED[_name] = "Backend"
for _name in ML_TOOLS:
    CURATED[_name] = "Machine Learning"
for _name in DATA_ENG_TOOLS:
    CURATED[_name] = "Data Engineering"
for _name in SECURITY_TOOLS:
    CURATED[_name] = "Security"
for _name in NETWORKING_TOOLS:
    CURATED[_name] = "Networking"
for _name in GAME_DEV_TOOLS:
    CURATED[_name] = "Game Development"
for _name in EMBEDDED_TOOLS:
    CURATED[_name] = "Embedded"
for _name in BLOCKCHAIN_TOOLS:
    CURATED[_name] = "Blockchain"

# Raw taxonomy `subdomain` tag -> normalized facet, used only as a
# fallback for tools with no curated classification. "Application
# Developer" and "Fullstack Dev" are intentionally excluded: they're
# broad, generic tags nearly every web-adjacent tool carries, and would
# otherwise blur every root decision toward "Full Stack".
SUBDOMAIN_FACET_MAP = {
    "Frontend": "Frontend",
    "UI UX Engineering": "Frontend",
    "Backend Engineering": "Backend",
    "Database Engineering": "Database",
    "Data Engineering": "Data Engineering",
    "DevOps": "DevOps",
    "SRE": "DevOps",
    "Performance & Capacity Engineering": "DevOps",
    "Cloud Engineering": "Cloud",
    "Platform Engineering": "Cloud",
    "Cloud Native Serverless Engineering": "Cloud",
    "MLOps": "Machine Learning",
    "AI Engineering": "Machine Learning",
    "ML AI Engineering": "Machine Learning",
    "Data Science": "Machine Learning",
    "Cyber Security": "Security",
    "Security Engineering": "Security",
    "QA Automation": "Testing",
    "QA Testing": "Testing",
    "Networking": "Networking",
    "Game Development": "Game Development",
    "Embedded Systems": "Embedded",
    "EMBEDDED_IOT_ENGINEERING": "Embedded",
    "Blockchain Web3": "Blockchain",
    "IT Support": "IT Support",
    "System IT": "IT Support",
}

# root_name -> facets it will absorb once triggered, and how each facet
# is relabeled as a bucket name under that specific root (a facet not
# present in the map keeps its own name as the bucket).
ROOT_FACETS: Dict[str, Set[str]] = {
    "Full Stack Development": {
        "Frontend", "Backend", "Database", "Messaging", "Caching",
        "DevOps", "Cloud", "Testing", "Language",
    },
    "Frontend Development": {
        "Frontend", "Language", "DevOps", "Cloud", "Testing",
    },
    "Backend Development": {
        "Backend", "Language", "Database", "Messaging", "Caching",
        "DevOps", "Cloud", "Testing",
    },
    "Machine Learning Engineering": {
        "Machine Learning", "Language", "Database", "DevOps", "Cloud",
    },
    "Data Engineering": {
        "Data Engineering", "Language", "Database", "Messaging", "DevOps",
        "Cloud",
    },
    "Security Engineering": {"Security", "Networking", "DevOps", "Cloud"},
    "Network Engineering": {"Networking", "Security", "DevOps"},
    "Game Development": {"Game Development", "Language"},
    "Embedded Systems Engineering": {"Embedded", "Language"},
    "Blockchain / Web3 Engineering": {"Blockchain", "Language"},
    "IT Support / Systems Administration": {
        "IT Support", "Networking", "Security",
    },
    "DevOps Engineering": {"DevOps", "Cloud"},
    "Cloud Engineering": {"Cloud", "DevOps"},
    "Database Engineering": {"Database", "Caching"},
}

# Priority order used to pick a bucket when a tool's facet set intersects
# an active root's absorbed facets in more than one place.
BUCKET_PRIORITY = [
    "Frontend", "Backend", "Machine Learning", "Data Engineering",
    "Security", "Networking", "Game Development", "Embedded", "Blockchain",
    "IT Support", "Database", "Messaging", "Caching", "DevOps", "Cloud",
    "Testing", "Language",
]


def _bucket_for_root(root_name: str, facet: str, tool_lower: str) -> str:
    if root_name == "Full Stack Development" and facet == "Language":
        return "Frontend" if tool_lower in FRONTEND_LANGS else "Backend"
    if root_name == "Backend Development" and facet == "Backend":
        return "Framework"
    return facet


# --------------------------------------------------------------------------
# Taxonomy loading + extraction
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.#+&/-][A-Za-z0-9]*)*")


class ToolTaxonomy:
    """Loads merged_tools.json and extracts canonical tool mentions from
    free text via greedy, longest-match, case-aware phrase matching.

    Runs on plain text tokens with no external NLP dependency — JD
    capability inference doesn't need a dependency parse, just reliable
    tool-name detection.
    """

    def __init__(self, path: Optional[str] = None):
        self.entries: Dict[str, dict] = {}          # canonical (lower) -> entry
        self.phrase_index: Dict[str, List[Tuple[Tuple[str, ...], str]]] = {}
        self._load(path or self._default_path())

    @staticmethod
    def _default_path() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), TOOLS_FILENAME)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            print(
                f"[jd.py] {path} not found — running with an empty tool taxonomy. "
                f"Tool extraction and capability inference will find nothing; "
                f"place merged_tools.json next to jd.py, or pass ToolTaxonomy(path=...), "
                f"to restore them.",
                file=sys.stderr,
            )
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        seen_phrases: Dict[Tuple[str, ...], str] = {}
        for entry in data:
            canonical = entry.get("tool")
            if not canonical:
                continue
            if " " in canonical and canonical.islower():
                # A handful of taxonomy entries are generic multi-word
                # concept labels filed as if they were tools ("machine
                # learning", "anomaly detection", "sentiment analysis").
                # Genuine multi-word tool/product names in this dataset
                # are consistently properly-cased, so this is a safe,
                # narrow filter — it doesn't touch legitimate lowercase
                # single-word tools (bcrypt, cors, dbt, esbuild, ...).
                continue
            self.entries[canonical.lower()] = entry
            names = [canonical] + list(entry.get("aliases") or [])
            for name in names:
                tokens = tuple(self._tokenize(name))
                if tokens:
                    seen_phrases.setdefault(tokens, canonical)

        for tokens, canonical in seen_phrases.items():
            self.phrase_index.setdefault(tokens[0], []).append((tokens, canonical))

        for candidates in self.phrase_index.values():
            candidates.sort(key=lambda c: -len(c[0]))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    @staticmethod
    def _looks_deliberately_capitalized(token: str) -> bool:
        if any(ch.isdigit() or not ch.isalpha() for ch in token):
            return True
        if token.isupper():
            return True
        return any(ch.isupper() for ch in token[1:])

    def extract(self, text: str) -> List[str]:
        """Return canonical tool names found in `text`, in order of first
        mention, deduplicated."""
        token_matches = list(_TOKEN_RE.finditer(text))
        surface = [m.group(0) for m in token_matches]
        # The token regex allows a trailing "." to support names like
        # "Node.js", which also means it swallows a sentence-ending
        # period ("AWS." -> "AWS."). No canonical tool name ends in a
        # literal period, so trailing dots are always sentence
        # punctuation and safe to strip.
        surface = [s[:-1] if s.endswith(".") else s for s in surface]
        lowered = [t.lower() for t in surface]
        n = len(surface)

        def crosses_list_boundary(start: int, span: int) -> bool:
            # A comma/semicolon/newline between two tokens means they're
            # separate list items, not one phrase - without this, a
            # taxonomy alias like "react redux" (an alias of "Redux")
            # would swallow "React, Redux, ..." as a single match and
            # silently drop React, which is common in comma-separated JD
            # skill lists.
            for k in range(start, start + span - 1):
                between = text[token_matches[k].end():token_matches[k + 1].start()]
                if any(ch in between for ch in ",;\n"):
                    return True
            return False

        found: List[str] = []
        seen: Set[str] = set()
        i = 0
        while i < n:
            candidates = self.phrase_index.get(lowered[i])
            matched = False
            if candidates:
                for tokens, canonical in candidates:
                    span = len(tokens)
                    if i + span > n or tuple(lowered[i:i + span]) != tokens:
                        continue
                    if span == 1 and surface[i].islower():
                        # Bare lowercase single-token alias ("go", "r", "ad")
                        # is too ambiguous with ordinary English words.
                        continue
                    if span > 1 and crosses_list_boundary(i, span):
                        continue
                    if canonical not in seen:
                        seen.add(canonical)
                        found.append(canonical)
                    i += span
                    matched = True
                    break
            if not matched:
                i += 1
        return found


# --------------------------------------------------------------------------
# Facet classification
# --------------------------------------------------------------------------

def _fallback_facets(entry: dict) -> Set[str]:
    """Facets derived from taxonomy subdomain tags, for tools with no
    curated classification (or for cross-listing purposes on top of a
    curated one)."""
    raw = entry.get("subdomain") or []
    if isinstance(raw, str):
        raw = [raw]
    facets = set()
    for tag in raw:
        facet = SUBDOMAIN_FACET_MAP.get(tag)
        if facet:
            facets.add(facet)
    return facets


def _curated_lookup(low: str) -> Optional[str]:
    """Curated-set lookup with a fallback for "Apache X" / "X" duplicate
    taxonomy entries (e.g. "Kafka" and "Apache Kafka" both exist as
    separate canonical tools that alias each other) — greedy alias
    matching can surface either form depending on JSON entry order, and
    both should classify the same way."""
    if low in CURATED:
        return CURATED[low]
    if low.startswith("apache "):
        return CURATED.get(low[len("apache "):])
    return None


def _facet_set(tool: str, entry: Optional[dict]) -> Tuple[str, Set[str], bool]:
    """Returns (primary_facet, full_facet_set, is_curated) for a tool.

    primary_facet drives the tool's natural single-category label and
    whether a rationale is worth generating; full_facet_set drives which
    additional active roots the tool cross-lists into.
    """
    low = tool.lower()
    curated = _curated_lookup(low)
    fallback = _fallback_facets(entry) if entry else set()
    if curated:
        return curated, ({curated} | fallback), True
    if fallback:
        # Deterministic primary pick: highest-priority facet present.
        primary = next((f for f in BUCKET_PRIORITY if f in fallback), next(iter(fallback)))
        return primary, fallback, False
    return "Tool", set(), False


def _trigger_facets(tools: List[str], taxonomy: ToolTaxonomy) -> Dict[str, bool]:
    """Which root-trigger signals are present, using ONLY curated
    memberships (or, for tools outside every curated set, their taxonomy
    fallback facet) — see module docstring for why tags like "Backend
    Engineering" don't get to trigger roots on their own."""
    triggers = defaultdict(bool)
    for tool in tools:
        low = tool.lower()
        if low in {"html", "css"}:
            # Bare markup/styling languages are a frontend signal on
            # their own, even though they classify as "Language".
            triggers["Frontend"] = True
        if low in BACKEND_LANGS:
            # Symmetric case: bare general-purpose backend languages are
            # a backend signal on their own, even though they too
            # classify as "Language". See BACKEND_LANGS for why.
            triggers["Backend"] = True

        curated = _curated_lookup(low)
        if curated:
            triggers[curated] = True
            continue

        entry = taxonomy.entries.get(low)
        if entry:
            for facet in _fallback_facets(entry):
                if facet in (
                    "Machine Learning", "Data Engineering", "Security",
                    "Networking", "Game Development", "Embedded",
                    "Blockchain", "IT Support",
                ):
                    triggers[facet] = True
    return triggers


def _decide_roots(triggers: Dict[str, bool]) -> List[str]:
    roots: List[str] = []
    has_fe, has_be = triggers.get("Frontend"), triggers.get("Backend")
    app_root = None
    if has_fe and has_be:
        app_root = "Full Stack Development"
    elif has_fe:
        app_root = "Frontend Development"
    elif has_be:
        app_root = "Backend Development"
    if app_root:
        roots.append(app_root)

    standalone_map = {
        "Machine Learning": "Machine Learning Engineering",
        "Data Engineering": "Data Engineering",
        "Security": "Security Engineering",
        "Networking": "Network Engineering",
        "Game Development": "Game Development",
        "Embedded": "Embedded Systems Engineering",
        "Blockchain": "Blockchain / Web3 Engineering",
        "IT Support": "IT Support / Systems Administration",
    }
    for facet, root_name in standalone_map.items():
        if triggers.get(facet):
            roots.append(root_name)

    # No app-layer or discipline-specific trigger at all: fall back to
    # whatever infra signal exists so the JD still gets a sensible root
    # instead of an empty graph.
    if not roots:
        if triggers.get("DevOps"):
            roots.append("DevOps Engineering")
        elif triggers.get("Cloud"):
            roots.append("Cloud Engineering")
        elif triggers.get("Database"):
            roots.append("Database Engineering")

    return roots


ROOT_RATIONALES = {
    "Full Stack Development": (
        "Both frontend technologies ({fe}) and backend technologies "
        "({be}) are present, so the JD spans the whole application stack "
        "rather than one layer of it."
    ),
    "Frontend Development": (
        "Only client-side/UI technologies ({fe}) are present, with no "
        "backend framework or general-purpose backend language mentioned "
        "— the role is scoped to the frontend."
    ),
    "Backend Development": (
        "Backend technologies ({be}) are present with no frontend "
        "framework/UI library — the role is scoped to server-side "
        "development."
    ),
}


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def build_jd_graph(jd_text: str, taxonomy: Optional[ToolTaxonomy] = None) -> Dict[str, Any]:
    taxonomy = taxonomy or ToolTaxonomy()
    tools = taxonomy.extract(jd_text)
    return _build_graph(tools, taxonomy)


# Tier ranking for combined_classification.json's "tier" field, used to
# keep the *strongest* tier when a tool is corroborated by more than one
# sub_requirement.
_TIER_RANK = {"CRITICAL": 3, "IMPORTANT": 2, "GENERIC": 1, "NON_MATCHABLE": 0}


def _parse_pct(value: Any) -> Optional[float]:
    """sentence_importance_analysis reports sentence-level importance as a
    string ('12%'); sub_requirement-level importance is already a plain
    number. Normalize both call sites to a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None


def extract_from_classification(
    classification: Dict[str, Any], taxonomy: Optional[ToolTaxonomy] = None
) -> Tuple[List[str], Dict[str, dict]]:
    """Extract canonical tools from a combined_classification.json-shaped
    JD analysis (tiered sub_requirements with match_keywords, plus a
    top-level extracted_skills list), instead of raw JD text.

    This is a richer signal than free text: match_keywords are already
    the analyst's/model's own pick of the concrete terms that matter, and
    each sub_requirement carries a tier (CRITICAL/IMPORTANT/GENERIC) and
    a dominant_role. Returns (ordered canonical tool list, tool name ->
    provenance dict) so that tier/importance/role can ride along onto the
    graph's Tool nodes for downstream candidate ranking.
    """
    taxonomy = taxonomy or ToolTaxonomy()

    importance_by_subid: Dict[str, Any] = {}
    for sentence in (classification.get("sentence_importance_analysis") or {}).get("sentences") or []:
        for sub in sentence.get("sub_requirement_importance") or []:
            if sub.get("sub_id"):
                importance_by_subid[sub["sub_id"]] = sub.get("importance_percentage")

    provenance: Dict[str, dict] = {}
    order: List[str] = []

    def record(tool: str, tier: Optional[str], importance: Optional[float],
               role: Optional[str], section: Optional[str]) -> None:
        if tool not in provenance:
            provenance[tool] = {
                "tier": tier, "importance_percentage": importance,
                "dominant_role": role, "section": section,
            }
            order.append(tool)
            return
        p = provenance[tool]
        if _TIER_RANK.get(tier, -1) > _TIER_RANK.get(p["tier"], -1):
            p["tier"] = tier
        if importance is not None and (p["importance_percentage"] is None or importance > p["importance_percentage"]):
            p["importance_percentage"] = importance
        if role and not p["dominant_role"]:
            p["dominant_role"] = role

    for sentence in classification.get("processed_sentences") or []:
        tier = sentence.get("tier")
        role = sentence.get("dominant_role")
        section = sentence.get("section")
        for sub in sentence.get("sub_requirements") or []:
            if not sub.get("matchable", True):
                continue
            blob = ", ".join(sub.get("match_keywords") or [])
            for tool in taxonomy.extract(blob):
                record(tool, tier, importance_by_subid.get(sub.get("sub_id")), role, section)

    # Supplemental pass over the flat extracted_skills list, for any term
    # a model surfaced there but didn't break out into a sub_requirement.
    skills_blob = ", ".join(classification.get("extracted_skills") or [])
    for tool in taxonomy.extract(skills_blob):
        record(tool, None, None, None, "extracted_skills")

    return order, provenance


def build_jd_graph_from_classification(
    classification: Dict[str, Any], taxonomy: Optional[ToolTaxonomy] = None
) -> Dict[str, Any]:
    """Build a JD graph from a combined_classification.json-shaped JD
    analysis instead of raw text. See extract_from_classification for how
    tools are pulled out of it."""
    taxonomy = taxonomy or ToolTaxonomy()
    tools, provenance = extract_from_classification(classification, taxonomy)
    result = _build_graph(tools, taxonomy, provenance)
    result["source_metadata"] = {
        "predicted_role": classification.get("predicted_role"),
        "job_role_domain": classification.get("job_role_domain"),
        "chatgpt_roles": classification.get("chatgpt_roles"),
        "job_logistics": classification.get("job_logistics"),
    }
    return result


def _build_graph(
    tools: List[str], taxonomy: ToolTaxonomy, provenance: Optional[Dict[str, dict]] = None
) -> Dict[str, Any]:
    if not tools:
        return {
            "tree": [],
            "graph": {"nodes": [], "relationships": []},
            "extracted_tools": [],
            "note": "No known technologies were recognized in this input.",
        }

    provenance = provenance or {}
    entries = {t: taxonomy.entries.get(t.lower()) for t in tools}
    classifications = {t: _facet_set(t, entries[t]) for t in tools}
    triggers = _trigger_facets(tools, taxonomy)
    root_names = _decide_roots(triggers)

    fe_tools = [t for t in tools if t.lower() in FRONTEND_FRAMEWORKS or t.lower() in {"html", "css"}]
    be_tools = [t for t in tools if t.lower() in BACKEND_FRAMEWORKS or t.lower() in BACKEND_LANGS]

    # root_name -> bucket_name -> list of tool dicts
    roots: Dict[str, Dict[str, List[dict]]] = {r: defaultdict(list) for r in root_names}

    for tool in tools:
        primary_facet, facet_set, is_curated = classifications[tool]
        entry = entries[tool]
        placed_in = []
        for root_name in root_names:
            absorbed = ROOT_FACETS.get(root_name, set())
            # The tool's own curated/primary facet wins whenever it applies
            # to this root — taxonomy fallback facets (e.g. PostgreSQL also
            # being tagged "Backend Engineering") should only decide
            # placement when the primary facet itself isn't relevant here,
            # otherwise a curated Database tool could get bucketed as
            # Backend on tag noise alone.
            if primary_facet in absorbed:
                chosen_facet = primary_facet
            else:
                hit = [f for f in BUCKET_PRIORITY if f in facet_set and f in absorbed]
                if not hit:
                    continue
                chosen_facet = hit[0]
            bucket = _bucket_for_root(root_name, chosen_facet, tool.lower())
            placed_in.append(root_name)
            roots[root_name][bucket].append(
                _tool_node(tool, entry, chosen_facet, is_curated, bucket, root_name, provenance.get(tool))
            )

        if not placed_in and root_names:
            # Long-tail tool with no facet match at all (e.g. a bare
            # "Tool" classification): attach it to the first root under a
            # generic bucket rather than silently dropping it.
            root_name = root_names[0]
            roots[root_name]["Other"].append(
                _tool_node(tool, entry, "Tool", is_curated, "Other", root_name, provenance.get(tool))
            )

    tree = []
    for root_name in root_names:
        node = {
            "name": root_name,
            "type": "Role/Capability",
            "domain": "Engineering",
            "subdomains": [
                {"name": bucket, "type": "Subdomain", "tools": tool_nodes}
                for bucket, tool_nodes in sorted(roots[root_name].items())
            ],
        }
        rationale_tmpl = ROOT_RATIONALES.get(root_name)
        if rationale_tmpl:
            node["rationale"] = rationale_tmpl.format(
                fe=", ".join(fe_tools) or "n/a", be=", ".join(be_tools) or "n/a"
            )
        else:
            trigger_tools = [
                t for t in tools
                if classifications[t][1] & ROOT_FACETS.get(root_name, set())
            ]
            node["rationale"] = (
                f"Inferred as a standalone capability because dedicated "
                f"{root_name}-specific tooling was found ({', '.join(trigger_tools[:6])})."
            )
        tree.append(node)

    graph = _to_graph(tree)
    return {
        "tree": tree,
        "graph": graph,
        "extracted_tools": tools,
    }


def _tool_node(tool: str, entry: Optional[dict], facet: str, is_curated: bool,
               bucket: str, root_name: str, provenance: Optional[dict] = None) -> dict:
    concepts: List[str] = []
    related: List[Tuple[str, float]] = []
    if entry:
        raw_sub = entry.get("subdomain") or []
        if isinstance(raw_sub, str):
            raw_sub = [raw_sub]
        sub_data = entry.get("subdomain_data") or {}
        # Pull concepts/related tools from whichever raw taxonomy tag(s)
        # map onto this tool's chosen facet, so the concepts shown match
        # the bucket the tool was actually placed in.
        for tag in raw_sub:
            if SUBDOMAIN_FACET_MAP.get(tag) == facet or (is_curated and tag in sub_data):
                data = sub_data.get(tag) or {}
                for c in data.get("concepts") or []:
                    if c not in concepts:
                        concepts.append(c)
                for rt in data.get("related_tools") or []:
                    related.append((rt.get("tool"), rt.get("weight", 0.0)))
        if not concepts:
            # Curated tools may not have a taxonomy tag equal to their
            # curated facet name (e.g. "DevOps" facet vs. tag "DevOps") —
            # fall back to whatever the entry has.
            for tag, data in sub_data.items():
                for c in data.get("concepts") or []:
                    if c not in concepts:
                        concepts.append(c)

    node = {
        "name": tool,
        "kind": bucket,
        "concepts": concepts[:6],
        "related_tools": [r for r, _ in sorted(related, key=lambda x: -x[1])[:5]],
    }
    if provenance:
        if provenance.get("tier"):
            node["tier"] = provenance["tier"]
        if provenance.get("importance_percentage") is not None:
            node["importance_percentage"] = provenance["importance_percentage"]
        if provenance.get("dominant_role"):
            node["dominant_role"] = provenance["dominant_role"]
    if not is_curated:
        raw_tags = entry.get("subdomain") if entry else None
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        node["rationale"] = (
            f"Placed under {root_name} > {bucket} based on taxonomy tag(s): "
            f"{', '.join(raw_tags or [])}."
        )
    return node


def _to_graph(tree: List[dict]) -> Dict[str, List[dict]]:
    nodes: Dict[str, dict] = {}
    rels: List[dict] = []
    tool_present: Set[str] = set()

    def add_node(node_id: str, label: str, props: dict):
        nodes.setdefault(node_id, {"id": node_id, "label": label, "properties": props})

    domain_id = "domain::Engineering"
    add_node(domain_id, "Domain", {"name": "Engineering"})

    for root in tree:
        root_id = f"root::{root['name']}"
        add_node(root_id, "Capability", {"name": root["name"], "rationale": root.get("rationale", "")})
        rels.append({"source": domain_id, "target": root_id, "type": "HAS_CAPABILITY"})

        for sub in root["subdomains"]:
            sub_id = f"subdomain::{root['name']}::{sub['name']}"
            add_node(sub_id, "Subdomain", {"name": sub["name"]})
            rels.append({"source": root_id, "target": sub_id, "type": "HAS_SUBDOMAIN"})

            for tool in sub["tools"]:
                tool_id = f"tool::{tool['name']}"
                props = {"name": tool["name"], "kind": tool["kind"]}
                for key in ("tier", "importance_percentage", "dominant_role"):
                    if key in tool:
                        props[key] = tool[key]
                add_node(tool_id, "Tool", props)
                rels.append({"source": sub_id, "target": tool_id, "type": "INCLUDES"})
                tool_present.add(tool["name"])

                for concept in tool["concepts"]:
                    concept_id = f"concept::{concept}"
                    add_node(concept_id, "Concept", {"name": concept})
                    rels.append({"source": tool_id, "target": concept_id, "type": "DEMONSTRATES_CONCEPT"})

    # Only link RELATED_TO between tools that both actually appear in
    # this JD's graph — otherwise every tool would pull in dozens of
    # taxonomy neighbors that were never mentioned.
    for root in tree:
        for sub in root["subdomains"]:
            for tool in sub["tools"]:
                for related in tool["related_tools"]:
                    if related in tool_present and related != tool["name"]:
                        rels.append({
                            "source": f"tool::{tool['name']}",
                            "target": f"tool::{related}",
                            "type": "RELATED_TO",
                        })

    return {"nodes": list(nodes.values()), "relationships": rels}


# --------------------------------------------------------------------------
# Human-readable rendering
# --------------------------------------------------------------------------

def render_tree(tree: List[dict]) -> str:
    lines = ["Engineering"]
    for ri, root in enumerate(tree):
        last_root = ri == len(tree) - 1
        lines.append(f"{'└── ' if last_root else '├── '}{root['name']}")
        prefix = "    " if last_root else "│   "
        if root.get("rationale"):
            lines.append(f"{prefix}    ({root['rationale']})")
        subs = root["subdomains"]
        for si, sub in enumerate(subs):
            last_sub = si == len(subs) - 1
            lines.append(f"{prefix}{'└── ' if last_sub else '├── '}{sub['name']}")
            sub_prefix = prefix + ("    " if last_sub else "│   ")
            for ti, tool in enumerate(sub["tools"]):
                last_tool = ti == len(sub["tools"]) - 1
                label = tool["name"]
                tags = []
                if tool.get("tier"):
                    tags.append(tool["tier"])
                if tool.get("importance_percentage") is not None:
                    tags.append(f"{tool['importance_percentage']}%")
                if tags:
                    label += f" [{', '.join(str(t) for t in tags)}]"
                lines.append(f"{sub_prefix}{'└── ' if last_tool else '├── '}{label}")
                if tool.get("rationale"):
                    tool_prefix = sub_prefix + ("    " if last_tool else "│   ")
                    lines.append(f"{tool_prefix}  ↳ {tool['rationale']}")
    return "\n".join(lines)


def compact_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """The smallest JSON shape that still carries every fact the tree
    itself doesn't already encode.

    The default `{"tree": ..., "graph": {"nodes": ..., "relationships": ...}}`
    is Neo4j-ingestion-shaped: every tool/concept is re-serialized a second
    time as graph nodes plus one relationship object per edge, which roughly
    quadruples the size of the tree for no benefit outside an actual graph
    import. On top of dropping that "graph" block (it's mechanically
    derivable from the tree via _to_graph if ever needed), this also drops:

      - `extracted_tools`: just the flattened set of every tool name
        already sitting under tree[*].subdomains[*] — zero new information.
      - `job_logistics` keys that are null (years_of_experience,
        work_mode, ... are null on most JDs; only the ones an analyst
        actually filled in are kept), and the whole key if nothing in it
        survived that.

    What's left: capability/rationale/subdomains, each tool's
    name/tier/importance/dominant_role, and predicted_role/domain/
    chatgpt_roles/non-null job_logistics as top-level context.
    """
    tree = []
    for root in result["tree"]:
        tree.append({
            "capability": root["name"],
            "rationale": root.get("rationale"),
            "subdomains": {
                sub["name"]: [
                    {
                        "name": t["name"],
                        "tier": t.get("tier"),
                        "importance_percentage": t.get("importance_percentage"),
                        "dominant_role": t.get("dominant_role"),
                    }
                    for t in sub["tools"]
                ]
                for sub in root["subdomains"]
            },
        })
    out = {"tree": tree}

    meta = result.get("source_metadata")
    if meta:
        out_meta = {k: v for k, v in meta.items() if k != "job_logistics" and v is not None}
        logistics = {k: v for k, v in (meta.get("job_logistics") or {}).items() if v is not None}
        if logistics:
            out_meta["job_logistics"] = logistics
        out["source_metadata"] = out_meta

    return out


def _actions_by_sentence(classification: Dict[str, Any]) -> Dict[int, List[dict]]:
    """Groups the classifier's responsibility_actions (action verbs + their
    direct-object target, e.g. {"actions": ["Build", "maintain"], "target":
    "data pipelines"}) by the sentence_id they were extracted from."""
    by_sentence: Dict[int, List[dict]] = defaultdict(list)
    for entry in classification.get("responsibility_actions") or []:
        sid = entry.get("sentence_id")
        if sid is None:
            continue
        by_sentence[sid].append({
            "actions": entry.get("actions") or [],
            "target": entry.get("target"),
        })
    return by_sentence


def unified_graph_json(classification: Dict[str, Any], result: Dict[str, Any],
                        taxonomy: Optional[ToolTaxonomy] = None) -> Dict[str, Any]:
    """The same role -> sentence -> sub_requirement -> tool graph that
    generate_dependency_trees.py renders as dependency_graph_unified.png,
    as JSON instead of a picture.

    Every node traces back to a specific field in combined_classification.json
    (nothing is asserted without a visible source): roles from chatgpt_roles/
    testlify_analysis, sentences from processed_sentences with their tier and
    role_attribution, sub_requirements with their sentence_importance_analysis
    percentage, tools recognized (via the same ToolTaxonomy used everywhere
    else in this module) inside each sub_requirement's match_keywords, and
    actions from responsibility_actions (each sentence's action verb(s) +
    the target they act on). tool_relations is the taxonomy's RELATED_TO
    layer, restricted to tools that actually appear in this JD — same
    restriction _build_graph already applies.
    """
    taxonomy = taxonomy or ToolTaxonomy()
    actions_by_sentence = _actions_by_sentence(classification)

    tool_props: Dict[str, dict] = {}
    for n in result["graph"]["nodes"]:
        if n["label"] == "Tool":
            tool_props[n["properties"]["name"]] = n["properties"]

    importance_by_subid: Dict[str, Any] = {}
    importance_by_sentence_id: Dict[int, float] = {}
    for sentence in (classification.get("sentence_importance_analysis") or {}).get("sentences") or []:
        sid = sentence.get("sentence_id")
        if sid is not None:
            importance_by_sentence_id[sid] = _parse_pct(sentence.get("importance_percentage"))
        for sub in sentence.get("sub_requirement_importance") or []:
            importance_by_subid[sub["sub_id"]] = sub.get("importance_percentage")

    # roles_ranked carries each role's own concept vocabulary (e.g. "Manual
    # testing" 95%, "Defect tracking" 80%) — the classifier's direct read of
    # what matters for that role, independent of whether any of those
    # concepts happen to also be a recognized tool name in merged_tools.json.
    # Most QA/analyst-style JDs are activities and domains, not named tools,
    # so this is often the richer signal; carried through here so a matcher
    # can score it (previously computed but dropped on the floor).
    roles_ranked = {
        r["role"]: r for r in (classification.get("testlify_analysis") or {}).get("roles_ranked") or []
    }
    roles = [
        {
            "name": role,
            "importance_pct": (roles_ranked.get(role) or {}).get("role_importance_pct"),
            "concepts": (roles_ranked.get(role) or {}).get("concepts") or [],
        }
        for role in (classification.get("chatgpt_roles") or [])
    ]

    tool_found_in: Dict[str, List[str]] = {}
    sentences = []
    for sentence in classification.get("processed_sentences") or []:
        subs = []
        for sub in sentence.get("sub_requirements") or []:
            sub_id = sub["sub_id"]
            tools_here = []
            if sub.get("matchable", True):
                blob = ", ".join(sub.get("match_keywords") or [])
                tools_here = taxonomy.extract(blob)
                for t in tools_here:
                    tool_found_in.setdefault(t, []).append(sub_id)
            subs.append({
                "sub_id": sub_id,
                "text": sub["text"],
                "importance_percentage": importance_by_subid.get(sub_id),
                "tools": tools_here,
            })
        sentence_actions = [
            {"id": f"a{sentence['id']}-{i}", "actions": act["actions"], "target": act["target"]}
            for i, act in enumerate(actions_by_sentence.get(sentence["id"], []))
        ]
        sentences.append({
            "id": sentence["id"],
            "text": sentence["text"],
            "tier": sentence.get("tier"),
            "importance_percentage": importance_by_sentence_id.get(sentence["id"]),
            "expected_depth": sentence.get("expected_depth"),
            "dominant_role": sentence.get("dominant_role"),
            "role_attribution": sentence.get("role_attribution"),
            "sub_requirements": subs,
            "actions": sentence_actions,
        })

    actions = [
        {**act, "sentence_id": sentence["id"]}
        for sentence in sentences
        for act in sentence["actions"]
    ]

    tools = [
        {
            "name": name,
            "tier": props.get("tier"),
            "importance_percentage": props.get("importance_percentage"),
            "dominant_role": props.get("dominant_role"),
            "found_in": tool_found_in.get(name, []),
        }
        for name, props in tool_props.items()
    ]

    seen = set()
    tool_relations = []
    for r in result["graph"]["relationships"]:
        if r["type"] != "RELATED_TO":
            continue
        src = r["source"].replace("tool::", "")
        tgt = r["target"].replace("tool::", "")
        if src not in tool_props or tgt not in tool_props:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        tool_relations.append({"source": src, "target": tgt})

    return {
        "roles": roles,
        "sentences": sentences,
        "tools": tools,
        "tool_relations": tool_relations,
        "actions": actions,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a JD Knowledge Graph from a job description.")
    parser.add_argument("text", nargs="?", help="Job description text (or omit and use --file / --classification / stdin)")
    parser.add_argument("--file", "-f", help="Read raw JD text from a file")
    parser.add_argument(
        "--classification", "-c",
        help="Read a combined_classification.json-shaped JD analysis instead of raw text "
             "(uses its match_keywords/extracted_skills, and carries tier/importance/role onto the graph)",
    )
    parser.add_argument(
        "--json-out", "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.json"),
        help="Write the machine-readable JSON to this path (default: output.json "
             "next to jd.py)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Print/write the full Neo4j-shaped tree+graph output (every tool/concept "
             "re-serialized as graph nodes and relationship edges) instead of the "
             "compact default (tree only: capability/rationale/subdomains/tier/"
             "importance/role). Only needed for actual graph-DB ingestion.",
    )
    parser.add_argument("--compact", action="store_true", help=argparse.SUPPRESS)  # now the default; kept as a no-op for old scripts/muscle memory
    args = parser.parse_args()

    default_classification = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "combined_classification.json"
    )
    if not args.classification and not args.file and not args.text \
            and os.path.exists(default_classification):
        # Nothing was passed at all and combined_classification.json sits
        # right next to this script — use it rather than blocking on
        # stdin (which may not even be a real TTY, e.g. under a
        # non-interactive shell) or dumping --help at someone who just
        # wants the graph.
        args.classification = default_classification
        print(f"(no input given — using {default_classification})\n")

    classification = None
    if args.classification:
        with open(args.classification, "r", encoding="utf-8") as f:
            classification = json.load(f)
        result = build_jd_graph_from_classification(classification)
    else:
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                jd_text = f.read()
        elif args.text:
            jd_text = args.text
        elif not sys.stdin.isatty():
            jd_text = sys.stdin.read()
        else:
            # No text/--file/--classification given, stdin isn't piped,
            # and there's no default classification file to fall back
            # to: reading stdin here would just block silently waiting
            # for input the user never sends. Show usage instead.
            parser.print_help()
            return
        result = build_jd_graph(jd_text)

    if classification is None and not result["tree"]:
        # No capability tree AND no classification (sentence/role) data to
        # fall back to — raw-text mode with zero recognized tools truly has
        # nothing left to show.
        print(result.get("note", "No graph produced."))
        return

    if result["tree"]:
        print(render_tree(result["tree"]))
        print()
        print(f"Extracted technologies: {', '.join(result['extracted_tools'])}")
    else:
        # The tool taxonomy matched nothing (or next to nothing) in this
        # JD — common for non-software domains (process/chemical/mechanical
        # engineering, etc.) since merged_tools.json is software-tooling-
        # flavored. The capability tree has nothing to show, but
        # combined_classification.json's own sentence/role/sub_requirement
        # structure below is still real, non-empty output.
        print("(No known technologies matched the tool taxonomy for this JD — "
              "showing the role/sentence/sub-requirement structure below instead.)")
    if result.get("source_metadata"):
        meta = result["source_metadata"]
        print(f"Source classifier's predicted role: {meta.get('predicted_role')} "
              f"(domain: {meta.get('job_role_domain')})")

    if args.full:
        output = {"tree": result["tree"], "graph": result["graph"]}
        if result.get("source_metadata"):
            output["source_metadata"] = result["source_metadata"]
    elif classification is not None:
        # Role -> sentence -> sub_requirement -> tool, same shape as
        # dependency_graph_unified.png — only possible when the input was
        # a combined_classification.json (raw JD text has no sentence/
        # sub_requirement/role_attribution structure to build this from).
        output = unified_graph_json(classification, result)
    else:
        output = compact_output(result)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON graph written to {args.json_out}")
    else:
        print()
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()