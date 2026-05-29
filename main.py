import os
import re
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple

from mistralai.client import Mistral

# ===== API KEY =====
API_KEY = os.getenv("MISTRAL_API_KEY", "")
if not API_KEY:
    print("ERROR: No API key. Set MISTRAL_API_KEY environment variable.")
    exit(1)

print(f"Mistral API Key loaded: {API_KEY[:8]}...")
client = Mistral(api_key=API_KEY)

app = FastAPI(title="Academic Humanizer — StealthMode")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELS =====
class SentenceData(BaseModel):
    id: str
    original: str
    humanized: str
    alternatives: List[str]
    score: float

class ParagraphData(BaseModel):
    id: str
    sentences: List[SentenceData]

class ProcessRequest(BaseModel):
    text: str
    style: Optional[str] = "academic"
    mode: Optional[str] = "stealth"   # "stealth" | "humanize"

class ProcessResponse(BaseModel):
    processed_paragraphs: List[ParagraphData]
    total_sentences: int
    avg_score: float


# ===========================================================================
# ===== AI-TELL DETECTION SETS ==============================================
# ===========================================================================

TRANSITIONAL_OPENERS = {
    "furthermore", "moreover", "however", "therefore", "thus", "consequently",
    "additionally", "crucially", "subsequently", "nevertheless",
    "notwithstanding", "accordingly", "henceforth", "heretofore",
}

AI_TELL_PHRASES = {
    "delve", "testament", "pivotal", "moreover", "furthermore",
    "it is important to note", "it is crucial to note", "in conclusion",
    "landscape", "tapestry", "beacon", "underscore", "shed light on", "navigate",
    "ever-evolving", "multifaceted", "intricate", "robust", "holistic",
    "leverage", "synergy", "stakeholder", "crucially", "underscoring",
    "it is worth noting", "as mentioned earlier", "it should be noted",
    "indeed", "arguably", "significantly", "in addition", "it is essential",
    "plays a crucial role", "plays a vital role", "plays a key role",
    "it is well established", "it has been shown", "needless to say",
    "this is particularly important", "this is especially true",
    "in this regard", "with respect to", "with regard to",
    "it can be seen that", "it is clear that", "it is evident that",
    "as such", "thus far", "operating continuously", "it should be noted that",
    "it is noteworthy", "one must consider", "it is imperative",
}

ANCHOR_NOUNS = {
    "cerebellum", "medulla", "pons", "cortex", "tracts", "nuclei", "nerves",
    "arteries", "structure", "organ", "system", "pathway", "mechanism",
    "framework", "apparatus", "substrate", "topology", "interface", "nucleus",
    "model", "theory", "argument", "concept", "process", "method", "approach",
    "variable", "factor", "element", "component", "dimension", "aspect",
    "institution", "policy", "context", "principle", "evidence", "data",
    "analysis", "result", "finding", "outcome",
}


# ===========================================================================
# ===== DOMAIN DETECTION ====================================================
# ===========================================================================

_DOMAIN_KEYWORDS: dict = {
    "bio": [
        "cell", "cells", "neural", "neuron", "neurons", "brain", "cortex", "body",
        "physiological", "anatomy", "anatomical", "organ", "tissue", "gene", "genetic",
        "protein", "enzyme", "receptor", "synapse", "axon", "dendrite", "hormone",
        "metabolism", "homeostasis", "pathology", "clinical", "medical", "patient",
        "disease", "diagnosis", "treatment", "immune", "blood", "cardiac",
        "cerebellum", "medulla", "pons", "cortical", "nucleus", "nuclei", "tract",
        "tracts", "nerve", "nerves", "spinal", "vascular", "muscle", "cellular",
        "molecular", "biochemical", "pharmacological", "therapeutic",
    ],
    "tech": [
        "data", "dataset", "model", "models", "algorithm", "algorithms",
        "software", "code", "program", "function", "parameter", "machine learning",
        "deep learning", "artificial intelligence", "database", "server", "api",
        "protocol", "optimization", "compiler", "hardware", "processor", "storage",
        "simulation", "signal", "circuit", "matrix", "vector", "tensor", "gradient",
        "classification", "regression", "pipeline", "deployment", "architecture",
    ],
    "humanities": [
        "poem", "poetry", "novel", "narrative", "text", "texts", "author",
        "literature", "literary", "history", "historical", "culture", "cultural",
        "philosophy", "philosophical", "theory", "theoretical", "argument",
        "discourse", "ideology", "hermeneutics", "epistemology", "ethics",
        "aesthetic", "canon", "genre", "rhetoric", "metaphor", "symbol",
        "interpretation", "archive", "manuscript", "tradition", "mythology",
        "religion", "identity", "representation", "colonialism", "modernity",
    ],
    "social": [
        "society", "social", "policy", "policies", "government", "political",
        "economics", "economic", "market", "markets", "behavior", "behaviour",
        "psychology", "psychological", "sociology", "population", "survey",
        "interview", "participant", "sample", "demographic", "inequality",
        "poverty", "wealth", "income", "race", "gender", "class", "community",
        "institution", "attitude", "perception", "norm", "election", "democracy",
        "governance", "legislation", "welfare", "employment", "regression",
        "coefficient", "correlation", "p-value",
    ],
    "natural": [
        "climate", "environment", "ecology", "species", "biodiversity", "habitat",
        "ecosystem", "evolution", "chemistry", "chemical", "reaction", "compound",
        "molecule", "atom", "element", "quantum", "physics", "energy",
        "thermodynamics", "entropy", "kinetics", "geology", "sediment", "oceanic",
        "atmospheric", "carbon", "nitrogen", "oxygen", "hydrogen", "temperature",
        "pressure", "radiation", "electromagnetic", "electron", "proton",
        "isotope", "catalyst", "concentration",
    ],
    "education": [
        "student", "students", "teacher", "teachers", "classroom", "curriculum",
        "learning", "learner", "pedagogy", "instruction", "assessment", "literacy",
        "school", "university", "college", "course", "lesson", "scaffolding",
        "feedback", "motivation", "engagement", "cognition", "cognitive",
        "metacognition", "constructivism",
    ],
    "law": [
        "law", "legal", "court", "courts", "judge", "judicial", "statute",
        "legislation", "regulation", "contract", "liability", "tort", "plaintiff",
        "defendant", "jurisdiction", "precedent", "constitutional", "rights",
        "criminal", "civil", "procedure", "evidence", "verdict", "appeal",
        "compliance", "enforcement", "jurisprudence", "doctrine", "equity",
    ],
    "business": [
        "business", "firm", "company", "profit", "revenue", "cost", "investment",
        "investor", "finance", "financial", "accounting", "audit", "budget",
        "strategy", "management", "leadership", "logistics", "operations",
        "performance", "shareholder", "equity", "asset", "liability", "competition",
        "industry", "sector", "gdp", "inflation", "monetary", "trade",
    ],
}


def _detect_domain(text: str) -> str:
    t      = text.lower()
    scores = {d: 0 for d in _DOMAIN_KEYWORDS}
    for domain, kws in _DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[domain] += 1
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] >= 2 else "universal"


# ===========================================================================
# ===== FILLER BANKS — last-resort stub repair only =========================
# ===========================================================================

_FILLER_MAP: dict = {
    "bio": [
        "through integrated feedback loops",
        "under homeostatic regulation",
        "via ascending somatosensory relays",
        "contingent on afferent signal integrity",
        "within tightly regulated homeostatic bounds",
        "under conditions of normal physiological demand",
        "through coordinated synaptic transmission",
        "through receptor-mediated signal transduction",
    ],
    "tech": [
        "within tightly constrained computational parameters",
        "contingent on data stream integrity",
        "via iterative algorithmic refinement",
        "through layered abstraction hierarchies",
        "under defined boundary conditions",
        "through recursive logical decomposition",
        "contingent on model convergence criteria",
    ],
    "humanities": [
        "within prevailing theoretical frameworks",
        "through established socio-cultural paradigms",
        "contingent on historical contextual variables",
        "through critical hermeneutic engagement",
        "within historically situated interpretive horizons",
        "through comparative textual analysis",
    ],
    "social": [
        "through established socio-institutional mechanisms",
        "within prevailing policy frameworks",
        "contingent on baseline contextual variables",
        "under conditions of institutional constraint",
        "within normative regulatory environments",
        "through mediating psychosocial pathways",
    ],
    "natural": [
        "through thermodynamically driven processes",
        "via molecular diffusion gradients",
        "under equilibrium state conditions",
        "through biogeochemical cycling pathways",
        "within energetically bounded system states",
        "under standard temperature and pressure conditions",
    ],
    "education": [
        "through scaffolded instructional sequences",
        "via formative assessment feedback loops",
        "under constructivist pedagogical frameworks",
        "across differentiated learning modalities",
    ],
    "law": [
        "within applicable statutory and regulatory frameworks",
        "contingent on jurisdictional precedent",
        "through established common law doctrine",
        "under constitutional due process constraints",
    ],
    "business": [
        "through market-driven competitive mechanisms",
        "via resource allocation optimization processes",
        "contingent on supply-chain operational integrity",
        "under conditions of information asymmetry",
    ],
    "universal": [
        "through inherently integrated processes",
        "within clearly defined boundaries",
        "under normal operational conditions",
        "via closely coordinated internal mechanisms",
        "contingent on underlying structural integrity",
        "across multiple interconnected dimensions",
        "through systematically organized pathways",
        "within established theoretical parameters",
        "under standard analytical conditions",
        "via well-documented empirical patterns",
        "through convergent lines of evidence",
        "within broadly accepted scholarly norms",
        "under conditions of internal consistency",
        "through mutually reinforcing causal chains",
        "across diverse empirical contexts",
        "within methodologically defined constraints",
        "through complementary analytical approaches",
        "under conditions of theoretical parsimony",
        "across multiple levels of analysis",
        "within the scope of the current analysis",
        "under well-specified model assumptions",
        "through iterative cycles of refinement",
        "across a range of empirical observations",
        "under conditions of analytical rigor",
        "through overlapping and reinforcing processes",
        "within the framework of existing scholarship",
        "under carefully controlled parameters",
        "through sustained empirical investigation",
        "within logically structured argument chains",
        "across both theoretical and applied dimensions",
    ],
}

HEDGING_PARENTHETICALS: list = [
    "(arguably)", "(presumably)", "(by extension)", "(notably)", "(evidently)",
    "(characteristically)", "(as expected)", "(in most cases)",
    "(broadly speaking)", "(on balance)", "(to some degree)", "(in principle)",
    "(under typical conditions)", "(in relative terms)", "(with few exceptions)",
    "(in theoretical terms)", "(empirically speaking)", "(all else being equal)",
    "(as conventionally understood)", "(by most accounts)",
    "(from a functional standpoint)", "(by current consensus)",
    "(in practical terms)",
]

SIGNPOST_OPENERS: list = [
    "From an analytical standpoint,",
    "Within this framework,",
    "Interestingly,",
    "Notably,",
    "In practice,",
    "Under these conditions,",
    "Specifically,",
    "In effect,",
    "Conversely,",
    "As expected,",
    "In this context,",
    "Evidently,",
    "Alternatively,",
    "In particular,",
    "By comparison,",
    "In such cases,",
    "Broadly speaking,",
    "Taken together,",
    "Viewed through this lens,",
    "On closer inspection,",
    "As the evidence suggests,",
    "By the same token,",
    "At the same time,",
    "In a related vein,",
    "Against this backdrop,",
    "In light of this,",
    "To this end,",
    "From this perspective,",
    "With this in mind,",
    "To a certain extent,",
    "Under closer scrutiny,",
    "At its core,",
    "In the aggregate,",
    "Fundamentally,",
]


def get_filler_phrase(sentence: str = "") -> str:
    domain = _detect_domain(sentence) if sentence else "universal"
    pool   = _FILLER_MAP.get(domain, _FILLER_MAP["universal"])
    if random.random() < 0.3:
        pool = _FILLER_MAP["universal"]
    return random.choice(pool)

def get_hedging_parenthetical() -> str:
    return random.choice(HEDGING_PARENTHETICALS)

def get_signpost_opener() -> str:
    return random.choice(SIGNPOST_OPENERS)


# ===========================================================================
# ===== TEXT UTILITIES ======================================================
# ===========================================================================

_ABBREV_RE = re.compile(
    r"\b(e\.g\.|i\.e\.|et al\.|Fig\.|Dr\.|Prof\.|vs\.|cf\.|ca\.|approx\.)\s"
)

def split_sentences(text: str) -> List[str]:
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    sents     = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected.strip())
    return [s.replace("\x00", ".").strip() for s in sents if s.strip()]

def split_paragraphs(text: str) -> List[List[str]]:
    paragraphs, current = [], []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(current)
                current = []
        else:
            current.append(stripped)
    if current:
        paragraphs.append(current)
    return [split_sentences(" ".join(para)) for para in paragraphs]

def is_markdown_heading(text: str) -> bool:
    return text.strip().startswith("#")

def is_markdown_list(text: str) -> bool:
    s = text.strip()
    return s.startswith(("* ", "- ")) or bool(re.match(r"^\d+\.", s))

def count_words(text: str) -> int:
    return len(text.split())

def _prepend_signpost(sent: str, opener: str) -> str:
    sent   = sent.strip()
    opener = opener.strip()
    if not sent:
        return sent
    body = sent[0].lower() + sent[1:] if opener.endswith(",") else sent
    return f"{opener} {body}"

def _insert_parenthetical_after_noun(sent: str, noun: str, p: str) -> str:
    pat = re.compile(r"(\b" + re.escape(noun) + r"\b)([,\.;:!?]?)", re.IGNORECASE)
    def rep(m):
        w, punc = m.group(1), m.group(2)
        return f"{w} {p}{punc}" if punc else f"{w} {p}"
    return pat.sub(rep, sent, count=1)


# ===========================================================================
# ===== SCORING =============================================================
# ===========================================================================

def score_sentence(sent: str) -> float:
    s     = sent.lower()
    words = sent.split()
    score = 0.0

    for tell in AI_TELL_PHRASES:
        if tell in s:
            score += 15

    if 15 <= len(words) <= 22:
        score += 10

    if words:
        first = words[0].lower().strip(",.!?;:")
        if first in TRANSITIONAL_OPENERS:
            score += 12

    if len(words) > 5:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < 0.5:
            score += 10

    if sent.count(",") > 3 or sent.count(";") > 2:
        score += 15

    if len(words) < 4 and not (sent.startswith("#") or sent.startswith("*")):
        score += 20

    return min(100.0, max(0.0, score))


# ===========================================================================
# ===== FILLER GATE =========================================================
# ===========================================================================

def _sentence_needs_filler(sentence: str, original: str) -> bool:
    """
    True ONLY when: sentence < 7 words, shortfall > 5, no trailing parenthetical.
    Single point of truth — nothing else in the pipeline injects fillers.
    """
    hum   = count_words(sentence)
    orig  = count_words(original)
    return hum < 7 and (orig - hum) > 5 and not sentence.rstrip(".!?").endswith(")")


# ===========================================================================
# ===== LENGTH ENFORCEMENT ==================================================
# ===========================================================================

def enforce_length_constraint(original: str, humanized: str, max_diff: int = 3) -> str:
    oc = count_words(original)
    hc = count_words(humanized)

    if abs(oc - hc) <= max_diff:
        return humanized

    if hc > oc + max_diff:
        words   = humanized.split()
        keep    = max(oc + max_diff - 1, min(oc, len(words)))
        trimmed = " ".join(words[:keep]).rstrip(",;—")
        return trimmed if trimmed[-1] in ".!?" else trimmed + "."

    if _sentence_needs_filler(humanized, original):
        humanized = humanized.rstrip(".") + " " + get_filler_phrase(humanized) + "."

    return humanized


def validate_and_correct_length(original: str, humanized: str, max_diff: int = 3) -> str:
    if abs(count_words(original) - count_words(humanized)) <= max_diff:
        return humanized
    return enforce_length_constraint(original, humanized, max_diff)


# ===========================================================================
# ===== DEEP RESTRUCTURE ENGINE (StealthMode core) ==========================
#
# StealthWriter's advantage is that it does not merely swap synonyms.
# It re-architects sentence structure itself: voice flipping, fronting,
# clause inversion, appositive insertion, and cleft construction.
# This engine replicates those transformations locally, before Mistral
# even sees the text, so Mistral only needs to polish, not rebuild.
# ===========================================================================

# Passive-to-active heuristic patterns
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|has been|have been|had been)\s+(\w+ed)\s+by\b",
    re.IGNORECASE,
)

# Common "It is X that" cleft patterns
_CLEFT_RE = re.compile(
    r"^It\s+(is|was)\s+(\w+)\s+that\s+", re.IGNORECASE
)

# Verb phrase rotation table — domain-agnostic verbs that detectors learn
_VERB_ROTATION: dict = {
    "regulates":   ["controls", "governs", "modulates", "directs", "coordinates"],
    "controls":    ["governs", "modulates", "regulates", "oversees", "directs"],
    "governs":     ["regulates", "controls", "shapes", "directs", "determines"],
    "modulates":   ["adjusts", "regulates", "governs", "tunes", "calibrates"],
    "influences":  ["shapes", "affects", "determines", "drives", "conditions"],
    "shapes":      ["influences", "determines", "drives", "conditions", "frames"],
    "determines":  ["governs", "shapes", "dictates", "establishes", "defines"],
    "demonstrates":["shows", "reveals", "indicates", "confirms", "illustrates"],
    "indicates":   ["shows", "suggests", "points to", "reveals", "confirms"],
    "suggests":    ["indicates", "implies", "points to", "proposes", "shows"],
    "provides":    ["offers", "supplies", "yields", "furnishes", "delivers"],
    "requires":    ["demands", "necessitates", "calls for", "needs", "entails"],
    "involves":    ["entails", "encompasses", "includes", "comprises", "concerns"],
    "allows":      ["enables", "permits", "facilitates", "makes possible", "supports"],
    "enables":     ["allows", "permits", "facilitates", "supports", "makes possible"],
    "produces":    ["generates", "yields", "creates", "results in", "gives rise to"],
    "generates":   ["produces", "yields", "creates", "gives rise to", "brings about"],
    "contains":    ["holds", "comprises", "encompasses", "includes", "houses"],
    "comprises":   ["contains", "encompasses", "includes", "consists of", "incorporates"],
    "represents":  ["constitutes", "embodies", "reflects", "stands for", "signifies"],
    "constitutes": ["represents", "forms", "makes up", "amounts to", "embodies"],
    "highlights":  ["underlines", "points to", "draws attention to", "marks", "flags"],
    "reflects":    ["mirrors", "indicates", "captures", "embodies", "represents"],
    "supports":    ["backs", "corroborates", "substantiates", "underpins", "confirms"],
    "establishes": ["sets out", "defines", "determines", "confirms", "lays out"],
    "facilitates": ["enables", "supports", "promotes", "aids", "helps drive"],
}

# Word-level synonym table — extended
_WORD_REPLACEMENTS: dict = {
    r"\bimportant\b":              ["key", "critical", "central", "essential", "primary"],
    r"\bplays a critical role\b":  ["is essential", "is central", "drives", "underpins"],
    r"\bplays a vital role\b":     ["is essential", "is critical", "anchors", "drives"],
    r"\bplays a key role\b":       ["is central", "is integral", "drives", "underpins"],
    r"\bis located\b":             ["lies", "sits", "is found", "is situated", "resides"],
    r"\bis composed of\b":         ["contains", "comprises", "includes", "incorporates"],
    r"\bacts as\b":                ["works as", "functions as", "serves as", "operates as"],
    r"\bdue to\b":                 ["because of", "owing to", "as a result of", "from"],
    r"\boverall\b":                ["in sum", "taken together", "collectively", "broadly"],
    r"\badditionally\b":           ["also", "further", "beyond this", "on top of this"],
    r"\bhowever\b":                ["yet", "though", "even so", "still", "that said"],
    r"\btherefore\b":              ["thus", "hence", "for this reason", "so"],
    r"\bconsequently\b":           ["as a result", "thereby", "hence", "this means"],
    r"\bregulates\b":              ["controls", "governs", "modulates", "directs"],
    r"\bcontains\b":               ["holds", "possesses", "encompasses", "incorporates"],
    r"\bresponsible for\b":        ["accountable for", "tasked with", "integral to"],
    r"\bassociated with\b":        ["linked to", "tied to", "connected with", "related to"],
    r"\binvolved in\b":            ["engaged in", "contributing to", "implicated in"],
    r"\bconsists of\b":            ["comprises", "is made up of", "incorporates"],
    r"\bpart of\b":                ["component of", "element of", "constituent of"],
    r"\bfunction\b":               ["role", "purpose", "operation", "capacity"],
    r"\bstructure\b":              ["anatomy", "architecture", "morphology", "configuration"],
    r"\bprocess\b":                ["mechanism", "procedure", "pathway", "cascade"],
    r"\bcontrol\b":                ["regulation", "oversight", "direction", "governance"],
    r"\bphenomenon\b":             ["occurrence", "manifestation", "observation", "finding"],
    r"\bframework\b":              ["schema", "construct", "architecture", "scaffold"],
    r"\bfurthermore\b":            ["beyond this", "building on this", "relatedly"],
    r"\bmoreover\b":               ["beyond this", "equally", "in the same vein"],
    r"\bin addition\b":            ["beyond this", "on top of this", "also"],
    r"\bsignificantly\b":          ["markedly", "substantially", "considerably", "notably"],
    r"\bimportantly\b":            ["critically", "centrally", "above all", "most tellingly"],
    r"\bnovel\b":                  ["new", "original", "distinct", "fresh"],
    r"\bpropose\b":                ["suggest", "advance", "put forward", "argue"],
    r"\butilize\b":                ["use", "employ", "apply", "draw on"],
    r"\butilises\b":               ["uses", "employs", "applies", "draws on"],
    r"\bobtain\b":                 ["get", "gain", "achieve", "derive", "acquire"],
    r"\bdemonstrate\b":            ["show", "reveal", "confirm", "illustrate", "indicate"],
    r"\bsubsequent\b":             ["later", "following", "next", "ensuing"],
    r"\bprior to\b":               ["before", "ahead of", "preceding"],
    r"\bin order to\b":            ["to", "so as to", "with the aim of"],
    r"\bapproximately\b":          ["roughly", "around", "nearly", "about"],
    r"\bnevertheless\b":           ["still", "even so", "that said", "yet"],
    r"\bnotwithstanding\b":        ["despite this", "even so", "still", "regardless"],
}

_COMPILED_REPLACEMENTS = {
    re.compile(pat, re.IGNORECASE): choices
    for pat, choices in _WORD_REPLACEMENTS.items()
}


def _rotate_verbs(text: str, used: set) -> str:
    """Replace detected verbs with rotation alternatives, tracking what was used."""
    for verb, alternatives in _VERB_ROTATION.items():
        pattern = re.compile(r"\b" + re.escape(verb) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            fresh = [a for a in alternatives if a not in used]
            if fresh:
                replacement = random.choice(fresh)
                used.add(replacement)
                text = pattern.sub(replacement, text, count=1)
    return text


def _flip_cleft(sent: str) -> str:
    """Remove 'It is X that' construction — a strong AI tell."""
    m = _CLEFT_RE.match(sent)
    if m:
        remainder = sent[m.end():]
        adj       = m.group(2)
        return f"{remainder.strip().rstrip('.')} — {adj}.".strip()
    return sent


def _front_adverbial(sent: str) -> str:
    """
    Move a trailing prepositional phrase to sentence-front for variation.
    e.g. 'Cells divide rapidly under hypoxic conditions.'
         → 'Under hypoxic conditions, cells divide rapidly.'
    Only fires when sentence ends with a short 'under/in/through/via' phrase.
    """
    pattern = re.compile(
        r"^(.{20,}?)\s+(under|in|through|via|across|within)\s+([^.]{4,40})\.$",
        re.IGNORECASE,
    )
    m = pattern.match(sent.strip())
    if m:
        body    = m.group(1).strip()
        prep    = m.group(2)
        phrase  = m.group(3).strip()
        body    = body[0].lower() + body[1:]
        return f"{prep.capitalize()} {phrase}, {body}."
    return sent


def _split_at_semicolon_candidate(sent: str) -> str:
    """
    Split an overly long sentence at a natural comma boundary into two sentences.
    Only fires when sentence > 32 words and there is a clear mid-sentence comma.
    """
    words = sent.split()
    if len(words) < 32:
        return sent
    half = len(words) // 2
    # Find closest comma to halfway point
    for delta in range(0, half):
        for idx in [half + delta, half - delta]:
            if 0 < idx < len(words):
                w = words[idx]
                if w.endswith(","):
                    left  = " ".join(words[:idx]).rstrip(",") + "."
                    right = " ".join(words[idx + 1:])
                    if right:
                        right = right[0].upper() + right[1:]
                        if right[-1] not in ".!?":
                            right += "."
                        return left + " " + right
    return sent


def deep_restructure(sent: str, used_verbs: set, index: int) -> str:
    """
    Apply structural transformations before synonym swapping.
    Each transformation is gated so it only fires when the sentence
    meets minimum length/content criteria.
    """
    s = sent.strip()

    # 1. Remove cleft constructions
    s = _flip_cleft(s)

    # 2. Split overlong sentences
    s = _split_at_semicolon_candidate(s)

    # 3. Front short trailing adverbials (every 3rd sentence)
    if index % 3 == 0:
        s = _front_adverbial(s)

    # 4. Rotate repeated verbs
    s = _rotate_verbs(s, used_verbs)

    # 5. Synonym replacement
    for pattern, choices in _COMPILED_REPLACEMENTS.items():
        s = pattern.sub(lambda m, c=choices: random.choice(c), s)

    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


# ===========================================================================
# ===== HUMAN IMPERFECTION ENGINE ===========================================
#
# AI detectors measure two signals:
#   • Perplexity  — how predictable each word choice is
#   • Burstiness  — how uniform sentence lengths are
#
# This engine targets both by injecting controlled unpredictability:
# short fragments, natural asides, direct rhetorical questions (rare),
# and deliberate length asymmetry — exactly what StealthWriter does.
# ===========================================================================

_SHORT_CONNECTIVES = [
    "This matters.",
    "The distinction is real.",
    "Not all cases are equal.",
    "Context determines the outcome.",
    "The pattern holds consistently.",
    "Exceptions exist, but they are rare.",
    "The implications are direct.",
    "The mechanism is well documented.",
    "Evidence supports this view.",
    "This point deserves emphasis.",
]

_NATURAL_ASIDES = [
    "— a detail often overlooked —",
    "— worth noting here —",
    "— and this is key —",
    "— though rarely discussed —",
    "— a nuance that matters —",
    "— the distinction is critical —",
]


def inject_human_imperfection(sentences: List[str], mode: str = "stealth") -> List[str]:
    """
    Stealth mode: aggressive imperfection injection for maximum detection bypass.
    Humanize mode: lighter touch, preserves flow.

    Techniques:
      A. Short punchy sentence insertion after every 4th long sentence
      B. Natural aside insertion into 1 in 6 sentences (stealth only)
      C. Occasional fragment (stealth only)
      D. Sentence-start variety enforcement
    """
    if not sentences:
        return sentences

    result      = list(sentences)
    insert_rate = 0.25 if mode == "stealth" else 0.10
    aside_rate  = 0.17 if mode == "stealth" else 0.05

    processed = []
    for i, sent in enumerate(result):
        words = sent.split()

        # A. Insert short punchy sentence after long sentences
        if len(words) > 26 and random.random() < insert_rate:
            processed.append(sent)
            processed.append(random.choice(_SHORT_CONNECTIVES))
            continue

        # B. Insert natural aside into mid-sentence (stealth only)
        if (mode == "stealth"
                and len(words) > 12
                and random.random() < aside_rate
                and "—" not in sent):
            mid   = len(words) // 2
            aside = random.choice(_NATURAL_ASIDES)
            sent  = " ".join(words[:mid]) + " " + aside + " " + " ".join(words[mid:])
            sent  = re.sub(r"\s+", " ", sent).strip()
            if sent[-1] not in ".!?":
                sent += "."

        processed.append(sent)

    return processed


# ===========================================================================
# ===== BURSTINESS ENGINE ===================================================
# Targets the burstiness metric directly — creates the dramatic length
# variation that AI detectors expect from human writing.
# ===========================================================================

def syntactic_burstiness_engine(sentences: List[str]) -> List[str]:
    """
    Compression-only burstiness. No filler injection.
    Creates: Long → Short → Medium → Fragment → Long rhythm.
    """
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result      = []

    for i, sent in enumerate(sentences):
        words       = sent.split()
        current_len = len(words)
        pattern     = i % 5

        if pattern == 0 and current_len > 30:
            target = max(int(current_len * 0.82), 15)
            sent   = " ".join(words[:target]).rstrip(",;—")
            if sent[-1] not in ".!?":
                sent += "."

        elif pattern == 1 and current_len > 20:
            target = max(int(current_len * 0.60), 8)
            sent   = " ".join(words[:target]) + "."

        elif pattern == 2 and ";" not in sent and current_len > 14:
            mid  = current_len // 2
            sent = " ".join(words[:mid]) + "; " + " ".join(words[mid:])

        elif pattern == 3 and current_len > 14:
            sent = " ".join(words[:5]) + "."

        # pattern 4: untouched — natural rhythm needs unmodified sentences

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        result.append(sent)

    new_total = sum(count_words(s) for s in result)
    if abs(new_total - total_words) > int(total_words * 0.10):
        diff = new_total - total_words
        if diff > 0:
            idx = max(range(len(result)), key=lambda i: count_words(result[i]))
            w   = result[idx].split()
            result[idx] = " ".join(w[:max(len(w) - diff, 3)]) + "."

    return result


# ===========================================================================
# ===== OBFUSCATION LAYER ===================================================
# ===========================================================================

MODIFICATION_RATE = 0.32

def final_obfuscation_layer(text: str, rate: float = MODIFICATION_RATE) -> str:
    sentences = split_sentences(text)
    processed = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words or len(words) < 6:
            processed.append(sent)
            continue

        if random.random() > rate:
            processed.append(sent)
            continue

        technique = i % 4

        if technique == 0 and len(words) > 10:
            if "," in sent and len(words) > 12:
                parts = sent.split(",", 1)
                if len(parts[0].split()) >= 5 and len(parts[1].split()) >= 5:
                    sent = parts[0] + "; " + parts[1].strip()

        elif technique == 1 and len(words) > 8:
            for word in words:
                clean = word.lower().strip(",.!?;:")
                if clean in ANCHOR_NOUNS:
                    sent = _insert_parenthetical_after_noun(
                        sent, clean, get_hedging_parenthetical()
                    )
                    break

        elif technique == 2 and len(words) > 12:
            break_words = {"which", "where", "when", "while", "although"}
            for idx, word in enumerate(words):
                if word.lower() in break_words and 3 < idx < len(words) - 4:
                    fragment  = " ".join(words[:idx]).rstrip(",") + ". "
                    remainder = " ".join(words[idx:])
                    remainder = remainder[0].upper() + remainder[1:]
                    sent = fragment + remainder
                    break

        elif technique == 3 and len(words) > 10 and " and " in sent:
            and_pos = sent.find(" and ")
            before  = sent[:and_pos].strip()
            after   = sent[and_pos + 5:].strip()
            clause_verbs = {
                "is", "are", "was", "were", "has", "have",
                "controls", "regulates", "modulates", "governs",
            }
            if (len(before.split()) > 4 and len(after.split()) > 4
                    and any(v in before.lower().split() for v in clause_verbs)):
                sent = sent.replace(" and ", ", consequently, ", 1)

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        processed.append(sent)

    return " ".join(processed)


# ===========================================================================
# ===== SIGNPOST LAYER — reduced rate =======================================
# ===========================================================================

SIGNPOST_RATE = 0.13

def apply_signpost_openers(text: str, rate: float = SIGNPOST_RATE) -> str:
    sentences = split_sentences(text)
    processed = []
    for sent in sentences:
        words = sent.split()
        if (len(words) > 5
                and not is_markdown_heading(sent)
                and not is_markdown_list(sent)
                and random.random() < rate):
            sent = _prepend_signpost(sent, get_signpost_opener())
            sent = re.sub(r"\s+", " ", sent).strip()
            if sent[-1] not in ".!?":
                sent += "."
        processed.append(sent)
    return " ".join(processed)


# ===========================================================================
# ===== REPETITION ELIMINATION ==============================================
# ===========================================================================

def eliminate_repetition(text: str) -> str:
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return text

    processed    = []
    used_bigrams: set = set()

    for idx, sent in enumerate(sentences):
        words   = sent.lower().split()
        bigrams = {
            words[i].strip(",.!?;:") + " " + words[i + 1].strip(",.!?;:")
            for i in range(len(words) - 1)
        }
        overlap = len(bigrams & used_bigrams) / len(bigrams) if bigrams else 0

        if overlap > 0.30 and len(words) > 6:
            sent = " ".join(words[:5]) + "."

        used_bigrams.update(bigrams)
        if len(processed) % 5 == 0:
            used_bigrams = set()

        processed.append(sent)

    return " ".join(processed)


# ===========================================================================
# ===== LOCAL FALLBACK ======================================================
# ===========================================================================

def local_humanize(sent: str, index: int) -> str:
    words = sent.split()
    if not words:
        return sent

    h         = sent
    used_verbs: set = set()

    for pattern, choices in _COMPILED_REPLACEMENTS.items():
        h = pattern.sub(lambda m, c=choices: random.choice(c), h)

    h = _rotate_verbs(h, used_verbs)

    if index % 3 == 0 and len(words) > 8:
        h = " ".join(words[:6]) + "."
    elif index % 3 == 2 and "," in h:
        h = h.replace(",", ";", 1)

    if (index % 7 == 0 and len(words) > 5
            and not is_markdown_heading(h) and not is_markdown_list(h)):
        h = _prepend_signpost(h, get_signpost_opener())

    return h


# ===========================================================================
# ===== SYSTEM PROMPTS ======================================================
# Two prompts: STEALTH (aggressive bypass) and HUMANIZE (natural flow)
# ===========================================================================

_SYSTEM_STEALTH = """You are an elite AI-detection bypass specialist and senior academic editor, fluent across all scholarly disciplines — STEM, social sciences, humanities, law, business, and education.

Your mission: Transform AI-generated academic text so that it scores 0% on ALL AI detectors (Turnitin AI, ZeroGPT, GPTZero, Copyleaks, Winston AI, Originality.ai) while meeting peer-reviewed journal publication standards.

AI detectors measure two primary signals. You must attack both:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL 1 — PERPLEXITY (word unpredictability)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI text is statistically predictable — every word choice is the most likely next token. To defeat this:
- Choose the second-best or third-best word, not the obvious one.
- Use specific, concrete nouns over abstract ones where possible.
- Occasionally use a slightly unusual but correct word order.
- Insert a short, direct declarative sentence where the detector expects a continuation clause.
- Use domain-specific technical vocabulary correctly — detectors cannot penalize precise jargon.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL 2 — BURSTINESS (sentence length variation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI text produces uniform sentence lengths. Human writers do not. You must enforce extreme length variation:
- Short sentences (4–8 words): place after every long sentence
- Long sentences (25–35 words): embed subordinate clauses, qualifying phrases, and technical precision
- Medium sentences (12–18 words): used for transition and summary
- Fragment sentences (3–6 words, no main verb): once per paragraph for impact
- NEVER produce three consecutive sentences of similar length

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 0 — MANDATORY DOMAIN DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing a single word, identify the academic field: biology, medicine, neuroscience, computer science, engineering, history, literature, philosophy, sociology, economics, law, education, or other.
Use ONLY terminology native to that field. Never import vocabulary from an unrelated discipline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURAL TRANSFORMATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Apply at least 3 of these per paragraph:
1. VOICE FLIP: Convert passive voice to active, or active to passive — never keep the same voice as the original
2. CLAUSE INVERSION: Move subordinate clauses from sentence-end to sentence-start
3. ADVERBIAL FRONTING: Pull trailing prepositional phrases to the front: "Cells divide under hypoxia" → "Under hypoxic conditions, cells divide"
4. APPOSITIVE INSERTION: Insert a brief parenthetical or appositive phrase mid-sentence to break the pattern
5. CLEFT REMOVAL: Eliminate all "It is X that..." and "There is/are..." constructions — rewrite as direct statements
6. COMPOUND SPLITTING: Break long compound sentences into two — one long, one short

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTELY BANNED PHRASES — never use
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, leverage, synergy, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is crucial to note, it is worth noting, as mentioned earlier, it should be noted, indeed, arguably, significantly, in addition, plays a crucial role, plays a vital role, plays a key role, it is well established, it is clear that, it is evident that, needless to say, in this regard, with respect to, as such, thus far, heretofore, it can be seen that, it is noteworthy, one must consider, it is imperative, needless to say, this is especially true, this is particularly important

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANNED TRANSITIONAL OPENERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER start a sentence with: Furthermore, Moreover, Consequently, Additionally, Subsequently, Nevertheless, Notwithstanding, Accordingly, Henceforth, Heretofore.
Instead let the logic of the preceding sentence drive the next sentence naturally — no mechanical connectors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HUMAN AUTHENTICITY MARKERS — use sparingly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use "we" or "our" 1–2 times per paragraph maximum
- Use semicolons for compound sentences; avoid em-dashes
- One hedging parenthetical per paragraph at most: (notably), (evidently), (under normal conditions), (by extension), (presumably)
- Start sentences with: subject, short participial phrase, fronted object, concise conditional — vary constantly
- "But" or "Yet" to open a sentence for contrast: once per paragraph maximum
- Occasional direct rhetorical statement (not a question): e.g. "The data do not lie." — once per section

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERB PHRASE ROTATION — MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Never repeat the same verb phrase twice in one paragraph. Rotate by domain:
Science/Bio:   regulates → controls → governs → modulates → mediates → coordinates
Social/Policy: influences → shapes → determines → drives → underpins → constrains
Humanities:    argues → contends → posits → maintains → asserts → suggests
Tech/Data:     processes → computes → executes → evaluates → optimizes → transforms
Business/Econ: generates → yields → produces → drives → sustains → captures
General:       demonstrates → indicates → reveals → reflects → highlights → confirms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEXICAL DENSITY BALANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After presenting a complex technical concept, follow immediately with a short, direct clarifying sentence (4–9 words). Do not stack dense technical jargon across three consecutive sentences without a relief sentence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCLUSION RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rewrite conclusions with shorter, more direct sentences. Break symmetry — do NOT end with parallel clause lists. Synthesize, do not repeat.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT CONTENT CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. WORD COUNT: Match each original sentence within +/- 2 words
2. ZERO OMISSION: Every fact, figure, named entity, citation, qualifier, and technical variable must survive. Restructuring is allowed; omission is not
3. CITATIONS: Keep (Author, 2020), [1], [1-3] exactly as written
4. MARKDOWN: Preserve all # headings, ## subheadings, * bullets, 1. numbered lists exactly
5. STANDARD: Write like a tenured professor with 30 years of publishing experience in the detected field

OUTPUT ONLY VALID JSON — no explanation, no preamble:
{"processed_paragraphs":[{"sentences":[{"original":"exact original text","humanized":"rewritten text","alternatives":["alt1","alt2","alt3"]}]}]}"""


_SYSTEM_HUMANIZE = """You are an elite academic editor fluent across all scholarly disciplines — STEM, social sciences, humanities, law, business, and education. Rewrite AI-generated academic text to read naturally, like it was written by a leading human expert, while maintaining peer-reviewed journal standards.

STEP 0 — DOMAIN DETECTION (mandatory):
Identify the field before writing. Use only field-native terminology.

GUIDELINE 1 — SENTENCE RHYTHM:
Vary lengths: short (5–9 words) → long (20–32 words) → medium (12–18 words). Never three consecutive similar-length sentences.

GUIDELINE 2 — DICTION:
Remove: delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, landscape, tapestry, robust, holistic, leverage, synergy, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, significantly, in addition, plays a crucial/vital/key role, needless to say, it is clear that, as such, heretofore.

GUIDELINE 3 — LEXICAL DENSITY:
Balance technical jargon with direct clarification. After a complex concept, add a grounding sentence.

GUIDELINE 4 — ORGANIC TRANSITIONS:
No mechanical connectors at sentence starts. Let logic drive structure. Vary openers constantly.

GUIDELINE 5 — CONCLUSIONS:
Shorter, more direct sentences. Break parallel symmetry.

VERB ROTATION — never repeat same verb phrase in a paragraph:
Science/Bio: regulates → controls → governs → modulates → mediates
Social: influences → shapes → determines → drives → underpins
Humanities: argues → contends → posits → maintains → asserts
Tech: processes → computes → executes → evaluates → optimizes
Business: generates → yields → produces → drives → sustains
General: demonstrates → indicates → reveals → reflects → confirms

CONSTRAINTS:
- Match word count within +/- 2 words per sentence
- Zero omission of facts, figures, citations, named entities
- Preserve all markdown headings and list items exactly
- Write at tenured professor level

OUTPUT ONLY VALID JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact text","humanized":"rewrite","alternatives":["alt1","alt2","alt3"]}]}]}"""


# ===========================================================================
# ===== CORRECTION LOOP =====================================================
# ===========================================================================

def correction_loop(original: str, humanized: str, max_attempts: int = 2) -> str:
    oc = count_words(original)
    hc = count_words(humanized)

    if abs(oc - hc) <= 3:
        return humanized

    for attempt in range(max_attempts):
        try:
            prompt = (
                f"Length constraint violation.\n"
                f"Original: {oc} words. Your rewrite: {hc} words.\n\n"
                f"Original: \"{original}\"\n"
                f"Your Rewrite: \"{humanized}\"\n\n"
                f"Adjust rewrite to match EXACTLY {oc} words (+/- 2). "
                f"Do NOT omit any facts, figures, named entities, or qualifications. "
                f"Maintain academic quality and domain-appropriate terminology. "
                f"Output ONLY the corrected sentence — no quotes, no explanation."
            )
            resp = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            corrected = resp.choices[0].message.content.strip().strip("\"'")
            if abs(oc - count_words(corrected)) <= 3:
                return corrected
            humanized = corrected
            hc        = count_words(humanized)
        except Exception as e:
            print(f"Correction loop attempt {attempt + 1} failed: {e}")
            break

    return enforce_length_constraint(original, humanized, max_diff=3)


# ===========================================================================
# ===== MAIN PROCESSING =====================================================
# ===========================================================================

def humanize_with_mistral(
    paragraphs: List[List[str]],
    style: str,
    mode: str = "stealth",
) -> List[ParagraphData]:

    print(f"CALLING MISTRAL — mode={mode}, paragraphs={len(paragraphs)}")

    system_prompt = _SYSTEM_STEALTH if mode == "stealth" else _SYSTEM_HUMANIZE

    # Build numbered input with word counts
    lines = []
    for i, para in enumerate(paragraphs):
        lines.append(f"Paragraph {i + 1}:")
        for j, s in enumerate(para):
            lines.append(f"{j + 1}. [{count_words(s)} words] {s}")
        lines.append("")

    data          = None
    mistral_error = None

    try:
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Style: {style} | Mode: {mode}\n\n"
                        "Rewrite this academic text following all guidelines above. "
                        "Word counts per sentence are in [brackets] — match within +/- 2 words. "
                        "Preserve every fact, figure, citation, and named entity. "
                        "Enforce extreme sentence length variation — no uniform structures. "
                        "Eliminate ALL banned transitional phrases. "
                        "Apply structural transformations (voice flip, clause inversion, "
                        "adverbial fronting, cleft removal) as instructed. "
                        "Preserve all markdown headings and list items exactly:\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.75 if mode == "stealth" else 0.68,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()

        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text  = text.strip()
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start: end + 1]

        data = json.loads(text)
        print("Mistral JSON parsed successfully")

    except Exception as e:
        mistral_error = str(e)
        print(f"Mistral FAILED: {e}")

    if not data or not data.get("processed_paragraphs"):
        print(f"Using LOCAL FALLBACK. Error: {mistral_error}")
        data = {
            "processed_paragraphs": [
                {
                    "sentences": [
                        {
                            "original": s,
                            "humanized": local_humanize(s, j),
                            "alternatives": [
                                local_humanize(s, j + 10),
                                local_humanize(s, j + 20),
                                local_humanize(s, j + 30),
                            ],
                        }
                        for j, s in enumerate(para)
                    ]
                }
                for para in paragraphs
            ]
        }

    result: List[ParagraphData] = []

    for i, para in enumerate(data.get("processed_paragraphs", [])):
        para_sentences = []
        used_verbs: set = set()

        # ── Pass 1: collect, correct lengths, deep-restructure ──────────────
        for j, sent in enumerate(para.get("sentences", [])):
            orig = sent.get("original", "")
            h    = sent.get("humanized", "") or local_humanize(orig, j)
            h    = correction_loop(orig, h)
            h    = validate_and_correct_length(orig, h, max_diff=3)
            # Deep restructure pass
            h    = deep_restructure(h, used_verbs, j)
            h    = validate_and_correct_length(orig, h, max_diff=3)
            para_sentences.append({
                "orig":     orig,
                "hum":      h,
                "raw_alts": sent.get("alternatives", [])[:3],
            })

        # ── Pass 2: burstiness + human imperfection at paragraph level ──────
        humanized_only  = [s["hum"] for s in para_sentences]
        burst           = syntactic_burstiness_engine(humanized_only)
        burst           = inject_human_imperfection(burst, mode=mode)

        # Realign burst list with para_sentences (injection may add extra sentences)
        # Extra injected sentences do not have originals — handle gracefully
        aligned_burst: List[Tuple[Optional[dict], str]] = []
        orig_idx = 0
        for sent_text in burst:
            if orig_idx < len(para_sentences):
                sd = para_sentences[orig_idx]
                if sent_text in _SHORT_CONNECTIVES or any(a in sent_text for a in _NATURAL_ASIDES):
                    aligned_burst.append((None, sent_text))
                else:
                    aligned_burst.append((sd, sent_text))
                    orig_idx += 1
            else:
                aligned_burst.append((None, sent_text))

        final_sentences: List[SentenceData] = []

        for j, (sent_data, h) in enumerate(aligned_burst):
            # Injected sentences (short connectives, asides) — no original
            if sent_data is None:
                # Add as a non-scored pass-through sentence
                h = re.sub(r"\s+", " ", h).strip()
                if h and h[-1] not in ".!?":
                    h += "."
                final_sentences.append(SentenceData(
                    id=f"p{i}-s{j}-injected",
                    original="",
                    humanized=h,
                    alternatives=[],
                    score=0.0,
                ))
                continue

            orig_sent = sent_data["orig"]
            h = validate_and_correct_length(orig_sent, h, max_diff=3)
            h = apply_signpost_openers(h)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            h = validate_and_correct_length(orig_sent, h, max_diff=3)
            score = score_sentence(h)

            # Build alternatives
            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                alt = alt or local_humanize(orig_sent, idx + 100)
                alt = correction_loop(orig_sent, alt)
                alt = validate_and_correct_length(orig_sent, alt, max_diff=3)
                alt = deep_restructure(alt, set(), idx)
                alt = final_obfuscation_layer(alt)
                alt = eliminate_repetition(alt)
                alt = validate_and_correct_length(orig_sent, alt, max_diff=3)
                alt = re.sub(r"\s+", " ", alt).strip()
                if alt and alt[-1] not in ".!?":
                    alt += "."
                clean_alts.append(alt)

            orig_lower  = orig_sent.lower().strip()
            unique_alts: List[str] = []
            seen_lowers: set = set()

            for alt in clean_alts:
                al = alt.lower().strip()
                if al != orig_lower and al not in seen_lowers:
                    unique_alts.append(alt)
                    seen_lowers.add(al)

            seed = 200
            while len(unique_alts) < 3:
                fallback = local_humanize(orig_sent, seed)
                fallback = validate_and_correct_length(orig_sent, fallback, max_diff=3)
                fl = fallback.lower().strip()
                if fl != orig_lower and fl not in seen_lowers:
                    unique_alts.append(fallback)
                    seen_lowers.add(fl)
                seed += 50

            final_sentences.append(SentenceData(
                id=f"p{i}-s{j}",
                original=orig_sent,
                humanized=h,
                alternatives=unique_alts[:3],
                score=score,
            ))

        result.append(ParagraphData(id=f"para-{i}", sentences=final_sentences))

    print(f"RETURNING {len(result)} paragraphs")
    return result


# ===========================================================================
# ===== API ENDPOINTS =======================================================
# ===========================================================================

@app.post("/api/process-text", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    mode      = req.mode if req.mode in ("stealth", "humanize") else "stealth"
    pars      = split_paragraphs(req.text)
    processed = humanize_with_mistral(pars, req.style, mode=mode)
    all_s     = [s for p in processed for s in p.sentences if s.original]
    avg       = sum(s.score for s in all_s) / len(all_s) if all_s else 0
    return ProcessResponse(
        processed_paragraphs=processed,
        total_sentences=len(all_s),
        avg_score=round(avg, 1),
    )

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "mistral-large-latest", "modes": ["stealth", "humanize"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
