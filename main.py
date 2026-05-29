import os
import re
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from mistralai.client import Mistral

# ===== API KEY =====
API_KEY = os.getenv("MISTRAL_API_KEY", "")
if not API_KEY:
    print("ERROR: No API key. Set MISTRAL_API_KEY environment variable.")
    exit(1)

print(f"Mistral API Key loaded: {API_KEY[:8]}...")
client = Mistral(api_key=API_KEY)

app = FastAPI(title="Academic Humanizer")
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
# ===== AI-TELL DETECTION ===================================================
# ===========================================================================

TRANSITIONAL_OPENERS = {
    "furthermore", "moreover", "however", "therefore", "thus", "consequently",
    "additionally", "crucially", "subsequently", "nevertheless",
    "notwithstanding", "accordingly", "henceforth", "heretofore",
}

AI_TELL_PHRASES = {
    "delve", "testament", "pivotal", "moreover", "furthermore",
    "it is important to note", "it is crucial to note", "in conclusion",
    "landscape", "tapestry", "beacon", "underscore", "shed light on",
    "ever-evolving", "multifaceted", "intricate", "robust", "holistic",
    "leverage", "synergy", "stakeholder", "crucially", "underscoring",
    "it is worth noting", "as mentioned earlier", "it should be noted",
    "indeed", "arguably", "significantly", "in addition", "it is essential",
    "plays a crucial role", "plays a vital role", "plays a key role",
    "it is well established", "needless to say",
    "this is particularly important", "this is especially true",
    "it can be seen that", "it is clear that", "it is evident that",
    "as such", "thus far", "operating continuously",
    "it is noteworthy", "one must consider", "it is imperative",
    "merely scratches the surface", "navigate", "and this is key",
}

ANCHOR_NOUNS = {
    "cerebellum", "medulla", "pons", "cortex", "tracts", "nuclei", "nerves",
    "arteries", "structure", "organ", "system", "pathway", "mechanism",
    "framework", "apparatus", "substrate", "nucleus", "model", "theory",
    "argument", "concept", "process", "method", "approach", "variable",
    "factor", "element", "component", "dimension", "institution", "policy",
    "context", "principle", "evidence", "data", "analysis", "result",
    "finding", "outcome",
}


# ===========================================================================
# ===== DOMAIN DETECTION ====================================================
# ===========================================================================

_DOMAIN_KEYWORDS: dict = {
    "bio": [
        "cell", "cells", "neural", "neuron", "neurons", "brain", "cortex",
        "physiological", "anatomy", "anatomical", "organ", "tissue", "gene",
        "genetic", "protein", "enzyme", "receptor", "synapse", "axon",
        "hormone", "metabolism", "homeostasis", "pathology", "clinical",
        "medical", "patient", "disease", "diagnosis", "treatment", "immune",
        "blood", "cardiac", "cerebellum", "medulla", "pons", "cortical",
        "nucleus", "nuclei", "tract", "tracts", "nerve", "nerves", "spinal",
        "vascular", "muscle", "cellular", "molecular", "biochemical",
        "pharmacological", "therapeutic",
    ],
    "tech": [
        "data", "dataset", "model", "models", "algorithm", "algorithms",
        "software", "code", "program", "function", "parameter",
        "machine learning", "deep learning", "artificial intelligence",
        "database", "server", "api", "protocol", "optimization", "compiler",
        "hardware", "processor", "storage", "simulation", "signal", "circuit",
        "matrix", "vector", "tensor", "gradient", "classification",
        "regression", "pipeline", "deployment", "architecture",
    ],
    "humanities": [
        "poem", "poetry", "novel", "narrative", "text", "texts", "author",
        "literature", "literary", "history", "historical", "culture",
        "cultural", "philosophy", "philosophical", "theory", "theoretical",
        "argument", "discourse", "ideology", "epistemology", "ethics",
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
        "institution", "attitude", "perception", "norm", "election",
        "democracy", "governance", "legislation", "welfare", "employment",
        "regression", "coefficient", "correlation", "p-value",
    ],
    "natural": [
        "climate", "environment", "ecology", "species", "biodiversity",
        "habitat", "ecosystem", "evolution", "chemistry", "chemical",
        "reaction", "compound", "molecule", "atom", "element", "quantum",
        "physics", "energy", "thermodynamics", "entropy", "geology",
        "sediment", "tectonic", "oceanic", "atmospheric", "carbon",
        "nitrogen", "oxygen", "hydrogen", "temperature", "pressure",
        "radiation", "electron", "proton", "isotope", "catalyst",
    ],
    "education": [
        "student", "students", "teacher", "teachers", "classroom",
        "curriculum", "learning", "learner", "pedagogy", "instruction",
        "assessment", "literacy", "school", "university", "college",
        "course", "lesson", "scaffolding", "feedback", "motivation",
        "engagement", "cognition", "cognitive", "metacognition",
    ],
    "law": [
        "law", "legal", "court", "courts", "judge", "judicial", "statute",
        "legislation", "regulation", "contract", "liability", "tort",
        "plaintiff", "defendant", "jurisdiction", "precedent",
        "constitutional", "rights", "criminal", "civil", "procedure",
        "evidence", "verdict", "appeal", "compliance", "jurisprudence",
    ],
    "business": [
        "business", "firm", "company", "profit", "revenue", "cost",
        "investment", "investor", "finance", "financial", "accounting",
        "audit", "budget", "strategy", "management", "leadership",
        "logistics", "operations", "performance", "shareholder", "equity",
        "asset", "liability", "competition", "industry", "sector", "gdp",
        "inflation", "monetary", "trade",
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
        "contingent on afferent signal integrity",
        "within tightly regulated homeostatic bounds",
        "under conditions of normal physiological demand",
        "through coordinated synaptic transmission",
    ],
    "tech": [
        "within tightly constrained computational parameters",
        "contingent on data stream integrity",
        "via iterative algorithmic refinement",
        "through layered abstraction hierarchies",
        "under defined boundary conditions",
    ],
    "humanities": [
        "within prevailing theoretical frameworks",
        "through established socio-cultural paradigms",
        "contingent on historical contextual variables",
        "through critical hermeneutic engagement",
    ],
    "social": [
        "through established socio-institutional mechanisms",
        "within prevailing policy frameworks",
        "under conditions of institutional constraint",
        "through mediating psychosocial pathways",
    ],
    "natural": [
        "through thermodynamically driven processes",
        "via molecular diffusion gradients",
        "under equilibrium state conditions",
        "within energetically bounded system states",
    ],
    "education": [
        "through scaffolded instructional sequences",
        "via formative assessment feedback loops",
        "under constructivist pedagogical frameworks",
    ],
    "law": [
        "within applicable statutory and regulatory frameworks",
        "contingent on jurisdictional precedent",
        "under constitutional due process constraints",
    ],
    "business": [
        "through market-driven competitive mechanisms",
        "under conditions of information asymmetry",
        "via transaction cost minimization mechanisms",
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
        "within methodologically defined constraints",
        "through complementary analytical approaches",
        "under conditions of theoretical parsimony",
        "across multiple levels of analysis",
        "within the scope of the current analysis",
        "under well-specified model assumptions",
        "through iterative cycles of refinement",
        "across a range of empirical observations",
        "within the framework of existing scholarship",
        "under carefully controlled parameters",
        "through sustained empirical investigation",
    ],
}

# Hedging parentheticals — used only once per paragraph by obfuscation layer
HEDGING_PARENTHETICALS: list = [
    "(arguably)", "(presumably)", "(by extension)", "(notably)",
    "(evidently)", "(as expected)", "(in most cases)", "(broadly speaking)",
    "(on balance)", "(to some degree)", "(in principle)",
    "(under typical conditions)", "(in relative terms)",
    "(in theoretical terms)", "(empirically speaking)",
    "(all else being equal)", "(as conventionally understood)",
    "(from a functional standpoint)", "(by current consensus)",
    "(in practical terms)",
]

# Signpost openers — used at low rate, only for opener variety
SIGNPOST_OPENERS: list = [
    "Within this framework,",
    "Specifically,",
    "In effect,",
    "Conversely,",
    "As expected,",
    "In this context,",
    "Alternatively,",
    "In particular,",
    "By comparison,",
    "Broadly speaking,",
    "Taken together,",
    "On closer inspection,",
    "As the evidence suggests,",
    "By the same token,",
    "At the same time,",
    "In a related vein,",
    "Against this backdrop,",
    "In light of this,",
    "From this perspective,",
    "With this in mind,",
    "To a certain extent,",
    "Under closer scrutiny,",
    "At its core,",
    "In the aggregate,",
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
    pat = re.compile(
        r"(\b" + re.escape(noun) + r"\b)([,\.;:!?]?)", re.IGNORECASE
    )
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
    True ONLY when sentence < 7 words AND shortfall > 5 words
    AND no trailing parenthetical already present.
    Single gating point — nothing else injects fillers.
    """
    hum  = count_words(sentence)
    orig = count_words(original)
    return (
        hum < 7
        and (orig - hum) > 5
        and not sentence.rstrip(".!?").endswith(")")
    )


# ===========================================================================
# ===== LENGTH ENFORCEMENT ==================================================
# ===========================================================================

def enforce_length_constraint(
    original: str, humanized: str, max_diff: int = 3
) -> str:
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


def validate_and_correct_length(
    original: str, humanized: str, max_diff: int = 3
) -> str:
    if abs(count_words(original) - count_words(humanized)) <= max_diff:
        return humanized
    return enforce_length_constraint(original, humanized, max_diff)


# ===========================================================================
# ===== WORD REPLACEMENT MAP ================================================
# Conservative — only replaces the clearest AI-tell words.
# Does NOT stack replacements or inflate diction.
# ===========================================================================

_WORD_REPLACEMENTS: dict = {
    r"\bimportant\b":              ["key", "critical", "central", "essential"],
    r"\bplays a critical role\b":  ["is essential", "is central", "underpins"],
    r"\bplays a vital role\b":     ["is essential", "is critical", "anchors"],
    r"\bplays a key role\b":       ["is central", "is integral", "drives"],
    r"\bis located\b":             ["lies", "sits", "is found", "resides"],
    r"\bis composed of\b":         ["contains", "comprises", "incorporates"],
    r"\bacts as\b":                ["works as", "functions as", "serves as"],
    r"\bdue to\b":                 ["because of", "owing to", "as a result of"],
    r"\boverall\b":                ["in sum", "taken together", "broadly"],
    r"\badditionally\b":           ["also", "further", "beyond this"],
    r"\bhowever\b":                ["yet", "though", "even so", "that said"],
    r"\btherefore\b":              ["thus", "hence", "for this reason"],
    r"\bconsequently\b":           ["as a result", "thereby", "hence"],
    r"\bregulates\b":              ["controls", "governs", "modulates"],
    r"\bcontains\b":               ["holds", "encompasses", "incorporates"],
    r"\bresponsible for\b":        ["accountable for", "tasked with"],
    r"\bassociated with\b":        ["linked to", "tied to", "related to"],
    r"\binvolved in\b":            ["engaged in", "contributing to"],
    r"\bconsists of\b":            ["comprises", "is made up of"],
    r"\bpart of\b":                ["component of", "element of"],
    r"\bfunction\b":               ["role", "purpose", "capacity"],
    r"\bprocess\b":                ["mechanism", "procedure", "pathway"],
    r"\bcontrol\b":                ["regulation", "oversight", "governance"],
    r"\bphenomenon\b":             ["occurrence", "observation", "finding"],
    r"\bfurthermore\b":            ["beyond this", "building on this"],
    r"\bmoreover\b":               ["beyond this", "equally"],
    r"\bin addition\b":            ["beyond this", "also"],
    r"\bsignificantly\b":          ["markedly", "substantially", "considerably"],
    r"\butilize\b":                ["use", "employ", "apply"],
    r"\butilises\b":               ["uses", "employs", "applies"],
    r"\bdemonstrate\b":            ["show", "reveal", "confirm", "illustrate"],
    r"\bsubsequent\b":             ["later", "following", "ensuing"],
    r"\bin order to\b":            ["to", "so as to"],
    r"\bnevertheless\b":           ["still", "even so", "that said"],
    r"\bnotwithstanding\b":        ["despite this", "even so", "still"],
}

_COMPILED_REPLACEMENTS = {
    re.compile(pat, re.IGNORECASE): choices
    for pat, choices in _WORD_REPLACEMENTS.items()
}


# ===========================================================================
# ===== STRUCTURAL TRANSFORMATION ENGINE ====================================
#
# Three clean, grammar-safe transformations only.
# No punctuation injection, no fake asides, no em-dash interruptions.
# ===========================================================================

def _flip_cleft(sent: str) -> str:
    """Remove 'It is/was X that ...' — a reliable AI-generation tell."""
    pat = re.compile(r"^It\s+(is|was)\s+(\w+)\s+that\s+", re.IGNORECASE)
    m   = pat.match(sent)
    if m:
        remainder = sent[m.end():].strip().rstrip(".")
        return remainder[0].upper() + remainder[1:] + "."
    return sent


def _front_adverbial(sent: str) -> str:
    """
    Move a short trailing prepositional phrase to the sentence front.
    Only fires on sentences longer than 15 words with a clean trailing phrase.
    e.g. 'Cells divide rapidly under hypoxic conditions.'
         → 'Under hypoxic conditions, cells divide rapidly.'
    """
    words = sent.split()
    if len(words) < 15:
        return sent

    pat = re.compile(
        r"^(.{30,}?)\s+(under|in|through|via|across|within)\s+([^,.]{4,35})\.$",
        re.IGNORECASE,
    )
    m = pat.match(sent.strip())
    if m:
        body   = m.group(1).strip()
        prep   = m.group(2)
        phrase = m.group(3).strip()
        body   = body[0].lower() + body[1:]
        return f"{prep.capitalize()} {phrase}, {body}."
    return sent


def _split_overlong(sent: str) -> str:
    """
    Split sentences over 35 words at the nearest comma to the midpoint.
    Produces two grammatically complete sentences.
    Does NOT add semicolons — splits cleanly.
    """
    words = sent.split()
    if len(words) < 35:
        return sent

    half = len(words) // 2
    for delta in range(0, half):
        for idx in [half + delta, half - delta]:
            if 0 < idx < len(words):
                if words[idx - 1].endswith(","):
                    left  = " ".join(words[:idx]).rstrip(",") + "."
                    right = " ".join(words[idx:])
                    if right:
                        right = right[0].upper() + right[1:]
                        if right[-1] not in ".!?":
                            right += "."
                        return left + " " + right
    return sent


def structural_transform(sent: str, index: int) -> str:
    """
    Apply one structural transformation per sentence.
    Transformations are gated — only fire when criteria are met.
    No punctuation injection, no filler, no asides.
    """
    s = sent.strip()
    s = _flip_cleft(s)
    s = _split_overlong(s)
    if index % 4 == 0:
        s = _front_adverbial(s)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


# ===========================================================================
# ===== BURSTINESS ENGINE — natural rhythm, not mechanical compression ======
# ===========================================================================

def syntactic_burstiness_engine(sentences: List[str]) -> List[str]:
    """
    Creates sentence-length variation through selective compression only.
    Fires conservatively — only compresses sentences that are genuinely
    too long. Does not inject fillers, fragments, or semicolons.
    """
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result      = []

    for i, sent in enumerate(sentences):
        words       = sent.split()
        current_len = len(words)

        # Only compress sentences that are genuinely long (> 32 words)
        if current_len > 32 and i % 3 == 0:
            target = max(int(current_len * 0.80), 18)
            sent   = " ".join(words[:target]).rstrip(",;—")
            if sent[-1] not in ".!?":
                sent += "."

        # Mild compression for moderately long sentences on every 5th
        elif current_len > 22 and i % 5 == 1:
            target = max(int(current_len * 0.72), 12)
            sent   = " ".join(words[:target]) + "."

        # All other sentences: pass through completely unchanged
        # Mistral's own length variation handles the rhythm

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        result.append(sent)

    # Keep total word count drift within 8%
    new_total = sum(count_words(s) for s in result)
    if abs(new_total - total_words) > int(total_words * 0.08):
        diff = new_total - total_words
        if diff > 0:
            idx = max(range(len(result)), key=lambda x: count_words(result[x]))
            w   = result[idx].split()
            result[idx] = " ".join(w[:max(len(w) - diff, 4)]) + "."

    return result


# ===========================================================================
# ===== OBFUSCATION LAYER ===================================================
# Conservative rate. Semicolons only at genuine clause boundaries.
# Hedging parentheticals used at most once per call.
# ===========================================================================

MODIFICATION_RATE   = 0.25   # 1 in 4 sentences maximum
_hedging_used_flag  = False   # module-level flag — reset per paragraph


def final_obfuscation_layer(text: str, rate: float = MODIFICATION_RATE) -> str:
    global _hedging_used_flag
    _hedging_used_flag = False

    sentences = split_sentences(text)
    processed = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words or len(words) < 8:
            processed.append(sent)
            continue

        if random.random() > rate:
            processed.append(sent)
            continue

        technique = i % 3

        # Technique 0: semicolon substitution — only between two full clauses
        # Requires both sides to have a verb; prevents "word; word" patterns
        if technique == 0 and len(words) > 14 and "," in sent:
            parts = sent.split(",", 1)
            left  = parts[0].split()
            right = parts[1].split()
            left_has_verb  = any(w.lower() in {
                "is", "are", "was", "were", "has", "have", "had",
                "contains", "includes", "requires", "provides",
                "shows", "reveals", "controls", "governs", "drives",
            } for w in left)
            right_has_verb = any(w.lower() in {
                "is", "are", "was", "were", "has", "have", "had",
                "contains", "includes", "requires", "provides",
                "shows", "reveals", "controls", "governs", "drives",
            } for w in right)
            if (len(left) >= 6 and len(right) >= 6
                    and left_has_verb and right_has_verb):
                sent = parts[0] + "; " + parts[1].strip()

        # Technique 1: one hedging parenthetical per paragraph, after anchor noun
        elif (technique == 1
              and len(words) > 8
              and not _hedging_used_flag):
            for word in words:
                clean = word.lower().strip(",.!?;:")
                if clean in ANCHOR_NOUNS:
                    sent = _insert_parenthetical_after_noun(
                        sent, clean, get_hedging_parenthetical()
                    )
                    _hedging_used_flag = True
                    break

        # Technique 2: clause split at subordinating conjunction
        elif technique == 2 and len(words) > 16:
            break_words = {"which", "where", "when", "while", "although"}
            for idx, word in enumerate(words):
                if word.lower() in break_words and 5 < idx < len(words) - 5:
                    fragment  = " ".join(words[:idx]).rstrip(",") + ". "
                    remainder = " ".join(words[idx:])
                    remainder = remainder[0].upper() + remainder[1:]
                    sent = fragment + remainder
                    break

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        processed.append(sent)

    return " ".join(processed)


# ===========================================================================
# ===== SIGNPOST LAYER — very low rate, stable openers only ================
# ===========================================================================

SIGNPOST_RATE = 0.10   # 1 in 10 sentences maximum


def apply_signpost_openers(text: str, rate: float = SIGNPOST_RATE) -> str:
    sentences = split_sentences(text)
    processed = []
    for sent in sentences:
        words = sent.split()
        if (len(words) > 7
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
    """
    Fallback: vocabulary substitution and light restructuring only.
    No filler appended, no stylistic markers injected.
    """
    words = sent.split()
    if not words:
        return sent

    h = sent
    for pattern, choices in _COMPILED_REPLACEMENTS.items():
        h = pattern.sub(lambda m, c=choices: random.choice(c), h)

    # Light structural variation — compress only, no expansion
    if index % 3 == 0 and len(words) > 10:
        h = " ".join(words[:7]) + "."
    elif index % 3 == 2 and "," in h and len(words) > 12:
        # Only replace comma with semicolon when both sides are substantial
        parts = h.split(",", 1)
        if len(parts[0].split()) >= 5 and len(parts[1].split()) >= 5:
            h = parts[0] + ";" + parts[1]

    if (index % 8 == 0
            and len(words) > 6
            and not is_markdown_heading(h)
            and not is_markdown_list(h)):
        h = _prepend_signpost(h, get_signpost_opener())

    return h


# ===========================================================================
# ===== SYSTEM PROMPTS — Two modes ==========================================
# ===========================================================================

_SYSTEM_STEALTH = """You are an elite academic editor and AI-detection bypass specialist fluent across all scholarly disciplines — STEM, social sciences, humanities, law, business, and education.

Your task: rewrite AI-generated academic text so it reads as natural human writing and scores 0% on all AI detectors (Turnitin AI, ZeroGPT, GPTZero, Copyleaks, Winston AI, Originality.ai) while meeting peer-reviewed journal standards.

━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 0 — DOMAIN DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing, identify the field: biology, medicine, neuroscience, CS, engineering, history, literature, philosophy, sociology, economics, law, education, or other. Use ONLY terminology native to that field. Never import vocabulary from an unrelated discipline.

━━━━━━━━━━━━━━━━━━━━━━━━━
TONE STABILITY — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━
Maintain one consistent tone throughout: the tone of a senior academic writing for their specialist peers. Do NOT mix:
- Undergraduate explanation with journal abstraction
- Rhetorical commentary with technical exposition
- Journalistic phrasing with scientific precision

Every sentence must sound like it came from the same writer, in the same register, in the same document.

━━━━━━━━━━━━━━━━━━━━━━━━━
SENTENCE LENGTH VARIATION
━━━━━━━━━━━━━━━━━━━━━━━━━
AI detectors measure burstiness — how uniform sentence lengths are. You must vary lengths naturally:
- Short (5–10 words): used after a dense technical statement to give the reader air
- Medium (13–20 words): the default register for analytical exposition
- Long (22–32 words): used for complex causal or conditional explanations, never longer
- NEVER produce three consecutive sentences of similar length
- Do NOT manufacture brevity — shorten only when the logic genuinely calls for it

━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━
Apply these where the text naturally permits — not mechanically on every sentence:
1. Vary sentence openings: subject-first, short participial phrase, fronted prepositional phrase, direct object fronted for emphasis — never the same pattern twice in a row
2. Remove "It is X that..." cleft constructions — rewrite as direct statements
3. Break genuinely overlong sentences (>32 words) into two complete sentences
4. Shift between active and passive voice naturally — do not force either
5. Do NOT insert em-dashes, asides, interruption phrases, or rhetorical commentary

━━━━━━━━━━━━━━━━━━━━━━━━━
LEXICAL DENSITY
━━━━━━━━━━━━━━━━━━━━━━━━━
Balance technical vocabulary with accessible explanation. After a dense technical concept, follow with a clear, direct statement. Do NOT stack three or more dense technical sentences without a relief sentence. Clarity ranks above sophistication.

━━━━━━━━━━━━━━━━━━━━━━━━━
BANNED PHRASES — never use
━━━━━━━━━━━━━━━━━━━━━━━━━
delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, leverage, synergy, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is crucial to note, it is worth noting, as mentioned earlier, it should be noted, indeed, arguably, significantly, in addition, plays a crucial role, plays a vital role, plays a key role, it is well established, it is clear that, it is evident that, needless to say, in this regard, with respect to, as such, thus far, heretofore, it can be seen that, it is noteworthy, one must consider, it is imperative, merely scratches the surface, and this is key, navigate

━━━━━━━━━━━━━━━━━━━━━━━━━
BANNED SENTENCE OPENERS
━━━━━━━━━━━━━━━━━━━━━━━━━
Never start a sentence with: Furthermore, Moreover, Consequently, Additionally, Subsequently, Nevertheless, Notwithstanding, Accordingly, Henceforth, Heretofore, Indeed, Significantly, Importantly.
Let the logic of one sentence drive the opening of the next naturally.

━━━━━━━━━━━━━━━━━━━━━━━━━
VERB VARIATION
━━━━━━━━━━━━━━━━━━━━━━━━━
Never repeat the same main verb twice in one paragraph. Rotate naturally:
Science/Bio:   regulates → controls → governs → modulates → mediates
Social/Policy: influences → shapes → determines → drives → conditions
Humanities:    argues → contends → posits → maintains → asserts
Tech/Data:     processes → executes → evaluates → optimizes → transforms
Business:      generates → yields → produces → drives → sustains
General:       shows → reveals → indicates → reflects → confirms

━━━━━━━━━━━━━━━━━━━━━━━━━
CONCLUSION RULE
━━━━━━━━━━━━━━━━━━━━━━━━━
Concluding paragraphs must use shorter, more direct sentences. Do not end with a list of parallel implications. Synthesize — do not summarize mechanically.

━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT CONSTRAINTS — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━
1. Word count: match each original sentence within +/- 2 words
2. Zero omission: every fact, figure, named entity, citation, qualifier, and variable must survive intact. Restructuring is permitted; omission is not
3. Citations: keep (Author, 2020), [1], [1-3] exactly as written
4. Markdown: preserve all # headings, ## subheadings, * bullets, 1. numbered lists exactly
5. Register: write at the level of a tenured professor publishing in their field

OUTPUT ONLY VALID JSON — no explanation, no preamble, no markdown fences:
{"processed_paragraphs":[{"sentences":[{"original":"exact original text","humanized":"rewritten sentence","alternatives":["alt1","alt2","alt3"]}]}]}"""


_SYSTEM_HUMANIZE = """You are an elite academic editor fluent across all scholarly disciplines — STEM, social sciences, humanities, law, business, and education.

Rewrite AI-generated academic text to read naturally, as though written by a leading human expert in the field, while maintaining peer-reviewed journal standards.

DOMAIN DETECTION (mandatory first):
Identify the field before writing. Use only field-native terminology. Maintain a single consistent academic tone throughout — do not mix registers.

SENTENCE RHYTHM:
Vary lengths: short (5–10 words) after dense statements; medium (13–20 words) as the default; long (22–32 words) for complex explanations. Never three consecutive similar-length sentences.

DICTION — remove these phrases entirely:
delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, landscape, tapestry, robust, holistic, leverage, synergy, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, significantly, in addition, plays a crucial/vital/key role, needless to say, it is clear that, as such, heretofore, it is noteworthy, merely scratches the surface, and this is key.

LEXICAL DENSITY:
Balance technical jargon with direct clarification. After a complex concept, follow with a direct, accessible statement. Do not stack jargon.

TRANSITIONS:
No mechanical connectors at sentence starts (Furthermore, Moreover, Consequently, Additionally, Subsequently, Nevertheless). Let logic drive the next sentence naturally.

VERB VARIATION:
Never repeat the same verb twice in a paragraph.

CONTENT CONSTRAINTS:
- Match word count within +/- 2 words per sentence
- Zero omission of facts, figures, citations, named entities
- Preserve all markdown headings and list items exactly
- Write at tenured-professor level

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
                f"Rewrite to match EXACTLY {oc} words (+/- 2 words). "
                f"Do NOT omit any facts, figures, named entities, or qualifications. "
                f"Maintain stable academic register and domain-appropriate terminology. "
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
                        "Rewrite this academic text following all guidelines. "
                        "Word counts per sentence are in [brackets] — match within +/- 2 words. "
                        "Preserve every fact, figure, citation, and named entity without exception. "
                        "Maintain one stable academic register throughout — do not shift tone. "
                        "Vary sentence lengths naturally. "
                        "Remove all banned phrases and transitional openers. "
                        "Preserve all markdown headings and list items exactly:\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.70,
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

        for j, sent_data in enumerate(para.get("sentences", [])):
            orig = sent_data.get("original", "")
            h    = sent_data.get("humanized", "") or local_humanize(orig, j)
            h    = correction_loop(orig, h)
            h    = validate_and_correct_length(orig, h, max_diff=3)
            # Structural transforms — clean, grammar-safe only
            h    = structural_transform(h, j)
            h    = validate_and_correct_length(orig, h, max_diff=3)
            para_sentences.append({
                "orig":     orig,
                "hum":      h,
                "raw_alts": sent_data.get("alternatives", [])[:3],
            })

        # Burstiness — conservative compression only
        humanized_only  = [s["hum"] for s in para_sentences]
        burst_sentences = syntactic_burstiness_engine(humanized_only)

        for j, (sent_data, h) in enumerate(zip(para_sentences, burst_sentences)):
            orig_sent = sent_data["orig"]
            h = validate_and_correct_length(orig_sent, h, max_diff=3)
            h = apply_signpost_openers(h)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            h = validate_and_correct_length(orig_sent, h, max_diff=3)
            score = score_sentence(h)

            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                alt = alt or local_humanize(orig_sent, idx + 100)
                alt = correction_loop(orig_sent, alt)
                alt = validate_and_correct_length(orig_sent, alt, max_diff=3)
                alt = structural_transform(alt, idx)
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

            para_sentences[j] = SentenceData(
                id=f"p{i}-s{j}",
                original=orig_sent,
                humanized=h,
                alternatives=unique_alts[:3],
                score=score,
            )

        result.append(ParagraphData(id=f"para-{i}", sentences=para_sentences))

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
    all_s     = [s for p in processed for s in p.sentences]
    avg       = sum(s.score for s in all_s) / len(all_s) if all_s else 0
    return ProcessResponse(
        processed_paragraphs=processed,
        total_sentences=len(all_s),
        avg_score=round(avg, 1),
    )

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model":  "mistral-large-latest",
        "modes":  ["stealth", "humanize"],
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
