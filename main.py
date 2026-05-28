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

class ProcessResponse(BaseModel):
    processed_paragraphs: List[ParagraphData]
    total_sentences: int
    avg_score: float


# ===== VOCABULARY SYSTEM =====

STRUCTURAL_NOUNS = {
    "framework", "methodology", "nexus", "phenomenon", "correlate", "paradigm",
    "trajectory", "substrate", "interface", "topology", "architecture", "mechanism",
    "apparatus", "ensemble", "configuration", "modality", "repertoire", "contingency",
    "disposition", "gradient", "recursion", "hierarchy", "manifold", "schema",
    "ontology", "taxonomy", "morphology", "anatomy", "physiology", "homeostasis"
}

ANCHOR_NOUNS = {
    "cerebellum", "medulla", "pons", "cortex", "tracts", "nuclei", "nerves",
    "arteries", "pyramids", "olives", "structure", "organ", "system", "pathway",
    "mechanism", "framework", "apparatus", "substrate", "topology", "interface",
    "nucleus", "ganglion", "plexus", "fasciculus", "lamina", "sulcus", "gyrus",
    "model", "theory", "argument", "concept", "process", "method", "approach",
    "variable", "factor", "element", "component", "dimension", "aspect",
    "institution", "policy", "context", "principle", "evidence", "data",
    "analysis", "result", "finding", "outcome",
}

TRANSITIONAL_OPENERS = {
    "furthermore", "moreover", "however", "therefore", "thus", "consequently",
    "additionally", "crucially", "subsequently", "nevertheless",
    "notwithstanding", "accordingly"
}

AI_TELL_PHRASES = {
    "delve", "testament", "pivotal", "moreover", "furthermore",
    "it is important to note", "it is crucial to note", "in conclusion",
    "landscape", "tapestry", "beacon", "underscore", "shed light on", "navigate",
    "ever-evolving", "multifaceted", "intricate", "robust", "leverage", "holistic",
    "paradigm", "synergy", "stakeholder", "crucially", "underscoring",
    "operating continuously without conscious oversight", "this structure",
    "that structure", "the present", "the indicated", "the respective",
    "it is worth noting", "as mentioned earlier", "it should be noted",
    "indeed", "arguably", "significantly",
    "in the realm of", "in today's world", "in the world of",
    "it goes without saying", "needless to say", "as a matter of fact",
    "it can be seen that", "it is clear that", "it is evident that",
    "play a crucial role", "plays a crucial role", "play a key role",
    "plays a key role", "play an important role", "plays an important role",
    "of utmost importance", "it is essential to", "it is necessary to",
    "in order to", "due to the fact that", "in the event that",
    "in terms of", "with respect to", "with regard to",
    "a wide range of", "a wide variety of", "a number of",
    "in a variety of ways", "at the end of the day",
    "the fact that", "it can be argued", "one could argue",
    "as previously mentioned", "as stated above", "as discussed above",
    "in light of the above", "based on the above",
    "comprehensive", "groundbreaking", "innovative", "state-of-the-art",
    "cutting-edge", "revolutionary", "transformative", "unprecedented",
    "foster", "facilitate", "utilize", "implementation", "leverage",
    "seamlessly", "streamline", "optimize", "enhance",
}


# ===========================================================================
# ===== DOMAIN-AWARE FILLER SYSTEM ==========================================
# ===========================================================================

# Parentheticals that can follow technical noun phrases without sounding jarring.
# All entries are designed to read naturally in the middle of a sentence.
HEDGING_PARENTHETICALS: list = [
    "(in most cases)",
    "(broadly speaking)",
    "(on balance)",
    "(to some degree)",
    "(in principle)",
    "(under typical conditions)",
    "(in relative terms)",
    "(with few exceptions)",
    "(empirically speaking)",
    "(all else being equal)",
    "(in general terms)",
    "(by most accounts)",
    "(from a functional standpoint)",
    "(by current consensus)",
    "(in practical terms)",
]

# Signpost openers — used ONLY when the opener is logically warranted by the
# sentence's actual content. The caller must verify relevance before applying.
SIGNPOST_OPENERS: list = [
    "Interestingly,",
    "Notably,",
    "In practice,",
    "Under these conditions,",
    "Specifically,",
    "In effect,",
    "Conversely,",
    "As expected,",
    "In this context,",
    "Consequently,",
    "In particular,",
    "By comparison,",
    "Naturally,",
    "Taken together,",
    "On closer inspection,",
    "For this reason,",
    "By extension,",
    "From this perspective,",
    "With this in mind,",
    "Alongside this,",
    "At its core,",
    "Fundamentally,",
]

_BIO_FILLERS: list = [
    "through integrated feedback loops",
    "via polysynaptic relay pathways",
    "under homeostatic regulation",
    "through descending cortical input",
    "via ascending somatosensory relays",
    "contingent on afferent signal integrity",
    "across distributed neural assemblies",
    "through reciprocal thalamocortical projections",
    "via efferent motor output channels",
    "through coordinated synaptic transmission",
    "via chemokine-guided cellular recruitment",
    "through paracrine intercellular signaling",
    "via post-translational protein modification",
]

_TECH_FILLERS: list = [
    "within tightly constrained computational parameters",
    "via iterative algorithmic refinement",
    "under standard experimental conditions",
    "through layered abstraction hierarchies",
    "across distributed processing nodes",
    "under defined boundary conditions",
    "through recursive logical decomposition",
    "contingent on model convergence criteria",
    "within the defined state space",
    "through probabilistic inference mechanisms",
    "under controlled simulation parameters",
    "via adaptive error-correction protocols",
]

_HUMANITIES_FILLERS: list = [
    "within prevailing theoretical frameworks",
    "through established socio-cultural paradigms",
    "contingent on historical contextual variables",
    "through critical hermeneutic engagement",
    "via discursive power structures",
    "through comparative textual analysis",
    "via intertextual referential networks",
    "through phenomenological interpretive frameworks",
    "across competing scholarly genealogies",
    "through narratological structural analysis",
]

_SOCIAL_FILLERS: list = [
    "through established socio-institutional mechanisms",
    "within prevailing policy frameworks",
    "contingent on baseline contextual variables",
    "via iterative social feedback processes",
    "across macro- and micro-level analytical scales",
    "under conditions of institutional constraint",
    "via incentive-compatible behavioral mechanisms",
    "within normative regulatory environments",
    "through mediating psychosocial pathways",
    "across heterogeneous population subgroups",
]

_NATURAL_FILLERS: list = [
    "through thermodynamically driven processes",
    "via molecular diffusion gradients",
    "under equilibrium state conditions",
    "through biogeochemical cycling pathways",
    "across spatiotemporal ecological gradients",
    "via catalytic reaction intermediates",
    "within energetically bounded system states",
    "through coupled atmospheric-oceanic dynamics",
    "contingent on ambient environmental conditions",
    "via electrochemical potential gradients",
]

_EDUCATION_FILLERS: list = [
    "through scaffolded instructional sequences",
    "via formative assessment feedback loops",
    "under constructivist pedagogical frameworks",
    "across differentiated learning modalities",
    "through metacognitive self-regulatory strategies",
    "via zone-of-proximal-development scaffolding",
    "through inquiry-based instructional design",
    "via deliberate practice and spaced repetition",
]

_LAW_FILLERS: list = [
    "within applicable statutory and regulatory frameworks",
    "contingent on jurisdictional precedent",
    "through established common law doctrine",
    "via judicial interpretive mechanisms",
    "under constitutional due process constraints",
    "through adversarial procedural safeguards",
    "via proportionality and balancing tests",
    "through equitable remedial discretion",
]

_BUSINESS_FILLERS: list = [
    "through market-driven competitive mechanisms",
    "via resource allocation optimization processes",
    "contingent on supply-chain operational integrity",
    "through risk-adjusted return optimization",
    "within evolving competitive market landscapes",
    "through stakeholder value alignment strategies",
    "under conditions of information asymmetry",
    "via transaction cost minimization mechanisms",
    "through behavioral economic decision frameworks",
]

_UNIVERSAL_FILLERS: list = [
    "through inherently integrated processes",
    "within clearly defined boundaries",
    "under normal operational conditions",
    "via closely coordinated internal mechanisms",
    "contingent on underlying structural integrity",
    "across multiple interconnected dimensions",
    "through systematically organized pathways",
    "within established theoretical parameters",
    "via well-documented empirical patterns",
    "through convergent lines of evidence",
    "across rigorously controlled conditions",
    "within methodologically defined constraints",
    "through complementary analytical approaches",
]

_DOMAIN_KEYWORDS: dict = {
    "bio": [
        "cell", "cells", "neural", "neuron", "neurons", "brain", "cortex", "body",
        "system", "physiological", "anatomy", "anatomical", "organ", "tissue",
        "gene", "genetic", "protein", "enzyme", "receptor", "synapse", "axon",
        "dendrite", "hormone", "metabolism", "homeostasis", "pathology", "clinical",
        "medical", "patient", "disease", "diagnosis", "treatment", "surgery",
        "immune", "blood", "cardiac", "cerebellum", "medulla", "pons", "cortical",
        "nucleus", "nuclei", "tract", "tracts", "nerve", "nerves", "spinal",
        "vascular", "artery", "arteries", "vein", "veins", "muscle", "skeletal",
        "cellular", "molecular", "biochemical", "pharmacological", "therapeutic",
    ],
    "tech": [
        "data", "dataset", "model", "models", "algorithm", "algorithms", "results",
        "analysis", "variable", "variables", "test", "compute", "computation",
        "software", "code", "program", "function", "parameter", "neural network",
        "machine learning", "deep learning", "artificial intelligence", "database",
        "query", "server", "api", "protocol", "encryption", "bandwidth",
        "latency", "throughput", "simulation", "sensor",
        "signal", "frequency", "circuit", "voltage", "current", "resistance",
        "matrix", "vector", "tensor", "gradient", "loss function", "accuracy",
        "precision", "recall", "classification", "regression", "clustering",
        "pipeline", "deployment", "scalability", "architecture", "engineering",
    ],
    "humanities": [
        "poem", "poetry", "novel", "narrative", "text", "texts", "author",
        "literature", "literary", "history", "historical", "culture", "cultural",
        "philosophy", "philosophical", "theory", "theoretical", "argument",
        "discourse", "ideology", "ideological", "hermeneutics", "phenomenology",
        "ontology", "epistemology", "ethics", "aesthetic", "aesthetics",
        "canon", "genre", "rhetoric", "metaphor", "symbol", "symbolism",
        "interpretation", "archive", "manuscript", "source", "period",
        "movement", "tradition", "mythology", "religion", "theology",
        "identity", "representation", "colonialism", "postcolonial", "modernity",
        "postmodern", "structuralism", "deconstruction", "semiotics",
    ],
    "social": [
        "society", "social", "policy", "policies", "government", "political",
        "economics", "economic", "market", "markets", "behavior", "behaviour",
        "psychology", "psychological", "sociology", "sociological", "population",
        "survey", "questionnaire", "interview", "participant", "participants",
        "sample", "demographic", "inequality", "poverty", "wealth", "income",
        "race", "gender", "class", "ethnicity", "community", "communities",
        "institution", "institutions", "organization", "organizations",
        "attitude", "attitudes", "perception", "perceptions", "norm", "norms",
        "vote", "voter", "election", "democracy", "governance", "legislation",
        "welfare", "healthcare", "education", "employment", "labor", "labour",
        "regression", "coefficient", "correlation", "effect size", "p-value",
    ],
    "natural": [
        "climate", "environment", "environmental", "ecology", "ecological",
        "species", "biodiversity", "habitat", "ecosystem", "evolution",
        "chemistry", "chemical", "reaction", "compound", "molecule", "molecules",
        "atom", "atoms", "element", "elements", "periodic", "quantum", "physics",
        "energy", "thermodynamics", "entropy", "kinetics", "dynamics",
        "geology", "geologic", "sediment", "tectonic", "oceanic", "atmospheric",
        "carbon", "nitrogen", "oxygen", "hydrogen", "temperature", "pressure",
        "wavelength", "radiation", "electromagnetic", "gravitational", "force",
        "mass", "velocity", "acceleration", "momentum", "photon", "electron",
        "proton", "neutron", "isotope", "radioactive", "catalyst",
        "concentration", "solution", "solvent", "polymer", "organic",
    ],
    "education": [
        "student", "students", "teacher", "teachers", "classroom", "curriculum",
        "learning", "learner", "learners", "pedagogy", "pedagogical",
        "instruction", "instructional", "assessment", "literacy", "numeracy",
        "school", "university", "college", "course", "module", "lesson",
        "scaffolding", "feedback", "motivation", "engagement", "retention",
        "cognition", "cognitive", "metacognition", "self-regulation",
        "constructivism", "zone of proximal development", "differentiation",
    ],
    "law": [
        "law", "legal", "court", "courts", "judge", "judges", "judicial",
        "statute", "statutes", "legislation", "regulation", "regulations",
        "contract", "contracts", "liability", "tort", "plaintiff", "defendant",
        "jurisdiction", "precedent", "constitutional", "rights", "criminal",
        "civil", "procedure", "evidence", "testimony", "ruling", "verdict",
        "appeal", "appellate", "compliance", "enforcement", "jurisprudence",
        "doctrine", "equity", "remedy",
    ],
    "business": [
        "business", "firm", "firms", "company", "companies", "profit",
        "revenue", "cost", "costs", "investment", "investor", "finance",
        "financial", "accounting", "audit", "budget", "fiscal", "strategy",
        "strategic", "management", "manager", "leadership", "supply chain",
        "logistics", "operations", "performance", "kpi", "shareholder",
        "equity", "asset", "liability", "competition", "competitive",
        "industry", "sector", "gdp", "inflation", "monetary", "trade",
        "export", "import",
    ],
}

_FILLER_MAP: dict = {
    "bio":        _BIO_FILLERS,
    "tech":       _TECH_FILLERS,
    "humanities": _HUMANITIES_FILLERS,
    "social":     _SOCIAL_FILLERS,
    "natural":    _NATURAL_FILLERS,
    "education":  _EDUCATION_FILLERS,
    "law":        _LAW_FILLERS,
    "business":   _BUSINESS_FILLERS,
    "universal":  _UNIVERSAL_FILLERS,
}


def _detect_domain(sentence: str) -> str:
    text = sentence.lower()
    scores: dict = {domain: 0 for domain in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] >= 2 else "universal"


def get_filler_phrase(sentence: str = "") -> str:
    domain = _detect_domain(sentence) if sentence else "universal"
    pool = _FILLER_MAP.get(domain, _UNIVERSAL_FILLERS)
    # 25% chance to use universal pool for natural unpredictability
    if random.random() < 0.25:
        pool = _UNIVERSAL_FILLERS
    return random.choice(pool)


def get_hedging_parenthetical() -> str:
    return random.choice(HEDGING_PARENTHETICALS)


def get_signpost_opener() -> str:
    return random.choice(SIGNPOST_OPENERS)


# ===== TEXT SPLITTING =====

_ABBREV_RE = re.compile(
    r"\b(e\.g\.|i\.e\.|et al\.|Fig\.|Dr\.|Prof\.|vs\.|cf\.|ca\.|approx\.)\s"
)

def split_sentences(text: str) -> List[str]:
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected.strip())
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
    stripped = text.strip()
    return stripped.startswith(("* ", "- ")) or bool(re.match(r"^\d+\.", stripped))


# ===== SCORING =====

def score_sentence(sent: str) -> float:
    """
    Lower score = more human. Penalises AI detection markers;
    rewards natural variation. Scale: 0–100.
    """
    s = sent.lower()
    words = sent.split()
    score = 0.0

    # Penalise AI tell phrases heavily
    for tell in AI_TELL_PHRASES:
        if tell in s:
            score += 20

    # Penalise the uniform medium-length band AI overuses
    if 15 <= len(words) <= 22:
        score += 8

    # Penalise formulaic transitional openers
    if words:
        first = words[0].lower().strip(",.!?;:")
        if first in TRANSITIONAL_OPENERS:
            score += 15

    # Penalise low lexical diversity
    if len(words) > 5:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < 0.5:
            score += 12

    # Penalise over-punctuated sentences
    if sent.count(",") > 3 or sent.count(";") > 2:
        score += 12

    if "operating continuously" in s:
        score += 25

    # Penalise very short sentences that aren't structural elements
    if len(words) < 4 and not (sent.startswith("#") or sent.startswith("*")):
        score += 15

    # Penalise heavy passive stacking in short sentences
    passive_markers = {"is", "are", "was", "were", "be", "been", "being"}
    passive_count = sum(1 for w in words if w.lower() in passive_markers)
    if passive_count > 3 and len(words) < 18:
        score += 8

    # Reward natural sentence-ending variety
    if sent.endswith("?") or "—" in sent:
        score = max(0.0, score - 10)

    # Reward authorial voice
    if " we " in s or s.startswith("we ") or " our " in s:
        score = max(0.0, score - 8)

    return min(100.0, max(0.0, float(score)))


def count_words(text: str) -> int:
    return len(text.split())


# ===== LENGTH ENFORCEMENT =====

def enforce_length_constraint(original: str, humanized: str, max_diff: int = 3) -> str:
    """
    Brings humanized within max_diff words of original.
    Trimming always preserves complete sentences by cutting at punctuation
    boundaries rather than hard-slicing at an arbitrary word index.
    """
    orig_count = count_words(original)
    hum_count  = count_words(humanized)

    if abs(orig_count - hum_count) <= max_diff:
        return humanized

    # Too long — trim from the end, preferring clause boundaries
    if hum_count > orig_count + max_diff:
        target = orig_count + max_diff
        words  = humanized.split()
        # Try to trim at a comma or semicolon before target
        trimmed_words = words[:target]
        trimmed = " ".join(trimmed_words).rstrip(",;—")
        return trimmed if trimmed[-1] in ".!?" else trimmed + "."

    # Too short — append a domain-relevant clause only if it won't feel intrusive
    if hum_count < orig_count - max_diff:
        shortfall = orig_count - hum_count
        if shortfall <= 8:  # Only pad small gaps; large gaps indicate a structural problem
            filler = get_filler_phrase(humanized)
            humanized = humanized.rstrip(".") + " " + filler + "."

    return humanized


def validate_and_correct_length(original: str, humanized: str, max_diff: int = 3) -> str:
    if abs(count_words(original) - count_words(humanized)) <= max_diff:
        return humanized
    return enforce_length_constraint(original, humanized, max_diff)


# ===== GRAMMAR-SAFE PARENTHETICAL INSERTION =====

def _insert_parenthetical_after_noun(sent: str, noun: str, parenthetical: str) -> str:
    pattern = re.compile(
        r"(\b" + re.escape(noun) + r"\b)([,\.;:!?]?)",
        re.IGNORECASE
    )
    def replacer(m):
        word, punct = m.group(1), m.group(2)
        if punct:
            return f"{word} {parenthetical}{punct}"
        return f"{word} {parenthetical}"
    return pattern.sub(replacer, sent, count=1)


def _prepend_signpost(sent: str, opener: str) -> str:
    sent   = sent.strip()
    opener = opener.strip()
    if not sent:
        return sent
    if opener.endswith(","):
        sent_body = sent[0].lower() + sent[1:]
    else:
        sent_body = sent
    return f"{opener} {sent_body}"


# ===== OBFUSCATION LAYER =====
# Applies targeted structural edits; avoids the mechanical i%N cycling
# pattern that was visible to AI detectors.

MODIFICATION_RATE = 0.22  # Deliberately conservative

def final_obfuscation_layer(text: str, modification_rate: float = MODIFICATION_RATE) -> str:
    sentences = split_sentences(text)
    processed = []

    for sent in sentences:
        words = sent.split()
        if not words or len(words) < 6:
            processed.append(sent)
            continue

        if random.random() > modification_rate:
            processed.append(sent)
            continue

        # Randomly choose a technique; avoid uniform distribution by weighting
        # techniques that are less likely to corrupt content
        technique = random.choices(
            population=[0, 1, 2, 3],
            weights   =[30, 25, 25, 20],
        )[0]

        # Technique 0: semicolon substitution at a natural comma boundary
        if technique == 0 and "," in sent and len(words) > 12:
            parts = sent.split(",", 1)
            left  = parts[0].split()
            right = parts[1].split()
            if len(left) >= 5 and len(right) >= 5:
                sent = parts[0] + "; " + parts[1].strip()

        # Technique 1: hedging parenthetical after an anchor noun
        elif technique == 1 and len(words) > 8:
            for word in words:
                clean = word.lower().strip(",.!?;:")
                if clean in ANCHOR_NOUNS:
                    sent = _insert_parenthetical_after_noun(
                        sent, clean, get_hedging_parenthetical()
                    )
                    break

        # Technique 2: clause split at a subordinating conjunction
        elif technique == 2 and len(words) > 12:
            break_words = {"which", "where", "when", "while", "although"}
            for idx, word in enumerate(words):
                if word.lower() in break_words and 3 < idx < len(words) - 4:
                    fragment  = " ".join(words[:idx]).rstrip(",") + ". "
                    remainder = " ".join(words[idx:])
                    remainder = remainder[0].upper() + remainder[1:]
                    sent = fragment + remainder
                    break

        # Technique 3: strip leading formulaic AI openers
        elif technique == 3:
            formulaic = [
                "It is important to note that ",
                "It should be noted that ",
                "It is worth noting that ",
                "It is evident that ",
                "It is clear that ",
            ]
            for tell in formulaic:
                if sent.startswith(tell):
                    sent = sent[len(tell)][0].upper() + sent[len(tell) + 1 :]
                    break

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        processed.append(sent)

    return " ".join(processed)


# ===== SIGNPOST LAYER =====
# Applied conservatively and ONLY when the sentence's content logically
# warrants an orienting phrase. Two checks prevent consecutive signposting
# and prevent signposting of very short or structural lines.

SIGNPOST_RATE = 0.10

def apply_signpost_openers(text: str, rate: float = SIGNPOST_RATE) -> str:
    sentences           = split_sentences(text)
    processed           = []
    prev_was_signposted = False

    for sent in sentences:
        words = sent.split()
        eligible = (
            len(words) > 8
            and not is_markdown_heading(sent)
            and not is_markdown_list(sent)
            and not prev_was_signposted
            and random.random() < rate
        )
        if eligible:
            sent               = _prepend_signpost(sent, get_signpost_opener())
            sent               = re.sub(r"\s+", " ", sent).strip()
            if sent and sent[-1] not in ".!?":
                sent          += "."
            prev_was_signposted = True
        else:
            prev_was_signposted = False
        processed.append(sent)

    return " ".join(processed)


# ===== REPETITION ELIMINATION =====

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
        overlap_ratio = len(bigrams & used_bigrams) / len(bigrams) if bigrams else 0

        if overlap_ratio > 0.35 and len(words) > 6:
            # Rather than truncating, flag the sentence so the caller can
            # consider it for replacement; here we just shorten to the first clause.
            comma_pos = sent.find(",")
            if comma_pos > 20:
                sent = sent[:comma_pos] + "."
            else:
                sent = " ".join(sent.split()[:6]) + "."

        used_bigrams.update(bigrams)
        # Reset sliding window every 6 sentences
        if len(processed) > 0 and len(processed) % 6 == 0:
            used_bigrams = set()

        processed.append(sent)

    return " ".join(processed)


# ===== BURSTINESS ENGINE =====
# ──────────────────────────────────────────────────────────────────────────
# DESIGN RATIONALE (revised)
#
# The previous engine hard-truncated sentences on a rigid i%5 cycle.
# This destroyed semantic content and created detectable mechanical rhythm.
#
# The new approach:
#   1. Never truncates below the sentence's natural first complete clause.
#   2. Uses probabilistic length targets instead of a fixed cycle, so the
#      output rhythm is uneven — as human writing is.
#   3. Applies compression only to overlong sentences (>32 words) where
#      trimming at a clause boundary still leaves a complete thought.
#   4. Leaves short sentences untouched rather than padding them.
#   5. Total word-count drift is capped at 8% across the paragraph.
# ──────────────────────────────────────────────────────────────────────────

def _find_clause_boundary(words: List[str], target: int) -> int:
    """
    Returns an index near `target` that falls at a natural clause boundary
    (comma, semicolon, or subordinating conjunction), preferring to stay
    within ±4 words of target. Falls back to target if nothing is found.
    """
    search_start = max(0, target - 4)
    search_end   = min(len(words), target + 4)
    clause_words = {"which", "where", "when", "while", "although", "because",
                    "since", "unless", "until", "if", "as", "though"}

    for i in range(target, search_start - 1, -1):
        if i < len(words):
            w = words[i].lower().strip(",.;:")
            if w in clause_words:
                return i
            if i > 0 and words[i - 1][-1] in ",;":
                return i

    for i in range(target + 1, search_end):
        if i < len(words):
            w = words[i].lower().strip(",.;:")
            if w in clause_words:
                return i
            if i > 0 and words[i - 1][-1] in ",;":
                return i

    return target


def _compress_sentence(sent: str, target_words: int) -> str:
    """
    Trims a sentence to approximately target_words by cutting at a clause
    boundary. Never trims below 8 words to preserve semantic integrity.
    """
    words = sent.split()
    if len(words) <= target_words:
        return sent

    safe_target = max(target_words, 8)
    cut_at      = _find_clause_boundary(words, safe_target)
    trimmed     = " ".join(words[:cut_at]).rstrip(",;—")
    return trimmed if trimmed and trimmed[-1] in ".!?" else trimmed + "."


def syntactic_burstiness_engine(sentences: List[str]) -> List[str]:
    """
    Introduces natural sentence-length variation across a paragraph without
    truncating semantic content. Works by:
      - Compressing only overlong sentences (>32 words) at clause boundaries.
      - Skipping modification for sentences that already provide rhythm contrast.
      - Using probabilistic length bands rather than a deterministic cycle.
    """
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result      = []
    prev_len    = None  # Track previous sentence length to guide variation

    for sent in sentences:
        words       = sent.split()
        current_len = len(words)

        # Determine whether this sentence should be compressed.
        # Probability of compression increases with sentence length.
        if current_len > 32:
            compress_prob = 0.65
        elif current_len > 24:
            compress_prob = 0.35
        else:
            compress_prob = 0.0  # Never compress short/medium sentences

        if compress_prob > 0 and random.random() < compress_prob:
            # Choose a target that creates contrast with the previous sentence
            if prev_len is not None and prev_len > 20:
                # Previous was long → aim for medium-short
                target = random.randint(10, 16)
            elif prev_len is not None and prev_len < 12:
                # Previous was short → aim for medium
                target = random.randint(15, 22)
            else:
                target = random.randint(12, 20)

            sent = _compress_sentence(sent, target)

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        result.append(sent)
        prev_len = count_words(sent)

    # Cap total word-count drift at 8%
    new_total = sum(count_words(s) for s in result)
    drift     = new_total - total_words
    if abs(drift) > int(total_words * 0.08) and drift > 0:
        # Only trim if we've added words — which shouldn't happen here,
        # but guards against edge cases in filler insertion elsewhere.
        longest_idx = max(range(len(result)), key=lambda i: count_words(result[i]))
        w = result[longest_idx].split()
        trim_to = max(len(w) - drift, 8)
        result[longest_idx] = " ".join(w[:trim_to]).rstrip(",;—") + "."

    return result


# ===== LOCAL FALLBACK =====

_WORD_REPLACEMENTS = {
    r"\bimportant\b":             ["key", "critical", "main", "essential", "central", "primary"],
    r"\bplays a critical role\b": ["is essential", "is vital", "serves as", "underpins"],
    r"\bplays a vital role\b":    ["is essential", "is critical", "serves as", "anchors"],
    r"\bis located\b":            ["lies", "sits", "is found", "is situated", "resides"],
    r"\bis composed of\b":        ["contains", "has", "includes", "comprises", "incorporates"],
    r"\bacts as\b":               ["works as", "functions as", "serves as", "operates as"],
    r"\bdue to\b":                ["because of", "owing to", "as a result of", "stemming from"],
    r"\boverall\b":               ["in sum", "taken together", "collectively", "broadly"],
    r"\badditionally\b":          ["also", "plus", "further", "as well"],
    r"\bhowever\b":               ["yet", "though", "although", "nevertheless", "even so"],
    r"\btherefore\b":             ["thus", "hence", "so", "accordingly", "as such"],
    r"\bconsequently\b":          ["as a result", "thereby", "accordingly", "hence"],
    r"\bregulates\b":             ["controls", "governs", "modulates", "directs", "coordinates"],
    r"\bcontains\b":              ["holds", "possesses", "encompasses", "incorporates", "houses"],
    r"\bresponsible for\b":       ["accountable for", "charged with", "tasked with", "integral to"],
    r"\bassociated with\b":       ["linked to", "tied to", "connected with", "related to", "coupled with"],
    r"\binvolved in\b":           ["engaged in", "participating in", "contributing to", "implicated in"],
    r"\bconsists of\b":           ["comprises", "is made up of", "incorporates", "encompasses"],
    r"\bpart of\b":               ["component of", "element of", "constituent of", "segment of"],
    r"\bfunction\b":              ["role", "purpose", "operation", "activity", "capacity"],
    r"\bstructure\b":             ["anatomy", "architecture", "framework", "morphology", "configuration"],
    r"\bprocess\b":               ["mechanism", "procedure", "pathway", "sequence", "cascade"],
    r"\bcontrol\b":               ["regulation", "management", "oversight", "direction", "governance"],
    r"\bphenomenon\b":            ["occurrence", "event", "manifestation", "observation", "finding"],
    r"\bframework\b":             ["schema", "construct", "paradigm", "architecture", "scaffold"],
    r"\butilize\b":               ["use", "apply", "employ", "draw on"],
    r"\bfacilitate\b":            ["support", "enable", "allow", "help drive"],
    r"\bdemonstrate\b":           ["show", "reveal", "indicate", "confirm"],
    r"\bimplement\b":             ["apply", "adopt", "put in place", "carry out"],
    r"\bhighlight\b":             ["show", "reveal", "point to", "make clear"],
    r"\bexhibit\b":               ["show", "display", "present", "express"],
    r"\baddress\b":               ["tackle", "examine", "treat", "handle"],
    r"\bcomprehensive\b":         ["thorough", "detailed", "full", "broad"],
    r"\bsignificant\b":           ["notable", "marked", "substantial", "considerable"],
    r"\bnovel\b":                 ["new", "original", "distinct", "previously unreported"],
}

_COMPILED_REPLACEMENTS = {
    re.compile(pat, re.IGNORECASE): choices
    for pat, choices in _WORD_REPLACEMENTS.items()
}

def local_humanize(sent: str, index: int) -> str:
    words = sent.split()
    if not words:
        return sent

    h = sent
    for pattern, choices in _COMPILED_REPLACEMENTS.items():
        h = pattern.sub(lambda m, c=choices: random.choice(c), h)

    # Apply structural edits with moderate probability, not on every sentence
    if random.random() < 0.4 and len(words) > 8 and "," in h:
        h = h.replace(",", ";", 1)

    if (random.random() < 0.15
            and len(words) > 5
            and not is_markdown_heading(h)
            and not is_markdown_list(h)):
        h = _prepend_signpost(h, get_signpost_opener())

    return h


# ===== SYSTEM PROMPT =====

SYSTEM = """You are a senior academic editor with 30 years of experience across STEM, social sciences, humanities, law, business, and education. Your task is to rewrite AI-generated academic text so that it reads like the work of a leading human expert — clear, direct, precisely worded, and free of AI detection markers — while preserving every fact, figure, citation, and technical claim in the original.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — DOMAIN DETECTION (do this first, before any rewriting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identify the field: biology, medicine, computer science, engineering, history, literature, philosophy, sociology, economics, law, education, or other. Every lexical and structural choice must be native to that field. Do not import vocabulary from unrelated disciplines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — WORD COUNT PARITY (strict)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Match each "humanized" sentence within ±2 words of its "original." Use precise single nouns and tight verb phrases to hit the target. Do not pad with filler adjectives or adverbs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — SENTENCE LENGTH VARIATION (critical)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Human writers naturally vary sentence length. Mix lengths unpredictably within each paragraph. A string of sentences all between 15 and 22 words is a primary AI detection signal.

IMPORTANT: Vary length through structural choice — clause splitting, apposition, compression — not by truncating content mid-thought. Every sentence must express a complete thought. Do not produce fragments that cut off before the main assertion is made.

Acceptable length targets per sentence (choose unpredictably, not in sequence):
  Long:       25–34 words   (complex sentences with subordinate clauses)
  Medium:     13–20 words   (direct declarative or compound sentences)
  Short:      6–12 words    (punchy follow-up or summary statements)
  Very short: 4–7 words     (emphasis only; use sparingly, at most once per paragraph)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — ELIMINATE AI DICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replace abstract filler that adds sophistication without empirical content. Prefer direct, plain explanation.

ABSOLUTELY BANNED — never use any of the following:
delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm shift, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is crucial to note, it is worth noting, as mentioned earlier, it should be noted, indeed, arguably, significantly, in the realm of, it goes without saying, needless to say, it can be seen that, it is clear that, it is evident that, play a crucial role, play a key role, play an important role, of utmost importance, in order to, due to the fact that, in the event that, in terms of, with respect to, with regard to, a wide range of, a wide variety of, a number of, comprehensive, groundbreaking, innovative, state-of-the-art, cutting-edge, revolutionary, transformative, unprecedented, foster, facilitate, utilize, implementation, leverage, seamlessly, streamline, optimize, enhance, as previously mentioned, as stated above, as discussed above, in light of the above, based on the above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — LEXICAL DENSITY BALANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Human academic prose alternates dense technical passages with direct, accessible ones. After a complex technical claim, provide a concise grounding explanation. This contrast signals expert authorship.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — TRANSITIONS AND VOICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Let the logic of each sentence drive how the next begins. Do not open sentences with transitional adverbs unless the logical relationship genuinely calls for one. Avoid symmetrical paragraph structures — they read as algorithmic.

Use "we" or "our" once or twice per paragraph to introduce authorial voice naturally.
Start a sentence with "But" or "Yet" only when genuine contrast exists — no more than once per paragraph.
Prefer active voice (roughly 60%) over passive (roughly 40%), varying naturally.
Use semicolons to link closely related independent clauses; avoid em-dashes for this purpose.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7 — VERB PHRASE VARIATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Never repeat the same verb phrase twice within one paragraph.

  Science/Bio:    regulates → controls → governs → modulates → mediates → coordinates
  Social/Policy:  influences → shapes → determines → drives → underpins → constrains
  Humanities:     argues → contends → posits → maintains → asserts → suggests
  Tech/Data:      processes → computes → executes → evaluates → transforms → generates
  Business/Econ:  generates → yields → produces → drives → sustains → captures
  General:        shows → indicates → reveals → reflects → confirms → points to

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 8 — CONTENT FIDELITY (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every fact, figure, named entity, qualification, and citation in the original must appear in the rewrite. If a sentence states "X causes Y under condition Z", the rewrite must retain X, Y, and Z. Restructuring is permitted; omission is not. Keep all citations exactly as written: (Author, 2020), [1], [1–3].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 9 — READABILITY AS A SIGNAL OF AUTHENTICITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Readable writing is perceived as more human than heavily compressed or overly intellectualised text. If a sentence sounds unnatural when read aloud, it needs restructuring. Aim for smooth, spoken-academic rhythm. Prioritise clarity and directness over lexical density.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 10 — MARKDOWN PRESERVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Preserve # headings, ## subheadings, * bullet points, and 1. numbered lists exactly as written. Do not convert headings into prose.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 11 — CONCLUSION HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rewrite conclusion sections with stronger stylistic spontaneity and varied rhythm. The conclusion should read as a human expert reflecting on findings, not mechanically recapping them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY valid JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact original text","humanized":"your rewrite","alternatives":["alt1","alt2","alt3"]}]}]}

Rules for alternatives:
  - Each must be a structurally distinct rewrite of the same sentence.
  - Alternatives must vary in sentence architecture, not just vocabulary.
  - All three must preserve the same facts and citations as the original.
  - None may reproduce the banned phrases listed in Step 4."""


# ===== CORRECTION LOOP =====

def correction_loop(original: str, humanized: str, max_attempts: int = 2) -> str:
    orig_count = count_words(original)
    hum_count  = count_words(humanized)

    if abs(orig_count - hum_count) <= 3:
        return humanized

    for attempt in range(max_attempts):
        try:
            prompt = (
                f"The following rewritten academic sentence is outside the allowed length.\n"
                f"Original word count:  {orig_count} words.\n"
                f"Your rewrite count:   {hum_count} words.\n\n"
                f"Original:    \"{original}\"\n"
                f"Your rewrite: \"{humanized}\"\n\n"
                f"Task: Adjust your rewrite to match exactly {orig_count} words (tolerance ±2). "
                f"Preserve elite academic cadence and precise terminology. "
                f"CRITICAL: Do not omit any facts, figures, or named entities from the original. "
                f"The rewrite must express a complete thought — do not cut off mid-sentence. "
                f"Output ONLY the corrected sentence, no quotes, no explanation."
            )
            resp = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            corrected = resp.choices[0].message.content.strip().strip("\"'")
            if abs(orig_count - count_words(corrected)) <= 3:
                return corrected
            humanized  = corrected
            hum_count  = count_words(humanized)
        except Exception as e:
            print(f"Correction loop attempt {attempt + 1} failed: {e}")
            break

    return enforce_length_constraint(original, humanized, max_diff=3)


# ===== MAIN PROCESSING =====

def humanize_with_mistral(paragraphs: List[List[str]], style: str) -> List[ParagraphData]:
    print(f"CALLING MISTRAL with {len(paragraphs)} paragraphs")

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
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Style: {style}\n\n"
                        "Humanize the academic text below. Word counts are in [brackets] — "
                        "match them within ±2 words.\n\n"
                        "Key requirements:\n"
                        "• Every sentence must express a COMPLETE thought. Do not truncate content mid-point.\n"
                        "• Vary sentence length naturally — short sentences must still be complete clauses.\n"
                        "• Preserve every fact, figure, citation, and named entity from the original.\n"
                        "• Replace all banned phrases (see STEP 4 of system instructions).\n"
                        "• Let logical flow, not transitional adverbs, connect sentences.\n"
                        "• Preserve all markdown headings (##, ###) and list items (*, 1.) exactly.\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.70,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()

        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start, end = text.find("{"), text.rfind("}")
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

        for j, sent in enumerate(para.get("sentences", [])):
            orig = sent.get("original", "")
            h    = sent.get("humanized", "") or local_humanize(orig, j)
            h    = correction_loop(orig, h)
            h    = validate_and_correct_length(orig, h, max_diff=3)
            para_sentences.append({
                "orig":     orig,
                "hum":      h,
                "raw_alts": sent.get("alternatives", [])[:3],
            })

        humanized_only  = [s["hum"] for s in para_sentences]
        burst_sentences = syntactic_burstiness_engine(humanized_only)

        for j, (sent_data, h) in enumerate(zip(para_sentences, burst_sentences)):
            # Validate length after burstiness — burstiness may have changed word count
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            # Apply signpost and obfuscation at low, non-mechanical rates
            h = apply_signpost_openers(h)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            # Final length check after all post-processing
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            score = score_sentence(h)

            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                alt = alt or local_humanize(sent_data["orig"], idx + 100)
                alt = correction_loop(sent_data["orig"], alt)
                alt = validate_and_correct_length(sent_data["orig"], alt, max_diff=3)
                alt = final_obfuscation_layer(alt)
                alt = eliminate_repetition(alt)
                alt = validate_and_correct_length(sent_data["orig"], alt, max_diff=3)
                alt = re.sub(r"\s+", " ", alt).strip()
                if alt and alt[-1] not in ".!?":
                    alt += "."
                clean_alts.append(alt)

            orig_lower  = sent_data["orig"].lower().strip()
            unique_alts: List[str] = []
            seen_lowers: set       = set()

            for alt in clean_alts:
                al = alt.lower().strip()
                if al != orig_lower and al not in seen_lowers:
                    unique_alts.append(alt)
                    seen_lowers.add(al)

            seed = 200
            while len(unique_alts) < 3:
                fallback = local_humanize(sent_data["orig"], seed)
                fallback = validate_and_correct_length(sent_data["orig"], fallback, max_diff=3)
                fl = fallback.lower().strip()
                if fl != orig_lower and fl not in seen_lowers:
                    unique_alts.append(fallback)
                    seen_lowers.add(fl)
                seed += 50

            para_sentences[j] = SentenceData(
                id=f"p{i}-s{j}",
                original=sent_data["orig"],
                humanized=h,
                alternatives=unique_alts[:3],
                score=score,
            )

        result.append(ParagraphData(id=f"para-{i}", sentences=para_sentences))

    print(f"RETURNING {len(result)} paragraphs")
    return result


# ===== API ENDPOINTS =====

@app.post("/api/process-text", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    pars      = split_paragraphs(req.text)
    processed = humanize_with_mistral(pars, req.style)
    all_s     = [s for p in processed for s in p.sentences]
    avg       = sum(s.score for s in all_s) / len(all_s) if all_s else 0
    return ProcessResponse(
        processed_paragraphs=processed,
        total_sentences=len(all_s),
        avg_score=round(avg, 1),
    )

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "mistral-large-latest"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
