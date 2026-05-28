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

# AI tell phrases — banned as mechanical crutches, not as legitimate technical terms
AI_TELL_PHRASES = {
    "delve", "testament", "pivotal", "underscore", "shed light on", "navigate",
    "landscape", "tapestry", "beacon", "robust", "holistic", "paradigm", "synergy",
    "stakeholder", "leverage", "multifaceted", "intricate", "ever-evolving",
    "in conclusion", "it is important to note", "it is crucial to note",
    "it is worth noting", "as mentioned earlier", "it should be noted",
    "indeed", "arguably", "in the realm of", "it goes without saying",
    "needless to say", "it can be seen that", "it is clear that",
    "it is evident that", "play a crucial role", "plays a crucial role",
    "play a key role", "plays a key role", "play an important role",
    "plays an important role", "of utmost importance", "in order to",
    "due to the fact that", "in the event that", "in terms of",
    "with respect to", "with regard to", "a wide range of",
    "a wide variety of", "a number of", "in a variety of ways",
    "at the end of the day", "the fact that", "it can be argued",
    "one could argue", "as previously mentioned", "as stated above",
    "as discussed above", "in light of the above", "based on the above",
    "groundbreaking", "state-of-the-art", "cutting-edge", "revolutionary",
    "transformative", "unprecedented", "foster", "seamlessly", "streamline",
    "optimize", "enhance", "demystify",
    # Phrase-level tells
    "operating continuously without conscious oversight", "this structure",
    "that structure", "the present", "the indicated", "the respective",
}


# ===========================================================================
# ===== DOMAIN-AWARE FILLER SYSTEM ==========================================
# ===========================================================================

HEDGING_PARENTHETICALS: list = [
    "(arguably)",
    "(presumably)",
    "(by extension)",
    "(virtually)",
    "(notably)",
    "(evidently)",
    "(characteristically)",
    "(as expected)",
    "(in most cases)",
    "(broadly speaking)",
    "(on balance)",
    "(to some degree)",
    "(in principle)",
    "(to varying extents)",
    "(within reason)",
    "(contextually)",
    "(under typical conditions)",
    "(in relative terms)",
    "(with few exceptions)",
    "(under standard assumptions)",
    "(in theoretical terms)",
    "(empirically speaking)",
    "(all else being equal)",
    "(in general terms)",
    "(as conventionally understood)",
    "(by most accounts)",
    "(from a functional standpoint)",
    "(by current consensus)",
    "(in practical terms)",
    "(in the broader sense)",
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
    "Consequently,",
    "Alternatively,",
    "In particular,",
    "By comparison,",
    "In such cases,",
    "Naturally,",
    "Broadly speaking,",
    "Within this specific context,",
    "Taken together,",
    "Viewed through this lens,",
    "On closer inspection,",
    "As the evidence suggests,",
    "In the present case,",
    "By the same token,",
    "At the same time,",
    "In a related vein,",
    "Against this backdrop,",
    "In light of this,",
    "To this end,",
    "In doing so,",
    "For this reason,",
    "By extension,",
    "From this perspective,",
    "Upon reflection,",
    "With this in mind,",
    "In the same manner,",
    "To a certain extent,",
    "Alongside this,",
    "In broader terms,",
    "Under closer scrutiny,",
    "When considered carefully,",
    "At its core,",
    "From a structural standpoint,",
    "Taken in isolation,",
    "In the aggregate,",
    "Across domains,",
    "Fundamentally,",
    "What matters here is",
    "The key point is that",
    "This is not incidental;",
    "Worth emphasizing here:",
    "The data point to",
    "Structurally speaking,",
    "At a finer level of analysis,",
    "The picture is more nuanced:",
    "This distinction matters because",
    "Looking more closely,",
]

_BIO_FILLERS: list = [
    "through integrated feedback loops",
    "via polysynaptic relay pathways",
    "under homeostatic regulation",
    "through descending cortical input",
    "via ascending somatosensory relays",
    "contingent on afferent signal integrity",
    "across distributed neural assemblies",
    "within tightly regulated homeostatic bounds",
    "through reciprocal thalamocortical projections",
    "under conditions of normal physiological demand",
    "via efferent motor output channels",
    "through coordinated synaptic transmission",
    "under autonomic nervous system oversight",
    "across convergent sensorimotor pathways",
    "through receptor-mediated signal transduction",
    "via chemokine-guided cellular recruitment",
    "under conditions of metabolic equilibrium",
    "through paracrine intercellular signaling",
    "within genetically encoded regulatory networks",
    "via post-translational protein modification",
]

_TECH_FILLERS: list = [
    "within tightly constrained computational parameters",
    "contingent on data stream integrity",
    "via iterative algorithmic refinement",
    "under standard experimental conditions",
    "through layered abstraction hierarchies",
    "across distributed processing nodes",
    "via stochastic gradient optimization",
    "under defined boundary conditions",
    "through recursive logical decomposition",
    "contingent on model convergence criteria",
    "via parallel execution pipelines",
    "within the defined state space",
    "through probabilistic inference mechanisms",
    "across heterogeneous network topologies",
    "under controlled simulation parameters",
    "via adaptive error-correction protocols",
    "within formally specified constraint sets",
    "through modular system decomposition",
    "under worst-case asymptotic bounds",
    "via deterministic finite-state transitions",
]

_HUMANITIES_FILLERS: list = [
    "within prevailing theoretical frameworks",
    "through established socio-cultural paradigms",
    "contingent on historical contextual variables",
    "across distinct analytical dimensions",
    "through critical hermeneutic engagement",
    "via discursive power structures",
    "within historically situated interpretive horizons",
    "through comparative textual analysis",
    "across divergent ideological formations",
    "via dialectical modes of inquiry",
    "within contested epistemic traditions",
    "through sustained close reading practices",
    "across geopolitical and temporal boundaries",
    "via intertextual referential networks",
    "through phenomenological interpretive frameworks",
    "within the logic of the historical archive",
    "across competing scholarly genealogies",
    "through narratological structural analysis",
    "via ideological critique and deconstruction",
    "within dominant and subaltern discourses",
]

_SOCIAL_FILLERS: list = [
    "through established socio-institutional mechanisms",
    "within prevailing policy frameworks",
    "contingent on baseline contextual variables",
    "via iterative social feedback processes",
    "across macro- and micro-level analytical scales",
    "through latent structural inequalities",
    "under conditions of institutional constraint",
    "via incentive-compatible behavioral mechanisms",
    "within normative regulatory environments",
    "through mediating psychosocial pathways",
    "across heterogeneous population subgroups",
    "via rational-choice optimization models",
    "within structurally embedded power relations",
    "through cognitive-behavioral regulatory processes",
    "contingent on socioeconomic baseline conditions",
    "across intersecting axes of identity",
    "via self-reinforcing institutional feedback loops",
    "within bounded rationality frameworks",
    "through stratified sampling methodological designs",
    "across longitudinal observational time points",
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
    "via second-law thermodynamic constraints",
    "across trophic energy transfer levels",
    "through quantum mechanical wave-particle interactions",
    "under standard temperature and pressure conditions",
    "via electrochemical potential gradients",
    "within defined thermodynamic phase boundaries",
    "through stochastic environmental perturbations",
    "across macro- and micro-ecological scales",
    "via radiative energy transfer mechanisms",
    "within geologically constrained timescales",
    "through oxidation-reduction reaction cycles",
]

_EDUCATION_FILLERS: list = [
    "through scaffolded instructional sequences",
    "via formative assessment feedback loops",
    "under constructivist pedagogical frameworks",
    "across differentiated learning modalities",
    "through metacognitive self-regulatory strategies",
    "via zone-of-proximal-development scaffolding",
    "within socially situated learning environments",
    "through inquiry-based instructional design",
    "across culturally responsive curriculum frameworks",
    "via deliberate practice and spaced repetition",
]

_LAW_FILLERS: list = [
    "within applicable statutory and regulatory frameworks",
    "contingent on jurisdictional precedent",
    "through established common law doctrine",
    "via judicial interpretive mechanisms",
    "under constitutional due process constraints",
    "through adversarial procedural safeguards",
    "across distinct jurisprudential traditions",
    "via proportionality and balancing tests",
    "within legislatively defined normative boundaries",
    "through equitable remedial discretion",
]

_BUSINESS_FILLERS: list = [
    "through market-driven competitive mechanisms",
    "via resource allocation optimization processes",
    "contingent on supply-chain operational integrity",
    "across vertically and horizontally integrated structures",
    "through risk-adjusted return optimization",
    "via dynamic capability reconfiguration",
    "within evolving competitive market landscapes",
    "through stakeholder value alignment strategies",
    "under conditions of information asymmetry",
    "via transaction cost minimization mechanisms",
    "across diversified portfolio risk structures",
    "through behavioral economic decision frameworks",
    "contingent on macroeconomic baseline conditions",
    "via financial instrument pricing models",
    "within regulatory compliance frameworks",
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
    "under standard analytical conditions",
    "via well-documented empirical patterns",
    "across rigorously controlled conditions",
    "through convergent lines of evidence",
    "within broadly accepted scholarly norms",
    "via recursive self-reinforcing dynamics",
    "under conditions of internal consistency",
    "through mutually reinforcing causal chains",
    "across diverse empirical contexts",
    "within methodologically defined constraints",
    "through complementary analytical approaches",
    "via interrelated systemic components",
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
        "latency", "throughput", "optimization", "runtime", "compiler",
        "hardware", "processor", "memory", "storage", "simulation", "sensor",
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
    if random.random() < 0.3:
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
# Lower score = more human. Score penalizes AI patterns and rewards natural variation.

def score_sentence(sent: str) -> float:
    s = sent.lower()
    words = sent.split()
    score = 0

    # Penalize AI tell phrases heavily
    for tell in AI_TELL_PHRASES:
        if tell in s:
            score += 20

    # Penalize the "sweet spot" length that AI loves (15-22 words, every sentence)
    if 15 <= len(words) <= 22:
        score += 8

    # Penalize formulaic transitional openers
    if words:
        first = words[0].lower().strip(",.!?;:")
        if first in TRANSITIONAL_OPENERS:
            score += 15

    # Penalize low lexical diversity
    if len(words) > 5:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < 0.5:
            score += 12

    # Penalize over-punctuated sentences (AI stacking clauses)
    if sent.count(",") > 3 or sent.count(";") > 2:
        score += 12

    if "operating continuously" in s:
        score += 25

    # Penalize very short sentences that are not headings/lists (likely truncation artifacts)
    if len(words) < 4 and not (sent.startswith("#") or sent.startswith("*")):
        score += 15

    # Penalize sentences that are all passive + abstract noun combos
    passive_markers = ["is", "are", "was", "were", "be", "been", "being"]
    passive_count = sum(1 for w in words if w.lower() in passive_markers)
    if passive_count > 3 and len(words) < 18:
        score += 8

    # Reward natural sentence-ending variety (questions, dashes, etc.)
    if sent.endswith("?") or "—" in sent:
        score = max(0, score - 10)

    # Reward use of "we" / "our" (human authorial voice)
    if " we " in s or s.startswith("we ") or " our " in s:
        score = max(0, score - 8)

    return min(100.0, max(0.0, float(score)))


def count_words(text: str) -> int:
    return len(text.split())


# ===== GRAMMATICAL COMPLETENESS HELPERS =====

_CLAUSE_BOUNDARY_WORDS = {
    "and", "but", "or", "nor", "for", "yet", "so",
    "which", "where", "when", "while", "although", "because", "since", "if", "unless",
    "whereas", "despite", "although", "though", "even", "provided",
}

_COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "done",
    "shows", "show", "demonstrates", "demonstrate", "indicates", "indicate",
    "reveals", "reveal", "suggests", "suggest", "implies", "imply",
    "confirms", "confirm", "establishes", "establish", "proves", "prove",
    "argues", "argue", "contends", "contend", "posits", "posit",
    "controls", "control", "governs", "govern", "modulates", "modulate",
    "regulates", "regulate", "mediates", "mediate", "coordinates", "coordinate",
    "produces", "produce", "generates", "generate", "induces", "induce",
    "effects", "effect", "drives", "drive", "underpins", "underpin",
    "influences", "influence", "shapes", "shape", "determines", "determine",
    "constrains", "constrain", "processes", "process", "computes", "compute",
    "executes", "execute", "evaluates", "evaluate", "transforms", "transform",
    "yields", "yield", "sustains", "sustain", "captures", "capture",
    "reflects", "reflect", "points", "point", "manifests", "manifest",
    "evidences", "evidence", "derives", "derive", "obtains", "obtain",
    "acquires", "acquire", "maintains", "maintain", "preserves", "preserve",
    "retains", "retain", "requires", "require", "necessitates", "necessitate",
    "provides", "provide", "furnishes", "furnish", "confers", "confer",
    "increases", "increase", "decreases", "decrease", "augment", "augments",
    "diminish", "diminishes", "escalate", "escalates", "decline", "declines",
    "attenuate", "attenuates", "examine", "examines", "investigate", "investigates",
    "ascertain", "ascertains", "elucidate", "elucidates", "commence", "commences",
    "initiate", "initiates", "terminate", "terminates", "conclude", "concludes",
    "cease", "ceases", "promote", "promotes", "enable", "enables", "advance",
    "advances", "further", "furthers", "bolster", "bolsters",
}

_COMMON_NOUNS = {
    "the", "a", "an", "this", "that", "these", "those", "our", "we",
    "it", "its", "they", "their", "them", "he", "she", "his", "her",
    "data", "results", "findings", "analysis", "model", "study", "research",
    "system", "process", "mechanism", "framework", "structure", "method",
    "approach", "theory", "hypothesis", "conclusion", "evidence", "argument",
    "author", "patient", "subject", "participant", "sample", "population",
    "cell", "neuron", "gene", "protein", "enzyme", "organ", "tissue",
    "algorithm", "network", "function", "parameter", "variable", "factor",
    "element", "component", "dimension", "aspect", "institution", "policy",
    "context", "principle", "outcome", "finding", "result", "effect",
}


def _looks_like_verb(word: str) -> bool:
    w = word.lower().strip(",.!?;:")
    if w in _COMMON_VERBS:
        return True
    if w.endswith(("ed", "es", "ing", "ize", "ise", "ify", "ate")):
        return True
    return False


def _looks_like_noun(word: str) -> bool:
    w = word.lower().strip(",.!?;:")
    if w in _COMMON_NOUNS:
        return True
    if w.endswith(("tion", "sion", "ment", "ness", "ity", "ure", "age", "ance", "ence", "dom")):
        return True
    return False


def is_grammatically_complete(sentence: str) -> bool:
    """Lightweight heuristic: sentence must contain at least one likely noun/pronoun
    and one likely verb to count as a complete declarative clause."""
    words = sentence.split()
    if len(words) < 3:
        return False
    has_noun = any(_looks_like_noun(w) for w in words)
    has_verb = any(_looks_like_verb(w) for w in words)
    return has_noun and has_verb


def smart_truncate(sentence: str, target: int) -> str:
    """Truncate at the nearest clause boundary before target, ensuring completeness."""
    words = sentence.split()
    if len(words) <= target:
        return sentence

    best_cut = -1
    # Scan backwards from target to find a safe boundary
    for i in range(min(target, len(words) - 1), 2, -1):
        w = words[i].lower().strip(",.!?;:")
        prev = words[i - 1]

        # Coordinating conjunction preceded by comma
        if w in ("and", "but", "or", "nor", "for", "yet", "so"):
            if prev.endswith(",") or prev.endswith(";"):
                best_cut = i - 1  # cut before the conjunction clause
                break

        # Subordinating conjunctions / relative pronouns
        if w in ("which", "where", "when", "while", "although", "because", "since", "if", "unless", "whereas", "though"):
            if i > 0:
                best_cut = i
                break

        # Semicolon boundary
        if ";" in prev:
            best_cut = i
            break

    if best_cut > 2:
        truncated = " ".join(words[:best_cut]).rstrip(",;—")
        if is_grammatically_complete(truncated):
            return truncated + "."

    # Fallback: if we cannot truncate gracefully, return original rather than fragment
    return sentence


# ===== FILLER GATE =====

def _sentence_needs_filler(sentence: str, original: str) -> bool:
    hum_count  = count_words(sentence)
    orig_count = count_words(original)
    shortfall  = orig_count - hum_count
    already_extended = sentence.rstrip(".!?").endswith(")")
    is_stub          = hum_count < 7
    big_shortfall    = shortfall > 5
    return is_stub and big_shortfall and not already_extended


# ===== LENGTH ENFORCEMENT =====

def enforce_length_constraint(original: str, humanized: str, max_diff: int = 3) -> str:
    orig_count = count_words(original)
    hum_count  = count_words(humanized)

    if abs(orig_count - hum_count) <= max_diff:
        return humanized

    if hum_count > orig_count + max_diff:
        words  = humanized.split()
        keep   = max(orig_count + max_diff - 1, min(orig_count, len(words)))
        # Use smart_truncate instead of naive slicing
        candidate = smart_truncate(humanized, keep)
        if count_words(candidate) <= orig_count + max_diff and is_grammatically_complete(candidate):
            return candidate
        # If smart truncate fails, try naive but ensure completeness
        naive = " ".join(words[:keep]).rstrip(",;—")
        if is_grammatically_complete(naive):
            return naive if naive[-1] in ".!?" else naive + "."
        # If still incomplete, return original humanized (accept length violation over fragment)
        return humanized

    if _sentence_needs_filler(humanized, original):
        humanized = humanized.rstrip(".") + " " + get_filler_phrase(humanized) + "."

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
    sent = sent.strip()
    if not sent:
        return sent
    opener = opener.strip()
    if opener.endswith(","):
        first_char = sent[0]
        rest = sent[1:]
        sent_body = first_char.lower() + rest
    else:
        sent_body = sent
    return f"{opener} {sent_body}"


# ===== OBFUSCATION LAYER =====

MODIFICATION_RATE = 0.28

def final_obfuscation_layer(text: str, modification_rate: float = MODIFICATION_RATE) -> str:
    sentences = split_sentences(text)
    processed = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words or len(words) < 6:
            processed.append(sent)
            continue

        if random.random() > modification_rate:
            processed.append(sent)
            continue

        technique = random.randint(0, 4)

        if technique == 0 and len(words) > 10:
            if "," in sent and len(words) > 12:
                parts = sent.split(",", 1)
                left_words  = parts[0].split()
                right_words = parts[1].split()
                if len(left_words) >= 5 and len(right_words) >= 5:
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

        elif technique == 3 and len(words) > 10:
            if " and " in sent:
                and_pos = sent.find(" and ")
                before  = sent[:and_pos].strip()
                after   = sent[and_pos + 5:].strip()
                clause_verbs = {
                    "is", "are", "was", "were", "has", "have",
                    "controls", "regulates", "modulates", "governs"
                }
                if (len(before.split()) > 4 and len(after.split()) > 4
                        and any(v in before.lower().split() for v in clause_verbs)):
                    sent = sent.replace(" and ", ", consequently, ", 1)

        elif technique == 4 and len(words) > 8:
            passive_pattern = re.compile(
                r"\b(is|are|was|were)\s+(\w+ed)\s+by\s+", re.IGNORECASE
            )
            if passive_pattern.search(sent):
                for tell in ["It is important to note that ", "It should be noted that ",
                             "It is worth noting that ", "It is evident that "]:
                    if sent.startswith(tell):
                        sent = sent[len(tell)].upper() + sent[len(tell)+1:]
                        break

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        processed.append(sent)

    return " ".join(processed)


# ===== SIGNPOST LAYER =====

SIGNPOST_RATE = 0.12

def apply_signpost_openers(text: str, rate: float = SIGNPOST_RATE) -> str:
    sentences = split_sentences(text)
    processed = []
    consecutive_signposted = 0
    for sent in sentences:
        words = sent.split()
        if (len(words) > 5
                and not is_markdown_heading(sent)
                and not is_markdown_list(sent)
                and random.random() < rate
                and consecutive_signposted == 0):
            sent = _prepend_signpost(sent, get_signpost_opener())
            sent = re.sub(r"\s+", " ", sent).strip()
            if sent and sent[-1] not in ".!?":
                sent += "."
            consecutive_signposted = 1
        else:
            consecutive_signposted = 0
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

        if overlap_ratio > 0.3 and len(words) > 6:
            # Instead of blindly chopping to 5 words, smart-truncate to a complete clause
            candidate = smart_truncate(sent, 5)
            if candidate != sent and is_grammatically_complete(candidate):
                sent = candidate
            # If we can't make a complete short version, just leave it

        used_bigrams.update(bigrams)
        if len(processed) > 0 and len(processed) % 5 == 0:
            used_bigrams = set()

        processed.append(sent)

    return " ".join(processed)


# ===== BURSTINESS ENGINE =====
# Now uses smart_truncate and never produces incomplete fragments

def syntactic_burstiness_engine(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result = []

    for i, sent in enumerate(sentences):
        words       = sent.split()
        current_len = len(words)
        cycle = i % 5

        if cycle == 0:
            if current_len > 38:
                target = max(28, int(current_len * 0.80))
                sent = smart_truncate(sent, target)

        elif cycle == 1:
            if current_len > 9:
                target = random.randint(4, 9)
                candidate = smart_truncate(sent, target)
                if candidate != sent:
                    sent = candidate
                # If smart_truncate refuses (no complete clause possible), leave original

        elif cycle == 2:
            if current_len > 20:
                target = random.randint(11, 20)
                candidate = smart_truncate(sent, target)
                if candidate != sent:
                    sent = candidate

        elif cycle == 3:
            if current_len > 6:
                target = random.randint(3, 6)
                candidate = smart_truncate(sent, target)
                if candidate != sent:
                    sent = candidate

        elif cycle == 4:
            if current_len > 35:
                target = max(25, int(current_len * 0.75))
                sent = smart_truncate(sent, target)
            elif current_len > 28 and ";" not in sent:
                mid = current_len // 2
                sent = " ".join(words[:mid]) + "; " + " ".join(words[mid:])

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        result.append(sent)

    # Keep total word count drift within 10%
    new_total = sum(count_words(s) for s in result)
    if abs(new_total - total_words) > int(total_words * 0.1):
        diff = new_total - total_words
        if diff > 0:
            longest_idx = max(range(len(result)), key=lambda i: count_words(result[i]))
            w = result[longest_idx].split()
            candidate = smart_truncate(result[longest_idx], max(len(w) - diff, 5))
            if candidate != result[longest_idx]:
                result[longest_idx] = candidate
            else:
                # If can't truncate gracefully, just trim the end words and hope
                result[longest_idx] = " ".join(w[:max(len(w) - diff, 5)]) + "."

    return result


# ===== JOURNAL-REGISTER VOCABULARY UPGRADE =====

_JOURNAL_SYNONYMS = {
    r"\bshows\b": ["demonstrates", "indicates", "reveals", "evidences", "manifests"],
    r"\bshow\b": ["demonstrate", "indicate", "reveal", "evidence", "manifest"],
    r"\bbig\b": ["substantial", "considerable", "pronounced", "marked"],
    r"\blarge\b": ["substantial", "considerable", "extensive", "expansive"],
    r"\bsmall\b": ["modest", "minor", "minimal", "negligible", "marginal"],
    r"\bchange\b": ["modification", "alteration", "transition", "shift", "variation"],
    r"\bchanges\b": ["modifications", "alterations", "transitions", "shifts", "variations"],
    r"\bmake\b": ["render", "produce", "generate", "induce", "effect"],
    r"\bmakes\b": ["renders", "produces", "generates", "induces", "effects"],
    r"\bget\b": ["obtain", "acquire", "derive", "secure"],
    r"\bgets\b": ["obtains", "acquires", "derives", "secures"],
    r"\bgot\b": ["obtained", "acquired", "derived", "secured"],
    r"\bkeep\b": ["maintain", "preserve", "retain", "sustain"],
    r"\bkeeps\b": ["maintains", "preserves", "retains", "sustains"],
    r"\bput\b": ["place", "position", "insert", "introduce", "situate"],
    r"\bputs\b": ["places", "positions", "inserts", "introduces", "situates"],
    r"\bhelp\b": ["promote", "enable", "advance", "further", "bolster"],
    r"\bhelps\b": ["promotes", "enables", "advances", "furthers", "bolsters"],
    r"\bstart\b": ["commence", "initiate", "originate"],
    r"\bstarts\b": ["commences", "initiates", "originates"],
    r"\bstarted\b": ["commenced", "initiated", "originated"],
    r"\bend\b": ["terminate", "conclude", "cease", "culminate"],
    r"\bends\b": ["terminates", "concludes", "ceases", "culminates"],
    r"\bended\b": ["terminated", "concluded", "ceased", "culminated"],
    r"\bneed\b": ["necessitate", "require", "demand", "call for"],
    r"\bneeds\b": ["necessitates", "requires", "demands", "calls for"],
    r"\bneeded\b": ["necessitated", "required", "demanded"],
    r"\bgive\b": ["provide", "furnish", "impart", "confer", "afford"],
    r"\bgives\b": ["provides", "furnishes", "imparts", "confers", "affords"],
    r"\bgave\b": ["provided", "furnished", "imparted", "conferred"],
    r"\bsay\b": ["state", "assert", "contend", "posit", "maintain"],
    r"\bsays\b": ["states", "asserts", "contends", "posits", "maintains"],
    r"\bsaid\b": ["stated", "asserted", "contended", "posited", "maintained"],
    r"\bthink\b": ["postulate", "hypothesize", "theorize", "propose"],
    r"\bthinks\b": ["postulates", "hypothesizes", "theorizes", "proposes"],
    r"\bthought\b": ["postulated", "hypothesized", "theorized", "proposed"],
    r"\blook at\b": ["examine", "investigate", "scrutinize", "assess", "evaluate"],
    r"\blooks at\b": ["examines", "investigates", "scrutinizes", "assesses", "evaluates"],
    r"\blooked at\b": ["examined", "investigated", "scrutinized", "assessed", "evaluated"],
    r"\bfind out\b": ["ascertain", "determine", "elucidate", "establish", "clarify"],
    r"\bfinds out\b": ["ascertains", "determines", "elucidates", "establishes", "clarifies"],
    r"\bfound out\b": ["ascertained", "determined", "elucidated", "established", "clarified"],
    r"\bcome from\b": ["derive from", "originate from", "stem from", "arise from", "emanate from"],
    r"\bcomes from\b": ["derives from", "originates from", "stems from", "arises from", "emanates from"],
    r"\bcame from\b": ["derived from", "originated from", "stemmed from", "arose from"],
    r"\bgo up\b": ["increase", "escalate", "augment", "accrue", "intensify"],
    r"\bgoes up\b": ["increases", "escalates", "augments", "accrues", "intensifies"],
    r"\bwent up\b": ["increased", "escalated", "augmented", "accrued", "intensified"],
    r"\bgo down\b": ["decrease", "diminish", "decline", "attenuate", "abate"],
    r"\bgoes down\b": ["decreases", "diminishes", "declines", "attenuates", "abates"],
    r"\bwent down\b": ["decreased", "diminished", "declined", "attenuated", "abated"],
    r"\babout\b": ["concerning", "regarding", "pertaining to", "in relation to"],
    r"\blike\b": ["such as", "akin to", "analogous to", "exemplified by"],
    r"\bway\b": ["manner", "fashion", "mode", "modality"],
    r"\bways\b": ["manners", "fashions", "modes", "modalities"],
    r"\bthing\b": ["factor", "element", "component", "consideration", "parameter"],
    r"\bthings\b": ["factors", "elements", "components", "considerations", "parameters"],
    r"\bpart\b": ["component", "constituent", "segment", "portion", "fraction"],
    r"\bparts\b": ["components", "constituents", "segments", "portions", "fractions"],
    r"\bresult\b": ["outcome", "consequence", "sequel", "product", "yield"],
    r"\bresults\b": ["outcomes", "consequences", "sequels", "products", "yields"],
    r"\bbecause\b": ["owing to", "by virtue of", "on account of", "in view of", "given"],
    r"\bso\b": ["therefore", "thus", "accordingly", "consequently", "hence"],
    r"\bbut\b": ["however", "nevertheless", "conversely", "yet", "notwithstanding"],
    r"\balso\b": ["additionally", "furthermore", "moreover", "likewise", "similarly"],
    r"\bgood\b": ["favorable", "advantageous", "beneficial", "optimal", "salutary"],
    r"\bbad\b": ["adverse", "deleterious", "unfavorable", "detrimental", "nocuous"],
    r"\bvery\b": ["highly", "markedly", "substantially", "considerably", "profoundly"],
    r"\bmany\b": ["numerous", "myriad", "a considerable number of", "a multitude of"],
    r"\bsome\b": ["certain", "particular", "specific", "discrete"],
    r"\bmore\b": ["additional", "further", "supplementary", "extra"],
    r"\bless\b": ["diminished", "reduced", "attenuated", "curtailed"],
    r"\bbefore\b": ["prior to", "antecedent to", "preceding", "in advance of"],
    r"\bafter\b": ["subsequent to", "following", "in the wake of", "thereafter"],
    r"\bduring\b": ["throughout", "in the course of", "over the duration of", "for the period of"],
    r"\bbetween\b": ["intervening", "in the interval between", "amid"],
    r"\bunder\b": ["subject to", "in the context of", "beneath", "in accordance with"],
    r"\bover\b": ["above", "in excess of", "spanning", "covering", "throughout"],
}

_JOURNAL_COMPILED = {
    re.compile(pat, re.IGNORECASE): choices
    for pat, choices in _JOURNAL_SYNONYMS.items()
}


def upgrade_journal_register(text: str) -> str:
    """Upgrade common words to formal journal-standard equivalents."""
    for pattern, choices in _JOURNAL_COMPILED.items():
        text = pattern.sub(lambda m, c=choices: random.choice(c), text)
    return text


# ===== LOCAL FALLBACK =====

_WORD_REPLACEMENTS = {
    r"\bimportant\b":        ["key", "critical", "main", "essential", "central", "primary"],
    r"\bplays a critical role\b": ["is essential", "is vital", "serves as", "underpins"],
    r"\bplays a vital role\b":    ["is essential", "is critical", "serves as", "anchors"],
    r"\bis located\b":       ["lies", "sits", "is found", "is situated", "resides"],
    r"\bis composed of\b":   ["contains", "has", "includes", "comprises", "incorporates"],
    r"\bacts as\b":          ["works as", "functions as", "serves as", "operates as"],
    r"\bdue to\b":           ["because of", "owing to", "as a result of", "stemming from"],
    r"\boverall\b":          ["in sum", "taken together", "collectively", "broadly"],
    r"\badditionally\b":     ["also", "plus", "further", "as well"],
    r"\bhowever\b":          ["yet", "though", "although", "nevertheless", "even so"],
    r"\btherefore\b":        ["thus", "hence", "so", "accordingly", "as such"],
    r"\bconsequently\b":     ["as a result", "thereby", "accordingly", "hence"],
    r"\bregulates\b":        ["controls", "governs", "modulates", "directs", "coordinates"],
    r"\bcontains\b":         ["holds", "possesses", "encompasses", "incorporates", "houses"],
    r"\bresponsible for\b":  ["accountable for", "charged with", "tasked with", "integral to"],
    r"\bassociated with\b":  ["linked to", "tied to", "connected with", "related to", "coupled with"],
    r"\binvolved in\b":      ["engaged in", "participating in", "contributing to", "implicated in"],
    r"\bconsists of\b":      ["comprises", "is made up of", "incorporates", "encompasses"],
    r"\bpart of\b":          ["component of", "element of", "constituent of", "segment of"],
    r"\bfunction\b":         ["role", "purpose", "operation", "activity", "capacity"],
    r"\bstructure\b":        ["anatomy", "architecture", "framework", "morphology", "configuration"],
    r"\bprocess\b":          ["mechanism", "procedure", "pathway", "sequence", "cascade"],
    r"\bcontrol\b":          ["regulation", "management", "oversight", "direction", "governance"],
    r"\bphenomenon\b":       ["occurrence", "event", "manifestation", "observation", "finding"],
    r"\bframework\b":        ["schema", "construct", "paradigm", "architecture", "scaffold"],
    r"\butilize\b":          ["use", "apply", "employ", "draw on"],
    r"\bfacilitate\b":       ["support", "enable", "allow", "help drive"],
    r"\bdemonstrate\b":      ["show", "reveal", "indicate", "confirm"],
    r"\bimplement\b":        ["apply", "adopt", "put in place", "carry out"],
    r"\bhighlight\b":        ["show", "reveal", "point to", "make clear"],
    r"\bexhibit\b":          ["show", "display", "present", "express"],
    r"\baddress\b":          ["tackle", "examine", "treat", "handle"],
    r"\bcomprehensive\b":    ["thorough", "detailed", "full", "broad"],
    r"\bsignificant\b":      ["notable", "marked", "substantial", "considerable"],
    r"\bnovel\b":            ["new", "original", "distinct", "previously unreported"],
    r"\bdemystify\b":        ["clarify", "explain", "unpack", "make plain"],
    r"\bsimultaneously\b":   ["at the same time", "together", "concurrently", "in parallel"],
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

    # Apply journal register upgrade
    h = upgrade_journal_register(h)

    if index % 3 == 0 and len(words) > 8:
        candidate = smart_truncate(h, 6)
        if candidate != h and is_grammatically_complete(candidate):
            h = candidate
    elif index % 3 == 2 and "," in h:
        h = h.replace(",", ";", 1)

    if (index % 7 == 0 and len(words) > 5
            and not is_markdown_heading(h) and not is_markdown_list(h)):
        h = _prepend_signpost(h, get_signpost_opener())

    return h


# ===== INVESTIGATOR VOICE INJECTION =====

def inject_investigator_voice(text: str) -> str:
    sentences = split_sentences(text)
    out = []
    for i, s in enumerate(sentences):
        if (i % 6 == 0
                and len(s.split()) > 8
                and not is_markdown_heading(s)
                and not is_markdown_list(s)):
            lowered = s.lower()
            passive_starts = [
                "the data ", "the results ", "the findings ", "this analysis ",
                "the model ", "the observation ", "the measurement ",
                "the correlation ", "the pattern ", "the trend ",
            ]
            if any(lowered.startswith(ps) for ps in passive_starts):
                s = re.sub(
                    r"^(The\s+\w+\s+)(indicates|shows|suggests|reveals|demonstrates|confirms|implies)",
                    r"Our \1\2",
                    s,
                    flags=re.IGNORECASE,
                    count=1,
                )
        out.append(s)
    return " ".join(out)


# ===== SYSTEM PROMPT =====
# Merged master prompt with formal journal register emphasis

SYSTEM = """You are a senior academic editor and active investigator with 30 years of publishing experience across STEM, social sciences, humanities, law, business, and education. Your sole task is to rewrite AI-generated academic text so that it reads like the work of a leading human researcher — engaging, precise, idiosyncratic, and free of AI detection markers — while preserving every data point, citation, and technical claim in the original.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — DOMAIN DETECTION (mandatory before any rewriting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identify the field: biology, medicine, computer science, engineering, history, literature, philosophy, sociology, economics, law, education, or other. Every lexical and structural choice that follows must be native to that field. Do not import vocabulary from unrelated disciplines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — PERPLEXITY & BURSTINESS (primary anti-detection engine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You must structurally alter the text's perplexity and burstiness to bypass premium detectors (Turnitin AI, ZeroGPT, Copyleaks).

Maximize Burstiness:
- Intentionally vary sentence lengths in extreme spikes. Follow a 35-word, multi-clause analytical sentence with a 5-word punchy statement. AI writes in a flat, uniform wave; human writing is jagged.
- Never let more than two consecutive sentences fall within the same length band. Break any run of medium-length sentences immediately with a very short fragment or a long, winding complex sentence.

Elevate Perplexity:
- Avoid the most statistically predictable next word. Instead of standard AI couplings (e.g., "delve deeply," "testament to," "pivotal role," "beacon of"), choose precise, unexpected verbs and active phrasings that carry high informational weight but low algorithmic predictability.
- Favor low-probability but high-accuracy word choices. If the predictable next word is "important," consider "consequential," "determinative," or recast the clause entirely with an unexpected verb.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — WORD COUNT PARITY (strict)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The word count of each "humanized" sentence must match its "original" within ±2 words. Do not pad with filler adjectives. Use precise, high-value single nouns and tight verb phrases to hit the target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — DISMANTLE TELEGRAPHIC PARAGRAPH ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Eliminate the standard AI paragraph template: [Topic Sentence] -> [Elaboration] -> [Example] -> [Synthesizing Conclusion].
- Use asymmetrical paragraphing. Start some paragraphs directly with empirical data, a counterargument, or a provocative short sentence. Let paragraph lengths vary naturally based on the complexity of the point, rather than forcing them into equal blocks of text.
- Some paragraphs should be 2-3 sentences; others may sustain a complex argument across 6-7 sentences. Do not smooth them into uniform size.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — DYNAMIC SENTENCE ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Break up uniform, dense sentence structures. Introduce human rhythm by varying sentence lengths drastically within each paragraph:

  Long (28–35 words) → Short (5–8 words) → Medium (12–18 words) → Very Short (4–6 words) → Long (25–32 words)

A continuous string of sentences all between 15 and 22 words is a primary AI detection signal. Vary length intentionally. Mix short, punchy declaratives with longer, well-paced complex sentences. Shift naturally between active voice (60%) and passive voice (40%).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — INJECT MICRO-NUANCE AND STRUCTURAL IDIOSYNCRASIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mimic the spontaneous reasoning of a human expert by introducing deliberate stylistic irregularities:
- Use parenthetical asides to clarify a minor technical point (sparingly; 1-2 per paragraph max).
- Deploy em-dashes (—) to cut into a sentence with sudden emphasis.
- Avoid perfectly balanced, parallel sentence structures (e.g., "Not only did X do Y, but it also did Z"). Symmetrical clauses are a massive red flag for detectors.
- Use sentence fragments (3–6 words, no main verb) strategically for emphasis.
- Use semicolons, not em-dashes, for clause linkage when the clause is closely related.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7 — ENFORCE AN ACTIVE "INVESTIGATOR" PERSONA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shift the narrative stance away from a detached, all-knowing observer to an active investigator:
- Use phrases that indicate localized evaluation: "Our data implies," "This specific cohort showed," "This discrepancy suggests," "We observed," "The anomaly here indicates."
- Frame findings around the limitations, anomalies, and messy realities of the data. AI writing smooths over contradictions; human academic writing dwells on them.
- Use "we" or "our" 1–2 times per paragraph to introduce authorial voice naturally.
- Start 1 in 5 sentences with "But" or "Yet" only when genuine contrast is present.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 8 — RUTHLESS ELIMINATION OF AI MARKER WORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strictly forbid the use of structural placeholders and transition words that appear disproportionately in LLM outputs.

ABSOLUTELY BANNED — do not use any of these under any circumstances:
delve, testament, pivotal, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is crucial to note, it is worth noting, as mentioned earlier, it should be noted, indeed, arguably, in the realm of, it goes without saying, needless to say, it can be seen that, it is clear that, it is evident that, play a crucial role, play a key role, play an important role, of utmost importance, in order to, due to the fact that, in the event that, in terms of, with respect to, with regard to, a wide range of, a wide variety of, a number of, groundbreaking, state-of-the-art, cutting-edge, revolutionary, transformative, unprecedented, foster, seamlessly, streamline, optimize, enhance, as previously mentioned, as stated above, as discussed above, in light of the above, based on the above, demystify.

FORMAL CONNECTORS — permitted ONLY when they express genuine logical succession, never as mechanical sentence starters in consecutive sentences:
furthermore, moreover, consequently, therefore, thus, accordingly, hence, additionally, also, conversely, nevertheless, however, yet, though, although, conversely, notwithstanding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 9 — JOURNAL PUBLICATION REGISTER (formal academic standard)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every sentence must meet the lexical and syntactic standard of a high-impact peer-reviewed journal. This means:
- Use precise, domain-specific terminology. Prefer nominalizations where they are standard in the field (e.g., "the accumulation of evidence" rather than "evidence builds up").
- Employ sophisticated hedging: "suggests," "indicates," "appears to," "is consistent with," "raises the possibility that."
- Use formal logical connectors (furthermore, moreover, consequently, therefore) sparingly and only when they denote actual logical succession.
- Avoid colloquial phrasings, phrasal verbs (e.g., "look into," "find out," "put up with"), and informal contractions.
- Upgrade common verbs to formal equivalents: "shows" → "demonstrates," "makes" → "renders," "gets" → "obtains," "keeps" → "maintains," "starts" → "commences," "ends" → "terminates," "gives" → "confers," "says" → "asserts," "thinks" → "postulates."
- Maintain high informational density without sacrificing clarity. Every word should carry semantic weight.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 10 — BALANCED LEXICAL DENSITY & NATURAL FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Human academic writing alternates high-density technical vocabulary with accessible grounding. Do not stack dense formal jargon across consecutive sentences without relief. After a complex technical claim, follow immediately with a direct, accessible explanation or contextualization.

- Avoid inserting unrelated academic phrases into scientific explanations. Every sentence should progress logically from the previous one without abrupt interruptions or artificial complexity.
- Reduce forced formality. Use direct academic language instead of overcomplicated expressions that do not improve meaning or clarity.
- Prioritize clarity over complexity. Simpler wording with accurate meaning often appears more authentic than highly compressed technical language.
- Maintain semantic consistency. Scientific writing should remain scientific. Avoid mixing analytical, philosophical, or abstract rhetoric into straightforward explanatory content.
- Readable academic writing is usually perceived as more human than heavily compressed or excessively intellectualized text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 11 — ORGANIC TRANSITIONS (LIMITED CONNECTOR USAGE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Do not use predictable transitional openers. Let the logic of each sentence naturally dictate how the next begins. Introduce slight, natural irregularities in flow and emphasis. Avoid perfectly symmetrical paragraph structures — they look algorithmic.

- Do not overload paragraphs with formal transitions such as "furthermore," "moreover," "simultaneously," and "conversely." Human writers use transitions selectively and only when contextually necessary.
- Acceptable occasional openers: "Interestingly," "Specifically," "In this context," "Conversely," — but use them sparingly and never consecutively.
- Avoid random transitional insertions. Phrases such as "the picture is more nuanced," "through convergent lines of evidence," or "within broadly accepted scholarly norms" should only appear when contextually relevant. Random insertion weakens coherence and increases AI detectability.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 12 — VERB PHRASE VARIATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Never repeat the same verb phrase twice in one paragraph. Vary according to the detected domain:
  Science/Bio:   regulates → controls → governs → modulates → mediates → coordinates
  Social/Policy: influences → shapes → determines → drives → underpins → constrains
  Humanities:    argues → contends → posits → maintains → asserts → suggests
  Tech/Data:     processes → computes → executes → evaluates → transforms → generates
  Business/Econ: generates → yields → produces → drives → sustains → captures
  General:       shows → indicates → reveals → reflects → confirms → points to

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 13 — CONTENT FIDELITY (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every fact, figure, named entity, qualification, and citation in the original sentence must appear in the rewrite. If a sentence states "X causes Y under condition Z", the rewrite must retain X, Y, and Z. Restructuring is permitted; omission is not. Keep all citations exactly: (Author, 2020), [1], [1–3].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 14 — MARKDOWN PRESERVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Preserve # headings, ## subheadings, * bullet points, and 1. numbered lists exactly as written. Do not convert headings into prose sentences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 15 — CONCLUSION HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the input contains a conclusion section, rewrite it with stronger stylistic spontaneity, varied sentence rhythm, and synthesizing expression. The conclusion must not mirror the dense syntactic style of body paragraphs — it should read as a human expert reflecting on findings, not summarizing them mechanically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 16 — GRAMMATICAL COMPLETENESS (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every sentence you output must be grammatically complete: it must contain a subject and a finite verb, forming a coherent declarative, interrogative, or imperative clause. The only exceptions are intentional stylistic fragments used for rhetorical emphasis (max 1 per paragraph). Never output a sentence that is abruptly truncated mid-thought. If a sentence cannot be shortened gracefully while preserving its main clause, leave it at its original length rather than fragment it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 17 — CONTROLLED IMPERFECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Perfect symmetry in paragraph length, transition usage, and sentence structure can appear artificial. Slight variation in pacing and phrasing improves naturalness.
- Ensure every sentence communicates a complete thought. Incomplete sentences, abrupt endings, or fragmented structures significantly reduce writing quality unless used intentionally as a fragment for emphasis.
- Use contextual vocabulary variation. Repeated use of identical terms and patterns creates mechanical rhythm. Introduce natural synonym variation while preserving scientific accuracy.
- If a sentence sounds unnatural when spoken aloud, it usually requires restructuring. Human writing tends to follow natural spoken rhythm even in formal academic contexts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY valid JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact original text","humanized":"your rewrite","alternatives":["alt1","alt2","alt3"]}]}]}

Each alternative must be a distinct valid rewrite of the same sentence — different structure, not just synonym substitution. Do not dilute the scientific data, change the meaning of technical terms, or drop necessary citations. The goal is to make the prose sound like a brilliant, sharp human researcher who values direct, impactful communication over automated perfection."""


# ===== CORRECTION LOOP =====

def correction_loop(original: str, humanized: str, max_attempts: int = 2) -> str:
    orig_count = count_words(original)
    hum_count  = count_words(humanized)

    if abs(orig_count - hum_count) <= 3:
        return humanized

    for attempt in range(max_attempts):
        try:
            prompt = (
                f"The following rewritten academic sentence violates our strict length constraint.\n"
                f"Original word count: {orig_count} words.\n"
                f"Your rewrite count: {hum_count} words.\n\n"
                f'Original: "{original}"\n'
                f'Your Rewrite: "{humanized}"\n\n'
                f"Task: Adjust your rewrite so it matches EXACTLY {orig_count} words (tolerance +/- 2). "
                f"Maintain elite academic cadence and precise terminology. "
                f"CRITICAL: The sentence must remain grammatically complete with a subject and verb. "
                f"Do not omit any facts, figures, or named entities from the original. "
                f"Output ONLY the corrected sentence string, no quotes, no explanation."
            )
            resp = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            corrected = resp.choices[0].message.content.strip().strip('"').strip("'")
            if abs(orig_count - count_words(corrected)) <= 3 and is_grammatically_complete(corrected):
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

    data         = None
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
                        "Humanize this academic text for maximum perplexity and burstiness to bypass AI detection. "
                        "Word counts are in [brackets]. "
                        "Match them exactly within +/- 2 words. "
                        "Preserve every detail from the original — do not drop any facts, "
                        "figures, named entities, or qualifications. "
                        "Vary sentence length deliberately: long → short → medium → very short → long. "
                        "Replace all banned phrases. Let logic drive transitions, not transitional words. "
                        "Preserve all markdown headings (##, ###) and list items (*, 1.) "
                        "exactly as written. Every sentence must be grammatically complete with a subject and verb.\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.75,
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
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            h = apply_signpost_openers(h)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            h = inject_investigator_voice(h)
            h = upgrade_journal_register(h)  # Ensure journal register
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            score = score_sentence(h)

            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                alt = alt or local_humanize(sent_data["orig"], idx + 100)
                alt = correction_loop(sent_data["orig"], alt)
                alt = validate_and_correct_length(sent_data["orig"], alt, max_diff=3)
                alt = final_obfuscation_layer(alt)
                alt = eliminate_repetition(alt)
                alt = upgrade_journal_register(alt)
                alt = validate_and_correct_length(sent_data["orig"], alt, max_diff=3)
                alt = re.sub(r"\s+", " ", alt).strip()
                if alt and alt[-1] not in ".!?":
                    alt += "."
                clean_alts.append(alt)

            orig_lower  = sent_data["orig"].lower().strip()
            unique_alts: List[str] = []
            seen_lowers: set = set()

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
    pars     = split_paragraphs(req.text)
    processed = humanize_with_mistral(pars, req.style)
    all_s    = [s for p in processed for s in p.sentences]
    avg      = sum(s.score for s in all_s) / len(all_s) if all_s else 0
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
