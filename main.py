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
    # extended for non-bio domains
    "model", "theory", "argument", "concept", "process", "method", "approach",
    "variable", "factor", "element", "component", "dimension", "aspect",
    "institution", "policy", "context", "framework", "paradigm", "principle",
    "evidence", "data", "analysis", "result", "finding", "outcome",
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
    "indeed", "arguably", "significantly"
}


# ===========================================================================
# ===== DOMAIN-AWARE FILLER SYSTEM ==========================================
# ===========================================================================

# ---------------------------------------------------------------------------
# Universal hedging parentheticals — safe in ANY discipline
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Universal signpost openers — safe in ANY discipline
# ---------------------------------------------------------------------------
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
    "Most importantly,",
    "In broader terms,",
    "Under closer scrutiny,",
    "When considered carefully,",
    "At its core,",
    "From a structural standpoint,",
    "Taken in isolation,",
    "In the aggregate,",
    "Across domains,",
    "Fundamentally,",
]

# ---------------------------------------------------------------------------
# Domain-specific filler phrase banks
# ---------------------------------------------------------------------------

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
_BIO_EXPANSION: list = [
    "(under normal physiological conditions)",
    "(a process essential for survival)",
    "(mediated by descending corticospinal tracts)",
    "(regulated through negative feedback mechanisms)",
    "(consistent with established neuroanatomical models)",
    "(dependent on intact afferent-efferent circuitry)",
    "(this occurs involuntarily)",
    "(a requirement that cannot be bypassed)",
    "(via receptor-ligand binding interactions)",
    "(under hormonal regulatory influence)",
    "(governed by enzymatic cascade reactions)",
    "(contingent on cellular membrane integrity)",
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
_TECH_EXPANSION: list = [
    "(within computational complexity bounds)",
    "(subject to hardware resource constraints)",
    "(under controlled benchmark conditions)",
    "(consistent with formal specification requirements)",
    "(assuming deterministic input conditions)",
    "(validated against held-out test data)",
    "(per established algorithmic conventions)",
    "(across multiple independent test runs)",
    "(subject to convergence guarantees)",
    "(as established in prior literature)",
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
_HUMANITIES_EXPANSION: list = [
    "(as the scholarly literature attests)",
    "(subject to interpretive contestation)",
    "(a point widely acknowledged in the field)",
    "(within its specific historical moment)",
    "(consistent with the theoretical tradition)",
    "(as subsequent historiography has confirmed)",
    "(a claim that warrants careful qualification)",
    "(a position not without its critics)",
    "(in the context of the broader debate)",
    "(as the primary sources make clear)",
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
_SOCIAL_EXPANSION: list = [
    "(as demonstrated in empirical studies)",
    "(consistent with the existing evidence base)",
    "(subject to individual-level variation)",
    "(a finding replicated across multiple contexts)",
    "(as the regression analysis confirms)",
    "(under conditions of ecological validity)",
    "(holding other variables constant)",
    "(as theory would predict)",
    "(across demographically diverse samples)",
    "(contingent on measurement reliability)",
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
_NATURAL_EXPANSION: list = [
    "(under controlled laboratory conditions)",
    "(consistent with thermodynamic principles)",
    "(as field observations confirm)",
    "(subject to ambient temperature effects)",
    "(across experimentally replicated trials)",
    "(within measurable detection thresholds)",
    "(as spectrometric analysis verifies)",
    "(under isothermal equilibrium conditions)",
    "(consistent with conservation of mass)",
    "(as validated by independent replication)",
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
_EDUCATION_EXPANSION: list = [
    "(as evidence-based pedagogy prescribes)",
    "(consistent with constructivist learning theory)",
    "(across diverse learner populations)",
    "(subject to instructional design constraints)",
    "(as classroom observation data confirms)",
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
_LAW_EXPANSION: list = [
    "(as established by binding precedent)",
    "(subject to legislative amendment)",
    "(within the bounds of constitutional authority)",
    "(per the court's interpretive holding)",
    "(consistent with the rule of law)",
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
_BUSINESS_EXPANSION: list = [
    "(as market data consistently shows)",
    "(subject to regulatory oversight)",
    "(under ceteris paribus conditions)",
    "(consistent with efficient market theory)",
    "(across multiple fiscal reporting periods)",
    "(as financial modeling confirms)",
    "(contingent on investor risk tolerance)",
    "(within accepted accounting standards)",
]

# Universal fallback — field-neutral, safe for any sentence
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
    "under conditions of theoretical parsimony",
    "across multiple levels of analysis",
    "through the interplay of structural factors",
    "within the scope of the current analysis",
    "via established disciplinary conventions",
    "under well-specified model assumptions",
    "through iterative cycles of refinement",
    "contingent on prior theoretical commitments",
    "across a range of empirical observations",
    "within systematically structured frameworks",
    "through hierarchically organized mechanisms",
    "via complementary theoretical perspectives",
    "under conditions of analytical rigor",
    "across temporally and spatially distinct cases",
    "through overlapping and reinforcing processes",
    "within the framework of existing scholarship",
    "via consistent methodological application",
    "under carefully controlled parameters",
    "across independent yet converging domains",
    "through sustained empirical investigation",
    "within logically structured argument chains",
    "via the application of formal analytical tools",
    "under conditions widely recognized in the field",
    "across both theoretical and applied dimensions",
    "through a principled sequence of analytical steps",
]
_UNIVERSAL_EXPANSION: list = [
    "(as the evidence consistently indicates)",
    "(a point well-established in the literature)",
    "(subject to methodological qualification)",
    "(under conditions of rigorous scrutiny)",
    "(as subsequent analysis confirms)",
    "(a finding robust across contexts)",
    "(in ways that merit further investigation)",
    "(consistent with prior theoretical accounts)",
    "(under the assumptions stated above)",
    "(an observation not without precedent)",
    "(as the broader literature attests)",
    "(subject to the caveats noted above)",
    "(across independently validated studies)",
    "(a claim well-supported by available evidence)",
    "(this remains a productive area of inquiry)",
    "(as logically follows from the premises)",
    "(in keeping with standard academic practice)",
    "(absent confounding variables)",
    "(across methodologically diverse approaches)",
    "(upon careful analytical examination)",
]

# ---------------------------------------------------------------------------
# Domain keyword detector
# ---------------------------------------------------------------------------
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

_EXPANSION_MAP: dict = {
    "bio":        _BIO_EXPANSION,
    "tech":       _TECH_EXPANSION,
    "humanities": _HUMANITIES_EXPANSION,
    "social":     _SOCIAL_EXPANSION,
    "natural":    _NATURAL_EXPANSION,
    "education":  _EDUCATION_EXPANSION,
    "law":        _LAW_EXPANSION,
    "business":   _BUSINESS_EXPANSION,
    "universal":  _UNIVERSAL_EXPANSION,
}


def _detect_domain(sentence: str) -> str:
    """
    Score the sentence against each domain's keyword list.
    Returns the best-matching domain key, or 'universal' if no domain
    scores at least 2 keyword hits.
    """
    text = sentence.lower()
    scores: dict = {domain: 0 for domain in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] >= 2 else "universal"


def get_filler_phrase(sentence: str = "") -> str:
    """
    Return a domain-appropriate filler expansion phrase.
    Pass the current sentence text for context detection.
    30 % of the time blends in a universal phrase to avoid monotony.
    """
    domain = _detect_domain(sentence) if sentence else "universal"
    pool = _FILLER_MAP.get(domain, _UNIVERSAL_FILLERS)
    if random.random() < 0.3:
        pool = _UNIVERSAL_FILLERS
    return random.choice(pool)


def get_expansion_parenthetical(sentence: str = "") -> str:
    """
    Return a domain-appropriate expansion parenthetical.
    Pass the current sentence text for context detection.
    """
    domain = _detect_domain(sentence) if sentence else "universal"
    pool = _EXPANSION_MAP.get(domain, _UNIVERSAL_EXPANSION)
    if random.random() < 0.3:
        pool = _UNIVERSAL_EXPANSION
    return random.choice(pool)


def get_hedging_parenthetical() -> str:
    """Return a hedging parenthetical — universally safe across all fields."""
    return random.choice(HEDGING_PARENTHETICALS)


def get_signpost_opener() -> str:
    """Return a signposting sentence opener — universally safe across all fields."""
    return random.choice(SIGNPOST_OPENERS)


# ===========================================================================
# ===== END FILLER SYSTEM ===================================================
# ===========================================================================


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
    s = sent.lower()
    words = sent.split()
    score = 0

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
    if "operating continuously" in s:
        score += 25

    if len(words) < 4 and not (sent.startswith("#") or sent.startswith("*")):
        score += 20

    return min(100.0, max(0.0, float(score)))


def count_words(text: str) -> int:
    return len(text.split())


# ===== LENGTH ENFORCEMENT =====

def enforce_length_constraint(original: str, humanized: str, max_diff: int = 3) -> str:
    orig_count = count_words(original)
    hum_count = count_words(humanized)

    if abs(orig_count - hum_count) <= max_diff:
        return humanized

    if hum_count > orig_count + max_diff:
        words = humanized.split()
        keep = max(orig_count + max_diff - 1, min(orig_count, len(words)))
        trimmed = " ".join(words[:keep]).rstrip(",;—")
        return trimmed if trimmed[-1] in ".!?" else trimmed + "."

    # Too short: append a context-aware filler phrase
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

MODIFICATION_RATE = 0.38

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

        technique = i % 4

        if technique == 0 and len(words) > 10:
            if "," in sent and len(words) > 12:
                parts = sent.split(",", 1)
                left_words = parts[0].split()
                right_words = parts[1].split()
                if len(left_words) >= 5 and len(right_words) >= 5:
                    sent = parts[0] + "; " + parts[1].strip()

        elif technique == 1 and len(words) > 8:
            for idx, word in enumerate(words):
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
                    fragment = " ".join(words[:idx]).rstrip(",") + ". "
                    remainder = " ".join(words[idx:])
                    remainder = remainder[0].upper() + remainder[1:]
                    sent = fragment + remainder
                    break

        elif technique == 3 and len(words) > 10:
            if " and " in sent:
                and_pos = sent.find(" and ")
                before, after = sent[:and_pos].strip(), sent[and_pos + 5:].strip()
                clause_verbs = {"is", "are", "was", "were", "has", "have",
                                "controls", "regulates", "modulates", "governs"}
                if (len(before.split()) > 4 and len(after.split()) > 4
                        and any(v in before.lower().split() for v in clause_verbs)):
                    sent = sent.replace(" and ", ", consequently, ", 1)

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        processed.append(sent)

    return " ".join(processed)


# ===== SIGNPOST LAYER =====

SIGNPOST_RATE = 0.20

def apply_signpost_openers(text: str, rate: float = SIGNPOST_RATE) -> str:
    sentences = split_sentences(text)
    processed = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        if (len(words) > 5
                and not is_markdown_heading(sent)
                and not is_markdown_list(sent)
                and random.random() < rate):
            sent = _prepend_signpost(sent, get_signpost_opener())
            sent = re.sub(r"\s+", " ", sent).strip()
            if sent and sent[-1] not in ".!?":
                sent += "."
        processed.append(sent)
    return " ".join(processed)


# ===== REPETITION ELIMINATION =====

def eliminate_repetition(text: str) -> str:
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return text

    processed = []
    used_bigrams: set = set()

    for idx, sent in enumerate(sentences):
        words = sent.lower().split()
        bigrams = {
            words[i].strip(",.!?;:") + " " + words[i + 1].strip(",.!?;:")
            for i in range(len(words) - 1)
        }
        overlap_ratio = len(bigrams & used_bigrams) / len(bigrams) if bigrams else 0

        if overlap_ratio > 0.3 and len(words) > 6:
            sent = " ".join(words[:5]) + "."

        used_bigrams.update(bigrams)
        if len(processed) > 0 and len(processed) % 5 == 0:
            used_bigrams = set()

        processed.append(sent)

    return " ".join(processed)


# ===== BURSTINESS ENGINE =====

def syntactic_burstiness_engine(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        current_len = len(words)
        pattern = i % 5

        if pattern == 0 and len(words) < int(current_len * 1.3):
            # LONG: embed a context-aware subordinate clause
            expansion = " through " + get_filler_phrase(sent) + "."
            sent = sent.rstrip(".") + expansion

        elif pattern == 1:
            # SHORT: compress
            target = max(int(current_len * 0.6), 4)
            if len(words) > target:
                sent = " ".join(words[:target]) + "."

        elif pattern == 2 and ";" not in sent and len(words) > 10:
            # MEDIUM with semicolon
            mid = len(words) // 2
            sent = " ".join(words[:mid]) + "; " + " ".join(words[mid:])

        elif pattern == 3 and len(words) > 7:
            # SHORT fragment
            sent = " ".join(words[:4]) + "."

        elif pattern == 4 and len(words) < 18:
            # LONG with context-aware parenthetical
            sent = sent.rstrip(".") + " " + get_expansion_parenthetical(sent) + "."

        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        result.append(sent)

    # Verify total word count drift stays within 10%
    new_total = sum(count_words(s) for s in result)
    if abs(new_total - total_words) > int(total_words * 0.1):
        diff = new_total - total_words
        if diff > 0:
            longest_idx = max(range(len(result)), key=lambda i: count_words(result[i]))
            w = result[longest_idx].split()
            result[longest_idx] = " ".join(w[:max(len(w) - diff, 3)]) + "."

    return result


# ===== LOCAL FALLBACK =====

_WORD_REPLACEMENTS = {
    r"\bimportant\b": ["key", "critical", "main", "essential", "central", "primary"],
    r"\bplays a critical role\b": ["is essential", "is vital", "serves as", "underpins"],
    r"\bplays a vital role\b": ["is essential", "is critical", "serves as", "anchors"],
    r"\bis located\b": ["lies", "sits", "is found", "is situated", "resides"],
    r"\bis composed of\b": ["contains", "has", "includes", "comprises", "incorporates"],
    r"\bacts as\b": ["works as", "functions as", "serves as", "operates as"],
    r"\bdue to\b": ["because of", "owing to", "as a result of", "stemming from"],
    r"\boverall\b": ["in sum", "taken together", "collectively", "broadly"],
    r"\badditionally\b": ["also", "plus", "further", "as well"],
    r"\bhowever\b": ["yet", "though", "although", "nevertheless", "even so"],
    r"\btherefore\b": ["thus", "hence", "so", "accordingly", "as such"],
    r"\bconsequently\b": ["as a result", "thereby", "accordingly", "hence"],
    r"\bregulates\b": ["controls", "governs", "modulates", "directs", "coordinates"],
    r"\bcontains\b": ["holds", "possesses", "encompasses", "incorporates", "houses"],
    r"\bresponsible for\b": ["accountable for", "charged with", "tasked with", "integral to"],
    r"\bassociated with\b": ["linked to", "tied to", "connected with", "related to", "coupled with"],
    r"\binvolved in\b": ["engaged in", "participating in", "contributing to", "implicated in"],
    r"\bconsists of\b": ["comprises", "is made up of", "incorporates", "encompasses"],
    r"\bpart of\b": ["component of", "element of", "constituent of", "segment of"],
    r"\bfunction\b": ["role", "purpose", "operation", "activity", "capacity"],
    r"\bstructure\b": ["anatomy", "architecture", "framework", "morphology", "configuration"],
    r"\bprocess\b": ["mechanism", "procedure", "pathway", "sequence", "cascade"],
    r"\bcontrol\b": ["regulation", "management", "oversight", "direction", "governance"],
    r"\bphenomenon\b": ["occurrence", "event", "manifestation", "observation", "finding"],
    r"\bframework\b": ["schema", "construct", "paradigm", "architecture", "scaffold"],
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

    if index % 3 == 0 and len(words) > 8:
        h = " ".join(words[:6]) + "."
    elif index % 3 == 1 and len(words) < 10:
        h = h.rstrip(".") + ", which remains essential for proper function."
    elif index % 3 == 2 and "," in h:
        h = h.replace(",", ";", 1)

    if (index % 7 == 0 and len(words) > 5
            and not is_markdown_heading(h) and not is_markdown_list(h)):
        opener = get_signpost_opener()
        h = _prepend_signpost(h, opener)

    return h


# ===== SYSTEM PROMPT =====

SYSTEM = """You are an elite academic editor fluent across all scholarly disciplines — including STEM, social sciences, humanities, law, business, and education. Rewrite AI-generated text into natural, publication-ready scholarly prose that passes Turnitin, ZeroGPT, and all AI detectors.

CRITICAL ARCHITECTURAL CONSTRAINTS:

1. STRICT WORD-COUNT PARITY (1:1 Density Ratio)
- For every sentence you process, calculate the EXACT word count in the "original" string.
- The "humanized" output must match that word count within +/- 2 words maximum.
- Do NOT pad with fluff adjectives or elongate text. Use singular high-value nouns and concise verb phrases.

2. THE SYNTACTIC COMPRESSION RULE
- If one sentence expands to 30+ words, the next sentence in the same paragraph MUST compress to 5-8 words.
- Never let total paragraph word count exceed input by more than 5%.
- Alternate: Long (28-35 words) → Short (5-8 words) → Medium (12-18) → Short (4-7) → Long (25-30).

3. BANNED PHRASES (ZeroGPT triggers)
NEVER USE: delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is worth noting, as mentioned earlier, it should be noted, indeed, arguably, significantly

4. DOMAIN DETECTION — MANDATORY FIRST STEP
- Before rewriting, identify the field of the input text: biology, medicine, computer science, engineering, history, literature, philosophy, sociology, economics, law, education, or other.
- Use ONLY terminology and phrasing native to that field.
- NEVER introduce vocabulary from an unrelated discipline into the rewrite.
- If the input is a history essay, write like a historian. If it is a CS paper, write like a computer scientist. If it is a legal brief, write like a lawyer.

5. PRECISE TERMINOLOGY
- Use terminology appropriate to the detected discipline of the input text.
- Mirror the technical register and vocabulary density of the original.
- Do not simplify specialized terms; replace them only with equally precise synonyms from the same field.

6. HUMAN QUIRKS FOR DETECTION BYPASS
- Use "we" or "our" occasionally (1-2 times per paragraph max).
- Start 1 in 5 sentences with "But" or "Yet" ONLY when creating contrast.
- Use sentence fragments (3-6 words, no verb) strategically.
- Use semicolons; not em-dashes.
- Use parentheticals sparingly: (notably), (evidently), (under normal conditions), (by extension), (presumably).
- Vary sentence openers: "Interestingly," "Specifically," "In this context," "Conversely," "As expected," "From an analytical standpoint,"

7. VERB PHRASE VARIATION
- Never repeat the same verb phrase twice in one paragraph.
- Vary according to the detected domain:
  Science/Bio:   "regulates" → "controls" → "governs" → "modulates" → "mediates" → "coordinates"
  Social/Policy: "influences" → "shapes" → "determines" → "drives" → "underpins" → "constrains"
  Humanities:    "argues" → "contends" → "posits" → "maintains" → "asserts" → "suggests"
  Tech/Data:     "processes" → "computes" → "executes" → "evaluates" → "optimizes" → "transforms"
  Business/Econ: "generates" → "yields" → "produces" → "drives" → "sustains" → "captures"
  General:       "demonstrates" → "indicates" → "reveals" → "reflects" → "highlights" → "confirms"

8. REPETITION ELIMINATION
- Never use the same noun phrase twice in one paragraph.

9. CITATION & MARKDOWN PRESERVATION
- Keep (Author, 2020), [1], [1-3] exactly as written.
- Preserve # headings, ## subheadings, * bullet points, 1. numbered lists EXACTLY.
- Do not turn "## Location" into a sentence. Headings must remain as: ## Heading Text

10. ACADEMIC TONE TARGET
- Write like a tenured professor with 30 years of publishing experience in the detected field.
- Use active voice 60% of the time, passive 40%.

OUTPUT ONLY VALID JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact text","humanized":"rewrite","alternatives":["alt1","alt2","alt3"]}]}]}"""

# ===== CORRECTION LOOP =====

def correction_loop(original: str, humanized: str, max_attempts: int = 2) -> str:
    orig_count = count_words(original)
    hum_count = count_words(humanized)

    if abs(orig_count - hum_count) <= 3:
        return humanized

    for attempt in range(max_attempts):
        try:
            prompt = (
                f"The following rewritten academic sentence violates our strict length constraint.\n"
                f"Original word count: {orig_count} words.\n"
                f"Your rewrite count: {hum_count} words.\n\n"
                f"Original: \"{original}\"\n"
                f"Your Rewrite: \"{humanized}\"\n\n"
                f"Task: Adjust your rewrite so it matches EXACTLY {orig_count} words (tolerance +/- 2). "
                f"Maintain elite academic cadence and precise terminology. "
                f"Output ONLY the corrected sentence string, no quotes, no explanation."
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
            humanized = corrected
            hum_count = count_words(humanized)
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

    data = None
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
                        "Humanize this academic text. Word counts are in [brackets]. "
                        "Match them exactly within +/- 2 words. Preserve all markdown headings "
                        "(##, ###) and list items (*, 1.) exactly as written:\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.7,
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
            h = sent.get("humanized", "") or local_humanize(orig, j)
            h = correction_loop(orig, h)
            h = validate_and_correct_length(orig, h, max_diff=3)
            para_sentences.append({
                "orig": orig,
                "hum": h,
                "raw_alts": sent.get("alternatives", [])[:3],
            })

        # Burstiness at paragraph level — pass sentence text for domain detection
        humanized_only = [s["hum"] for s in para_sentences]
        burst_sentences = syntactic_burstiness_engine(humanized_only)

        for j, (sent_data, h) in enumerate(zip(para_sentences, burst_sentences)):
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            h = apply_signpost_openers(h)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
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

            orig_lower = sent_data["orig"].lower().strip()
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
    pars = split_paragraphs(req.text)
    processed = humanize_with_mistral(pars, req.style)
    all_s = [s for p in processed for s in p.sentences]
    avg = sum(s.score for s in all_s) / len(all_s) if all_s else 0
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
