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
    "operating continuously without conscious oversight", "this structure",
    "that structure", "the present", "the indicated", "the respective",
}


# ===========================================================================
# ===== RESTRAINED FILLER SYSTEM ============================================
# ===========================================================================

# All fillers are neutral, academic, and register-consistent. No theatrical flair.
_UNIVERSAL_FILLERS: list = [
    "as described above.",
    "under standard conditions.",
    "within the experimental framework.",
    "as previously reported.",
    "under these parameters.",
    "as outlined in the methodology.",
    "within the defined scope.",
    "as noted earlier.",
    "under normal operating conditions.",
    "within the established model.",
]

_BIO_FILLERS: list = [
    "as previously described.",
    "under standard physiological conditions.",
    "across all experimental groups.",
    "within the defined parameters.",
    "as shown in the figure.",
]

_TECH_FILLERS: list = [
    "as previously described.",
    "under standard experimental conditions.",
    "across all test cases.",
    "within the defined parameters.",
    "as implemented in the methodology.",
]

_HUMANITIES_FILLERS: list = [
    "as discussed above.",
    "within the established framework.",
    "across the relevant literature.",
    "under the stated assumptions.",
    "as outlined previously.",
]

_SOCIAL_FILLERS: list = [
    "as previously described.",
    "within the sampled population.",
    "across all survey waves.",
    "under the stated conditions.",
    "as reported in the literature.",
]

_NATURAL_FILLERS: list = [
    "as previously described.",
    "under standard temperature and pressure.",
    "across all measured samples.",
    "within the defined system.",
    "as shown in the analysis.",
]

_EDUCATION_FILLERS: list = [
    "as previously described.",
    "within the study cohort.",
    "across all instructional conditions.",
    "under the stated parameters.",
    "as outlined in the design.",
]

_LAW_FILLERS: list = [
    "as previously described.",
    "within the applicable jurisdiction.",
    "under the stated precedent.",
    "across all relevant statutes.",
    "as outlined in the opinion.",
]

_BUSINESS_FILLERS: list = [
    "as previously described.",
    "within the analyzed period.",
    "across all reported segments.",
    "under the stated assumptions.",
    "as noted in the filing.",
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
# Lower score = more human. Penalizes AI patterns AND over-stylization.

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

    # Penalize theatrical / rhetorical language (new)
    theatrical = ["let us not forget", "wreak havoc", "doorstep", "theatrical", "dramatic"]
    for t in theatrical:
        if t in s:
            score += 18

    # Penalize overuse of metaphorical language
    metaphor_markers = ["journey", "battle", "war", "fight", "struggle", "triumph", "sweep"]
    for m in metaphor_markers:
        if m in s:
            score += 10

    # Reward natural sentence-ending variety
    if sent.endswith("?"):
        score = max(0, score - 10)

    # Reward use of "we" / "our" (human authorial voice) — but only sparingly
    if " we " in s or s.startswith("we ") or " our " in s:
        score = max(0, score - 5)

    return min(100.0, max(0.0, float(score)))


def count_words(text: str) -> int:
    return len(text.split())


# ===== GRAMMATICAL SAFETY HELPERS =====

_COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "shows", "show", "demonstrates", "demonstrate", "indicates", "indicate",
    "reveals", "reveal", "suggests", "suggest", "implies", "imply",
    "controls", "control", "governs", "govern", "modulates", "modulate",
    "regulates", "regulate", "mediates", "mediate", "coordinates", "coordinate",
    "produces", "produce", "generates", "generate", "processes", "process",
    "occurs", "occur", "results", "result", "leads", "lead",
    "contributes", "contribute", "plays", "play", "functions", "function",
    "operates", "operate", "acts", "act", "serves", "serve",
    "works", "work", "constitutes", "constitute", "represents", "represent",
    "reflects", "reflect", "underlies", "underlie", "derives", "derive",
    "requires", "require", "involves", "involve", "includes", "include",
    "comprises", "comprise", "contains", "contain", "exhibits", "exhibit",
    "manifests", "manifest", "presents", "present", "displays", "display",
    "confirms", "confirm", "establishes", "establish", "supports", "support",
    "explains", "explain", "accounts", "account", "determines", "determine",
    "defines", "define", "describes", "describe", "identifies", "identify",
    "recognizes", "recognize", "observes", "observe", "notes", "note",
    "finds", "find", "reports", "report", "states", "state",
    "argues", "argue", "claims", "claim", "asserts", "assert",
    "posits", "posit", "maintains", "maintain", "contends", "contend",
    "acknowledges", "acknowledge", "accepts", "accept", "rejects", "reject",
    "challenges", "challenge", "questions", "question", "examines", "examine",
    "investigates", "investigate", "evaluates", "evaluate", "assesses", "assess",
    "measures", "measure", "calculates", "calculate", "estimates", "estimate",
    "quantifies", "quantify", "characterizes", "characterize", "specifies", "specify",
}

def _has_verb(words: list[str]) -> bool:
    return any(w.lower().strip(",.!?;:") in _COMMON_VERBS for w in words)

def _safe_end(text: str) -> str:
    text = text.rstrip(",;—")
    if text and text[-1] not in ".!?":
        text += "."
    return text


# ===== LENGTH ENFORCEMENT =====

def enforce_length_constraint(original: str, humanized: str, max_diff: int = 4) -> str:
    """Ensure humanized text stays within word count bounds of original."""
    orig_count = count_words(original)
    hum_count = count_words(humanized)

    if abs(orig_count - hum_count) <= max_diff:
        return humanized

    if hum_count > orig_count + max_diff:
        words = humanized.split()
        keep = max(orig_count + max_diff - 1, 5)
        trimmed = " ".join(words[:keep])
        trimmed = _safe_end(trimmed)
        # Only accept if it still has a verb; otherwise return original humanized
        if _has_verb(trimmed.split()):
            return trimmed
        return humanized

    if hum_count < orig_count - max_diff:
        humanized = humanized.rstrip(".") + " " + get_filler_phrase(humanized)

    return humanized


# ===== OBFUSCATION LAYER (restrained) =====
# Only subtle, register-consistent modifications. No theatrical flourishes.

def final_obfuscation_layer(text: str) -> str:
    sentences = split_sentences(text)
    processed = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words or len(words) < 6:
            processed.append(sent)
            continue

        # Apply to ~25% of sentences max
        if random.random() > 0.25:
            processed.append(sent)
            continue

        technique = random.randint(0, 2)

        if technique == 0 and len(words) > 12:
            # One comma → semicolon, but only if both clauses are substantial
            if "," in sent:
                parts = sent.split(",", 1)
                left_words = parts[0].split()
                right_words = parts[1].split()
                if len(left_words) >= 5 and len(right_words) >= 5:
                    sent = parts[0] + "; " + parts[1].strip()

        elif technique == 1 and len(words) > 8:
            # Single subtle synonym swap (max 1 per sentence)
            swaps = {
                "the": ["this", "that"],
                "is": ["remains", "constitutes"],
                "are": ["constitute", "represent"],
                "and": ["while", "yet"],
                "to": ["toward"],
                "of": ["within"],
                "in": ["amid"],
                "for": ["regarding"],
                "with": ["via"],
                "by": ["through"],
            }
            swap_count = 0
            for idx, word in enumerate(words):
                clean = word.lower().strip(",.!?;:")
                if clean in swaps and swap_count < 1:
                    new_word = random.choice(swaps[clean])
                    if word[0].isupper():
                        new_word = new_word.capitalize()
                    if word[-1] in ",.!?;:":
                        new_word += word[-1]
                    words[idx] = new_word
                    swap_count += 1
            sent = " ".join(words)

        elif technique == 2 and len(words) > 14:
            # Split at relative pronoun only if it creates two complete sentences
            break_words = {"which", "where", "when", "while"}
            for idx, word in enumerate(words):
                if word.lower() in break_words and 4 < idx < len(words) - 4:
                    left = " ".join(words[:idx]).rstrip(",") + ". "
                    right = " ".join(words[idx:])
                    right = right[0].upper() + right[1:]
                    if _has_verb(left.split()) and _has_verb(right.split()):
                        sent = left + right
                    break

        sent = re.sub(r"\s+", " ", sent).strip()
        sent = _safe_end(sent)
        processed.append(sent)

    return " ".join(processed)


# ===== SIGNPOST LAYER (restrained) =====
# Reduced rate, only academic openers, never consecutive.

_SIGNPOST_OPENERS: list = [
    "Specifically,",
    "In this context,",
    "Consequently,",
    "Alternatively,",
    "Notably,",
    "In practice,",
    "Evidently,",
    "By comparison,",
    "Taken together,",
    "From this perspective,",
    "In such cases,",
    "Under these conditions,",
]

SIGNPOST_RATE = 0.08

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
            sent = _prepend_signpost(sent, random.choice(_SIGNPOST_OPENERS))
            sent = re.sub(r"\s+", " ", sent).strip()
            sent = _safe_end(sent)
            consecutive_signposted = 1
        else:
            consecutive_signposted = 0
        processed.append(sent)
    return " ".join(processed)


# ===== REPETITION ELIMINATION (gentle) =====
# Never chops blindly to 5 words. Only compresses if a complete clause can be formed.

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

        overlap = len(bigrams & used_bigrams)
        overlap_ratio = overlap / len(bigrams) if bigrams else 0

        if overlap_ratio > 0.35 and len(words) > 10:
            # Gentle compression to first half, but only if result is complete
            target = max(6, len(words) // 2)
            candidate = " ".join(words[:target]).rstrip(",;—")
            candidate = _safe_end(candidate)
            if _has_verb(candidate.split()):
                sent = candidate

        used_bigrams.update(bigrams)
        if len(processed) > 0 and len(processed) % 5 == 0:
            used_bigrams = set()

        processed.append(sent)

    return " ".join(processed)


# ===== BURSTINESS ENGINE (safe) =====
# Variation is introduced, but every sentence must remain grammatically complete.

def syntactic_burstiness_engine(sentences: List[str]) -> List[str]:
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        current_len = len(words)
        cycle = i % 5

        if cycle == 0 and current_len > 22:
            # Slight expansion with restrained academic clause
            target = int(current_len * 1.15)
            if len(words) < target:
                expansions = [
                    " as previously described.",
                    " under standard conditions.",
                    " across all groups.",
                    " within the defined parameters.",
                ]
                sent = sent.rstrip(".") + random.choice(expansions)

        elif cycle == 1 and current_len > 12:
            # Moderate compression — must retain verb
            target = max(int(current_len * 0.75), 6)
            if len(words) > target:
                candidate = " ".join(words[:target])
                candidate = _safe_end(candidate)
                if _has_verb(candidate.split()):
                    sent = candidate

        elif cycle == 2 and len(words) > 12:
            # Semicolon split at logical midpoint
            if ";" not in sent and "," in sent:
                mid = len(words) // 2
                left = " ".join(words[:mid])
                right = " ".join(words[mid:])
                if left and right:
                    sent = left + "; " + right

        elif cycle == 3 and current_len > 10:
            # Gentle compression — must retain verb
            target = max(int(current_len * 0.70), 5)
            if len(words) > target:
                candidate = " ".join(words[:target])
                candidate = _safe_end(candidate)
                if _has_verb(candidate.split()):
                    sent = candidate

        elif cycle == 4 and current_len > 18:
            # Slight expansion with restrained parenthetical
            if len(words) < 24:
                parentheticals = [
                    " (as expected).",
                    " (under standard conditions).",
                    " (see Methods).",
                    " (Figure 1).",
                ]
                sent = sent.rstrip(".") + random.choice(parentheticals)

        sent = re.sub(r"\s+", " ", sent).strip()
        sent = _safe_end(sent)
        result.append(sent)

    # Keep total word count drift within 10%
    new_total = sum(count_words(s) for s in result)
    if abs(new_total - total_words) > int(total_words * 0.1):
        diff = new_total - total_words
        if diff > 0:
            longest_idx = max(range(len(result)), key=lambda i: count_words(result[i]))
            w = result[longest_idx].split()
            target = max(len(w) - diff, 5)
            candidate = " ".join(w[:target]).rstrip(",;—")
            candidate = _safe_end(candidate)
            if _has_verb(candidate.split()):
                result[longest_idx] = candidate

    return result


# ===== JOURNAL-REGISTER VOCABULARY UPGRADE =====
# Only upgrades genuinely informal words. No forced theatrical substitutions.

_JOURNAL_SYNONYMS = {
    r"\bshows\b": ["demonstrates", "indicates", "reveals"],
    r"\bshow\b": ["demonstrate", "indicate", "reveal"],
    r"\bbig\b": ["substantial", "considerable", "pronounced"],
    r"\blarge\b": ["substantial", "considerable", "extensive"],
    r"\bsmall\b": ["modest", "minor", "minimal"],
    r"\bchange\b": ["modification", "alteration", "transition"],
    r"\bchanges\b": ["modifications", "alterations", "transitions"],
    r"\bmake\b": ["render", "produce", "generate"],
    r"\bmakes\b": ["renders", "produces", "generates"],
    r"\bget\b": ["obtain", "acquire", "derive"],
    r"\bgets\b": ["obtains", "acquires", "derives"],
    r"\bgot\b": ["obtained", "acquired", "derived"],
    r"\bkeep\b": ["maintain", "preserve", "retain"],
    r"\bkeeps\b": ["maintains", "preserves", "retains"],
    r"\bput\b": ["place", "position", "situate"],
    r"\bputs\b": ["places", "positions", "situates"],
    r"\bhelp\b": ["promote", "enable", "advance"],
    r"\bhelps\b": ["promotes", "enables", "advances"],
    r"\bstart\b": ["commence", "initiate"],
    r"\bstarts\b": ["commences", "initiates"],
    r"\bstarted\b": ["commenced", "initiated"],
    r"\bend\b": ["terminate", "conclude", "cease"],
    r"\bends\b": ["terminates", "concludes", "ceases"],
    r"\bended\b": ["terminated", "concluded", "ceased"],
    r"\bneed\b": ["necessitate", "require", "demand"],
    r"\bneeds\b": ["necessitates", "requires", "demands"],
    r"\bneeded\b": ["necessitated", "required", "demanded"],
    r"\bgive\b": ["provide", "furnish", "confer"],
    r"\bgives\b": ["provides", "furnishes", "confers"],
    r"\bgave\b": ["provided", "furnished", "conferred"],
    r"\bsay\b": ["state", "assert", "maintain"],
    r"\bsays\b": ["states", "asserts", "maintains"],
    r"\bsaid\b": ["stated", "asserted", "maintained"],
    r"\bthink\b": ["postulate", "hypothesize", "propose"],
    r"\bthinks\b": ["postulates", "hypothesizes", "proposes"],
    r"\bthought\b": ["postulated", "hypothesized", "proposed"],
    r"\blook at\b": ["examine", "investigate", "assess"],
    r"\blooks at\b": ["examines", "investigates", "assesses"],
    r"\blooked at\b": ["examined", "investigated", "assessed"],
    r"\bfind out\b": ["ascertain", "determine", "establish"],
    r"\bfinds out\b": ["ascertains", "determines", "establishes"],
    r"\bfound out\b": ["ascertained", "determined", "established"],
    r"\bcome from\b": ["derive from", "originate from", "stem from"],
    r"\bcomes from\b": ["derives from", "originates from", "stems from"],
    r"\bcame from\b": ["derived from", "originated from", "stemmed from"],
    r"\bgo up\b": ["increase", "escalate", "augment"],
    r"\bgoes up\b": ["increases", "escalates", "augments"],
    r"\bwent up\b": ["increased", "escalated", "augmented"],
    r"\bgo down\b": ["decrease", "diminish", "decline"],
    r"\bgoes down\b": ["decreases", "diminishes", "declines"],
    r"\bwent down\b": ["decreased", "diminished", "declined"],
    r"\babout\b": ["concerning", "regarding", "pertaining to"],
    r"\blike\b": ["such as", "akin to"],
    r"\bway\b": ["manner", "mode", "modality"],
    r"\bways\b": ["manners", "modes", "modalities"],
    r"\bthing\b": ["factor", "element", "component"],
    r"\bthings\b": ["factors", "elements", "components"],
    r"\bpart\b": ["component", "constituent", "segment"],
    r"\bparts\b": ["components", "constituents", "segments"],
    r"\bresult\b": ["outcome", "consequence", "product"],
    r"\bresults\b": ["outcomes", "consequences", "products"],
    r"\bbecause\b": ["owing to", "by virtue of", "given"],
    r"\bso\b": ["therefore", "thus", "accordingly"],
    r"\bbut\b": ["however", "nevertheless", "yet"],
    r"\balso\b": ["additionally", "furthermore", "moreover"],
    r"\bgood\b": ["favorable", "advantageous", "beneficial"],
    r"\bbad\b": ["adverse", "deleterious", "unfavorable"],
    r"\bvery\b": ["highly", "markedly", "substantially"],
    r"\bmany\b": ["numerous", "myriad"],
    r"\bsome\b": ["certain", "particular", "specific"],
    r"\bmore\b": ["additional", "further", "supplementary"],
    r"\bless\b": ["diminished", "reduced", "attenuated"],
    r"\bbefore\b": ["prior to", "preceding"],
    r"\bafter\b": ["subsequent to", "following"],
    r"\bduring\b": ["throughout", "in the course of"],
    r"\bbetween\b": ["intervening", "amid"],
    r"\bunder\b": ["subject to", "in accordance with"],
    r"\bover\b": ["above", "spanning", "throughout"],
}

_JOURNAL_COMPILED = {
    re.compile(pat, re.IGNORECASE): choices
    for pat, choices in _JOURNAL_SYNONYMS.items()
}

def upgrade_journal_register(text: str) -> str:
    for pattern, choices in _JOURNAL_COMPILED.items():
        text = pattern.sub(lambda m, c=choices: random.choice(c), text)
    return text


# ===== LOCAL FALLBACK (restrained) =====
# No forced "But"/"Yet" injection. No signpost injection. Simple synonym swaps only.

_WORD_REPLACEMENTS = {
    r"\bimportant\b":        ["key", "critical", "main", "essential"],
    r"\bplays a critical role\b": ["is essential", "is vital", "serves as"],
    r"\bplays a vital role\b":    ["is essential", "is critical", "serves as"],
    r"\bis located\b":       ["lies", "sits", "is found", "is situated"],
    r"\bis composed of\b":   ["contains", "has", "includes", "comprises"],
    r"\bacts as\b":          ["works as", "functions as", "serves as"],
    r"\bdue to\b":           ["because of", "owing to", "as a result of"],
    r"\boverall\b":          ["in sum", "taken together", "collectively"],
    r"\badditionally\b":     ["also", "plus", "further"],
    r"\bhowever\b":          ["yet", "though", "although", "nevertheless"],
    r"\btherefore\b":        ["thus", "hence", "accordingly"],
    r"\bconsequently\b":     ["as a result", "thereby", "accordingly"],
    r"\bregulates\b":        ["controls", "governs", "modulates", "directs"],
    r"\bcontains\b":         ["holds", "encompasses", "incorporates"],
    r"\bresponsible for\b":  ["accountable for", "integral to"],
    r"\bassociated with\b":  ["linked to", "connected with", "coupled with"],
    r"\binvolved in\b":      ["engaged in", "contributing to", "implicated in"],
    r"\bconsists of\b":      ["comprises", "incorporates", "encompasses"],
    r"\bpart of\b":          ["component of", "element of", "constituent of"],
    r"\bfunction\b":         ["role", "purpose", "operation", "capacity"],
    r"\bstructure\b":        ["architecture", "framework", "configuration"],
    r"\bprocess\b":          ["mechanism", "procedure", "pathway"],
    r"\bcontrol\b":          ["regulation", "oversight", "governance"],
    r"\bphenomenon\b":       ["occurrence", "manifestation", "finding"],
    r"\bframework\b":        ["schema", "construct", "scaffold"],
    r"\butilize\b":          ["use", "apply", "employ"],
    r"\bfacilitate\b":       ["support", "enable", "allow"],
    r"\bdemonstrate\b":      ["show", "reveal", "indicate", "confirm"],
    r"\bimplement\b":        ["apply", "adopt", "carry out"],
    r"\bhighlight\b":        ["show", "reveal", "point to"],
    r"\bexhibit\b":          ["show", "display", "present"],
    r"\baddress\b":          ["tackle", "examine", "treat"],
    r"\bcomprehensive\b":    ["thorough", "detailed", "full"],
    r"\bsignificant\b":      ["notable", "marked", "substantial"],
    r"\bnovel\b":            ["new", "original", "distinct"],
    r"\bsimultaneously\b":   ["at the same time", "concurrently", "in parallel"],
}

_COMPILED_REPLACEMENTS = {
    re.compile(pat, re.IGNORECASE): choices
    for pat, choices in _WORD_REPLACEMENTS.items()
}

def local_humanize(sent: str, index: int = 0) -> str:
    words = sent.split()
    if not words:
        return sent

    h = sent
    for pattern, choices in _COMPILED_REPLACEMENTS.items():
        h = pattern.sub(lambda m, c=choices: random.choice(c), h)

    h = upgrade_journal_register(h)
    return h


# ===== INVESTIGATOR VOICE (very subtle) =====
# Only changes specific passive constructions. Never forced.

def inject_investigator_voice(text: str) -> str:
    sentences = split_sentences(text)
    out = []
    for i, s in enumerate(sentences):
        if (i % 12 == 0
                and len(s.split()) > 8
                and not is_markdown_heading(s)
                and not is_markdown_list(s)):
            lowered = s.lower()
            if lowered.startswith("the data show") or lowered.startswith("the data indicate"):
                s = re.sub(
                    r"^(The\s+data\s+)(show|indicates|suggest|reveal|demonstrate|confirm|imply)",
                    r"Our data \2",
                    s,
                    flags=re.IGNORECASE,
                    count=1,
                )
        out.append(s)
    return " ".join(out)


# ===== SYSTEM PROMPT =====
# Restrained, precise, and register-consistent. No theatrical language.

SYSTEM = """You are a senior academic editor with 25 years of experience across STEM, social sciences, humanities, law, business, and education. Your task is to rewrite AI-generated academic text so that it reads like natural, rigorous scholarly prose written by a careful human researcher.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE PRINCIPLE: CONTROLLED CLARITY WITH NATURAL VARIATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Humanization is not decoration. It is controlled clarity with natural variation. Do not add rhetorical amplification, theatrical language, or metaphor stacking. Every sentence must remain precise, measured, and internally consistent with its paragraph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — DOMAIN DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identify the field: biology, medicine, computer science, engineering, history, literature, philosophy, sociology, economics, law, education, or other. Maintain that field's standard register throughout. Do not import vocabulary from unrelated disciplines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — REGISTER CONSISTENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Maintain a consistent academic register within every paragraph. Do not shift between scientific explanation and informal commentary. Avoid phrases like "let us not forget," "wreak havoc," "arrives at the doorstep," or any non-scientific metaphor. Metaphors, if used at all, must be minimal and functional — never decorative.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — WORD COUNT PARITY (strict)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The word count of each "humanized" sentence must match its "original" within ±2 words. Do not pad with filler. Use precise, high-value single nouns and tight verb phrases to hit the target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — NATURAL SENTENCE VARIATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vary sentence length to avoid the AI "flat line" where every sentence is 15–22 words. However, every sentence must remain grammatically complete with a clear subject and finite verb. No fragments. No unfinished thoughts.

Acceptable variation:
- Long analytical sentences (22–30 words) for complex claims
- Medium sentences (12–18 words) for supporting points
- Short complete sentences (6–10 words) for emphasis or transition

Never create fragments like "The mechanism is." or "In the context of these conditions, the alveoli are."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — GRAMMATICAL COMPLETENESS (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every sentence must contain a subject and a finite verb, forming a coherent declarative, interrogative, or imperative clause. If a sentence cannot be shortened gracefully while preserving its main clause, leave it at its original length. Never truncate mid-thought.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — ELIMINATE AI MARKERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strictly forbid these structural placeholders:
delve, testament, pivotal, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is crucial to note, it is worth noting, as mentioned earlier, it should be noted, indeed, arguably, in the realm of, it goes without saying, needless to say, it can be seen that, it is clear that, it is evident that, play a crucial role, play a key role, play an important role, of utmost importance, in order to, due to the fact that, in the event that, in terms of, with respect to, with regard to, a wide range of, a wide variety of, a number of, groundbreaking, state-of-the-art, cutting-edge, revolutionary, transformative, unprecedented, foster, seamlessly, streamline, optimize, enhance, demystify.

Permitted logical connectors (use sparingly, only for genuine logical succession):
furthermore, moreover, consequently, therefore, thus, accordingly, hence, additionally, also, conversely, nevertheless, however, yet, though, although, notwithstanding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7 — FORMAL ACADEMIC REGISTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every sentence must meet the standard of a high-impact peer-reviewed journal:
- Use precise, domain-specific terminology.
- Employ measured hedging: "suggests," "indicates," "appears to," "is consistent with."
- Avoid colloquial phrasings and phrasal verbs (e.g., "look into," "find out," "put up with").
- Upgrade informal verbs to formal equivalents where natural: "shows" → "demonstrates," "makes" → "renders," "gets" → "obtains," "keeps" → "maintains."
- Maintain high informational density. Every word should carry semantic weight.
- Prioritize clarity over complexity. Simpler wording with accurate meaning is preferable to compressed technical jargon.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 8 — ORGANIC TRANSITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Do not overload paragraphs with formal transitions. Human writers use them selectively and only when contextually necessary. Let the logic of each sentence naturally dictate how the next begins. Avoid perfectly symmetrical paragraph structures.

Acceptable occasional openers: "Specifically," "Notably," "In this context," "Conversely," — but use them sparingly and never consecutively.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 9 — VERB PHRASE VARIATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Never repeat the same verb phrase twice in one paragraph. Vary according to domain:
  Science/Bio:   regulates → controls → governs → modulates → directs
  Social/Policy: influences → shapes → determines → drives → constrains
  Humanities:    argues → contends → posits → maintains → suggests
  Tech/Data:     processes → computes → executes → evaluates → transforms
  Business/Econ: generates → yields → produces → drives → sustains
  General:       shows → indicates → reveals → reflects → confirms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 10 — CONTENT FIDELITY (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every fact, figure, named entity, qualification, and citation must appear in the rewrite. Restructuring is permitted; omission is not. Keep all citations exactly: (Author, 2020), [1], [1–3].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 11 — MARKDOWN PRESERVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Preserve # headings, ## subheadings, * bullet points, and 1. numbered lists exactly as written. Do not convert headings into prose sentences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 12 — CONTROLLED IMPERFECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Slight variation in pacing and phrasing improves naturalness, but never at the cost of clarity or completeness. Ensure every sentence communicates a complete thought. If a sentence sounds unnatural when spoken aloud, restructure it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY valid JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact original text","humanized":"your rewrite","alternatives":["alt1","alt2","alt3"]}]}]}

Each alternative must be a distinct valid rewrite — different structure, not just synonym substitution. Do not dilute scientific data, change technical meanings, or drop citations. The goal is prose that sounds like a careful, precise human researcher — never theatrical, always rigorous."""


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
                f"Maintain precise academic register and natural flow. "
                f"CRITICAL: The sentence must remain grammatically complete with a subject and verb. "
                f"Do not add theatrical language, metaphors, or informal commentary. "
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
            if abs(orig_count - count_words(corrected)) <= 3 and _has_verb(corrected.split()):
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
                        "Rewrite this academic text with controlled clarity and natural variation. "
                        "Word counts are in [brackets]. Match them within +/- 2 words. "
                        "Preserve every detail — do not drop any facts, figures, named entities, or qualifications. "
                        "Vary sentence length naturally, but every sentence must be grammatically complete. "
                        "Replace all banned phrases. Use logical connectors only when genuinely needed. "
                        "Preserve all markdown headings (##, ###) and list items (*, 1.) exactly as written.\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.6,
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
            h    = enforce_length_constraint(orig, h, max_diff=3)
            para_sentences.append({
                "orig":     orig,
                "hum":      h,
                "raw_alts": sent.get("alternatives", [])[:3],
            })

        humanized_only  = [s["hum"] for s in para_sentences]
        burst_sentences = syntactic_burstiness_engine(humanized_only)

        for j, (sent_data, h) in enumerate(zip(para_sentences, burst_sentences)):
            h = enforce_length_constraint(sent_data["orig"], h, max_diff=4)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            h = apply_signpost_openers(h)
            h = inject_investigator_voice(h)
            h = upgrade_journal_register(h)
            h = enforce_length_constraint(sent_data["orig"], h, max_diff=5)
            score = score_sentence(h)

            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                alt = alt or local_humanize(sent_data["orig"], idx + 100)
                alt = correction_loop(sent_data["orig"], alt)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=3)
                alt = final_obfuscation_layer(alt)
                alt = eliminate_repetition(alt)
                alt = upgrade_journal_register(alt)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=5)
                alt = re.sub(r"\s+", " ", alt).strip()
                alt = _safe_end(alt)
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
                fallback = enforce_length_constraint(sent_data["orig"], fallback, max_diff=3)
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
