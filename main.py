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


# ===== AI TELLS & BANNED PHRASES =====

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

META_PHRASES = {
    "in accordance with standard conditions",
    "within the defined scope",
    "subject to these parameters",
    "within the experimental framework",
    "as previously described",
    "under standard conditions",
    "within the established model",
    "as outlined in the methodology",
    "under these parameters",
    "within the defined parameters",
    "as noted earlier",
    "under normal operating conditions",
    "as previously reported",
    "subject to these conditions",
    "under the stated conditions",
    "within the applicable framework",
    "as conventionally understood",
    "within the stated boundaries",
    "subject to the outlined constraints",
    "as described in the preceding section",
    "under the aforementioned conditions",
    "within the parameters outlined above",
    "in line with standard protocol",
    "as per the established methodology",
    "within the confines of the model",
    "subject to the limitations described",
    "under the regulatory framework",
    "as documented in the literature",
    "within the prescribed parameters",
}

TRANSITIONAL_OPENERS = {
    "furthermore", "moreover", "however", "therefore", "thus", "consequently",
    "additionally", "crucially", "subsequently", "nevertheless",
    "notwithstanding", "accordingly"
}


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
            score += 20

    for meta in META_PHRASES:
        if meta in s:
            score += 18

    if 15 <= len(words) <= 22:
        score += 5

    if words:
        first = words[0].lower().strip(",.!?;:")
        if first in TRANSITIONAL_OPENERS:
            score += 10

    if len(words) > 5:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < 0.5:
            score += 8

    if sent.count(",") > 3 or sent.count(";") > 2:
        score += 8

    hedge_markers = ["appears to", "seems to", "tends to", "may be", "might be", "could be", "would be"]
    hedge_count = sum(1 for h in hedge_markers if h in s)
    if hedge_count > 2:
        score += 10

    if len(words) < 10 or len(words) > 28:
        score = max(0, score - 5)

    if " we " in s or s.startswith("we ") or " our " in s:
        score = max(0, score - 8)

    if sent.endswith("?"):
        score = max(0, score - 10)

    return min(100.0, max(0.0, float(score)))


def count_words(text: str) -> int:
    return len(text.split())


# ===== GRAMMATICAL SAFETY =====

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


# ===== LENGTH ENFORCEMENT (minimal) =====

def enforce_length_constraint(original: str, humanized: str, max_diff: int = 4) -> str:
    orig_count = count_words(original)
    hum_count = count_words(humanized)

    if abs(orig_count - hum_count) <= max_diff:
        return humanized

    if hum_count > orig_count + max_diff:
        words = humanized.split()
        keep = max(orig_count + max_diff - 1, 5)
        trimmed = " ".join(words[:keep])
        trimmed = _safe_end(trimmed)
        if _has_verb(trimmed.split()):
            return trimmed
        return humanized

    return humanized


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

        overlap = len(bigrams & used_bigrams)
        overlap_ratio = overlap / len(bigrams) if bigrams else 0

        if overlap_ratio > 0.4 and len(words) > 12:
            target = max(7, len(words) // 2)
            candidate = " ".join(words[:target]).rstrip(",;—")
            candidate = _safe_end(candidate)
            if _has_verb(candidate.split()):
                sent = candidate

        used_bigrams.update(bigrams)
        if len(processed) > 0 and len(processed) % 6 == 0:
            used_bigrams = set()

        processed.append(sent)

    return " ".join(processed)


# ===== NATURAL HUMAN VARIATION ENGINE =====
# This is the core anti-detection layer. It introduces genuine human statistical
# patterns: contractions, slight informality, varied sentence rhythm, and
# unpredictable phrasing that breaks AI token predictability.

_CONTRACTIONS = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "has not": "hasn't",
    "have not": "haven't",
    "had not": "hadn't",
    "will not": "won't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "should not": "shouldn't",
    "cannot": "can't",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "what is": "what's",
    "who is": "who's",
    "where is": "where's",
    "when is": "when's",
    "why is": "why's",
    "how is": "how's",
    "i am": "i'm",
    "you are": "you're",
    "we are": "we're",
    "they are": "they're",
    "he is": "he's",
    "she is": "she's",
    "it has": "it's",
    "that has": "that's",
    "there has": "there's",
    "i have": "i've",
    "you have": "you've",
    "we have": "we've",
    "they have": "they've",
    "i will": "i'll",
    "you will": "you'll",
    "we will": "we'll",
    "they will": "they'll",
    "he will": "he'll",
    "she will": "she'll",
    "it will": "it'll",
    "would have": "would've",
    "could have": "could've",
    "should have": "should've",
    "might have": "might've",
}

def apply_contractions(text: str, rate: float = 0.15) -> str:
    """Introduce natural contractions at ~15% rate to break formal AI patterns."""
    sentences = split_sentences(text)
    processed = []
    for sent in sentences:
        if random.random() < rate and not is_markdown_heading(sent) and not is_markdown_list(sent):
            # Apply 1-2 random contractions
            applied = 0
            items = list(_CONTRACTIONS.items())
            random.shuffle(items)
            for full, contracted in items:
                if full in sent.lower() and applied < 2:
                    # Case-insensitive replacement, preserve original case pattern
                    pattern = re.compile(re.escape(full), re.IGNORECASE)
                    sent = pattern.sub(contracted, sent, count=1)
                    applied += 1
        processed.append(sent)
    return " ".join(processed)


def apply_informal_markers(text: str, rate: float = 0.08) -> str:
    """Add occasional natural human markers that AI avoids."""
    sentences = split_sentences(text)
    processed = []
    
    informal_starters = [
        "honestly,",
        "look,",
        "the thing is,",
        "to be fair,",
        "admittedly,",
        "frankly,",
        "sure,",
        "now,",
        "so,",
        "well,",
    ]
    
    for i, sent in enumerate(sentences):
        words = sent.split()
        if (len(words) > 8 
            and not is_markdown_heading(sent) 
            and not is_markdown_list(sent)
            and random.random() < rate
            and i > 0):  # Never first sentence
            starter = random.choice(informal_starters)
            # Lowercase first word of original
            if sent[0].isupper():
                rest = sent[0].lower() + sent[1:]
            else:
                rest = sent
            sent = f"{starter} {rest}"
        processed.append(sent)
    return " ".join(processed)


def vary_punctuation(text: str) -> str:
    """Break uniform punctuation patterns with natural variation."""
    sentences = split_sentences(text)
    processed = []
    
    for i, sent in enumerate(sentences):
        # Occasional question instead of statement (rare, natural)
        if i > 0 and i % 12 == 7 and "?" not in sent and len(sent.split()) < 15:
            if random.random() < 0.3:
                # Convert a short declarative to rhetorical question
                words = sent.split()
                if len(words) > 5 and words[0].lower() in {"this", "that", "these", "those", "it"}:
                    sent = sent.rstrip(".") + "?"
        
        # Occasional dash usage (sparingly)
        if i % 15 == 3 and len(sent.split()) > 10 and "—" not in sent:
            if random.random() < 0.2 and "," in sent:
                # Replace one comma with em-dash for emphasis
                parts = sent.rsplit(", ", 1)
                if len(parts) == 2 and len(parts[1].split()) > 3:
                    sent = parts[0] + " — " + parts[1]
        
        processed.append(sent)
    return " ".join(processed)


def break_uniform_rhythm(text: str) -> str:
    """Introduce sentence length chaos — the #1 ZeroGPT bypass signal."""
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return text
    
    processed = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        count = len(words)
        
        # Force extreme variation every few sentences
        if i % 7 == 3 and count > 8:
            # Create a very short punchy sentence
            short = " ".join(words[:min(4, count)])
            short = _safe_end(short)
            if _has_verb(short.split()):
                processed.append(short)
                continue
        
        elif i % 11 == 5 and count < 10:
            # Elongate a short sentence with natural continuation
            continuations = [
                ", which follows from the above.",
                ", as one would expect.",
                ", a point worth emphasizing.",
                ", and this matters for the analysis.",
            ]
            sent = sent.rstrip(".") + random.choice(continuations)
        
        processed.append(sent)
    
    return " ".join(processed)


# ===== JOURNAL REGISTER (minimal) =====

_JOURNAL_SYNONYMS = {
    r"\bshows\b": ["demonstrates", "indicates", "reveals"],
    r"\bshow\b": ["demonstrate", "indicate", "reveal"],
    r"\bget\b": ["obtain", "acquire", "derive"],
    r"\bgets\b": ["obtains", "acquires", "derives"],
    r"\bgot\b": ["obtained", "acquired", "derived"],
    r"\bkeep\b": ["maintain", "preserve", "retain"],
    r"\bkeeps\b": ["maintains", "preserves", "retains"],
    r"\bput\b": ["place", "position", "situate"],
    r"\bputs\b": ["places", "positions", "situates"],
    r"\bstart\b": ["commence", "initiate"],
    r"\bstarts\b": ["commences", "initiates"],
    r"\bstarted\b": ["commenced", "initiated"],
    r"\bend\b": ["terminate", "conclude", "cease"],
    r"\bends\b": ["terminates", "concludes", "ceases"],
    r"\bended\b": ["terminated", "concluded", "ceased"],
    r"\bneed\b": ["necessitate", "require"],
    r"\bneeds\b": ["necessitates", "requires"],
    r"\bneeded\b": ["necessitated", "required"],
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
    r"\bway\b": ["manner", "mode"],
    r"\bways\b": ["manners", "modes"],
    r"\bthing\b": ["factor", "element", "component"],
    r"\bthings\b": ["factors", "elements", "components"],
    r"\bpart\b": ["component", "constituent", "segment"],
    r"\bparts\b": ["components", "constituents", "segments"],
    r"\bresult\b": ["outcome", "consequence", "product"],
    r"\bresults\b": ["outcomes", "consequences", "products"],
    r"\bbecause\b": ["owing to", "given"],
    r"\bso\b": ["therefore", "thus", "accordingly"],
    r"\bbut\b": ["however", "nevertheless", "yet"],
    r"\balso\b": ["additionally", "furthermore", "moreover"],
    r"\bgood\b": ["favorable", "advantageous", "beneficial"],
    r"\bbad\b": ["adverse", "deleterious", "unfavorable"],
    r"\bvery\b": ["highly", "markedly", "substantially"],
    r"\bmany\b": ["numerous", "myriad"],
    r"\bsome\b": ["certain", "particular", "specific"],
    r"\bmore\b": ["additional", "further"],
    r"\bless\b": ["diminished", "reduced"],
    r"\bbefore\b": ["prior to", "preceding"],
    r"\bafter\b": ["subsequent to", "following"],
    r"\bduring\b": ["throughout", "in the course of"],
    r"\bbetween\b": ["intervening", "amid"],
    r"\bunder\b": ["subject to"],
    r"\bover\b": ["above", "spanning"],
}

_JOURNAL_COMPILED = {
    re.compile(pat, re.IGNORECASE): choices
    for pat, choices in _JOURNAL_SYNONYMS.items()
}

def upgrade_journal_register(text: str) -> str:
    for pattern, choices in _JOURNAL_COMPILED.items():
        text = pattern.sub(lambda m, c=choices: random.choice(c), text)
    return text


# ===== LOCAL FALLBACK =====

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


# ===== SYSTEM PROMPT (anti-detection focused) =====

SYSTEM = """You are a human academic writer. Rewrite the provided text so it reads exactly like natural human-written academic prose — the kind a real student or researcher would submit.

Your goal is to make the text completely indistinguishable from human writing while keeping all facts, citations, and technical terms intact.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL ANTI-DETECTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. USE CONTRACTIONS NATURALLY
   Use "don't," "can't," "it's," "we've," "they're" occasionally. Real humans use contractions even in academic writing. Don't overdo it — 1-2 per paragraph.

2. VARY SENTENCE LENGTH DRAMATICALLY
   This is the #1 detector signal. Mix very short sentences (4-6 words) with medium (12-18) and long (25-35). Never let more than 2 consecutive sentences be the same length. Example pattern: 28 words → 7 words → 16 words → 5 words → 31 words.

3. START SENTENCES DIFFERENTLY
   Vary your openers. Subject → "However," → "This" → "But" → "In 2020," → "We" → "Notably," → "It" → "So," → "That" → "Yet" → "When" → "The" → "Our" → "Interestingly," → "They" → "But" → "This" → "We" → "It" → "However,"

4. USE INFORMAL MARKERS SPARINGLY
   Occasionally use "honestly," "look," "the thing is," "to be fair," "admittedly," "frankly," "sure," "now," "so," "well," at the start of sentences. Not every sentence — maybe 1-2 per paragraph.

5. ASK RHETORICAL QUESTIONS
   Very occasionally (1 per 2-3 paragraphs), end a short sentence with "?" instead of "." Example: "But does this hold across all populations?"

6. USE DASHES AND COLLOQUIAL PACING
   Use an em-dash (—) once or twice to break a sentence for emphasis. Use a colon for dramatic pause. Real humans punctuate unpredictably.

7. BREAK PARAGRAPH UNIFORMITY
   Let some paragraphs be 2 sentences. Others 5-6. Never uniform 4-sentence blocks.

8. USE "WE" AND "OUR" NATURALLY
   1-2 times per paragraph when discussing findings. "We found that..." "Our data suggest..." "We observed..."

9. DIRECT AND CLEAR
   No meta-language. No "in accordance with standard conditions." No "within the defined scope." Write like you're explaining to a peer, not filing a compliance report.

10. BANNED AI PHRASES
    Never use: delve, testament, pivotal, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is crucial to note, furthermore (as opener), moreover (as opener), crucially, arguably, indeed, needless to say, groundbreaking, cutting-edge, state-of-the-art, revolutionary, transformative, unprecedented, foster, seamlessly, streamline, optimize, enhance, demystify.

11. PRESERVE EVERYTHING
    Keep all facts, numbers, named entities, citations (Author, 2020), [1], [1-3], and technical terms exactly.

12. MATCH LENGTH WITHIN ±3 WORDS
    But don't pad artificially. If the original is 12 words, 9-15 is fine. Natural variation includes slightly shorter or longer sentences.

13. MARKDOWN PRESERVATION
    Keep headings (#, ##, ###) and list items (*, 1.) exactly as written.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY valid JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact original text","humanized":"your rewrite","alternatives":["alt1","alt2","alt3"]}]}]}

Each alternative must be structurally different — not just synonym swaps."""


# ===== ROBUST JSON PARSING =====

def robust_json_extract(text: str) -> dict:
    """Extract valid JSON from potentially malformed LLM output."""
    text = text.strip()
    
    # Remove markdown fences
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    # Aggressive repair: fix common LLM JSON errors
    # Fix unescaped quotes inside strings
    repaired = text[start:end+1] if start != -1 and end != -1 else text
    
    # Replace problematic patterns
    # Fix single quotes used as apostrophes inside double-quoted strings
    repaired = re.sub(r'(?<<=[a-zA-Z])\'(?=[a-zA-Z])', "''", repaired)  # temporarily mark
    repaired = re.sub(r'\'', '"', repaired)  # convert remaining single quotes
    repaired = re.sub(r"''", "'", repaired)  # restore apostrophes
    
    # Fix trailing commas
    repaired = re.sub(r',\s*}', '}', repaired)
    repaired = re.sub(r',\s*]', ']', repaired)
    
    # Fix missing quotes around keys
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        print(f"JSON repair failed: {e}")
        # Last resort: extract sentence data with regex
        return _fallback_parse(text)


def _fallback_parse(text: str) -> dict:
    """Emergency extraction when JSON is completely broken."""
    result = {"processed_paragraphs": []}
    
    # Try to find sentence pairs with regex
    pattern = r'"original"\s*:\s*"([^"]+)"\s*,\s*"humanized"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Group into a single paragraph (emergency fallback)
        sentences = []
        for orig, hum in matches:
            # Clean up escaped quotes
            orig = orig.replace('\\"', '"').replace('\\\\', '\\')
            hum = hum.replace('\\"', '"').replace('\\\\', '\\')
            sentences.append({
                "original": orig,
                "humanized": hum,
                "alternatives": [hum, hum, hum]  # Duplicates as emergency fallback
            })
        result["processed_paragraphs"].append({"sentences": sentences})
    
    return result


# ===== CORRECTION LOOP =====

def correction_loop(original: str, humanized: str, max_attempts: int = 2) -> str:
    orig_count = count_words(original)
    hum_count  = count_words(humanized)

    if abs(orig_count - hum_count) <= 3:
        return humanized

    for attempt in range(max_attempts):
        try:
            prompt = (
                f"Fix the word count. Original: {orig_count} words. "
                f"Yours: {hum_count} words. Target: {orig_count} (±2).\n\n"
                f'Original: "{original}"\n'
                f'Rewrite: "{humanized}"\n\n'
                f"Output ONLY the corrected sentence. Use contractions naturally. "
                f"Vary sentence structure. No meta-language. No banned AI phrases."
            )
            resp = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            corrected = resp.choices[0].message.content.strip().strip('"').strip("'")
            if abs(orig_count - count_words(corrected)) <= 3 and _has_verb(corrected.split()):
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
                        "Rewrite this academic text to read as completely natural human writing. "
                        "Use contractions, vary sentence length dramatically, start sentences differently, "
                        "add occasional informal markers, use rhetorical questions sparingly. "
                        "Word counts in [brackets] — match within ±3 words. "
                        "Preserve all facts, citations, and technical terms. "
                        "No meta-language. No banned phrases. "
                        "Preserve markdown headings and lists exactly.\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.65,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        print(f"RAW RESPONSE (first 500 chars): {text[:500]}")
        
        data = robust_json_extract(text)
        print("JSON parsed successfully")
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
            h = enforce_length_constraint(orig, h, max_diff=3)
            para_sentences.append({
                "orig": orig,
                "hum": h,
                "raw_alts": sent.get("alternatives", [])[:3],
            })

        # Apply anti-detection layers in specific order
        for j, sent_data in enumerate(para_sentences):
            h = sent_data["hum"]
            
            # Layer 1: Break uniform rhythm (sentence length chaos)
            h = break_uniform_rhythm(h)
            
            # Layer 2: Natural contractions
            h = apply_contractions(h, rate=0.12)
            
            # Layer 3: Occasional informal markers
            h = apply_informal_markers(h, rate=0.06)
            
            # Layer 4: Punctuation variation
            h = vary_punctuation(h)
            
            # Layer 5: Gentle repetition cleanup
            h = eliminate_repetition(h)
            
            # Layer 6: Minimal register upgrade
            h = upgrade_journal_register(h)
            
            # Layer 7: Final length and safety
            h = enforce_length_constraint(sent_data["orig"], h, max_diff=5)
            h = re.sub(r"\s{2,}", " ", h)
            h = _safe_end(h.strip())
            
            score = score_sentence(h)

            # Process alternatives
            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                alt = alt or local_humanize(sent_data["orig"], idx + 100)
                alt = correction_loop(sent_data["orig"], alt)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=3)
                alt = break_uniform_rhythm(alt)
                alt = apply_contractions(alt, rate=0.10)
                alt = eliminate_repetition(alt)
                alt = upgrade_journal_register(alt)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=5)
                alt = re.sub(r"\s{2,}", " ", alt)
                alt = _safe_end(alt.strip())
                clean_alts.append(alt)

            # Deduplicate
            orig_lower = sent_data["orig"].lower().strip()
            unique_alts: List[str] = []
            seen_lowers: set = set()

            for alt in clean_alts:
                al = alt.lower().strip()
                if al != orig_lower and al not in seen_lowers and len(alt.split()) > 3:
                    unique_alts.append(alt)
                    seen_lowers.add(al)

            seed = 200
            while len(unique_alts) < 3:
                fallback = local_humanize(sent_data["orig"], seed)
                fallback = break_uniform_rhythm(fallback)
                fallback = apply_contractions(fallback, rate=0.10)
                fallback = enforce_length_constraint(sent_data["orig"], fallback, max_diff=3)
                fl = fallback.lower().strip()
                if fl != orig_lower and fl not in seen_lowers and len(fallback.split()) > 3:
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
