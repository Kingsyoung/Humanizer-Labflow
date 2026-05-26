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
# Structured vocabulary pools by semantic category

STRUCTURAL_NOUNS = {
    "framework", "methodology", "nexus", "phenomenon", "correlate", "paradigm",
    "trajectory", "substrate", "interface", "topology", "architecture", "mechanism",
    "apparatus", "ensemble", "configuration", "modality", "repertoire", "contingency",
    "disposition", "gradient", "recursion", "hierarchy", "manifold", "schema",
    "ontology", "taxonomy", "morphology", "anatomy", "physiology", "homeostasis"
}

# Anatomical/technical nouns that can anchor parentheticals
ANCHOR_NOUNS = {
    "cerebellum", "medulla", "pons", "cortex", "tracts", "nuclei", "nerves",
    "arteries", "pyramids", "olives", "structure", "organ", "system", "pathway",
    "mechanism", "framework", "apparatus", "substrate", "topology", "interface",
    "nucleus", "ganglion", "plexus", "fasciculus", "lamina", "sulcus", "gyrus"
}

# O(1) lookup sets for connectors that trigger score penalties
TRANSITIONAL_OPENERS = {
    "furthermore", "moreover", "however", "therefore", "thus", "consequently",
    "additionally", "crucially", "additionally", "subsequently", "nevertheless",
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

# ── Hedging parentheticals (inserted mid-sentence after anchor nouns) ──────
HEDGING_PARENTHETICALS = [
    "(arguably)",
    "(presumably)",
    "(by extension)",
    "(virtually)",
    "(notably)",
    "(evidently)",
    "(under normal conditions)",
    "(under homeostatic regulation)",
    "(characteristically)",
    "(as expected)",
    "(in most cases)",
    "(physiologically speaking)",
]

# ── Signposting sentence-openers (prepended to whole sentences) ────────────
SIGNPOST_OPENERS = [
    "From an analytical standpoint,",
    "Within this framework,",
    "Crucially, this aligns with",
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
]

# ── Filler expansion phrases (for sentences that are too short) ────────────
FILLER_PHRASES = [
    "through integrated feedback loops",
    "via polysynaptic pathways",
    "under homeostatic regulation",
    "through descending cortical input",
    "via ascending somatosensory relays",
    "contingent on afferent signal integrity",
    "across distributed neural assemblies",
    "within tightly regulated homeostatic bounds",
    "through reciprocal thalamocortical projections",
    "under conditions of normal physiological demand",
]

# ── Expansion parentheticals (appended to short sentences) ────────────────
EXPANSION_PARENTHETICALS = [
    "(a requirement that cannot be bypassed)",
    "(this occurs involuntarily)",
    "(under normal physiological conditions)",
    "(a process essential for survival)",
    "(mediated by descending corticospinal tracts)",
    "(regulated through negative feedback mechanisms)",
    "(consistent with established neuroanatomical models)",
    "(dependent on intact afferent-efferent circuitry)",
]


def get_hedging_parenthetical() -> str:
    return random.choice(HEDGING_PARENTHETICALS)

def get_signpost_opener() -> str:
    return random.choice(SIGNPOST_OPENERS)

def get_filler_phrase() -> str:
    return random.choice(FILLER_PHRASES)

def get_expansion_parenthetical() -> str:
    return random.choice(EXPANSION_PARENTHETICALS)


# ===== TEXT SPLITTING =====

_ABBREV_RE = re.compile(
    r"\b(e\.g\.|i\.e\.|et al\.|Fig\.|Dr\.|Prof\.|vs\.|cf\.|ca\.|approx\.)\s"
)

def split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving abbreviations and citations."""
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected.strip())
    return [s.replace("\x00", ".").strip() for s in sents if s.strip()]

def split_paragraphs(text: str) -> List[List[str]]:
    """Split text into paragraphs, then sentences within each."""
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

    # O(1) phrase lookups
    for tell in AI_TELL_PHRASES:
        if tell in s:
            score += 15

    # Length in the "AI sweet spot"
    if 15 <= len(words) <= 22:
        score += 10

    # Transitional opener penalty — O(1) set lookup
    if words:
        first = words[0].lower().strip(",.!?;:")
        if first in TRANSITIONAL_OPENERS:
            score += 12

    # Low lexical diversity
    if len(words) > 5:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < 0.5:
            score += 10

    # Punctuation overload
    if sent.count(",") > 3 or sent.count(";") > 2:
        score += 15
    if "operating continuously" in s:
        score += 25

    # Broken fragment penalty
    if len(words) < 4 and not (sent.startswith("#") or sent.startswith("*")):
        score += 20

    return min(100.0, max(0.0, float(score)))


def count_words(text: str) -> int:
    return len(text.split())


# ===== LENGTH ENFORCEMENT =====

def enforce_length_constraint(original: str, humanized: str, max_diff: int = 3) -> str:
    """Trim or expand humanized text to match original word count within ±max_diff."""
    orig_count = count_words(original)
    hum_count = count_words(humanized)

    if abs(orig_count - hum_count) <= max_diff:
        return humanized

    if hum_count > orig_count + max_diff:
        words = humanized.split()
        keep = max(orig_count + max_diff - 1, min(orig_count, len(words)))
        trimmed = " ".join(words[:keep]).rstrip(",;—")
        return trimmed if trimmed[-1] in ".!?" else trimmed + "."

    # Too short: append filler phrase
    humanized = humanized.rstrip(".") + " " + get_filler_phrase() + "."
    return humanized

def validate_and_correct_length(original: str, humanized: str, max_diff: int = 3) -> str:
    if abs(count_words(original) - count_words(humanized)) <= max_diff:
        return humanized
    return enforce_length_constraint(original, humanized, max_diff)


# ===== GRAMMAR-SAFE PARENTHETICAL INSERTION =====

def _insert_parenthetical_after_noun(sent: str, noun: str, parenthetical: str) -> str:
    """
    Insert a parenthetical immediately after `noun` in `sent`,
    correctly handling trailing punctuation.

    e.g. "The cortex, ..."  →  "The cortex (notably), ..."
         "The cortex."      →  "The cortex (notably)."
    """
    # Pattern: noun optionally followed by punctuation
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
    """
    Prepend a signposting opener to a sentence.
    Handles capitalization: opener ends with comma → force next word lowercase.
    Opener ends without comma → keep case.
    """
    sent = sent.strip()
    if not sent:
        return sent

    # Opener already capitalised (it should be from our list)
    opener = opener.strip()

    # Force first word of original sentence to lowercase after a comma
    if opener.endswith(","):
        first_char = sent[0]
        rest = sent[1:]
        sent_body = first_char.lower() + rest
    else:
        # Opener ends with full word — add space, keep original case
        sent_body = sent

    return f"{opener} {sent_body}"


# ===== OBFUSCATION LAYER (GRAMMAR-AWARE) =====

# Configurable modification rate (0.0–1.0). Only this fraction of sentences
# will receive structural modifications, preserving natural cadence.
MODIFICATION_RATE = 0.38

def final_obfuscation_layer(text: str, modification_rate: float = MODIFICATION_RATE) -> str:
    """Apply subtle linguistic variations to a fraction of sentences."""
    sentences = split_sentences(text)
    processed = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words or len(words) < 6:
            processed.append(sent)
            continue

        # Pacing gate: only modify ~modification_rate of sentences
        if random.random() > modification_rate:
            processed.append(sent)
            continue

        technique = i % 4

        if technique == 0 and len(words) > 10:
            # SEMICOLON substitution — only between two independent clauses
            if "," in sent and len(words) > 12:
                parts = sent.split(",", 1)
                left_words = parts[0].split()
                right_words = parts[1].split()
                if len(left_words) >= 5 and len(right_words) >= 5:
                    sent = parts[0] + "; " + parts[1].strip()

        elif technique == 1 and len(words) > 8:
            # PARENTHETICAL after an anchor noun (grammar-safe)
            for idx, word in enumerate(words):
                clean = word.lower().strip(",.!?;:")
                if clean in ANCHOR_NOUNS:
                    sent = _insert_parenthetical_after_noun(sent, clean, get_hedging_parenthetical())
                    break

        elif technique == 2 and len(words) > 12:
            # CLAUSE SPLIT at subordinating conjunctions
            break_words = {"which", "where", "when", "while", "although"}
            for idx, word in enumerate(words):
                if word.lower() in break_words and 3 < idx < len(words) - 4:
                    fragment = " ".join(words[:idx]).rstrip(",") + ". "
                    remainder = " ".join(words[idx:])
                    remainder = remainder[0].upper() + remainder[1:]
                    sent = fragment + remainder
                    break

        elif technique == 3 and len(words) > 10:
            # COMPOUND SPLIT: replace "and" between two clauses with adverbial phrase
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

SIGNPOST_RATE = 0.20  # ~1 in 5 sentences receives a signpost opener

def apply_signpost_openers(text: str, rate: float = SIGNPOST_RATE) -> str:
    """Prepend signposting openers to a subset of sentences."""
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
    """Reduce conceptual repetition via bigram overlap tracking."""
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
    """Impose sentence-length variation for natural prose rhythm. No em-dashes."""
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        current_len = len(words)
        pattern = i % 5

        if pattern == 0 and len(words) < int(current_len * 1.3):
            # LONG: embed subordinate clause
            expansion = " through " + get_filler_phrase() + "."
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
            # LONG with parenthetical
            sent = sent.rstrip(".") + " " + get_expansion_parenthetical() + "."

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

# Word replacement map — patterns compiled once at module load
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
    """Fallback humanizer using vocabulary substitution and structural variation."""
    words = sent.split()
    if not words:
        return sent

    h = sent
    for pattern, choices in _COMPILED_REPLACEMENTS.items():
        h = pattern.sub(lambda m, c=choices: random.choice(c), h)

    # Structural variation based on sentence index
    if index % 3 == 0 and len(words) > 8:
        h = " ".join(words[:6]) + "."
    elif index % 3 == 1 and len(words) < 10:
        h = h.rstrip(".") + ", which remains essential for proper function."
    elif index % 3 == 2 and "," in h:
        h = h.replace(",", ";", 1)

    # Signpost opener (rate-controlled)
    if (index % 7 == 0 and len(words) > 5
            and not is_markdown_heading(h) and not is_markdown_list(h)):
        opener = get_signpost_opener()
        h = _prepend_signpost(h, opener)

    return h


# ===== SYSTEM PROMPT =====

SYSTEM = """You are an elite academic editor specializing in neuroanatomy and biomedical sciences. Rewrite AI-generated text into natural, publication-ready scholarly prose that passes Turnitin, ZeroGPT, and all AI detectors.

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

4. HUMAN QUIRKS FOR DETECTION BYPASS
- Use "we" or "our" occasionally (1-2 times per paragraph max).
- Start 1 in 5 sentences with "But" or "Yet" ONLY when creating contrast.
- Use sentence fragments (3-6 words, no verb) strategically.
- Use semicolons; not em-dashes.
- Use parentheticals sparingly: (notably), (evidently), (under normal conditions), (by extension), (presumably).
- Vary sentence openers: "Interestingly," "Specifically," "In this context," "Conversely," "As expected," "From an analytical standpoint,"

5. REPETITION ELIMINATION
- Never use the same noun phrase twice in one paragraph.
- Vary verb phrases: "regulates" → "controls" → "governs" → "modulates".

6. CITATION & MARKDOWN PRESERVATION
- Keep (Author, 2020), [1], [1-3] exactly as written.
- Preserve # headings, ## subheadings, * bullet points, 1. numbered lists EXACTLY.
- Do not turn "## Location" into a sentence. Headings must remain as: ## Heading Text

7. ACADEMIC TONE TARGET
- Write like a tenured professor with 30 years of publishing experience.
- Use precise terminology: "afferent pathways," "proprioceptive feedback," "vestibulocerebellar tracts."
- Use active voice 60% of the time, passive 40%.

OUTPUT ONLY VALID JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact text","humanized":"rewrite","alternatives":["alt1","alt2","alt3"]}]}]}"""


# ===== CORRECTION LOOP =====

def correction_loop(original: str, humanized: str, max_attempts: int = 2) -> str:
    """Re-query Mistral when length drift exceeds tolerance."""
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

        # Strip markdown code fences if present
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

        # Burstiness at paragraph level
        humanized_only = [s["hum"] for s in para_sentences]
        burst_sentences = syntactic_burstiness_engine(humanized_only)

        for j, (sent_data, h) in enumerate(zip(para_sentences, burst_sentences)):
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            # Apply signpost openers before obfuscation so capitalization is handled cleanly
            h = apply_signpost_openers(h)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            score = score_sentence(h)

            # Build unique alternatives
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
