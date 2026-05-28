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


# ===== AI TELLS & META-LANGUAGE =====

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

# Procedural meta-phrases that artificially inflate text — penalized heavily
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
    "within the defined scope",
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
# Lower = more human. Penalizes AI tells, meta-language, and over-formality.

def score_sentence(sent: str) -> float:
    s = sent.lower()
    words = sent.split()
    score = 0

    # AI tells
    for tell in AI_TELL_PHRASES:
        if tell in s:
            score += 20

    # Meta procedural phrases (major penalty)
    for meta in META_PHRASES:
        if meta in s:
            score += 18

    # Uniform AI length sweet spot
    if 15 <= len(words) <= 22:
        score += 5

    # Transitional opener overuse
    if words:
        first = words[0].lower().strip(",.!?;:")
        if first in TRANSITIONAL_OPENERS:
            score += 10

    # Low lexical diversity
    if len(words) > 5:
        unique_ratio = len({w.lower() for w in words}) / len(words)
        if unique_ratio < 0.5:
            score += 8

    # Over-punctuation
    if sent.count(",") > 3 or sent.count(";") > 2:
        score += 8

    # Over-hedging markers
    hedge_markers = ["appears to", "seems to", "tends to", "may be", "might be", "could be", "would be"]
    hedge_count = sum(1 for h in hedge_markers if h in s)
    if hedge_count > 2:
        score += 10

    # Reward natural length variation
    if len(words) < 10 or len(words) > 28:
        score = max(0, score - 5)

    # Reward direct "we" usage
    if " we " in s or s.startswith("we ") or " our " in s:
        score = max(0, score - 8)

    # Reward questions (rhetorical, natural)
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
    """Minimal length enforcement. Never append artificial fillers."""
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

    # If too short, do NOT add filler. Natural writing has short sentences.
    return humanized


# ===== REPETITION ELIMINATION (gentle) =====

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


# ===== JOURNAL REGISTER (minimal, only for informal words) =====

_JOURNAL_SYNONYMS = {
    r"\bshows\b": ["demonstrates", "indicates", "reveals"],
    r"\bshow\b": ["demonstrate", "indicate", "reveal"],
    r"\bbig\b": ["substantial", "considerable", "pronounced"],
    r"\blarge\b": ["substantial", "considerable", "extensive"],
    r"\bsmall\b": ["modest", "minor", "minimal"],
    r"\bmake\b": ["render", "produce", "generate"],
    r"\bmakes\b": ["renders", "produces", "generates"],
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


# ===== LOCAL FALLBACK (clean and direct) =====

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


# ===== SYSTEM PROMPT (fundamentally rewritten for directness) =====

SYSTEM = """You are an experienced academic editor. Rewrite the provided text so it reads like natural, clear academic prose written by a competent human researcher.

Your goal is to remove AI-generated patterns while preserving complete scientific accuracy, all citations, and every technical detail.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PRESERVE EVERYTHING
   Keep all facts, numbers, named entities, citations (Author, 2020), [1], [1-3], and technical terms exactly as they appear.

2. NO META-LANGUAGE
   Never add procedural or methodological qualifiers like "in accordance with standard conditions," "within the defined scope," "subject to these parameters," "as previously described," "under the stated conditions," or "within the experimental framework" unless they literally exist in the original sentence.

3. DIRECT AND CLEAR
   Write directly. Do not over-hedge. One layer of qualification is enough. Do not write "it might be suggested that it could potentially be considered that" — write "this suggests that" or "we found that."

4. NATURAL VARIATION
   Vary sentence length naturally. Use short sentences (6-10 words) for emphasis or transition. Use medium sentences (12-20 words) for explanation. Use longer sentences (22-30 words) only when necessary for complex claims. Never force all sentences into the same length.

5. GRAMMATICAL COMPLETENESS
   Every sentence must have a subject and a finite verb. No fragments. No truncated thoughts.

6. BANNED AI PHRASES
   Never use: delve, testament, pivotal, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, it is crucial to note, furthermore (as a mechanical opener), moreover (as a mechanical opener), crucially, arguably, indeed, needless to say, it goes without saying, groundbreaking, cutting-edge, state-of-the-art, revolutionary, transformative, unprecedented, foster, seamlessly, streamline, optimize, enhance, demystify.

7. NATURAL CONNECTORS
   Use transitions only when they express real logic: "However," "Thus," "Next," "In contrast," "Consequently." Do not start every sentence with a connector. Let the flow be organic.

8. AUTHORIAL VOICE
   Use "we" or "our" 1-2 times per paragraph if it feels natural, especially when discussing findings or methods. Example: "We observed that..." or "Our results indicate..."

9. SENTENCE STARTERS
   Vary how sentences begin. Start some with the subject. Start some with "However," or "Thus." Start some with "This finding..." or "These results..." Do not use the same pattern repeatedly.

10. CLARITY OVER COMPLEXITY
    Prefer clear, direct phrasing over compressed jargon. If a simpler word says the same thing, use it. Do not inflate vocabulary to sound "academic."

11. MARKDOWN PRESERVATION
    Keep headings (#, ##, ###) and list items (*, 1.) exactly as written.

12. WORD COUNT
    Match the original sentence length within ±3 words. Do not pad or cut to hit an artificial target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY valid JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact original text","humanized":"your rewrite","alternatives":["alt1","alt2","alt3"]}]}]}

Each alternative must be a genuinely different structural rewrite of the same sentence. Do not just swap synonyms."""


# ===== CORRECTION LOOP (simplified) =====

def correction_loop(original: str, humanized: str, max_attempts: int = 2) -> str:
    orig_count = count_words(original)
    hum_count  = count_words(humanized)

    if abs(orig_count - hum_count) <= 3:
        return humanized

    for attempt in range(max_attempts):
        try:
            prompt = (
                f"This rewrite is off by {abs(orig_count - hum_count)} words.\n"
                f"Original ({orig_count} words): {original}\n"
                f"Rewrite ({hum_count} words): {humanized}\n\n"
                f"Adjust to exactly {orig_count} words (±2 tolerance). "
                f"Keep all facts and terms. Write directly. No meta-language. "
                f"Output ONLY the corrected sentence, no quotes."
            )
            resp = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
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
                        "Rewrite this academic text to read as natural human-written prose. "
                        "Remove AI patterns. Preserve all facts and citations. "
                        "Word counts are in [brackets] — match within ±3 words. "
                        "No meta-language. Direct and clear. "
                        "Preserve markdown headings and lists exactly.\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            temperature=0.5,
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

        # Post-processing: minimal and clean
        for j, sent_data in enumerate(para_sentences):
            h = sent_data["hum"]
            
            # Gentle repetition cleanup only
            h = eliminate_repetition(h)
            
            # Minimal register upgrade (only for truly informal words)
            h = upgrade_journal_register(h)
            
            # Final loose length check
            h = enforce_length_constraint(sent_data["orig"], h, max_diff=5)
            
            # Clean spacing and punctuation
            h = re.sub(r"\s{2,}", " ", h)
            h = _safe_end(h.strip())
            
            score = score_sentence(h)

            # Process alternatives
            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                alt = alt or local_humanize(sent_data["orig"], idx + 100)
                alt = correction_loop(sent_data["orig"], alt)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=3)
                alt = eliminate_repetition(alt)
                alt = upgrade_journal_register(alt)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=5)
                alt = re.sub(r"\s{2,}", " ", alt)
                alt = _safe_end(alt.strip())
                clean_alts.append(alt)

            # Deduplicate alternatives
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
