import os
import re
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from mistralai.client import Mistral

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
    mode: Optional[str] = "stealth"

class ProcessResponse(BaseModel):
    processed_paragraphs: List[ParagraphData]
    total_sentences: int
    avg_score: float


# ===========================================================================
# ===== AI-TELL DETECTION (scoring only — not used to drive transforms) =====
# ===========================================================================

AI_TELL_PHRASES = {
    "delve", "testament", "pivotal", "moreover", "furthermore",
    "it is important to note", "it is crucial to note", "in conclusion",
    "landscape", "tapestry", "beacon", "underscore", "shed light on",
    "ever-evolving", "multifaceted", "intricate", "robust", "holistic",
    "leverage", "synergy", "crucially", "underscoring",
    "it is worth noting", "as mentioned earlier", "it should be noted",
    "indeed", "significantly", "in addition", "it is essential",
    "plays a crucial role", "plays a vital role", "plays a key role",
    "it is well established", "needless to say",
    "it is noteworthy", "one must consider", "it is imperative",
    "merely scratches the surface", "and this is key",
}

TRANSITIONAL_OPENERS = {
    "furthermore", "moreover", "however", "therefore", "thus", "consequently",
    "additionally", "crucially", "subsequently", "nevertheless",
    "notwithstanding", "accordingly", "henceforth", "heretofore",
}

def score_sentence(sent: str) -> float:
    s, words = sent.lower(), sent.split()
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
    return min(100.0, max(0.0, score))


# ===========================================================================
# ===== TEXT UTILITIES ======================================================
# ===========================================================================

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

def count_words(text: str) -> int:
    return len(text.split())

def is_markdown_heading(text: str) -> bool:
    return text.strip().startswith("#")

def is_markdown_list(text: str) -> bool:
    s = text.strip()
    return s.startswith(("* ", "- ")) or bool(re.match(r"^\d+\.", s))


# ===========================================================================
# ===== LIGHT CLEANUP — runs ONCE, after Mistral, no stacking ===============
# ===========================================================================

# Only the clearest mechanical AI tells — not stylistic preferences
_HARD_REPLACEMENTS = {
    re.compile(r"\bfurthermore\b", re.I):    ["beyond this", "building on this", "beyond that"],
    re.compile(r"\bmoreover\b", re.I):        ["equally", "beyond this", "on top of this"],
    re.compile(r"\badditionally\b", re.I):    ["also", "beyond this", "further"],
    re.compile(r"\bcrucially\b", re.I):       ["critically", "centrally", "above all"],
    re.compile(r"\bsubsequently\b", re.I):    ["later", "after this", "following this"],
    re.compile(r"\bnevertheless\b", re.I):    ["still", "even so", "that said"],
    re.compile(r"\bnotwithstanding\b", re.I): ["despite this", "even so"],
    re.compile(r"\baccordingly\b", re.I):     ["as a result", "for this reason"],
    re.compile(r"\butilize\b", re.I):         ["use", "employ", "apply"],
    re.compile(r"\butilises\b", re.I):        ["uses", "employs", "applies"],
    re.compile(r"\bin order to\b", re.I):     ["to"],
    re.compile(r"\bdemonstrate\b", re.I):     ["show", "reveal", "confirm"],
    re.compile(r"\bit is important to note\b", re.I): ["notably", "of note"],
    re.compile(r"\bit is worth noting\b", re.I):      ["notably", "of note"],
    re.compile(r"\bit is crucial to note\b", re.I):   ["critically", "of note"],
    re.compile(r"\bit should be noted\b", re.I):      ["notably"],
    re.compile(r"\bplays a (?:crucial|vital|key) role\b", re.I): ["is central", "is integral", "underpins"],
}

def _apply_hard_replacements(text: str) -> str:
    """Replace only the hardest AI-tell phrases. One pass. No chaining."""
    for pattern, choices in _HARD_REPLACEMENTS.items():
        def _rep(m, c=choices):
            replacement = random.choice(c)
            orig = m.group(0)
            # Preserve capitalisation of first word if sentence-opener
            if orig and orig[0].isupper() and replacement:
                replacement = replacement[0].upper() + replacement[1:]
            return replacement
        text = pattern.sub(_rep, text)
    return text


def _light_structural_fix(sent: str) -> str:
    """
    Two grammar-safe fixes only:
    1. Remove 'It is X that ...' cleft constructions
    2. Split sentences genuinely over 38 words at the nearest comma
    Nothing else — no injections, no additions, no em-dashes.
    """
    # Fix 1: cleft
    pat = re.compile(r"^It\s+(is|was)\s+(\w+)\s+that\s+", re.I)
    m = pat.match(sent)
    if m:
        remainder = sent[m.end():].strip().rstrip(".")
        sent = remainder[0].upper() + remainder[1:] + "."

    # Fix 2: split overlong
    words = sent.split()
    if len(words) > 38:
        half = len(words) // 2
        for delta in range(0, half):
            for idx in [half + delta, half - delta]:
                if 0 < idx < len(words) and words[idx - 1].endswith(","):
                    left = " ".join(words[:idx]).rstrip(",") + "."
                    right = " ".join(words[idx:])
                    if right:
                        right = right[0].upper() + right[1:]
                        if right[-1] not in ".!?":
                            right += "."
                        return left + " " + right
    return sent


def cleanup_pass(text: str) -> str:
    """
    Single cleanup pass: hard replacements + light structural fixes.
    Called ONCE per sentence after Mistral returns. Nothing else downstream.
    """
    sentences = split_sentences(text)
    cleaned = []
    for sent in sentences:
        if is_markdown_heading(sent) or is_markdown_list(sent):
            cleaned.append(sent)
            continue
        sent = _apply_hard_replacements(sent)
        sent = _light_structural_fix(sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        if sent and sent[-1] not in ".!?":
            sent += "."
        cleaned.append(sent)
    return " ".join(cleaned)


# ===========================================================================
# ===== SYSTEM PROMPT — rewritten from scratch ==============================
#
# PHILOSOPHY CHANGE:
# Old prompt told the model HOW to sound human (avoid X, vary Y, rotate Z).
# This made Mistral perform humanness as a task — detectable.
#
# New prompt gives Mistral a PERSONA and a GOAL: write as a specific type
# of human expert would naturally write. No mechanical instructions.
# The model generates authentically; we clean only the hardest tells after.
# ===========================================================================

_SYSTEM_PERSONA = """You are a senior academic — a tenured professor at a research university — revising a colleague's draft for submission. You write well because you have spent years writing, not because you follow rules. Your prose is direct, occasionally a bit dry, sometimes unexpectedly precise. You do not perform sophistication. You do not over-explain. You cut what is unnecessary.

Your task: rewrite each sentence so it sounds exactly like you wrote it originally — a real person, thinking through ideas on paper.

WHAT ACTUALLY DISTINGUISHES HUMAN ACADEMIC WRITING:

1. Sentence rhythm varies because the IDEAS vary in complexity, not because you deliberately mix lengths. A short sentence follows a dense one because the thought is complete, not to hit a target.

2. You occasionally use a slightly unusual word choice — not an elevated one, just the specific one that fits. You do not rotate synonyms to avoid repetition.

3. You write through ideas rather than announcing them. "The cerebellum coordinates voluntary movement" — not "It is worth noting that the cerebellum plays a key role in the coordination of voluntary movement."

4. You do not start sentences with: Furthermore, Moreover, Consequently, Additionally, Subsequently, Nevertheless, Notwithstanding, Significantly, Importantly, Crucially, Indeed.

5. You do not use: delve, testament, tapestry, landscape, beacon, holistic, robust, multifaceted, intricate, leverage, synergy, shed light on, ever-evolving, underscores, it is important to note, it is worth noting, plays a crucial/vital/key role, needless to say.

6. Transitions arise from logic, not connective tissue. The end of one sentence implies the start of the next.

7. Passive voice appears where appropriate — where the agent is unknown, unimportant, or where the object deserves the subject position. Not as a style choice.

ABSOLUTE CONSTRAINTS:
- Match original word count within +/- 3 words per sentence
- Preserve every fact, figure, citation, statistic, named entity, and qualifier exactly
- Keep all markdown headings, bullets, and numbered items unchanged
- Write in the register of the field — do not shift to a different discipline's vocabulary

OUTPUT ONLY VALID JSON — no preamble, no explanation, no markdown fences:
{"processed_paragraphs":[{"sentences":[{"original":"exact original sentence","humanized":"your rewrite","alternatives":["alt1","alt2","alt3"]}]}]}"""


# ===========================================================================
# ===== LOCAL FALLBACK ======================================================
# ===========================================================================

_FALLBACK_REPLACEMENTS = {
    re.compile(r"\bfurthermore\b", re.I):  "beyond this",
    re.compile(r"\bmoreover\b", re.I):     "equally",
    re.compile(r"\badditionally\b", re.I): "also",
    re.compile(r"\butilize\b", re.I):      "use",
    re.compile(r"\bin order to\b", re.I):  "to",
    re.compile(r"\bdemonstrate\b", re.I):  "show",
}

def local_humanize(sent: str) -> str:
    h = sent
    for pattern, replacement in _FALLBACK_REPLACEMENTS.items():
        h = pattern.sub(replacement, h)
    return h


# ===========================================================================
# ===== MAIN PROCESSING =====================================================
# ===========================================================================

def humanize_with_mistral(
    paragraphs: List[List[str]],
    style: str,
    mode: str = "stealth",
) -> List[ParagraphData]:

    print(f"CALLING MISTRAL — mode={mode}, paragraphs={len(paragraphs)}")

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
                {"role": "system", "content": _SYSTEM_PERSONA},
                {
                    "role": "user",
                    "content": (
                        f"Revise this academic text. Field/style: {style}.\n"
                        "Word counts per sentence are in [brackets] — stay within +/- 3 words.\n"
                        "Write as yourself — a senior academic revising a draft. "
                        "Do not follow rules. Write how you actually write.\n\n"
                        + "\n".join(lines)
                    ),
                },
            ],
            # Lower temperature = more conservative, more uniform = more detectable
            # 0.85 gives Mistral room to make genuine lexical choices
            temperature=0.85,
            max_tokens=4096,
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
            text = text[start:end + 1]
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
                            "humanized": local_humanize(s),
                            "alternatives": [local_humanize(s), local_humanize(s), local_humanize(s)],
                        }
                        for s in para
                    ]
                }
                for para in paragraphs
            ]
        }

    result: List[ParagraphData] = []

    for i, para_data in enumerate(data.get("processed_paragraphs", [])):
        para_sentences = []

        for j, sent_data in enumerate(para_data.get("sentences", [])):
            orig = sent_data.get("original", "")
            h = sent_data.get("humanized", "") or local_humanize(orig)

            # ONE cleanup pass — that's it
            h = cleanup_pass(h)

            # Soft word-count check — trim only if wildly over, never pad
            oc, hc = count_words(orig), count_words(h)
            if hc > oc + 5:
                words = h.split()
                h = " ".join(words[:oc + 3]).rstrip(",;—")
                if h[-1] not in ".!?":
                    h += "."

            score = score_sentence(h)

            # Alternatives — same single cleanup pass, no extras
            raw_alts = sent_data.get("alternatives", [])[:3]
            clean_alts: List[str] = []
            seen: set = set()
            orig_lower = orig.lower().strip()

            for alt in raw_alts:
                if not alt:
                    alt = local_humanize(orig)
                alt = cleanup_pass(alt)
                al = alt.lower().strip()
                if al != orig_lower and al not in seen:
                    clean_alts.append(alt)
                    seen.add(al)

            while len(clean_alts) < 3:
                fb = local_humanize(orig)
                fl = fb.lower().strip()
                if fl not in seen:
                    clean_alts.append(fb)
                    seen.add(fl)
                else:
                    clean_alts.append(orig)
                    break

            para_sentences.append(SentenceData(
                id=f"p{i}-s{j}",
                original=orig,
                humanized=h,
                alternatives=clean_alts[:3],
                score=score,
            ))

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
    mode = req.mode if req.mode in ("stealth", "humanize") else "stealth"
    pars = split_paragraphs(req.text)
    processed = humanize_with_mistral(pars, req.style, mode=mode)
    all_s = [s for p in processed for s in p.sentences]
    avg = sum(s.score for s in all_s) / len(all_s) if all_s else 0
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
