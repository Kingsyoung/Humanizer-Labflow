import os
import re
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import MistralClient as Mistral

import os
API_KEY = os.getenv("MISTRAL_API_KEY", "")
if not API_KEY:
    print("ERROR: No API key")
    exit(1)
client = Mistral(api_key=API_KEY)
app = FastAPI(title="Academic Humanizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://https://humanizer-frontend-seven.vercel.app",
        "https://humanizer-labflow-git-main-growthloom.vercel.app",
        "http://localhost:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
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

# ===== TEXT SPLITTING =====

def split_sentences(text):
    """Split text into sentences, preserving citations and abbreviations."""
    text = re.sub(r"\b(e\.g\.|i\.e\.|et al\.|Fig\.|Dr\.|Prof\.)\s", lambda m: m.group(0).replace(".", "\x00"), text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.replace("\x00", ".").strip() for s in sents if s.strip()]

def split_paragraphs(text):
    """Split text into paragraphs, preserving markdown structure."""
    lines = text.split("\n")
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(current)
                current = []
        else:
            current.append(stripped)
    if current:
        paragraphs.append(current)

    # Return as list of sentence lists, with metadata about markdown type
    result = []
    for para in paragraphs:
        para_text = " ".join(para)
        result.append(split_sentences(para_text))
    return result

def is_markdown_line(text):
    """Detect if text is a markdown heading, list item, or other structural element."""
    stripped = text.strip()
    if stripped.startswith("#"):
        return "heading"
    if stripped.startswith("* ") or stripped.startswith("- "):
        return "bullet"
    if stripped.startswith("1.") or stripped.startswith("2.") or re.match(r"^\d+\.", stripped):
        return "numbered"
    return None

# ===== SCORING =====

AI_TELLS = ["delve", "testament", "pivotal", "moreover", "furthermore", "it is important to note",
            "it is crucial to note", "in conclusion", "landscape", "tapestry", "beacon", "underscore",
            "shed light on", "navigate", "ever-evolving", "multifaceted", "intricate", "robust",
            "leverage", "holistic", "paradigm", "synergy", "stakeholder", "crucially", "underscoring",
            "operating continuously without conscious oversight", "this structure", "that structure",
            "the present", "the indicated", "the respective"]

def score_sentence(sent):
    s, words = sent.lower(), sent.split()
    score = sum(15 for t in AI_TELLS if t in s)
    if 15 <= len(words) <= 22: score += 10
    first = words[0].lower().strip(",.:") if words else ""
    if first in ["furthermore", "moreover", "however", "therefore", "thus", "consequently", "additionally", "crucially"]:
        score += 12
    unique = len(set(w.lower() for w in words))
    if len(words) > 5 and unique / len(words) < 0.5: score += 10
    for phrase in ["furthermore", "moreover", "in conclusion", "it is important to note", "it is crucial to note", "crucially"]:
        if phrase in s: score += 20
    # Penalize broken fragments
    if sent.count(",") > 3 or sent.count(";") > 2: score += 15
    if "operating continuously" in s: score += 25
    return min(100, max(0, score))

def count_words(text):
    return len(text.split())

# ===== LENGTH CONSTRAINTS =====

# VARIED filler phrases - never repeat the same one
def get_filler_phrase():
    fillers = [
        "a process that occurs automatically.",
        "this happens without voluntary input.",
        "such activity proceeds reflexively.",
        "the mechanism functions autonomously.",
        "this operates below conscious awareness.",
        "the response is involuntary.",
        "such control is automatic.",
        "the process remains unconscious."
    ]
    return random.choice(fillers)

def enforce_length_constraint(original, humanized, max_diff=4):
    """Ensure humanized text stays within word count bounds of original."""
    orig_count = count_words(original)
    hum_count = count_words(humanized)

    if abs(orig_count - hum_count) <= max_diff:
        return humanized

    if hum_count > orig_count + max_diff:
        words = humanized.split()
        keep = max(orig_count + max_diff - 1, 5)
        trimmed = " ".join(words[:keep])
        if trimmed[-1] in ',;—':
            trimmed = trimmed[:-1]
        if trimmed[-1] not in '.!?':
            trimmed += '.'
        return trimmed

    if hum_count < orig_count - max_diff:
        words_needed = orig_count - hum_count
        if words_needed <= 3:
            humanized = humanized.rstrip('.') + ' ' + get_filler_phrase()
        else:
            humanized = humanized.rstrip('.') + ' ' + get_filler_phrase()

    return humanized

# ===== OBfuscation LAYER =====

def final_obfuscation_layer(text):
    """Apply subtle linguistic variations. No technique overused."""
    sentences = split_sentences(text)
    processed = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words:
            continue

        # Only apply to ~30% of sentences, rotate techniques
        if i % 3 == 0 and len(words) > 8:
            # Swap 1 common word for synonym
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
                "from": ["originating in"],
                "as": ["functioning as"],
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

        elif i % 3 == 1 and len(words) > 12:
            # Use semicolon instead of comma (only once)
            if "," in sent:
                parts = sent.split(",", 1)
                sent = parts[0] + "; " + parts[1]

        elif i % 3 == 2 and len(words) > 10:
            # Brief parenthetical insertion
            insert_point = min(3, len(words) - 2)
            words.insert(insert_point, "(notably)")
            sent = " ".join(words)

        # Clean up
        sent = re.sub(r'\s+', ' ', sent).strip()
        if sent and sent[-1] not in '.!?':
            sent += '.'

        processed.append(sent)

    return " ".join(processed)

# ===== REPETITION ELIMINATION =====

def eliminate_repetition(text):
    """Reduce conceptual repetition within paragraphs using n-gram overlap."""
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return text

    processed = []
    used_bigrams = set()

    for sent in sentences:
        words = sent.lower().split()
        bigrams = set()
        for i in range(len(words) - 1):
            bg = words[i].strip(",.!?;:") + " " + words[i+1].strip(",.!?;:")
            bigrams.add(bg)

        # Check overlap with previous sentences
        overlap = len(bigrams & used_bigrams)
        overlap_ratio = overlap / len(bigrams) if bigrams else 0

        if overlap_ratio > 0.3 and len(words) > 6:
            # Compress repetitive sentence
            sent = ' '.join(words[:5]) + "."

        # Add bigrams to used set
        used_bigrams.update(bigrams)

        # Reset every 5 sentences to allow thematic continuation
        if len(processed) > 0 and len(processed) % 5 == 0:
            used_bigrams = set()

        processed.append(sent)

    return " ".join(processed)

# ===== BURSTINESS ENGINE =====

def syntactic_burstiness_engine(sentences):
    """Apply length variation patterns. No em-dashes."""
    if not sentences:
        return sentences

    total_words = sum(count_words(s) for s in sentences)
    result = []

    for i, sent in enumerate(sentences):
        words = sent.split()
        current_len = len(words)

        if i % 5 == 0:
            # LONG: expand with embedded clause
            target = int(current_len * 1.3)
            if len(words) < target:
                expansions = [
                    " through integrated feedback loops.",
                    " via polysynaptic pathways.",
                    " under homeostatic regulation.",
                    " through descending cortical input."
                ]
                sent = sent.rstrip('.') + random.choice(expansions)

        elif i % 5 == 1:
            # SHORT: compress
            target = max(int(current_len * 0.6), 4)
            if len(words) > target:
                sent = ' '.join(words[:target]) + '.'

        elif i % 5 == 2:
            # MEDIUM with semicolon
            if ';' not in sent and len(words) > 10:
                mid = len(words) // 2
                sent = ' '.join(words[:mid]) + '; ' + ' '.join(words[mid:])

        elif i % 5 == 3:
            # SHORT fragment
            if len(words) > 7:
                sent = ' '.join(words[:4]) + '.'

        elif i % 5 == 4:
            # LONG with parenthetical
            if len(words) < 18:
                parentheticals = [
                    " (a requirement that cannot be bypassed).",
                    " (this occurs involuntarily).",
                    " (under normal physiological conditions).",
                    " (a process essential for survival)."
                ]
                sent = sent.rstrip('.') + random.choice(parentheticals)

        sent = re.sub(r'\s+', ' ', sent).strip()
        if sent and sent[-1] not in '.!?':
            sent += '.'
        result.append(sent)

    # Verify total word count
    new_total = sum(count_words(s) for s in result)
    if abs(new_total - total_words) > int(total_words * 0.1):
        diff = new_total - total_words
        if diff > 0:
            longest_idx = max(range(len(result)), key=lambda i: count_words(result[i]))
            words = result[longest_idx].split()
            result[longest_idx] = ' '.join(words[:max(len(words) - diff, 3)]) + '.'

    return result

# ===== LOCAL FALLBACK =====

def local_humanize(sent, index):
    """Fallback humanization when Mistral fails."""
    words = sent.split()

    replacements = {
        r'\bimportant\b': random.choice(['key', 'critical', 'main']),
        r'\bplays a critical role\b': random.choice(['is essential', 'is vital', 'serves as']),
        r'\bplays a vital role\b': random.choice(['is essential', 'is critical', 'serves as']),
        r'\bis located\b': random.choice(['lies', 'sits', 'is found']),
        r'\bis composed of\b': random.choice(['contains', 'has', 'includes']),
        r'\bacts as\b': random.choice(['works as', 'functions as', 'serves as']),
        r'\bdue to\b': random.choice(['because of', 'owing to', 'as a result of']),
        r'\boverall\b': random.choice(['in sum', 'taken together', 'collectively']),
        r'\badditionally\b': random.choice(['also', 'plus', 'further']),
        r'\bhowever\b': random.choice(['yet', 'though', 'although']),
        r'\btherefore\b': random.choice(['thus', 'hence', 'so']),
        r'\bconsequently\b': random.choice(['as a result', 'thereby', 'accordingly']),
    }

    h = sent
    for pattern, replacement in replacements.items():
        h = re.sub(pattern, replacement, h, flags=re.IGNORECASE)

    # Vary sentence starters
    if index % 4 == 0 and not h.startswith('But') and not h.startswith('#') and not h.startswith('*'):
        h = 'But ' + h[0].lower() + h[1:]
    elif index % 4 == 2 and not h.startswith('Yet') and not h.startswith('#') and not h.startswith('*'):
        h = 'Yet ' + h[0].lower() + h[1:]

    return h

# ===== SYSTEM PROMPT =====

SYSTEM = """You are an elite academic editor. Rewrite AI-generated text into natural scholarly prose.

CRITICAL RULES:
1. STRICT LENGTH PARITY: Match original word count within +/- 3 words. NEVER add fluff.
2. SYNTACTIC COMPRESSION: If you expand one sentence, compress another in same paragraph.
3. THESIS RIGOR: Dense vocabulary. Use "epistemological", "heuristic" where natural.
4. BURSTINESS: Alternate lengths. Long (25+ words) → Short (5-8 words) → Medium (12-18) → Short (3-6).
5. PRESERVE CITATIONS: Keep (Author, 2020), [1], [1-3] exactly.
6. PRESERVE STRUCTURE: Keep headings (#, ##, ###), numbered lists, bullet points EXACTLY as markdown.
7. BANNED PHRASES: delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, operating continuously without conscious oversight, this structure, that structure, the present, the indicated, the respective
8. HUMAN QUIRKS: Use "we" occasionally. Start with "But", "Yet". Use fragments. Use semicolons; like this.
9. NEVER REPEAT: Don't use the same phrase twice in one paragraph.
10. WORD COUNT: If original is 12 words, output must be 10-15 words. No exceptions.
11. NO EM-DASHES: Use semicolons, colons, or parentheses instead of em-dashes.
12. HEADINGS: Keep markdown headings intact. Do not turn "## Location" into a sentence.
13. LISTS: Keep bullet points and numbered lists in markdown format.

OUTPUT ONLY JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact text","humanized":"rewrite","alternatives":["alt1","alt2","alt3"]}]}]}"""

# ===== MAIN PROCESSING =====

def humanize_with_mistral(paragraphs, style):
    print(f"CALLING MISTRAL with {len(paragraphs)} paragraphs")

    lines = []
    for i, para in enumerate(paragraphs):
        lines.append(f"Paragraph {i+1}:")
        for j, s in enumerate(para):
            lines.append(f"{j+1}. {s}")
        lines.append("")

    try:
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Style: {style}\n\nHumanize this academic text:\n\n" + "\n".join(lines)}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        text = resp.choices[0].message.content
        print(f"RAW: {text[:150]}...")

        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]

        data = json.loads(text)
    except Exception as e:
        print(f"Mistral failed: {e}")
        data = {
            "processed_paragraphs": [
                {
                    "sentences": [
                        {
                            "original": s,
                            "humanized": local_humanize(s, j),
                            "alternatives": [local_humanize(s, 0), local_humanize(s, 1), local_humanize(s, 2)]
                        } for j, s in enumerate(para)
                    ]
                } for para in paragraphs
            ]
        }

    result = []
    for i, para in enumerate(data.get("processed_paragraphs", [])):
        para_sentences = []

        for j, sent in enumerate(para.get("sentences", [])):
            orig = sent.get("original", "")
            h = sent.get("humanized", "")

            if not h:
                h = local_humanize(orig, j)

            # Apply constraints
            h = enforce_length_constraint(orig, h, max_diff=3)
            para_sentences.append({
                "orig": orig, 
                "hum": h, 
                "raw_alts": sent.get("alternatives", [])[:3]
            })

        # Apply burstiness at paragraph level
        humanized_only = [s["hum"] for s in para_sentences]
        burstiness_applied = syntactic_burstiness_engine(humanized_only)

        for j, (sent_data, h) in enumerate(zip(para_sentences, burstiness_applied)):
            # Apply obfuscation and repetition elimination
            h = enforce_length_constraint(sent_data["orig"], h, max_diff=4)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            h = enforce_length_constraint(sent_data["orig"], h, max_diff=5)

            score = score_sentence(h)

            # Process alternatives
            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                if not alt:
                    alt = local_humanize(sent_data["orig"], idx)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=4)
                alt = final_obfuscation_layer(alt)
                alt = eliminate_repetition(alt)
                alt = enforce_length_constraint(sent_data["orig"], alt, max_diff=5)
                alt = re.sub(r'\s+', ' ', alt).strip()
                if alt and alt[-1] not in '.!?':
                    alt += '.'
                clean_alts.append(alt)

            while len(clean_alts) < 3:
                clean_alts.append(local_humanize(sent_data["orig"], len(clean_alts)))

            para_sentences[j] = SentenceData(
                id=f"p{i}-s{j}",
                original=sent_data["orig"],
                humanized=h,
                alternatives=clean_alts[:3],
                score=score
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
        avg_score=round(avg, 1)
    )

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "mistral-large-latest"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
