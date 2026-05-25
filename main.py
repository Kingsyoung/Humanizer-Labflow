
import os
import re
import json
import random
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq

# ===== API KEY FROM ENVIRONMENT =====
API_KEY = os.getenv("GROQ_API_KEY", "")
if not API_KEY:
    print("ERROR: No API key. Set GROQ_API_KEY environment variable.")
    exit(1)

print(f"Groq API Key loaded: {API_KEY[:8]}...")
client = Groq(api_key=API_KEY)

# ===== FASTAPI APP (ONE TIME ONLY) =====
app = FastAPI(title="Academic Humanizer")

# ===== CORS (ONE TIME ONLY) =====
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

# ===== TEXT SPLITTING =====

def split_sentences(text):
    text = re.sub(r"\\b(e\\.g\\.|i\\.e\\.|et al\\.|Fig\\.|Dr\\.|Prof\\.)\\s", lambda m: m.group(0).replace(".", "\\x00"), text)
    sents = re.split(r"(?<=[.!?])\\s+(?=[A-Z])", text.strip())
    return [s.replace("\\x00", ".").strip() for s in sents if s.strip()]

def split_paragraphs(text):
    return [split_sentences(p.strip()) for p in text.split("\\n\\n") if p.strip()]

# ===== SCORING =====

AI_TELLS = ["delve", "testament", "pivotal", "moreover", "furthermore", "it is important to note",
            "it is crucial to note", "in conclusion", "landscape", "tapestry", "beacon", "underscore",
            "shed light on", "navigate", "ever-evolving", "multifaceted", "intricate", "robust",
            "leverage", "holistic", "paradigm", "synergy", "stakeholder", "crucially", "underscoring",
            "operating continuously without conscious oversight", "this structure", "that structure",
            "the present", "the indicated", "the respective", "it is worth noting", "as mentioned earlier",
            "it should be noted", "indeed", "arguably", "notably", "significantly"]

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
    # Penalize broken fragments and repetition
    if sent.count(",") > 3 or sent.count(";") > 2: score += 15
    if "operating continuously" in s: score += 25
    # Penalize short broken sentences
    if len(words) < 4 and not sent.startswith("#") and not sent.startswith("*"): score += 20
    return min(100, max(0, score))

def count_words(text):
    return len(text.split())

# ===== LENGTH CONSTRAINTS WITH CORRECTION LOOP =====

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

def enforce_length_constraint(original, humanized, max_diff=3):
    """Strict length enforcement with truncation/expansion."""
    orig_count = count_words(original)
    hum_count = count_words(humanized)
    
    if abs(orig_count - hum_count) <= max_diff:
        return humanized
    
    if hum_count > orig_count + max_diff:
        words = humanized.split()
        # Keep enough words to match original length
        keep = max(orig_count + max_diff - 1, min(orig_count, len(words)))
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

def validate_and_correct_length(original, humanized, max_diff=3):
    """Validate length and return corrected if needed."""
    orig_count = count_words(original)
    hum_count = count_words(humanized)
    
    if abs(orig_count - hum_count) <= max_diff:
        return humanized
    
    # If too far off, use strict enforcement
    return enforce_length_constraint(original, humanized, max_diff)

# ===== FINAL OBfuscation LAYER =====

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
        sent = re.sub(r'\\s+', ' ', sent).strip()
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

        sent = re.sub(r'\\s+', ' ', sent).strip()
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

# ===== LOCAL FALLBACK (IMPROVED — actually transforms text) =====

# BETTER sentence starters — varied and natural, not forced "But"/"Yet"
SENTENCE_STARTERS = [
    "Interestingly, ", "Notably, ", "In practice, ", "Under these conditions, ",
    "Specifically, ", "In effect, ", "Conversely, ", "As expected, ",
    "In this context, ", "Evidently, ", "Consequently, ", "Alternatively, ",
    "In particular, ", "By comparison, ", "In such cases, ", "Naturally, "
]

def local_humanize(sent, index):
    """Fallback when Groq fails. Actually transforms the text, not just echoes it."""
    words = sent.split()
    if not words:
        return sent

    # Step 1: Word replacements (24 patterns)
    replacements = {
        r'\\bimportant\\b': random.choice(['key', 'critical', 'main', 'essential']),
        r'\\bplays a critical role\\b': random.choice(['is essential', 'is vital', 'serves as']),
        r'\\bplays a vital role\\b': random.choice(['is essential', 'is critical', 'serves as']),
        r'\\bis located\\b': random.choice(['lies', 'sits', 'is found', 'is situated']),
        r'\\bis composed of\\b': random.choice(['contains', 'has', 'includes', 'comprises']),
        r'\\bacts as\\b': random.choice(['works as', 'functions as', 'serves as', 'operates as']),
        r'\\bdue to\\b': random.choice(['because of', 'owing to', 'as a result of', 'stemming from']),
        r'\\boverall\\b': random.choice(['in sum', 'taken together', 'collectively', 'broadly']),
        r'\\badditionally\\b': random.choice(['also', 'plus', 'further', 'moreover']),
        r'\\bhowever\\b': random.choice(['yet', 'though', 'although', 'nevertheless']),
        r'\\btherefore\\b': random.choice(['thus', 'hence', 'so', 'accordingly']),
        r'\\bconsequently\\b': random.choice(['as a result', 'thereby', 'accordingly', 'hence']),
        r'\\bregulates\\b': random.choice(['controls', 'governs', 'modulates', 'directs']),
        r'\\bcontains\\b': random.choice(['holds', 'possesses', 'encompasses', 'incorporates']),
        r'\\bresponsible for\\b': random.choice(['accountable for', 'charged with', 'tasked with']),
        r'\\bassociated with\\b': random.choice(['linked to', 'tied to', 'connected with', 'related to']),
        r'\\binvolved in\\b': random.choice(['engaged in', 'participating in', 'taking part in']),
        r'\\bconsists of\\b': random.choice(['comprises', 'is made up of', 'incorporates']),
        r'\\bpart of\\b': random.choice(['component of', 'element of', 'constituent of']),
        r'\\bfunction\\b': random.choice(['role', 'purpose', 'operation', 'activity']),
        r'\\bstructure\\b': random.choice(['anatomy', 'architecture', 'framework', 'morphology']),
        r'\\bprocess\\b': random.choice(['mechanism', 'procedure', 'pathway', 'sequence']),
        r'\\bcontrol\\b': random.choice(['regulation', 'management', 'oversight', 'direction']),
    }

    h = sent
    for pattern, replacement in replacements.items():
        h = re.sub(pattern, replacement, h, flags=re.IGNORECASE)

    # Step 2: Sentence structure variations
    if index % 3 == 0 and len(words) > 8:
        # Compress to fragment
        h = ' '.join(words[:6]) + '.'
    elif index % 3 == 1 and len(words) < 10:
        # Expand with clause
        h = h.rstrip('.') + ', which remains essential for proper function.'
    elif index % 3 == 2 and ',' in h:
        # Semicolon swap
        h = h.replace(',', ';', 1)

    # Step 3: Sentence starter variation (natural academic)
    if index % 5 == 0 and len(words) > 5 and not h.startswith('#') and not h.startswith('*'):
        starter = random.choice(SENTENCE_STARTERS)
        first_char = h[0]
        if first_char.isupper():
            h = starter + first_char.lower() + h[1:]
        else:
            h = starter + h

    return h

# ===== SYSTEM PROMPT (ELITE ACADEMIC EDITOR) =====

SYSTEM = """You are an elite academic editor specializing in neuroanatomy and biomedical sciences. Rewrite AI-generated text into natural, publication-ready scholarly prose that passes Turnitin, ZeroGPT, and all AI detectors.

CRITICAL ARCHITECTURAL CONSTRAINTS:

1. STRICT WORD-COUNT PARITY (1:1 Density Ratio)
- For every sentence you process, calculate the EXACT word count in the "original" string.
- The "humanized" output must match that word count within +/- 2 words maximum.
- Do NOT pad with fluff adjectives or elongate text. Use singular high-value nouns and concise verb phrases.
- Example: If original is 12 words, rewrite must be 10-14 words.

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
- Use parentheticals sparingly (notably), (evidently), (under normal conditions).
- Vary sentence openers: "Interestingly," "Specifically," "In this context," "Conversely," "As expected,"

5. REPETITION ELIMINATION
- Never use the same noun phrase twice in one paragraph.
- If "the cerebellum" appears in sentence 1, use "this structure," "the respective organ," or "it" in sentence 2.
- Vary verb phrases: "regulates" → "controls" → "governs" → "modulates".

6. CITATION & MARKDOWN PRESERVATION
- Keep (Author, 2020), [1], [1-3] exactly as written.
- Preserve # headings, ## subheadings, * bullet points, 1. numbered lists.
- Do not turn "## Location" into a sentence.

7. ACADEMIC TONE TARGET
- Write like a tenured professor with 30 years of publishing experience.
- Use precise terminology: "afferent pathways," "proprioceptive feedback," "vestibulocerebellar tracts."
- Avoid generic transitions. Each sentence should advance the argument or provide new anatomical detail.
- Use active voice 60% of the time, passive 40%.

OUTPUT ONLY VALID JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact text","humanized":"rewrite","alternatives":["alt1","alt2","alt3"]}]}]}"""

# ===== CORRECTION LOOP FOR LENGTH =====

def correction_loop(original, humanized, max_attempts=2):
    """Send back to LLM if length is too far off."""
    orig_count = count_words(original)
    hum_count = count_words(humanized)
    
    if abs(orig_count - hum_count) <= 3:
        return humanized
    
    for attempt in range(max_attempts):
        try:
            correction_prompt = f"""The following rewritten academic sentence violates our strict length constraint.
Original word count: {orig_count} words.
Your rewrite count: {hum_count} words.

Original: "{original}"
Your Rewrite: "{humanized}"

Task: Adjust your rewrite so it matches EXACTLY {orig_count} words (tolerance +/- 2 words). 
Maintain elite academic cadence and precise terminology. 
Use the SYNTACTIC COMPRESSION RULE: if original is long, keep it long; if short, keep it short.
Output ONLY the corrected sentence string, no quotes, no explanation."""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            corrected = resp.choices[0].message.content.strip().strip('"').strip("'")
            new_count = count_words(corrected)
            
            if abs(orig_count - new_count) <= 3:
                return corrected
            
            humanized = corrected  # Try again with this as base
            hum_count = new_count
            
        except Exception as e:
            print(f"Correction loop attempt {attempt+1} failed: {e}")
            break
    
    # If all corrections fail, use strict enforcement
    return enforce_length_constraint(original, humanized, max_diff=3)

# ===== MAIN PROCESSING =====

def humanize_with_groq(paragraphs, style):
    print(f"CALLING GROQ with {len(paragraphs)} paragraphs")
    lines = []
    for i, para in enumerate(paragraphs):
        lines.append(f"Paragraph {i+1}:")
        for j, s in enumerate(para):
            lines.append(f"{j+1}. [{count_words(s)} words] {s}")
        lines.append("")

    data = None
    groq_error = None

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Style: {style}\\n\\nHumanize this academic text. I have marked word counts in [brackets] for each sentence. Match them exactly within +/- 2 words:\\n\\n" + "\\n".join(lines)}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        text = resp.choices[0].message.content
        print(f"RAW: {text[:300]}...")

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
        print(f"Groq JSON parsed successfully")
    except Exception as e:
        groq_error = str(e)
        print(f"Groq FAILED: {e}")

    if not data or not data.get("processed_paragraphs"):
        print(f"Using LOCAL FALLBACK. Error was: {groq_error}")
        data = {
            "processed_paragraphs": [
                {
                    "sentences": [
                        {
                            "original": s,
                            "humanized": local_humanize(s, j),
                            "alternatives": [
                                local_humanize(s, 0),
                                local_humanize(s, 1),
                                local_humanize(s, 2)
                            ]
                        } for j, s in enumerate(para)
                    ]
                } for para in paragraphs
            ]
        }

    result = []
    for i, para in enumerate(data.get("processed_paragraphs", [])):
        para_sentences = []
        para_total_original = 0
        for j, sent in enumerate(para.get("sentences", [])):
            orig = sent.get("original", "")
            h = sent.get("humanized", "")
            if not h:
                h = local_humanize(orig, j)
            
            # CORRECTION LOOP: Send back to LLM if length is wrong
            h = correction_loop(orig, h)
            
            # Additional strict enforcement
            h = validate_and_correct_length(orig, h, max_diff=3)
            
            para_sentences.append({"orig": orig, "hum": h, "raw_alts": sent.get("alternatives", [])[:3]})
            para_total_original += count_words(orig)

        # Apply burstiness at paragraph level
        humanized_only = [s["hum"] for s in para_sentences]
        burstiness_applied = syntactic_burstiness_engine(humanized_only)

        for j, (sent_data, h) in enumerate(zip(para_sentences, burstiness_applied)):
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            h = final_obfuscation_layer(h)
            h = eliminate_repetition(h)
            h = validate_and_correct_length(sent_data["orig"], h, max_diff=3)
            score = score_sentence(h)

            # Process alternatives with correction loop
            clean_alts = []
            for idx, alt in enumerate(sent_data["raw_alts"]):
                if not alt:
                    alt = local_humanize(sent_data["orig"], idx)
                
                # Apply correction loop to alternatives too
                alt = correction_loop(sent_data["orig"], alt)
                alt = validate_and_correct_length(sent_data["orig"], alt, max_diff=3)
                alt = final_obfuscation_layer(alt)
                alt = eliminate_repetition(alt)
                alt = validate_and_correct_length(sent_data["orig"], alt, max_diff=3)
                alt = re.sub(r'\\s+', ' ', alt).strip()
                if alt and alt[-1] not in '.!?':
                    alt += '.'
                clean_alts.append(alt)

            # Ensure 3 unique alternatives, not duplicates of original
            while len(clean_alts) < 3:
                fallback = local_humanize(sent_data["orig"], len(clean_alts) + 10)
                fallback = correction_loop(sent_data["orig"], fallback)
                fallback = validate_and_correct_length(sent_data["orig"], fallback, max_diff=3)
                clean_alts.append(fallback)
            
            # Remove any duplicates of the original
            clean_alts = [alt for alt in clean_alts if alt.lower().strip() != sent_data["orig"].lower().strip()]
            while len(clean_alts) < 3:
                fallback = local_humanize(sent_data["orig"], len(clean_alts) + 20)
                fallback = validate_and_correct_length(sent_data["orig"], fallback, max_diff=3)
                clean_alts.append(fallback)

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
    processed = humanize_with_groq(pars, req.style)
    all_s = [s for p in processed for s in p.sentences]
    avg = sum(s.score for s in all_s) / len(all_s) if all_s else 0
    return ProcessResponse(
        processed_paragraphs=processed,
        total_sentences=len(all_s),
        avg_score=round(avg, 1)
    )

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "llama-3.3-70b-versatile"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
