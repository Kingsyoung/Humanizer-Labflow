import os

code = r'''import os
import re
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from mistralai import Mistral

API_KEY = "yUjr8SeWsnhS32Ec3mmrhEFae3cWTgZK"
if not API_KEY or API_KEY == "yUjr8SeWsnhS32Ec3mmrhEFae3cWTgZK":
    print("ERROR: Replace YOUR_ACTUAL_MISTRAL_KEY_HERE with your real key")
    exit(1)


client = Mistral(api_key=API_KEY)
app = FastAPI(title="Academic Humanizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

def split_sentences(text):
    text = re.sub(r"\b(e\.g\.|i\.e\.|et al\.|Fig\.|Dr\.|Prof\.)\s", lambda m: m.group(0).replace(".", "\x00"), text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.replace("\x00", ".").strip() for s in sents if s.strip()]

def split_paragraphs(text):
    return [split_sentences(p.strip()) for p in text.split("\n\n") if p.strip()]

AI_TELLS = ["delve", "testament", "pivotal", "moreover", "furthermore", "it is important to note",
            "it is crucial to note", "in conclusion", "landscape", "tapestry", "beacon", "underscore",
            "shed light on", "navigate", "ever-evolving", "multifaceted", "intricate", "robust",
            "leverage", "holistic", "paradigm", "synergy", "stakeholder", "crucially", "underscoring"]

ACADEMIC_TRANSITIONS = {
    "furthermore": ["Extending this observation", "Building upon this premise", "In tandem with these findings", "Concomitantly"],
    "moreover": ["Equally compelling", "A parallel consideration emerges", "This is reinforced by", "In the same vein"],
    "in conclusion": ["Ultimately, the data indicates", "Synthesizing these results", "In aggregate", "The cumulative evidence suggests"],
    "it is important to note": ["Notably", "Of paramount significance", "A critical caveat emerges", "Significantly"],
    "it is crucial to note": ["Notably", "Of paramount significance", "A critical caveat emerges", "Significantly"],
    "crucially": ["Notably", "Of paramount significance", "A critical caveat emerges", "Significantly"]
}

def score_sentence(sent):
    s, words = sent.lower(), sent.split()
    score = sum(15 for t in AI_TELLS if t in s)
    if 15 <= len(words) <= 20: score += 10
    first = words[0].lower().strip(",.:") if words else ""
    if first in ["furthermore", "moreover", "however", "therefore", "thus", "consequently", "additionally", "crucially"]:
        score += 12
    unique = len(set(w.lower() for w in words))
    if len(words) > 5 and unique / len(words) < 0.5: score += 15
    for phrase in ["furthermore", "moreover", "in conclusion", "it is important to note", "it is crucial to note", "crucially"]:
        if phrase in s: score += 20
    return min(100, max(0, score))

def post_process_humanize(text):
    sentences = split_sentences(text)
    processed = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        if i % 2 == 0 and len(words) < 15:
            sent = sent.rstrip(".") + ", a finding that challenges prevailing assumptions in the field."
        elif i % 2 == 1 and len(words) > 25:
            mid = len(words) // 2
            sent = " ".join(words[:mid]) + ". " + " ".join(words[mid:])
        sent_lower = sent.lower()
        for phrase, alts in ACADEMIC_TRANSITIONS.items():
            if phrase in sent_lower:
                sent = re.sub(re.escape(phrase), random.choice(alts), sent, flags=re.IGNORECASE)
        scholarly_swaps = {
            "important": random.choice(["critical", "paramount", "consequential", "material"]),
            "many": random.choice(["myriad", "multitudinous", "numerous", "variegated"]),
            "show": random.choice(["demonstrate", "evince", "manifest", "betoken"]),
            "use": random.choice(["utilize", "deploy", "employ", "leverage"]),
            "change": random.choice(["transformation", "metamorphosis", "transmutation", "paradigm shift"])
        }
        for old, new in scholarly_swaps.items():
            if old in sent.lower() and random.random() > 0.5:
                sent = re.sub(r"\b" + old + r"\b", new, sent, flags=re.IGNORECASE, count=1)
        if "," in sent and random.random() > 0.7:
            sent = sent.replace(",", "—", 1)
        if ";" not in sent and len(sent.split()) > 20 and random.random() > 0.6:
            words = sent.split()
            mid = len(words) // 2
            sent = " ".join(words[:mid]) + "; " + " ".join(words[mid:])
        if i % 3 == 2 and random.random() > 0.6:
            sent = sent.rstrip(".") + ". A striking deviation from expected outcomes."
        processed.append(sent)
    return " ".join(processed)

SYSTEM = """You are an elite academic thesis writer. Your task is to rewrite AI-generated text into first-class scholarly prose that evades AI detection.

STRICT REQUIREMENTS:
1. THESIS-LEVEL RIGOR: Maintain dense, authoritative academic vocabulary. Use terms like "epistemological", "heuristic", "ontological", "axiomatic", "dialectical" where appropriate.

2. BURSTINESS (Critical for evasion):
   - Write ONE extremely complex sentence (40+ words, multiple embedded clauses)
   - Follow with ONE short, direct sentence (5-10 words)
   - Then a medium sentence (15-20 words)
   - Then a fragment or abrupt statement
   - Pattern: LONG → SHORT → MEDIUM → FRAGMENT → LONG

3. PERPLEXITY (Critical for evasion):
   - Use unexpected but precise scholarly terms
   - Vary sentence openings: start with prepositional phrases, participial phrases, absolute constructions
   - NEVER start two consecutive sentences with the same grammatical structure
   - Use "we" or "this study" occasionally (human academics do this)

4. BANNED (Zero tolerance):
   delve, testament, pivotal, moreover, furthermore, crucially, underscore, shed light on, navigate, landscape, tapestry, beacon, robust, holistic, paradigm, synergy, stakeholder, leverage, multifaceted, intricate, ever-evolving, in conclusion, it is important to note, not only...but also

5. SCHOLARLY REPLACEMENTS:
   - "Furthermore" → "Extending this observation", "Building upon this premise", "In tandem with these findings"
   - "Moreover" → "Equally compelling", "A parallel consideration emerges"
   - "In conclusion" → "Ultimately, the data indicates", "Synthesizing these results"
   - "It is important to note" → "Notably", "Of paramount significance", "A critical caveat emerges"

6. HUMAN ACADEMIC QUIRKS:
   - Use em-dashes for abrupt shifts—like this
   - Use semicolons to join related independent clauses
   - Occasionally use "we argue" or "our analysis reveals"
   - Include one slightly unexpected comparison or metaphor per paragraph

OUTPUT ONLY JSON:
{"processed_paragraphs":[{"sentences":[{"original":"exact text","humanized":"rewrite following ALL rules","alternatives":["alt1 with different structure","alt2 with different vocabulary","alt3 with different voice"]}]}]}"""

def humanize_with_mistral(paragraphs, style):
    print(f"CALLING MISTRAL with {len(paragraphs)} paragraphs")
    lines = []
    for i, para in enumerate(paragraphs):
        lines.append(f"Paragraph {i+1}:")
        for j, s in enumerate(para):
            lines.append(f"{j+1}. {s}")
        lines.append("")
    resp = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Style: {style}\n\nHumanize this academic text:\n\n" + "\n".join(lines)}
        ],
        temperature=0.85,
        max_tokens=4000,
        response_format={"type": "json_object"}
    )
    text = resp.choices[0].message.content
    print(f"RAW RESPONSE: {text[:200]}...")
    data = json.loads(text.strip())
    result = []
    for i, para in enumerate(data.get("processed_paragraphs", [])):
        sents = []
        for j, sent in enumerate(para.get("sentences", [])):
            h = sent.get("humanized", "")
            h = post_process_humanize(h)
            score = score_sentence(h)
            attempts = 0
            while score > 40 and attempts < 2:
                print(f"  SENTENCE {j} SCORE {score} - CRITIC PASS {attempts+1}")
                feedback = f"This sentence scores {score}% AI probability. Rewrite with: "
                words = h.split()
                if len(words) < 12:
                    feedback += "MORE words (expand with clauses); "
                elif 15 <= len(words) <= 20:
                    feedback += "BREAK into two sentences or expand to 30+ words; "
                if len(set(w.lower() for w in words)) / len(words) > 0.65:
                    feedback += "use more specialized vocabulary; "
                feedback += "add em-dashes, semicolons, or a fragment. Keep thesis-level rigor."
                critic = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": feedback + "\n\nSentence: " + h}
                    ],
                    temperature=0.95,
                    max_tokens=500
                )
                h = critic.choices[0].message.content.strip("\"'\n")
                h = post_process_humanize(h)
                score = score_sentence(h)
                attempts += 1
                print(f"  NEW SCORE: {score}")
            sents.append(SentenceData(
                id=f"p{i}-s{j}",
                original=sent.get("original", ""),
                humanized=h,
                alternatives=sent.get("alternatives", [])[:3],
                score=score
            ))
        result.append(ParagraphData(id=f"para-{i}", sentences=sents))
    print(f"RETURNING {len(result)} paragraphs")
    return result

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
'''

with open("main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("main.py created successfully")