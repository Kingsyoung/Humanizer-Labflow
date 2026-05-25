import os

code = """\"use client\";

import React, { useState, useRef, useEffect } from \"react\";
import { Sparkles, FileText, CheckCircle2, RotateCcw, Copy, Check, Zap, Shield, AlertTriangle, BarChart3, Settings } from \"lucide-react\";

interface SentenceData {
  id: string;
  original: string;
  humanized: string;
  alternatives: string[];
  score: number;
}

interface ParagraphData {
  id: string;
  sentences: SentenceData[];
}

export default function HumanizerPlayground() {
  const [inputText, setInputText] = useState(\"\");
  const [isLoading, setIsLoading] = useState(false);
  const [paragraphs, setParagraphs] = useState<<ParagraphData[]>([]);
  const [selectedSentenceId, setSelectedSentenceId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [style, setStyle] = useState(\"academic\");
  const [showSettings, setShowSettings] = useState(false);
  const [stats, setStats] = useState({ total: 0, avgScore: 0, highRisk: 0, safe: 0 });
  const dropdownRef = useRef<<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setSelectedSentenceId(null);
      }
    }
    document.addEventListener(\"mousedown\", handleClickOutside);
    return () => document.removeEventListener(\"mousedown\", handleClickOutside);
  }, []);

  const handleHumanize = async () => {
    if (!inputText.trim()) return;
    setIsLoading(true);
    setSelectedSentenceId(null);

    try {
      const response = await fetch(\"http://localhost:8000/api/process-text\", {
        method: \"POST\",
        headers: { \"Content-Type\": \"application/json\" },
        body: JSON.stringify({ text: inputText, style }),
      });

      if (!response.ok) throw new Error(\"Processing failed\");
      
      const data = await response.json();
      setParagraphs(data.processed_paragraphs);
      
      const allSentences = data.processed_paragraphs.flatMap((p: ParagraphData) => p.sentences);
      setStats({
        total: data.total_sentences,
        avgScore: data.avg_score,
        highRisk: allSentences.filter((s: SentenceData) => s.score >= 70).length,
        safe: allSentences.filter((s: SentenceData) => s.score < 40).length
      });
    } catch (error) {
      console.error(\"Error:\", error);
      alert(\"Backend not running. Start it with: python main.py\");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAlternativeSelect = (paragraphId: string, sentenceId: string, alternativeText: string) => {
    setParagraphs(prev => prev.map(p => {
      if (p.id !== paragraphId) return p;
      return {
        ...p,
        sentences: p.sentences.map(s => {
          if (s.id !== sentenceId) return s;
          const newAlternatives = [s.humanized, ...s.alternatives.filter(alt => alt !== alternativeText)];
          return { ...s, humanized: alternativeText, alternatives: newAlternatives, score: Math.max(15, s.score - 25) };
        })
      };
    }));
    setSelectedSentenceId(null);
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getScoreColorClass = (score: number) => {
    if (score >= 80) return \"bg-red-500/10 hover:bg-red-500/20 border-b-2 border-red-400/40\";
    if (score >= 50) return \"bg-amber-500/10 hover:bg-amber-500/20 border-b-2 border-amber-400/40\";
    return \"bg-emerald-500/5 hover:bg-emerald-500/15 border-b-2 border-emerald-400/20\";
  };

  const getScoreBadge = (score: number) => {
    if (score >= 80) return { bg: \"bg-red-50\", text: \"text-red-600\", label: \"High Risk\" };
    if (score >= 50) return { bg: \"bg-amber-50\", text: \"text-amber-600\", label: \"Moderate\" };
    return { bg: \"bg-emerald-50\", text: \"text-emerald-600\", label: \"Safe\" };
  };

  return (
    <div className=\"min-h-screen bg-slate-50 text-slate-900 font-sans\">
      <header className=\"border-b border-slate-200 bg-white sticky top-0 z-40 px-6 py-4 flex items-center justify-between shadow-sm\">
        <div className=\"flex items-center space-x-3\">
          <div className=\"bg-indigo-600 p-2 rounded-xl text-white\">
            <Sparkles className=\"w-5 h-5\" />
          </div>
          <div>
            <h1 className=\"font-bold text-lg tracking-tight\">StealthWriter <span className=\"text-indigo-600\">Academic</span></h1>
            <p className=\"text-xs text-slate-500 font-medium tracking-wide\">HIGH-TEMPO SCHOLARLY HUMANIZER</p>
          </div>
        </div>
        <div className=\"flex items-center space-x-4\">
          <button onClick={() => setShowSettings(!showSettings)} className=\"text-slate-400 hover:text-slate-600 p-2 rounded-lg hover:bg-slate-100\">
            <Settings className=\"w-4 h-4\" />
          </button>
          <span className=\"text-xs bg-slate-100 text-slate-600 font-semibold px-2.5 py-1 rounded-full border border-slate-200\">Engine: Mistral-Large</span>
        </div>
      </header>

      {showSettings && (
        <div className=\"bg-white border-b border-slate-200 px-6 py-4\">
          <div className=\"max-w-7xl mx-auto flex items-center space-x-6\">
            <div className=\"flex items-center space-x-2\">
              <span className=\"text-sm font-medium text-slate-700\">Style:</span>
              <select value={style} onChange={(e) => setStyle(e.target.value)} className=\"text-sm border border-slate-200 rounded-lg px-3 py-1.5 bg-slate-50\">
                <option value=\"academic\">Academic</option>
                <option value=\"technical\">Technical</option>
                <option value=\"casual\">Casual</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {paragraphs.length > 0 && (
        <div className=\"bg-white border-b border-slate-200 px-6 py-3\">
          <div className=\"max-w-7xl mx-auto flex items-center space-x-8\">
            <div className=\"flex items-center space-x-2\">
              <BarChart3 className=\"w-4 h-4 text-indigo-500\" />
              <span className=\"text-sm font-semibold text-slate-700\">Analysis Complete</span>
            </div>
            <div className=\"text-xs\"><span className=\"text-slate-500\">Sentences:</span> <span className=\"font-bold\">{stats.total}</span></div>
            <div className=\"text-xs\"><span className=\"text-slate-500\">Avg Score:</span> <span className={\"font-bold \" + (stats.avgScore >= 70 ? \"text-red-600\" : stats.avgScore >= 50 ? \"text-amber-600\" : \"text-emerald-600\")}>{stats.avgScore}%</span></div>
            <div className=\"text-xs flex items-center space-x-1\"><AlertTriangle className=\"w-3 h-3 text-red-500\" /><span className=\"text-slate-500\">High Risk:</span> <span className=\"font-bold text-red-600\">{stats.highRisk}</span></div>
            <div className=\"text-xs flex items-center space-x-1\"><Shield className=\"w-3 h-3 text-emerald-500\" /><span className=\"text-slate-500\">Safe:</span> <span className=\"font-bold text-emerald-600\">{stats.safe}</span></div>
          </div>
        </div>
      )}

      <main className=\"max-w-7xl mx-auto p-6 md:p-8 grid grid-cols-1 lg:grid-cols-2 gap-6\" style={{minHeight: \"calc(100vh - 140px)\"}}>
        <div className=\"flex flex-col bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden\" style={{height: \"680px\"}}>
          <div className=\"flex items-center justify-between px-5 py-4 border-b border-slate-100 bg-slate-50/50\">
            <div className=\"flex items-center space-x-2 text-slate-700 font-semibold text-sm\">
              <FileText className=\"w-4 h-4 text-slate-500\" />
              <span>Draft Artificial Text</span>
            </div>
            <div className=\"flex items-center space-x-2\">
              <span className=\"text-xs text-slate-400\">{inputText.split(/\\s+/).filter(Boolean).length} words</span>
              <button onClick={() => setInputText(\"\")} className=\"text-slate-400 hover:text-slate-600 p-1 rounded-lg hover:bg-slate-100\">
                <RotateCcw className=\"w-4 h-4\" />
              </button>
            </div>
          </div>
          <textarea value={inputText} onChange={(e) => setInputText(e.target.value)} placeholder=\"Paste your standard AI-generated essay, literature review, or methodology draft here...\" className=\"flex-1 w-full p-6 resize-none focus:outline-none text-slate-700 leading-relaxed text-base\" />
          <div className=\"p-4 bg-slate-50 border-t border-slate-100 flex justify-between items-center\">
            <div className=\"text-xs text-slate-400 flex items-center space-x-1\">
              <Zap className=\"w-3 h-3\" />
              <span>Auto-detects AI tells</span>
            </div>
            <button onClick={handleHumanize} disabled={isLoading || !inputText.trim()} className=\"bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-200 disabled:text-slate-400 text-white font-semibold text-sm px-5 py-2.5 rounded-xl inline-flex items-center space-x-2 cursor-pointer disabled:cursor-not-allowed\">
              {isLoading ? (
                <><div className=\"w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin\" /><span>Humanizing Structure...</span></>
              ) : (
                <><Sparkles className=\"w-4 h-4\" /><span>Execute Academic Humanization</span></>
              )}
            </button>
          </div>
        </div>

        <div className=\"flex flex-col bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden\" style={{height: \"680px\"}}>
          <div className=\"flex items-center justify-between px-5 py-4 border-b border-slate-100 bg-slate-50/50\">
            <div className=\"flex items-center space-x-2 text-slate-700 font-semibold text-sm\">
              <CheckCircle2 className=\"w-4 h-4 text-indigo-500\" />
              <span>Interactive Scholarly Output</span>
            </div>
            {paragraphs.length > 0 && (
              <button onClick={() => { const fullText = paragraphs.map(p => p.sentences.map(s => s.humanized).join(\" \")).join(\"\\n\\n\"); copyToClipboard(fullText, \"all-text\"); }} className=\"text-xs font-semibold text-indigo-600 hover:text-indigo-700 inline-flex items-center space-x-1 px-2 py-1 rounded-lg hover:bg-indigo-50\">
                {copiedId === \"all-text\" ? <Check className=\"w-3.5 h-3.5\" /> : <Copy className=\"w-3.5 h-3.5\" />}
                <span>{copiedId === \"all-text\" ? \"Copied All!\" : \"Copy Full Output\"}</span>
              </button>
            )}
          </div>

          <div className=\"flex-1 p-6 overflow-y-auto space-y-6\" ref={dropdownRef}>
            {isLoading ? (
              <div className=\"h-full flex flex-col items-center justify-center space-y-3 text-slate-400\">
                <div className=\"w-8 h-8 border-3 border-indigo-600/20 border-t-indigo-600 rounded-full animate-spin\" />
                <p className=\"text-xs font-medium tracking-wide animate-pulse\">RECONSTRUCTING SYNTACTIC BURSTINESS...</p>
              </div>
            ) : paragraphs.length > 0 ? (
              paragraphs.map((paragraph) => (
                <p key={paragraph.id} className=\"leading-loose text-base text-slate-800\">
                  {paragraph.sentences.map((sentence) => {
                    const isSelected = selectedSentenceId === sentence.id;
                    const badge = getScoreBadge(sentence.score);
                    return (
                      <span key={sentence.id} className=\"relative inline\">
                        <span onClick={() => setSelectedSentenceId(isSelected ? null : sentence.id)} className={\"cursor-pointer transition-all duration-150 rounded-sm py-0.5 px-0.5 inline \" + getScoreColorClass(sentence.score) + (isSelected ? \" ring-2 ring-indigo-500 ring-offset-1 bg-indigo-500/10\" : \"\")} title={\"Score: \" + sentence.score + \"%\"}>
                          {sentence.humanized}{\" \"}
                        </span>

                        {isSelected && (
                          <div className=\"absolute left-0 top-full mt-2 w-full min-w-[380px] max-w-[520px] bg-white border border-slate-200 rounded-xl shadow-2xl z-50 overflow-hidden text-sm\">
                            <div className=\"bg-slate-50 px-4 py-3 border-b border-slate-100 flex justify-between items-center\">
                              <div className=\"flex items-center space-x-2\">
                                <span className=\"text-xs font-bold tracking-wide text-slate-500 uppercase\">Scholarly Alternatives</span>
                                <span className={\"text-[10px] font-bold px-2 py-0.5 rounded-full \" + badge.bg + \" \" + badge.text}>{badge.label} • {sentence.score}%</span>
                              </div>
                              <button onClick={() => copyToClipboard(sentence.humanized, sentence.id)} className=\"text-slate-400 hover:text-indigo-600 p-1 rounded\">
                                {copiedId === sentence.id ? <Check className=\"w-3.5 h-3.5\" /> : <Copy className=\"w-3.5 h-3.5\" />}
                              </button>
                            </div>
                            <div className=\"px-4 py-2 bg-amber-50/50 border-b border-slate-100\">
                              <span className=\"text-[10px] font-bold text-amber-600 uppercase\">Original</span>
                              <p className=\"text-xs text-slate-500 mt-0.5 italic\">{sentence.original}</p>
                            </div>
                            <div className=\"divide-y divide-slate-100 max-h-[280px] overflow-y-auto\">
                              {sentence.alternatives.map((alt, index) => (
                                <button key={index} onClick={() => handleAlternativeSelect(paragraph.id, sentence.id, alt)} className=\"w-full text-left p-3.5 hover:bg-slate-50 text-slate-700 hover:text-indigo-700 text-sm leading-relaxed font-medium group\">
                                  <div className=\"flex items-start space-x-2\">
                                    <span className=\"text-[10px] font-bold text-slate-300 mt-0.5 group-hover:text-indigo-400\">{index + 1}</span>
                                    <span>{alt}</span>
                                  </div>
                                </button>
                              ))}
                            </div>
                            <div className=\"bg-slate-50 p-2 border-t border-slate-100 flex justify-between items-center\">
                              <span className=\"text-[10px] text-slate-400\">Click to replace sentence</span>
                              <button onClick={() => setSelectedSentenceId(null)} className=\"text-[11px] font-bold text-slate-400 hover:text-slate-600 px-2 py-1 rounded hover:bg-slate-200\">Close</button>
                            </div>
                          </div>
                        )}
                      </span>
                    );
                  })}
                </p>
              ))
            ) : (
              <div className=\"h-full flex flex-col items-center justify-center text-center p-6 border-2 border-dashed border-slate-200 rounded-xl\">
                <Sparkles className=\"w-8 h-8 text-slate-300 mb-2\" />
                <p className=\"text-sm font-semibold text-slate-400\">No output ready yet</p>
                <p className=\"text-xs text-slate-400 max-w-xs mt-1\">Input your raw draft on the left and execute the engine.</p>
              </div>
            )}
          </div>

          {paragraphs.length > 0 && (
            <div className=\"px-5 py-3.5 bg-slate-50 border-t border-slate-100 flex space-x-6 items-center text-xs text-slate-500 font-medium\">
              <span className=\"flex items-center space-x-1.5\"><span className=\"w-2.5 h-2.5 rounded-sm bg-red-400/30 border border-red-400\" /><span>Highly Predictable (Fix)</span></span>
              <span className=\"flex items-center space-x-1.5\"><span className=\"w-2.5 h-2.5 rounded-sm bg-amber-400/30 border border-amber-400\" /><span>Moderate AI Signature</span></span>
              <span className=\"flex items-center space-x-1.5\"><span className=\"w-2.5 h-2.5 rounded-sm bg-emerald-400/20 border border-emerald-400\" /><span>Scholarly Flow Safe</span></span>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
"""

with open("src/app/page.tsx", "w", encoding="utf-8") as f:
    f.write(code)
print("File created successfully")

