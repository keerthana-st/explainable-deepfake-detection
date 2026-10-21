import React, { useState } from 'react';
import { Copy, Check, ArrowLeft } from 'lucide-react';
import type { AnalysisResult } from '../types';

interface RawOutputViewProps {
  analysisResult: AnalysisResult | null;
  onBack: () => void;
}

export const RawOutputView: React.FC<RawOutputViewProps> = ({
  analysisResult,
  onBack,
}) => {
  const [copied, setCopied] = useState(false);

  const payload = analysisResult || {
    filename: 'dog.jpg',
    global_verdict: 'AI-GENERATED / FAKE',
    confidence: 0.90,
    decision_basis: 'Cryptographic C2PA metadata confirms generative provenance (DALL-E 3).',
    primary_system: {
      frequency: { score: 0.82, is_suspicious: true },
      classifier: { score: 0.76, prediction: 'FAKE' },
      metadata: { flag: 'AI_PROVENANCE_C2PA', tool: 'DALLE-3' },
    },
    secondary_system: {
      regions: [
        { region: 'Face', prediction: 'Fake', confidence: 0.88 },
        { region: 'Clothing', prediction: 'Real', confidence: 0.64 },
        { region: 'Background', prediction: 'Real', confidence: 0.71 },
      ],
      regional_consistency: 'SUPPORTED',
    },
  };

  const jsonStr = JSON.stringify(payload, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-slate-700 text-xs font-semibold hover:bg-slate-50 transition cursor-pointer"
        >
          <ArrowLeft className="h-3 w-3" />
          <span>Back to Visual View</span>
        </button>

        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold transition cursor-pointer shadow-xs"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          <span>{copied ? 'Copied to Clipboard' : 'Copy JSON'}</span>
        </button>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-slate-900 text-slate-100 p-5 font-mono text-xs overflow-x-auto shadow-inner max-h-[600px]">
        <pre>{jsonStr}</pre>
      </div>
    </div>
  );
};
