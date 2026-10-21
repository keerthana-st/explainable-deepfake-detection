import React from 'react';
import { MessageSquareText, FileCheck } from 'lucide-react';

interface ExplanationCardProps {
  finalExplanation: string;
}

export const ExplanationCard: React.FC<ExplanationCardProps> = ({ finalExplanation }) => {
  return (
    <div className="bg-navy-900 border border-navy-700 rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-navy-800">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <MessageSquareText className="h-5 w-5" />
          </div>
          <div>
            <span className="text-[10px] font-mono uppercase text-slate-400">Forensic Audit Narrative</span>
            <h3 className="text-base font-bold text-white tracking-wide">
              Why did the system reach this result?
            </h3>
          </div>
        </div>
        <span className="text-xs font-mono text-cyan-400">Multi-Modal Triangulation</span>
      </div>

      <div className="bg-navy-950/80 border border-navy-800/80 rounded-xl p-5">
        <div className="text-xs font-mono text-slate-200 leading-relaxed whitespace-pre-wrap space-y-3">
          {finalExplanation ||
            'The Primary System evaluated the image globally using the 2D-DCT Frequency Agent and EfficientNet-B0 Classifier. The Secondary System segmented localized spatial patches to verify global consistency.'}
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between text-[11px] font-mono text-slate-400">
        <span className="flex items-center gap-1.5">
          <FileCheck className="h-3.5 w-3.5 text-emerald-400" />
          Audited across Spectral, Spatial, Metadata & Regional dimensions
        </span>
        <span>Deterministic Decision Trail</span>
      </div>
    </div>
  );
};
