import React from 'react';
import { Waves, Eye, FileSearch, GitMerge } from 'lucide-react';
import type { PrimarySystemPayload } from '../types';

interface AgentCardProps {
  primary: PrimarySystemPayload;
}

export const AgentCard: React.FC<AgentCardProps> = ({ primary }) => {
  const { frequency, classifier, metadata, disagreement } = primary;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-mono font-bold tracking-wider text-slate-300 uppercase flex items-center gap-2">
          <span>Primary System Multi-Agent Evidence</span>
        </h3>
        <span className="text-xs font-mono text-cyan-400">Independent Expert Verification</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1. Frequency Agent */}
        <div className="bg-navy-900/90 border border-navy-700 rounded-xl p-5 space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-cyan-400">
                <Waves className="h-5 w-5" />
                <h4 className="text-sm font-bold text-white">Frequency Agent</h4>
              </div>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                  frequency.is_suspicious
                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                    : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                }`}
              >
                {frequency.is_suspicious ? 'GRID ANOMALIES' : 'NATURAL DECAY'}
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Anomaly Score:</span>
                <span className="font-bold text-white">{(frequency.score * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Spectral Energy:</span>
                <span className="text-slate-200">{frequency.raw_energy.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Domain:</span>
                <span className="text-slate-200">2D-DCT 8×8 Blocks</span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 line-clamp-3 bg-navy-950/60 p-2.5 rounded-lg border border-navy-800">
            {frequency.reasoning}
          </p>
        </div>

        {/* 2. Classifier Agent */}
        <div className="bg-navy-900/90 border border-navy-700 rounded-xl p-5 space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-cyan-400">
                <Eye className="h-5 w-5" />
                <h4 className="text-sm font-bold text-white">Vision Classifier</h4>
              </div>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                  classifier.prediction === 'FAKE'
                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                    : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                }`}
              >
                {classifier.prediction}
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>P(Fake):</span>
                <span className="font-bold text-rose-400">{(classifier.score * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>P(Real):</span>
                <span className="font-bold text-emerald-400">
                  {((classifier.probabilities?.REAL ?? (1 - classifier.score)) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Disparity Ratio:</span>
                <span className="text-slate-200">{classifier.disparity_ratio.toFixed(2)}×</span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 line-clamp-3 bg-navy-950/60 p-2.5 rounded-lg border border-navy-800">
            {classifier.reasoning}
          </p>
        </div>

        {/* 3. Metadata Agent */}
        <div className="bg-navy-900/90 border border-navy-700 rounded-xl p-5 space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-cyan-400">
                <FileSearch className="h-5 w-5" />
                <h4 className="text-sm font-bold text-white">Metadata Agent</h4>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-navy-800 text-slate-300 border border-navy-700">
                {metadata.flag}
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>C2PA Signed:</span>
                <span className={metadata.c2pa_present ? 'text-emerald-400 font-bold' : 'text-slate-400'}>
                  {metadata.c2pa_present ? 'Manifest Found' : 'None'}
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Disclosed Tool:</span>
                <span className="text-slate-200 truncate max-w-[110px]" title={metadata.disclosed_tool || 'None'}>
                  {metadata.disclosed_tool || 'None'}
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>DQT Profile:</span>
                <span className="text-slate-200 truncate max-w-[110px]" title={metadata.dqt_profile || 'Default/Stripped'}>
                  {metadata.dqt_profile || 'Default/Stripped'}
                </span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 line-clamp-3 bg-navy-950/60 p-2.5 rounded-lg border border-navy-800">
            {metadata.reasoning}
          </p>
        </div>

        {/* 4. Disagreement Layer */}
        <div className="bg-navy-900/90 border border-navy-700 rounded-xl p-5 space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-cyan-400">
                <GitMerge className="h-5 w-5" />
                <h4 className="text-sm font-bold text-white">Disagreement Layer</h4>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                ZERO-FUSION
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Conflict Type:</span>
                <span className="font-bold text-white truncate max-w-[120px]" title={disagreement.conflict_type}>
                  {disagreement.conflict_type}
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Arbitration Mode:</span>
                <span className="text-slate-200">State Matrix</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Deciding Badge:</span>
                <span className="text-cyan-300 font-semibold">{primary.badge}</span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 line-clamp-3 bg-navy-950/60 p-2.5 rounded-lg border border-navy-800">
            {disagreement.arbitration_reasoning}
          </p>
        </div>
      </div>
    </div>
  );
};
