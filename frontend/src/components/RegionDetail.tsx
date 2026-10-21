import React from 'react';
import { X, Layers, Brain, Waves } from 'lucide-react';
import type { RegionData } from '../types';
import { getVerdictTheme } from '../utils/verdictTheme';

interface RegionDetailProps {
  region: RegionData | null;
  onClose: () => void;
}

export const RegionDetail: React.FC<RegionDetailProps> = ({ region, onClose }) => {
  if (!region) return null;

  const theme = getVerdictTheme(region.verdict || region.region_verdict);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-navy-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-navy-900 border border-navy-700 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden relative">
        {/* Header */}
        <div className="p-5 border-b border-navy-800 flex items-center justify-between bg-navy-950">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-xl ${theme.badgeBg} border ${theme.badgeBorder}`}>
              <Layers className={`h-5 w-5 ${theme.badgeText}`} />
            </div>
            <div>
              <span className="text-[10px] font-mono uppercase text-slate-400">Localized Inspection</span>
              <h3 className="text-lg font-bold text-white uppercase font-mono">{region.region_id}</h3>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg bg-navy-800 hover:bg-navy-700 text-slate-400 hover:text-white transition cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body Metrics */}
        <div className="p-6 space-y-6">
          {/* Verdict Banner */}
          <div className={`p-4 rounded-xl border ${theme.cardBorder} ${theme.badgeBg} flex items-center justify-between`}>
            <div>
              <span className="text-xs font-mono uppercase text-slate-300">Regional Verdict</span>
              <h4 className={`text-xl font-black ${theme.badgeText} mt-0.5`}>
                {region.verdict || region.region_verdict}
              </h4>
            </div>
            <div className="text-right">
              <span className="text-xs font-mono uppercase text-slate-300">Combined Suspicion</span>
              <p className="text-xl font-mono font-bold text-white mt-0.5">
                {(region.combined_score * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Metric Comparison Grid */}
          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-800">
              <div className="flex items-center gap-1.5 text-rose-400 mb-1">
                <Brain className="h-4 w-4" />
                <span className="font-semibold">Vision Classifier Fake %</span>
              </div>
              <p className="text-lg font-bold text-white">{(region.fake_probability * 100).toFixed(1)}%</p>
              <span className="text-[10px] text-slate-400">P(Real): {(region.real_probability * 100).toFixed(1)}%</span>
            </div>

            <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-800">
              <div className="flex items-center gap-1.5 text-cyan-400 mb-1">
                <Waves className="h-4 w-4" />
                <span className="font-semibold">Frequency Anomaly %</span>
              </div>
              <p className="text-lg font-bold text-white">{(region.frequency_score * 100).toFixed(1)}%</p>
              <span className="text-[10px] text-slate-400">
                Grid Artifacts: {region.frequency_anomaly ? 'Detected' : 'Clean'}
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-800">
              <span className="text-slate-400 block mb-1">Multi-Agent Agreement</span>
              <p className="text-sm font-bold text-white">{region.agreement_type || 'CONCORDANT'}</p>
            </div>

            <div className="p-3.5 rounded-xl bg-navy-950 border border-navy-800">
              <span className="text-slate-400 block mb-1">Biometric / Segment Type</span>
              <p className="text-sm font-bold text-white capitalize">{region.region_type}</p>
            </div>
          </div>

          {/* Why Section */}
          <div className="space-y-2">
            <h5 className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-bold">
              Why did the system reach this regional verdict?
            </h5>
            <div className="p-4 rounded-xl bg-navy-950/80 border border-navy-800 text-xs text-slate-300 leading-relaxed font-mono">
              {region.explanation ||
                `The Vision Classifier and Frequency Agent evaluated this ${region.region_type} patch independently. Signals show ${
                  region.combined_score >= 0.7 ? 'concentrated synthesis anomalies' : 'authentic natural properties'
                }.`}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-navy-950 border-t border-navy-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-navy-800 hover:bg-navy-700 text-white font-mono text-xs font-semibold transition cursor-pointer"
          >
            Close Inspection
          </button>
        </div>
      </div>
    </div>
  );
};
