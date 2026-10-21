import React from 'react';
import { AlertTriangle, HelpCircle, CheckCircle2 } from 'lucide-react';
import type { RegionalConsistency } from '../types';

interface ConsistencyCardProps {
  consistency?: RegionalConsistency | null;
}

export const ConsistencyCard: React.FC<ConsistencyCardProps> = ({ consistency }) => {
  if (!consistency) {
    return null;
  }

  const { status, explanation, suspicious_regions, authentic_regions, requires_review, warning } = consistency;

  const isSupported = status === 'SUPPORTED';
  const isContradicted = status === 'CONTRADICTED';

  const getStatusBadge = () => {
    if (isSupported) {
      return (
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-xs font-bold">
          <CheckCircle2 className="h-4 w-4" />
          <span>CONSISTENCY: SUPPORTED</span>
        </div>
      );
    }
    if (isContradicted) {
      return (
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/30 text-rose-400 font-mono text-xs font-bold animate-pulse">
          <AlertTriangle className="h-4 w-4" />
          <span>CONSISTENCY: CONTRADICTED (AUDIT ALERT)</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 font-mono text-xs font-bold">
        <HelpCircle className="h-4 w-4" />
        <span>CONSISTENCY: {status}</span>
      </div>
    );
  };

  return (
    <div
      className={`rounded-2xl p-6 border transition-all ${
        isContradicted
          ? 'bg-rose-950/20 border-rose-500/40 shadow-lg shadow-rose-500/10'
          : isSupported
          ? 'bg-emerald-950/20 border-emerald-500/30'
          : 'bg-navy-900 border-navy-700'
      }`}
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 font-semibold">
            Global–Regional Forensic Consistency
          </span>
          <h3 className="text-lg font-bold text-white mt-0.5">
            Cross-Level Evidence Concordance
          </h3>
        </div>
        <div>{getStatusBadge()}</div>
      </div>

      {warning && (
        <div className="mb-4 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-mono flex items-start gap-2.5">
          <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <b className="uppercase tracking-wider">Audit Alert:</b> {warning}
          </div>
        </div>
      )}

      <p className="text-sm text-slate-300 leading-relaxed font-mono mb-4 bg-navy-950/60 p-4 rounded-xl border border-navy-800">
        {explanation}
      </p>

      {/* Region Attribution Breakdown */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
        <div className="p-3 rounded-xl bg-navy-950 border border-navy-800">
          <span className="text-slate-400 block mb-1">Supporting Suspicious Regions:</span>
          {suspicious_regions && suspicious_regions.length > 0 ? (
            <div className="flex flex-wrap gap-1.5 mt-1">
              {suspicious_regions.map((reg) => (
                <span key={reg} className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30 font-bold">
                  {reg}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-slate-500 italic">None detected</span>
          )}
        </div>

        <div className="p-3 rounded-xl bg-navy-950 border border-navy-800">
          <span className="text-slate-400 block mb-1">Normal / Authentic Regions:</span>
          {authentic_regions && authentic_regions.length > 0 ? (
            <div className="flex flex-wrap gap-1.5 mt-1">
              {authentic_regions.map((reg) => (
                <span key={reg} className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold">
                  {reg}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-slate-500 italic">None detected</span>
          )}
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-navy-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
        <span>🔒 Non-Override Invariant: Global verdict remains preserved as the master truth.</span>
        <span>Review Flag: {requires_review ? 'YES (AUDIT REQUIRED)' : 'CLEAN'}</span>
      </div>
    </div>
  );
};
