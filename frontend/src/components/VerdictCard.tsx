import React from 'react';
import { ShieldCheck, AlertTriangle, HelpCircle, CheckCircle2 } from 'lucide-react';
import { getVerdictTheme } from '../utils/verdictTheme';

interface VerdictCardProps {
  verdict: string;
  confidence: number;
  confidenceDisplay: string;
  decisionBasis: string;
  badge: string;
  latencySeconds?: number;
}

export const VerdictCard: React.FC<VerdictCardProps> = ({
  verdict,
  confidence,
  confidenceDisplay,
  decisionBasis,
  badge,
  latencySeconds,
}) => {
  const theme = getVerdictTheme(verdict);

  const renderIcon = () => {
    switch (theme.category) {
      case 'REAL':
        return <CheckCircle2 className="h-10 w-10 text-emerald-400" />;
      case 'AI-GENERATED':
        return <AlertTriangle className="h-10 w-10 text-rose-400" />;
      case 'AI-EDITED':
        return <ShieldCheck className="h-10 w-10 text-orange-400" />;
      default:
        return <HelpCircle className="h-10 w-10 text-amber-400" />;
    }
  };

  return (
    <div
      className={`rounded-2xl p-6 bg-gradient-to-r ${theme.gradient} border ${theme.cardBorder} ${theme.glow} transition-all duration-300 relative overflow-hidden`}
    >
      {/* Decorative background grid pattern */}
      <div className="absolute inset-0 opacity-10 forensic-grid pointer-events-none" />

      <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        {/* Left: Verdict info */}
        <div className="flex items-center space-x-5">
          <div className={`p-3.5 rounded-2xl ${theme.badgeBg} border ${theme.badgeBorder}`}>
            {renderIcon()}
          </div>
          <div>
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                Primary System Global Finding
              </span>
              <span className={`text-[11px] font-mono px-2 py-0.5 rounded-full border ${theme.badgeBg} ${theme.badgeText} ${theme.badgeBorder}`}>
                {badge}
              </span>
            </div>
            <h2 className="text-3xl font-extrabold tracking-tight text-white mt-1">
              {theme.label}
            </h2>
            <p className="text-sm text-slate-300 mt-1 font-medium max-w-xl">
              Basis: <span className="text-white font-semibold">{decisionBasis}</span>
            </p>
          </div>
        </div>

        {/* Right: Confidence Metric Card */}
        <div className="flex items-center gap-6 self-stretch md:self-auto justify-end border-t md:border-t-0 md:border-l border-white/10 pt-4 md:pt-0 md:pl-8">
          <div>
            <div className="text-xs font-mono text-slate-400 uppercase tracking-wider text-right">
              Calibrated Confidence
            </div>
            <div className="text-3xl font-mono font-bold text-white text-right mt-0.5">
              {confidenceDisplay}
            </div>
            {latencySeconds && (
              <div className="text-[11px] font-mono text-slate-400 text-right mt-1">
                Latency: <span className="text-cyan-400">{latencySeconds}s</span>
              </div>
            )}
          </div>

          {/* Mini gauge circle */}
          <div className="relative w-16 h-16 flex items-center justify-center shrink-0">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path
                className="text-navy-950"
                strokeWidth="3.5"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                strokeWidth="3.5"
                strokeDasharray={`${Math.min(100, Math.max(0, confidence * 100))}, 100`}
                stroke={theme.accentHex}
                strokeLinecap="round"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <span className="absolute text-xs font-mono font-bold text-white">
              {Math.round(confidence * 100)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
