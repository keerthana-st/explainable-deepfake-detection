import React from 'react';
import { ChevronRight } from 'lucide-react';
import type { RegionData } from '../types';
import { getVerdictTheme } from '../utils/verdictTheme';

interface RegionTableProps {
  regions: RegionData[];
  onSelectRegion: (region: RegionData) => void;
  selectedRegionId?: string | null;
}

export const RegionTable: React.FC<RegionTableProps> = ({
  regions,
  onSelectRegion,
  selectedRegionId,
}) => {
  if (!regions || regions.length === 0) {
    return (
      <div className="p-8 text-center bg-navy-900/60 rounded-xl border border-navy-800 text-slate-500 font-mono text-xs">
        No sub-regions detected or triggered for this image.
      </div>
    );
  }

  return (
    <div className="bg-navy-900 border border-navy-700 rounded-2xl overflow-hidden shadow-lg">
      <div className="p-4 bg-navy-950/80 border-b border-navy-800 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide uppercase font-mono">
            Secondary System: Regional Analysis Breakdown
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Independent multi-agent evaluation per localized spatial segment
          </p>
        </div>
        <span className="text-xs font-mono text-cyan-400 px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20">
          {regions.length} Segments Analyzed
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-navy-950/60 text-slate-400 uppercase text-[11px] border-b border-navy-800">
            <tr>
              <th className="py-3 px-4">Region ID</th>
              <th className="py-3 px-4">Type</th>
              <th className="py-3 px-4 text-right">Classifier Fake %</th>
              <th className="py-3 px-4 text-right">Frequency Score %</th>
              <th className="py-3 px-4 text-center">Agreement</th>
              <th className="py-3 px-4">Verdict</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-800/80 text-slate-200">
            {regions.map((region) => {
              const theme = getVerdictTheme(region.verdict || region.region_verdict);
              const isSelected = selectedRegionId === region.region_id;

              return (
                <tr
                  key={region.region_id}
                  onClick={() => onSelectRegion(region)}
                  className={`hover:bg-navy-800/60 cursor-pointer transition-colors ${
                    isSelected ? 'bg-blue-600/15 border-l-2 border-cyan-400' : ''
                  }`}
                >
                  <td className="py-3.5 px-4 font-bold text-white flex items-center gap-2">
                    <span className="text-cyan-400">#</span>
                    {region.region_id}
                  </td>
                  <td className="py-3.5 px-4 text-slate-300 capitalize">{region.region_type}</td>
                  <td className="py-3.5 px-4 text-right font-semibold text-rose-400">
                    {(region.fake_probability * 100).toFixed(1)}%
                  </td>
                  <td className="py-3.5 px-4 text-right font-semibold text-slate-300">
                    {(region.frequency_score * 100).toFixed(1)}%
                  </td>
                  <td className="py-3.5 px-4 text-center">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold border ${
                        region.agent_agreement
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                      }`}
                    >
                      {region.agreement_type || (region.agent_agreement ? 'CONCORDANT' : 'DISAGREEMENT')}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold border ${theme.badgeBg} ${theme.badgeText} ${theme.badgeBorder}`}
                    >
                      {region.verdict || region.region_verdict}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 font-semibold"
                    >
                      <span>Details</span>
                      <ChevronRight className="h-3.5 w-3.5" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
