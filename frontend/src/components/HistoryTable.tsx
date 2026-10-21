import React from 'react';
import { ExternalLink, Calendar, Layers, Image as ImageIcon } from 'lucide-react';
import type { HistoryItem } from '../types';
import { getVerdictTheme } from '../utils/verdictTheme';

interface HistoryTableProps {
  history: HistoryItem[];
  onSelectEntry: (id: string) => void;
}

export const HistoryTable: React.FC<HistoryTableProps> = ({ history, onSelectEntry }) => {
  if (!history || history.length === 0) {
    return (
      <div className="bg-navy-900 border border-navy-700 rounded-2xl p-12 text-center">
        <ImageIcon className="h-12 w-12 mx-auto text-slate-600 mb-3" />
        <h3 className="text-base font-bold text-white mb-1">No previous analyses</h3>
        <p className="text-xs text-slate-400 font-mono">
          Upload an image on the Analyze page to build your forensic audit history.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-navy-900 border border-navy-700 rounded-2xl overflow-hidden shadow-xl">
      <div className="p-5 bg-navy-950/80 border-b border-navy-800 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-white uppercase font-mono tracking-wider">
            Forensic Audit Log
          </h3>
          <p className="text-xs text-slate-400 mt-0.5 font-mono">Historical multi-agent investigations</p>
        </div>
        <span className="text-xs font-mono text-cyan-400 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20">
          {history.length} Records
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs font-mono">
          <thead className="bg-navy-950/60 text-slate-400 uppercase text-[11px] border-b border-navy-800">
            <tr>
              <th className="py-3.5 px-4">Timestamp</th>
              <th className="py-3.5 px-4">Filename</th>
              <th className="py-3.5 px-4">Global Verdict</th>
              <th className="py-3.5 px-4 text-right">Confidence</th>
              <th className="py-3.5 px-4 text-center">Regions</th>
              <th className="py-3.5 px-4 text-center">Consistency</th>
              <th className="py-3.5 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-800/80 text-slate-200">
            {history.map((item) => {
              const theme = getVerdictTheme(item.global_verdict);
              const formattedDate = new Date(item.timestamp).toLocaleString();

              return (
                <tr
                  key={item.id}
                  className="hover:bg-navy-800/60 cursor-pointer transition-colors"
                  onClick={() => onSelectEntry(item.id)}
                >
                  <td className="py-3.5 px-4 text-slate-400 flex items-center gap-2">
                    <Calendar className="h-3.5 w-3.5 text-slate-500" />
                    <span>{formattedDate}</span>
                  </td>
                  <td className="py-3.5 px-4 font-bold text-white truncate max-w-[180px]">
                    {item.filename}
                  </td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${theme.badgeBg} ${theme.badgeText} ${theme.badgeBorder}`}
                    >
                      {item.global_verdict}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-right font-bold text-white">
                    {item.confidence_display || `${Math.round(item.confidence * 100)}%`}
                  </td>
                  <td className="py-3.5 px-4 text-center">
                    <span className="inline-flex items-center gap-1 text-slate-300">
                      <Layers className="h-3 w-3 text-cyan-400" />
                      {item.region_count}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-center">
                    <span className="text-[10px] font-semibold text-slate-300">
                      {item.consistency_status}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectEntry(item.id);
                      }}
                      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-navy-800 hover:bg-blue-600/30 text-cyan-400 hover:text-cyan-300 font-bold border border-navy-700 transition"
                    >
                      <span>View</span>
                      <ExternalLink className="h-3 w-3" />
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
