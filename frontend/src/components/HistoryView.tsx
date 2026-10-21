import React from 'react';
import { 
  History as HistoryIcon, 
  RefreshCw, 
  ArrowRight
} from 'lucide-react';
import type { HistoryItem } from '../types';

interface HistoryViewProps {
  history: HistoryItem[];
  onSelectEntry: (id: string) => void;
  onRefresh: () => void;
  isRefreshing: boolean;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  onSelectEntry,
  onRefresh,
  isRefreshing,
}) => {
  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      
      {/* Header */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <HistoryIcon className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-slate-900">
              Analysis History
            </h2>
            <p className="text-xs text-slate-500">
              Verifiable forensic audit trail of all previous image scans
            </p>
          </div>
        </div>

        <button
          onClick={onRefresh}
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold transition cursor-pointer shadow-xs"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin text-blue-600' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs">
        {history.length === 0 ? (
          <div className="text-center py-12 text-slate-400 text-xs">
            No previous analysis records found. Upload an image to start auditing.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50/60">
                  <th className="py-3 px-4">Date & Time</th>
                  <th className="py-3 px-4">File Name</th>
                  <th className="py-3 px-4">Verdict</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4">Regions</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.map((item) => {
                  const isFake = item.global_verdict.includes('AI') || item.global_verdict.includes('FAKE');
                  const dateStr = new Date(item.timestamp).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  });

                  return (
                    <tr key={item.id} className="hover:bg-slate-50/80 transition">
                      <td className="py-3 px-4 text-slate-500 whitespace-nowrap">
                        {dateStr}
                      </td>
                      <td className="py-3 px-4 font-mono font-semibold text-slate-800">
                        {item.filename}
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-bold ${
                          isFake 
                            ? 'bg-rose-50 text-rose-700 border border-rose-200' 
                            : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        }`}>
                          {item.global_verdict}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-700">
                        {item.confidence_display || (item.confidence >= 1 ? `${item.confidence}%` : `${(item.confidence * 100).toFixed(1)}%`)}
                      </td>
                      <td className="py-3 px-4 text-slate-600">
                        {item.region_count} regions ({item.suspicious_region_count} suspicious)
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={() => onSelectEntry(item.id)}
                          className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-blue-50 hover:bg-blue-100 text-blue-700 text-xs font-semibold transition cursor-pointer"
                        >
                          <span>Inspect</span>
                          <ArrowRight className="h-3 w-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
};
