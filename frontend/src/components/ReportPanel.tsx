import React from 'react';
import { Printer, FileText, ShieldCheck, FileJson } from 'lucide-react';
import type { AnalysisResult } from '../types';
import { getVerdictTheme } from '../utils/verdictTheme';

interface ReportPanelProps {
  result: AnalysisResult | null;
}

export const ReportPanel: React.FC<ReportPanelProps> = ({ result }) => {
  if (!result) {
    return (
      <div className="bg-navy-900 border border-navy-700 rounded-2xl p-12 text-center">
        <FileText className="h-12 w-12 mx-auto text-slate-600 mb-3" />
        <h3 className="text-base font-bold text-white mb-1">No Active Forensic Report</h3>
        <p className="text-xs text-slate-400 font-mono">
          Run an analysis on the Analyze page to generate and export an executive forensic audit report.
        </p>
      </div>
    );
  }

  const theme = getVerdictTheme(result.global_verdict);

  const downloadJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `forensic_report_${result.id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-6">
      {/* Action Bar */}
      <div className="flex items-center justify-between bg-navy-900/80 border border-navy-700 p-4 rounded-2xl">
        <div>
          <h3 className="text-sm font-bold text-white uppercase font-mono">Forensic Audit Export</h3>
          <p className="text-xs text-slate-400 font-mono">Target: {result.filename}</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={downloadJson}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-navy-800 hover:bg-navy-700 border border-navy-700 text-cyan-400 text-xs font-mono font-bold transition cursor-pointer"
          >
            <FileJson className="h-4 w-4" />
            <span>Export JSON</span>
          </button>
          <button
            onClick={handlePrint}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white text-xs font-mono font-bold shadow-md shadow-cyan-500/20 transition cursor-pointer"
          >
            <Printer className="h-4 w-4" />
            <span>Print / Save PDF</span>
          </button>
        </div>
      </div>

      {/* Printable Report Document Card */}
      <div className="bg-navy-950 border border-navy-800 rounded-2xl p-8 shadow-2xl font-mono text-xs space-y-6 print:bg-white print:text-black print:border-none">
        {/* Document Header */}
        <div className="border-b border-navy-800 pb-5 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <ShieldCheck className="h-5 w-5 text-cyan-400 print:text-black" />
              <h2 className="text-lg font-bold text-white print:text-black">
                EXPLAINABLE DEEPFAKE FORENSIC AUDIT DOSSIER
              </h2>
            </div>
            <p className="text-[11px] text-slate-400 print:text-slate-600">
              Multi-Agent Neural Architecture • University Research Verification Suite
            </p>
          </div>
          <div className="text-right text-[11px] text-slate-400 print:text-slate-600">
            <div>Report ID: {result.id.slice(0, 8)}</div>
            <div>Generated: {new Date(result.timestamp).toUTCString()}</div>
          </div>
        </div>

        {/* Section 1: Global Finding */}
        <div className="p-4 rounded-xl bg-navy-900 border border-navy-800 print:border-gray-300">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">
            1. Master Global Finding (WHAT)
          </span>
          <div className="flex items-center justify-between mt-2">
            <span className={`text-xl font-black ${theme.badgeText} print:text-black`}>
              {result.global_verdict}
            </span>
            <span className="text-sm font-bold text-white print:text-black">
              Calibrated Confidence: {result.confidence_display}
            </span>
          </div>
          <p className="text-slate-300 mt-1 print:text-slate-700">Arbitration Basis: {result.decision_basis}</p>
        </div>

        {/* Section 2: Multi-Agent Evidence */}
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold block mb-2">
            2. Multi-Agent Evidence Triangulation (WHY)
          </span>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-3 bg-navy-900 rounded-lg border border-navy-800 print:border-gray-200">
              <b className="text-cyan-400 print:text-black">Vision Classifier</b>
              <p className="text-slate-300 mt-1">Prediction: {result.primary_system.classifier.prediction}</p>
              <p className="text-slate-300">Fake Probability: {(result.primary_system.classifier.score * 100).toFixed(1)}%</p>
              <p className="text-slate-400 text-[10px] mt-1">{result.primary_system.classifier.reasoning}</p>
            </div>
            <div className="p-3 bg-navy-900 rounded-lg border border-navy-800 print:border-gray-200">
              <b className="text-cyan-400 print:text-black">Frequency Agent</b>
              <p className="text-slate-300 mt-1">Status: {result.primary_system.frequency.is_suspicious ? 'Grid Anomalies' : 'Natural'}</p>
              <p className="text-slate-300">Anomaly Score: {(result.primary_system.frequency.score * 100).toFixed(1)}%</p>
              <p className="text-slate-400 text-[10px] mt-1">{result.primary_system.frequency.reasoning}</p>
            </div>
            <div className="p-3 bg-navy-900 rounded-lg border border-navy-800 print:border-gray-200">
              <b className="text-cyan-400 print:text-black">Metadata Provenance</b>
              <p className="text-slate-300 mt-1">Flag: {result.primary_system.metadata.flag}</p>
              <p className="text-slate-300">Tool: {result.primary_system.metadata.disclosed_tool || 'None'}</p>
              <p className="text-slate-400 text-[10px] mt-1">{result.primary_system.metadata.reasoning}</p>
            </div>
          </div>
        </div>

        {/* Section 3: Regional Breakdown */}
        {result.secondary_system?.regions && result.secondary_system.regions.length > 0 && (
          <div>
            <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold block mb-2">
              3. Spatial Regional Breakdown (WHERE)
            </span>
            <table className="w-full text-left border border-navy-800 rounded-lg overflow-hidden">
              <thead className="bg-navy-900 text-slate-400">
                <tr>
                  <th className="p-2">Region</th>
                  <th className="p-2">Type</th>
                  <th className="p-2">Classifier</th>
                  <th className="p-2">Frequency</th>
                  <th className="p-2">Verdict</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-800 text-slate-300">
                {result.secondary_system.regions.map((reg) => (
                  <tr key={reg.region_id}>
                    <td className="p-2 font-bold">{reg.region_id}</td>
                    <td className="p-2">{reg.region_type}</td>
                    <td className="p-2">{(reg.fake_probability * 100).toFixed(1)}%</td>
                    <td className="p-2">{(reg.frequency_score * 100).toFixed(1)}%</td>
                    <td className="p-2 font-bold">{reg.verdict || reg.region_verdict}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Section 4: Narrative */}
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold block mb-2">
            4. Detailed Forensic Narrative
          </span>
          <div className="p-4 bg-navy-900 rounded-xl border border-navy-800 leading-relaxed text-slate-300 whitespace-pre-wrap">
            {result.final_explanation}
          </div>
        </div>
      </div>
    </div>
  );
};
