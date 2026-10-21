import React, { useState } from 'react';
import { ArrowLeft, ChevronDown, ChevronUp, Layers } from 'lucide-react';
import type { AnalysisResult, RegionData } from '../types';
import { RegionTable } from './RegionTable';
import { RegionDetail } from './RegionDetail';
import { ReportPanel } from './ReportPanel';

interface SimpleResultViewProps {
  result: AnalysisResult;
  onBack: () => void;
}

export const SimpleResultView: React.FC<SimpleResultViewProps> = ({ result, onBack }) => {
  const [activeImageTab, setActiveImageTab] = useState<'original' | 'regions' | 'heatmap' | 'gradcam'>('original');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState<RegionData | null>(null);

  // Normalize verdict for display matching Screenshot 3
  const isFake = result.global_verdict.includes('AI-GENERATED') || result.global_verdict.includes('FAKE');
  const isEdited = result.global_verdict.includes('AI-EDITED') || result.global_verdict.includes('INPAINT');
  const isReal = result.global_verdict.includes('REAL') || result.global_verdict.includes('AUTHENTIC');

  const verdictLabel = isFake ? 'Fake' : isReal ? 'Real' : isEdited ? 'AI-Edited' : 'Uncertain';
  const verdictPillColor = isFake
    ? 'bg-[#E15252] text-white'
    : isReal
    ? 'bg-[#10B981] text-white'
    : isEdited
    ? 'bg-[#F97316] text-white'
    : 'bg-[#EAB308] text-white';

  const progressColor = isFake
    ? 'bg-[#E15252]'
    : isReal
    ? 'bg-[#10B981]'
    : isEdited
    ? 'bg-[#F97316]'
    : 'bg-[#EAB308]';

  const confidencePct = Math.round(result.confidence * 1000) / 10;

  // Visual URLs
  const originalUrl = result.preview_url;
  const regionsUrl = result.visualizations?.regions_overlay;
  const heatmapUrl = result.visualizations?.frequency_heatmap;
  const gradcamUrl = result.visualizations?.gradcam;

  const currentDisplayUrl = () => {
    switch (activeImageTab) {
      case 'regions':
        return regionsUrl || originalUrl;
      case 'heatmap':
        return heatmapUrl || originalUrl;
      case 'gradcam':
        return gradcamUrl || originalUrl;
      case 'original':
      default:
        return originalUrl;
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-in fade-in duration-300">
      {/* Top Header Matching Screenshot 3 */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Image Analysis Result
        </h2>
        <button
          onClick={onBack}
          className="flex items-center space-x-1.5 text-sm font-medium text-blue-400 hover:text-blue-300 transition cursor-pointer"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Upload</span>
        </button>
      </div>

      {/* Main 2-Column Section Matching Screenshot 3 */}
      <div className="bg-[#0D152D] border border-[#19264D] rounded-2xl p-6 shadow-xl grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
        {/* Left: Analyzed Image */}
        <div className="space-y-3">
          <div className="relative w-full h-[320px] rounded-xl overflow-hidden bg-[#0A1024] border border-[#1C2C57] flex items-center justify-center p-2">
            {currentDisplayUrl() ? (
              <img
                src={currentDisplayUrl()}
                alt={activeImageTab}
                className="max-h-full max-w-full object-contain rounded-lg"
              />
            ) : (
              <div className="text-slate-500 text-xs font-mono">Image preview unavailable</div>
            )}

            {/* Quick mini-toggle on hover or at bottom */}
            <div className="absolute top-3 right-3 flex items-center gap-1 bg-[#090F22]/90 backdrop-blur-md p-1 rounded-lg border border-[#1C2C57] text-[11px]">
              <button
                onClick={() => setActiveImageTab('original')}
                className={`px-2 py-0.5 rounded transition ${
                  activeImageTab === 'original' ? 'bg-blue-600 text-white font-medium' : 'text-slate-400 hover:text-white'
                }`}
              >
                Original
              </button>
              {regionsUrl && (
                <button
                  onClick={() => setActiveImageTab('regions')}
                  className={`px-2 py-0.5 rounded transition ${
                    activeImageTab === 'regions' ? 'bg-blue-600 text-white font-medium' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Regions
                </button>
              )}
            </div>
          </div>

          <p className="text-xs text-center text-slate-400 capitalize font-medium">
            {activeImageTab === 'original' ? 'Original Image' : `${activeImageTab} View`}
          </p>
        </div>

        {/* Right: Verdict, Confidence & Explanation Matching Screenshot 3 */}
        <div className="space-y-6">
          {/* Verdict Pill */}
          <div>
            <span
              className={`inline-block px-8 py-2 rounded-full text-lg font-bold tracking-wide shadow-md ${verdictPillColor}`}
            >
              {verdictLabel}
            </span>
          </div>

          {/* Confidence Score */}
          <div className="space-y-2">
            <span className="text-xs font-medium text-slate-400 tracking-wider">
              Confidence Score
            </span>
            <div className="text-4xl font-extrabold text-white tracking-tight">
              {confidencePct}%
            </div>

            {/* Progress Bar */}
            <div className="w-full h-2.5 rounded-full bg-[#15203D] overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${progressColor}`}
                style={{ width: `${Math.min(100, Math.max(5, confidencePct))}%` }}
              />
            </div>
          </div>

          {/* Explanation Box */}
          <div className="bg-[#090F22] border border-[#1C2C57] rounded-xl p-4 space-y-2">
            <h4 className="text-xs font-bold text-slate-200 tracking-wide uppercase">
              Explanation
            </h4>
            <p className="text-xs text-slate-300 leading-relaxed">
              {result.final_explanation ||
                `The image shows ${
                  isFake ? 'signs of manipulation' : 'authentic camera properties'
                } based on frequency analysis, inconsistencies in facial features and metadata analysis.`}
            </p>
          </div>
        </div>
      </div>

      {/* Bottom Row of 4 Agent Cards Matching Screenshot 3 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Frequency Analysis */}
        <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex flex-col justify-between space-y-3">
          <span className="text-xs font-semibold text-slate-300">
            Frequency Analysis
          </span>

          <div className="h-28 rounded-lg overflow-hidden bg-[#0A1024] border border-[#1C2C57] flex items-center justify-center">
            {heatmapUrl ? (
              <img src={heatmapUrl} alt="Frequency" className="h-full w-full object-cover" />
            ) : (
              <div className="text-[11px] text-slate-500 font-mono text-center p-2">
                2D-DCT Anomaly: {(result.primary_system.frequency.score * 100).toFixed(1)}%
              </div>
            )}
          </div>

          <div className="text-[11px] text-slate-400">
            Status: <b className={result.primary_system.frequency.is_suspicious ? 'text-rose-400' : 'text-emerald-400'}>
              {result.primary_system.frequency.is_suspicious ? 'Grid Anomaly' : 'Natural Decay'}
            </b>
          </div>
        </div>

        {/* Card 2: Classifier Result */}
        <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex flex-col justify-between space-y-3">
          <span className="text-xs font-semibold text-slate-300">
            Classifier Result
          </span>

          <div className="h-28 rounded-lg bg-[#0A1024] border border-[#1C2C57] flex flex-col items-center justify-center p-2 text-center">
            <span
              className={`text-xl font-black ${
                result.primary_system.classifier.prediction === 'FAKE' ? 'text-[#E15252]' : 'text-[#10B981]'
              }`}
            >
              {result.primary_system.classifier.prediction === 'FAKE' ? 'Fake' : 'Real'}
            </span>
            <span className="text-xs font-mono text-slate-400 mt-1">
              ({(result.primary_system.classifier.score * 100).toFixed(1)}%)
            </span>
          </div>

          <div className="text-[11px] text-slate-400 truncate">
            Model: <span className="text-slate-300">EfficientNet-B0</span>
          </div>
        </div>

        {/* Card 3: Metadata Analysis */}
        <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex flex-col justify-between space-y-3">
          <span className="text-xs font-semibold text-slate-300">
            Metadata Analysis
          </span>

          <div className="h-28 rounded-lg bg-[#0A1024] border border-[#1C2C57] flex flex-col items-center justify-center p-3 text-center">
            <p className="text-xs text-slate-300 leading-relaxed font-medium">
              {result.primary_system.metadata.c2pa_present
                ? 'C2PA Manifest Verified'
                : result.primary_system.metadata.disclosed_tool
                ? `Tool: ${result.primary_system.metadata.disclosed_tool}`
                : 'No authentic metadata found'}
            </p>
          </div>

          <div className="text-[11px] text-slate-400 truncate">
            Flag: <span className="text-slate-300">{result.primary_system.metadata.flag}</span>
          </div>
        </div>

        {/* Card 4: Grad-CAM Heatmap */}
        <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex flex-col justify-between space-y-3">
          <span className="text-xs font-semibold text-slate-300">
            Grad-CAM Heatmap
          </span>

          <div className="h-28 rounded-lg overflow-hidden bg-[#0A1024] border border-[#1C2C57] flex items-center justify-center">
            {gradcamUrl ? (
              <img src={gradcamUrl} alt="Grad-CAM" className="h-full w-full object-cover" />
            ) : (
              <div className="text-[11px] text-slate-500 font-mono text-center p-2">
                Attention Map Active
              </div>
            )}
          </div>

          <div className="text-[11px] text-slate-400">
            Focus: <span className="text-slate-300">Attribution Overlay</span>
          </div>
        </div>
      </div>

      {/* Expandable Advanced Research & Regional Forensics */}
      <div className="pt-2">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full py-3 px-4 rounded-xl bg-[#090F22] hover:bg-[#0D152D] border border-[#1C2C57] text-xs font-medium text-slate-300 flex items-center justify-between transition cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue-400" />
            <span>Detailed Region-Level Analysis & Forensic Dossier ({result.region_count} Segments)</span>
          </div>
          {showAdvanced ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
        </button>

        {showAdvanced && (
          <div className="mt-4 space-y-6 animate-in fade-in duration-200">
            <RegionTable
              regions={result.secondary_system?.regions || []}
              onSelectRegion={setSelectedRegion}
              selectedRegionId={selectedRegion?.region_id}
            />

            <ReportPanel result={result} />
          </div>
        )}
      </div>

      {/* Region Detail Modal if opened */}
      <RegionDetail region={selectedRegion} onClose={() => setSelectedRegion(null)} />
    </div>
  );
};
