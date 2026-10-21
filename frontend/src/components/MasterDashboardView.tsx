import React, { useRef, useState } from 'react';
import { 
  UploadCloud, 
  CheckCircle2, 
  AlertTriangle, 
  Sparkles, 
  Waves, 
  Brain, 
  FileText, 
  Database, 
  Layers, 
  Play, 
  X,
  FileCheck
} from 'lucide-react';
import type { AnalysisResult, Preset } from '../types';

interface MasterDashboardViewProps {
  analysisResult: AnalysisResult | null;
  presets: Preset[];
  onStartAnalysis: (file?: File, presetId?: string) => void;
  isLoading: boolean;
}

export const MasterDashboardView: React.FC<MasterDashboardViewProps> = ({
  analysisResult,
  presets,
  onStartAnalysis,
  isLoading,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('portrait_real');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileMeta, setFileMeta] = useState({
    name: 'image.jpg',
    size: '1.2 MB',
    resolution: '1000 × 1000',
    type: 'Image',
    date: 'Oct 6, 2026, 03:15 PM',
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const img = new Image();
    img.onload = () => {
      setFileMeta({
        name: file.name,
        size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        resolution: `${img.width} × ${img.height}`,
        type: 'Image',
        date: new Date().toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
      });
    };
    img.src = url;
  };

  const handlePresetSelect = (presetId: string) => {
    setSelectedPresetId(presetId);
    setSelectedFile(null);
    setPreviewUrl(null);
    const preset = presets.find((p) => p.id === presetId);
    setFileMeta({
      name: preset ? `${preset.name.split('(')[0].trim().toLowerCase().replace(/\s+/g, '_')}.jpg` : 'image.jpg',
      size: '1.2 MB',
      resolution: '1000 × 1000',
      type: 'Image',
      date: 'Oct 6, 2026, 03:15 PM',
    });
  };

  const triggerAnalyze = () => {
    if (selectedFile) {
      onStartAnalysis(selectedFile, undefined);
    } else {
      onStartAnalysis(undefined, selectedPresetId || 'portrait_real');
    }
  };

  // Derive display values from analysis result or standard illustrative defaults
  const isFake = analysisResult 
    ? (analysisResult.global_verdict.includes('AI') || analysisResult.global_verdict.includes('FAKE'))
    : true; // Default illustrative matching screenshot
  
  const verdictText = analysisResult 
    ? (isFake ? 'FAKE' : 'REAL') 
    : 'FAKE';
  
  const confidenceScore = analysisResult 
    ? Math.round(analysisResult.confidence * 1000) / 10 
    : 93.4;

  const freqSuspicious = analysisResult 
    ? analysisResult.primary_system.frequency.is_suspicious 
    : true;

  const classifierPred = analysisResult 
    ? (analysisResult.primary_system.classifier.prediction === 'FAKE' ? 'Fake' : 'Real') 
    : 'Fake';

  const classifierScore = analysisResult 
    ? (analysisResult.primary_system.classifier.score * 100).toFixed(1) 
    : '93.4';

  const metadataFlag = analysisResult 
    ? analysisResult.primary_system.metadata.flag 
    : 'Inconclusive';

  const regionsData = analysisResult?.secondary_system?.regions && analysisResult.secondary_system.regions.length > 0 
    ? analysisResult.secondary_system.regions 
    : [
        {
          region_id: '1',
          region_type: 'Face',
          verdict: 'Fake',
          fake_probability: 0.961,
          explanation: 'Manipulation traces in facial features and texture.',
        },
        {
          region_id: '2',
          region_type: 'Background',
          verdict: 'Real',
          fake_probability: 0.127,
          explanation: 'No significant manipulation detected.',
        },
        {
          region_id: '3',
          region_type: 'Hair',
          verdict: 'Uncertain',
          fake_probability: 0.625,
          explanation: 'Inconsistent boundaries detected.',
        },
      ];

  const primaryImageDisplay = analysisResult?.visualizations?.gradcam ||
    analysisResult?.visualizations?.regions_overlay ||
    analysisResult?.preview_url ||
    previewUrl ||
    'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=600&q=80';

  const frequencyImageDisplay = analysisResult?.visualizations?.frequency_heatmap;

  return (
    <div className="space-y-6 pb-12 animate-in fade-in duration-300">
      {/* Page Title & Subtitle Matching Mockup */}
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Explainable Deepfake Detection
        </h2>
        <p className="text-sm text-slate-400 mt-0.5">
          Image authenticity analysis with transparent evidence
        </p>
      </div>

      {/* ======================================================== */}
      {/* TOP ROW: TWO CARDS SIDE BY SIDE                          */}
      {/* ======================================================== */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Card 1: Upload Image for Analysis (7 cols) */}
        <div className="lg:col-span-7 bg-[#0D152D] border border-[#19264D] rounded-2xl p-5 flex flex-col justify-between shadow-xl">
          <div>
            <div className="flex items-center space-x-2.5 mb-1">
              <div className="h-6 w-6 rounded-md bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
                <UploadCloud className="h-4 w-4" />
              </div>
              <h3 className="text-sm font-bold text-white tracking-wide">
                Upload Image for Analysis
              </h3>
            </div>
            <p className="text-xs text-slate-400 pl-8.5 mb-4">
              Upload an image to detect if it is real or fake
            </p>

            {/* Inner Upload / Preview Container */}
            <div className="bg-[#090F22] border border-[#1A284F] rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center space-x-3.5 w-full sm:w-auto">
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  className="w-16 h-16 rounded-lg overflow-hidden bg-[#131E3D] border border-[#223363] shrink-0 flex items-center justify-center cursor-pointer hover:border-blue-500 transition"
                >
                  <img
                    src={primaryImageDisplay}
                    alt="Target"
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-semibold text-white truncate max-w-[150px]">
                      {fileMeta.name}
                    </span>
                    <button 
                      onClick={() => {
                        setSelectedFile(null);
                        setPreviewUrl(null);
                      }}
                      className="text-slate-500 hover:text-white"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono">
                    <span>{fileMeta.size}</span>
                    <span>{fileMeta.resolution}</span>
                  </div>
                  <p className="text-[10px] text-slate-500">
                    Supported formats: JPG, JPEG, PNG (Max 10MB)
                  </p>
                </div>
              </div>

              {/* Hidden file input */}
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                accept=".jpg,.jpeg,.png"
                className="hidden"
              />

              {/* Analyze Image Action Button */}
              <button
                onClick={triggerAnalyze}
                disabled={isLoading}
                className="w-full sm:w-auto px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-semibold text-sm shadow-md shadow-blue-600/30 flex items-center justify-center space-x-2 transition cursor-pointer shrink-0"
              >
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>{isLoading ? 'Analyzing...' : 'Analyze Image'}</span>
              </button>
            </div>
          </div>

          {/* Quick Presets Bar */}
          <div className="pt-3 border-t border-[#14203B] flex items-center justify-between text-xs">
            <span className="text-slate-400 text-[11px]">Quick Benchmarks:</span>
            <div className="flex items-center space-x-2">
              {presets.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => handlePresetSelect(preset.id)}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-medium border transition cursor-pointer ${
                    selectedPresetId === preset.id
                      ? 'bg-blue-600/20 text-cyan-400 border-blue-500'
                      : 'bg-[#0A1024] text-slate-400 border-[#19264D] hover:text-white'
                  }`}
                >
                  {preset.name.split('(')[0].trim()}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Card 2: Global Verdict (5 cols) */}
        <div className="lg:col-span-5 bg-[#0D152D] border border-[#19264D] rounded-2xl p-5 flex flex-col justify-between shadow-xl">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <div className="h-5 w-5 rounded bg-purple-600/30 border border-purple-500/40 flex items-center justify-center text-purple-300">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-wide">
                  Global Verdict
                </h3>
              </div>

              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-purple-950/60 text-purple-300 border border-purple-700/50">
                Result from Primary + Secondary System
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
              {/* Verdict Banner & Confidence */}
              <div className="sm:col-span-6 space-y-2">
                <div
                  className={`w-full py-2.5 px-4 rounded-xl flex items-center justify-center space-x-2 font-black text-xl tracking-wider shadow-lg ${
                    isFake
                      ? 'bg-gradient-to-r from-red-600 to-rose-500 text-white shadow-rose-600/20'
                      : 'bg-gradient-to-r from-emerald-600 to-teal-500 text-white shadow-emerald-600/20'
                  }`}
                >
                  {isFake ? <AlertTriangle className="h-5 w-5" /> : <CheckCircle2 className="h-5 w-5" />}
                  <span>{verdictText}</span>
                </div>

                <div className="space-y-1 pt-1">
                  <span className="text-[11px] text-slate-400">Confidence Score (Illustrative)</span>
                  <div className="text-3xl font-extrabold text-white tracking-tight">
                    {confidenceScore}%
                  </div>
                  <div className="w-full h-2 rounded-full bg-[#15203D] overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        isFake ? 'bg-red-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${confidenceScore}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Metadata Info List */}
              <div className="sm:col-span-6 space-y-1.5 text-[11px] font-mono pl-0 sm:pl-3 sm:border-l border-[#1A284F]">
                <div className="flex justify-between text-slate-400">
                  <span>File Name</span>
                  <span className="text-slate-200 truncate max-w-[90px]">{fileMeta.name}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Media Type</span>
                  <span className="text-slate-200">{fileMeta.type}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Resolution</span>
                  <span className="text-slate-200">{fileMeta.resolution}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>File Size</span>
                  <span className="text-slate-200">{fileMeta.size}</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Analysis Date</span>
                  <span className="text-slate-200 text-[10px]">{fileMeta.date}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ======================================================== */}
      {/* SECTION 1: PRIMARY SYSTEM (Runs first — Core novelty)    */}
      {/* ======================================================== */}
      <div className="bg-[#0B1229] border border-purple-800/40 rounded-2xl p-5 shadow-xl space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="h-6 w-6 rounded-md bg-purple-600/20 border border-purple-500/40 flex items-center justify-center text-purple-300">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white tracking-wide">
                1. Primary System <span className="text-purple-400 text-xs font-normal">(Runs first — Core novelty)</span>
              </h3>
              <p className="text-xs text-slate-400">
                Three independent agents analyze the image and their results are compared for the global verdict.
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>Completed</span>
          </div>
        </div>

        {/* 3 Agents Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* 1. Frequency Agent */}
          <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex justify-between space-x-3">
            <div className="space-y-2 flex-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-blue-400">
                  <Waves className="h-4 w-4" />
                  <span className="text-xs font-bold text-white">Frequency Agent</span>
                </div>
                <span
                  className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                    freqSuspicious
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  }`}
                >
                  {freqSuspicious ? 'Suspicious' : 'Normal'}
                </span>
              </div>

              <ul className="text-[11px] text-slate-300 space-y-1">
                <li>• DCT & spectral analysis</li>
                <li>• Frequency artifacts detected</li>
                <li>• Possible generative traces</li>
              </ul>
            </div>

            {/* DCT Spectrum Thumbnail */}
            <div className="w-24 h-20 rounded-lg overflow-hidden bg-[#070D1B] border border-[#1C2C57] shrink-0 flex items-center justify-center">
              {frequencyImageDisplay ? (
                <img src={frequencyImageDisplay} alt="DCT" className="w-full h-full object-cover" />
              ) : (
                <div 
                  className="w-full h-full opacity-70"
                  style={{
                    backgroundImage: 'radial-gradient(circle, #38bdf8 1px, transparent 1px)',
                    backgroundSize: '6px 6px'
                  }}
                />
              )}
            </div>
          </div>

          {/* 2. Classifier Agent */}
          <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex justify-between space-x-3">
            <div className="space-y-2 flex-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-cyan-400">
                  <Brain className="h-4 w-4" />
                  <span className="text-xs font-bold text-white">Classifier Agent</span>
                </div>
                <span
                  className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                    classifierPred === 'Fake'
                      ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                      : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  }`}
                >
                  {classifierPred}
                </span>
              </div>

              <ul className="text-[11px] text-slate-300 space-y-1">
                <li>• Vision model (EfficientNet)</li>
                <li>• {classifierPred} ({classifierScore}%)</li>
                <li>• High confidence</li>
              </ul>
            </div>

            {/* Heatmap Thumbnail */}
            <div className="w-24 h-20 rounded-lg overflow-hidden bg-[#070D1B] border border-[#1C2C57] shrink-0 flex items-center justify-center">
              <img
                src={primaryImageDisplay}
                alt="Heatmap"
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* 3. Metadata Agent */}
          <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex justify-between space-x-3">
            <div className="space-y-2 flex-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-emerald-400">
                  <FileText className="h-4 w-4" />
                  <span className="text-xs font-bold text-white">Metadata Agent</span>
                </div>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                  {metadataFlag}
                </span>
              </div>

              <ul className="text-[11px] text-slate-300 space-y-1">
                <li>• EXIF, XMP, C2PA analysis</li>
                <li>• No authentic metadata found</li>
                <li>• Possible editing history</li>
              </ul>
            </div>

            {/* Metadata Summary Attributes Box */}
            <div className="p-2 rounded-lg bg-[#070D1B] border border-[#1C2C57] text-[10px] font-mono text-slate-400 space-y-0.5 shrink-0 justify-center flex flex-col">
              <div>Camera: <span className="text-slate-300">Not found</span></div>
              <div>Software: <span className="text-slate-300">Unknown</span></div>
              <div>Date: <span className="text-slate-300">N/A</span></div>
              <div>GPS: <span className="text-slate-300">Not found</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* ======================================================== */}
      {/* SECTION 2: SECONDARY SYSTEM (Runs after primary)         */}
      {/* ======================================================== */}
      <div className="bg-[#0B132B] border border-emerald-800/40 rounded-2xl p-5 shadow-xl space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="h-6 w-6 rounded-md bg-emerald-600/20 border border-emerald-500/40 flex items-center justify-center text-emerald-300">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white tracking-wide">
                2. Secondary System <span className="text-emerald-400 text-xs font-normal">(Runs after primary completes)</span>
              </h3>
              <p className="text-xs text-slate-400">
                Image is segmented into regions and analyzed to identify region-level manipulation patterns.
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>Completed</span>
          </div>
        </div>

        {/* 2 Panels Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
          {/* Panel 1: Region Segmentation & Heatmap (6 cols) */}
          <div className="lg:col-span-6 bg-[#0D152D] border border-[#19264D] rounded-xl p-4 space-y-3 flex flex-col justify-between">
            <h4 className="text-xs font-bold text-white tracking-wide">
              Region Segmentation & Heatmap
            </h4>

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center">
              {/* Segmented Image with Bounding Boxes */}
              <div className="sm:col-span-5 relative h-44 rounded-xl overflow-hidden bg-[#070D1B] border border-[#1C2C57]">
                <img
                  src={primaryImageDisplay}
                  alt="Segmentation"
                  className="w-full h-full object-cover"
                />

                {/* Overlaid Target Bounding Boxes Matching Mockup */}
                <div className="absolute inset-x-6 top-8 bottom-6 border-2 border-red-500/80 rounded">
                  <span className="absolute -top-3 -left-2 bg-red-600 text-white font-bold text-[10px] px-1 rounded">1</span>
                </div>
                <div className="absolute left-2 bottom-3 w-8 h-8 border-2 border-blue-500/80 rounded">
                  <span className="absolute -top-2.5 -left-1 bg-blue-600 text-white font-bold text-[9px] px-1 rounded">2</span>
                </div>
                <div className="absolute right-2 top-10 w-8 h-12 border-2 border-emerald-500/80 rounded">
                  <span className="absolute -top-2.5 -left-1 bg-emerald-600 text-white font-bold text-[9px] px-1 rounded">3</span>
                </div>
              </div>

              {/* 3 Region Explanations */}
              <div className="sm:col-span-7 space-y-2 text-xs">
                {/* Region 1 */}
                <div className="flex items-start space-x-2 bg-[#090F22] p-2 rounded-lg border border-[#1A284F]">
                  <span className="h-4 w-4 rounded bg-red-600 text-white font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                    1
                  </span>
                  <div>
                    <div className="flex items-center space-x-1.5">
                      <span className="font-bold text-white text-[11px]">Face Region</span>
                      <span className="bg-red-500/20 text-red-400 px-1.5 py-0.2 rounded text-[10px] font-bold">Fake</span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-0.5">Manipulation traces in facial features and texture.</p>
                  </div>
                </div>

                {/* Region 2 */}
                <div className="flex items-start space-x-2 bg-[#090F22] p-2 rounded-lg border border-[#1A284F]">
                  <span className="h-4 w-4 rounded bg-blue-600 text-white font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                    2
                  </span>
                  <div>
                    <div className="flex items-center space-x-1.5">
                      <span className="font-bold text-white text-[11px]">Background</span>
                      <span className="bg-emerald-500/20 text-emerald-400 px-1.5 py-0.2 rounded text-[10px] font-bold">Real</span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-0.5">No significant manipulation detected.</p>
                  </div>
                </div>

                {/* Region 3 */}
                <div className="flex items-start space-x-2 bg-[#090F22] p-2 rounded-lg border border-[#1A284F]">
                  <span className="h-4 w-4 rounded bg-emerald-600 text-white font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                    3
                  </span>
                  <div>
                    <div className="flex items-center space-x-1.5">
                      <span className="font-bold text-white text-[11px]">Hair Region</span>
                      <span className="bg-amber-500/20 text-amber-400 px-1.5 py-0.2 rounded text-[10px] font-bold">Uncertain</span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-0.5">Inconsistent boundaries detected.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Panel 2: Region-Level Results Table (6 cols) */}
          <div className="lg:col-span-6 bg-[#0D152D] border border-[#19264D] rounded-xl p-4 flex flex-col justify-between">
            <h4 className="text-xs font-bold text-white tracking-wide mb-2">
              Region-Level Results
            </h4>

            <div className="overflow-x-auto flex-1">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-[#090F22] text-slate-400 text-[10px] border-b border-[#1A284F]">
                  <tr>
                    <th className="py-2 px-3">Region</th>
                    <th className="py-2 px-3">Label</th>
                    <th className="py-2 px-3">Confidence (Illustrative)</th>
                    <th className="py-2 px-3">Explanation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#152245] text-slate-300 text-[11px]">
                  {regionsData.slice(0, 3).map((reg, idx) => {
                    const isRegFake = reg.verdict === 'Fake' || (reg.fake_probability > 0.5);
                    const isRegUncertain = reg.verdict === 'Uncertain' || (reg.fake_probability >= 0.4 && reg.fake_probability <= 0.6);
                    const pillColor = isRegFake
                      ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                      : isRegUncertain
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
                    const pillLabel = isRegFake ? 'Fake' : isRegUncertain ? 'Uncertain' : 'Real';
                    const conf = (reg.fake_probability > 0.5 ? reg.fake_probability : (1 - reg.fake_probability)) * 100;

                    return (
                      <tr key={idx} className="hover:bg-[#111B38]">
                        <td className="py-2.5 px-3 font-semibold text-white">{reg.region_type}</td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${pillColor}`}>
                            {pillLabel}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-bold text-white">{conf.toFixed(1)}%</td>
                        <td className="py-2.5 px-3 text-[10px] text-slate-400">{reg.explanation}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* ======================================================== */}
      {/* SECTION 3: EVIDENCE COMBINATION (Final decision)         */}
      {/* ======================================================== */}
      <div className="bg-[#091522] border border-cyan-800/40 rounded-2xl p-5 shadow-xl space-y-4">
        {/* Header */}
        <div className="flex items-center space-x-2">
          <div className="h-6 w-6 rounded-md bg-cyan-600/20 border border-cyan-500/40 flex items-center justify-center text-cyan-300">
            <Database className="h-4 w-4" />
          </div>
          <h3 className="text-sm font-bold text-white tracking-wide">
            3. Evidence Combination <span className="text-cyan-400 text-xs font-normal">(Final decision with explanations)</span>
          </h3>
        </div>

        {/* 3 Columns Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 items-stretch">
          {/* Column 1: Evidence Combination Bullets */}
          <div className="bg-[#0D182E] border border-[#19264D] rounded-xl p-4 space-y-3">
            <div className="flex items-center space-x-2 text-cyan-400">
              <Database className="h-4 w-4" />
              <h4 className="text-xs font-bold text-white">Evidence Combination</h4>
            </div>
            <ul className="text-xs text-slate-300 space-y-1.5 leading-relaxed">
              <li>• Combines region-level results</li>
              <li>• Identifies manipulation patterns</li>
              <li>• Checks provenance and agent agreement</li>
              <li>• Generates final explanation</li>
            </ul>
          </div>

          {/* Column 2: Region Contribution to Final Verdict (Bar Chart) */}
          <div className="bg-[#0D182E] border border-[#19264D] rounded-xl p-4 space-y-3">
            <h4 className="text-xs font-bold text-white">
              Region Contribution to Final Verdict
            </h4>

            <div className="space-y-2 text-xs font-mono pt-1">
              {/* Face Bar */}
              <div className="flex items-center space-x-2">
                <span className="w-16 text-slate-400 text-[11px] text-right">Face</span>
                <div className="flex-1 h-3 rounded-full bg-[#15203D] overflow-hidden">
                  <div className="h-full rounded-full bg-red-500" style={{ width: '60%' }} />
                </div>
                <span className="text-white text-[11px] font-bold w-8">60%</span>
              </div>

              {/* Background Bar */}
              <div className="flex items-center space-x-2">
                <span className="w-16 text-slate-400 text-[11px] text-right">Background</span>
                <div className="flex-1 h-3 rounded-full bg-[#15203D] overflow-hidden">
                  <div className="h-full rounded-full bg-blue-500" style={{ width: '25%' }} />
                </div>
                <span className="text-white text-[11px] font-bold w-8">25%</span>
              </div>

              {/* Hair Bar */}
              <div className="flex items-center space-x-2">
                <span className="w-16 text-slate-400 text-[11px] text-right">Hair</span>
                <div className="flex-1 h-3 rounded-full bg-[#15203D] overflow-hidden">
                  <div className="h-full rounded-full bg-amber-500" style={{ width: '15%' }} />
                </div>
                <span className="text-white text-[11px] font-bold w-8">15%</span>
              </div>

              {/* Axis ticks */}
              <div className="flex justify-between text-[9px] text-slate-500 pt-1 pl-18 pr-8">
                <span>0</span>
                <span>20</span>
                <span>40</span>
                <span>60</span>
                <span>80</span>
                <span>100</span>
              </div>
              <p className="text-[10px] text-slate-500 text-center">Contribution (%)</p>
            </div>
          </div>

          {/* Column 3: Final Explanation Narrative */}
          <div className="bg-[#0D182E] border border-[#19264D] rounded-xl p-4 space-y-2 flex flex-col justify-between">
            <div>
              <div className="flex items-center space-x-2 text-emerald-400 mb-2">
                <FileCheck className="h-4 w-4" />
                <h4 className="text-xs font-bold text-white">Final Explanation (Illustrative)</h4>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                The image shows clear signs of manipulation in facial regions, including inconsistencies in texture and frequency patterns. The classifier also indicates a high probability of being fake. No authentic metadata was found. Based on the combined evidence, the image is predicted as <b className={isFake ? 'text-red-400' : 'text-emerald-400'}>{verdictText}</b>.
              </p>
            </div>

            <div className="pt-2 text-[10px] text-slate-500 border-t border-[#162547] flex items-center justify-between">
              <span>Non-Override Deterministic Invariant</span>
              <span>Audit Ready</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
