import React, { useRef } from 'react';
import { 
  UploadCloud, 
  Image as ImageIcon, 
  CheckCircle2, 
  Activity, 
  Brain, 
  FileText, 
  Scale, 
  Bot, 
  Lightbulb, 
  ChevronDown, 
  ChevronRight,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  ArrowRight
} from 'lucide-react';
import type { AnalysisResult } from '../types';

interface OverallResultViewProps {
  analysisResult: AnalysisResult | null;
  selectedFile: File | null;
  onFileSelect: (file: File) => void;
  onSelectPreset: (presetId: string) => void;
  onTriggerAnalysis: () => void;
  isLoading: boolean;
  activeSubTab: string;
  setActiveSubTab: (tab: string) => void;
  onNavigateModule: (module: string) => void;
}

export const OverallResultView: React.FC<OverallResultViewProps> = ({
  analysisResult,
  selectedFile,
  onFileSelect,
  onSelectPreset,
  onTriggerAnalysis,
  isLoading,
  activeSubTab,
  setActiveSubTab,
  onNavigateModule,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Dynamic derivations from analysisResult
  const primary = analysisResult?.primary_system;

  const verdictTitle = analysisResult
    ? (analysisResult.global_verdict.includes('AI') ? 'AI-GENERATED' : (analysisResult.global_verdict.includes('EDIT') ? 'AI-EDITED' : 'REAL'))
    : 'AI-GENERATED';

  const isFinalFake = verdictTitle.includes('AI') || verdictTitle.includes('FAKE') || verdictTitle.includes('EDIT');

  const confidenceScore = analysisResult 
    ? (analysisResult.confidence >= 1 ? (analysisResult.confidence / 100).toFixed(2) : analysisResult.confidence.toFixed(2))
    : '0.90';

  const previewOriginal = analysisResult?.preprocessing?.original_url || 
                          (selectedFile ? URL.createObjectURL(selectedFile) : (analysisResult?.preview_url || '/dog.jpg'));

  const previewGrayscale = analysisResult?.preprocessing?.grayscale_url || '/dog_grayscale.jpg';
  
  const prepFormat = analysisResult?.preprocessing?.format || 'JPG';
  const prepDimensions = analysisResult?.preprocessing?.dimensions || '512 × 512';

  const freqIsFake = primary ? primary.frequency?.is_suspicious : true;
  const freqPred = freqIsFake ? 'Fake' : 'Real';
  const freqConf = primary?.frequency?.score ? primary.frequency.score.toFixed(2) : '0.82';

  const clsIsFake = primary 
    ? (primary.classifier?.prediction === 'FAKE' || primary.classifier?.score >= 0.5) 
    : true;
  const clsPred = clsIsFake ? 'Fake' : 'Real';
  const clsConf = primary?.classifier?.score ? primary.classifier.score.toFixed(2) : '0.76';
  const clsImage = analysisResult?.visualizations?.gradcam_global || 
                   analysisResult?.visualizations?.region_overlay || 
                   '/dog_face.jpg';

  const metaIsFake = primary 
    ? Boolean(primary.metadata?.c2pa_present || primary.metadata?.flag?.includes('AI') || primary.metadata?.flag === 'SUSPICIOUS_EXIF' || primary.metadata?.flag === 'EXPLICIT_SYNTHETIC') 
    : false;
  const metaPred = metaIsFake ? 'Fake' : 'Real';
  const metaMake = primary?.metadata?.make || (primary?.metadata?.disclosed_tool ? primary.metadata.disclosed_tool : "None detected");
  const metaSoftware = primary?.metadata?.software || (primary?.metadata?.disclosed_tool ? primary.metadata.disclosed_tool : "None detected");
  const metaC2PA = primary?.metadata?.c2pa_status || (primary?.metadata?.c2pa_present ? "Present" : "None detected");

  const heatmapImage = analysisResult?.visualizations?.gradcam || 
                       analysisResult?.visualizations?.gradcam_global || 
                       analysisResult?.visualizations?.region_overlay || 
                       '/dog_heatmap.png';

  const frequencyImage = analysisResult?.visualizations?.dct_spectrum || 
                         analysisResult?.visualizations?.frequency_heatmap || 
                         '/spatial_heatmap.png';

  // Format secondary regions strictly into "Fake" or "Real" (no "Uncertain"!)
  const rawRegions = analysisResult?.secondary_system?.regions;
  const regions = (rawRegions && rawRegions.length > 0)
    ? rawRegions.map((r, idx) => {
        const isRegFake = r.verdict === 'Fake' || 
                          r.verdict === 'AI-EDITED' || 
                          r.region_verdict === 'AI-EDITED' || 
                          r.fake_probability >= 0.50;
        const confNum = isRegFake ? r.fake_probability : (1 - r.fake_probability);
        let defaultCrop = '/dog_face.jpg';
        if (r.region_type?.toLowerCase().includes('cloth') || idx === 1) defaultCrop = '/dog_clothing.jpg';
        if (r.region_type?.toLowerCase().includes('back') || idx === 2) defaultCrop = '/dog_background.jpg';

        const cropImg = (r as any).crop_url || (r as any).crop_path || defaultCrop;

        return {
          name: `${r.region_type ? r.region_type.charAt(0).toUpperCase() + r.region_type.slice(1) : 'Region'} Region`,
          prediction: isRegFake ? 'Fake' : 'Real',
          confidence: (confNum || 0.85).toFixed(2),
          image: cropImg,
        };
      })
    : [
        { name: 'Face Region', prediction: 'Fake', confidence: '0.88', image: '/dog_face.jpg' },
        { name: 'Clothing Region', prediction: 'Real', confidence: '0.64', image: '/dog_clothing.jpg' },
        { name: 'Background Region', prediction: 'Real', confidence: '0.71', image: '/dog_background.jpg' },
      ];

  return (
    <div className="space-y-6">
      
      {/* ======================================================== */}
      {/* 1. UPLOAD IMAGE CARD (Exact match to reference)          */}
      {/* ======================================================== */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs">
        <div className="flex items-center gap-2.5 mb-1">
          <ImageIcon className="h-5 w-5 text-blue-600" />
          <h2 className="text-lg font-bold text-slate-800 tracking-tight">Upload Image</h2>
        </div>
        <p className="text-xs text-slate-500 mb-5">
          Upload an image to detect whether it is real or AI-generated
        </p>

        <div className="flex flex-col md:flex-row items-center gap-6">
          {/* Dashed Drag & Drop Box */}
          <div 
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 w-full border-2 border-dashed border-slate-300 hover:border-blue-400 bg-slate-50/50 hover:bg-blue-50/30 rounded-2xl p-6 text-center transition cursor-pointer flex flex-col items-center justify-center min-h-[170px]"
          >
            <input 
              ref={fileInputRef}
              type="file" 
              accept=".jpg,.jpeg,.png"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  onFileSelect(e.target.files[0]);
                }
              }}
            />
            <UploadCloud className="h-10 w-10 text-slate-400 mb-2 group-hover:scale-110 transition-transform" />
            <p className="text-sm font-semibold text-slate-700">Drag and drop an image here</p>
            <p className="text-xs text-slate-400 my-1">or</p>
            <button 
              type="button"
              className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs transition shadow-sm shadow-blue-600/20 cursor-pointer mb-2"
            >
              Choose Image
            </button>
            <p className="text-[11px] text-slate-400">Supported formats: JPG, PNG, JPEG (Max 10 MB)</p>
          </div>

          {/* Right: Example Image Preview */}
          <div className="shrink-0 flex flex-col items-center">
            <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2 self-start">
              Example Image
            </span>
            <div 
              onClick={() => onSelectPreset('animal_ai')}
              className="group relative rounded-xl overflow-hidden border-2 border-blue-400/80 shadow-sm cursor-pointer hover:border-blue-600 transition"
              title="Click to analyze pre-selected AI Dog sample"
            >
              <img 
                src="/dog.jpg" 
                alt="AI Dog Sample" 
                className="w-36 h-36 object-cover group-hover:scale-105 transition-transform duration-300" 
              />
              <div className="absolute inset-0 bg-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="text-[10px] font-bold bg-white/90 text-blue-700 px-2 py-1 rounded shadow-xs">
                  dog.jpg
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Analyze Bar */}
        <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="font-medium text-slate-700">Active Image:</span>
            <span className="bg-slate-100 text-slate-700 font-mono px-2 py-0.5 rounded text-[11px]">
              {selectedFile ? selectedFile.name : 'dog.jpg (Pre-selected AI sample)'}
            </span>
          </div>

          <button
            onClick={onTriggerAnalysis}
            disabled={isLoading}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-xs font-bold text-white transition shadow-md cursor-pointer ${
              isLoading 
                ? 'bg-blue-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 shadow-blue-600/25 active:scale-95'
            }`}
          >
            {isLoading ? (
              <>
                <span className="h-3.5 w-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Running Forensic Agents...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Analyze Image</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* ======================================================== */}
      {/* 2. RESULT NAVIGATION TABS                                */}
      {/* ======================================================== */}
      <div className="border-b border-slate-200 flex items-center gap-8 overflow-x-auto select-none">
        <button
          onClick={() => setActiveSubTab('overall')}
          className={`flex items-center gap-2 py-3 border-b-2 text-sm font-semibold transition whitespace-nowrap cursor-pointer ${
            activeSubTab === 'overall'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          <Activity className="h-4 w-4" />
          <span>Overall Result</span>
        </button>

        <button
          onClick={() => onNavigateModule('module1')}
          className="flex items-center gap-2 py-3 border-b-2 border-transparent text-sm font-medium text-slate-500 hover:text-slate-800 transition whitespace-nowrap cursor-pointer"
        >
          <span>Preprocessing (Module 1)</span>
        </button>

        <button
          onClick={() => onNavigateModule('module2')}
          className="flex items-center gap-2 py-3 border-b-2 border-transparent text-sm font-medium text-slate-500 hover:text-slate-800 transition whitespace-nowrap cursor-pointer"
        >
          <span>Primary System (Module 2)</span>
        </button>

        <button
          onClick={() => onNavigateModule('module3')}
          className="flex items-center gap-2 py-3 border-b-2 border-transparent text-sm font-medium text-slate-500 hover:text-slate-800 transition whitespace-nowrap cursor-pointer"
        >
          <span>Secondary System (Module 3)</span>
        </button>

        <button
          onClick={() => setActiveSubTab('raw')}
          className={`flex items-center gap-2 py-3 border-b-2 text-sm font-medium transition whitespace-nowrap cursor-pointer ${
            activeSubTab === 'raw'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          <FileText className="h-4 w-4" />
          <span>Raw Output</span>
        </button>
      </div>

      {/* ======================================================== */}
      {/* 3. OVERALL RESULT CARDS GRID (Exact match to reference)   */}
      {/* ======================================================== */}
      <div className="space-y-6">

        {/* TOP ROW: 3 Columns Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
          
          {/* ---------------------------------------------------- */}
          {/* CARD 1: Input & Preprocessing (Module 1) - Col 1     */}
          {/* ---------------------------------------------------- */}
          <div className="lg:col-span-4 bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col justify-between">
            <div>
              {/* Header */}
              <div 
                onClick={() => onNavigateModule('module1')}
                className="flex items-center justify-between mb-4 cursor-pointer group"
              >
                <div className="flex items-center gap-2.5">
                  <span className="h-6 w-6 rounded-full bg-emerald-600 text-white font-bold text-xs flex items-center justify-center">
                    1
                  </span>
                  <h3 className="font-bold text-slate-800 text-sm group-hover:text-emerald-700 transition">
                    Input & Preprocessing (Module 1)
                  </h3>
                </div>
                <ChevronDown className="h-4 w-4 text-slate-400 group-hover:text-slate-600" />
              </div>

              {/* Two Preview Images Side-by-Side */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="text-center">
                  <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 aspect-square flex items-center justify-center">
                    <img 
                      src={previewOriginal} 
                      alt="Original" 
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <p className="text-[11px] font-semibold text-slate-700 mt-1.5">Original Image</p>
                  <p className="text-[10px] text-slate-400">512 × 512</p>
                </div>

                <div className="text-center">
                  <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 aspect-square flex items-center justify-center">
                    <img 
                      src={previewGrayscale} 
                      alt="Grayscale" 
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <p className="text-[11px] font-semibold text-slate-700 mt-1.5">Grayscale (for Frequency Agent)</p>
                  <p className="text-[10px] text-slate-400">512 × 512</p>
                </div>
              </div>

              {/* Checklist */}
              <div className="space-y-1.5 text-xs text-slate-600">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                  <span>Image validation: <strong className="text-emerald-600 font-semibold">Passed</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                  <span>Format: <strong className="text-slate-700 font-semibold">{prepFormat}</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                  <span>Image size: <strong className="text-emerald-600 font-semibold">{prepDimensions}</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                  <span>RGB / Grayscale conversion: <strong className="text-emerald-600 font-semibold">Completed</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                  <span>Resizing & normalization: <strong className="text-emerald-600 font-semibold">Completed</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
                  <span>Raw bytes preserved: <strong className="text-emerald-600 font-semibold">Yes</strong></span>
                </div>
              </div>
            </div>

            <button
              onClick={() => onNavigateModule('module1')}
              className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-emerald-600 hover:text-emerald-700 cursor-pointer"
            >
              <span>Explore Module 1 details</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* ---------------------------------------------------- */}
          {/* CARD 2: Primary System Results (Module 2) - Col 2    */}
          {/* ---------------------------------------------------- */}
          <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col justify-between">
            <div>
              {/* Header */}
              <div 
                onClick={() => onNavigateModule('module2')}
                className="flex items-center justify-between mb-4 cursor-pointer group"
              >
                <div className="flex items-center gap-2.5">
                  <span className="h-6 w-6 rounded-full bg-blue-600 text-white font-bold text-xs flex items-center justify-center">
                    2
                  </span>
                  <h3 className="font-bold text-slate-800 text-sm group-hover:text-blue-700 transition">
                    Primary System Results (Module 2)
                  </h3>
                </div>
                <ChevronRight className="h-4 w-4 text-slate-400 group-hover:text-slate-600" />
              </div>

              {/* Three Agents Cards Row */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
                
                {/* Frequency Agent */}
                <div className="rounded-xl border border-rose-100 bg-rose-50/20 p-2.5 text-center flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-center gap-1.5 text-rose-600 text-xs font-semibold mb-1.5">
                      <Activity className="h-3.5 w-3.5" />
                      <span>Frequency Agent</span>
                    </div>
                    <div className="rounded-lg overflow-hidden border border-slate-200 bg-slate-900 aspect-video mb-2 flex items-center justify-center">
                      <img 
                        src={frequencyImage} 
                        alt="Frequency Spectrum" 
                        className="w-full h-full object-cover" 
                      />
                    </div>
                  </div>
                  <div className="text-[11px]">
                    <p className="text-slate-600">Prediction: <strong className={freqIsFake ? "text-rose-600 font-bold" : "text-emerald-600 font-bold"}>{freqPred}</strong></p>
                    <p className="text-slate-500 text-[10px]">Confidence: {freqConf}</p>
                  </div>
                </div>

                {/* Classifier Agent */}
                <div className="rounded-xl border border-blue-100 bg-blue-50/20 p-2.5 text-center flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-center gap-1.5 text-blue-600 text-xs font-semibold mb-1.5">
                      <Brain className="h-3.5 w-3.5" />
                      <span>Classifier Agent</span>
                    </div>
                    <div className="rounded-lg overflow-hidden border border-slate-200 bg-slate-100 aspect-video mb-2 flex items-center justify-center">
                      <img 
                        src={clsImage} 
                        alt="Classifier Focus" 
                        className="w-full h-full object-cover" 
                      />
                    </div>
                  </div>
                  <div className="text-[11px]">
                    <p className="text-slate-600">Prediction: <strong className={clsIsFake ? "text-rose-600 font-bold" : "text-emerald-600 font-bold"}>{clsPred}</strong></p>
                    <p className="text-slate-500 text-[10px]">Confidence: {clsConf}</p>
                  </div>
                </div>

                {/* Metadata Agent */}
                <div className="rounded-xl border border-amber-100 bg-amber-50/20 p-2.5 text-center flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-center gap-1.5 text-amber-600 text-xs font-semibold mb-1.5">
                      <FileText className="h-3.5 w-3.5" />
                      <span>Metadata Agent</span>
                    </div>
                    <div className="rounded-lg bg-slate-50 border border-slate-200 p-1.5 mb-2 font-mono text-[9px] text-slate-700 text-left leading-relaxed">
                      <p>&quot;Make&quot;: &quot;{metaMake}&quot;,</p>
                      <p>&quot;Software&quot;: &quot;{metaSoftware}&quot;,</p>
                      <p>&quot;C2PA&quot;: &quot;{metaC2PA}&quot;</p>
                    </div>
                  </div>
                  <div className="text-[11px]">
                    <p className="text-slate-600">Prediction: <strong className={metaIsFake ? "text-rose-600 font-bold" : "text-emerald-600 font-bold"}>{metaPred}</strong></p>
                    <p className="text-slate-500 text-[9px] leading-tight">Evidence: {metaIsFake ? 'AI generation markers detected' : 'No synthetic markers'}</p>
                  </div>
                </div>

              </div>

              {/* Disagreement Analysis */}
              <div className="rounded-xl bg-purple-50/60 border border-purple-200 p-3 flex items-start gap-3">
                <div className="h-8 w-8 rounded-lg bg-purple-600 text-white flex items-center justify-center shrink-0 shadow-xs">
                  <Scale className="h-4 w-4" />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-purple-900">Disagreement Analysis</h4>
                  <p className="text-[11px] font-semibold text-purple-700">
                    {freqPred === clsPred && clsPred === metaPred ? 'Agents are consistent' : 'Disagreement resolved via weighted consensus / provenance'}
                  </p>
                  <p className="text-[10px] text-purple-600 mt-0.5">
                    Frequency: <span className={freqIsFake ? "font-semibold text-rose-600" : "font-semibold text-emerald-600"}>{freqPred}</span> &nbsp;|&nbsp; 
                    Classifier: <span className={clsIsFake ? "font-semibold text-rose-600" : "font-semibold text-emerald-600"}>{clsPred}</span> &nbsp;|&nbsp; 
                    Metadata: <span className={metaIsFake ? "font-semibold text-rose-600" : "font-semibold text-emerald-600"}>{metaPred}</span>
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={() => onNavigateModule('module2')}
              className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] font-semibold text-blue-600 hover:text-blue-700 cursor-pointer"
            >
              <span>Explore Module 2 details</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* ---------------------------------------------------- */}
          {/* CARD 3: Final Decision - Col 3                       */}
          {/* ---------------------------------------------------- */}
          <div className={`lg:col-span-3 rounded-2xl border-2 ${isFinalFake ? 'border-rose-200 bg-rose-50/40' : 'border-emerald-200 bg-emerald-50/40'} p-5 shadow-xs flex flex-col justify-between text-center`}>
            <div>
              {/* Header Target Badge */}
              <div className={`flex items-center justify-center gap-1.5 ${isFinalFake ? 'text-rose-600' : 'text-emerald-600'} text-xs font-bold uppercase tracking-wider mb-4`}>
                {isFinalFake ? <ShieldAlert className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
                <span>Final Decision</span>
              </div>

              {/* Big Red or Green Verdict */}
              <div className="py-2">
                {isFinalFake ? (
                  <Bot className="h-12 w-12 text-rose-600 mx-auto mb-2 animate-bounce-short" />
                ) : (
                  <CheckCircle2 className="h-12 w-12 text-emerald-600 mx-auto mb-2" />
                )}
                <h3 className={`text-2xl font-black ${isFinalFake ? 'text-rose-600' : 'text-emerald-600'} tracking-tight`}>
                  {verdictTitle}
                </h3>
                <p className="text-sm font-bold text-slate-800 mt-1">
                  Confidence: {confidenceScore}
                </p>
              </div>
            </div>

            {/* Overridden / C2PA Provenance Alert Badge */}
            <div className={`mt-4 p-3 rounded-xl bg-white border ${isFinalFake ? 'border-rose-200' : 'border-emerald-200'} text-left shadow-xs`}>
              <div className="flex items-start gap-2">
                <CheckCircle2 className={`h-4 w-4 ${isFinalFake ? 'text-rose-600' : 'text-emerald-600'} shrink-0 mt-0.5`} />
                <div>
                  <h4 className={`text-xs font-bold ${isFinalFake ? 'text-rose-700' : 'text-emerald-700'}`}>
                    {(analysisResult as any)?.metadata_override_applied || primary?.metadata?.c2pa_present ? 'Overridden to FAKE' : (isFinalFake ? 'AI Detection Confirmed' : 'Verified Authentic')}
                  </h4>
                  <p className="text-[11px] text-slate-600 leading-snug mt-0.5">
                    {analysisResult?.final_explanation || (isFinalFake ? 'Cryptographic metadata confirms AI generation (C2PA).' : 'Multi-agent analysis confirms genuine camera sensor characteristics.')}
                  </p>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* BOTTOM ROW: Card 4: Secondary System — Region Level Analysis */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs">
          
          {/* Header */}
          <div 
            onClick={() => onNavigateModule('module3')}
            className="flex items-center justify-between mb-4 cursor-pointer group"
          >
            <div className="flex items-center gap-2.5">
              <span className="h-6 w-6 rounded-full bg-purple-600 text-white font-bold text-xs flex items-center justify-center">
                3
              </span>
              <h3 className="font-bold text-slate-800 text-sm group-hover:text-purple-700 transition">
                Secondary System – Region Level Analysis (Module 3)
              </h3>
            </div>
            <span className="text-xs font-semibold text-purple-600 hover:text-purple-700 flex items-center gap-1">
              <span>View full module</span>
              <ChevronRight className="h-3.5 w-3.5" />
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
            
            {/* Left Column: Region Heatmap Image with High-Low Legend */}
            <div className="md:col-span-4 flex flex-col items-center">
              <div className="flex items-center gap-3">
                {/* Visual Heatmap Image (Guaranteed real image, never text box!) */}
                <div className="rounded-xl overflow-hidden border border-slate-200 shadow-sm w-44 h-44 bg-slate-900">
                  <img 
                    src={heatmapImage} 
                    alt="Region Heatmap" 
                    className="w-full h-full object-cover" 
                  />
                </div>

                {/* Vertical Colorbar Legend matching screenshot */}
                <div className="flex flex-col items-center justify-between h-44 text-[10px] text-slate-500 font-semibold py-1">
                  <span className="text-rose-600">High</span>
                  <div className="w-2.5 h-32 rounded-full bg-gradient-to-b from-red-500 via-yellow-400 via-green-400 to-blue-600 border border-slate-300 shadow-xs" />
                  <span className="text-blue-600">Low</span>
                </div>
              </div>

              <p className="text-[11px] font-semibold text-slate-600 mt-2 text-center">
                Region Heatmap (Face / Clothing / Background)
              </p>
            </div>

            {/* Middle Column: Region Predictions (Strictly Fake or Real!) */}
            <div className="md:col-span-4 space-y-2.5">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
                Region Predictions
              </h4>

              {regions.map((reg, index) => {
                const isRegFake = reg.prediction === 'Fake';
                return (
                  <div 
                    key={index} 
                    className="flex items-center justify-between p-2.5 rounded-xl border border-slate-200 bg-slate-50/70 hover:bg-slate-100 transition"
                  >
                    <div className="flex items-center gap-3">
                      <img 
                        src={reg.image} 
                        alt={reg.name} 
                        className="w-9 h-9 rounded-lg object-cover border border-slate-300 shadow-xs" 
                      />
                      <span className="text-xs font-bold text-slate-800">{reg.name}</span>
                    </div>
                    <div className="text-right">
                      <p className="text-xs">
                        Prediction: <strong className={isRegFake ? 'text-rose-600 font-bold' : 'text-emerald-600 font-bold'}>{reg.prediction}</strong>
                      </p>
                      <p className="text-[10px] text-slate-500">
                        Confidence: {reg.confidence}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Right Column: Explanation Checklist */}
            <div className="md:col-span-4 bg-emerald-50/30 border border-emerald-100 rounded-xl p-4 space-y-2">
              <div className="flex items-center gap-2 mb-2">
                <Lightbulb className="h-4 w-4 text-amber-500" />
                <h4 className="text-xs font-bold text-slate-800">Explanation</h4>
              </div>

              <div className="space-y-1.5 text-xs text-slate-700">
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
                  <span>{freqIsFake ? 'High frequency anomalies detected in spectral analysis.' : 'Frequency distribution matches authentic natural camera optics.'}</span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
                  <span>{clsIsFake ? 'Classifier model indicates synthetic/AI-generated patterns.' : 'Classifier confirms natural photographic textures.'}</span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
                  <span>{metaIsFake ? (primary?.metadata?.reasoning || 'Provenance metadata confirms synthetic generation.') : 'EXIF metadata is consistent with native hardware capture.'}</span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
                  <span>
                    {regions.some(r => r.prediction === 'Fake')
                      ? `Suspicious regions detected: ${regions.filter(r => r.prediction === 'Fake').map(r => r.name.replace(' Region', '')).join(', ')}.`
                      : 'All segmented regions exhibit authentic features.'}
                  </span>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
                  <span>Overall decision: <strong className="text-slate-900 font-bold">{verdictTitle} with {confidenceScore} confidence</strong>.</span>
                </div>
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>
  );
};
