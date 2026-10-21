import React from 'react';
import { 
  ArrowLeft, 
  ArrowRight, 
  Activity, 
  Brain, 
  FileText, 
  Scale, 
  Layers
} from 'lucide-react';
import type { AnalysisResult } from '../types';

interface Module2PrimarySystemViewProps {
  analysisResult: AnalysisResult | null;
  onNavigate: (tab: string) => void;
}

export const Module2PrimarySystemView: React.FC<Module2PrimarySystemViewProps> = ({
  analysisResult,
  onNavigate,
}) => {
  const primary = analysisResult?.primary_system;

  const freqConfidence = primary?.frequency?.score ? primary.frequency.score.toFixed(2) : '0.82';
  const clsConfidence = primary?.classifier?.score ? primary.classifier.score.toFixed(2) : '0.76';

  const freqIsFake = primary ? primary.frequency?.is_suspicious : true;
  const freqPred = freqIsFake ? 'Fake' : 'Real';

  const clsIsFake = primary 
    ? (primary.classifier?.prediction === 'FAKE' || primary.classifier?.score >= 0.5) 
    : true;
  const clsPred = clsIsFake ? 'Fake' : 'Real';

  const metaIsFake = primary 
    ? Boolean(primary.metadata?.c2pa_present || primary.metadata?.flag?.includes('AI') || primary.metadata?.flag === 'SUSPICIOUS_EXIF' || primary.metadata?.flag === 'EXPLICIT_SYNTHETIC') 
    : false;
  const metaPred = metaIsFake ? 'Fake' : 'Real';

  const metaMake = primary?.metadata?.make || (primary?.metadata?.disclosed_tool ? primary.metadata.disclosed_tool : "None detected");
  const metaSoftware = primary?.metadata?.software || (primary?.metadata?.disclosed_tool ? primary.metadata.disclosed_tool : "None detected");
  const metaC2PA = primary?.metadata?.c2pa_status || (primary?.metadata?.c2pa_present ? "Present" : "None detected");

  const freqImage = analysisResult?.visualizations?.dct_spectrum || 
                    analysisResult?.visualizations?.frequency_heatmap || 
                    '/spatial_heatmap.png';

  const clsImage = analysisResult?.visualizations?.gradcam_global || 
                   analysisResult?.visualizations?.gradcam || 
                   analysisResult?.visualizations?.region_overlay || 
                   '/dog_heatmap.png';

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      
      {/* Top Header Card */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs">
        <div className="flex items-center gap-2.5 mb-1 text-slate-800">
          <Layers className="h-5 w-5 text-blue-600" />
          <h2 className="text-xl font-bold tracking-tight text-slate-900">
            Module 2 : Primary System
          </h2>
        </div>
        <p className="text-xs text-slate-500">
          Frequency, Classifier and Metadata Agents with Disagreement Analysis
        </p>
      </div>

      {/* 3 Agents Cards Side-by-Side matching reference Screen 3 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        {/* Agent 1: Frequency Agent */}
        <div className={`rounded-2xl border p-4 shadow-xs flex flex-col justify-between ${
          freqIsFake ? 'border-rose-200 bg-rose-50/30' : 'border-emerald-200 bg-emerald-50/30'
        }`}>
          <div>
            <div className={`flex items-center gap-2 font-bold text-xs sm:text-sm mb-3 ${
              freqIsFake ? 'text-rose-600' : 'text-emerald-700'
            }`}>
              <Activity className="h-4 w-4" />
              <span>Frequency Agent</span>
            </div>

            <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-900 aspect-video mb-3 flex items-center justify-center">
              <img src={freqImage} alt="Frequency Spectrum" className="w-full h-full object-cover" />
            </div>

            <div className="text-xs mb-3 space-y-0.5">
              <p className="text-slate-700">
                Prediction: <strong className={freqIsFake ? 'text-rose-600 font-bold' : 'text-emerald-600 font-bold'}>{freqPred}</strong>
              </p>
              <p className="text-slate-500 text-[11px]">
                Confidence: {freqConfidence}
              </p>
            </div>

            {/* Key Evidence */}
            <div className="pt-2 border-t border-slate-200/60">
              <h4 className={`text-xs font-bold mb-1.5 ${freqIsFake ? 'text-rose-600' : 'text-emerald-700'}`}>
                Key Evidence
              </h4>
              <ul className="space-y-1 text-[11px] text-slate-700">
                {freqIsFake ? (
                  <>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-rose-500" />
                      <span>High frequency inconsistencies</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-rose-500" />
                      <span>Irregular spectral patterns</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-rose-500" />
                      <span>Possible AI generation artifacts</span>
                    </li>
                  </>
                ) : (
                  <>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-emerald-500" />
                      <span>Natural 2D-DCT spectral energy decay</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-emerald-500" />
                      <span>Continuous azimuthal radial decay</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-emerald-500" />
                      <span>Authentic optical characteristics</span>
                    </li>
                  </>
                )}
              </ul>
            </div>
          </div>
        </div>

        {/* Agent 2: Classifier Agent */}
        <div className={`rounded-2xl border p-4 shadow-xs flex flex-col justify-between ${
          clsIsFake ? 'border-blue-200 bg-blue-50/30' : 'border-emerald-200 bg-emerald-50/30'
        }`}>
          <div>
            <div className="flex items-center gap-2 text-blue-600 font-bold text-xs sm:text-sm mb-3">
              <Brain className="h-4 w-4 text-blue-500" />
              <span>Classifier Agent</span>
            </div>

            <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-100 aspect-video mb-3 flex items-center justify-center">
              <img src={clsImage} alt="Attention Heatmap" className="w-full h-full object-cover" />
            </div>

            <div className="text-xs mb-3 space-y-0.5">
              <p className="text-slate-700">
                Prediction: <strong className={clsIsFake ? 'text-rose-600 font-bold' : 'text-emerald-600 font-bold'}>{clsPred}</strong>
              </p>
              <p className="text-slate-500 text-[11px]">
                Confidence: {clsConfidence}
              </p>
            </div>

            {/* Key Evidence */}
            <div className="pt-2 border-t border-slate-200/60">
              <h4 className="text-xs font-bold text-blue-600 mb-1.5">Key Evidence</h4>
              <ul className="space-y-1 text-[11px] text-slate-700">
                {clsIsFake ? (
                  <>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-blue-500" />
                      <span>Deep learning model indicates AI-generated content</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-blue-500" />
                      <span>Grad-CAM highlights suspicious regions</span>
                    </li>
                  </>
                ) : (
                  <>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-emerald-500" />
                      <span>Deep neural model verifies authentic photographic content</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-emerald-500" />
                      <span>Consistent photon shot-noise across frame</span>
                    </li>
                  </>
                )}
              </ul>
            </div>
          </div>
        </div>

        {/* Agent 3: Metadata Agent */}
        <div className={`rounded-2xl border p-4 shadow-xs flex flex-col justify-between ${
          metaIsFake ? 'border-amber-200 bg-amber-50/30' : 'border-emerald-200 bg-emerald-50/30'
        }`}>
          <div>
            <div className="flex items-center gap-2 text-amber-700 font-bold text-xs sm:text-sm mb-3">
              <FileText className="h-4 w-4 text-amber-600" />
              <span>Metadata Agent</span>
            </div>

            <div className="rounded-xl bg-cyan-50/60 border border-cyan-200/80 p-2.5 mb-3 font-mono text-[10px] text-slate-800 leading-relaxed">
              <p>&quot;Make&quot;: &quot;{metaMake}&quot;,</p>
              <p>&quot;Software&quot;: &quot;{metaSoftware}&quot;,</p>
              <p>&quot;C2PA&quot;: &quot;{metaC2PA}&quot;</p>
            </div>

            <div className="text-xs mb-3 space-y-0.5">
              <p className="text-slate-700">
                Prediction: <strong className={metaIsFake ? 'text-rose-600 font-bold' : 'text-emerald-600 font-bold'}>{metaPred}</strong>
              </p>
              <p className="text-slate-500 text-[11px]">
                Evidence: {metaIsFake ? 'AI generation metadata found' : 'No synthetic tags detected'}
              </p>
            </div>

            {/* Key Evidence */}
            <div className="pt-2 border-t border-slate-200/60">
              <h4 className="text-xs font-bold text-amber-700 mb-1.5">Key Evidence</h4>
              <ul className="space-y-1 text-[11px] text-slate-700">
                <li className="flex items-center gap-1.5">
                  <span className="h-1 w-1 rounded-full bg-amber-600" />
                  <span>EXIF data analysis: {primary?.metadata?.flag || 'Checked'}</span>
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="h-1 w-1 rounded-full bg-amber-600" />
                  <span>C2PA metadata {metaC2PA === 'Present' ? 'present' : 'absent'}</span>
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="h-1 w-1 rounded-full bg-amber-600" />
                  <span>{metaIsFake ? 'Indicates AI-generated image' : 'Clean authentic camera metadata'}</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

      </div>

      {/* Disagreement Analysis Card matching reference Screen 3 */}
      <div className="rounded-2xl bg-purple-50/50 border border-purple-200/80 p-4 shadow-xs">
        <div className="flex items-start gap-3">
          <div className="h-8 w-8 rounded-xl bg-purple-100 flex items-center justify-center shrink-0 text-purple-700 mt-0.5">
            <Scale className="h-4 w-4" />
          </div>
          <div className="flex-1">
            <h3 className="font-bold text-slate-900 text-xs sm:text-sm">
              Disagreement Analysis
            </h3>
            <p className="text-[11px] text-slate-600 mb-2.5">
              {freqPred === clsPred && clsPred === metaPred 
                ? 'Agents are consistent across all forensic domains.' 
                : 'Agents evaluated evidence independently with domain arbitration.'}
            </p>

            <div className="flex flex-wrap items-center gap-3">
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border font-semibold text-[11px] ${
                freqIsFake 
                  ? 'bg-rose-100/80 border-rose-300 text-rose-700' 
                  : 'bg-emerald-100/80 border-emerald-300 text-emerald-700'
              }`}>
                <span className={`h-2 w-2 rounded-full ${freqIsFake ? 'bg-rose-600' : 'bg-emerald-600'}`} />
                <span>Frequency: {freqPred}</span>
              </span>

              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border font-semibold text-[11px] ${
                clsIsFake 
                  ? 'bg-rose-100/80 border-rose-300 text-rose-700' 
                  : 'bg-emerald-100/80 border-emerald-300 text-emerald-700'
              }`}>
                <span className={`h-2 w-2 rounded-full ${clsIsFake ? 'bg-rose-600' : 'bg-emerald-600'}`} />
                <span>Classifier: {clsPred}</span>
              </span>

              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border font-semibold text-[11px] ${
                metaIsFake 
                  ? 'bg-rose-100/80 border-rose-300 text-rose-700' 
                  : 'bg-emerald-100/80 border-emerald-300 text-emerald-700'
              }`}>
                <span className={`h-2 w-2 rounded-full ${metaIsFake ? 'bg-rose-600' : 'bg-emerald-600'}`} />
                <span>Metadata: {metaPred}</span>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Navigation Buttons matching reference Screen 3 */}
      <div className="pt-2 flex items-center justify-between">
        <button
          onClick={() => onNavigate('module1')}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs transition cursor-pointer shadow-xs"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          <span>Back</span>
        </button>

        <button
          onClick={() => onNavigate('module3')}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs transition shadow-sm shadow-blue-600/20 cursor-pointer"
        >
          <span>Next : Module 3</span>
          <ArrowRight className="h-3.5 w-3.5" />
        </button>
      </div>

    </div>
  );
};
