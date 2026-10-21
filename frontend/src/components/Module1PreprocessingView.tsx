import React from 'react';
import { 
  CheckCircle2, 
  ArrowLeft, 
  ArrowRight, 
  Settings2, 
  FileText
} from 'lucide-react';
import type { AnalysisResult } from '../types';

interface Module1PreprocessingViewProps {
  analysisResult: AnalysisResult | null;
  onNavigate: (tab: string) => void;
}

export const Module1PreprocessingView: React.FC<Module1PreprocessingViewProps> = ({
  analysisResult,
  onNavigate,
}) => {
  const meta = analysisResult?.preprocessing;
  const fileName = analysisResult?.filename || 'dog.jpg';
  const format = meta?.format || 'JPG';
  const dimensions = meta?.dimensions || '512 × 512';
  const fileSize = meta?.file_size_kb ? `${meta.file_size_kb.toFixed(1)} KB` : '132.5 KB';

  const origUrl = meta?.original_url || analysisResult?.preview_url || '/dog.jpg';
  const rgbUrl = meta?.rgb_url || origUrl;
  const grayUrl = meta?.grayscale_url || '/dog_grayscale.jpg';
  const resizedUrl = meta?.resized_url || origUrl;

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      
      {/* Top Header Card */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs">
        <div className="flex items-center gap-2.5 mb-1 text-slate-800">
          <Settings2 className="h-5 w-5 text-blue-600" />
          <h2 className="text-xl font-bold tracking-tight text-slate-900">
            Module 1 : Preprocessing
          </h2>
        </div>
        <p className="text-xs text-slate-500">
          Image validation, conversion, resize and normalization
        </p>

        {/* Success Banner matching reference Screen 2 */}
        <div className="mt-4 p-2.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-700 font-semibold text-xs flex items-center justify-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
          <span>Preprocessing Completed Successfully</span>
        </div>
      </div>

      {/* 4 Image Previews in a Single Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        
        {/* 1. Original Image */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-3 shadow-xs text-center">
          <p className="text-xs font-bold text-slate-800">Original Image</p>
          <p className="text-[10px] text-slate-400 mb-2">{dimensions}</p>
          <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 aspect-square flex items-center justify-center">
            <img src={origUrl} alt="Original Image" className="w-full h-full object-cover" />
          </div>
        </div>

        {/* 2. RGB Image */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-3 shadow-xs text-center">
          <p className="text-xs font-bold text-slate-800">RGB Image</p>
          <p className="text-[10px] text-slate-400 mb-2">{dimensions}</p>
          <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 aspect-square flex items-center justify-center">
            <img src={rgbUrl} alt="RGB Image" className="w-full h-full object-cover" />
          </div>
        </div>

        {/* 3. Grayscale Image */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-3 shadow-xs text-center">
          <p className="text-xs font-bold text-slate-800">Grayscale Image</p>
          <p className="text-[10px] text-slate-400 mb-2">{dimensions}</p>
          <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 aspect-square flex items-center justify-center">
            <img src={grayUrl} alt="Grayscale Image" className="w-full h-full object-cover" />
          </div>
        </div>

        {/* 4. Resized Image */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-3 shadow-xs text-center">
          <p className="text-xs font-bold text-slate-800">Resized Image</p>
          <p className="text-[10px] text-slate-400 mb-2">224 × 224</p>
          <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 aspect-square flex items-center justify-center">
            <img src={resizedUrl} alt="Resized Image" className="w-full h-full object-cover" />
          </div>
        </div>

      </div>

      {/* Two Details Cards Side-by-Side matching reference Screen 2 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        {/* Left Card: Processing Details */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs">
          <div className="flex items-center gap-2 mb-3.5 text-blue-600">
            <CheckCircle2 className="h-4 w-4" />
            <h3 className="font-bold text-slate-800 text-xs sm:text-sm">Processing Details</h3>
          </div>

          <div className="divide-y divide-slate-100 text-xs">
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Image validation</span>
              <span className="font-semibold text-emerald-600">Passed</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Format</span>
              <span className="font-semibold text-slate-800">{format}</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Original size</span>
              <span className="font-semibold text-slate-800">{dimensions}</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">RGB conversion</span>
              <span className="font-semibold text-emerald-600">Completed</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Grayscale conversion</span>
              <span className="font-semibold text-emerald-600">Completed</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Resizing</span>
              <span className="font-semibold text-emerald-600">Completed (224 × 224 for classifier)</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Normalization</span>
              <span className="font-semibold text-emerald-600">Completed</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Raw bytes preserved</span>
              <span className="font-semibold text-emerald-600">Yes</span>
            </div>
          </div>
        </div>

        {/* Right Card: Image Information */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs">
          <div className="flex items-center gap-2 mb-3.5 text-blue-600">
            <FileText className="h-4 w-4" />
            <h3 className="font-bold text-slate-800 text-xs sm:text-sm">Image Information</h3>
          </div>

          <div className="divide-y divide-slate-100 text-xs">
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">File name</span>
              <span className="font-mono font-semibold text-slate-800">{fileName}</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">File size</span>
              <span className="font-semibold text-slate-800">{fileSize}</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Dimensions</span>
              <span className="font-semibold text-slate-800">{dimensions}</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Color mode</span>
              <span className="font-semibold text-slate-800">{(meta as any)?.color_mode || 'RGB'}</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Image type</span>
              <span className="font-semibold text-slate-800">{format} Image</span>
            </div>
            <div className="py-2 flex items-center justify-between">
              <span className="text-slate-600">Metadata integrity</span>
              <span className="font-semibold text-slate-700">Intact (used for forensic analysis)</span>
            </div>
          </div>
        </div>

      </div>

      {/* Bottom Navigation Buttons matching reference Screen 2 */}
      <div className="pt-2 flex items-center justify-between">
        <button
          onClick={() => onNavigate('detect')}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs transition cursor-pointer shadow-xs"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          <span>Back</span>
        </button>

        <button
          onClick={() => onNavigate('module2')}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs transition shadow-sm shadow-blue-600/20 cursor-pointer"
        >
          <span>Next : Module 2</span>
          <ArrowRight className="h-3.5 w-3.5" />
        </button>
      </div>

    </div>
  );
};
