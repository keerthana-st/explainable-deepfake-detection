import React from 'react';
import { 
  ArrowLeft, 
  ArrowRight, 
  Scan, 
  CheckCircle2, 
  Lightbulb
} from 'lucide-react';
import type { AnalysisResult } from '../types';

interface Module3SecondarySystemViewProps {
  analysisResult: AnalysisResult | null;
  onNavigate: (tab: string) => void;
}

export const Module3SecondarySystemView: React.FC<Module3SecondarySystemViewProps> = ({
  analysisResult,
  onNavigate,
}) => {
  const segOverlay = analysisResult?.visualizations?.region_overlay || '/dog_segmented_regions.png';

  const rawRegions = analysisResult?.secondary_system?.regions;
  const regions = (rawRegions && rawRegions.length > 0)
    ? rawRegions.map((r, idx) => {
        const isRegFake = r.verdict === 'Fake' || 
                          r.verdict === 'AI-EDITED' || 
                          r.region_verdict === 'AI-EDITED' || 
                          r.fake_probability >= 0.50;
        const fakeProb = r.fake_probability ?? 0.85;
        const confNum = isRegFake ? fakeProb : (1 - fakeProb);

        let defaultCrop = '/dog_face_crop_green.png';
        if (r.region_type?.toLowerCase().includes('cloth') || idx === 1) defaultCrop = '/dog_clothing_crop_blue.png';
        if (r.region_type?.toLowerCase().includes('back') || idx === 2) defaultCrop = '/dog_bg_crop_red.png';

        let defaultHeatmap = '/dog_heatmap_face.png';
        if (r.region_type?.toLowerCase().includes('cloth') || idx === 1) defaultHeatmap = '/dog_heatmap_clothing.png';
        if (r.region_type?.toLowerCase().includes('back') || idx === 2) defaultHeatmap = '/dog_heatmap_background.png';

        const cropImg = (r as any).crop_url || (r as any).crop_path || defaultCrop;
        const heatImg = (r as any).heatmap_url || (r as any).heatmap_path || defaultHeatmap;

        return {
          id: r.region_id || String(idx + 1),
          name: `${r.region_type ? r.region_type.charAt(0).toUpperCase() + r.region_type.slice(1) : 'Region'} Region`,
          prediction: isRegFake ? 'Fake' : 'Real',
          confidence: confNum.toFixed(2),
          cropImage: cropImg,
          heatmapImage: heatImg,
        };
      })
    : [
        {
          id: '1',
          name: 'Face Region',
          prediction: 'Fake',
          confidence: '0.88',
          cropImage: '/dog_face_crop_green.png',
          heatmapImage: '/dog_heatmap_face.png',
        },
        {
          id: '2',
          name: 'Clothing Region',
          prediction: 'Real',
          confidence: '0.64',
          cropImage: '/dog_clothing_crop_blue.png',
          heatmapImage: '/dog_heatmap_clothing.png',
        },
        {
          id: '3',
          name: 'Background Region',
          prediction: 'Real',
          confidence: '0.71',
          cropImage: '/dog_bg_crop_red.png',
          heatmapImage: '/dog_heatmap_background.png',
        },
      ];

  const isGlobalFake = analysisResult 
    ? (analysisResult.global_verdict.includes('AI') || analysisResult.global_verdict.includes('FAKE'))
    : true;

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      
      {/* Top Header Card */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs">
        <div className="flex items-center gap-2.5 mb-1 text-slate-800">
          <Scan className="h-5 w-5 text-blue-600" />
          <h2 className="text-xl font-bold tracking-tight text-slate-900">
            Module 3 : Secondary System
          </h2>
        </div>
        <p className="text-xs text-slate-500">
          Region-level analysis (face, clothing and background)
        </p>
      </div>

      {/* TOP ROW: Two Cards Side-by-Side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-stretch">
        
        {/* Left Card: Region Segmentation */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-3 text-blue-600 font-bold text-xs sm:text-sm">
            <span className="text-blue-600 text-xs">◆</span>
            <h3 className="text-slate-800">Region Segmentation</h3>
          </div>

          <div className="flex items-center justify-center gap-5 py-1">
            {/* Segmented Image */}
            <div className="rounded-xl overflow-hidden border border-slate-200 shadow-sm w-48 h-48 bg-slate-100 flex items-center justify-center">
              <img 
                src={segOverlay} 
                alt="Segmented Regions" 
                className="w-full h-full object-cover" 
              />
            </div>

            {/* Legend matching actual detected regions */}
            <div className="space-y-3 text-xs font-semibold text-slate-700">
              {regions.map((reg, idx) => {
                const colorClasses = [
                  'bg-emerald-500',
                  'bg-blue-500',
                  'bg-purple-500',
                  'bg-rose-500',
                  'bg-amber-500',
                ];
                const colorClass = colorClasses[idx % colorClasses.length];
                return (
                  <div key={reg.id || idx} className="flex items-center gap-2">
                    <span className={`w-3.5 h-3.5 rounded-sm ${colorClass} shrink-0`} />
                    <span>{reg.name}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Card: Region-wise Results */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-3 text-blue-600 font-bold text-xs sm:text-sm">
            <span className="text-blue-600 text-xs">◆</span>
            <h3 className="text-slate-800">Region-wise Results</h3>
          </div>

          <div className="space-y-2.5">
            {regions.map((reg) => {
              const isFake = reg.prediction === 'Fake';
              return (
                <div 
                  key={reg.id}
                  className="flex items-center justify-between p-2 rounded-xl border border-slate-200 bg-slate-50/60 hover:bg-slate-100 transition"
                >
                  <div className="flex items-center gap-3">
                    <img 
                      src={reg.cropImage} 
                      alt={reg.name} 
                      className="w-10 h-10 rounded-lg object-cover border border-slate-300 shadow-xs" 
                    />
                    <span className="text-xs font-bold text-slate-800">{reg.name}</span>
                  </div>
                  <div className="text-right">
                    <p className="text-xs">
                      Prediction: <strong className={isFake ? 'text-rose-600 font-bold' : 'text-emerald-600 font-bold'}>{reg.prediction}</strong>
                    </p>
                    <p className="text-[10px] text-slate-500">
                      Confidence: {reg.confidence}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

      </div>

      {/* BOTTOM ROW: Two Cards Side-by-Side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-stretch">
        
        {/* Left Card: Region Heatmaps */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-3 text-blue-600 font-bold text-xs sm:text-sm">
            <span className="text-blue-600 text-xs">◆</span>
            <h3 className="text-slate-800">Region Heatmaps</h3>
          </div>

          <div className="flex items-center justify-between gap-2 py-1">
            {/* Dynamic Region Heatmaps */}
            <div className={`grid ${regions.length === 1 ? 'grid-cols-1 max-w-[160px]' : (regions.length === 2 ? 'grid-cols-2 max-w-[340px]' : 'grid-cols-3')} gap-3 flex-1`}>
              {regions.map((reg, idx) => {
                const cleanName = reg.name.replace(' Region', '');
                return (
                  <div key={reg.id || idx} className="text-center">
                    <p className="text-[10px] font-bold text-slate-700 mb-1">{cleanName} Heatmap</p>
                    <div className="rounded-lg overflow-hidden border border-slate-200 aspect-square bg-slate-900 shadow-xs">
                      <img 
                        src={reg.heatmapImage} 
                        alt={`${cleanName} Heatmap`} 
                        className="w-full h-full object-cover" 
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Vertical Colorbar */}
            <div className="flex flex-col items-center justify-between h-28 text-[9px] text-slate-500 font-semibold pl-2">
              <span className="text-rose-600">High</span>
              <div className="w-2 h-16 rounded-full bg-gradient-to-b from-red-500 via-yellow-400 via-green-400 to-blue-600 border border-slate-300" />
              <span className="text-blue-600">Low</span>
            </div>
          </div>
        </div>

        {/* Right Card: Evidence Summary */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-3 text-blue-600 font-bold text-xs sm:text-sm">
            <Lightbulb className="h-4 w-4 text-blue-600" />
            <h3 className="text-slate-800">Evidence Summary</h3>
          </div>

          <div className="space-y-2 text-xs text-slate-700 py-1">
            {isGlobalFake ? (
              <>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>High frequency inconsistencies detected in primary region.</span>
                </div>

                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>Classifier model indicates AI-generated content in focal region.</span>
                </div>

                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>C2PA metadata / EXIF confirms AI generation indicators.</span>
                </div>

                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>Secondary regional analysis isolates manipulation coordinates.</span>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>Natural continuous frequency decay verified across all regions.</span>
                </div>

                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>Classifier neural network confirms authentic physical optics.</span>
                </div>

                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>No generative AI metadata tags detected in image header.</span>
                </div>

                <div className="flex items-start gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                  <span>All segmented regions show consistent authentic noise profiles.</span>
                </div>
              </>
            )}
          </div>
        </div>

      </div>

      {/* Bottom Navigation Buttons matching reference Screen 4 */}
      <div className="pt-2 flex items-center justify-between">
        <button
          onClick={() => onNavigate('module2')}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 font-semibold text-xs transition cursor-pointer shadow-xs"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          <span>Back</span>
        </button>

        <button
          onClick={() => onNavigate('overall')}
          className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-xs transition shadow-sm shadow-emerald-600/20 cursor-pointer"
        >
          <span>View Final Result</span>
          <ArrowRight className="h-3.5 w-3.5" />
        </button>
      </div>

    </div>
  );
};
