import React, { useState } from 'react';
import { Layers, Eye, Flame, BrainCircuit, Info } from 'lucide-react';
import type { RegionData } from '../types';

interface ImageViewerProps {
  originalUrl?: string;
  regionOverlayUrl?: string;
  heatmapUrl?: string;
  gradcamUrl?: string;
  regions?: RegionData[];
  onSelectRegion?: (region: RegionData) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  originalUrl,
  regionOverlayUrl,
  heatmapUrl,
  gradcamUrl,
  regions = [],
  onSelectRegion,
}) => {
  const [activeTab, setActiveTab] = useState<'original' | 'regions' | 'heatmap' | 'gradcam'>('regions');

  const tabs = [
    { id: 'original', label: 'Original', icon: Eye, available: !!originalUrl },
    { id: 'regions', label: 'Regions', icon: Layers, available: !!regionOverlayUrl },
    { id: 'heatmap', label: 'Heatmap', icon: Flame, available: !!heatmapUrl },
    { id: 'gradcam', label: 'Grad-CAM', icon: BrainCircuit, available: !!gradcamUrl },
  ] as const;

  const currentDisplayUrl = () => {
    switch (activeTab) {
      case 'original':
        return originalUrl;
      case 'regions':
        return regionOverlayUrl || originalUrl;
      case 'heatmap':
        return heatmapUrl || originalUrl;
      case 'gradcam':
        return gradcamUrl || originalUrl;
    }
  };

  return (
    <div className="bg-navy-900 border border-navy-700 rounded-2xl overflow-hidden flex flex-col h-full">
      {/* Header Tabs */}
      <div className="flex items-center justify-between border-b border-navy-700/80 px-4 py-2.5 bg-navy-950/70">
        <div className="flex items-center space-x-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                disabled={!tab.available}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                  isActive
                    ? 'bg-blue-600/20 text-cyan-400 border border-cyan-500/30'
                    : tab.available
                    ? 'text-slate-400 hover:text-white hover:bg-navy-800'
                    : 'text-slate-600 cursor-not-allowed'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        <span className="text-[11px] font-mono text-slate-400 hidden sm:inline-block">
          Interactive Forensic Inspection
        </span>
      </div>

      {/* Main Image Display Area */}
      <div className="relative flex-1 min-h-[380px] max-h-[520px] bg-navy-950 flex items-center justify-center p-4 overflow-hidden">
        {currentDisplayUrl() ? (
          <img
            src={currentDisplayUrl()}
            alt={activeTab}
            className="max-h-full max-w-full object-contain rounded-lg shadow-lg border border-navy-800"
          />
        ) : (
          <div className="text-center text-slate-500">
            <Info className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-xs font-mono">Visualization not available for this tab</p>
          </div>
        )}

        {/* Legend / Overlay details */}
        {activeTab === 'heatmap' && (
          <div className="absolute bottom-4 left-4 right-4 bg-navy-900/90 backdrop-blur-md border border-navy-700 rounded-xl p-3 flex items-center justify-between">
            <span className="text-xs font-mono text-slate-300">Spatial Suspicion Density</span>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-emerald-400">Low (Authentic)</span>
              <div className="h-2 w-32 rounded-full bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500" />
              <span className="text-[10px] font-mono text-rose-400">High (Suspicious)</span>
            </div>
          </div>
        )}

        {activeTab === 'regions' && (
          <div className="absolute bottom-4 left-4 right-4 bg-navy-900/90 backdrop-blur-md border border-navy-700 rounded-xl p-2.5 flex items-center justify-between">
            <div className="flex items-center gap-4 text-xs font-mono">
              <span className="flex items-center gap-1.5 text-rose-400">
                <span className="h-2.5 w-2.5 rounded-full bg-rose-500" />
                AI-EDITED
              </span>
              <span className="flex items-center gap-1.5 text-emerald-400">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                REAL
              </span>
              <span className="flex items-center gap-1.5 text-amber-400">
                <span className="h-2.5 w-2.5 rounded-full bg-amber-500" />
                UNCERTAIN
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-slate-400 hidden sm:inline-block">
                {regions.length} segmented regions
              </span>
              {regions.length > 0 && onSelectRegion && (
                <button
                  onClick={() => onSelectRegion(regions[0])}
                  className="text-[10px] font-mono text-cyan-400 hover:text-cyan-300 underline"
                >
                  Quick Inspect #1
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Grad-CAM Model Attribution Disclaimer */}
      {activeTab === 'gradcam' && (
        <div className="p-3 bg-navy-950/80 border-t border-navy-800 text-xs font-mono text-slate-400 flex items-start gap-2">
          <Info className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
          <span>
            <b>Scientific Notice:</b> Grad-CAM indicates spatial regions receiving stronger neural network visual attribution. It represents model reasoning, not standalone legal ground truth.
          </span>
        </div>
      )}
    </div>
  );
};
