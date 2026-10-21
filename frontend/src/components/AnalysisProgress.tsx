import React, { useEffect, useState } from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

interface AnalysisProgressProps {
  isLoading: boolean;
}

interface Step {
  id: number;
  label: string;
  subtext: string;
  category: 'prep' | 'primary' | 'secondary' | 'report';
}

const STEPS: Step[] = [
  { id: 1, label: 'Image Ingestion & Format Validation', subtext: 'Raw byte validation & EXIF preservation', category: 'prep' },
  { id: 2, label: 'Forensic Normalization (RGB & Grayscale)', subtext: 'Decoupled representation generation', category: 'prep' },
  { id: 3, label: 'Frequency Agent (2D-DCT Spectral Analysis)', subtext: '8×8 block transform & radial decay falloff', category: 'primary' },
  { id: 4, label: 'Vision Classifier (EfficientNet-B0)', subtext: 'Multi-crop spatial feature & sensor noise check', category: 'primary' },
  { id: 5, label: 'Metadata & Provenance Agent', subtext: 'C2PA cryptographic manifests & JPEG DQT table match', category: 'primary' },
  { id: 6, label: 'Disagreement Arbitration Layer', subtext: 'Zero-fusion categorical state matrix evaluation', category: 'primary' },
  { id: 7, label: 'Secondary Region Segmentation', subtext: 'Face, body, attire, and background spatial masks', category: 'secondary' },
  { id: 8, label: 'Region-Level Multi-Agent Inference', subtext: 'Independent localized feature extraction', category: 'secondary' },
  { id: 9, label: 'Global–Regional Consistency Evaluation', subtext: 'Non-override localized evidence concordance check', category: 'secondary' },
  { id: 10, label: 'Explainability Synthesis & Visualization', subtext: 'Grad-CAM, 2D heatmaps, and forensic narrative', category: 'report' },
];

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({ isLoading }) => {
  const [activeStep, setActiveStep] = useState(1);

  useEffect(() => {
    if (!isLoading) {
      setActiveStep(1);
      return;
    }

    // Progress smoothly through the stages while backend is computing
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev < STEPS.length ? prev + 1 : prev));
    }, 1200);

    return () => clearInterval(interval);
  }, [isLoading]);

  return (
    <div className="bg-navy-900 border border-navy-700/80 rounded-2xl p-8 max-w-3xl mx-auto shadow-2xl relative overflow-hidden">
      {/* Background Animated Radar Sweep */}
      <div className="absolute inset-x-0 h-2 bg-gradient-to-r from-transparent via-cyan-400 to-transparent opacity-60 animate-radar pointer-events-none" />

      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono font-medium mb-3">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          <span>MULTI-AGENT PIPELINE EXECUTING</span>
        </div>
        <h3 className="text-2xl font-extrabold text-white">Forensic Analysis In Progress</h3>
        <p className="text-sm text-slate-400 font-mono mt-0.5">
          Examining spatial, spectral, metadata, and regional domains simultaneously...
        </p>
      </div>

      {/* Stepper Timeline */}
      <div className="space-y-3 relative">
        <div className="absolute left-4.5 top-3 bottom-3 w-0.5 bg-navy-800 pointer-events-none" />

        {STEPS.map((step) => {
          const isCompleted = activeStep > step.id;
          const isCurrent = activeStep === step.id;

          return (
            <div
              key={step.id}
              className={`flex items-start gap-4 p-3 rounded-xl transition-all ${
                isCurrent
                  ? 'bg-blue-600/10 border border-cyan-500/30'
                  : isCompleted
                  ? 'bg-navy-950/40 opacity-90'
                  : 'opacity-40'
              }`}
            >
              <div className="relative z-10 shrink-0 mt-0.5">
                {isCompleted ? (
                  <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                ) : isCurrent ? (
                  <Loader2 className="h-5 w-5 text-cyan-400 animate-spin" />
                ) : (
                  <Circle className="h-5 w-5 text-slate-600" />
                )}
              </div>

              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4
                    className={`text-sm font-semibold ${
                      isCurrent ? 'text-cyan-300' : isCompleted ? 'text-slate-200' : 'text-slate-500'
                    }`}
                  >
                    {step.label}
                  </h4>
                  <span className="text-[10px] font-mono uppercase text-slate-400">
                    {step.category}
                  </span>
                </div>
                <p className="text-xs text-slate-400 font-mono mt-0.5">{step.subtext}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
