import React from 'react';
import { ArrowRight } from 'lucide-react';

interface WorkflowIndicatorProps {
  currentStep?: number;
  onStepClick?: (stepIndex: number) => void;
}

export const WorkflowIndicator: React.FC<WorkflowIndicatorProps> = ({ 
  currentStep = 1,
  onStepClick 
}) => {
  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs">
      <div className="flex items-center justify-around gap-2 max-w-3xl mx-auto">
        
        {/* Step 1: Preprocessing */}
        <div 
          onClick={() => onStepClick?.(1)}
          className={`flex items-center gap-3 px-5 py-2.5 rounded-xl border transition cursor-pointer select-none ${
            currentStep === 1 
              ? 'bg-emerald-50/70 border-emerald-200 shadow-xs' 
              : 'bg-slate-50 border-slate-200/80 hover:bg-emerald-50/30'
          }`}
        >
          <div className="h-7 w-7 rounded-full bg-emerald-600 text-white font-bold text-xs flex items-center justify-center shrink-0 shadow-xs">
            1
          </div>
          <span className="font-bold text-slate-800 text-xs sm:text-sm">
            Preprocessing
          </span>
        </div>

        <ArrowRight className="h-4 w-4 text-slate-300 shrink-0" />

        {/* Step 2: Primary System */}
        <div 
          onClick={() => onStepClick?.(2)}
          className={`flex items-center gap-3 px-5 py-2.5 rounded-xl border transition cursor-pointer select-none ${
            currentStep === 2 
              ? 'bg-blue-50/70 border-blue-200 shadow-xs' 
              : 'bg-slate-50 border-slate-200/80 hover:bg-blue-50/30'
          }`}
        >
          <div className="h-7 w-7 rounded-full bg-blue-600 text-white font-bold text-xs flex items-center justify-center shrink-0 shadow-xs">
            2
          </div>
          <span className="font-bold text-slate-800 text-xs sm:text-sm">
            Primary System
          </span>
        </div>

        <ArrowRight className="h-4 w-4 text-slate-300 shrink-0" />

        {/* Step 3: Secondary System */}
        <div 
          onClick={() => onStepClick?.(3)}
          className={`flex items-center gap-3 px-5 py-2.5 rounded-xl border transition cursor-pointer select-none ${
            currentStep === 3 
              ? 'bg-purple-50/70 border-purple-200 shadow-xs' 
              : 'bg-slate-50 border-slate-200/80 hover:bg-purple-50/30'
          }`}
        >
          <div className="h-7 w-7 rounded-full bg-purple-600 text-white font-bold text-xs flex items-center justify-center shrink-0 shadow-xs">
            3
          </div>
          <span className="font-bold text-slate-800 text-xs sm:text-sm">
            Secondary System
          </span>
        </div>

      </div>
    </div>
  );
};
