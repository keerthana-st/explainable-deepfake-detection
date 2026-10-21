import React from 'react';

export const AboutView: React.FC = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-10 animate-in fade-in duration-300">
      {/* Header Matching Screenshot 4 */}
      <div className="space-y-2">
        <h2 className="text-2xl font-bold text-white tracking-tight">
          About the Project
        </h2>
        <h3 className="text-lg font-semibold text-blue-400">
          Explainable Deepfake Detection System
        </h3>
        <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
          This project uses multiple AI agents to detect manipulated images and videos and provides explainable results to help users understand the decision.
        </p>
      </div>

      {/* How it Works Section Matching Screenshot 4 */}
      <div className="space-y-4">
        <h4 className="text-base font-bold text-white tracking-wide">
          How it Works
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 relative">
          {/* Step 1: Upload Media (Blue) */}
          <div className="bg-[#0B1530] border border-blue-500/50 rounded-2xl p-5 flex items-center space-x-4 shadow-lg">
            <div className="h-10 w-10 rounded-xl bg-blue-600 flex items-center justify-center text-white font-bold text-base shrink-0 shadow-md shadow-blue-500/30">
              1
            </div>
            <div>
              <h5 className="text-sm font-bold text-white">Upload Media</h5>
              <p className="text-xs text-slate-400 mt-0.5">Image or Video</p>
            </div>
          </div>

          {/* Step 2: AI Analysis (Purple) */}
          <div className="bg-[#150F2E] border border-purple-500/50 rounded-2xl p-5 flex items-center space-x-4 shadow-lg">
            <div className="h-10 w-10 rounded-xl bg-purple-600 flex items-center justify-center text-white font-bold text-base shrink-0 shadow-md shadow-purple-500/30">
              2
            </div>
            <div>
              <h5 className="text-sm font-bold text-white">AI Analysis</h5>
              <p className="text-xs text-slate-400 mt-0.5">Multiple Agents</p>
            </div>
          </div>

          {/* Step 3: Explainability (Green) */}
          <div className="bg-[#0C1E1B] border border-emerald-500/50 rounded-2xl p-5 flex items-center space-x-4 shadow-lg">
            <div className="h-10 w-10 rounded-xl bg-emerald-600 flex items-center justify-center text-white font-bold text-base shrink-0 shadow-md shadow-emerald-500/30">
              3
            </div>
            <div>
              <h5 className="text-sm font-bold text-white">Explainability</h5>
              <p className="text-xs text-slate-400 mt-0.5">Heatmaps & Insights</p>
            </div>
          </div>

          {/* Step 4: Final Result (Orange) */}
          <div className="bg-[#24170E] border border-amber-500/50 rounded-2xl p-5 flex items-center space-x-4 shadow-lg">
            <div className="h-10 w-10 rounded-xl bg-amber-600 flex items-center justify-center text-white font-bold text-base shrink-0 shadow-md shadow-amber-500/30">
              4
            </div>
            <div>
              <h5 className="text-sm font-bold text-white">Final Result</h5>
              <p className="text-xs text-slate-400 mt-0.5">Real / Fake / Uncertain</p>
            </div>
          </div>
        </div>
      </div>

      {/* Detection Agents Section Matching Screenshot 4 */}
      <div className="space-y-4">
        <h4 className="text-base font-bold text-white tracking-wide">
          Detection Agents
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Frequency Agent */}
          <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-5 space-y-2.5">
            <h5 className="text-sm font-bold text-white">
              Frequency Agent
            </h5>
            <p className="text-xs text-slate-400 leading-relaxed">
              Analyzes frequency domain features.
            </p>
          </div>

          {/* Classifier Agent */}
          <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-5 space-y-2.5">
            <h5 className="text-sm font-bold text-white">
              Classifier Agent
            </h5>
            <p className="text-xs text-slate-400 leading-relaxed">
              Uses deep learning models for classification.
            </p>
          </div>

          {/* Metadata Agent */}
          <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-5 space-y-2.5">
            <h5 className="text-sm font-bold text-white">
              Metadata Agent
            </h5>
            <p className="text-xs text-slate-400 leading-relaxed">
              Checks metadata and provenance information.
            </p>
          </div>

          {/* Decision Agent */}
          <div className="bg-[#0D152D] border border-[#19264D] rounded-xl p-5 space-y-2.5">
            <h5 className="text-sm font-bold text-white">
              Decision Agent
            </h5>
            <p className="text-xs text-slate-400 leading-relaxed">
              Combines results and provides final prediction.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
