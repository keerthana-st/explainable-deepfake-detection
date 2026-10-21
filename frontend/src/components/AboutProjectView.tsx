import { 
  ShieldCheck, 
  Workflow, 
  Scan, 
  CheckCircle2, 
  Lightbulb,
  GraduationCap
} from 'lucide-react';

export const AboutProjectView: React.FC = () => {
  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      
      {/* Page Header */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs">
        <div className="flex items-center gap-3 mb-2">
          <div className="h-10 w-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
              About the Project
            </h2>
            <p className="text-xs text-slate-500">
              An Explainable Deepfake Detection System Using Multiple Agents
            </p>
          </div>
        </div>
        <p className="text-xs text-slate-600 leading-relaxed max-w-4xl mt-3">
          Deepfake detection often suffers from the black-box dilemma: monolithic deep learning models produce binary decisions without transparent justification. This research platform introduces a multi-agent cooperative architecture that segregates frequency-domain signal analysis, deep visual representation modeling, cryptographic C2PA provenance validation, and spatial regional attention mapping.
        </p>
      </div>

      {/* ======================================================== */}
      {/* 1. PRIMARY SYSTEM FLOW (Required by User!)              */}
      {/* ======================================================== */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2.5 text-blue-600 pb-2 border-b border-slate-100">
          <Workflow className="h-5 w-5" />
          <h3 className="text-base font-bold text-slate-900 tracking-tight">
            1. Primary System Flow
          </h3>
        </div>

        <p className="text-xs text-slate-600">
          The Primary System receives the uploaded input image, applies strict integrity preprocessing, invokes three independent forensic agents simultaneously, and resolves inter-agent conflict via categorical arbitration:
        </p>

        {/* Visual Pipeline Steps */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3.5 pt-2">
          
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Step 1</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Input Image</h4>
            </div>
            <p className="text-[11px] text-slate-500 mt-2">
              User uploads JPG/PNG media for forensic scrutiny.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-emerald-50/60 border border-emerald-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider">Step 2</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Preprocessing</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Magic bytes validation, RGB conversion, grayscale luma, and tensor normalization.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-blue-50/60 border border-blue-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wider">Step 3</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Multi-Agent Suite</h4>
            </div>
            <div className="space-y-1 text-[10px] text-slate-700 font-medium mt-2">
              <p className="bg-white/80 py-0.5 rounded">Frequency Agent</p>
              <p className="bg-white/80 py-0.5 rounded">Classifier Agent</p>
              <p className="bg-white/80 py-0.5 rounded">Metadata Agent</p>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-purple-50/60 border border-purple-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-purple-600 uppercase tracking-wider">Step 4</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Disagreement Layer</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Zero-fusion arbitration and C2PA cryptographic provenance override engine.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-rose-50/60 border border-rose-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-rose-600 uppercase tracking-wider">Step 5</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Global Verdict</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              REAL, AI-EDITED, or AI-GENERATED with calibrated confidence score.
            </p>
          </div>

        </div>
      </div>

      {/* ======================================================== */}
      {/* 2. SECONDARY SYSTEM FLOW (Required by User!)            */}
      {/* ======================================================== */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2.5 text-purple-600 pb-2 border-b border-slate-100">
          <Scan className="h-5 w-5" />
          <h3 className="text-base font-bold text-slate-900 tracking-tight">
            2. Secondary System Flow
          </h3>
        </div>

        <p className="text-xs text-slate-600">
          The Secondary System provides deep spatial explainability. It isolates semantic regions of interest, conducts micro-level frequency and visual tests per patch, generates Grad-CAM activation maps, and verifies regional consistency:
        </p>

        {/* Visual Pipeline Steps */}
        <div className="grid grid-cols-1 md:grid-cols-6 gap-3 pt-2">
          
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Step 1</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Global Trigger</h4>
            </div>
            <p className="text-[11px] text-slate-500 mt-2">
              Secondary system activates based on configured trigger policy.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-purple-50/60 border border-purple-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-purple-600 uppercase tracking-wider">Step 2</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Region Segmentation</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Extracts Face, Clothing/Body, and Background bounding boxes.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-blue-50/60 border border-blue-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wider">Step 3</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Regional Tests</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Runs localized classifier logits & localized 2D-DCT spectral energy decay.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-amber-50/60 border border-amber-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-amber-600 uppercase tracking-wider">Step 4</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Evidence Fusion</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Evaluates agreement per region, assigning strictly <strong className="text-rose-600">Fake</strong> or <strong className="text-emerald-600">Real</strong>.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-rose-50/60 border border-rose-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-rose-600 uppercase tracking-wider">Step 5</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Grad-CAM Heatmap</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Computes gradients of target class to visualize exact spatial attention pixels.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-emerald-50/60 border border-emerald-200 text-center flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-emerald-600 uppercase tracking-wider">Step 6</span>
              <h4 className="text-xs font-bold text-slate-800 mt-1">Consistency Audit</h4>
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Validates that localized regional findings support the global verdict.
            </p>
          </div>

        </div>
      </div>

      {/* Research & Academic Demonstration Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="flex items-center gap-2 text-blue-600">
            <Lightbulb className="h-5 w-5" />
            <h4 className="font-bold text-slate-800 text-sm">Key Innovations</h4>
          </div>

          <ul className="space-y-2 text-xs text-slate-600">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Zero-Fusion Disagreement Layer:</strong> Does not average probabilities; instead reasons categorically over domain conflicts.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>C2PA Provenance Primacy:</strong> Seamlessly combines cryptographic identity verification with empirical deep learning.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
              <span><strong>Non-Override Secondary System:</strong> Region analysis adds granular explanation without corrupting global decisions.</span>
            </li>
          </ul>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="flex items-center gap-2 text-purple-600">
            <GraduationCap className="h-5 w-5" />
            <h4 className="font-bold text-slate-800 text-sm">Academic & Review Suitability</h4>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            Engineered specifically for SIH, research symposiums, and faculty project demonstrations. Every decision is transparently mapped to physical frequencies, deep neural logits, and tamper-evident image metadata.
          </p>

          <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-100">
            <span>Platform Version 2.0.0</span>
            <span>FastAPI + PyTorch + React</span>
          </div>
        </div>

      </div>

    </div>
  );
};
