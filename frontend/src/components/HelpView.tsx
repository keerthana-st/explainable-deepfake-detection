import React from 'react';
import { 
  HelpCircle, 
  UploadCloud, 
  Layers, 
  Flame, 
  FileQuestion
} from 'lucide-react';

export const HelpView: React.FC = () => {
  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      
      {/* Header */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs">
        <div className="flex items-center gap-3 mb-2">
          <div className="h-10 w-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <HelpCircle className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
              Help & Documentation
            </h2>
            <p className="text-xs text-slate-500">
              User guide, interpretation manual, and forensic FAQ
            </p>
          </div>
        </div>
      </div>

      {/* Guide Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        
        {/* Step 1 */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="h-9 w-9 rounded-xl bg-blue-50 flex items-center justify-center text-blue-600">
            <UploadCloud className="h-5 w-5" />
          </div>
          <h3 className="font-bold text-slate-800 text-sm">1. Uploading Images</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Drag and drop or select an image from your device. Supported file formats include JPG, JPEG, and PNG up to 10 MB. Alternatively, click the <strong>dog.jpg</strong> pre-selected sample for immediate demonstration.
          </p>
        </div>

        {/* Step 2 */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="h-9 w-9 rounded-xl bg-purple-50 flex items-center justify-center text-purple-600">
            <Layers className="h-5 w-5" />
          </div>
          <h3 className="font-bold text-slate-800 text-sm">2. Multi-Agent Inspection</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Inspect the findings from the Frequency Agent (DCT spectrum), Classifier Agent (neural probability), and Metadata Agent (C2PA/EXIF). The Disagreement Layer reconciles conflicts transparently.
          </p>
        </div>

        {/* Step 3 */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs space-y-3">
          <div className="h-9 w-9 rounded-xl bg-rose-50 flex items-center justify-center text-rose-600">
            <Flame className="h-5 w-5" />
          </div>
          <h3 className="font-bold text-slate-800 text-sm">3. Spatial Heatmap & Regions</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            The Secondary System highlights spatial manipulation with Grad-CAM heatmaps (warm red = synthetic focus) and evaluates segmented regions as strictly <strong>Fake</strong> or <strong>Real</strong>.
          </p>
        </div>

      </div>

      {/* FAQ Section */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 text-slate-900 pb-2 border-b border-slate-100">
          <FileQuestion className="h-5 w-5 text-blue-600" />
          <h3 className="font-bold text-sm">Frequently Asked Questions</h3>
        </div>

        <div className="space-y-3 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-slate-800 mb-1">What does &quot;Overridden to FAKE&quot; mean?</h4>
            <p className="text-slate-600 leading-relaxed">
              When an image contains valid cryptographic C2PA provenance indicating generation via DALL-E, Midjourney, or similar generative AI software, the system overrides any ambiguous classifier score to FAKE with 0.90+ confidence.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-slate-800 mb-1">Why are regional predictions strictly Fake or Real?</h4>
            <p className="text-slate-600 leading-relaxed">
              To provide definitive, unambiguous forensic clarity for human investigators, the secondary system calculates weighted evidence across vision and frequency domains and delivers a definitive binary classification for each segmented anatomical patch.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-slate-800 mb-1">Are my uploaded images stored permanently?</h4>
            <p className="text-slate-600 leading-relaxed">
              Images are stored locally within the outputs workspace directory for the purpose of the historical audit ledger and can be reviewed or cleared from the History panel at any time.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
};
