import React, { useRef } from 'react';
import { 
  UploadCloud, 
  Image as ImageIcon, 
  ArrowRight
} from 'lucide-react';
import { WorkflowIndicator } from './WorkflowIndicator';

interface DetectImageViewProps {
  selectedFile: File | null;
  onFileSelect: (file: File) => void;
  onSelectPreset: (presetId: string) => void;
  onNavigateNext: () => void;
  isLoading?: boolean;
}

export const DetectImageView: React.FC<DetectImageViewProps> = ({
  selectedFile,
  onFileSelect,
  onSelectPreset,
  onNavigateNext,
  isLoading = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      
      {/* 3 Connected Workflow Indicator Pills matching reference Screen 1 */}
      <WorkflowIndicator currentStep={1} onStepClick={() => {}} />

      {/* Upload Image Main Card matching reference Screen 1 */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs">
        
        {/* Card Header */}
        <div className="flex items-center gap-2.5 mb-1 text-slate-800">
          <ImageIcon className="h-5 w-5 text-blue-600" />
          <h2 className="text-lg font-bold tracking-tight text-slate-900">
            Upload Image
          </h2>
        </div>
        <p className="text-xs text-slate-500 mb-6">
          Upload an image to detect whether it is real or AI-generated
        </p>

        {/* Drag & Drop Area and Example Image Container */}
        <div className="flex flex-col md:flex-row items-center gap-6">
          
          {/* Dashed Drag and Drop Dropzone */}
          <div 
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 w-full border-2 border-dashed border-slate-300 hover:border-blue-500 bg-slate-50/50 hover:bg-blue-50/20 rounded-2xl p-8 text-center transition cursor-pointer flex flex-col items-center justify-center min-h-[190px]"
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
            <UploadCloud className="h-10 w-10 text-slate-400 mb-2" />
            <p className="text-xs sm:text-sm font-semibold text-slate-700">
              Drag and drop an image here
            </p>
            <p className="text-xs text-slate-400 my-1">or</p>
            <button 
              type="button"
              className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs transition shadow-sm shadow-blue-600/20 cursor-pointer mb-2"
            >
              Choose Image
            </button>
            <p className="text-[11px] text-slate-400">
              Supported formats: JPG, PNG, JPEG (Max 10 MB)
            </p>
          </div>

          {/* Right Column: Example Image dog.jpg */}
          <div className="shrink-0 flex flex-col items-center">
            <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2 self-start">
              Example Image
            </span>
            <div 
              onClick={() => onSelectPreset('animal_ai')}
              className="group relative rounded-xl overflow-hidden border border-slate-200 shadow-sm cursor-pointer hover:border-blue-500 transition"
              title="Pre-selected dog.jpg sample"
            >
              <img 
                src="/dog.jpg" 
                alt="Example Dog" 
                className="w-40 h-40 object-cover group-hover:scale-105 transition-transform duration-300" 
              />
              <div className="absolute inset-0 bg-blue-600/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="text-[11px] font-bold bg-white/95 text-blue-700 px-2 py-1 rounded shadow-xs">
                  dog.jpg
                </span>
              </div>
            </div>
          </div>

        </div>

        {/* Selected file preview banner if any */}
        {selectedFile && (
          <div className="mt-4 p-3 rounded-xl bg-blue-50 border border-blue-200 text-blue-900 text-xs flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg overflow-hidden border border-blue-300 shrink-0 bg-white">
                <img 
                  src={URL.createObjectURL(selectedFile)} 
                  alt="Selected preview" 
                  className="w-full h-full object-cover" 
                />
              </div>
              <div>
                <span className="font-bold">{selectedFile.name}</span>
                <p className="text-[11px] text-blue-600">{(selectedFile.size / 1024).toFixed(1)} KB — Ready to Preprocess</p>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Right: Next Button matching reference Screen 1 */}
        <div className="mt-6 flex items-center justify-end">
          <button
            onClick={onNavigateNext}
            disabled={isLoading}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-xs text-white transition shadow-sm cursor-pointer ${
              isLoading 
                ? 'bg-blue-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 shadow-blue-600/20 active:scale-95'
            }`}
          >
            {isLoading ? (
              <>
                <span className="h-3.5 w-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Preprocessing Image...</span>
              </>
            ) : (
              <>
                <span>Next</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </>
            )}
          </button>
        </div>

      </div>

    </div>
  );
};
