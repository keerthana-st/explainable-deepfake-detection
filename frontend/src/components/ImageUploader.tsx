import React, { useRef, useState } from 'react';
import { UploadCloud, X, Sparkles, CheckCircle2 } from 'lucide-react';
import type { Preset } from '../types';

interface ImageUploaderProps {
  onFileSelected: (file: File) => void;
  onPresetSelected: (presetId: string) => void;
  presets: Preset[];
  selectedFile: File | null;
  selectedPresetId: string | null;
  onClear: () => void;
  onStartAnalysis: () => void;
  isLoading: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onFileSelected,
  onPresetSelected,
  presets,
  selectedFile,
  selectedPresetId,
  onClear,
  onStartAnalysis,
  isLoading,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageMeta, setImageMeta] = useState<{ width: number; height: number; size: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
      alert('Please upload a valid JPG, JPEG, or PNG image.');
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const img = new Image();
    img.onload = () => {
      setImageMeta({
        width: img.width,
        height: img.height,
        size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      });
    };
    img.src = url;

    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handlePresetClick = (preset: Preset) => {
    setPreviewUrl(null);
    setImageMeta({
      width: 1024,
      height: 1024,
      size: '1.4 MB',
    });
    onPresetSelected(preset.id);
  };

  const clearSelection = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setImageMeta(null);
    onClear();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Title & Subtitle Matching Screenshot 2 */}
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Analyze Image
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload an image to detect if it is real or fake
        </p>
      </div>

      {/* Main Upload Card */}
      <div className="bg-[#0D152D] border border-[#19264D] rounded-2xl p-8 shadow-xl space-y-6">
        {/* Dashed Drag & Drop Box Matching Screenshot 2 */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all ${
            isDragOver
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-[#263768] hover:border-blue-500/60 bg-[#0A1024] hover:bg-[#0D1530]'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            accept=".jpg,.jpeg,.png"
            className="hidden"
          />

          {/* Cloud Icon */}
          <div className="mx-auto w-14 h-14 rounded-full bg-[#131E3D] flex items-center justify-center text-blue-400 mb-3">
            <UploadCloud className="h-7 w-7" />
          </div>

          <p className="text-base font-medium text-slate-200">
            Drag & drop an image here
          </p>
          <p className="text-xs text-slate-500 my-2">or</p>

          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="px-6 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition-all shadow-md shadow-blue-600/30 cursor-pointer"
          >
            Choose File
          </button>

          <p className="text-xs text-slate-500 mt-4">
            Supported formats: JPG, JPEG, PNG (Max 10MB)
          </p>
        </div>

        {/* Selected Image Preview Card Matching Screenshot 2 */}
        {(selectedFile || selectedPresetId) && (
          <div className="bg-[#090F22] border border-[#1C2C57] rounded-xl p-4 flex items-center justify-between animate-in fade-in duration-200">
            <div className="flex items-center space-x-4">
              <div className="w-16 h-16 rounded-lg overflow-hidden bg-[#131E3D] border border-[#223363] shrink-0 flex items-center justify-center">
                {previewUrl ? (
                  <img src={previewUrl} alt="Thumbnail" className="w-full h-full object-cover" />
                ) : (
                  <div className="text-blue-400 text-xs font-mono font-bold text-center px-1">
                    Preset
                  </div>
                )}
              </div>

              <div>
                <h4 className="text-sm font-semibold text-white truncate max-w-xs md:max-w-md">
                  {selectedFile?.name || presets.find(p => p.id === selectedPresetId)?.name || 'Selected Image'}
                </h4>
                <div className="flex items-center space-x-3 text-xs text-slate-400 mt-1">
                  <span>{imageMeta?.size || '1.2 MB'}</span>
                  <span>•</span>
                  <span>{imageMeta ? `${imageMeta.width} × ${imageMeta.height}` : '1000 × 1000'}</span>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={clearSelection}
              disabled={isLoading}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-[#15203D] transition cursor-pointer"
              title="Remove"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        )}

        {/* Big Blue "Analyze Image" Button Matching Screenshot 2 */}
        <button
          type="button"
          onClick={onStartAnalysis}
          disabled={(!selectedFile && !selectedPresetId) || isLoading}
          className="w-full py-3.5 px-6 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 text-white font-semibold text-base shadow-lg shadow-blue-600/30 transition-all cursor-pointer flex items-center justify-center space-x-2"
        >
          <span>{isLoading ? 'Analyzing...' : 'Analyze Image'}</span>
        </button>
      </div>

      {/* Quick Benchmark Presets */}
      <div className="pt-2">
        <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-2.5">
          <Sparkles className="h-3.5 w-3.5 text-blue-400" />
          <span>Or test immediately with benchmark samples:</span>
        </div>
        <div className="flex flex-wrap gap-2.5">
          {presets.map((preset) => {
            const isSelected = selectedPresetId === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => handlePresetClick(preset)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer flex items-center gap-1.5 ${
                  isSelected
                    ? 'bg-blue-600/20 text-blue-400 border-blue-500'
                    : 'bg-[#0D152D] text-slate-400 border-[#19264D] hover:text-white hover:border-slate-500'
                }`}
              >
                <span>{preset.name.split('(')[0].trim()}</span>
                {isSelected && <CheckCircle2 className="h-3.5 w-3.5 text-blue-400" />}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
