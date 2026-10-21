import React, { useRef, useState } from 'react';
import { Video, Film, Sparkles } from 'lucide-react';

export const VideoAnalyzeView: React.FC = () => {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleVideo = (file: File) => {
    setSelectedVideo(file);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-in fade-in duration-300">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Analyze Video
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Upload a video file to detect facial manipulations and temporal deepfakes
        </p>
      </div>

      <div className="bg-[#0D152D] border border-[#19264D] rounded-2xl p-8 shadow-xl space-y-6">
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-[#263768] hover:border-blue-500/60 bg-[#0A1024] hover:bg-[#0D1530] rounded-2xl p-10 text-center cursor-pointer transition-all"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files?.[0] && handleVideo(e.target.files[0])}
            accept=".mp4,.avi,.mov,.mkv"
            className="hidden"
          />

          <div className="mx-auto w-14 h-14 rounded-full bg-[#131E3D] flex items-center justify-center text-blue-400 mb-3">
            <Video className="h-7 w-7" />
          </div>

          <p className="text-base font-medium text-slate-200">
            Drag & drop a video here
          </p>
          <p className="text-xs text-slate-500 my-2">or</p>

          <button
            type="button"
            className="px-6 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition-all shadow-md shadow-blue-600/30 cursor-pointer"
          >
            Choose Video File
          </button>

          <p className="text-xs text-slate-500 mt-4">
            Supported formats: MP4, AVI, MOV (Max 50MB)
          </p>
        </div>

        {selectedVideo && (
          <div className="bg-[#090F22] border border-[#1C2C57] rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Film className="h-6 w-6 text-blue-400" />
              <div>
                <p className="text-sm font-semibold text-white">{selectedVideo.name}</p>
                <p className="text-xs text-slate-400">{(selectedVideo.size / (1024 * 1024)).toFixed(1)} MB</p>
              </div>
            </div>
            <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2 py-1 rounded border border-cyan-500/20">
              Ready
            </span>
          </div>
        )}

        <div className="bg-[#0A1024] border border-[#1C2C57] rounded-xl p-4 text-xs text-slate-400 flex items-start space-x-3">
          <Sparkles className="h-4 w-4 text-blue-400 shrink-0 mt-0.5" />
          <div>
            <b className="text-slate-200">Temporal & Frame-Level Multi-Agent Architecture:</b>
            <p className="mt-1 leading-relaxed">
              Videos are extracted at 5 FPS keyframes and concurrently dispatched to the 2D-DCT Frequency Agent and EfficientNet-B0 Classifier to detect face-swap flicker and temporal synthesis artifacts.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
