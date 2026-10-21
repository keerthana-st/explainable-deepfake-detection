import React from 'react';
import { 
  ShieldCheck, 
  ArrowRight, 
  PlaySquare, 
  FileCheck2, 
  Layers, 
  Search
} from 'lucide-react';

interface LandingPageProps {
  onAnalyze: () => void;
  onAbout: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onAnalyze, onAbout }) => {
  return (
    <div className="min-h-screen bg-[#070D1B] text-slate-100 flex flex-col justify-between selection:bg-blue-500/30 selection:text-blue-300">
      {/* Top Navbar Matching Screenshot 1 */}
      <header className="h-20 px-8 max-w-7xl w-full mx-auto flex items-center justify-between border-b border-[#14203B]/60">
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => {}}>
          <div className="h-9 w-9 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <span className="text-lg font-bold tracking-tight text-white">
            DeepFake Detector
          </span>
        </div>

        {/* Center / Right Navigation Links */}
        <div className="flex items-center space-x-8">
          <nav className="hidden md:flex items-center space-x-7 text-sm font-medium text-slate-300">
            <button onClick={() => {}} className="text-white hover:text-blue-400 transition cursor-pointer">
              Home
            </button>
            <button onClick={onAbout} className="hover:text-white transition cursor-pointer">
              Features
            </button>
            <button onClick={onAbout} className="hover:text-white transition cursor-pointer">
              How It Works
            </button>
            <button onClick={onAbout} className="hover:text-white transition cursor-pointer">
              About
            </button>
          </nav>

          <button
            onClick={onAnalyze}
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/30 transition-all cursor-pointer"
          >
            Get Started
          </button>
        </div>
      </header>

      {/* Hero Section Matching Screenshot 1 */}
      <main className="max-w-7xl w-full mx-auto px-8 py-12 md:py-20 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center flex-1">
        {/* Left Headline & CTAs */}
        <div className="lg:col-span-7 space-y-6">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-white tracking-tight leading-[1.15]">
            Explainable <br />
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Deepfake Detection
            </span> <br />
            System
          </h1>

          <p className="text-base md:text-lg text-slate-300 max-w-xl leading-relaxed">
            Detect manipulated images and videos with AI and get clear, explainable results.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={onAnalyze}
              className="px-7 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-lg shadow-blue-600/30 flex items-center space-x-2 transition cursor-pointer"
            >
              <span>Analyze Media</span>
              <ArrowRight className="h-4 w-4" />
            </button>

            <button
              onClick={onAbout}
              className="px-7 py-3.5 rounded-xl bg-[#0F1832] hover:bg-[#162347] border border-[#223363] text-slate-200 font-semibold text-sm transition cursor-pointer"
            >
              Learn More
            </button>
          </div>
        </div>

        {/* Right: Face with Cybernetic Scan Grid Visual Matching Screenshot 1 */}
        <div className="lg:col-span-5 flex justify-center">
          <div className="relative w-72 h-88 md:w-80 md:h-96 rounded-2xl overflow-hidden shadow-2xl border border-[#1E3060] bg-[#0A1226] group">
            {/* Background Portrait Photo */}
            <img
              src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=800&q=80"
              alt="Deepfake Facial Scan"
              className="w-full h-full object-cover"
            />

            {/* Right Half: Cyan Digital Cybernetic Overlay */}
            <div className="absolute inset-y-0 right-0 w-1/2 bg-blue-600/30 backdrop-blur-[1px] border-l-2 border-cyan-400 flex items-center justify-center overflow-hidden">
              {/* Scanline Grid */}
              <div 
                className="absolute inset-0 opacity-40 pointer-events-none"
                style={{
                  backgroundImage: 'radial-gradient(circle, #38bdf8 1.5px, transparent 1.5px)',
                  backgroundSize: '12px 12px'
                }}
              />
              <div className="absolute inset-x-0 h-1 bg-cyan-300 opacity-80 animate-radar pointer-events-none" />
            </div>

            {/* Target Reticle Brackets */}
            <div className="absolute top-4 left-4 w-5 h-5 border-t-2 border-l-2 border-cyan-400 pointer-events-none" />
            <div className="absolute top-4 right-4 w-5 h-5 border-t-2 border-r-2 border-cyan-400 pointer-events-none" />
            <div className="absolute bottom-4 left-4 w-5 h-5 border-b-2 border-l-2 border-cyan-400 pointer-events-none" />
            <div className="absolute bottom-4 right-4 w-5 h-5 border-b-2 border-r-2 border-cyan-400 pointer-events-none" />
          </div>
        </div>
      </main>

      {/* Bottom 4 Feature Pills Matching Screenshot 1 */}
      <footer className="border-t border-[#14203B]/80 bg-[#091024]/80 backdrop-blur-sm py-8 px-8">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          <div className="flex flex-col items-center space-y-2">
            <div className="p-2.5 rounded-xl bg-blue-600/10 border border-blue-500/20 text-blue-400">
              <PlaySquare className="h-5 w-5" />
            </div>
            <span className="text-xs font-semibold text-slate-300">
              Image & Video Analysis
            </span>
          </div>

          <div className="flex flex-col items-center space-y-2">
            <div className="p-2.5 rounded-xl bg-cyan-600/10 border border-cyan-500/20 text-cyan-400">
              <FileCheck2 className="h-5 w-5" />
            </div>
            <span className="text-xs font-semibold text-slate-300">
              Explainable AI Results
            </span>
          </div>

          <div className="flex flex-col items-center space-y-2">
            <div className="p-2.5 rounded-xl bg-purple-600/10 border border-purple-500/20 text-purple-400">
              <Layers className="h-5 w-5" />
            </div>
            <span className="text-xs font-semibold text-slate-300">
              Multiple Detection Agents
            </span>
          </div>

          <div className="flex flex-col items-center space-y-2">
            <div className="p-2.5 rounded-xl bg-emerald-600/10 border border-emerald-500/20 text-emerald-400">
              <Search className="h-5 w-5" />
            </div>
            <span className="text-xs font-semibold text-slate-300">
              Detailed Forensic Insights
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};
