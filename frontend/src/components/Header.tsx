import React from 'react';
import { Shield, Home, Info, HelpCircle } from 'lucide-react';

interface HeaderProps {
  currentTab: string;
  onNavigate: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ currentTab, onNavigate }) => {
  return (
    <header className="h-20 bg-white border-b border-slate-200 px-6 md:px-8 flex items-center justify-between sticky top-0 z-30 shadow-xs">
      {/* Brand & Subtitle matching reference screenshot */}
      <div 
        onClick={() => onNavigate('detect')}
        className="flex items-center gap-3.5 cursor-pointer group select-none"
      >
        <div className="h-10 w-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
          <Shield className="h-6 w-6 fill-white/20 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight text-slate-900 group-hover:text-blue-600 transition-colors">
            Deepfake Detection
          </h1>
          <p className="text-xs text-slate-500 font-normal">
            An Explainable Deepfake Detection System Using Multiple Agents
          </p>
        </div>
      </div>

      {/* Right Action Buttons */}
      <div className="flex items-center gap-2">
        {/* Home Button (Active Pill in screenshot) */}
        <button
          onClick={() => onNavigate('detect')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition cursor-pointer ${
            currentTab === 'detect'
              ? 'bg-blue-50 text-blue-600 border border-blue-200 shadow-xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <Home className="h-4 w-4" />
          <span>Home</span>
        </button>

        {/* About Button */}
        <button
          onClick={() => onNavigate('about')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-sm font-medium transition cursor-pointer ${
            currentTab === 'about'
              ? 'bg-blue-50 text-blue-600 border border-blue-200'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <Info className="h-4 w-4" />
          <span>About</span>
        </button>

        {/* Help Button */}
        <button
          onClick={() => onNavigate('help')}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-sm font-medium transition cursor-pointer ${
            currentTab === 'help'
              ? 'bg-blue-50 text-blue-600 border border-blue-200'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <HelpCircle className="h-4 w-4" />
          <span>Help</span>
        </button>
      </div>
    </header>
  );
};
