import React from 'react';
import { Bell } from 'lucide-react';

interface NavbarProps {
  onGoHome?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onGoHome }) => {
  return (
    <header className="h-16 bg-[#080E1E]/80 backdrop-blur-md border-b border-[#15203B] px-8 flex items-center justify-between sticky top-0 z-20">
      <div>
        {/* Left spacing */}
      </div>

      <div className="flex items-center space-x-5">
        {/* Notification Bell */}
        <button 
          className="text-slate-400 hover:text-white transition relative p-1.5 rounded-lg hover:bg-[#111A33] cursor-pointer"
          title="Notifications"
        >
          <Bell className="h-5 w-5" />
          <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-blue-500" />
        </button>

        {/* User Pill Matching Screenshot: K Keerthana */}
        <div 
          onClick={onGoHome}
          className="flex items-center space-x-2.5 pl-2 cursor-pointer hover:opacity-90 transition"
          title="View Landing Page"
        >
          <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">
            K
          </div>
          <span className="text-sm font-semibold text-white tracking-wide">
            Keerthana
          </span>
        </div>
      </div>
    </header>
  );
};
