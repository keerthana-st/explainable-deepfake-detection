import React from 'react';
import { 
  Image as ImageIcon, 
  Settings2, 
  Layers, 
  Scan, 
  History as HistoryIcon, 
  Info, 
  HelpCircle 
} from 'lucide-react';

interface SidebarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, setCurrentTab }) => {
  const navItems = [
    { id: 'detect', label: 'Detect Image', icon: ImageIcon },
    { id: 'module1', label: 'Module 1 Preprocessing', icon: Settings2 },
    { id: 'module2', label: 'Module 2 Primary System', icon: Layers },
    { id: 'module3', label: 'Module 3 Secondary System', icon: Scan },
    { id: 'history', label: 'History', icon: HistoryIcon },
    { id: 'about', label: 'About Project', icon: Info },
    { id: 'help', label: 'Help', icon: HelpCircle },
  ];

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col justify-between shrink-0 min-h-[calc(100vh-5rem)] p-4 select-none">
      <div className="space-y-1.5">
        <div className="px-3 py-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
          Navigation
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentTab(item.id)}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition cursor-pointer text-left ${
                isActive
                  ? 'bg-blue-50 text-blue-600 font-semibold border border-blue-100 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`}
            >
              <Icon className={`h-4.5 w-4.5 shrink-0 ${isActive ? 'text-blue-600' : 'text-slate-500'}`} />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* Forensic Engine Status Tag */}
      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5 text-xs text-slate-600">
        <div className="flex items-center justify-between font-semibold text-slate-700">
          <span>Engine Status</span>
          <span className="flex items-center gap-1.5 text-emerald-600 font-medium">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            Online
          </span>
        </div>
        <p className="text-[11px] text-slate-500 leading-relaxed">
          Primary & Secondary multi-agent arbitration pipeline active.
        </p>
      </div>
    </aside>
  );
};
