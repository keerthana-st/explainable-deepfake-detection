import React, { useState, useEffect, useCallback } from 'react';
import { AlertCircle } from 'lucide-react';

import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { DetectImageView } from './components/DetectImageView';
import { Module1PreprocessingView } from './components/Module1PreprocessingView';
import { Module2PrimarySystemView } from './components/Module2PrimarySystemView';
import { Module3SecondarySystemView } from './components/Module3SecondarySystemView';
import { OverallResultView } from './components/OverallResultView';
import { RawOutputView } from './components/RawOutputView';
import { HistoryView } from './components/HistoryView';
import { AboutProjectView } from './components/AboutProjectView';
import { HelpView } from './components/HelpView';

import { forensicApi } from './api/client';
import type { AnalysisResult, HistoryItem } from './types';

export const App: React.FC = () => {
  // Navigation tabs: 'detect' | 'module1' | 'module2' | 'module3' | 'overall' | 'history' | 'about' | 'help'
  const [currentTab, setCurrentTab] = useState<string>('detect');
  const [activeSubTab, setActiveSubTab] = useState<string>('overall');

  // File and Analysis State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('animal_ai');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // History & Refresh
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // Load history on mount
  const refreshHistory = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const data = await forensicApi.getHistory().catch(() => []);
      setHistory(data);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  // Execute analysis and navigate into Module 1 Preprocessing
  const handleNextFromDetect = async () => {
    setIsLoading(true);
    setError(null);
    try {
      let res: AnalysisResult;
      if (selectedFile) {
        res = await forensicApi.analyzeImage(selectedFile);
      } else {
        res = await forensicApi.analyzePreset(selectedPresetId || 'animal_ai');
      }

      setAnalysisResult(res);
      refreshHistory();
      // Flow directly into Module 1: Preprocessing for the given image!
      setCurrentTab('module1');
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Preprocessing & analysis failed. Verify backend API is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Analysis Trigger (Overall Result)
  const handleStartAnalysis = async (file?: File, presetId?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      let res: AnalysisResult;
      if (file) {
        res = await forensicApi.analyzeImage(file);
      } else if (presetId) {
        res = await forensicApi.analyzePreset(presetId);
      } else if (selectedFile) {
        res = await forensicApi.analyzeImage(selectedFile);
      } else {
        res = await forensicApi.analyzePreset(selectedPresetId || 'animal_ai');
      }

      setAnalysisResult(res);
      refreshHistory();
      setCurrentTab('overall');
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Analysis failed. Ensure backend API is active on port 8000.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectHistoryRecord = async (id: string) => {
    setIsLoading(true);
    try {
      const detail = await forensicApi.getHistoryDetail(id);
      setAnalysisResult(detail);
      setCurrentTab('overall');
    } catch (err) {
      console.error('Failed to fetch history detail:', err);
      setError('Could not retrieve historical record.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-800 flex flex-col antialiased selection:bg-blue-600/20 selection:text-blue-700">
      
      {/* Top Header matching reference screenshot */}
      <Header 
        currentTab={currentTab} 
        onNavigate={(tab) => {
          setCurrentTab(tab);
          setError(null);
        }} 
      />

      {/* Main Layout: Left Sidebar + Content Area */}
      <div className="flex-1 flex flex-col md:flex-row max-w-[1600px] w-full mx-auto">
        
        {/* Left Sidebar with 7 exact items */}
        <Sidebar 
          currentTab={currentTab} 
          setCurrentTab={(tab) => {
            setCurrentTab(tab);
            setError(null);
          }} 
        />

        {/* Content Body */}
        <main className="flex-1 p-5 md:p-7 min-w-0">
          
          {/* Error Banner */}
          {error && (
            <div className="mb-5 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-rose-600 shrink-0" />
                <span>{error}</span>
              </div>
              <button 
                onClick={() => setError(null)} 
                className="text-xs font-bold text-rose-800 hover:text-rose-950 cursor-pointer ml-3"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* PAGE 1: Detect Image (Screen 1 Upload View) */}
          {currentTab === 'detect' && (
            <DetectImageView
              selectedFile={selectedFile}
              onFileSelect={(file) => {
                setSelectedFile(file);
                setError(null);
              }}
              onSelectPreset={(pid) => {
                setSelectedPresetId(pid);
                setSelectedFile(null);
                setError(null);
              }}
              onNavigateNext={handleNextFromDetect}
              isLoading={isLoading}
            />
          )}

          {/* PAGE 2: Module 1 Preprocessing (Screen 2 View) */}
          {currentTab === 'module1' && (
            <Module1PreprocessingView
              analysisResult={analysisResult}
              onNavigate={(tab) => setCurrentTab(tab)}
            />
          )}

          {/* PAGE 3: Module 2 Primary System (Screen 3 View) */}
          {currentTab === 'module2' && (
            <Module2PrimarySystemView
              analysisResult={analysisResult}
              onNavigate={(tab) => setCurrentTab(tab)}
            />
          )}

          {/* PAGE 4: Module 3 Secondary System (Screen 4 View) */}
          {currentTab === 'module3' && (
            <Module3SecondarySystemView
              analysisResult={analysisResult}
              onNavigate={(tab) => setCurrentTab(tab)}
            />
          )}

          {/* PAGE 5: Overall Result View (Full multi-agent dashboard from media_1792126705769_89a6750b.jpg) */}
          {currentTab === 'overall' && (
            activeSubTab === 'raw' ? (
              <RawOutputView
                analysisResult={analysisResult}
                onBack={() => setActiveSubTab('overall')}
              />
            ) : (
              <OverallResultView
                analysisResult={analysisResult}
                selectedFile={selectedFile}
                onFileSelect={(file) => {
                  setSelectedFile(file);
                  handleStartAnalysis(file, undefined);
                }}
                onSelectPreset={(pid) => {
                  setSelectedPresetId(pid);
                  setSelectedFile(null);
                  handleStartAnalysis(undefined, pid);
                }}
                onTriggerAnalysis={() => handleStartAnalysis(selectedFile || undefined, selectedPresetId)}
                isLoading={isLoading}
                activeSubTab={activeSubTab}
                setActiveSubTab={setActiveSubTab}
                onNavigateModule={(mod) => setCurrentTab(mod)}
              />
            )
          )}

          {/* PAGE 6: History */}
          {currentTab === 'history' && (
            <HistoryView
              history={history}
              onSelectEntry={handleSelectHistoryRecord}
              onRefresh={refreshHistory}
              isRefreshing={isRefreshing}
            />
          )}

          {/* PAGE 7: About Project (Dual flows: Primary & Secondary!) */}
          {currentTab === 'about' && (
            <AboutProjectView />
          )}

          {/* PAGE 8: Help */}
          {currentTab === 'help' && (
            <HelpView />
          )}

        </main>
      </div>

    </div>
  );
};

export default App;
