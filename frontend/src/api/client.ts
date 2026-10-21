import axios from 'axios';
import type { AnalysisResult, DashboardStats, HistoryItem, Preset, SystemStatus } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 minutes for deep inference
});

export const forensicApi = {
  // Analyze uploaded image file
  async analyzeImage(file: File): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<AnalysisResult>('/api/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  // Analyze preset image
  async analyzePreset(presetId: string): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>('/api/analyze-preset', { preset_id: presetId });
    return response.data;
  },

  // Fetch available presets
  async getPresets(): Promise<Preset[]> {
    const response = await api.get<Preset[]>('/api/presets');
    return response.data;
  },

  // Fetch dashboard stats
  async getStats(): Promise<DashboardStats> {
    const response = await api.get<DashboardStats>('/api/stats');
    return response.data;
  },

  // Fetch history list
  async getHistory(): Promise<HistoryItem[]> {
    const response = await api.get<HistoryItem[]>('/api/history');
    return response.data;
  },

  // Fetch full analysis detail by id
  async getHistoryDetail(id: string): Promise<AnalysisResult> {
    const response = await api.get<AnalysisResult>(`/api/history/${id}`);
    return response.data;
  },

  // Fetch system health and agent status
  async getSystemStatus(): Promise<SystemStatus> {
    const response = await api.get<SystemStatus>('/api/system-status');
    return response.data;
  },
};
