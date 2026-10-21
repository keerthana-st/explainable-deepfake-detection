export interface FrequencyEvidence {
  score: number;
  is_suspicious: boolean;
  raw_energy: number;
  reasoning: string;
  low_freq_energy_pct?: number;
  mid_freq_energy_pct?: number;
  high_freq_energy_pct?: number;
  energy_ratio_hf?: number;
  energy_ratio_hl?: number;
  spectral_entropy?: number;
}

export interface ClassifierEvidence {
  score: number;
  prediction: string;
  probabilities: Record<string, number>;
  noise_variance: number;
  disparity_ratio: number;
  reasoning: string;
}

export interface MetadataEvidence {
  flag: string;
  disclosed_tool: string | null;
  c2pa_present: boolean;
  c2pa_status?: string | null;
  c2pa_claim_generator?: string | null;
  make?: string | null;
  model?: string | null;
  software?: string | null;
  dqt_profile: string | null;
  reasoning: string;
}

export interface DisagreementEvidence {
  conflict_type: string;
  arbitration_reasoning: string;
}

export interface PrimarySystemPayload {
  global_verdict: string;
  confidence: number;
  confidence_display: string;
  decision_basis: string;
  badge: string;
  frequency: FrequencyEvidence;
  classifier: ClassifierEvidence;
  metadata: MetadataEvidence;
  disagreement: DisagreementEvidence;
}

export interface RegionData {
  region_id: string;
  region_type: string;
  bbox: number[]; // [ymin, xmin, ymax, xmax]
  classifier_prediction: string;
  real_probability: number;
  fake_probability: number;
  classifier_score?: number;
  frequency_score: number;
  frequency_anomaly: boolean;
  agent_agreement: boolean;
  agreement_type: string;
  agreement?: string;
  combined_score: number;
  region_verdict: string;
  verdict?: string;
  confidence: string | number;
  explanation: string;
  gradcam_path?: string | null;
}

export interface RegionalConsistency {
  status: 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'CONTRADICTED' | 'INCONCLUSIVE' | string;
  explanation: string;
  suspicious_regions: string[];
  authentic_regions: string[];
  requires_review: boolean;
  warning?: string | null;
}

export interface SecondarySystemPayload {
  enabled: boolean;
  executed: boolean;
  trigger_mode?: string;
  trigger_fired?: boolean;
  region_count: number;
  suspicious_region_count: number;
  regions: RegionData[];
  regional_consistency?: RegionalConsistency | null;
  suspicious_regions?: string[];
  visualizations: Record<string, string>;
  summary_explanation?: string;
}

export interface PreprocessingInfo {
  validation_passed?: boolean;
  format?: string;
  dimensions?: string;
  width?: number;
  height?: number;
  channels?: number;
  rgb_converted?: boolean;
  grayscale_converted?: boolean;
  resized?: boolean;
  normalized?: boolean;
  raw_bytes_preserved?: boolean;
  file_size_kb?: number;
  original_url?: string;
  rgb_url?: string;
  grayscale_url?: string;
  resized_url?: string;
}

export interface AnalysisResult {
  id: string;
  timestamp: string;
  filename: string;
  latency_seconds: number;
  status: string;
  global_verdict: string;
  confidence: number;
  confidence_display: string;
  decision_basis: string;
  badge: string;
  region_count: number;
  suspicious_region_count: number;
  consistency_status: string;
  preview_url?: string;
  preprocessing?: PreprocessingInfo;
  primary_system: PrimarySystemPayload;
  secondary_system?: SecondarySystemPayload | null;
  visualizations: Record<string, string>;
  final_explanation: string;
}

export interface DashboardStats {
  total_analyzed: number;
  ai_generated_count: number;
  ai_generated_pct: number;
  ai_edited_count: number;
  ai_edited_pct: number;
  real_count: number;
  real_pct: number;
  uncertain_count: number;
  uncertain_pct: number;
  recent_week_count: number;
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  filename: string;
  global_verdict: string;
  confidence: number;
  confidence_display: string;
  region_count: number;
  suspicious_region_count: number;
  consistency_status: string;
  preview_url?: string;
}

export interface SystemStatus {
  status: string;
  primary_system: string;
  secondary_system: string;
  ai_model: string;
  device: string;
  classifier: string;
  frequency_agent: string;
  metadata_agent: string;
  disagreement_layer: string;
  timestamp: string;
}

export interface Preset {
  id: string;
  name: string;
  description: string;
  available: boolean;
}
