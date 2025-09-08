// Training types for Persian Legal AI Frontend
export interface TrainingSessionRequest {
  model_type: 'dora' | 'qr_adaptor';
  model_name: string;
  config: Record<string, any>;
  data_source: 'sample' | 'qavanin' | 'majlis';
  task_type: 'text_classification' | 'question_answering' | 'text_generation';
}

export interface TrainingSessionResponse {
  session_id: string;
  status: string;
  message: string;
  created_at: string;
}

export interface TrainingProgress {
  data_loaded: boolean;
  model_initialized: boolean;
  training_started: boolean;
  training_completed: boolean;
  train_samples: number;
  eval_samples: number;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
}

export interface TrainingMetrics {
  total_steps?: number;
  total_epochs?: number;
  total_loss?: number;
  current_loss?: number;
  learning_rate?: number;
  accuracy?: number;
  f1_score?: number;
  precision?: number;
  recall?: number;
  training_time?: number;
  gpu_utilization?: number;
  memory_usage?: number;
}

export interface TrainingSessionStatus {
  session_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  progress: TrainingProgress;
  metrics: TrainingMetrics;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  failed_at?: string;
  error_message?: string;
}

export interface TrainingLog {
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  message: string;
  details?: Record<string, any>;
}

export interface VerifiedTrainingSession {
  session_id: string;
  status: string;
  progress: Record<string, any>;
  metrics: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface DatasetInfo {
  name: string;
  size: number;
  quality_score: number;
  last_updated: string;
  source: string;
}