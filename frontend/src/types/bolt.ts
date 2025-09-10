// Bolt-specific type definitions
export namespace Bolt {
  export interface TrainingSession {
    id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    modelId: string;
    config: TrainingConfig;
    startTime?: Date;
    endTime?: Date;
    metrics?: TrainingMetrics;
  }

  export interface TrainingConfig {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    rank: number;
    alpha: number;
    target_modules: string[];
    quantization_bits: number;
    adaptive_rank: boolean;
  }

  export interface TrainingMetrics {
    loss: number;
    accuracy: number;
    cpu: number;
    memory: number;
    gpu: number;
  }

  export interface Model {
    id: string;
    name: string;
    status: 'active' | 'training' | 'inactive';
    accuracy: number;
    lastTrained: Date;
  }

  export interface Document {
    id: string;
    name: string;
    type: string;
    size: number;
    status: 'uploaded' | 'processing' | 'processed' | 'failed';
    uploadDate: Date;
    processedDate?: Date;
  }

  export interface Analytics {
    totalDocuments: number;
    processedDocuments: number;
    successRate: number;
    averageProcessingTime: number;
  }
}
