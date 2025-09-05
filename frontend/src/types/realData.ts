// REAL TypeScript interfaces - no mock data

export interface RealTeamMember {
  id: number;
  name: string;
  email: string;
  role: string;
  status: 'online' | 'offline' | 'busy' | 'away';
  phone?: string;
  department: string;
  location: string;
  experienceYears: number;
  skills: string[];
  permissions: string[];
  projects: string[];
  joinDate: string;
  lastActive: string;
  isActive: boolean;
  avatar: string;
  totalTasks: number;
  completedTasks: number;
  activeProjects: number;
  performanceScore: number;
}

export interface RealModelTraining {
  id: number;
  name: string;
  description?: string;
  status: 'pending' | 'training' | 'completed' | 'error' | 'paused';
  progress: number;
  accuracy: number;
  loss: number;
  parameters?: Record<string, any>;
  framework: string;
  doraRank: number;
  epochs: number;
  currentEpoch: number;
  startTime?: string;
  endTime?: string;
  timeRemaining?: string;
  datasetSize: number;
  modelSize: number;
  gpuUsage: number;
  memoryUsage: number;
  createdAt: string;
  updatedAt: string;
  teamMemberId?: number;
}

export interface RealSystemMetrics {
  id?: number;
  timestamp: string;
  cpuUsage: number;
  memoryUsage: number;
  gpuUsage: number;
  diskUsage: number;
  networkIn: number;
  networkOut: number;
  temperature: number;
  powerConsumption: number;
  activeConnections: number;
  queueSize: number;
  errorCount: number;
  isHealthy: boolean;
}

export interface RealAnalyticsData {
  id: number;
  timestamp: string;
  metricName: string;
  metricValue: number;
  metricType?: string;
  labels?: Record<string, any>;
  source?: string;
  description?: string;
}

export interface RealLegalDocument {
  id: number;
  title: string;
  content: string;
  documentType?: string;
  language: string;
  source?: string;
  category?: string;
  tags: string[];
  wordCount: number;
  pageCount: number;
  processingStatus: string;
  confidenceScore: number;
  createdAt: string;
  updatedAt: string;
  isPublic: boolean;
  metadata?: Record<string, any>;
}

export interface RealTrainingJob {
  id: number;
  name: string;
  description?: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  modelType?: string;
  datasetId?: string;
  hyperparameters?: Record<string, any>;
  metrics?: Record<string, any>;
  startTime?: string;
  endTime?: string;
  estimatedDuration?: number;
  actualDuration?: number;
  gpuRequirements?: Record<string, any>;
  memoryRequirements?: number;
  createdAt: string;
  updatedAt: string;
  assignedToId?: number;
}

export interface RealUser {
  id: number;
  username: string;
  email: string;
  fullName?: string;
  role: string;
  isActive: boolean;
  isVerified: boolean;
  createdAt: string;
  lastLogin?: string;
  preferences?: string;
  avatarUrl?: string;
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

// Error types
export interface ApiError {
  detail: string;
  status_code: number;
  timestamp: string;
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'system_metrics' | 'training_update' | 'notification' | 'error';
  data: any;
  timestamp: string;
}