// Dashboard Types for Persian Legal AI System

export interface TeamMember {
  id: number;
  name: string;
  email: string;
  role: string;
  status: string;
  phone?: string;
  department: string;
  location: string;
  experienceYears: number;
  skills: string[];
  permissions: string[];
  projects: string[];
  joinDate: Date;
  lastActive: Date;
  isActive: boolean;
  avatar: string;
  totalTasks: number;
  completedTasks: number;
  activeProjects: number;
  performanceScore: number;
}

export interface ModelTraining {
  id: number;
  name: string;
  type: string;
  status: string;
  progress: number;
  accuracy: number;
  loss: number;
  epochs: number;
  framework: string;
  parameters: string;
  doraRank: number;
  lastUpdated: Date;
  createdAt: Date;
}

export interface SystemMetric {
  title: string;
  value: number;
  unit: string;
  icon: any;
  color: string;
  details: string;
  chart: string;
}

export interface LogEntry {
  id: number;
  timestamp: Date;
  level: string;
  component: string;
  category: string;
  message: string;
  details?: string;
}

export interface NewMember {
  name: string;
  email: string;
  role: string;
  phone: string;
  permissions: string[];
  projects: string[];
}

export interface AppContextType {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  autoRefresh: boolean;
  setAutoRefresh: (refresh: boolean) => void;
  refreshInterval: number;
  setRefreshInterval: (interval: number) => void;
  isConnected: boolean | null;
  connectionTesting: boolean;
  testConnection: () => void;
  models: ModelTraining[];
  setModels: (models: ModelTraining[] | ((prev: ModelTraining[]) => ModelTraining[])) => void;
  realTimeData: any[];
  teamMembers: TeamMember[];
  systemLogs: LogEntry[];
  setSystemLogs: (logs: LogEntry[] | ((prev: LogEntry[]) => LogEntry[])) => void;
}