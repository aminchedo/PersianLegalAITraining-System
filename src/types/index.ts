// Base types
export interface BaseEntity {
  id: string
  createdAt: string
  updatedAt: string
}

// User types
export interface User extends BaseEntity {
  username: string
  email: string
  firstName: string
  lastName: string
  role: UserRole
  avatar?: string
  phone?: string
  nationalId?: string
  isActive: boolean
  lastLoginAt?: string
  preferences: UserPreferences
}

export type UserRole = 'admin' | 'user' | 'viewer'

export interface UserPreferences {
  theme: 'light' | 'dark'
  language: 'fa' | 'en'
  notifications: {
    email: boolean
    browser: boolean
    mobile: boolean
  }
  privacy: {
    profileVisibility: 'public' | 'private'
    dataSharing: boolean
  }
}

// Document types
export interface Document extends BaseEntity {
  title: string
  originalFilename: string
  filename: string
  mimeType: string
  size: number
  path: string
  uploadedBy: string
  status: DocumentStatus
  description?: string
  tags: string[]
  metadata: DocumentMetadata
  classification?: DocumentClassification
  processingHistory: ProcessingHistoryEntry[]
  // Additional properties for scraped documents
  scraped_at?: string
  source?: string
  category?: string
  url?: string
  content?: string
}

export type DocumentStatus = 'uploaded' | 'processing' | 'processed' | 'failed'

export interface DocumentMetadata {
  pages?: number
  words?: number
  characters?: number
  language?: string
  encoding?: string
  extractedText?: string
  thumbnailPath?: string
}

export interface DocumentClassification {
  primaryCategory: string
  confidence: number
  categories: ClassificationResult[]
  modelVersion: string
  processedAt: string
}

export interface ClassificationResult {
  category: string
  confidence: number
  description?: string
}

export interface ProcessingHistoryEntry {
  id: string
  action: ProcessingAction
  status: ProcessingStatus
  startedAt: string
  completedAt?: string
  duration?: number
  result?: any
  error?: string
  metadata?: Record<string, any>
}

export type ProcessingAction = 'upload' | 'text_extraction' | 'classification' | 'training' | 'analysis'
export type ProcessingStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

// Classification types
export interface ClassificationRequest {
  text: string
  options?: ClassificationOptions
}

export interface ClassificationOptions {
  confidenceThreshold?: number
  maxResults?: number
  categories?: string[]
  modelVersion?: string
}

export interface ClassificationResponse {
  text: string
  classification: ClassificationResult[]
  primaryCategory: string
  confidence: number
  modelVersion: string
  processedAt: string
  processingTime: number
}

export interface BatchClassificationRequest {
  documents: string[] // Document IDs
  options?: ClassificationOptions
}

export interface BatchClassificationJob extends BaseEntity {
  name: string
  totalDocuments: number
  processedDocuments: number
  status: BatchJobStatus
  startedAt?: string
  completedAt?: string
  results?: BatchJobResults
  options?: ClassificationOptions
  createdBy: string
}

export type BatchJobStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'

export interface BatchJobResults {
  successful: number
  failed: number
  averageConfidence: number
  categories: Record<string, number>
  errors?: string[]
}

// Training types
export interface TrainingSession extends BaseEntity {
  name: string
  description?: string
  dataset: TrainingDataset
  model: ModelConfiguration
  status: TrainingStatus
  progress: number
  startedAt?: string
  completedAt?: string
  duration?: number
  metrics?: TrainingMetrics
  logs?: TrainingLog[]
  createdBy: string
}

export type TrainingStatus = 'pending' | 'preparing' | 'training' | 'validating' | 'completed' | 'failed' | 'cancelled'

export interface TrainingDataset {
  id: string
  name: string
  description?: string
  totalSamples: number
  trainingSamples: number
  validationSamples: number
  testSamples: number
  categories: Record<string, number>
  createdAt: string
}

export interface ModelConfiguration {
  id: string
  name: string
  type: ModelType
  version: string
  parameters: Record<string, any>
  architecture?: string
}

export type ModelType = 'transformer' | 'lstm' | 'cnn' | 'bert' | 'custom'

export interface TrainingMetrics {
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  loss: number
  validationLoss: number
  confusionMatrix?: number[][]
  classificationReport?: Record<string, any>
}

export interface TrainingLog {
  timestamp: string
  level: LogLevel
  message: string
  metadata?: Record<string, any>
}

export type LogLevel = 'debug' | 'info' | 'warning' | 'error'

// Analytics types
export interface AnalyticsData {
  overview: OverviewStats
  documents: DocumentStats
  classification: ClassificationStats
  training: TrainingStats
  usage: UsageStats
  performance: PerformanceStats
}

export interface OverviewStats {
  totalDocuments: number
  totalClassifications: number
  totalUsers: number
  systemUptime: number
  lastUpdated: string
}

export interface DocumentStats {
  uploadedToday: number
  uploadedThisWeek: number
  uploadedThisMonth: number
  totalSize: number
  averageSize: number
  byType: Record<string, number>
  byStatus: Record<string, number>
}

export interface ClassificationStats {
  totalClassifications: number
  averageConfidence: number
  byCategory: Record<string, number>
  accuracyTrend: TimeSeriesData[]
  confidenceTrend: TimeSeriesData[]
}

export interface TrainingStats {
  totalSessions: number
  completedSessions: number
  averageAccuracy: number
  modelVersions: number
  trainingTime: number
}

export interface UsageStats {
  activeUsers: number
  apiCalls: number
  storageUsed: number
  bandwidthUsed: number
  dailyActivity: TimeSeriesData[]
}

export interface PerformanceStats {
  averageResponseTime: number
  systemLoad: number
  memoryUsage: number
  diskUsage: number
  errorRate: number
  uptimePercentage: number
}

export interface TimeSeriesData {
  timestamp: string
  value: number
  label?: string
}

// System types
export interface SystemHealth {
  status: SystemStatus
  version: string
  uptime: number
  components: ComponentHealth[]
  lastChecked: string
}

export type SystemStatus = 'healthy' | 'degraded' | 'down'

export interface ComponentHealth {
  name: string
  status: ComponentStatus
  responseTime?: number
  lastChecked: string
  details?: Record<string, any>
}

export type ComponentStatus = 'online' | 'offline' | 'warning' | 'error'

// API types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  message?: string
  errors?: ApiError[]
  meta?: ApiMeta
}

export interface ApiError {
  code: string
  message: string
  field?: string
  details?: any
}

export interface ApiMeta {
  pagination?: PaginationMeta
  timestamp: string
  version: string
  requestId: string
}

export interface PaginationMeta {
  page: number
  pageSize: number
  totalItems: number
  totalPages: number
  hasNext: boolean
  hasPrevious: boolean
}

export interface PaginationParams {
  page?: number
  pageSize?: number
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
  filters?: Record<string, any>
}

// UI types
export interface SelectOption {
  value: string | number
  label: string
  disabled?: boolean
  icon?: React.ComponentType<any>
}

export interface TableColumn<T = any> {
  key: string
  title: string
  dataIndex?: keyof T
  render?: (value: any, record: T, index: number) => React.ReactNode
  sortable?: boolean
  filterable?: boolean
  width?: number | string
  align?: 'left' | 'center' | 'right'
}

export interface TableProps<T = any> {
  columns: TableColumn<T>[]
  data: T[]
  loading?: boolean
  pagination?: PaginationMeta
  onPageChange?: (page: number) => void
  onSort?: (field: string, order: 'asc' | 'desc') => void
  onFilter?: (filters: Record<string, any>) => void
  rowKey?: string | ((record: T) => string)
  selectedRows?: string[]
  onSelectionChange?: (selectedRows: string[]) => void
}

export interface ChartData {
  labels: string[]
  datasets: ChartDataset[]
}

export interface ChartDataset {
  label: string
  data: number[]
  backgroundColor?: string | string[]
  borderColor?: string | string[]
  borderWidth?: number
  fill?: boolean
}

// Form types
export interface FormField {
  name: string
  label: string
  type: FieldType
  required?: boolean
  placeholder?: string
  options?: SelectOption[]
  validation?: ValidationRule[]
  dependencies?: FormDependency[]
}

export type FieldType = 'text' | 'email' | 'password' | 'number' | 'select' | 'multiselect' | 'textarea' | 'checkbox' | 'radio' | 'file' | 'date' | 'datetime'

export interface ValidationRule {
  type: ValidationType
  value?: any
  message: string
}

export type ValidationType = 'required' | 'minLength' | 'maxLength' | 'pattern' | 'custom'

export interface FormDependency {
  field: string
  condition: DependencyCondition
  value: any
  action: DependencyAction
}

export type DependencyCondition = 'equals' | 'notEquals' | 'contains' | 'greaterThan' | 'lessThan'
export type DependencyAction = 'show' | 'hide' | 'enable' | 'disable' | 'required'

// Notification types
export interface Notification extends BaseEntity {
  title: string
  message: string
  type: NotificationType
  read: boolean
  userId: string
  data?: Record<string, any>
  expiresAt?: string
}

export type NotificationType = 'info' | 'success' | 'warning' | 'error' | 'system'

// Search types
export interface SearchRequest {
  query: string
  filters?: SearchFilters
  pagination?: PaginationParams
  highlight?: boolean
}

export interface SearchFilters {
  documentType?: string[]
  category?: string[]
  dateRange?: {
    from: string
    to: string
  }
  confidence?: {
    min: number
    max: number
  }
  tags?: string[]
  status?: string[]
}

export interface SearchResult {
  id: string
  title: string
  content: string
  highlights?: string[]
  score: number
  metadata: Record<string, any>
}

export interface SearchResponse {
  documents: Document[]
  total: number
  aggregations?: Record<string, any>
  suggestions?: string[]
  took: number
}

// Event types
export interface SystemEvent {
  id: string
  type: EventType
  source: string
  timestamp: string
  data: Record<string, any>
  userId?: string
  sessionId?: string
}

export type EventType = 'user_action' | 'system_event' | 'error' | 'performance' | 'security'

// Scraping types
export interface ScrapingStatus {
  is_running: boolean
  documents_scraped: number
  errors?: string[]
  last_scrape_time?: string
  current_source?: string
  started_at?: string
}

// Constants
export const IRANIAN_LEGAL_SOURCES = {
  'divan-edari': 'دیوان عدالت اداری',
  'divanali': 'دیوان عالی کشور',
  'qanoon': 'مرکز قوانین و مقررات',
  'majlis': 'مجلس شورای اسلامی',
  'dadgostari': 'دادگستری',
  'other': 'سایر منابع'
} as const;

export type LegalSourceKey = keyof typeof IRANIAN_LEGAL_SOURCES;

// Configuration types
export interface AppConfig {
  app: {
    name: string
    version: string
    environment: string
    baseUrl: string
  }
  api: {
    baseUrl: string
    timeout: number
    retries: number
  }
  upload: {
    maxFileSize: number
    allowedTypes: string[]
    uploadPath: string
  }
  classification: {
    defaultModel: string
    confidenceThreshold: number
    maxTextLength: number
  }
  security: {
    jwtSecret: string
    sessionTimeout: number
    maxLoginAttempts: number
  }
  features: {
    registration: boolean
    fileUpload: boolean
    batchProcessing: boolean
    analytics: boolean
  }
}