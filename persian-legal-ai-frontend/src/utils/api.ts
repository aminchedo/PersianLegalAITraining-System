import axios, { AxiosResponse, AxiosError } from 'axios'

// API Base Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      if (typeof window !== 'undefined') {
        localStorage.removeItem('authToken')
        window.location.href = '/auth/login'
      }
    }
    return Promise.reject(error)
  }
)

// API Response Types
export interface ApiResponse<T = any> {
  data: T
  message?: string
  status: 'success' | 'error'
}

export interface SystemHealth {
  status: 'healthy' | 'unhealthy'
  database_connected: boolean
  ai_model_loaded: boolean
  version: string
  uptime: string
  memory_usage: number
  cpu_usage: number
}

export interface ClassificationRequest {
  text: string
  model_type?: string
}

export interface ClassificationResponse {
  text: string
  classification: { [key: string]: number }
  confidence: number
  predicted_class: string
  processing_time: number
  model_version: string
}

export interface Document {
  id: string
  title: string
  content: string
  type: 'contract' | 'law' | 'regulation' | 'judgment' | 'other'
  status: 'processed' | 'processing' | 'pending' | 'error'
  classification?: string
  confidence?: number
  created_at: string
  updated_at: string
  file_size: number
  file_type: string
  metadata?: any
}

export interface TrainingSession {
  id: string
  name: string
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed'
  progress: number
  created_at: string
  started_at?: string
  completed_at?: string
  metrics?: {
    accuracy: number
    loss: number
    f1_score: number
    precision: number
    recall: number
  }
  hyperparameters: {
    learning_rate: number
    batch_size: number
    epochs: number
    max_length: number
  }
  model_type: string
  dataset_size: number
  current_epoch?: number
  estimated_time_remaining?: string
  logs?: TrainingLog[]
}

export interface TrainingLog {
  timestamp: string
  epoch: number
  step: number
  loss: number
  accuracy: number
  message: string
  level: 'info' | 'warning' | 'error'
}

export interface CreateTrainingSessionRequest {
  name: string
  model_type: string
  hyperparameters: {
    learning_rate: number
    batch_size: number
    epochs: number
    max_length: number
  }
  dataset_ids: string[]
}

// API Functions

// System APIs
export const getSystemHealth = async (): Promise<SystemHealth> => {
  const response = await apiClient.get<ApiResponse<SystemHealth>>('/system/health')
  return response.data.data
}

export const getSystemStats = async () => {
  const response = await apiClient.get<ApiResponse>('/system/stats')
  return response.data.data
}

// Classification APIs
export const classifyText = async (request: ClassificationRequest): Promise<ClassificationResponse> => {
  const response = await apiClient.post<ApiResponse<ClassificationResponse>>('/ai/classify', request)
  return response.data.data
}

export const batchClassify = async (texts: string[]) => {
  const response = await apiClient.post<ApiResponse>('/ai/classify/batch', { texts })
  return response.data.data
}

export const getClassificationHistory = async (page = 1, limit = 20) => {
  const response = await apiClient.get<ApiResponse>(`/ai/classify/history?page=${page}&limit=${limit}`)
  return response.data.data
}

// Document APIs
export const getDocuments = async (page = 1, limit = 20, type?: string, status?: string) => {
  const params = new URLSearchParams({
    page: page.toString(),
    limit: limit.toString(),
  })
  
  if (type) params.append('type', type)
  if (status) params.append('status', status)
  
  const response = await apiClient.get<ApiResponse<{ documents: Document[], total: number, page: number }>>(`/documents?${params}`)
  return response.data.data
}

export const getDocument = async (id: string): Promise<Document> => {
  const response = await apiClient.get<ApiResponse<Document>>(`/documents/${id}`)
  return response.data.data
}

export const uploadDocument = async (file: File, metadata?: any) => {
  const formData = new FormData()
  formData.append('file', file)
  if (metadata) {
    formData.append('metadata', JSON.stringify(metadata))
  }
  
  const response = await apiClient.post<ApiResponse>('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data.data
}

export const deleteDocument = async (id: string) => {
  const response = await apiClient.delete<ApiResponse>(`/documents/${id}`)
  return response.data.data
}

export const searchDocuments = async (query: string, filters?: any) => {
  const response = await apiClient.post<ApiResponse>('/documents/search', { query, filters })
  return response.data.data
}

// Training APIs
export const getTrainingSessions = async (page = 1, limit = 20) => {
  const response = await apiClient.get<ApiResponse<{ sessions: TrainingSession[], total: number, page: number }>>(`/training/sessions?page=${page}&limit=${limit}`)
  return response.data.data
}

export const getTrainingSession = async (id: string): Promise<TrainingSession> => {
  const response = await apiClient.get<ApiResponse<TrainingSession>>(`/training/sessions/${id}`)
  return response.data.data
}

export const createTrainingSession = async (request: CreateTrainingSessionRequest): Promise<TrainingSession> => {
  const response = await apiClient.post<ApiResponse<TrainingSession>>('/training/sessions', request)
  return response.data.data
}

export const startTrainingSession = async (id: string) => {
  const response = await apiClient.post<ApiResponse>(`/training/sessions/${id}/start`)
  return response.data.data
}

export const pauseTrainingSession = async (id: string) => {
  const response = await apiClient.post<ApiResponse>(`/training/sessions/${id}/pause`)
  return response.data.data
}

export const stopTrainingSession = async (id: string) => {
  const response = await apiClient.post<ApiResponse>(`/training/sessions/${id}/stop`)
  return response.data.data
}

export const getTrainingLogs = async (sessionId: string, page = 1, limit = 50) => {
  const response = await apiClient.get<ApiResponse<{ logs: TrainingLog[], total: number }>>(`/training/sessions/${sessionId}/logs?page=${page}&limit=${limit}`)
  return response.data.data
}

// Model APIs
export const getModels = async () => {
  const response = await apiClient.get<ApiResponse>('/models')
  return response.data.data
}

export const getModel = async (id: string) => {
  const response = await apiClient.get<ApiResponse>(`/models/${id}`)
  return response.data.data
}

export const activateModel = async (id: string) => {
  const response = await apiClient.post<ApiResponse>(`/models/${id}/activate`)
  return response.data.data
}

export const deactivateModel = async (id: string) => {
  const response = await apiClient.post<ApiResponse>(`/models/${id}/deactivate`)
  return response.data.data
}

export const deleteModel = async (id: string) => {
  const response = await apiClient.delete<ApiResponse>(`/models/${id}`)
  return response.data.data
}

// Analytics APIs
export const getAnalytics = async (timeRange = '7d') => {
  const response = await apiClient.get<ApiResponse>(`/analytics?range=${timeRange}`)
  return response.data.data
}

export const getPerformanceMetrics = async (modelId?: string) => {
  const params = modelId ? `?model_id=${modelId}` : ''
  const response = await apiClient.get<ApiResponse>(`/analytics/performance${params}`)
  return response.data.data
}

export const generateReport = async (type: string, filters?: any) => {
  const response = await apiClient.post<ApiResponse>('/analytics/reports', { type, filters })
  return response.data.data
}

// User APIs
export const getUsers = async (page = 1, limit = 20) => {
  const response = await apiClient.get<ApiResponse>(`/users?page=${page}&limit=${limit}`)
  return response.data.data
}

export const getUser = async (id: string) => {
  const response = await apiClient.get<ApiResponse>(`/users/${id}`)
  return response.data.data
}

export const updateUser = async (id: string, data: any) => {
  const response = await apiClient.put<ApiResponse>(`/users/${id}`, data)
  return response.data.data
}

export const deleteUser = async (id: string) => {
  const response = await apiClient.delete<ApiResponse>(`/users/${id}`)
  return response.data.data
}

// Settings APIs
export const getSettings = async () => {
  const response = await apiClient.get<ApiResponse>('/settings')
  return response.data.data
}

export const updateSettings = async (settings: any) => {
  const response = await apiClient.put<ApiResponse>('/settings', settings)
  return response.data.data
}

// Export the API client for custom requests
export default apiClient