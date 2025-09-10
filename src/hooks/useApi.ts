import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Generic API hook
export function useApi<T>(
  key: string | string[],
  fetcher: () => Promise<T>,
  options?: {
    enabled?: boolean
    refetchInterval?: number
  }
) {
  return useQuery({
    queryKey: Array.isArray(key) ? key : [key],
    queryFn: fetcher,
    enabled: options?.enabled,
    refetchInterval: options?.refetchInterval,
  })
}

// System health check
export function useSystemHealth() {
  return useApi('system-health', async () => {
    const response = await api.get('/system/health')
    return response.data
  }, {
    refetchInterval: 30000, // Check every 30 seconds
  })
}

// Document operations
export function useDocuments() {
  return useApi('documents', async () => {
    const response = await api.get('/documents')
    return response.data
  })
}

export function useDocument(id: string) {
  return useApi(['document', id], async () => {
    const response = await api.get(`/documents/${id}`)
    return response.data
  }, {
    enabled: !!id,
  })
}

export function useUploadDocument() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await api.post('/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })
}

// Classification operations
export function useClassifyText() {
  return useMutation({
    mutationFn: async (data: { text: string; options?: any }) => {
      const response = await api.post('/ai/classify', data)
      return response.data
    },
  })
}

export function useClassificationHistory() {
  return useApi('classification-history', async () => {
    const response = await api.get('/ai/classification/history')
    return response.data
  })
}

export function useBatchClassification() {
  return useMutation({
    mutationFn: async (data: { documents: string[]; options?: any }) => {
      const response = await api.post('/ai/classify/batch', data)
      return response.data
    },
  })
}

// Training operations
export function useTrainingStatus() {
  return useApi('training-status', async () => {
    const response = await api.get('/ai/training/status')
    return response.data
  }, {
    refetchInterval: 5000, // Check every 5 seconds during training
  })
}

export function useStartTraining() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (data: { dataset: string; config?: any }) => {
      const response = await api.post('/ai/training/start', data)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-status'] })
    },
  })
}

// Analytics operations
export function useAnalytics(period: string = '30d') {
  return useApi(['analytics', period], async () => {
    const response = await api.get(`/analytics?period=${period}`)
    return response.data
  })
}

export function usePerformanceMetrics() {
  return useApi('performance-metrics', async () => {
    const response = await api.get('/analytics/performance')
    return response.data
  }, {
    refetchInterval: 60000, // Refresh every minute
  })
}

// Settings operations
export function useSettings() {
  return useApi('settings', async () => {
    const response = await api.get('/settings')
    return response.data
  })
}

export function useUpdateSettings() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (data: any) => {
      const response = await api.put('/settings', data)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
    },
  })
}

// Search operations
export function useSearch(query: string, filters?: any) {
  return useApi(['search', query, filters], async () => {
    const response = await api.post('/search', { query, filters })
    return response.data
  }, {
    enabled: !!query && query.length > 2,
  })
}

export { api }