'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  Database, 
  FileText, 
  Search, 
  Upload, 
  BarChart3, 
  Settings, 
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Users,
  Zap,
  TrendingUp,
  Shield,
  Server,
  Cpu,
  HardDrive,
  Wifi,
  RefreshCw
} from 'lucide-react'
import { toast, Toaster } from 'react-hot-toast'

// Types
interface SystemHealth {
  status: 'healthy' | 'degraded' | 'critical'
  timestamp: string
  database_connected: boolean
  ai_model_loaded: boolean
  version: string
  uptime: string
  memory_usage: number
  cpu_usage: number
  disk_usage: number
}

interface ClassificationResult {
  text: string
  classification: Record<string, number>
  confidence: number
  predicted_class: string
  timestamp: string
}

interface DocumentStats {
  total_documents: number
  total_size: string
  last_updated: string
  categories: Record<string, number>
}

interface ActivityLog {
  id: string
  type: 'classification' | 'upload' | 'search' | 'training'
  description: string
  timestamp: string
  status: 'success' | 'error' | 'warning'
}

// API Service
class ApiService {
  private baseUrl: string

  constructor() {
    this.baseUrl = process.env.NODE_ENV === 'production' 
      ? 'https://persian-legal-ai-backend.vercel.app'
      : 'http://localhost:8000'
  }

  async fetchSystemHealth(): Promise<SystemHealth> {
    try {
      const response = await fetch(`${this.baseUrl}/api/system/health`)
      if (!response.ok) throw new Error('Failed to fetch system health')
      return await response.json()
    } catch (error) {
      console.error('System health fetch error:', error)
      throw error
    }
  }

  async classifyText(text: string): Promise<ClassificationResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/ai/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, include_confidence: true })
      })
      if (!response.ok) throw new Error('Classification failed')
      return await response.json()
    } catch (error) {
      console.error('Classification error:', error)
      throw error
    }
  }

  async getDocumentStats(): Promise<DocumentStats> {
    try {
      const response = await fetch(`${this.baseUrl}/api/documents/stats`)
      if (!response.ok) throw new Error('Failed to fetch document stats')
      return await response.json()
    } catch (error) {
      console.error('Document stats fetch error:', error)
      throw error
    }
  }
}

// Main Dashboard Component
export default function CompletePersianAIDashboard() {
  // State Management
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null)
  const [documentStats, setDocumentStats] = useState<DocumentStats | null>(null)
  const [classificationText, setClassificationText] = useState('')
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [refreshing, setRefreshing] = useState(false)
  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([])

  const apiService = new ApiService()

  // Effects
  useEffect(() => {
    initializeDashboard()
    const interval = setInterval(refreshSystemHealth, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  // Functions
  const initializeDashboard = async () => {
    setRefreshing(true)
    try {
      await Promise.all([
        refreshSystemHealth(),
        refreshDocumentStats()
      ])
      toast.success('Dashboard بارگذاری شد', { duration: 2000 })
    } catch (error) {
      toast.error('خطا در بارگذاری Dashboard')
    } finally {
      setRefreshing(false)
    }
  }

  const refreshSystemHealth = async () => {
    try {
      const health = await apiService.fetchSystemHealth()
      setSystemHealth(health)
    } catch (error) {
      console.error('Failed to refresh system health:', error)
      // Fallback mock data for demo
      setSystemHealth({
        status: 'degraded',
        timestamp: new Date().toISOString(),
        database_connected: false,
        ai_model_loaded: false,
        version: '2.0.0',
        uptime: '2 hours',
        memory_usage: 67,
        cpu_usage: 34,
        disk_usage: 45
      })
    }
  }

  const refreshDocumentStats = async () => {
    try {
      const stats = await apiService.getDocumentStats()
      setDocumentStats(stats)
    } catch (error) {
      console.error('Failed to refresh document stats:', error)
      // Fallback mock data
      setDocumentStats({
        total_documents: 1250,
        total_size: '45.7 MB',
        last_updated: new Date().toISOString(),
        categories: {
          'قرارداد': 450,
          'قانون': 320,
          'رأی دادگاه': 280,
          'آیین‌نامه': 200
        }
      })
    }
  }

  const handleClassification = async () => {
    if (!classificationText.trim()) {
      toast.error('لطفاً متنی وارد کنید')
      return
    }

    setLoading(true)
    try {
      const result = await apiService.classifyText(classificationText)
      setClassificationResult(result)
      
      // Add to activity log
      addActivityLog({
        type: 'classification',
        description: `طبقه‌بندی متن: ${result.predicted_class}`,
        status: 'success'
      })
      
      toast.success('طبقه‌بندی با موفقیت انجام شد')
    } catch (error) {
      toast.error('خطا در طبقه‌بندی متن')
      addActivityLog({
        type: 'classification',
        description: 'خطا در طبقه‌بندی متن',
        status: 'error'
      })
    } finally {
      setLoading(false)
    }
  }

  const addActivityLog = (activity: Omit<ActivityLog, 'id' | 'timestamp'>) => {
    const newActivity: ActivityLog = {
      ...activity,
      id: Date.now().toString(),
      timestamp: new Date().toISOString()
    }
    setActivityLogs(prev => [newActivity, ...prev.slice(0, 9)]) // Keep last 10
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500 bg-green-50'
      case 'degraded': return 'text-yellow-500 bg-yellow-50'
      case 'critical': return 'text-red-500 bg-red-50'
      default: return 'text-gray-500 bg-gray-50'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-5 w-5" />
      case 'degraded': return <AlertCircle className="h-5 w-5" />
      case 'critical': return <AlertCircle className="h-5 w-5" />
      default: return <Clock className="h-5 w-5" />
    }
  }

  // Render Functions
  const renderSystemStatus = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {/* System Health */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white p-6 rounded-lg shadow-md border"
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">وضعیت سیستم</h3>
            <p className={`text-sm font-medium px-2 py-1 rounded-full ${getStatusColor(systemHealth?.status || 'unknown')}`}>
              {systemHealth?.status?.toUpperCase() || 'UNKNOWN'}
            </p>
          </div>
          <div className={getStatusColor(systemHealth?.status || 'unknown')}>
            {getStatusIcon(systemHealth?.status || 'unknown')}
          </div>
        </div>
      </motion.div>

      {/* Database Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white p-6 rounded-lg shadow-md border"
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">پایگاه داده</h3>
            <p className="text-sm text-gray-500">
              {systemHealth?.database_connected ? 'متصل' : 'قطع شده'}
            </p>
          </div>
          <Database className={`h-8 w-8 ${
            systemHealth?.database_connected ? 'text-green-500' : 'text-red-500'
          }`} />
        </div>
      </motion.div>

      {/* AI Model Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white p-6 rounded-lg shadow-md border"
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">مدل هوش مصنوعی</h3>
            <p className="text-sm text-gray-500">
              {systemHealth?.ai_model_loaded ? 'بارگذاری شده' : 'غیرفعال'}
            </p>
          </div>
          <Brain className={`h-8 w-8 ${
            systemHealth?.ai_model_loaded ? 'text-green-500' : 'text-red-500'
          }`} />
        </div>
      </motion.div>

      {/* Version Info */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-white p-6 rounded-lg shadow-md border"
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">نسخه سیستم</h3>
            <p className="text-sm text-gray-500">
              {systemHealth?.version || 'نامشخص'}
            </p>
          </div>
          <Settings className="h-8 w-8 text-indigo-500" />
        </div>
      </motion.div>
    </div>
  )

  const renderClassificationInterface = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-md p-6 mb-8"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-900">طبقه‌بندی متن حقوقی</h2>
        <Brain className="h-6 w-6 text-indigo-500" />
      </div>
      
      <div className="space-y-4">
        <textarea
          value={classificationText}
          onChange={(e) => setClassificationText(e.target.value)}
          placeholder="متن حقوقی خود را برای طبقه‌بندی وارد کنید..."
          className="w-full h-32 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
          dir="rtl"
        />
        
        <button
          onClick={handleClassification}
          disabled={loading || !classificationText.trim()}
          className="bg-indigo-600 text-white px-6 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
        >
          {loading ? (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
          ) : (
            <Search className="h-4 w-4" />
          )}
          <span className="mr-2">{loading ? 'در حال پردازش...' : 'طبقه‌بندی'}</span>
        </button>
      </div>

      {/* Classification Results */}
      <AnimatePresence>
        {classificationResult && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-6 p-4 bg-gray-50 rounded-md border"
          >
            <h3 className="text-lg font-medium text-gray-900 mb-3">نتایج طبقه‌بندی</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 mb-2">
                  <strong>دسته پیش‌بینی شده:</strong> {classificationResult.predicted_class}
                </p>
                <p className="text-sm text-gray-600">
                  <strong>اطمینان:</strong> {(classificationResult.confidence * 100).toFixed(1)}%
                </p>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">توزیع احتمالات:</h4>
                <div className="space-y-1">
                  {Object.entries(classificationResult.classification).map(([category, score]) => (
                    <div key={category} className="flex items-center space-x-2">
                      <span className="text-xs w-16 text-gray-600">{category}:</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${score * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-600 w-12">
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )

  const renderActivityLogs = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-md p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-900">فعالیت‌های اخیر</h2>
        <Activity className="h-6 w-6 text-indigo-500" />
      </div>
      
      <div className="space-y-3">
        {activityLogs.length > 0 ? (
          activityLogs.map((log) => (
            <div key={log.id} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-md">
              <div className={`w-2 h-2 rounded-full ${
                log.status === 'success' ? 'bg-green-500' :
                log.status === 'error' ? 'bg-red-500' : 'bg-yellow-500'
              }`}></div>
              <div className="flex-1">
                <p className="text-sm text-gray-900">{log.description}</p>
                <p className="text-xs text-gray-500">
                  {new Date(log.timestamp).toLocaleString('fa-IR')}
                </p>
              </div>
            </div>
          ))
        ) : (
          <p className="text-gray-500 text-center py-4">هیچ فعالیتی ثبت نشده است</p>
        )}
      </div>
    </motion.div>
  )

  // Main Render
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100" dir="rtl">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-indigo-600" />
              <h1 className="text-2xl font-bold text-gray-900 mr-3">
                سامانه هوش مصنوعی حقوقی فارسی
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={initializeDashboard}
                disabled={refreshing}
                className="p-2 text-gray-500 hover:text-gray-700 disabled:opacity-50"
              >
                <RefreshCw className={`h-5 w-5 ${refreshing ? 'animate-spin' : ''}`} />
              </button>
              
              {systemHealth && (
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    systemHealth.status === 'healthy' ? 'bg-green-500' : 
                    systemHealth.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}></div>
                  <span className={`text-sm font-medium ${getStatusColor(systemHealth.status).split(' ')[0]}`}>
                    {systemHealth.status.toUpperCase()}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderSystemStatus()}
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            {renderClassificationInterface()}
          </div>
          <div>
            {renderActivityLogs()}
          </div>
        </div>
      </main>
    </div>
  )
}