import React, { useState, useEffect } from 'react'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import { Button } from '../../src/components/ui/Button'
import { Input } from '../../src/components/ui/Input'
import { Badge } from '../../src/components/ui/Badge'
import {
  AcademicCapIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  ChartBarIcon,
  ClockIcon,
  CpuChipIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  PlusIcon
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'

interface TrainingSession {
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
}

export default function TrainingDashboard() {
  const [sessions, setSessions] = useState<TrainingSession[]>([])
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)

  // Mock training sessions data
  useEffect(() => {
    const mockSessions: TrainingSession[] = [
      {
        id: 'session-1',
        name: 'آموزش مدل طبقه‌بندی قراردادها',
        status: 'running',
        progress: 65,
        created_at: '2024-01-15T10:30:00Z',
        started_at: '2024-01-15T10:35:00Z',
        metrics: {
          accuracy: 0.87,
          loss: 0.23,
          f1_score: 0.85,
          precision: 0.89,
          recall: 0.82
        },
        hyperparameters: {
          learning_rate: 2e-5,
          batch_size: 16,
          epochs: 10,
          max_length: 512
        },
        model_type: 'BERT-Persian',
        dataset_size: 5000,
        current_epoch: 7,
        estimated_time_remaining: '45 دقیقه'
      },
      {
        id: 'session-2',
        name: 'آموزش مدل شناسایی احکام قضایی',
        status: 'completed',
        progress: 100,
        created_at: '2024-01-14T09:00:00Z',
        started_at: '2024-01-14T09:05:00Z',
        completed_at: '2024-01-14T11:30:00Z',
        metrics: {
          accuracy: 0.92,
          loss: 0.15,
          f1_score: 0.90,
          precision: 0.93,
          recall: 0.88
        },
        hyperparameters: {
          learning_rate: 1e-5,
          batch_size: 32,
          epochs: 15,
          max_length: 1024
        },
        model_type: 'RoBERTa-Persian',
        dataset_size: 8000
      },
      {
        id: 'session-3',
        name: 'آموزش مدل تحلیل متون حقوقی',
        status: 'paused',
        progress: 30,
        created_at: '2024-01-16T14:00:00Z',
        started_at: '2024-01-16T14:05:00Z',
        metrics: {
          accuracy: 0.71,
          loss: 0.45,
          f1_score: 0.68,
          precision: 0.74,
          recall: 0.65
        },
        hyperparameters: {
          learning_rate: 3e-5,
          batch_size: 8,
          epochs: 20,
          max_length: 768
        },
        model_type: 'GPT-Persian',
        dataset_size: 12000,
        current_epoch: 6
      }
    ]

    setTimeout(() => {
      setSessions(mockSessions)
      setLoading(false)
    }, 1000)
  }, [])

  const handleSessionControl = (sessionId: string, action: 'start' | 'pause' | 'stop') => {
    setSessions(prev => prev.map(session => {
      if (session.id === sessionId) {
        switch (action) {
          case 'start':
            return { ...session, status: 'running' as const }
          case 'pause':
            return { ...session, status: 'paused' as const }
          case 'stop':
            return { ...session, status: 'completed' as const, progress: 100 }
          default:
            return session
        }
      }
      return session
    }))
  }

  const getStatusBadge = (status: TrainingSession['status']) => {
    switch (status) {
      case 'running':
        return <Badge variant="primary">در حال اجرا</Badge>
      case 'completed':
        return <Badge variant="success">تکمیل شده</Badge>
      case 'paused':
        return <Badge variant="warning">متوقف شده</Badge>
      case 'failed':
        return <Badge variant="error">ناموفق</Badge>
      default:
        return <Badge variant="secondary">در انتظار</Badge>
    }
  }

  const getStatusIcon = (status: TrainingSession['status']) => {
    switch (status) {
      case 'running':
        return <PlayIcon className="h-4 w-4 text-blue-600" />
      case 'completed':
        return <CheckCircleIcon className="h-4 w-4 text-green-600" />
      case 'paused':
        return <PauseIcon className="h-4 w-4 text-yellow-600" />
      case 'failed':
        return <ExclamationTriangleIcon className="h-4 w-4 text-red-600" />
      default:
        return <ClockIcon className="h-4 w-4 text-gray-600" />
    }
  }

  const runningSessionsCount = sessions.filter(s => s.status === 'running').length
  const completedSessionsCount = sessions.filter(s => s.status === 'completed').length
  const avgAccuracy = sessions
    .filter(s => s.metrics)
    .reduce((acc, s) => acc + (s.metrics?.accuracy || 0), 0) / 
    sessions.filter(s => s.metrics).length

  return (
    <MainLayout title="آموزش مدل - سامانه هوش مصنوعی حقوقی فارسی">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="heading-2 text-gray-900 flex items-center">
              <AcademicCapIcon className="h-8 w-8 text-indigo-600 ml-3" />
              آموزش مدل
            </h1>
            <p className="paragraph-normal text-gray-600">
              مدیریت و نظارت بر جلسات آموزش مدل‌های هوش مصنوعی
            </p>
          </div>
          <Button 
            variant="primary" 
            onClick={() => setShowCreateModal(true)}
            className="flex items-center"
          >
            <PlusIcon className="h-4 w-4 ml-2" />
            جلسه آموزش جدید
          </Button>
        </div>

        {/* Training Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <PlayIcon className="h-8 w-8 text-blue-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {runningSessionsCount}
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    جلسات فعال
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <CheckCircleIcon className="h-8 w-8 text-green-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {completedSessionsCount}
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    جلسات تکمیل شده
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ChartBarIcon className="h-8 w-8 text-purple-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {(avgAccuracy * 100).toFixed(1)}%
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    میانگین دقت
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Training Sessions */}
        <Card>
          <CardHeader>
            <CardTitle>جلسات آموزش</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                <span className="mr-3 text-persian-primary ui-text-medium">در حال بارگذاری...</span>
              </div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-8">
                <AcademicCapIcon className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900 text-persian-primary">
                  هیچ جلسه آموزشی یافت نشد
                </h3>
                <p className="mt-1 text-sm text-gray-500 text-persian-primary">
                  برای شروع، جلسه آموزش جدید ایجاد کنید.
                </p>
                <div className="mt-6">
                  <Button variant="primary" onClick={() => setShowCreateModal(true)}>
                    <PlusIcon className="h-4 w-4 ml-2" />
                    ایجاد جلسه آموزش
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {sessions.map((session) => (
                  <motion.div
                    key={session.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3 space-x-reverse">
                        {getStatusIcon(session.status)}
                        <div>
                          <h3 className="text-lg font-medium text-gray-900 text-persian-primary">
                            {session.name}
                          </h3>
                          <div className="flex items-center space-x-2 space-x-reverse mt-1">
                            {getStatusBadge(session.status)}
                            <span className="text-sm text-gray-500 text-persian-primary">
                              {session.model_type}
                            </span>
                            <span className="text-sm text-gray-500 text-persian-primary">
                              {session.dataset_size.toLocaleString('fa')} نمونه
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2 space-x-reverse">
                        {session.status === 'running' && (
                          <>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleSessionControl(session.id, 'pause')}
                            >
                              <PauseIcon className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleSessionControl(session.id, 'stop')}
                            >
                              <StopIcon className="h-4 w-4" />
                            </Button>
                          </>
                        )}
                        {session.status === 'paused' && (
                          <Button
                            variant="primary"
                            size="sm"
                            onClick={() => handleSessionControl(session.id, 'start')}
                          >
                            <PlayIcon className="h-4 w-4" />
                          </Button>
                        )}
                        <Button variant="ghost" size="sm">
                          جزئیات
                        </Button>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="mt-4">
                      <div className="flex justify-between text-sm text-gray-600 text-persian-primary mb-1">
                        <span>پیشرفت آموزش</span>
                        <span>{session.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <motion.div
                          className="bg-blue-600 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${session.progress}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                      {session.current_epoch && (
                        <div className="flex justify-between text-xs text-gray-500 text-persian-primary mt-1">
                          <span>
                            دوره {session.current_epoch} از {session.hyperparameters.epochs}
                          </span>
                          {session.estimated_time_remaining && (
                            <span>زمان باقی‌مانده: {session.estimated_time_remaining}</span>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Metrics */}
                    {session.metrics && (
                      <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="text-center">
                          <div className="text-lg font-semibold text-gray-900 text-persian-primary">
                            {(session.metrics.accuracy * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500 text-persian-primary">دقت</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-semibold text-gray-900 text-persian-primary">
                            {session.metrics.loss.toFixed(3)}
                          </div>
                          <div className="text-xs text-gray-500 text-persian-primary">خطا</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-semibold text-gray-900 text-persian-primary">
                            {(session.metrics.f1_score * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500 text-persian-primary">F1</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-semibold text-gray-900 text-persian-primary">
                            {(session.metrics.precision * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500 text-persian-primary">دقت مثبت</div>
                        </div>
                        <div className="text-center">
                          <div className="text-lg font-semibold text-gray-900 text-persian-primary">
                            {(session.metrics.recall * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-gray-500 text-persian-primary">بازخوانی</div>
                        </div>
                      </div>
                    )}

                    {/* Hyperparameters */}
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <h4 className="text-sm font-medium text-gray-900 text-persian-primary mb-2">
                        پارامترهای آموزش
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600 text-persian-primary">
                        <div>
                          <span className="font-medium">نرخ یادگیری:</span> {session.hyperparameters.learning_rate}
                        </div>
                        <div>
                          <span className="font-medium">اندازه دسته:</span> {session.hyperparameters.batch_size}
                        </div>
                        <div>
                          <span className="font-medium">تعداد دوره:</span> {session.hyperparameters.epochs}
                        </div>
                        <div>
                          <span className="font-medium">حداکثر طول:</span> {session.hyperparameters.max_length}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Create Session Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
            <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
              <div className="mt-3">
                <h3 className="text-lg font-medium text-gray-900 text-persian-primary mb-4">
                  ایجاد جلسه آموزش جدید
                </h3>
                <div className="space-y-4">
                  <Input label="نام جلسه" placeholder="نام جلسه آموزش را وارد کنید" />
                  <Input label="نوع مدل" placeholder="BERT-Persian" />
                  <Input label="نرخ یادگیری" placeholder="2e-5" type="number" />
                  <Input label="اندازه دسته" placeholder="16" type="number" />
                  <Input label="تعداد دوره" placeholder="10" type="number" />
                </div>
                <div className="flex justify-end space-x-2 space-x-reverse mt-6">
                  <Button 
                    variant="outline" 
                    onClick={() => setShowCreateModal(false)}
                  >
                    لغو
                  </Button>
                  <Button variant="primary">
                    شروع آموزش
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </MainLayout>
  )
}