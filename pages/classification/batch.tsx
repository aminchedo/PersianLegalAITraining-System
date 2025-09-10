'use client'

import React, { useState } from 'react'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import Button from '../../src/components/ui/Button'
import Badge from '../../src/components/ui/Badge'
import Loading from '../../src/components/ui/Loading'
import Alert from '../../src/components/ui/Alert'
import {
  CpuChipIcon,
  DocumentTextIcon,
  PlayIcon,
  PauseIcon,
  StopIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import { motion, AnimatePresence } from 'framer-motion'

interface BatchJob {
  id: string
  name: string
  totalDocuments: number
  processedDocuments: number
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed'
  startTime?: string
  endTime?: string
  results?: {
    successful: number
    failed: number
    accuracy: number
  }
}

export default function BatchProcessingPage() {
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([])
  const [batchJobs, setBatchJobs] = useState<BatchJob[]>([
    {
      id: '1',
      name: 'پردازش اسناد قراردادی',
      totalDocuments: 150,
      processedDocuments: 150,
      status: 'completed',
      startTime: '1402/09/15 - 14:30',
      endTime: '1402/09/15 - 15:45',
      results: {
        successful: 147,
        failed: 3,
        accuracy: 94.2
      }
    },
    {
      id: '2',
      name: 'تحلیل اسناد قانونی',
      totalDocuments: 89,
      processedDocuments: 45,
      status: 'running',
      startTime: '1402/09/16 - 09:15',
    },
    {
      id: '3',
      name: 'طبقه‌بندی آیین‌نامه‌ها',
      totalDocuments: 67,
      processedDocuments: 0,
      status: 'pending',
    }
  ])

  const availableDocuments = [
    { id: '1', name: 'قرارداد خرید املاک.pdf', type: 'قرارداد' },
    { id: '2', name: 'آیین‌نامه داخلی.docx', type: 'آیین‌نامه' },
    { id: '3', name: 'قانون کار جدید.pdf', type: 'قانون' },
    { id: '4', name: 'رأی دیوان عدالت.pdf', type: 'رأی قضایی' },
    { id: '5', name: 'مقررات صادرات.doc', type: 'مقررات' },
  ]

  const handleDocumentSelect = (docId: string) => {
    setSelectedDocuments(prev => 
      prev.includes(docId) 
        ? prev.filter(id => id !== docId)
        : [...prev, docId]
    )
  }

  const handleSelectAll = () => {
    setSelectedDocuments(
      selectedDocuments.length === availableDocuments.length 
        ? [] 
        : availableDocuments.map(doc => doc.id)
    )
  }

  const startBatchProcessing = () => {
    if (selectedDocuments.length === 0) return

    const newJob: BatchJob = {
      id: Date.now().toString(),
      name: `پردازش دسته‌ای ${selectedDocuments.length} سند`,
      totalDocuments: selectedDocuments.length,
      processedDocuments: 0,
      status: 'running',
      startTime: new Date().toLocaleDateString('fa-IR') + ' - ' + new Date().toLocaleTimeString('fa-IR', { hour: '2-digit', minute: '2-digit' })
    }

    setBatchJobs(prev => [newJob, ...prev])
    setSelectedDocuments([])

    // Simulate processing
    simulateProcessing(newJob.id)
  }

  const simulateProcessing = (jobId: string) => {
    const interval = setInterval(() => {
      setBatchJobs(prev => 
        prev.map(job => {
          if (job.id === jobId && job.status === 'running') {
            const newProcessed = Math.min(job.processedDocuments + Math.floor(Math.random() * 3) + 1, job.totalDocuments)
            
            if (newProcessed >= job.totalDocuments) {
              clearInterval(interval)
              return {
                ...job,
                processedDocuments: newProcessed,
                status: 'completed',
                endTime: new Date().toLocaleDateString('fa-IR') + ' - ' + new Date().toLocaleTimeString('fa-IR', { hour: '2-digit', minute: '2-digit' }),
                results: {
                  successful: Math.floor(job.totalDocuments * 0.95),
                  failed: Math.floor(job.totalDocuments * 0.05),
                  accuracy: 95.2 + Math.random() * 3
                }
              }
            }
            
            return { ...job, processedDocuments: newProcessed }
          }
          return job
        })
      )
    }, 1000)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="success" size="sm">تکمیل شده</Badge>
      case 'running':
        return <Badge variant="warning" size="sm">در حال اجرا</Badge>
      case 'paused':
        return <Badge variant="secondary" size="sm">متوقف شده</Badge>
      case 'failed':
        return <Badge variant="danger" size="sm">ناموفق</Badge>
      default:
        return <Badge variant="info" size="sm">در انتظار</Badge>
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />
      case 'running':
        return <div className="h-5 w-5 animate-spin rounded-full border-2 border-yellow-500 border-t-transparent" />
      case 'failed':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
      default:
        return <CpuChipIcon className="h-5 w-5 text-gray-400" />
    }
  }

  return (
    <MainLayout
      title="پردازش دسته‌ای - سامانه هوش مصنوعی حقوقی فارسی"
      description="پردازش و طبقه‌بندی دسته‌ای اسناد حقوقی"
    >
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center">
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-gradient-to-br from-purple-500 to-purple-700 rounded-2xl">
              <CpuChipIcon className="h-12 w-12 text-white" />
            </div>
          </div>
          <h1 className="heading-1 text-gray-900">
            پردازش دسته‌ای اسناد
          </h1>
          <p className="paragraph-large text-gray-600 max-w-2xl mx-auto">
            چندین سند را همزمان انتخاب کنید و به صورت دسته‌ای طبقه‌بندی کنید
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Document Selection */}
          <div className="lg:col-span-2 space-y-6">
            {/* Available Documents */}
            <Card>
              <CardHeader>
                <div className="flex justify-between items-center">
                  <CardTitle>انتخاب اسناد برای پردازش</CardTitle>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSelectAll}
                  >
                    {selectedDocuments.length === availableDocuments.length ? 'لغو انتخاب همه' : 'انتخاب همه'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {availableDocuments.map((doc) => (
                    <div
                      key={doc.id}
                      className={`flex items-center justify-between p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedDocuments.includes(doc.id)
                          ? 'border-primary-500 bg-primary-50'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                      onClick={() => handleDocumentSelect(doc.id)}
                    >
                      <div className="flex items-center space-x-3 space-x-reverse">
                        <input
                          type="checkbox"
                          checked={selectedDocuments.includes(doc.id)}
                          onChange={() => handleDocumentSelect(doc.id)}
                          className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <DocumentTextIcon className="h-5 w-5 text-gray-400" />
                        <div>
                          <p className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                            {doc.name}
                          </p>
                          <p className="ui-text-small text-gray-500 text-persian-primary">
                            نوع: {doc.type}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {selectedDocuments.length > 0 && (
                  <div className="mt-6 pt-4 border-t border-gray-200">
                    <div className="flex justify-between items-center">
                      <span className="ui-text-medium text-gray-700 text-persian-primary">
                        {selectedDocuments.length} سند انتخاب شده
                      </span>
                      <Button
                        onClick={startBatchProcessing}
                        leftIcon={<PlayIcon className="h-5 w-5" />}
                      >
                        شروع پردازش دسته‌ای
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Batch Jobs */}
            <Card>
              <CardHeader>
                <CardTitle>تاریخچه پردازش‌های دسته‌ای</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <AnimatePresence>
                    {batchJobs.map((job) => (
                      <motion.div
                        key={job.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="border border-gray-200 rounded-lg p-4"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center space-x-3 space-x-reverse">
                            {getStatusIcon(job.status)}
                            <div>
                              <h3 className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                                {job.name}
                              </h3>
                              <p className="ui-text-small text-gray-500 text-persian-primary">
                                {job.totalDocuments} سند
                              </p>
                            </div>
                          </div>
                          {getStatusBadge(job.status)}
                        </div>

                        {/* Progress Bar */}
                        {(job.status === 'running' || job.status === 'completed') && (
                          <div className="mb-3">
                            <div className="flex justify-between items-center mb-1">
                              <span className="ui-text-small text-gray-600 text-persian-primary">
                                پیشرفت
                              </span>
                              <span className="ui-text-small text-gray-600 persian-numbers">
                                {job.processedDocuments} از {job.totalDocuments}
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${(job.processedDocuments / job.totalDocuments) * 100}%` }}
                                transition={{ duration: 0.5 }}
                                className="bg-primary-600 h-2 rounded-full"
                              />
                            </div>
                          </div>
                        )}

                        {/* Time Info */}
                        <div className="flex justify-between items-center text-sm text-gray-500">
                          <div className="text-persian-primary">
                            {job.startTime && (
                              <span>شروع: {job.startTime}</span>
                            )}
                            {job.endTime && (
                              <span className="mr-4">پایان: {job.endTime}</span>
                            )}
                          </div>
                          
                          {job.status === 'running' && (
                            <div className="flex space-x-2 space-x-reverse">
                              <Button variant="outline" size="sm" leftIcon={<PauseIcon className="h-4 w-4" />}>
                                توقف
                              </Button>
                              <Button variant="ghost" size="sm" leftIcon={<StopIcon className="h-4 w-4" />}>
                                لغو
                              </Button>
                            </div>
                          )}
                        </div>

                        {/* Results */}
                        {job.results && (
                          <div className="mt-4 pt-4 border-t border-gray-200">
                            <div className="grid grid-cols-3 gap-4 text-center">
                              <div>
                                <p className="ui-text-large font-medium text-green-600 persian-numbers">
                                  {job.results.successful}
                                </p>
                                <p className="ui-text-xs text-gray-500 text-persian-primary">موفق</p>
                              </div>
                              <div>
                                <p className="ui-text-large font-medium text-red-600 persian-numbers">
                                  {job.results.failed}
                                </p>
                                <p className="ui-text-xs text-gray-500 text-persian-primary">ناموفق</p>
                              </div>
                              <div>
                                <p className="ui-text-large font-medium text-blue-600 persian-numbers">
                                  {job.results.accuracy.toFixed(1)}%
                                </p>
                                <p className="ui-text-xs text-gray-500 text-persian-primary">دقت</p>
                              </div>
                            </div>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Current Status */}
            <Card>
              <CardHeader>
                <CardTitle>وضعیت فعلی</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      در انتظار
                    </span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">
                      {batchJobs.filter(job => job.status === 'pending').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      در حال اجرا
                    </span>
                    <span className="ui-text-small font-medium text-yellow-600 persian-numbers">
                      {batchJobs.filter(job => job.status === 'running').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      تکمیل شده
                    </span>
                    <span className="ui-text-small font-medium text-green-600 persian-numbers">
                      {batchJobs.filter(job => job.status === 'completed').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      ناموفق
                    </span>
                    <span className="ui-text-small font-medium text-red-600 persian-numbers">
                      {batchJobs.filter(job => job.status === 'failed').length}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Processing Tips */}
            <Card>
              <CardHeader>
                <CardTitle>نکات مهم</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 ui-text-small text-gray-600 text-persian-primary">
                  <p>• حداکثر ۱۰۰ سند در هر دسته قابل پردازش است</p>
                  <p>• پردازش بسته به حجم اسناد زمان می‌برد</p>
                  <p>• می‌توانید چندین دسته همزمان اجرا کنید</p>
                  <p>• نتایج به صورت خودکار ذخیره می‌شود</p>
                </div>
              </CardContent>
            </Card>

            {/* System Resources */}
            <Card>
              <CardHeader>
                <CardTitle>منابع سیستم</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="ui-text-small text-gray-600 text-persian-primary">CPU</span>
                      <span className="ui-text-small text-gray-600 persian-numbers">45%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: '45%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="ui-text-small text-gray-600 text-persian-primary">RAM</span>
                      <span className="ui-text-small text-gray-600 persian-numbers">68%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-green-600 h-2 rounded-full" style={{ width: '68%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="ui-text-small text-gray-600 text-persian-primary">GPU</span>
                      <span className="ui-text-small text-gray-600 persian-numbers">23%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-purple-600 h-2 rounded-full" style={{ width: '23%' }} />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}