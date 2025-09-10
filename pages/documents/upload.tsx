'use client'

import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import Button from '../../src/components/ui/Button'
import Input from '../../src/components/ui/Input'
import Alert from '../../src/components/ui/Alert'
import Badge from '../../src/components/ui/Badge'
import {
  CloudArrowUpIcon,
  DocumentTextIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import { motion, AnimatePresence } from 'framer-motion'

interface UploadedFile {
  file: File
  id: string
  status: 'uploading' | 'success' | 'error'
  progress: number
  error?: string
}

export default function UploadPage() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [description, setDescription] = useState('')

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      status: 'uploading',
      progress: 0,
    }))

    setUploadedFiles(prev => [...prev, ...newFiles])
    
    // Simulate upload progress
    newFiles.forEach(uploadFile => {
      simulateUpload(uploadFile.id)
    })
  }, [])

  const simulateUpload = (fileId: string) => {
    const interval = setInterval(() => {
      setUploadedFiles(prev => 
        prev.map(file => {
          if (file.id === fileId) {
            const newProgress = Math.min(file.progress + Math.random() * 20, 100)
            
            if (newProgress >= 100) {
              clearInterval(interval)
              // Randomly succeed or fail
              const success = Math.random() > 0.2
              return {
                ...file,
                progress: 100,
                status: success ? 'success' : 'error',
                error: success ? undefined : 'خطا در آپلود فایل'
              }
            }
            
            return { ...file, progress: newProgress }
          }
          return file
        })
      )
    }, 500)
  }

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId))
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
  })

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase()
    return <DocumentTextIcon className="h-8 w-8 text-blue-600" />
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
      default:
        return null
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 بایت'
    const k = 1024
    const sizes = ['بایت', 'کیلوبایت', 'مگابایت', 'گیگابایت']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const handleSubmit = () => {
    setIsUploading(true)
    // Simulate API call
    setTimeout(() => {
      setIsUploading(false)
      // Handle success
    }, 2000)
  }

  return (
    <MainLayout
      title="آپلود سند - سامانه هوش مصنوعی حقوقی فارسی"
      description="آپلود و پردازش اسناد حقوقی جدید"
    >
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="heading-1 text-gray-900">
            آپلود سند جدید
          </h1>
          <p className="paragraph-normal text-gray-600">
            اسناد حقوقی خود را آپلود کنید تا به صورت خودکار طبقه‌بندی شوند
          </p>
        </div>

        {/* Upload Instructions */}
        <Alert variant="info">
          <div>
            <p className="font-medium">راهنمای آپلود:</p>
            <ul className="mt-2 space-y-1 text-sm">
              <li>• فرمت‌های پشتیبانی شده: PDF، DOC، DOCX، TXT</li>
              <li>• حداکثر اندازه فایل: ۱۰ مگابایت</li>
              <li>• می‌توانید چندین فایل را همزمان آپلود کنید</li>
              <li>• پس از آپلود، فایل‌ها به صورت خودکار طبقه‌بندی می‌شوند</li>
            </ul>
          </div>
        </Alert>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* Drag & Drop Zone */}
            <Card>
              <CardContent>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                    isDragActive
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <input {...getInputProps()} />
                  <CloudArrowUpIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  {isDragActive ? (
                    <p className="ui-text-large text-primary-600 text-persian-primary">
                      فایل‌ها را اینجا رها کنید...
                    </p>
                  ) : (
                    <div>
                      <p className="ui-text-large text-gray-600 text-persian-primary mb-2">
                        فایل‌ها را اینجا بکشید یا کلیک کنید
                      </p>
                      <p className="ui-text-small text-gray-400 text-persian-primary">
                        PDF، DOC، DOCX، TXT تا ۱۰ مگابایت
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Description */}
            <Card>
              <CardHeader>
                <CardTitle>توضیحات (اختیاری)</CardTitle>
              </CardHeader>
              <CardContent>
                <Input
                  placeholder="توضیحات مربوط به اسناد آپلود شده..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  fullWidth
                />
              </CardContent>
            </Card>

            {/* Uploaded Files */}
            {uploadedFiles.length > 0 && (
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <CardTitle>فایل‌های آپلود شده</CardTitle>
                    <Badge variant="info" size="sm">
                      {uploadedFiles.length} فایل
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <AnimatePresence>
                      {uploadedFiles.map((uploadFile) => (
                        <motion.div
                          key={uploadFile.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          className="flex items-center space-x-4 space-x-reverse p-4 border border-gray-200 rounded-lg"
                        >
                          <div className="flex-shrink-0">
                            {getFileIcon(uploadFile.file.name)}
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <p className="ui-text-medium font-medium text-gray-900 text-persian-primary truncate">
                              {uploadFile.file.name}
                            </p>
                            <p className="ui-text-small text-gray-500 text-persian-primary">
                              {formatFileSize(uploadFile.file.size)}
                            </p>
                            
                            {uploadFile.status === 'uploading' && (
                              <div className="mt-2">
                                <div className="flex justify-between items-center mb-1">
                                  <span className="ui-text-xs text-gray-500 text-persian-primary">
                                    در حال آپلود...
                                  </span>
                                  <span className="ui-text-xs text-gray-500 text-persian-primary">
                                    {Math.round(uploadFile.progress)}%
                                  </span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div
                                    className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                                    style={{ width: `${uploadFile.progress}%` }}
                                  />
                                </div>
                              </div>
                            )}
                            
                            {uploadFile.error && (
                              <p className="ui-text-xs text-red-600 text-persian-primary mt-1">
                                {uploadFile.error}
                              </p>
                            )}
                          </div>
                          
                          <div className="flex items-center space-x-2 space-x-reverse">
                            {getStatusIcon(uploadFile.status)}
                            <button
                              onClick={() => removeFile(uploadFile.id)}
                              className="text-gray-400 hover:text-red-500 transition-colors"
                            >
                              <XMarkIcon className="h-5 w-5" />
                            </button>
                          </div>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Submit Button */}
            {uploadedFiles.length > 0 && (
              <div className="flex justify-end">
                <Button
                  loading={isUploading}
                  onClick={handleSubmit}
                  size="lg"
                  leftIcon={<CheckCircleIcon className="h-5 w-5" />}
                >
                  شروع پردازش
                </Button>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Upload Stats */}
            <Card>
              <CardHeader>
                <CardTitle>آمار آپلود</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      کل فایل‌ها
                    </span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">
                      {uploadedFiles.length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      موفق
                    </span>
                    <span className="ui-text-small font-medium text-green-600 persian-numbers">
                      {uploadedFiles.filter(f => f.status === 'success').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      در حال پردازش
                    </span>
                    <span className="ui-text-small font-medium text-yellow-600 persian-numbers">
                      {uploadedFiles.filter(f => f.status === 'uploading').length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      خطا
                    </span>
                    <span className="ui-text-small font-medium text-red-600 persian-numbers">
                      {uploadedFiles.filter(f => f.status === 'error').length}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Recent Uploads */}
            <Card>
              <CardHeader>
                <CardTitle>آپلودهای اخیر</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <DocumentTextIcon className="h-5 w-5 text-gray-400" />
                    <div className="flex-1 min-w-0">
                      <p className="ui-text-small text-gray-900 text-persian-primary truncate">
                        قرارداد خرید.pdf
                      </p>
                      <p className="ui-text-xs text-gray-500 text-persian-primary">
                        ۲ ساعت پیش
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <DocumentTextIcon className="h-5 w-5 text-gray-400" />
                    <div className="flex-1 min-w-0">
                      <p className="ui-text-small text-gray-900 text-persian-primary truncate">
                        آیین‌نامه شرکت.docx
                      </p>
                      <p className="ui-text-xs text-gray-500 text-persian-primary">
                        دیروز
                      </p>
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