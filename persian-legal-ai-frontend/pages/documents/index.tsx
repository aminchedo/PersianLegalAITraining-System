import React, { useState, useEffect } from 'react'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import { Button } from '../../src/components/ui/Button'
import { Input } from '../../src/components/ui/Input'
import { Badge } from '../../src/components/ui/Badge'
import {
  DocumentTextIcon,
  MagnifyingGlassIcon,
  CloudArrowUpIcon,
  FolderIcon,
  EyeIcon,
  TrashIcon,
  PencilIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'

interface Document {
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
}

export default function DocumentsIndex() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedType, setSelectedType] = useState<string>('all')

  // Mock documents data
  useEffect(() => {
    const mockDocuments: Document[] = [
      {
        id: 'doc-1',
        title: 'قرارداد خرید و فروش ملک',
        content: 'این قرارداد بین خریدار و فروشنده منعقد می‌شود...',
        type: 'contract',
        status: 'processed',
        classification: 'قرارداد',
        confidence: 0.95,
        created_at: '2024-01-15T10:30:00Z',
        updated_at: '2024-01-15T10:35:00Z',
        file_size: 245760,
        file_type: 'PDF'
      },
      {
        id: 'doc-2',
        title: 'قانون مدنی - بخش تعهدات',
        content: 'تعهدات قراردادی و غیرقراردادی...',
        type: 'law',
        status: 'processed',
        classification: 'قانون',
        confidence: 0.98,
        created_at: '2024-01-14T09:00:00Z',
        updated_at: '2024-01-14T09:05:00Z',
        file_size: 1024000,
        file_type: 'PDF'
      },
      {
        id: 'doc-3',
        title: 'حکم دادگاه در پرونده ملکی',
        content: 'دادگاه پس از بررسی اوراق و مستندات...',
        type: 'judgment',
        status: 'processing',
        created_at: '2024-01-16T14:00:00Z',
        updated_at: '2024-01-16T14:05:00Z',
        file_size: 512000,
        file_type: 'PDF'
      },
      {
        id: 'doc-4',
        title: 'آیین‌نامه اجرایی قانون کار',
        content: 'این آیین‌نامه به منظور اجرای قانون کار...',
        type: 'regulation',
        status: 'pending',
        created_at: '2024-01-16T15:30:00Z',
        updated_at: '2024-01-16T15:30:00Z',
        file_size: 768000,
        file_type: 'PDF'
      }
    ]

    setTimeout(() => {
      setDocuments(mockDocuments)
      setLoading(false)
    }, 1000)
  }, [])

  const getStatusBadge = (status: Document['status']) => {
    switch (status) {
      case 'processed':
        return <Badge variant="success">پردازش شده</Badge>
      case 'processing':
        return <Badge variant="primary">در حال پردازش</Badge>
      case 'pending':
        return <Badge variant="warning">در انتظار</Badge>
      case 'error':
        return <Badge variant="error">خطا</Badge>
      default:
        return <Badge variant="secondary">نامشخص</Badge>
    }
  }

  const getTypeLabel = (type: Document['type']) => {
    switch (type) {
      case 'contract':
        return 'قرارداد'
      case 'law':
        return 'قانون'
      case 'regulation':
        return 'آیین‌نامه'
      case 'judgment':
        return 'حکم'
      default:
        return 'سایر'
    }
  }

  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         doc.content.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesType = selectedType === 'all' || doc.type === selectedType
    return matchesSearch && matchesType
  })

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 بایت'
    const k = 1024
    const sizes = ['بایت', 'کیلوبایت', 'مگابایت', 'گیگابایت']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <MainLayout title="مدیریت اسناد - سامانه هوش مصنوعی حقوقی فارسی">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="heading-2 text-gray-900 flex items-center">
              <FolderIcon className="h-8 w-8 text-indigo-600 ml-3" />
              مدیریت اسناد
            </h1>
            <p className="paragraph-normal text-gray-600">
              مشاهده، جستجو و مدیریت اسناد حقوقی
            </p>
          </div>
          <div className="flex space-x-3 space-x-reverse">
            <Link href="/documents/search">
              <Button variant="outline">
                <MagnifyingGlassIcon className="h-4 w-4 ml-2" />
                جستجوی پیشرفته
              </Button>
            </Link>
            <Link href="/documents/upload">
              <Button variant="primary">
                <CloudArrowUpIcon className="h-4 w-4 ml-2" />
                آپلود سند
              </Button>
            </Link>
          </div>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <DocumentTextIcon className="h-8 w-8 text-blue-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {documents.length}
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    کل اسناد
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-green-100 rounded-lg flex items-center justify-center">
                    <span className="text-green-600 font-bold text-sm">✓</span>
                  </div>
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {documents.filter(d => d.status === 'processed').length}
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    پردازش شده
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                    <span className="text-yellow-600 font-bold text-sm">⏳</span>
                  </div>
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {documents.filter(d => d.status === 'processing' || d.status === 'pending').length}
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    در انتظار
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 bg-purple-100 rounded-lg flex items-center justify-center">
                    <span className="text-purple-600 font-bold text-sm">📊</span>
                  </div>
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {documents.filter(d => d.classification).length}
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    طبقه‌بندی شده
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters and Search */}
        <Card>
          <CardContent className="p-6">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <Input
                  placeholder="جستجو در اسناد..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <div className="sm:w-48">
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-persian-primary"
                >
                  <option value="all">همه انواع</option>
                  <option value="contract">قرارداد</option>
                  <option value="law">قانون</option>
                  <option value="regulation">آیین‌نامه</option>
                  <option value="judgment">حکم</option>
                  <option value="other">سایر</option>
                </select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Documents List */}
        <Card>
          <CardHeader>
            <CardTitle>لیست اسناد</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                <span className="mr-3 text-persian-primary ui-text-medium">در حال بارگذاری...</span>
              </div>
            ) : filteredDocuments.length === 0 ? (
              <div className="text-center py-8">
                <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900 text-persian-primary">
                  هیچ سندی یافت نشد
                </h3>
                <p className="mt-1 text-sm text-gray-500 text-persian-primary">
                  برای شروع، سند جدید آپلود کنید.
                </p>
                <div className="mt-6">
                  <Link href="/documents/upload">
                    <Button variant="primary">
                      <CloudArrowUpIcon className="h-4 w-4 ml-2" />
                      آپلود سند جدید
                    </Button>
                  </Link>
                </div>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        عنوان سند
                      </th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        نوع
                      </th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        وضعیت
                      </th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        طبقه‌بندی
                      </th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        اندازه فایل
                      </th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        تاریخ ایجاد
                      </th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        عملیات
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredDocuments.map((document) => (
                      <tr key={document.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <DocumentTextIcon className="h-5 w-5 text-gray-400 ml-3" />
                            <div>
                              <div className="text-sm font-medium text-gray-900 text-persian-primary">
                                {document.title}
                              </div>
                              <div className="text-sm text-gray-500 text-persian-primary">
                                {document.file_type}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <Badge variant="secondary">
                            {getTypeLabel(document.type)}
                          </Badge>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {getStatusBadge(document.status)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {document.classification ? (
                            <div>
                              <div className="text-sm text-gray-900 text-persian-primary">
                                {document.classification}
                              </div>
                              {document.confidence && (
                                <div className="text-xs text-gray-500 text-persian-primary">
                                  اطمینان: {(document.confidence * 100).toFixed(1)}%
                                </div>
                              )}
                            </div>
                          ) : (
                            <span className="text-sm text-gray-500 text-persian-primary">-</span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-persian-primary">
                          {formatFileSize(document.file_size)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-persian-primary">
                          {new Date(document.created_at).toLocaleDateString('fa-IR')}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <div className="flex space-x-2 space-x-reverse">
                            <Button variant="ghost" size="sm">
                              <EyeIcon className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="sm">
                              <PencilIcon className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="sm">
                              <TrashIcon className="h-4 w-4" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </MainLayout>
  )
}