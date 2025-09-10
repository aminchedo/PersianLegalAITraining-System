'use client'

import React, { useState } from 'react'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import Button from '../../src/components/ui/Button'
import Input from '../../src/components/ui/Input'
import Badge from '../../src/components/ui/Badge'
import {
  DocumentTextIcon,
  MagnifyingGlassIcon,
  CloudArrowUpIcon,
  EyeIcon,
  TrashIcon,
  FunnelIcon,
  CalendarIcon,
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'
import Link from 'next/link'

interface Document {
  id: string
  title: string
  type: string
  size: string
  uploadDate: string
  status: 'processed' | 'processing' | 'error'
  classification?: string
  confidence?: number
}

const mockDocuments: Document[] = [
  {
    id: '1',
    title: 'قرارداد خرید املاک تجاری.pdf',
    type: 'PDF',
    size: '2.3 MB',
    uploadDate: '1402/09/15',
    status: 'processed',
    classification: 'قرارداد',
    confidence: 95.8,
  },
  {
    id: '2',
    title: 'آیین‌نامه داخلی شرکت.docx',
    type: 'DOCX',
    size: '1.8 MB',
    uploadDate: '1402/09/14',
    status: 'processed',
    classification: 'آیین‌نامه',
    confidence: 89.2,
  },
  {
    id: '3',
    title: 'لایحه قانونی جدید.pdf',
    type: 'PDF',
    size: '4.1 MB',
    uploadDate: '1402/09/13',
    status: 'processing',
  },
  {
    id: '4',
    title: 'رأی دیوان عدالت اداری.pdf',
    type: 'PDF',
    size: '1.2 MB',
    uploadDate: '1402/09/12',
    status: 'processed',
    classification: 'رأی قضایی',
    confidence: 92.5,
  },
  {
    id: '5',
    title: 'تفسیر قانونی ماده 10.txt',
    type: 'TXT',
    size: '0.5 MB',
    uploadDate: '1402/09/11',
    status: 'error',
  },
]

function DocumentCard({ document }: { document: Document }) {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'processed':
        return <Badge variant="success" size="sm">پردازش شده</Badge>
      case 'processing':
        return <Badge variant="warning" size="sm">در حال پردازش</Badge>
      case 'error':
        return <Badge variant="danger" size="sm">خطا</Badge>
      default:
        return <Badge variant="secondary" size="sm">نامشخص</Badge>
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card hover>
        <CardContent>
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3 space-x-reverse flex-1">
              <div className="p-2 bg-blue-50 rounded-lg">
                <DocumentTextIcon className="h-6 w-6 text-blue-600" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="ui-text-medium font-medium text-gray-900 text-persian-primary truncate">
                  {document.title}
                </h3>
                <div className="flex items-center space-x-4 space-x-reverse mt-1">
                  <span className="ui-text-xs text-gray-500 text-persian-primary">
                    {document.type}
                  </span>
                  <span className="ui-text-xs text-gray-500 text-persian-primary">
                    {document.size}
                  </span>
                  <span className="ui-text-xs text-gray-500 text-persian-primary">
                    {document.uploadDate}
                  </span>
                </div>
                {document.classification && (
                  <div className="flex items-center space-x-2 space-x-reverse mt-2">
                    <Badge variant="primary" size="sm">
                      {document.classification}
                    </Badge>
                    <span className="ui-text-xs text-gray-500 text-persian-primary">
                      اطمینان: {document.confidence}٪
                    </span>
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center space-x-2 space-x-reverse">
              {getStatusBadge(document.status)}
            </div>
          </div>
          
          <div className="flex justify-between items-center mt-4 pt-4 border-t border-gray-200">
            <div className="flex space-x-2 space-x-reverse">
              <Link href={`/documents/${document.id}`}>
                <Button variant="outline" size="sm" leftIcon={<EyeIcon className="h-4 w-4" />}>
                  مشاهده
                </Button>
              </Link>
              <Button variant="ghost" size="sm" leftIcon={<TrashIcon className="h-4 w-4" />}>
                حذف
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export default function DocumentsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [sortBy, setSortBy] = useState('date')

  const filteredDocuments = mockDocuments.filter(doc => 
    doc.title.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <MainLayout
      title="مدیریت اسناد - سامانه هوش مصنوعی حقوقی فارسی"
      description="مدیریت، جستجو و طبقه‌بندی اسناد حقوقی"
    >
      <div className="space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
          <div>
            <h1 className="heading-1 text-gray-900">
              مدیریت اسناد
            </h1>
            <p className="paragraph-normal text-gray-600">
              مدیریت، جستجو و طبقه‌بندی اسناد حقوقی خود
            </p>
          </div>
          <div className="flex space-x-3 space-x-reverse">
            <Link href="/documents/upload">
              <Button leftIcon={<CloudArrowUpIcon className="h-5 w-5" />}>
                آپلود سند جدید
              </Button>
            </Link>
            <Link href="/documents/search">
              <Button variant="outline" leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}>
                جستجوی پیشرفته
              </Button>
            </Link>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-primary-600 mb-0 persian-numbers">
                  {mockDocuments.length}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">
                  کل اسناد
                </p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-green-600 mb-0 persian-numbers">
                  {mockDocuments.filter(d => d.status === 'processed').length}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">
                  پردازش شده
                </p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-yellow-600 mb-0 persian-numbers">
                  {mockDocuments.filter(d => d.status === 'processing').length}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">
                  در حال پردازش
                </p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-red-600 mb-0 persian-numbers">
                  {mockDocuments.filter(d => d.status === 'error').length}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">
                  خطا
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters and Search */}
        <Card>
          <CardContent>
            <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 sm:space-x-reverse">
              <div className="flex-1">
                <Input
                  placeholder="جستجو در اسناد..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}
                />
              </div>
              <div className="flex space-x-3 space-x-reverse">
                <Button variant="outline" size="sm" leftIcon={<FunnelIcon className="h-4 w-4" />}>
                  فیلتر
                </Button>
                <Button variant="outline" size="sm" leftIcon={<CalendarIcon className="h-4 w-4" />}>
                  تاریخ
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Documents List */}
        <div className="space-y-4">
          {filteredDocuments.length > 0 ? (
            filteredDocuments.map((document, index) => (
              <motion.div
                key={document.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <DocumentCard document={document} />
              </motion.div>
            ))
          ) : (
            <Card>
              <CardContent>
                <div className="text-center py-12">
                  <DocumentTextIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="heading-4 text-gray-900">
                    هیچ سندی یافت نشد
                  </h3>
                  <p className="paragraph-normal text-gray-600">
                    سند جدید آپلود کنید یا جستجوی خود را تغییر دهید
                  </p>
                  <div className="mt-6">
                    <Link href="/documents/upload">
                      <Button leftIcon={<CloudArrowUpIcon className="h-5 w-5" />}>
                        آپلود سند جدید
                      </Button>
                    </Link>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Pagination */}
        {filteredDocuments.length > 0 && (
          <div className="flex justify-center">
            <div className="flex space-x-2 space-x-reverse">
              <Button variant="outline" size="sm">
                قبلی
              </Button>
              <Button variant="primary" size="sm">
                1
              </Button>
              <Button variant="outline" size="sm">
                2
              </Button>
              <Button variant="outline" size="sm">
                3
              </Button>
              <Button variant="outline" size="sm">
                بعدی
              </Button>
            </div>
          </div>
        )}
      </div>
    </MainLayout>
  )
}