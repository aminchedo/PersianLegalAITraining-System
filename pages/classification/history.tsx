'use client'

import React, { useState } from 'react'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import Button from '../../src/components/ui/Button'
import Input from '../../src/components/ui/Input'
import Badge from '../../src/components/ui/Badge'
import {
  ClockIcon,
  MagnifyingGlassIcon,
  DocumentTextIcon,
  EyeIcon,
  ArrowDownTrayIcon,
  FunnelIcon,
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'

interface ClassificationRecord {
  id: string
  text: string
  result: string
  confidence: number
  timestamp: string
  type: 'single' | 'batch'
  status: 'completed' | 'failed'
  processingTime: number
}

const mockHistory: ClassificationRecord[] = [
  {
    id: '1',
    text: 'این قرارداد در تاریخ ۱۴۰۲/۰۹/۱۰ بین طرفین منعقد گردیده...',
    result: 'قرارداد',
    confidence: 95.8,
    timestamp: '1402/09/16 - 14:30',
    type: 'single',
    status: 'completed',
    processingTime: 1.2
  },
  {
    id: '2',
    text: 'بر اساس ماده ۱۰ قانون مدنی، هر شخص حقیقی...',
    result: 'قانون',
    confidence: 92.4,
    timestamp: '1402/09/16 - 12:15',
    type: 'single',
    status: 'completed',
    processingTime: 0.8
  },
  {
    id: '3',
    text: 'پردازش دسته‌ای ۲۳ سند',
    result: 'متنوع',
    confidence: 88.7,
    timestamp: '1402/09/15 - 16:45',
    type: 'batch',
    status: 'completed',
    processingTime: 45.6
  },
  {
    id: '4',
    text: 'دیوان عدالت اداری در رأی شماره ۱۲۳۴...',
    result: 'رأی قضایی',
    confidence: 97.2,
    timestamp: '1402/09/15 - 11:20',
    type: 'single',
    status: 'completed',
    processingTime: 1.5
  },
  {
    id: '5',
    text: 'خطا در پردازش متن',
    result: '-',
    confidence: 0,
    timestamp: '1402/09/15 - 09:30',
    type: 'single',
    status: 'failed',
    processingTime: 0.3
  }
]

export default function ClassificationHistoryPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<'all' | 'single' | 'batch'>('all')
  const [filterStatus, setFilterStatus] = useState<'all' | 'completed' | 'failed'>('all')

  const filteredHistory = mockHistory.filter(record => {
    const matchesSearch = record.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         record.result.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesType = filterType === 'all' || record.type === filterType
    const matchesStatus = filterStatus === 'all' || record.status === filterStatus
    
    return matchesSearch && matchesType && matchesStatus
  })

  const getStatusBadge = (status: string) => {
    return status === 'completed' 
      ? <Badge variant="success" size="sm">موفق</Badge>
      : <Badge variant="danger" size="sm">ناموفق</Badge>
  }

  const getTypeBadge = (type: string) => {
    return type === 'single'
      ? <Badge variant="info" size="sm">تکی</Badge>
      : <Badge variant="secondary" size="sm">دسته‌ای</Badge>
  }

  return (
    <MainLayout
      title="تاریخچه طبقه‌بندی - سامانه هوش مصنوعی حقوقی فارسی"
      description="مشاهده تاریخچه و نتایج طبقه‌بندی‌های انجام شده"
    >
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="heading-1 text-gray-900">
            تاریخچه طبقه‌بندی
          </h1>
          <p className="paragraph-normal text-gray-600">
            تمام طبقه‌بندی‌های انجام شده و نتایج آن‌ها
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-primary-600 mb-0 persian-numbers">
                  {mockHistory.length}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">کل طبقه‌بندی</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-green-600 mb-0 persian-numbers">
                  {mockHistory.filter(r => r.status === 'completed').length}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">موفق</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-red-600 mb-0 persian-numbers">
                  {mockHistory.filter(r => r.status === 'failed').length}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">ناموفق</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-3 text-blue-600 mb-0 persian-numbers">
                  {(mockHistory.filter(r => r.status === 'completed').reduce((sum, r) => sum + r.confidence, 0) / 
                    mockHistory.filter(r => r.status === 'completed').length).toFixed(1)}%
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">میانگین دقت</p>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Filters */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>فیلترها</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Search */}
                  <div>
                    <label className="form-label">جستجو</label>
                    <Input
                      placeholder="جستجو در متن یا نتایج..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}
                    />
                  </div>

                  {/* Type Filter */}
                  <div>
                    <label className="form-label">نوع پردازش</label>
                    <select
                      value={filterType}
                      onChange={(e) => setFilterType(e.target.value as any)}
                      className="form-input w-full"
                    >
                      <option value="all">همه</option>
                      <option value="single">تکی</option>
                      <option value="batch">دسته‌ای</option>
                    </select>
                  </div>

                  {/* Status Filter */}
                  <div>
                    <label className="form-label">وضعیت</label>
                    <select
                      value={filterStatus}
                      onChange={(e) => setFilterStatus(e.target.value as any)}
                      className="form-input w-full"
                    >
                      <option value="all">همه</option>
                      <option value="completed">موفق</option>
                      <option value="failed">ناموفق</option>
                    </select>
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    fullWidth
                    onClick={() => {
                      setSearchQuery('')
                      setFilterType('all')
                      setFilterStatus('all')
                    }}
                  >
                    پاک کردن فیلترها
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card>
              <CardHeader>
                <CardTitle>آمار سریع</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">امروز</span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">12</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">این هفته</span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">47</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">این ماه</span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">203</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* History List */}
          <div className="lg:col-span-3 space-y-6">
            {/* Actions */}
            <Card>
              <CardContent>
                <div className="flex justify-between items-center">
                  <span className="ui-text-medium text-gray-700 text-persian-primary">
                    {filteredHistory.length} مورد یافت شد
                  </span>
                  <div className="flex space-x-2 space-x-reverse">
                    <Button variant="outline" size="sm" leftIcon={<ArrowDownTrayIcon className="h-4 w-4" />}>
                      خروجی Excel
                    </Button>
                    <Button variant="outline" size="sm" leftIcon={<FunnelIcon className="h-4 w-4" />}>
                      فیلتر پیشرفته
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* History Records */}
            <div className="space-y-4">
              {filteredHistory.length > 0 ? (
                filteredHistory.map((record, index) => (
                  <motion.div
                    key={record.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <Card hover>
                      <CardContent>
                        <div className="flex items-start justify-between">
                          <div className="flex items-start space-x-4 space-x-reverse flex-1">
                            <div className="p-2 bg-gray-50 rounded-lg">
                              {record.type === 'single' ? (
                                <DocumentTextIcon className="h-6 w-6 text-gray-600" />
                              ) : (
                                <ClockIcon className="h-6 w-6 text-gray-600" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-3 space-x-reverse mb-2">
                                {getTypeBadge(record.type)}
                                {getStatusBadge(record.status)}
                                <span className="ui-text-xs text-gray-500 text-persian-primary">
                                  {record.timestamp}
                                </span>
                              </div>
                              
                              <p className="ui-text-medium text-gray-900 text-persian-primary mb-2 truncate">
                                {record.text}
                              </p>
                              
                              {record.status === 'completed' && (
                                <div className="flex items-center space-x-4 space-x-reverse">
                                  <div className="flex items-center space-x-2 space-x-reverse">
                                    <Badge variant="primary" size="sm">
                                      {record.result}
                                    </Badge>
                                    <span className="ui-text-xs text-gray-500 text-persian-primary">
                                      اطمینان: <span className="persian-numbers">{record.confidence}%</span>
                                    </span>
                                  </div>
                                  <span className="ui-text-xs text-gray-500 text-persian-primary">
                                    زمان پردازش: <span className="persian-numbers">{record.processingTime}</span> ثانیه
                                  </span>
                                </div>
                              )}
                              
                              {record.status === 'failed' && (
                                <p className="ui-text-small text-red-600 text-persian-primary">
                                  خطا در پردازش متن
                                </p>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex space-x-2 space-x-reverse">
                            <Button variant="outline" size="sm" leftIcon={<EyeIcon className="h-4 w-4" />}>
                              جزئیات
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))
              ) : (
                <Card>
                  <CardContent>
                    <div className="text-center py-12">
                      <ClockIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <h3 className="heading-4 text-gray-900">
                        هیچ رکوردی یافت نشد
                      </h3>
                      <p className="paragraph-normal text-gray-600">
                        فیلترهای خود را تغییر دهید یا طبقه‌بندی جدیدی انجام دهید
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Pagination */}
            {filteredHistory.length > 0 && (
              <div className="flex justify-center">
                <div className="flex space-x-2 space-x-reverse">
                  <Button variant="outline" size="sm">قبلی</Button>
                  <Button variant="primary" size="sm">1</Button>
                  <Button variant="outline" size="sm">2</Button>
                  <Button variant="outline" size="sm">3</Button>
                  <Button variant="outline" size="sm">بعدی</Button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}