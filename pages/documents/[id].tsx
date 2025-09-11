'use client'

import React, { useState } from 'react'
import { useRouter } from 'next/router'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import Button from '../../src/components/ui/Button'
import Badge from '../../src/components/ui/Badge'
import {
  DocumentTextIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  TrashIcon,
  PencilSquareIcon,
  EyeIcon,
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'

export default function DocumentDetailPage() {
  const router = useRouter()
  const { id } = router.query

  // Mock document data
  const document = {
    id: id,
    title: 'قرارداد خرید املاک تجاری.pdf',
    type: 'PDF',
    size: '2.3 MB',
    uploadDate: '1402/09/15',
    status: 'processed',
    classification: {
      primary: 'قرارداد',
      confidence: 95.8,
      categories: [
        { name: 'قرارداد', confidence: 95.8 },
        { name: 'املاک', confidence: 87.2 },
        { name: 'تجاری', confidence: 82.1 },
      ]
    },
    content: `این قرارداد در تاریخ ۱۴۰۲/۰۹/۱۰ بین آقای احمد محمدی (خریدار) و خانم فاطمه احمدی (فروشنده) منعقد گردیده است.

موضوع قرارداد: خرید و فروش ملک تجاری واقع در تهران، خیابان ولیعصر، پلاک ۱۲۳

مبلغ قرارداد: ۵,۰۰۰,۰۰۰,۰۰۰ ریال (پنج میلیارد ریال)

شرایط پرداخت:
- پیش پرداخت: ۱,۰۰۰,۰۰۰,۰۰۰ ریال
- باقی مانده: در ۱۲ قسط ماهانه

تعهدات فروشنده:
- تحویل ملک در تاریخ ۱۴۰۲/۱۰/۰۱
- انتقال اسناد مالکیت
- تخلیه ملک از هرگونه متصرف

تعهدات خریدار:
- پرداخت مبلغ قرارداد طبق شرایط مذکور
- پرداخت هزینه‌های انتقال سند
- بیمه ملک

در صورت نقض قرارداد از سوی هر یک از طرفین، طرف متضرر حق مطالبه خسارت را خواهد داشت.`,
    metadata: {
      pages: 15,
      words: 2847,
      characters: 18923,
      language: 'فارسی',
      encoding: 'UTF-8',
    },
    tags: ['قرارداد', 'املاک', 'تجاری', 'خرید', 'فروش'],
    relatedDocuments: [
      { id: '2', title: 'سند مالکیت ملک.pdf', similarity: 85 },
      { id: '3', title: 'گواهی عدم بدهی.pdf', similarity: 72 },
      { id: '4', title: 'نقشه ملک.dwg', similarity: 68 },
    ]
  }

  const [activeTab, setActiveTab] = useState<'content' | 'metadata' | 'classification' | 'related'>('content')

  return (
    <MainLayout
      title={`${document.title} - مشاهده سند`}
      description="مشاهده جزئیات و محتوای سند حقوقی"
    >
      <div className="space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
          <div className="flex items-center space-x-4 space-x-reverse">
            <div className="p-3 bg-blue-50 rounded-lg">
              <DocumentTextIcon className="h-8 w-8 text-blue-600" />
            </div>
            <div>
              <h1 className="heading-2 text-gray-900 mb-0">
                {document.title}
              </h1>
              <div className="flex items-center space-x-4 space-x-reverse mt-2">
                <Badge variant="success" size="sm">پردازش شده</Badge>
                <span className="ui-text-small text-gray-500 text-persian-primary">
                  آپلود شده در {document.uploadDate}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex space-x-3 space-x-reverse">
            <Button variant="outline" size="sm" leftIcon={<ArrowDownTrayIcon className="h-4 w-4" />}>
              دانلود
            </Button>
            <Button variant="outline" size="sm" leftIcon={<ShareIcon className="h-4 w-4" />}>
              اشتراک
            </Button>
            <Button variant="outline" size="sm" leftIcon={<PencilSquareIcon className="h-4 w-4" />}>
              ویرایش
            </Button>
            <Button variant="ghost" size="sm" leftIcon={<TrashIcon className="h-4 w-4" />}>
              حذف
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-4 text-primary-600 mb-0 persian-numbers">
                  {document.metadata.pages}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">صفحه</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-4 text-green-600 mb-0 persian-numbers">
                  {document.metadata.words.toLocaleString('fa')}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">کلمه</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-4 text-blue-600 mb-0 persian-numbers">
                  {document.classification.confidence}%
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">اطمینان</p>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <div className="text-center">
                <p className="heading-4 text-purple-600 mb-0">
                  {document.size}
                </p>
                <p className="ui-text-small text-gray-600 text-persian-primary">اندازه</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8 space-x-reverse">
            {[
              { id: 'content', label: 'محتوای سند', icon: EyeIcon },
              { id: 'metadata', label: 'اطلاعات فنی', icon: DocumentTextIcon },
              { id: 'classification', label: 'طبقه‌بندی', icon: DocumentTextIcon },
              { id: 'related', label: 'اسناد مرتبط', icon: DocumentTextIcon },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 space-x-reverse py-4 px-1 border-b-2 font-medium ui-text-small transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="h-5 w-5" />
                <span className="text-persian-primary">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {activeTab === 'content' && (
            <Card>
              <CardHeader>
                <CardTitle>محتوای سند</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose max-w-none text-persian-primary">
                  <div className="whitespace-pre-line paragraph-normal text-gray-700">
                    {document.content}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {activeTab === 'metadata' && (
            <Card>
              <CardHeader>
                <CardTitle>اطلاعات فنی سند</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div>
                      <label className="form-label">نام فایل</label>
                      <p className="ui-text-medium text-gray-900 text-persian-primary">
                        {document.title}
                      </p>
                    </div>
                    <div>
                      <label className="form-label">نوع فایل</label>
                      <p className="ui-text-medium text-gray-900 text-persian-primary">
                        {document.type}
                      </p>
                    </div>
                    <div>
                      <label className="form-label">اندازه</label>
                      <p className="ui-text-medium text-gray-900 text-persian-primary">
                        {document.size}
                      </p>
                    </div>
                    <div>
                      <label className="form-label">زبان</label>
                      <p className="ui-text-medium text-gray-900 text-persian-primary">
                        {document.metadata.language}
                      </p>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <label className="form-label">تعداد صفحات</label>
                      <p className="ui-text-medium text-gray-900 persian-numbers">
                        {document.metadata.pages}
                      </p>
                    </div>
                    <div>
                      <label className="form-label">تعداد کلمات</label>
                      <p className="ui-text-medium text-gray-900 persian-numbers">
                        {document.metadata.words.toLocaleString('fa')}
                      </p>
                    </div>
                    <div>
                      <label className="form-label">تعداد کاراکترها</label>
                      <p className="ui-text-medium text-gray-900 persian-numbers">
                        {document.metadata.characters.toLocaleString('fa')}
                      </p>
                    </div>
                    <div>
                      <label className="form-label">کدگذاری</label>
                      <p className="ui-text-medium text-gray-900 text-persian-primary">
                        {document.metadata.encoding}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Tags */}
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <label className="form-label">برچسب‌ها</label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {document.tags.map((tag, index) => (
                      <Badge key={index} variant="secondary" size="sm">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {activeTab === 'classification' && (
            <Card>
              <CardHeader>
                <CardTitle>نتایج طبقه‌بندی هوشمند</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h3 className="ui-text-large font-medium text-gray-900 text-persian-primary mb-4">
                      طبقه‌بندی اصلی
                    </h3>
                    <div className="flex items-center space-x-4 space-x-reverse">
                      <Badge variant="primary" size="lg">
                        {document.classification.primary}
                      </Badge>
                      <span className="ui-text-medium text-gray-600 text-persian-primary">
                        میزان اطمینان: <span className="persian-numbers font-medium text-green-600">
                          {document.classification.confidence}%
                        </span>
                      </span>
                    </div>
                  </div>

                  <div>
                    <h3 className="ui-text-large font-medium text-gray-900 text-persian-primary mb-4">
                      تمام طبقه‌بندی‌ها
                    </h3>
                    <div className="space-y-3">
                      {document.classification.categories.map((category, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <span className="ui-text-medium text-gray-900 text-persian-primary">
                            {category.name}
                          </span>
                          <div className="flex items-center space-x-3 space-x-reverse">
                            <div className="w-24 bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-primary-600 h-2 rounded-full"
                                style={{ width: `${category.confidence}%` }}
                              />
                            </div>
                            <span className="ui-text-small text-gray-600 persian-numbers">
                              {category.confidence}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {activeTab === 'related' && (
            <Card>
              <CardHeader>
                <CardTitle>اسناد مرتبط</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {document.relatedDocuments.map((relatedDoc) => (
                    <div
                      key={relatedDoc.id}
                      className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center space-x-3 space-x-reverse">
                        <DocumentTextIcon className="h-6 w-6 text-gray-400" />
                        <div>
                          <p className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                            {relatedDoc.title}
                          </p>
                          <p className="ui-text-small text-gray-500 text-persian-primary">
                            شباهت: <span className="persian-numbers">{relatedDoc.similarity}%</span>
                          </p>
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
                        مشاهده
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </motion.div>
      </div>
    </MainLayout>
  )
}