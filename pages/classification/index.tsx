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
  ClipboardDocumentIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import { motion, AnimatePresence } from 'framer-motion'

interface ClassificationResult {
  category: string
  confidence: number
  description: string
}

export default function ClassificationPage() {
  const [inputText, setInputText] = useState('')
  const [isClassifying, setIsClassifying] = useState(false)
  const [results, setResults] = useState<ClassificationResult[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleClassify = async () => {
    if (!inputText.trim()) {
      setError('لطفاً متن مورد نظر را وارد کنید')
      return
    }

    setIsClassifying(true)
    setError(null)
    setResults(null)

    // Simulate API call
    setTimeout(() => {
      const mockResults: ClassificationResult[] = [
        {
          category: 'قرارداد',
          confidence: 95.8,
          description: 'این متن حاوی شرایط و ضوابط قراردادی است'
        },
        {
          category: 'حقوق مدنی',
          confidence: 87.2,
          description: 'محتوای مربوط به حقوق مدنی و روابط اشخاص'
        },
        {
          category: 'املاک',
          confidence: 82.1,
          description: 'مطالب مرتبط با معاملات املاک و مستغلات'
        },
        {
          category: 'تجاری',
          confidence: 76.4,
          description: 'جنبه‌های تجاری و اقتصادی موضوع'
        }
      ]
      
      setResults(mockResults)
      setIsClassifying(false)
    }, 2000)
  }

  const handleClear = () => {
    setInputText('')
    setResults(null)
    setError(null)
  }

  const sampleTexts = [
    'این قرارداد در تاریخ ۱۴۰۲/۰۹/۱۰ بین طرفین منعقد گردیده و شامل شرایط خرید و فروش ملک می‌باشد...',
    'بر اساس ماده ۱۰ قانون مدنی، هر شخص حقیقی از زمان تولد تا فوت دارای شخصیت حقوقی است...',
    'شرکت مذکور متعهد است تا پایان سال مالی جاری، کلیه تعهدات مالی خود را ایفا نماید...'
  ]

  return (
    <MainLayout
      title="طبقه‌بندی هوشمند متن - سامانه هوش مصنوعی حقوقی فارسی"
      description="طبقه‌بندی خودکار متون حقوقی با استفاده از هوش مصنوعی"
    >
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center">
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-gradient-to-br from-primary-500 to-primary-700 rounded-2xl">
              <CpuChipIcon className="h-12 w-12 text-white" />
            </div>
          </div>
          <h1 className="heading-1 text-gray-900">
            طبقه‌بندی هوشمند متن حقوقی
          </h1>
          <p className="paragraph-large text-gray-600 max-w-2xl mx-auto">
            متن حقوقی خود را وارد کنید تا به صورت خودکار طبقه‌بندی شود و دسته‌بندی دقیق آن مشخص گردد
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Text Input */}
            <Card>
              <CardHeader>
                <CardTitle>متن مورد نظر را وارد کنید</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="متن حقوقی خود را در اینجا وارد کنید..."
                    className="w-full h-64 p-4 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-persian-primary form-input"
                    dir="rtl"
                  />
                  
                  <div className="flex justify-between items-center">
                    <span className="ui-text-small text-gray-500 text-persian-primary">
                      {inputText.length} کاراکتر
                    </span>
                    <div className="flex space-x-3 space-x-reverse">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleClear}
                        disabled={!inputText && !results}
                      >
                        پاک کردن
                      </Button>
                      <Button
                        onClick={handleClassify}
                        loading={isClassifying}
                        disabled={!inputText.trim()}
                        leftIcon={<SparklesIcon className="h-5 w-5" />}
                      >
                        طبقه‌بندی کن
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Sample Texts */}
            <Card>
              <CardHeader>
                <CardTitle>متون نمونه</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {sampleTexts.map((text, index) => (
                    <button
                      key={index}
                      onClick={() => setInputText(text)}
                      className="w-full p-3 text-right bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
                    >
                      <p className="ui-text-small text-gray-700 text-persian-primary truncate">
                        {text}
                      </p>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Error */}
            {error && (
              <Alert variant="error" dismissible onDismiss={() => setError(null)}>
                {error}
              </Alert>
            )}

            {/* Loading */}
            {isClassifying && (
              <Card>
                <CardContent>
                  <div className="text-center py-12">
                    <Loading size="lg" text="در حال تجزیه و تحلیل متن..." />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Results */}
            <AnimatePresence>
              {results && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <Card>
                    <CardHeader>
                      <div className="flex justify-between items-center">
                        <CardTitle>نتایج طبقه‌بندی</CardTitle>
                        <Badge variant="success" size="sm">
                          تحلیل کامل شد
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {results.map((result, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.1 }}
                            className="p-4 border border-gray-200 rounded-lg hover:shadow-sm transition-shadow"
                          >
                            <div className="flex justify-between items-start mb-2">
                              <h3 className="ui-text-large font-medium text-gray-900 text-persian-primary">
                                {result.category}
                              </h3>
                              <div className="flex items-center space-x-2 space-x-reverse">
                                <div className="w-16 bg-gray-200 rounded-full h-2">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${result.confidence}%` }}
                                    transition={{ duration: 1, delay: index * 0.1 }}
                                    className="bg-primary-600 h-2 rounded-full"
                                  />
                                </div>
                                <span className="ui-text-small font-medium text-primary-600 persian-numbers">
                                  {result.confidence}%
                                </span>
                              </div>
                            </div>
                            <p className="ui-text-small text-gray-600 text-persian-primary">
                              {result.description}
                            </p>
                          </motion.div>
                        ))}
                      </div>

                      {/* Actions */}
                      <div className="flex justify-center space-x-4 space-x-reverse mt-6 pt-6 border-t border-gray-200">
                        <Button variant="outline" leftIcon={<DocumentTextIcon className="h-5 w-5" />}>
                          ذخیره نتایج
                        </Button>
                        <Button variant="outline" leftIcon={<ClipboardDocumentIcon className="h-5 w-5" />}>
                          کپی نتایج
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Statistics */}
            <Card>
              <CardHeader>
                <CardTitle>آمار طبقه‌بندی</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      امروز
                    </span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">
                      ۴۷
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      این هفته
                    </span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">
                      ۲۸۳
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      این ماه
                    </span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">
                      ۱,۲۴۵
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      کل
                    </span>
                    <span className="ui-text-small font-medium text-gray-900 persian-numbers">
                      ۱۵,۶۷۸
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Popular Categories */}
            <Card>
              <CardHeader>
                <CardTitle>پرکاربردترین دسته‌ها</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { name: 'قرارداد', count: 1245, percentage: 35 },
                    { name: 'قانون', count: 987, percentage: 28 },
                    { name: 'آیین‌نامه', count: 654, percentage: 18 },
                    { name: 'رأی قضایی', count: 432, percentage: 12 },
                    { name: 'سایر', count: 234, percentage: 7 },
                  ].map((category, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="ui-text-small text-gray-700 text-persian-primary">
                        {category.name}
                      </span>
                      <div className="flex items-center space-x-2 space-x-reverse">
                        <div className="w-12 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${category.percentage}%` }}
                          />
                        </div>
                        <span className="ui-text-xs text-gray-500 persian-numbers">
                          {category.count}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Help */}
            <Card>
              <CardHeader>
                <CardTitle>راهنمای استفاده</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 ui-text-small text-gray-600 text-persian-primary">
                  <p>• متن حقوقی خود را در کادر بالا وارد کنید</p>
                  <p>• از متون نمونه برای آزمایش استفاده کنید</p>
                  <p>• نتایج با درصد اطمینان نمایش داده می‌شود</p>
                  <p>• می‌توانید نتایج را ذخیره یا کپی کنید</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}