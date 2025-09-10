import React, { useState } from 'react'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import { Button } from '../../src/components/ui/Button'
import { Input } from '../../src/components/ui/Input'
import { Badge } from '../../src/components/ui/Badge'
import {
  CpuChipIcon,
  DocumentTextIcon,
  ChartBarIcon,
  SparklesIcon
} from '@heroicons/react/24/outline'

interface ClassificationResult {
  predicted_class: string
  confidence: number
  probabilities: { [key: string]: number }
  processing_time: number
}

export default function ClassificationIndex() {
  const [inputText, setInputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleClassify = async () => {
    if (!inputText.trim()) {
      setError('لطفاً متن مورد نظر را وارد کنید')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      const mockResult: ClassificationResult = {
        predicted_class: 'قرارداد',
        confidence: 0.87,
        probabilities: {
          'قرارداد': 0.87,
          'قانون': 0.08,
          'آیین‌نامه': 0.03,
          'حکم': 0.02
        },
        processing_time: 1.2
      }
      
      setResult(mockResult)
    } catch (err) {
      setError('خطا در طبقه‌بندی متن. لطفاً مجدداً تلاش کنید.')
    } finally {
      setLoading(false)
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return <Badge variant="success">اطمینان بالا</Badge>
    if (confidence >= 0.6) return <Badge variant="warning">اطمینان متوسط</Badge>
    return <Badge variant="error">اطمینان پایین</Badge>
  }

  return (
    <MainLayout title="طبقه‌بندی هوشمند - سامانه هوش مصنوعی حقوقی فارسی">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="heading-2 text-gray-900 flex items-center">
              <CpuChipIcon className="h-8 w-8 text-indigo-600 ml-3" />
              طبقه‌بندی هوشمند
            </h1>
            <p className="paragraph-normal text-gray-600">
              طبقه‌بندی خودکار متون حقوقی با استفاده از هوش مصنوعی
            </p>
          </div>
        </div>

        {/* Classification Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <DocumentTextIcon className="h-6 w-6 text-blue-600 ml-2" />
              ورود متن برای طبقه‌بندی
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <label className="form-label text-gray-700">
                  متن حقوقی خود را وارد کنید
                </label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="متن حقوقی خود را در اینجا وارد کنید..."
                  className="w-full h-48 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-vertical text-persian-primary"
                  disabled={loading}
                />
                <p className="form-help">
                  متن باید حداقل 50 کاراکتر باشد. برای نتایج بهتر، متن کامل‌تر وارد کنید.
                </p>
              </div>

              {error && (
                <div className="alert-persian-error">
                  {error}
                </div>
              )}

              <div className="flex justify-end">
                <Button 
                  variant="primary" 
                  onClick={handleClassify}
                  loading={loading}
                  disabled={!inputText.trim() || loading}
                >
                  <SparklesIcon className="h-4 w-4 ml-2" />
                  طبقه‌بندی
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Classification Result */}
        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center">
                  <ChartBarIcon className="h-6 w-6 text-green-600 ml-2" />
                  نتایج طبقه‌بندی
                </div>
                {getConfidenceBadge(result.confidence)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Main Result */}
                <div className="text-center p-6 bg-gray-50 rounded-lg">
                  <div className="text-3xl font-bold text-gray-900 text-persian-primary mb-2">
                    {result.predicted_class}
                  </div>
                  <div className={`text-xl font-semibold ${getConfidenceColor(result.confidence)}`}>
                    {(result.confidence * 100).toFixed(1)}% اطمینان
                  </div>
                  <p className="text-sm text-gray-600 text-persian-primary mt-2">
                    زمان پردازش: {result.processing_time} ثانیه
                  </p>
                </div>

                {/* Probability Distribution */}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 text-persian-primary mb-4">
                    توزیع احتمالات
                  </h3>
                  <div className="space-y-3">
                    {Object.entries(result.probabilities)
                      .sort(([,a], [,b]) => b - a)
                      .map(([category, probability]) => (
                        <div key={category} className="flex items-center justify-between">
                          <div className="flex items-center flex-1">
                            <span className="text-sm font-medium text-gray-900 text-persian-primary w-20">
                              {category}
                            </span>
                            <div className="flex-1 mx-4">
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                                  style={{ width: `${probability * 100}%` }}
                                />
                              </div>
                            </div>
                            <span className="text-sm text-gray-600 text-persian-primary w-12 text-left">
                              {(probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Detailed Analysis */}
                <div className="border-t border-gray-200 pt-6">
                  <h3 className="text-lg font-medium text-gray-900 text-persian-primary mb-4">
                    تحلیل تفصیلی
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <div className="text-sm font-medium text-blue-900 text-persian-primary">
                        دسته پیش‌بینی شده
                      </div>
                      <div className="text-lg font-bold text-blue-600 text-persian-primary">
                        {result.predicted_class}
                      </div>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg">
                      <div className="text-sm font-medium text-green-900 text-persian-primary">
                        میزان اطمینان
                      </div>
                      <div className="text-lg font-bold text-green-600 text-persian-primary">
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <div className="text-sm font-medium text-purple-900 text-persian-primary">
                        زمان پردازش
                      </div>
                      <div className="text-lg font-bold text-purple-600 text-persian-primary">
                        {result.processing_time}s
                      </div>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex justify-end space-x-3 space-x-reverse border-t border-gray-200 pt-6">
                  <Button variant="outline">
                    ذخیره نتیجه
                  </Button>
                  <Button variant="outline">
                    اصلاح طبقه‌بندی
                  </Button>
                  <Button variant="primary">
                    طبقه‌بندی متن جدید
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="text-center">
                <DocumentTextIcon className="h-12 w-12 text-blue-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 text-persian-primary mb-2">
                  پردازش دسته‌ای
                </h3>
                <p className="text-sm text-gray-600 text-persian-primary mb-4">
                  طبقه‌بندی همزمان چندین سند
                </p>
                <Button variant="outline" size="sm">
                  شروع پردازش
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="text-center">
                <ChartBarIcon className="h-12 w-12 text-green-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 text-persian-primary mb-2">
                  تاریخچه طبقه‌بندی
                </h3>
                <p className="text-sm text-gray-600 text-persian-primary mb-4">
                  مشاهده طبقه‌بندی‌های قبلی
                </p>
                <Button variant="outline" size="sm">
                  مشاهده تاریخچه
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="text-center">
                <CpuChipIcon className="h-12 w-12 text-purple-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 text-persian-primary mb-2">
                  تنظیمات مدل
                </h3>
                <p className="text-sm text-gray-600 text-persian-primary mb-4">
                  تنظیم پارامترهای طبقه‌بندی
                </p>
                <Button variant="outline" size="sm">
                  تنظیمات
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </MainLayout>
  )
}