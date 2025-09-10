'use client'

import React, { useState } from 'react'
import MainLayout from '../../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../../src/components/ui/Card'
import Button from '../../src/components/ui/Button'
import Input from '../../src/components/ui/Input'
import Badge from '../../src/components/ui/Badge'
import {
  MagnifyingGlassIcon,
  FunnelIcon,
  DocumentTextIcon,
  CalendarIcon,
  TagIcon,
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'

export default function SearchPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [isSearching, setIsSearching] = useState(false)
  const [filters, setFilters] = useState({
    documentType: '',
    dateRange: '',
    classification: '',
    confidence: 0,
  })

  const handleSearch = async () => {
    setIsSearching(true)
    // Simulate search API call
    setTimeout(() => {
      setIsSearching(false)
      // Set mock results
    }, 1500)
  }

  return (
    <MainLayout
      title="جستجوی اسناد - سامانه هوش مصنوعی حقوقی فارسی"
      description="جستجوی پیشرفته در اسناد حقوقی"
    >
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="heading-1 text-gray-900">
            جستجوی پیشرفته اسناد
          </h1>
          <p className="paragraph-normal text-gray-600">
            با استفاده از فیلترهای مختلف، اسناد مورد نظر خود را پیدا کنید
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Search Filters */}
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>فیلترهای جستجو</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Document Type */}
                  <div>
                    <label className="form-label">نوع سند</label>
                    <select
                      value={filters.documentType}
                      onChange={(e) => setFilters(prev => ({ ...prev, documentType: e.target.value }))}
                      className="form-input w-full"
                    >
                      <option value="">همه انواع</option>
                      <option value="contract">قرارداد</option>
                      <option value="law">قانون</option>
                      <option value="regulation">آیین‌نامه</option>
                      <option value="judgment">رأی قضایی</option>
                    </select>
                  </div>

                  {/* Date Range */}
                  <div>
                    <label className="form-label">بازه زمانی</label>
                    <select
                      value={filters.dateRange}
                      onChange={(e) => setFilters(prev => ({ ...prev, dateRange: e.target.value }))}
                      className="form-input w-full"
                    >
                      <option value="">همه تاریخ‌ها</option>
                      <option value="today">امروز</option>
                      <option value="week">هفته گذشته</option>
                      <option value="month">ماه گذشته</option>
                      <option value="year">سال گذشته</option>
                    </select>
                  </div>

                  {/* Classification */}
                  <div>
                    <label className="form-label">طبقه‌بندی</label>
                    <select
                      value={filters.classification}
                      onChange={(e) => setFilters(prev => ({ ...prev, classification: e.target.value }))}
                      className="form-input w-full"
                    >
                      <option value="">همه طبقه‌ها</option>
                      <option value="civil">حقوق مدنی</option>
                      <option value="criminal">حقوق جزا</option>
                      <option value="commercial">حقوق تجاری</option>
                      <option value="administrative">حقوق اداری</option>
                    </select>
                  </div>

                  {/* Confidence Level */}
                  <div>
                    <label className="form-label">حداقل میزان اطمینان</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={filters.confidence}
                      onChange={(e) => setFilters(prev => ({ ...prev, confidence: parseInt(e.target.value) }))}
                      className="w-full"
                    />
                    <div className="flex justify-between ui-text-xs text-gray-500 text-persian-primary">
                      <span>0%</span>
                      <span className="persian-numbers">{filters.confidence}%</span>
                      <span>100%</span>
                    </div>
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    fullWidth
                    onClick={() => setFilters({ documentType: '', dateRange: '', classification: '', confidence: 0 })}
                  >
                    پاک کردن فیلترها
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Search Results */}
          <div className="lg:col-span-3 space-y-6">
            {/* Search Bar */}
            <Card>
              <CardContent>
                <div className="flex space-x-4 space-x-reverse">
                  <div className="flex-1">
                    <Input
                      placeholder="جستجو در محتوای اسناد، عنوان، یا کلمات کلیدی..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}
                      onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                    />
                  </div>
                  <Button
                    loading={isSearching}
                    onClick={handleSearch}
                    leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}
                  >
                    جستجو
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Search Results */}
            {searchQuery && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Card>
                  <CardHeader>
                    <div className="flex justify-between items-center">
                      <CardTitle>نتایج جستجو</CardTitle>
                      <Badge variant="info" size="sm">
                        ۱۲ نتیجه یافت شد
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Mock Search Results */}
                      {[1, 2, 3, 4, 5].map((item) => (
                        <div
                          key={item}
                          className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                        >
                          <div className="flex items-start space-x-4 space-x-reverse">
                            <div className="p-2 bg-blue-50 rounded-lg">
                              <DocumentTextIcon className="h-6 w-6 text-blue-600" />
                            </div>
                            <div className="flex-1">
                              <h3 className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                                قرارداد خرید املاک تجاری شماره {item}
                              </h3>
                              <p className="ui-text-small text-gray-600 text-persian-primary mt-1">
                                این سند حاوی شرایط و ضوابط خرید املاک تجاری بوده و شامل جزئیات قیمت، 
                                زمان تحویل و تعهدات طرفین می‌باشد...
                              </p>
                              <div className="flex items-center space-x-4 space-x-reverse mt-2">
                                <Badge variant="primary" size="sm">قرارداد</Badge>
                                <span className="ui-text-xs text-gray-500 text-persian-primary">
                                  اطمینان: ۹۵%
                                </span>
                                <span className="ui-text-xs text-gray-500 text-persian-primary">
                                  ۱۴۰۲/۰۹/۱۵
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Pagination */}
                    <div className="flex justify-center mt-8">
                      <div className="flex space-x-2 space-x-reverse">
                        <Button variant="outline" size="sm">قبلی</Button>
                        <Button variant="primary" size="sm">1</Button>
                        <Button variant="outline" size="sm">2</Button>
                        <Button variant="outline" size="sm">3</Button>
                        <Button variant="outline" size="sm">بعدی</Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Search Tips */}
            {!searchQuery && (
              <Card>
                <CardHeader>
                  <CardTitle>راهنمای جستجو</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3 space-x-reverse">
                      <MagnifyingGlassIcon className="h-5 w-5 text-primary-600 mt-1" />
                      <div>
                        <h4 className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                          جستجوی ساده
                        </h4>
                        <p className="ui-text-small text-gray-600 text-persian-primary">
                          کلمات کلیدی خود را وارد کنید. مثال: "قرارداد خرید"
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3 space-x-reverse">
                      <FunnelIcon className="h-5 w-5 text-primary-600 mt-1" />
                      <div>
                        <h4 className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                          استفاده از فیلترها
                        </h4>
                        <p className="ui-text-small text-gray-600 text-persian-primary">
                          از فیلترهای سمت راست برای محدود کردن نتایج استفاده کنید
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3 space-x-reverse">
                      <TagIcon className="h-5 w-5 text-primary-600 mt-1" />
                      <div>
                        <h4 className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                          جستجوی پیشرفته
                        </h4>
                        <p className="ui-text-small text-gray-600 text-persian-primary">
                          از علائم "+" و "-" برای شامل یا حذف کردن کلمات استفاده کنید
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}