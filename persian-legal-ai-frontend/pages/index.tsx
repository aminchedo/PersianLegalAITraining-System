import React from 'react'
import MainLayout from '../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../src/components/ui/Card'
import { Button } from '../src/components/ui/Button'
import { Badge } from '../src/components/ui/Badge'
import {
  DocumentTextIcon,
  CpuChipIcon,
  ChartBarIcon,
  AcademicCapIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'

export default function Dashboard() {
  // Mock data - in real app, this would come from API
  const systemStatus = {
    status: 'healthy',
    database_connected: true,
    ai_model_loaded: true,
    version: '2.0.0',
    uptime: '15 روز، 3 ساعت'
  }

  const stats = {
    totalDocuments: 1247,
    classifiedToday: 89,
    trainingProgress: 75,
    accuracy: 94.2
  }

  const recentActivity = [
    {
      id: 1,
      type: 'classification',
      message: 'طبقه‌بندی 15 سند جدید انجام شد',
      timestamp: '10 دقیقه پیش',
      status: 'success'
    },
    {
      id: 2,
      type: 'training',
      message: 'جلسه آموزش جدید شروع شد',
      timestamp: '25 دقیقه پیش',
      status: 'info'
    },
    {
      id: 3,
      type: 'document',
      message: '3 سند جدید آپلود شد',
      timestamp: '1 ساعت پیش',
      status: 'success'
    },
    {
      id: 4,
      type: 'system',
      message: 'بروزرسانی سیستم انجام شد',
      timestamp: '2 ساعت پیش',
      status: 'warning'
    }
  ]

  const quickActions = [
    {
      title: 'آپلود سند جدید',
      description: 'افزودن سند جدید به سیستم',
      href: '/documents/upload',
      icon: DocumentTextIcon,
      color: 'bg-blue-500'
    },
    {
      title: 'طبقه‌بندی متن',
      description: 'طبقه‌بندی هوشمند متن حقوقی',
      href: '/classification',
      icon: CpuChipIcon,
      color: 'bg-green-500'
    },
    {
      title: 'شروع آموزش',
      description: 'آموزش مدل جدید',
      href: '/training',
      icon: AcademicCapIcon,
      color: 'bg-purple-500'
    },
    {
      title: 'مشاهده گزارش‌ها',
      description: 'آنالیز و گزارش‌گیری',
      href: '/analytics',
      icon: ChartBarIcon,
      color: 'bg-orange-500'
    }
  ]

  return (
    <MainLayout title="داشبورد اصلی - سامانه هوش مصنوعی حقوقی فارسی">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="heading-2 text-gray-900">سامانه هوش مصنوعی حقوقی فارسی</h1>
            <p className="paragraph-normal text-gray-600">
              خوش آمدید! آخرین وضعیت سیستم و عملکرد را مشاهده کنید.
            </p>
          </div>
          <div className="flex items-center space-x-3 space-x-reverse">
            <Badge variant={systemStatus.status === 'healthy' ? 'success' : 'error'}>
              {systemStatus.status === 'healthy' ? 'سیستم سالم' : 'نیاز به بررسی'}
            </Badge>
            <Button variant="primary">
              تنظیمات سریع
            </Button>
          </div>
        </div>

        {/* System Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <CheckCircleIcon className="h-6 w-6 text-green-500 ml-2" />
              وضعیت سیستم
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="flex items-center">
                <div className={`h-3 w-3 rounded-full ml-2 ${
                  systemStatus.database_connected ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-persian-primary ui-text-small">
                  {systemStatus.database_connected ? 'پایگاه داده متصل' : 'پایگاه داده قطع'}
                </span>
              </div>
              <div className="flex items-center">
                <div className={`h-3 w-3 rounded-full ml-2 ${
                  systemStatus.ai_model_loaded ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-persian-primary ui-text-small">
                  {systemStatus.ai_model_loaded ? 'مدل AI بارگذاری شده' : 'مدل AI بارگذاری نشده'}
                </span>
              </div>
              <div className="flex items-center">
                <ClockIcon className="h-4 w-4 text-gray-400 ml-2" />
                <span className="text-persian-primary ui-text-small">
                  مدت فعالیت: {systemStatus.uptime}
                </span>
              </div>
              <div className="flex items-center">
                <span className="text-persian-primary ui-text-small">
                  نسخه: {systemStatus.version}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <DocumentTextIcon className="h-8 w-8 text-blue-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {stats.totalDocuments.toLocaleString('fa')}
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
                  <CpuChipIcon className="h-8 w-8 text-green-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {stats.classifiedToday.toLocaleString('fa')}
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    طبقه‌بندی امروز
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <AcademicCapIcon className="h-8 w-8 text-purple-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {stats.trainingProgress}%
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    پیشرفت آموزش
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ArrowTrendingUpIcon className="h-8 w-8 text-orange-600" />
                </div>
                <div className="mr-4">
                  <div className="text-2xl font-bold text-gray-900 text-persian-primary">
                    {stats.accuracy}%
                  </div>
                  <p className="text-persian-primary ui-text-small text-gray-600">
                    دقت سیستم
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>عملیات سریع</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {quickActions.map((action, index) => (
                  <Link key={index} href={action.href}>
                    <div className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer">
                      <div className="flex items-center">
                        <div className={`p-2 rounded-lg ${action.color}`}>
                          <action.icon className="h-6 w-6 text-white" />
                        </div>
                        <div className="mr-3">
                          <h3 className="text-sm font-medium text-gray-900 text-persian-primary">
                            {action.title}
                          </h3>
                          <p className="text-xs text-gray-600 text-persian-primary">
                            {action.description}
                          </p>
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <CardTitle>فعالیت‌های اخیر</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivity.map((activity) => (
                  <div key={activity.id} className="flex items-center">
                    <div className={`flex-shrink-0 h-2 w-2 rounded-full ${
                      activity.status === 'success' ? 'bg-green-500' :
                      activity.status === 'warning' ? 'bg-yellow-500' :
                      activity.status === 'error' ? 'bg-red-500' : 'bg-blue-500'
                    }`}></div>
                    <div className="mr-3 flex-1">
                      <p className="text-sm text-gray-900 text-persian-primary">
                        {activity.message}
                      </p>
                      <p className="text-xs text-gray-500 text-persian-primary">
                        {activity.timestamp}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 pt-4 border-t border-gray-200">
                <Link href="/activity">
                  <Button variant="ghost" size="sm" className="w-full">
                    مشاهده همه فعالیت‌ها
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </MainLayout>
  )
}