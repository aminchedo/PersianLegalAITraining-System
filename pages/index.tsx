'use client'

import React from 'react'
import MainLayout from '../src/components/layout/MainLayout'
import { Card, CardHeader, CardTitle, CardContent } from '../src/components/ui/Card'
import Button from '../src/components/ui/Button'
import Badge from '../src/components/ui/Badge'
import {
  DocumentTextIcon,
  CpuChipIcon,
  ChartBarIcon,
  CloudArrowUpIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'
import Link from 'next/link'

interface StatCardProps {
  title: string
  value: string | number
  change: string
  changeType: 'positive' | 'negative' | 'neutral'
  icon: React.ComponentType<any>
}

function StatCard({ title, value, change, changeType, icon: Icon }: StatCardProps) {
  const changeColors = {
    positive: 'text-green-600',
    negative: 'text-red-600',
    neutral: 'text-gray-600',
  }

  return (
    <Card hover className="h-full">
      <CardContent>
        <div className="flex items-center justify-between">
          <div>
            <p className="ui-text-small text-gray-600 text-persian-primary">
              {title}
            </p>
            <p className="heading-3 text-gray-900 mb-0 persian-numbers">
              {value}
            </p>
            <p className={`ui-text-xs ${changeColors[changeType]} text-persian-primary`}>
              {change}
            </p>
          </div>
          <div className="p-3 bg-primary-50 rounded-lg">
            <Icon className="h-6 w-6 text-primary-600" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

interface QuickActionProps {
  title: string
  description: string
  href: string
  icon: React.ComponentType<any>
  variant?: 'primary' | 'secondary'
}

function QuickAction({ title, description, href, icon: Icon, variant = 'primary' }: QuickActionProps) {
  return (
    <Link href={href}>
      <Card hover className="h-full cursor-pointer">
        <CardContent>
          <div className="flex items-start space-x-4 space-x-reverse">
            <div className={`p-3 rounded-lg ${
              variant === 'primary' 
                ? 'bg-primary-50' 
                : 'bg-gray-50'
            }`}>
              <Icon className={`h-6 w-6 ${
                variant === 'primary' 
                  ? 'text-primary-600' 
                  : 'text-gray-600'
              }`} />
            </div>
            <div className="flex-1">
              <h3 className="ui-text-medium font-medium text-gray-900 text-persian-primary">
                {title}
              </h3>
              <p className="ui-text-small text-gray-600 text-persian-primary mt-1">
                {description}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}

interface ActivityItem {
  id: string
  type: 'classification' | 'upload' | 'training' | 'report'
  title: string
  description: string
  time: string
  status: 'success' | 'warning' | 'error' | 'info'
}

function ActivityFeed() {
  const activities: ActivityItem[] = [
    {
      id: '1',
      type: 'classification',
      title: 'طبقه‌بندی سند جدید',
      description: 'سند قرارداد خرید با دقت ۹۵٪ طبقه‌بندی شد',
      time: '۵ دقیقه پیش',
      status: 'success',
    },
    {
      id: '2',
      type: 'upload',
      title: 'آپلود دسته‌ای اسناد',
      description: '۲۳ سند جدید به سیستم اضافه شد',
      time: '۱۵ دقیقه پیش',
      status: 'info',
    },
    {
      id: '3',
      type: 'training',
      title: 'آموزش مدل',
      description: 'آموزش مدل جدید با موفقیت آغاز شد',
      time: '۳۰ دقیقه پیش',
      status: 'info',
    },
    {
      id: '4',
      type: 'report',
      title: 'گزارش خطا',
      description: 'خطا در پردازش ۲ سند شناسایی شد',
      time: '۱ ساعت پیش',
      status: 'error',
    },
  ]

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
      default:
        return <ClockIcon className="h-5 w-5 text-blue-500" />
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>فعالیت‌های اخیر</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {activities.map((activity) => (
            <div key={activity.id} className="flex items-start space-x-3 space-x-reverse">
              <div className="flex-shrink-0 mt-1">
                {getStatusIcon(activity.status)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="ui-text-small font-medium text-gray-900 text-persian-primary">
                  {activity.title}
                </p>
                <p className="ui-text-xs text-gray-500 text-persian-primary">
                  {activity.description}
                </p>
                <p className="ui-text-xs text-gray-400 text-persian-primary mt-1">
                  {activity.time}
                </p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export default function HomePage() {
  return (
    <MainLayout
      title="داشبورد اصلی - سامانه هوش مصنوعی حقوقی فارسی"
      description="داشبورد اصلی سامانه جامع پردازش و طبقه‌بندی اسناد حقوقی با هوش مصنوعی"
    >
      <div className="space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
          <div>
            <h1 className="heading-1 text-gray-900">
              سامانه هوش مصنوعی حقوقی فارسی
            </h1>
            <p className="paragraph-normal text-gray-600">
              داشبورد مدیریت و کنترل سیستم طبقه‌بندی هوشمند اسناد حقوقی
            </p>
          </div>
          <div className="flex space-x-3 space-x-reverse">
            <Badge variant="success">آنلاین</Badge>
            <Badge variant="info">نسخه ۲.۰.۰</Badge>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <StatCard
              title="کل اسناد"
              value="۱,۲۳۴"
              change="+۱۲٪ نسبت به ماه قبل"
              changeType="positive"
              icon={DocumentTextIcon}
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <StatCard
              title="طبقه‌بندی امروز"
              value="۸۷"
              change="+۵٪ نسبت به دیروز"
              changeType="positive"
              icon={CpuChipIcon}
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <StatCard
              title="دقت مدل"
              value="۹۴.۵٪"
              change="+۲.۱٪ بهبود"
              changeType="positive"
              icon={ChartBarIcon}
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.4 }}
          >
            <StatCard
              title="وضعیت سیستم"
              value="عالی"
              change="۹۹.۹٪ آپتایم"
              changeType="positive"
              icon={CpuChipIcon}
            />
          </motion.div>
        </div>

        {/* Quick Actions */}
        <div>
          <h2 className="heading-3 text-gray-900 mb-6">
            عملیات سریع
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.5 }}
            >
              <QuickAction
                title="طبقه‌بندی متن"
                description="متن حقوقی خود را وارد کنید و طبقه‌بندی دریافت کنید"
                href="/classification"
                icon={CpuChipIcon}
              />
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.6 }}
            >
              <QuickAction
                title="آپلود سند"
                description="سند جدید خود را آپلود کنید و پردازش کنید"
                href="/documents/upload"
                icon={CloudArrowUpIcon}
                variant="secondary"
              />
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.7 }}
            >
              <QuickAction
                title="مشاهده گزارش‌ها"
                description="آمار و گزارش‌های تفصیلی سیستم را مشاهده کنید"
                href="/analytics"
                icon={ChartBarIcon}
                variant="secondary"
              />
            </motion.div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Activity Feed */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.8 }}
            className="lg:col-span-2"
          >
            <ActivityFeed />
          </motion.div>

          {/* System Status */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.9 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>وضعیت سیستم</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      سرور اصلی
                    </span>
                    <Badge variant="success" size="sm">آنلاین</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      پایگاه داده
                    </span>
                    <Badge variant="success" size="sm">متصل</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      مدل هوش مصنوعی
                    </span>
                    <Badge variant="success" size="sm">فعال</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="ui-text-small text-gray-600 text-persian-primary">
                      API سرویس
                    </span>
                    <Badge variant="warning" size="sm">کند</Badge>
                  </div>
                </div>
                
                <div className="mt-6 pt-4 border-t border-gray-200">
                  <Link href="/settings/system">
                    <Button variant="outline" size="sm" fullWidth>
                      مشاهده جزئیات
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </MainLayout>
  )
}