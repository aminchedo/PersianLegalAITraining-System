'use client'

import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { motion } from 'framer-motion'
import {
  HomeIcon,
  DocumentTextIcon,
  CpuChipIcon,
  ChartBarIcon,
  CogIcon,
  AcademicCapIcon,
  FolderIcon,
  MagnifyingGlassIcon,
  CloudArrowUpIcon,
  ClockIcon,
  UsersIcon,
  ArrowTrendingUpIcon,
  DocumentChartBarIcon,
  Cog6ToothIcon,
  UserGroupIcon
} from '@heroicons/react/24/outline'
import { useUIContext } from '../../contexts/UIContext'
import { Badge } from '../ui/Badge'

interface NavigationItem {
  name: string
  href: string
  icon: React.ComponentType<any>
  badge?: string | number
  children?: NavigationItem[]
}

const navigation: NavigationItem[] = [
  {
    name: 'داشبورد اصلی',
    href: '/',
    icon: HomeIcon
  },
  {
    name: 'مدیریت اسناد',
    href: '/documents',
    icon: FolderIcon,
    badge: 'جدید',
    children: [
      { name: 'لیست اسناد', href: '/documents', icon: DocumentTextIcon },
      { name: 'جستجوی اسناد', href: '/documents/search', icon: MagnifyingGlassIcon },
      { name: 'آپلود سند', href: '/documents/upload', icon: CloudArrowUpIcon }
    ]
  },
  {
    name: 'طبقه‌بندی هوشمند',
    href: '/classification',
    icon: CpuChipIcon,
    children: [
      { name: 'طبقه‌بندی متن', href: '/classification', icon: CpuChipIcon },
      { name: 'پردازش دسته‌ای', href: '/classification/batch', icon: CpuChipIcon },
      { name: 'تاریخچه طبقه‌بندی', href: '/classification/history', icon: ClockIcon }
    ]
  },
  {
    name: 'آموزش مدل',
    href: '/training',
    icon: AcademicCapIcon,
    children: [
      { name: 'داشبورد آموزش', href: '/training', icon: ChartBarIcon },
      { name: 'جلسات آموزش', href: '/training/sessions', icon: ClockIcon },
      { name: 'مدیریت مدل‌ها', href: '/training/models', icon: CpuChipIcon }
    ]
  },
  {
    name: 'آنالیز و گزارش',
    href: '/analytics',
    icon: ChartBarIcon,
    children: [
      { name: 'داشبورد آنالیز', href: '/analytics', icon: ArrowTrendingUpIcon },
      { name: 'تولید گزارش', href: '/analytics/reports', icon: DocumentChartBarIcon },
      { name: 'متریک‌های عملکرد', href: '/analytics/performance', icon: ChartBarIcon }
    ]
  },
  {
    name: 'تنظیمات',
    href: '/settings',
    icon: CogIcon,
    children: [
      { name: 'تنظیمات سیستم', href: '/settings/system', icon: Cog6ToothIcon },
      { name: 'مدیریت کاربران', href: '/settings/users', icon: UserGroupIcon }
    ]
  }
]

export default function Sidebar() {
  const router = useRouter()
  const { sidebarOpen } = useUIContext()

  const isActive = (href: string) => {
    if (href === '/') {
      return router.pathname === '/'
    }
    return router.pathname.startsWith(href)
  }

  const isChildActive = (children: NavigationItem[]) => {
    return children.some(child => isActive(child.href))
  }

  return (
    <motion.aside
      initial={false}
      animate={{
        width: sidebarOpen ? 256 : 64,
        transition: { duration: 0.3, ease: 'easeInOut' }
      }}
      className="fixed top-16 right-0 bottom-0 z-30 bg-white shadow-lg border-l border-gray-200 overflow-hidden"
    >
      <div className="h-full overflow-y-auto py-4">
        <nav className="space-y-1 px-2">
          {navigation.map((item) => (
            <div key={item.name}>
              {/* Main Navigation Item */}
              <Link
                href={item.href}
                className={`group flex items-center px-3 py-3 rounded-lg transition-all duration-200 ${
                  isActive(item.href) || (item.children && isChildActive(item.children))
                    ? 'bg-indigo-50 text-indigo-700 border-r-2 border-indigo-500'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <item.icon 
                  className={`h-6 w-6 ml-3 transition-colors ${
                    isActive(item.href) || (item.children && isChildActive(item.children))
                      ? 'text-indigo-500'
                      : 'text-gray-400 group-hover:text-gray-500'
                  }`}
                />
                
                <motion.div
                  initial={false}
                  animate={{
                    opacity: sidebarOpen ? 1 : 0,
                    width: sidebarOpen ? 'auto' : 0,
                    transition: { duration: 0.2 }
                  }}
                  className="flex items-center justify-between flex-1 overflow-hidden"
                >
                  <span className="text-persian-primary ui-text-medium font-medium whitespace-nowrap">
                    {item.name}
                  </span>
                  
                  {item.badge && (
                    <Badge variant="primary" size="sm">
                      {item.badge}
                    </Badge>
                  )}
                </motion.div>
              </Link>

              {/* Sub Navigation */}
              {item.children && sidebarOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ 
                    opacity: isActive(item.href) || isChildActive(item.children) ? 1 : 0,
                    height: isActive(item.href) || isChildActive(item.children) ? 'auto' : 0,
                    transition: { duration: 0.3 }
                  }}
                  className="overflow-hidden"
                >
                  <div className="pr-6 space-y-1 mt-2">
                    {item.children.map((child) => (
                      <Link
                        key={child.name}
                        href={child.href}
                        className={`group flex items-center px-3 py-2 rounded-md transition-all duration-200 ${
                          isActive(child.href)
                            ? 'bg-indigo-100 text-indigo-700'
                            : 'text-gray-500 hover:bg-gray-50 hover:text-gray-700'
                        }`}
                      >
                        <child.icon 
                          className={`h-4 w-4 ml-2 transition-colors ${
                            isActive(child.href)
                              ? 'text-indigo-500'
                              : 'text-gray-400 group-hover:text-gray-500'
                          }`}
                        />
                        <span className="text-persian-primary ui-text-small whitespace-nowrap">
                          {child.name}
                        </span>
                      </Link>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          ))}
        </nav>
      </div>
    </motion.aside>
  )
}