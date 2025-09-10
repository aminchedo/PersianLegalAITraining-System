'use client'

import React, { Fragment } from 'react'
import { Menu, Transition } from '@headlessui/react'
import { 
  Bars3Icon, 
  BellIcon, 
  UserCircleIcon,
  SunIcon,
  MoonIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon
} from '@heroicons/react/24/outline'
import { useUIContext } from '../../contexts/UIContext'
import { useAuth } from '../../contexts/AuthContext'
import { Badge } from '../ui/Badge'

export default function Header() {
  const { sidebarOpen, setSidebarOpen, theme, setTheme, notifications } = useUIContext()
  const { user, logout } = useAuth()

  const unreadNotifications = notifications.filter(n => !n.read).length

  return (
    <header className="bg-white shadow-sm border-b border-gray-200 fixed top-0 left-0 right-0 z-40">
      <div className="mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Right Side - Logo and Menu Toggle */}
          <div className="flex items-center">
            <button
              type="button"
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <span className="sr-only">باز کردن منوی اصلی</span>
              <Bars3Icon className="h-6 w-6" />
            </button>
            
            <div className="flex-shrink-0 flex items-center mr-4">
              <div className="flex items-center">
                <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">ح</span>
                </div>
                <div className="mr-3 hidden sm:block">
                  <h1 className="text-lg font-semibold text-gray-900 text-persian-primary">
                    سامانه هوش مصنوعی حقوقی
                  </h1>
                </div>
              </div>
            </div>
          </div>

          {/* Left Side - User Menu and Actions */}
          <div className="flex items-center space-x-4 space-x-reverse">
            {/* Theme Toggle */}
            <button
              type="button"
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            >
              <span className="sr-only">تغییر تم</span>
              {theme === 'light' ? (
                <MoonIcon className="h-6 w-6" />
              ) : (
                <SunIcon className="h-6 w-6" />
              )}
            </button>

            {/* Notifications */}
            <button
              type="button"
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 relative"
            >
              <span className="sr-only">مشاهده اعلان‌ها</span>
              <BellIcon className="h-6 w-6" />
              {unreadNotifications > 0 && (
                <div className="absolute -top-1 -left-1">
                  <Badge variant="error" size="sm">
                    {unreadNotifications > 99 ? '99+' : unreadNotifications}
                  </Badge>
                </div>
              )}
            </button>

            {/* User Menu */}
            <Menu as="div" className="relative">
              <div>
                <Menu.Button className="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                  <span className="sr-only">باز کردن منوی کاربر</span>
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <div className="flex-shrink-0">
                      {user?.avatar ? (
                        <img
                          className="h-8 w-8 rounded-full"
                          src={user.avatar}
                          alt={user.name}
                        />
                      ) : (
                        <UserCircleIcon className="h-8 w-8 text-gray-400" />
                      )}
                    </div>
                    <div className="hidden md:block text-right">
                      <div className="text-sm font-medium text-gray-700 text-persian-primary">
                        {user?.name || 'کاربر میهمان'}
                      </div>
                      <div className="text-xs text-gray-500 text-persian-primary">
                        {user?.role === 'admin' ? 'مدیر سیستم' : 
                         user?.role === 'analyst' ? 'تحلیلگر' : 'کاربر'}
                      </div>
                    </div>
                  </div>
                </Menu.Button>
              </div>

              <Transition
                as={Fragment}
                enter="transition ease-out duration-100"
                enterFrom="transform opacity-0 scale-95"
                enterTo="transform opacity-100 scale-100"
                leave="transition ease-in duration-75"
                leaveFrom="transform opacity-100 scale-100"
                leaveTo="transform opacity-0 scale-95"
              >
                <Menu.Items className="origin-top-left absolute left-0 mt-2 w-48 rounded-md shadow-lg py-1 bg-white ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
                  <div className="px-4 py-3 border-b border-gray-100">
                    <p className="text-sm text-persian-primary">وارد شده به عنوان</p>
                    <p className="text-sm font-medium text-gray-900 text-persian-primary truncate">
                      {user?.email}
                    </p>
                  </div>

                  <Menu.Item>
                    {({ active }) => (
                      <a
                        href="/profile"
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } flex items-center px-4 py-2 text-sm text-gray-700 text-persian-primary`}
                      >
                        <UserCircleIcon className="ml-3 h-5 w-5 text-gray-400" />
                        پروفایل کاربری
                      </a>
                    )}
                  </Menu.Item>

                  <Menu.Item>
                    {({ active }) => (
                      <a
                        href="/settings"
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } flex items-center px-4 py-2 text-sm text-gray-700 text-persian-primary`}
                      >
                        <Cog6ToothIcon className="ml-3 h-5 w-5 text-gray-400" />
                        تنظیمات
                      </a>
                    )}
                  </Menu.Item>

                  <div className="border-t border-gray-100">
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={logout}
                          className={`${
                            active ? 'bg-gray-100' : ''
                          } flex items-center w-full text-right px-4 py-2 text-sm text-gray-700 text-persian-primary`}
                        >
                          <ArrowRightOnRectangleIcon className="ml-3 h-5 w-5 text-gray-400" />
                          خروج از سیستم
                        </button>
                      )}
                    </Menu.Item>
                  </div>
                </Menu.Items>
              </Transition>
            </Menu>
          </div>
        </div>
      </div>
    </header>
  )
}