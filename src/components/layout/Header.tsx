'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import {
  Bars3Icon,
  BellIcon,
  MagnifyingGlassIcon,
  UserCircleIcon,
  SunIcon,
  MoonIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
} from '@heroicons/react/24/outline'
import { Menu, Transition } from '@headlessui/react'
import { Fragment } from 'react'
import { useUIContext } from '../../contexts/UIContext'
import { useAuth } from '../../hooks/useAuth'
import Input from '../ui/Input'
import Badge from '../ui/Badge'

export default function Header() {
  const { sidebarOpen, setSidebarOpen, theme, setTheme } = useUIContext()
  const { user, logout } = useAuth()
  const [searchQuery, setSearchQuery] = useState('')

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light')
  }

  return (
    <header className="fixed top-0 right-0 left-0 z-40 bg-white border-b border-gray-200 shadow-sm">
      <div className="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        {/* Right side - Logo and Menu Toggle */}
        <div className="flex items-center space-x-4 space-x-reverse">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-colors"
          >
            <Bars3Icon className="h-6 w-6" />
          </button>

          <Link href="/" className="flex items-center space-x-3 space-x-reverse">
            <div className="h-8 w-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">ح</span>
            </div>
            <div className="hidden sm:block">
              <h1 className="heading-4 text-gray-900 mb-0">
                سامانه هوش مصنوعی حقوقی
              </h1>
            </div>
          </Link>
        </div>

        {/* Center - Search */}
        <div className="flex-1 max-w-lg mx-8 hidden md:block">
          <Input
            type="text"
            placeholder="جستجو در سیستم..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}
            className="w-full"
          />
        </div>

        {/* Left side - Actions and User Menu */}
        <div className="flex items-center space-x-4 space-x-reverse">
          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-colors"
          >
            {theme === 'light' ? (
              <MoonIcon className="h-6 w-6" />
            ) : (
              <SunIcon className="h-6 w-6" />
            )}
          </button>

          {/* Notifications */}
          <div className="relative">
            <button className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-colors">
              <BellIcon className="h-6 w-6" />
              <Badge
                variant="danger"
                size="sm"
                className="absolute -top-1 -left-1 h-5 w-5 flex items-center justify-center p-0 text-xs"
              >
                3
              </Badge>
            </button>
          </div>

          {/* User Menu */}
          <Menu as="div" className="relative">
            <Menu.Button className="flex items-center space-x-2 space-x-reverse p-2 rounded-lg hover:bg-gray-100 transition-colors">
              <UserCircleIcon className="h-8 w-8 text-gray-500" />
              <div className="hidden sm:block text-right">
                <p className="ui-text-small font-medium text-gray-700">
                  {user?.name || 'کاربر مهمان'}
                </p>
                <p className="caption-text text-gray-500">
                  {user?.role === 'admin' ? 'مدیر سیستم' : 'کاربر'}
                </p>
              </div>
            </Menu.Button>

            <Transition
              as={Fragment}
              enter="transition ease-out duration-100"
              enterFrom="transform opacity-0 scale-95"
              enterTo="transform opacity-100 scale-100"
              leave="transition ease-in duration-75"
              leaveFrom="transform opacity-100 scale-100"
              leaveTo="transform opacity-0 scale-95"
            >
              <Menu.Items className="absolute left-0 mt-2 w-56 origin-top-left divide-y divide-gray-100 rounded-lg bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                <div className="p-1">
                  <Menu.Item>
                    {({ active }) => (
                      <Link
                        href="/profile"
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } group flex w-full items-center rounded-md px-3 py-2 ui-text-small text-gray-700`}
                      >
                        <UserCircleIcon className="ml-2 h-5 w-5" />
                        پروفایل کاربری
                      </Link>
                    )}
                  </Menu.Item>
                  <Menu.Item>
                    {({ active }) => (
                      <Link
                        href="/settings"
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } group flex w-full items-center rounded-md px-3 py-2 ui-text-small text-gray-700`}
                      >
                        <Cog6ToothIcon className="ml-2 h-5 w-5" />
                        تنظیمات
                      </Link>
                    )}
                  </Menu.Item>
                </div>
                <div className="p-1">
                  <Menu.Item>
                    {({ active }) => (
                      <button
                        onClick={logout}
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } group flex w-full items-center rounded-md px-3 py-2 ui-text-small text-red-600`}
                      >
                        <ArrowRightOnRectangleIcon className="ml-2 h-5 w-5" />
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

      {/* Mobile Search */}
      <div className="md:hidden px-4 pb-3 border-t border-gray-200">
        <Input
          type="text"
          placeholder="جستجو در سیستم..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          leftIcon={<MagnifyingGlassIcon className="h-5 w-5" />}
          fullWidth
        />
      </div>
    </header>
  )
}