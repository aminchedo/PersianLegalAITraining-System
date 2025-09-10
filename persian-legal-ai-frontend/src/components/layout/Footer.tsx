'use client'

import React from 'react'
import Link from 'next/link'

export default function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-white border-t border-gray-200 mt-auto">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="md:flex md:items-center md:justify-between">
          <div className="flex justify-center space-x-6 space-x-reverse md:order-2">
            <Link
              href="/about"
              className="text-gray-400 hover:text-gray-500 text-persian-primary ui-text-small"
            >
              درباره ما
            </Link>
            <Link
              href="/privacy"
              className="text-gray-400 hover:text-gray-500 text-persian-primary ui-text-small"
            >
              حریم خصوصی
            </Link>
            <Link
              href="/terms"
              className="text-gray-400 hover:text-gray-500 text-persian-primary ui-text-small"
            >
              شرایط استفاده
            </Link>
            <Link
              href="/contact"
              className="text-gray-400 hover:text-gray-500 text-persian-primary ui-text-small"
            >
              تماس با ما
            </Link>
            <Link
              href="/help"
              className="text-gray-400 hover:text-gray-500 text-persian-primary ui-text-small"
            >
              راهنما
            </Link>
          </div>
          
          <div className="mt-8 md:mt-0 md:order-1">
            <div className="flex items-center justify-center md:justify-start">
              <div className="flex items-center">
                <div className="h-6 w-6 bg-indigo-600 rounded flex items-center justify-center ml-2">
                  <span className="text-white font-bold text-xs">ح</span>
                </div>
                <p className="text-gray-400 text-persian-primary ui-text-small">
                  © {currentYear} سامانه هوش مصنوعی حقوقی فارسی. تمامی حقوق محفوظ است.
                </p>
              </div>
            </div>
            <p className="mt-2 text-center md:text-right text-xs text-gray-400 text-persian-primary">
              ساخته شده با ❤️ برای بهبود سیستم قضایی ایران
            </p>
          </div>
        </div>
        
        <div className="mt-6 border-t border-gray-200 pt-6">
          <div className="text-center md:text-right">
            <p className="text-xs text-gray-400 text-persian-primary">
              این سامانه با استفاده از فناوری‌های پیشرفته هوش مصنوعی و پردازش زبان طبیعی فارسی توسعه یافته است.
              برای حمایت فنی با تیم پشتیبانی تماس بگیرید.
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}