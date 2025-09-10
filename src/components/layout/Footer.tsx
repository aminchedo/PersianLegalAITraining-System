'use client'

import React from 'react'
import Link from 'next/link'

export default function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-white border-t border-gray-200 mt-auto">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          {/* Copyright */}
          <div className="text-center md:text-right">
            <p className="ui-text-small text-gray-500 text-persian-primary">
              © {currentYear} سامانه هوش مصنوعی حقوقی فارسی. تمامی حقوق محفوظ است.
            </p>
          </div>

          {/* Links */}
          <div className="flex space-x-6 space-x-reverse">
            <Link
              href="/privacy"
              className="ui-text-small text-gray-500 hover:text-gray-700 transition-colors text-persian-primary"
            >
              حریم خصوصی
            </Link>
            <Link
              href="/terms"
              className="ui-text-small text-gray-500 hover:text-gray-700 transition-colors text-persian-primary"
            >
              شرایط استفاده
            </Link>
            <Link
              href="/support"
              className="ui-text-small text-gray-500 hover:text-gray-700 transition-colors text-persian-primary"
            >
              پشتیبانی
            </Link>
            <Link
              href="/docs"
              className="ui-text-small text-gray-500 hover:text-gray-700 transition-colors text-persian-primary"
            >
              مستندات
            </Link>
          </div>

          {/* Version */}
          <div className="text-center md:text-left">
            <span className="ui-text-xs text-gray-400 text-persian-primary">
              نسخه 2.0.0
            </span>
          </div>
        </div>
      </div>
    </footer>
  )
}