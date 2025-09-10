'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useRouter } from 'next/router'
import Head from 'next/head'
import Header from './Header'
import Sidebar from './Sidebar'
import Footer from './Footer'
import { useUIContext } from '../../contexts/UIContext'
import { useAuth } from '../../contexts/AuthContext'
import { Toaster } from 'react-hot-toast'

interface MainLayoutProps {
  children: React.ReactNode
  title?: string
  description?: string
  showSidebar?: boolean
  showFooter?: boolean
  className?: string
}

export default function MainLayout({
  children,
  title = 'سامانه هوش مصنوعی حقوقی فارسی',
  description = 'سامانه جامع پردازش و طبقه‌بندی اسناد حقوقی با استفاده از هوش مصنوعی',
  showSidebar = true,
  showFooter = true,
  className = ''
}: MainLayoutProps) {
  const router = useRouter()
  const { sidebarOpen, setSidebarOpen, theme } = useUIContext()
  const { user, isLoading } = useAuth()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    const handleRouteChange = () => {
      if (window.innerWidth < 1024) {
        setSidebarOpen(false)
      }
    }

    router.events.on('routeChangeComplete', handleRouteChange)
    return () => {
      router.events.off('routeChangeComplete', handleRouteChange)
    }
  }, [router.events, setSidebarOpen])

  if (!mounted) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-persian-primary ui-text-medium text-gray-600">
            در حال بارگذاری...
          </p>
        </div>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta charSet="UTF-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <link rel="icon" href="/favicon.ico" />
        
        {/* Persian Language Meta */}
        <meta name="language" content="fa" />
        <meta name="dir" content="rtl" />
        
        {/* Open Graph */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content={title} />
        <meta property="og:description" content={description} />
        <meta property="og:locale" content="fa_IR" />
        
        {/* Twitter Card */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={title} />
        <meta name="twitter:description" content={description} />
      </Head>

      <div 
        className={`min-h-screen bg-gray-50 transition-colors duration-200 ${
          theme === 'dark' ? 'dark' : ''
        }`}
        dir="rtl"
      >
        <Toaster 
          position="top-right"
          containerClassName="text-right"
          toastOptions={{
            duration: 4000,
            style: {
              fontFamily: 'var(--font-primary)',
              direction: 'rtl',
              textAlign: 'right'
            }
          }}
        />

        {/* Header */}
        <Header />

        <div className="flex">
          {/* Sidebar */}
          {showSidebar && (
            <>
              {/* Mobile Sidebar Overlay */}
              <AnimatePresence>
                {sidebarOpen && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-20 lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                  >
                    <div className="absolute inset-0 bg-black bg-opacity-50" />
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Sidebar */}
              <Sidebar />
            </>
          )}

          {/* Main Content */}
          <main 
            className={`flex-1 transition-all duration-300 ${
              showSidebar 
                ? sidebarOpen 
                  ? 'lg:mr-64' 
                  : 'lg:mr-16'
                : ''
            } ${className}`}
          >
            <div className="min-h-screen flex flex-col">
              {/* Page Content */}
              <div className="flex-1 p-4 sm:p-6 lg:p-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="max-w-full"
                >
                  {children}
                </motion.div>
              </div>

              {/* Footer */}
              {showFooter && <Footer />}
            </div>
          </main>
        </div>

        {/* Loading Overlay */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-center justify-center bg-white bg-opacity-90"
            >
              <div className="text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-600 mx-auto mb-4"></div>
                <p className="text-persian-primary ui-text-large text-gray-600">
                  در حال پردازش...
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  )
}