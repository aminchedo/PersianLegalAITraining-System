'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'

interface UIContextType {
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
  theme: 'light' | 'dark'
  setTheme: (theme: 'light' | 'dark') => void
  loading: boolean
  setLoading: (loading: boolean) => void
  notifications: Notification[]
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void
  removeNotification: (id: string) => void
  language: 'fa' | 'en'
  setLanguage: (lang: 'fa' | 'en') => void
}

interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  timestamp: Date
  duration?: number
  read?: boolean
}

const UIContext = createContext<UIContextType | undefined>(undefined)

export function UIProvider({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [loading, setLoading] = useState(false)
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [language, setLanguage] = useState<'fa' | 'en'>('fa')

  // Load preferences from localStorage
  useEffect(() => {
    const savedSidebarState = localStorage.getItem('sidebarOpen')
    const savedTheme = localStorage.getItem('theme')
    const savedLanguage = localStorage.getItem('language')

    if (savedSidebarState !== null) {
      setSidebarOpen(JSON.parse(savedSidebarState))
    }

    if (savedTheme) {
      setTheme(savedTheme as 'light' | 'dark')
    }

    if (savedLanguage) {
      setLanguage(savedLanguage as 'fa' | 'en')
    }

    // Set initial sidebar state based on screen size
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarOpen(false)
      }
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Save preferences to localStorage
  useEffect(() => {
    localStorage.setItem('sidebarOpen', JSON.stringify(sidebarOpen))
  }, [sidebarOpen])

  useEffect(() => {
    localStorage.setItem('theme', theme)
    // Apply theme to document
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  useEffect(() => {
    localStorage.setItem('language', language)
    // Apply language to document
    document.documentElement.lang = language
    document.documentElement.dir = language === 'fa' ? 'rtl' : 'ltr'
  }, [language])

  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const newNotification: Notification = {
      ...notification,
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date()
    }

    setNotifications(prev => [newNotification, ...prev])

    // Auto remove notification
    if (notification.duration !== 0) {
      setTimeout(() => {
        removeNotification(newNotification.id)
      }, notification.duration || 5000)
    }
  }

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id))
  }

  const value: UIContextType = {
    sidebarOpen,
    setSidebarOpen,
    theme,
    setTheme,
    loading,
    setLoading,
    notifications,
    addNotification,
    removeNotification,
    language,
    setLanguage
  }

  return (
    <UIContext.Provider value={value}>
      {children}
    </UIContext.Provider>
  )
}

export function useUIContext() {
  const context = useContext(UIContext)
  if (context === undefined) {
    throw new Error('useUIContext must be used within a UIProvider')
  }
  return context
}

export type { UIContextType, Notification }