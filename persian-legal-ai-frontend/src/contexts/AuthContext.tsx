'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'
import { useRouter } from 'next/router'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'user' | 'analyst'
  avatar?: string
  permissions: string[]
  lastLogin?: Date
  isActive: boolean
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<boolean>
  logout: () => void
  updateUser: (userData: Partial<User>) => void
  hasPermission: (permission: string) => boolean
  hasRole: (role: string) => boolean
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  // Check authentication status on mount
  useEffect(() => {
    checkAuthStatus()
  }, [])

  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem('authToken')
      if (!token) {
        setIsLoading(false)
        return
      }

      // Verify token with backend
      const response = await fetch('/api/auth/verify', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const userData = await response.json()
        setUser({
          id: userData.id,
          name: userData.name,
          email: userData.email,
          role: userData.role,
          avatar: userData.avatar,
          permissions: userData.permissions || [],
          lastLogin: userData.last_login ? new Date(userData.last_login) : undefined,
          isActive: userData.is_active
        })
      } else {
        // Token is invalid
        localStorage.removeItem('authToken')
        setUser(null)
      }
    } catch (error) {
      console.error('Auth check failed:', error)
      localStorage.removeItem('authToken')
      setUser(null)
    } finally {
      setIsLoading(false)
    }
  }

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      setIsLoading(true)

      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
      })

      if (response.ok) {
        const data = await response.json()
        
        // Store token
        localStorage.setItem('authToken', data.token)
        
        // Set user data
        setUser({
          id: data.user.id,
          name: data.user.name,
          email: data.user.email,
          role: data.user.role,
          avatar: data.user.avatar,
          permissions: data.user.permissions || [],
          lastLogin: new Date(),
          isActive: data.user.is_active
        })

        return true
      } else {
        const errorData = await response.json()
        throw new Error(errorData.message || 'ورود ناموفق')
      }
    } catch (error) {
      console.error('Login failed:', error)
      return false
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    // Clear local storage
    localStorage.removeItem('authToken')
    
    // Clear user state
    setUser(null)
    
    // Redirect to login
    router.push('/auth/login')
  }

  const updateUser = (userData: Partial<User>) => {
    if (user) {
      setUser({ ...user, ...userData })
    }
  }

  const hasPermission = (permission: string): boolean => {
    if (!user) return false
    return user.permissions.includes(permission) || user.role === 'admin'
  }

  const hasRole = (role: string): boolean => {
    if (!user) return false
    return user.role === role || user.role === 'admin'
  }

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    logout,
    updateUser,
    hasPermission,
    hasRole
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export type { User, AuthContextType }