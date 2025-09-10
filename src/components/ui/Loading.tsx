'use client'

import React from 'react'
import { clsx } from 'clsx'

interface LoadingProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  text?: string
  fullScreen?: boolean
  className?: string
}

export function Loading({
  size = 'md',
  text,
  fullScreen = false,
  className,
}: LoadingProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
    xl: 'h-16 w-16',
  }

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
    xl: 'text-xl',
  }

  const spinner = (
    <div
      className={clsx(
        'animate-spin rounded-full border-2 border-gray-300 border-t-primary-600',
        sizeClasses[size]
      )}
    />
  )

  const content = (
    <div className={clsx('flex flex-col items-center space-y-3', className)}>
      {spinner}
      {text && (
        <p className={clsx(
          'text-persian-primary text-gray-600',
          textSizeClasses[size]
        )}>
          {text}
        </p>
      )}
    </div>
  )

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-white bg-opacity-90">
        {content}
      </div>
    )
  }

  return content
}

interface LoadingSkeletonProps {
  lines?: number
  className?: string
}

export function LoadingSkeleton({ lines = 3, className }: LoadingSkeletonProps) {
  return (
    <div className={clsx('animate-pulse space-y-3', className)}>
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="space-y-2">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        </div>
      ))}
    </div>
  )
}

interface LoadingButtonProps {
  loading?: boolean
  children: React.ReactNode
  className?: string
  onClick?: () => void
}

export function LoadingButton({
  loading = false,
  children,
  className,
  onClick,
}: LoadingButtonProps) {
  return (
    <button
      className={clsx(
        'inline-flex items-center justify-center px-4 py-2 rounded-lg',
        'bg-primary-600 text-white font-medium',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'transition-colors duration-200',
        className
      )}
      disabled={loading}
      onClick={onClick}
    >
      {loading && (
        <div className="ml-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
      )}
      <span className="text-persian-primary">{children}</span>
    </button>
  )
}

export default Loading