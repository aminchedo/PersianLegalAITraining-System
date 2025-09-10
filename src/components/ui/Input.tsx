'use client'

import React from 'react'
import { clsx } from 'clsx'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  helper?: string
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  fullWidth?: boolean
}

export function Input({
  label,
  error,
  helper,
  leftIcon,
  rightIcon,
  fullWidth = false,
  className,
  id,
  ...props
}: InputProps) {
  const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`

  return (
    <div className={clsx('space-y-1', fullWidth && 'w-full')}>
      {label && (
        <label htmlFor={inputId} className="form-label text-gray-700">
          {label}
        </label>
      )}
      
      <div className="relative">
        {leftIcon && (
          <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
            <span className="text-gray-400">{leftIcon}</span>
          </div>
        )}
        
        <input
          id={inputId}
          className={clsx(
            'form-input block w-full rounded-lg border-gray-300 shadow-sm transition-colors duration-200',
            'focus:border-primary-500 focus:ring-primary-500 focus:ring-1',
            'placeholder:text-gray-400',
            error && 'error-border',
            leftIcon && 'pr-10',
            rightIcon && 'pl-10',
            className
          )}
          {...props}
        />
        
        {rightIcon && (
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <span className="text-gray-400">{rightIcon}</span>
          </div>
        )}
      </div>
      
      {helper && !error && (
        <p className="form-help">{helper}</p>
      )}
      
      {error && (
        <p className="form-error">{error}</p>
      )}
    </div>
  )
}

export default Input