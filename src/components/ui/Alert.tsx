'use client'

import React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { clsx } from 'clsx'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XCircleIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'

const alertVariants = cva(
  'rounded-lg p-4 border',
  {
    variants: {
      variant: {
        success: 'bg-green-50 border-green-200 text-green-800',
        warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
        error: 'bg-red-50 border-red-200 text-red-800',
        info: 'bg-blue-50 border-blue-200 text-blue-800',
      },
    },
    defaultVariants: {
      variant: 'info',
    },
  }
)

const iconMap = {
  success: CheckCircleIcon,
  warning: ExclamationTriangleIcon,
  error: XCircleIcon,
  info: InformationCircleIcon,
}

interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {
  title?: string
  children: React.ReactNode
  dismissible?: boolean
  onDismiss?: () => void
  icon?: boolean
}

export function Alert({
  className,
  variant = 'info',
  title,
  children,
  dismissible = false,
  onDismiss,
  icon = true,
  ...props
}: AlertProps) {
  const IconComponent = iconMap[variant!]

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={clsx(alertVariants({ variant, className }))}
      {...props}
    >
      <div className="flex">
        {icon && (
          <div className="flex-shrink-0 ml-2">
            <IconComponent className="h-5 w-5" />
          </div>
        )}
        <div className="flex-1">
          {title && (
            <h3 className="ui-text-medium font-medium mb-1">
              {title}
            </h3>
          )}
          <div className="ui-text-small text-persian-primary">
            {children}
          </div>
        </div>
        {dismissible && (
          <div className="flex-shrink-0 mr-2">
            <button
              className="inline-flex text-gray-400 hover:text-gray-600 transition-colors"
              onClick={onDismiss}
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
        )}
      </div>
    </motion.div>
  )
}

interface AlertListProps {
  alerts: Array<{
    id: string
    variant: 'success' | 'warning' | 'error' | 'info'
    title?: string
    message: string
    dismissible?: boolean
  }>
  onDismiss?: (id: string) => void
}

export function AlertList({ alerts, onDismiss }: AlertListProps) {
  return (
    <div className="space-y-3">
      <AnimatePresence>
        {alerts.map((alert) => (
          <Alert
            key={alert.id}
            variant={alert.variant}
            title={alert.title}
            dismissible={alert.dismissible}
            onDismiss={() => onDismiss?.(alert.id)}
          >
            {alert.message}
          </Alert>
        ))}
      </AnimatePresence>
    </div>
  )
}

export default Alert