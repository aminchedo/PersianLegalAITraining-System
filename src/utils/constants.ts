// Application Constants
export const APP_NAME = 'سامانه هوش مصنوعی حقوقی فارسی'
export const APP_VERSION = '2.0.0'
export const APP_DESCRIPTION = 'سامانه جامع پردازش و طبقه‌بندی اسناد حقوقی با استفاده از هوش مصنوعی'

// API Configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api'
export const API_TIMEOUT = 10000 // 10 seconds

// Pagination
export const DEFAULT_PAGE_SIZE = 10
export const MAX_PAGE_SIZE = 100

// File Upload
export const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB
export const ALLOWED_FILE_TYPES = [
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain'
]
export const ALLOWED_FILE_EXTENSIONS = ['.pdf', '.doc', '.docx', '.txt']

// Classification
export const MIN_CONFIDENCE_THRESHOLD = 0.5
export const MAX_TEXT_LENGTH = 10000
export const MIN_TEXT_LENGTH = 10

// Document Types
export const DOCUMENT_TYPES = {
  CONTRACT: 'قرارداد',
  LAW: 'قانون',
  REGULATION: 'آیین‌نامه',
  JUDGMENT: 'رأی قضایی',
  LEGAL_OPINION: 'نظریه حقوقی',
  OTHER: 'سایر'
} as const

// Classification Categories
export const CLASSIFICATION_CATEGORIES = {
  CIVIL_LAW: 'حقوق مدنی',
  CRIMINAL_LAW: 'حقوق جزا',
  COMMERCIAL_LAW: 'حقوق تجاری',
  ADMINISTRATIVE_LAW: 'حقوق اداری',
  CONSTITUTIONAL_LAW: 'حقوق قانون اساسی',
  INTERNATIONAL_LAW: 'حقوق بین‌الملل',
  FAMILY_LAW: 'حقوق خانواده',
  PROPERTY_LAW: 'حقوق اموال',
  CONTRACT_LAW: 'حقوق قراردادها',
  OTHER: 'سایر'
} as const

// Status Types
export const STATUS_TYPES = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled'
} as const

// Priority Levels
export const PRIORITY_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  URGENT: 'urgent'
} as const

// Theme Configuration
export const THEME_COLORS = {
  PRIMARY: '#3B82F6',
  SECONDARY: '#6B7280',
  SUCCESS: '#10B981',
  WARNING: '#F59E0B',
  ERROR: '#EF4444',
  INFO: '#06B6D4'
} as const

// Local Storage Keys
export const STORAGE_KEYS = {
  THEME: 'persian-legal-ai-theme',
  USER: 'persian-legal-ai-user',
  TOKEN: 'persian-legal-ai-token',
  PREFERENCES: 'persian-legal-ai-preferences',
  RECENT_SEARCHES: 'persian-legal-ai-recent-searches'
} as const

// Date Formats
export const DATE_FORMATS = {
  PERSIAN_SHORT: 'yyyy/MM/dd',
  PERSIAN_LONG: 'yyyy/MM/dd - HH:mm',
  DISPLAY: 'dd/MM/yyyy',
  API: 'yyyy-MM-dd\'T\'HH:mm:ss.SSSxxx'
} as const

// Validation Rules
export const VALIDATION_RULES = {
  MIN_PASSWORD_LENGTH: 8,
  MAX_PASSWORD_LENGTH: 128,
  MIN_USERNAME_LENGTH: 3,
  MAX_USERNAME_LENGTH: 50,
  EMAIL_REGEX: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  PHONE_REGEX: /^(\+98|0)?9\d{9}$/,
  NATIONAL_ID_REGEX: /^\d{10}$/
} as const

// Animation Durations
export const ANIMATION_DURATION = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500
} as const

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'خطا در اتصال به سرور',
  UNAUTHORIZED: 'دسترسی غیرمجاز',
  FORBIDDEN: 'عدم دسترسی',
  NOT_FOUND: 'موردی یافت نشد',
  SERVER_ERROR: 'خطای داخلی سرور',
  VALIDATION_ERROR: 'خطا در اعتبارسنجی داده‌ها',
  FILE_TOO_LARGE: 'اندازه فایل بیش از حد مجاز است',
  INVALID_FILE_TYPE: 'نوع فایل پشتیبانی نمی‌شود',
  REQUIRED_FIELD: 'این فیلد الزامی است'
} as const

// Success Messages
export const SUCCESS_MESSAGES = {
  SAVED: 'با موفقیت ذخیره شد',
  UPDATED: 'با موفقیت به‌روزرسانی شد',
  DELETED: 'با موفقیت حذف شد',
  UPLOADED: 'فایل با موفقیت آپلود شد',
  CLASSIFIED: 'طبقه‌بندی با موفقیت انجام شد',
  SENT: 'با موفقیت ارسال شد'
} as const

// Routes
export const ROUTES = {
  HOME: '/',
  DOCUMENTS: '/documents',
  DOCUMENT_DETAIL: '/documents/[id]',
  DOCUMENT_UPLOAD: '/documents/upload',
  DOCUMENT_SEARCH: '/documents/search',
  CLASSIFICATION: '/classification',
  CLASSIFICATION_BATCH: '/classification/batch',
  CLASSIFICATION_HISTORY: '/classification/history',
  TRAINING: '/training',
  TRAINING_SESSIONS: '/training/sessions',
  TRAINING_MODELS: '/training/models',
  ANALYTICS: '/analytics',
  ANALYTICS_REPORTS: '/analytics/reports',
  ANALYTICS_PERFORMANCE: '/analytics/performance',
  SETTINGS: '/settings',
  SETTINGS_SYSTEM: '/settings/system',
  SETTINGS_USERS: '/settings/users'
} as const

export type DocumentType = keyof typeof DOCUMENT_TYPES
export type ClassificationCategory = keyof typeof CLASSIFICATION_CATEGORIES
export type StatusType = keyof typeof STATUS_TYPES
export type PriorityLevel = keyof typeof PRIORITY_LEVELS