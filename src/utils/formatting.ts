import { format, parseISO } from 'date-fns'

/**
 * Format bytes to human readable format
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 بایت'

  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['بایت', 'کیلوبایت', 'مگابایت', 'گیگابایت', 'ترابایت']

  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

/**
 * Format number to Persian digits
 */
export function toPersianDigits(str: string | number): string {
  const persianDigits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
  return str.toString().replace(/\d/g, (digit) => persianDigits[parseInt(digit)])
}

/**
 * Convert Persian digits to English
 */
export function toEnglishDigits(str: string): string {
  const englishDigits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  const persianDigits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
  
  let result = str
  persianDigits.forEach((persian, index) => {
    result = result.replace(new RegExp(persian, 'g'), englishDigits[index])
  })
  
  return result
}

/**
 * Format number with Persian thousand separators
 */
export function formatNumber(num: number): string {
  return toPersianDigits(num.toLocaleString('fa-IR'))
}

/**
 * Format percentage
 */
export function formatPercentage(value: number, decimals = 1): string {
  return toPersianDigits(value.toFixed(decimals)) + '٪'
}

/**
 * Format currency (Rial)
 */
export function formatCurrency(amount: number): string {
  return toPersianDigits(amount.toLocaleString('fa-IR')) + ' ریال'
}

/**
 * Format date to Persian
 */
export function formatDate(date: string | Date, formatStr = 'yyyy/MM/dd'): string {
  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    const formatted = format(dateObj, formatStr)
    return toPersianDigits(formatted)
  } catch {
    return 'تاریخ نامعتبر'
  }
}

/**
 * Format date with time
 */
export function formatDateTime(date: string | Date): string {
  return formatDate(date, 'yyyy/MM/dd - HH:mm')
}

/**
 * Format relative time (time ago)
 */
export function formatRelativeTime(date: string | Date): string {
  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date
    const now = new Date()
    const diffInSeconds = Math.floor((now.getTime() - dateObj.getTime()) / 1000)

    if (diffInSeconds < 60) {
      return 'چند ثانیه پیش'
    } else if (diffInSeconds < 3600) {
      const minutes = Math.floor(diffInSeconds / 60)
      return `${toPersianDigits(minutes)} دقیقه پیش`
    } else if (diffInSeconds < 86400) {
      const hours = Math.floor(diffInSeconds / 3600)
      return `${toPersianDigits(hours)} ساعت پیش`
    } else if (diffInSeconds < 2592000) {
      const days = Math.floor(diffInSeconds / 86400)
      return `${toPersianDigits(days)} روز پیش`
    } else if (diffInSeconds < 31536000) {
      const months = Math.floor(diffInSeconds / 2592000)
      return `${toPersianDigits(months)} ماه پیش`
    } else {
      const years = Math.floor(diffInSeconds / 31536000)
      return `${toPersianDigits(years)} سال پیش`
    }
  } catch {
    return 'زمان نامعتبر'
  }
}

/**
 * Format duration in seconds to human readable
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${toPersianDigits(seconds.toFixed(1))} ثانیه`
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${toPersianDigits(minutes)} دقیقه و ${toPersianDigits(remainingSeconds.toFixed(0))} ثانیه`
  } else {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${toPersianDigits(hours)} ساعت و ${toPersianDigits(minutes)} دقیقه`
  }
}

/**
 * Truncate text with ellipsis
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

/**
 * Format confidence score
 */
export function formatConfidence(confidence: number): string {
  return `${formatPercentage(confidence * 100)} اطمینان`
}

/**
 * Format file extension
 */
export function formatFileExtension(filename: string): string {
  const ext = filename.split('.').pop()?.toUpperCase()
  return ext || 'نامشخص'
}

/**
 * Format classification result
 */
export function formatClassificationResult(category: string, confidence: number): string {
  return `${category} (${formatPercentage(confidence * 100)})`
}

/**
 * Format phone number
 */
export function formatPhoneNumber(phone: string): string {
  const cleaned = toEnglishDigits(phone.replace(/\D/g, ''))
  
  if (cleaned.startsWith('98')) {
    const number = cleaned.substring(2)
    return `+۹۸ ${toPersianDigits(number.substring(0, 3))} ${toPersianDigits(number.substring(3, 6))} ${toPersianDigits(number.substring(6))}`
  } else if (cleaned.startsWith('0')) {
    const number = cleaned.substring(1)
    return `${toPersianDigits(number.substring(0, 3))} ${toPersianDigits(number.substring(3, 6))} ${toPersianDigits(number.substring(6))}`
  }
  
  return toPersianDigits(phone)
}

/**
 * Format national ID
 */
export function formatNationalId(id: string): string {
  const cleaned = toEnglishDigits(id.replace(/\D/g, ''))
  if (cleaned.length === 10) {
    return `${toPersianDigits(cleaned.substring(0, 3))}-${toPersianDigits(cleaned.substring(3, 9))}-${toPersianDigits(cleaned.substring(9))}`
  }
  return toPersianDigits(id)
}

/**
 * Format address
 */
export function formatAddress(address: string, maxLength = 100): string {
  return truncateText(address, maxLength)
}