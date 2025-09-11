import { z } from 'zod'
import { VALIDATION_RULES, ERROR_MESSAGES } from './constants'

/**
 * Common validation schemas
 */

// Email validation
export const emailSchema = z
  .string()
  .min(1, ERROR_MESSAGES.REQUIRED_FIELD)
  .email('آدرس ایمیل معتبر نیست')

// Password validation
export const passwordSchema = z
  .string()
  .min(VALIDATION_RULES.MIN_PASSWORD_LENGTH, `رمز عبور باید حداقل ${VALIDATION_RULES.MIN_PASSWORD_LENGTH} کاراکتر باشد`)
  .max(VALIDATION_RULES.MAX_PASSWORD_LENGTH, `رمز عبور نباید بیش از ${VALIDATION_RULES.MAX_PASSWORD_LENGTH} کاراکتر باشد`)

// Username validation
export const usernameSchema = z
  .string()
  .min(VALIDATION_RULES.MIN_USERNAME_LENGTH, `نام کاربری باید حداقل ${VALIDATION_RULES.MIN_USERNAME_LENGTH} کاراکتر باشد`)
  .max(VALIDATION_RULES.MAX_USERNAME_LENGTH, `نام کاربری نباید بیش از ${VALIDATION_RULES.MAX_USERNAME_LENGTH} کاراکتر باشد`)
  .regex(/^[a-zA-Z0-9_]+$/, 'نام کاربری فقط می‌تواند شامل حروف انگلیسی، اعداد و خط زیر باشد')

// Phone validation
export const phoneSchema = z
  .string()
  .min(1, ERROR_MESSAGES.REQUIRED_FIELD)
  .regex(VALIDATION_RULES.PHONE_REGEX, 'شماره موبایل معتبر نیست')

// National ID validation
export const nationalIdSchema = z
  .string()
  .min(1, ERROR_MESSAGES.REQUIRED_FIELD)
  .regex(VALIDATION_RULES.NATIONAL_ID_REGEX, 'کد ملی معتبر نیست')
  .refine(validateNationalId, 'کد ملی معتبر نیست')

// Text classification schema
export const textClassificationSchema = z.object({
  text: z
    .string()
    .min(10, 'متن باید حداقل ۱۰ کاراکتر باشد')
    .max(10000, 'متن نباید بیش از ۱۰۰۰۰ کاراکتر باشد'),
  options: z.object({
    confidence_threshold: z.number().min(0).max(1).optional(),
    max_results: z.number().min(1).max(10).optional()
  }).optional()
})

// Document upload schema
export const documentUploadSchema = z.object({
  file: z.instanceof(File, { message: 'فایل انتخاب نشده است' }),
  description: z.string().max(500, 'توضیحات نباید بیش از ۵۰۰ کاراکتر باشد').optional()
})

// User registration schema
export const userRegistrationSchema = z.object({
  username: usernameSchema,
  email: emailSchema,
  password: passwordSchema,
  confirmPassword: z.string(),
  firstName: z.string().min(1, ERROR_MESSAGES.REQUIRED_FIELD).max(50, 'نام نباید بیش از ۵۰ کاراکتر باشد'),
  lastName: z.string().min(1, ERROR_MESSAGES.REQUIRED_FIELD).max(50, 'نام خانوادگی نباید بیش از ۵۰ کاراکتر باشد'),
  phone: phoneSchema.optional(),
  nationalId: nationalIdSchema.optional()
}).refine((data) => data.password === data.confirmPassword, {
  message: 'رمز عبور و تکرار آن مطابقت ندارند',
  path: ['confirmPassword']
})

// User login schema
export const userLoginSchema = z.object({
  identifier: z.string().min(1, 'نام کاربری یا ایمیل الزامی است'),
  password: z.string().min(1, 'رمز عبور الزامی است')
})

// Settings schema
export const settingsSchema = z.object({
  theme: z.enum(['light', 'dark']).optional(),
  language: z.enum(['fa', 'en']).optional(),
  notifications: z.object({
    email: z.boolean().optional(),
    browser: z.boolean().optional(),
    mobile: z.boolean().optional()
  }).optional(),
  privacy: z.object({
    profileVisibility: z.enum(['public', 'private']).optional(),
    dataSharing: z.boolean().optional()
  }).optional()
})

/**
 * Validation helper functions
 */

/**
 * Validate Iranian national ID
 */
export function validateNationalId(nationalId: string): boolean {
  if (!/^\d{10}$/.test(nationalId)) return false
  
  const digits = nationalId.split('').map(Number)
  const checkDigit = digits[9]
  
  let sum = 0
  for (let i = 0; i < 9; i++) {
    sum += digits[i] * (10 - i)
  }
  
  const remainder = sum % 11
  
  if (remainder < 2) {
    return checkDigit === remainder
  } else {
    return checkDigit === 11 - remainder
  }
}

/**
 * Validate file type
 */
export function validateFileType(file: File, allowedTypes: string[]): boolean {
  return allowedTypes.includes(file.type)
}

/**
 * Validate file size
 */
export function validateFileSize(file: File, maxSize: number): boolean {
  return file.size <= maxSize
}

/**
 * Validate text length
 */
export function validateTextLength(text: string, minLength: number, maxLength: number): boolean {
  return text.length >= minLength && text.length <= maxLength
}

/**
 * Validate URL
 */
export function validateURL(url: string): boolean {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

/**
 * Validate Persian text
 */
export function validatePersianText(text: string): boolean {
  const persianRegex = /^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\.,;:!?()[\]{}"'-]+$/
  return persianRegex.test(text)
}

/**
 * Sanitize HTML
 */
export function sanitizeHTML(html: string): string {
  const div = document.createElement('div')
  div.textContent = html
  return div.innerHTML
}

/**
 * Validate password strength
 */
export function validatePasswordStrength(password: string): {
  isValid: boolean
  score: number
  feedback: string[]
} {
  const feedback: string[] = []
  let score = 0

  if (password.length >= 8) {
    score += 1
  } else {
    feedback.push('رمز عبور باید حداقل ۸ کاراکتر باشد')
  }

  if (/[a-z]/.test(password)) {
    score += 1
  } else {
    feedback.push('رمز عبور باید شامل حروف کوچک باشد')
  }

  if (/[A-Z]/.test(password)) {
    score += 1
  } else {
    feedback.push('رمز عبور باید شامل حروف بزرگ باشد')
  }

  if (/\d/.test(password)) {
    score += 1
  } else {
    feedback.push('رمز عبور باید شامل اعداد باشد')
  }

  if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
    score += 1
  } else {
    feedback.push('رمز عبور باید شامل کاراکترهای خاص باشد')
  }

  return {
    isValid: score >= 3,
    score,
    feedback
  }
}

/**
 * Form validation utilities
 */
export class FormValidator {
  private errors: Record<string, string> = {}

  addError(field: string, message: string): void {
    this.errors[field] = message
  }

  removeError(field: string): void {
    delete this.errors[field]
  }

  getError(field: string): string | undefined {
    return this.errors[field]
  }

  hasErrors(): boolean {
    return Object.keys(this.errors).length > 0
  }

  getErrors(): Record<string, string> {
    return this.errors
  }

  clear(): void {
    this.errors = {}
  }

  validate<T>(schema: z.ZodSchema<T>, data: unknown): T | null {
    try {
      const result = schema.parse(data)
      this.clear()
      return result
    } catch (error) {
      if (error instanceof z.ZodError) {
        this.clear()
        error.errors.forEach((err) => {
          const field = err.path.join('.')
          this.addError(field, err.message)
        })
      }
      return null
    }
  }
}

// Export validation schemas for reuse
export const validationSchemas = {
  email: emailSchema,
  password: passwordSchema,
  username: usernameSchema,
  phone: phoneSchema,
  nationalId: nationalIdSchema,
  textClassification: textClassificationSchema,
  documentUpload: documentUploadSchema,
  userRegistration: userRegistrationSchema,
  userLogin: userLoginSchema,
  settings: settingsSchema
}