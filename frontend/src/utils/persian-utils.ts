/**
 * Persian Legal AI - Utility Functions
 * توابع کمکی سیستم هوش مصنوعی حقوقی فارسی
 */

// Persian number conversion
export const toPersianNumbers = (input: string | number): string => {
  const persianNumbers = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
  const englishNumbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  
  let result = input.toString();
  for (let i = 0; i < englishNumbers.length; i++) {
    result = result.replace(new RegExp(englishNumbers[i], 'g'), persianNumbers[i]);
  }
  return result;
};

// English number conversion
export const toEnglishNumbers = (input: string): string => {
  const persianNumbers = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
  const englishNumbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  
  let result = input;
  for (let i = 0; i < persianNumbers.length; i++) {
    result = result.replace(new RegExp(persianNumbers[i], 'g'), englishNumbers[i]);
  }
  return result;
};

// Persian date formatting
export const formatPersianDate = (date: Date | string): string => {
  const d = new Date(date);
  const persianMonths = [
    'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
    'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'
  ];
  
  // Simple Gregorian to Persian approximation
  const year = d.getFullYear();
  const month = d.getMonth();
  const day = d.getDate();
  
  const persianYear = year - 621;
  const persianMonth = persianMonths[month] || persianMonths[0];
  
  return `${toPersianNumbers(day)} ${persianMonth} ${toPersianNumbers(persianYear)}`;
};

// Persian time formatting
export const formatPersianTime = (date: Date | string): string => {
  const d = new Date(date);
  const hours = d.getHours();
  const minutes = d.getMinutes();
  const seconds = d.getSeconds();
  
  return `${toPersianNumbers(hours.toString().padStart(2, '0'))}:${toPersianNumbers(minutes.toString().padStart(2, '0'))}:${toPersianNumbers(seconds.toString().padStart(2, '0'))}`;
};

// Persian relative time
export const formatPersianRelativeTime = (date: Date | string): string => {
  const now = new Date();
  const targetDate = new Date(date);
  const diffMs = now.getTime() - targetDate.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  
  if (diffMinutes < 1) return 'همین الان';
  if (diffMinutes < 60) return `${toPersianNumbers(diffMinutes)} دقیقه پیش`;
  if (diffHours < 24) return `${toPersianNumbers(diffHours)} ساعت پیش`;
  if (diffDays < 7) return `${toPersianNumbers(diffDays)} روز پیش`;
  
  return formatPersianDate(date);
};

// Persian text direction detection
export const hasPersianText = (text: string): boolean => {
  const persianRegex = /[\u0600-\u06FF\u0750-\u077F]/;
  return persianRegex.test(text);
};

// Persian text normalization
export const normalizePersianText = (text: string): string => {
  // Normalize Persian/Arabic characters
  return text
    .replace(/ي/g, 'ی')  // Arabic yeh to Persian yeh
    .replace(/ك/g, 'ک')  // Arabic kaf to Persian kaf
    .replace(/‌+/g, '‌')  // Multiple ZWNJs to single
    .replace(/\s+/g, ' ') // Multiple spaces to single
    .trim();
};

// Persian currency formatting
export const formatPersianCurrency = (amount: number): string => {
  const formatted = new Intl.NumberFormat('fa-IR', {
    style: 'currency',
    currency: 'IRR',
    minimumFractionDigits: 0,
  }).format(amount);
  
  return toPersianNumbers(formatted);
};

// Persian number formatting with thousands separator
export const formatPersianNumber = (num: number): string => {
  const formatted = new Intl.NumberFormat('fa-IR').format(num);
  return toPersianNumbers(formatted);
};

// Persian file size formatting
export const formatPersianFileSize = (bytes: number): string => {
  const units = ['بایت', 'کیلوبایت', 'مگابایت', 'گیگابایت', 'ترابایت'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  const formattedSize = size < 10 ? size.toFixed(1) : Math.round(size).toString();
  return `${toPersianNumbers(formattedSize)} ${units[unitIndex]}`;
};

// Persian percentage formatting
export const formatPersianPercentage = (value: number): string => {
  return `${toPersianNumbers(Math.round(value))}٪`;
};

// Persian status translations
export const translateStatus = (status: string): string => {
  const statusMap: Record<string, string> = {
    'pending': 'در انتظار',
    'running': 'در حال اجرا',
    'completed': 'تکمیل شده',
    'failed': 'ناموفق',
    'cancelled': 'لغو شده',
    'paused': 'متوقف شده',
    'error': 'خطا',
    'success': 'موفق',
    'warning': 'هشدار',
    'info': 'اطلاعات',
    'active': 'فعال',
    'inactive': 'غیرفعال',
    'online': 'آنلاین',
    'offline': 'آفلاین',
    'connecting': 'در حال اتصال',
    'disconnected': 'قطع شده',
    'loading': 'در حال بارگذاری',
    'ready': 'آماده',
    'busy': 'مشغول',
    'available': 'در دسترس',
    'unavailable': 'در دسترس نیست'
  };
  
  return statusMap[status.toLowerCase()] || status;
};

// Persian priority translations
export const translatePriority = (priority: string): string => {
  const priorityMap: Record<string, string> = {
    'low': 'کم',
    'medium': 'متوسط',
    'high': 'بالا',
    'critical': 'بحرانی',
    'urgent': 'فوری'
  };
  
  return priorityMap[priority.toLowerCase()] || priority;
};

// Persian sorting for arrays
export const sortPersianText = (arr: string[]): string[] => {
  return arr.sort((a, b) => a.localeCompare(b, 'fa-IR'));
};

// Persian search/filter function
export const searchPersianText = (text: string, query: string): boolean => {
  const normalizedText = normalizePersianText(text.toLowerCase());
  const normalizedQuery = normalizePersianText(query.toLowerCase());
  
  return normalizedText.includes(normalizedQuery);
};

// Persian validation functions
export const validatePersianName = (name: string): boolean => {
  const persianNameRegex = /^[\u0600-\u06FF\s]+$/;
  return persianNameRegex.test(name.trim());
};

export const validatePersianPhoneNumber = (phone: string): boolean => {
  const persianPhoneRegex = /^(\+98|0)?9\d{9}$/;
  const englishPhone = toEnglishNumbers(phone.replace(/\s/g, ''));
  return persianPhoneRegex.test(englishPhone);
};

export const validatePersianEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Persian text truncation
export const truncatePersianText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

// Persian word count
export const countPersianWords = (text: string): number => {
  const normalizedText = normalizePersianText(text);
  return normalizedText.split(/\s+/).filter(word => word.length > 0).length;
};

// Persian text statistics
export const getPersianTextStats = (text: string) => {
  const normalizedText = normalizePersianText(text);
  const words = normalizedText.split(/\s+/).filter(word => word.length > 0);
  const sentences = normalizedText.split(/[.!?؟]+/).filter(s => s.trim().length > 0);
  const paragraphs = normalizedText.split(/\n\s*\n/).filter(p => p.trim().length > 0);
  
  return {
    characters: text.length,
    charactersNoSpaces: text.replace(/\s/g, '').length,
    words: words.length,
    sentences: sentences.length,
    paragraphs: paragraphs.length,
    averageWordsPerSentence: sentences.length > 0 ? Math.round(words.length / sentences.length) : 0
  };
};

// Persian color utilities
export const getPersianStatusColor = (status: string): string => {
  const colorMap: Record<string, string> = {
    'success': '#10b981',
    'error': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6',
    'pending': '#6b7280',
    'running': '#8b5cf6',
    'completed': '#10b981',
    'failed': '#ef4444'
  };
  
  return colorMap[status.toLowerCase()] || '#6b7280';
};

// Persian keyboard layout detection
export const isPersianKeyboard = (text: string): boolean => {
  const persianChars = text.match(/[\u0600-\u06FF]/g) || [];
  const englishChars = text.match(/[a-zA-Z]/g) || [];
  
  return persianChars.length > englishChars.length;
};

// Persian RTL text direction
export const getTextDirection = (text: string): 'rtl' | 'ltr' => {
  return hasPersianText(text) ? 'rtl' : 'ltr';
};

// Persian accessibility helpers
export const getPersianAriaLabel = (key: string): string => {
  const ariaLabels: Record<string, string> = {
    'close': 'بستن',
    'open': 'باز کردن',
    'menu': 'منو',
    'search': 'جستجو',
    'filter': 'فیلتر',
    'sort': 'مرتب‌سازی',
    'refresh': 'به‌روزرسانی',
    'edit': 'ویرایش',
    'delete': 'حذف',
    'save': 'ذخیره',
    'cancel': 'لغو',
    'submit': 'ارسال',
    'next': 'بعدی',
    'previous': 'قبلی',
    'first': 'اولین',
    'last': 'آخرین',
    'loading': 'در حال بارگذاری',
    'error': 'خطا',
    'success': 'موفقیت',
    'warning': 'هشدار'
  };
  
  return ariaLabels[key.toLowerCase()] || key;
};

// Export all utilities as default object
export default {
  toPersianNumbers,
  toEnglishNumbers,
  formatPersianDate,
  formatPersianTime,
  formatPersianRelativeTime,
  hasPersianText,
  normalizePersianText,
  formatPersianCurrency,
  formatPersianNumber,
  formatPersianFileSize,
  formatPersianPercentage,
  translateStatus,
  translatePriority,
  sortPersianText,
  searchPersianText,
  validatePersianName,
  validatePersianPhoneNumber,
  validatePersianEmail,
  truncatePersianText,
  countPersianWords,
  getPersianTextStats,
  getPersianStatusColor,
  isPersianKeyboard,
  getTextDirection,
  getPersianAriaLabel
};