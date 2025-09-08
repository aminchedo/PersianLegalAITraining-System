#!/bin/bash
# 🚀 اسکریپت نصب و راه‌اندازی کامل داشبورد پیشرفته
# Persian Legal AI Dashboard - Complete Setup Script

echo "⚖️  شروع نصب سیستم هوش مصنوعی حقوقی فارسی"
echo "=================================================="

# رنگ‌ها برای خروجی زیبا
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# تابع برای چاپ پیام‌های رنگی
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_step() {
    echo -e "${PURPLE}🔧 $1${NC}"
}

# بررسی وجود Node.js
check_nodejs() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js موجود است: $NODE_VERSION"
        return 0
    else
        print_error "Node.js یافت نشد"
        return 1
    fi
}

# بررسی وجود npm
check_npm() {
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        print_success "npm موجود است: $NPM_VERSION"
        return 0
    else
        print_error "npm یافت نشد"
        return 1
    fi
}

# نصب Node.js اگر موجود نباشد
install_nodejs() {
    print_step "نصب Node.js..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install node
        else
            print_error "Homebrew یافت نشد. لطفاً Node.js را دستی نصب کنید"
            print_info "از این لینک: https://nodejs.org/"
            exit 1
        fi
    else
        print_warning "سیستم‌عامل شناسایی نشد"
        print_info "لطفاً Node.js را دستی نصب کنید از: https://nodejs.org/"
        exit 1
    fi
}

# ایجاد ساختار پروژه
create_project_structure() {
    print_step "ایجاد ساختار پروژه..."
    
    # ایجاد پوشه اصلی
    mkdir -p persian-legal-ai-dashboard
    cd persian-legal-ai-dashboard
    
    # ایجاد پوشه‌های فرعی
    mkdir -p src/components src/hooks src/api src/utils public
    
    print_success "ساختار پروژه ایجاد شد"
}

# ایجاد package.json
create_package_json() {
    print_step "ایجاد package.json..."
    
    cat > package.json << 'EOF'
{
  "name": "persian-legal-ai-dashboard",
  "version": "1.0.0",
  "description": "داشبورد پیشرفته سیستم هوش مصنوعی حقوقی فارسی",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "recharts": "^2.8.0",
    "lucide-react": "^0.294.0",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "devDependencies": {
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.31"
  }
}
EOF
    
    print_success "package.json ایجاد شد"
}

# نصب Tailwind CSS
setup_tailwind() {
    print_step "راه‌اندازی Tailwind CSS..."
    
    # نصب Tailwind
    npm install -D tailwindcss postcss autoprefixer
    npx tailwindcss init -p
    
    # تنظیم tailwind.config.js
    cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'vazir': ['Vazirmatn', 'Tahoma', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      colors: {
        'blue': {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        'purple': {
          50: '#faf5ff',
          100: '#f3e8ff',
          200: '#e9d5ff',
          300: '#d8b4fe',
          400: '#c084fc',
          500: '#a855f7',
          600: '#9333ea',
          700: '#7c3aed',
          800: '#6b21a8',
          900: '#581c87',
        },
        'green': {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        }
      }
    },
  },
  plugins: [],
}
EOF
    
    print_success "Tailwind CSS راه‌اندازی شد"
}

# ایجاد فایل‌های CSS
create_css_files() {
    print_step "ایجاد فایل‌های CSS..."
    
    cat > src/index.css << 'EOF'
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: 'Vazirmatn', 'Tahoma', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  direction: rtl;
}

* {
  box-sizing: border-box;
}

.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}

.scrollbar-hide::-webkit-scrollbar {
  display: none;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #3b82f6;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #2563eb;
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.fade-in {
  animation: fadeIn 0.3s ease-out;
}

/* RTL specific styles */
[dir="rtl"] .ml-auto {
  margin-left: 0;
  margin-right: auto;
}

[dir="rtl"] .mr-auto {
  margin-right: 0;
  margin-left: auto;
}
EOF
    
    print_success "فایل‌های CSS ایجاد شدند"
}

# ایجاد index.js
create_index_js() {
    print_step "ایجاد index.js..."
    
    cat > src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF
    
    print_success "index.js ایجاد شد"
}

# ایجاد App.js
create_app_js() {
    print_step "ایجاد App.js..."
    
    cat > src/App.js << 'EOF'
import React from 'react';
import CompletePersianAIDashboard from './components/CompletePersianAIDashboard';

function App() {
  return (
    <div className="App">
      <CompletePersianAIDashboard />
    </div>
  );
}

export default App;
EOF
    
    print_success "App.js ایجاد شد"
}

# ایجاد public/index.html
create_index_html() {
    print_step "ایجاد index.html..."
    
    cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="fa" dir="rtl">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#3b82f6" />
    <meta
      name="description"
      content="سیستم هوش مصنوعی حقوقی فارسی - داشبورد پیشرفته"
    />
    <link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />
    <link rel="manifest" href="%PUBLIC_URL%/manifest.json" />
    <title>⚖️ داشبورد AI حقوقی فارسی</title>
    
    <!-- Preload fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    
    <style>
      /* Loading screen */
      .loading-screen {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        color: white;
        font-family: 'Vazirmatn', sans-serif;
      }
      
      .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
        margin-bottom: 20px;
      }
      
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
      
      .loading-text {
        font-size: 18px;
        margin-bottom: 10px;
      }
      
      .loading-subtext {
        font-size: 14px;
        opacity: 0.8;
      }
    </style>
  </head>
  <body>
    <noscript>برای اجرای این برنامه نیاز به JavaScript دارید.</noscript>
    
    <!-- Loading Screen -->
    <div id="loading-screen" class="loading-screen">
      <div class="loading-spinner"></div>
      <div class="loading-text">⚖️ سیستم هوش مصنوعی حقوقی فارسی</div>
      <div class="loading-subtext">در حال بارگیری داشبورد پیشرفته...</div>
    </div>
    
    <div id="root"></div>
    
    <script>
      // Hide loading screen when React loads
      window.addEventListener('load', function() {
        setTimeout(function() {
          const loadingScreen = document.getElementById('loading-screen');
          if (loadingScreen) {
            loadingScreen.style.opacity = '0';
            loadingScreen.style.transition = 'opacity 0.5s ease-out';
            setTimeout(function() {
              loadingScreen.style.display = 'none';
            }, 500);
          }
        }, 1000);
      });
    </script>
  </body>
</html>
EOF
    
    print_success "index.html ایجاد شد"
}

# کپی کردن کامپوننت اصلی
copy_main_component() {
    print_step "آماده‌سازی کامپوننت اصلی..."
    
    # ایجاد پوشه components
    mkdir -p src/components
    
    print_info "لطفاً کامپوننت CompletePersianAIDashboard را از artifact کپی کرده و در فایل زیر قرار دهید:"
    print_warning "src/components/CompletePersianAIDashboard.js"
    
    echo ""
    echo "📋 مراحل کپی کردن:"
    echo "1. کامپوننت React را از artifact بالا انتخاب و کپی کنید"
    echo "2. فایل جدیدی با نام CompletePersianAIDashboard.js در پوشه src/components ایجاد کنید"
    echo "3. کامپوننت کپی شده را در آن قرار دهید"
    echo ""
}

# نصب dependencies
install_dependencies() {
    print_step "نصب وابستگی‌ها..."
    
    if npm install; then
        print_success "وابستگی‌ها با موفقیت نصب شدند"
    else
        print_error "خطا در نصب وابستگی‌ها"
        exit 1
    fi
}

# ایجاد اسکریپت‌های اضافی
create_additional_scripts() {
    print_step "ایجاد اسکریپت‌های کمکی..."
    
    # اسکریپت راه‌اندازی سریع
    cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 راه‌اندازی داشبورد..."
npm start
EOF
    chmod +x start.sh
    
    # اسکریپت build
    cat > build.sh << 'EOF'
#!/bin/bash
echo "📦 ساخت نسخه تولید..."
npm run build
echo "✅ نسخه تولید در پوشه build ایجاد شد"
EOF
    chmod +x build.sh
    
    print_success "اسکریپت‌های کمکی ایجاد شدند"
}

# ایجاد README
create_readme() {
    print_step "ایجاد README..."
    
    cat > README.md << 'EOF'
# 🚀 داشبورد پیشرفته سیستم هوش مصنوعی حقوقی فارسی

## ⚖️ ویژگی‌ها

- 📊 داشبورد real-time با نمودارهای تعاملی
- 🧠 مدیریت مدل‌های هوش مصنوعی 
- 📚 نظارت بر جمع‌آوری داده‌های حقوقی فارسی
- 🎛️ کنترل‌های پیشرفته آموزش
- 📈 آنالیز عملکرد و پیش‌بینی هوشمند
- 🔔 سیستم اعلان‌های پیشرفته
- 🎨 طراحی مدرن با پشتیبانی کامل از RTL فارسی

## 🛠️ نصب و راه‌اندازی

### پیش‌نیازها
- Node.js 16.0.0 یا بالاتر
- npm یا yarn

### راه‌اندازی سریع

```bash
# نصب وابستگی‌ها
npm install

# راه‌اندازی سرور توسعه
npm start

# یا استفاده از اسکریپت سریع
./start.sh
```

داشبورد در آدرس http://localhost:3000 در دسترس خواهد بود.

### ساخت نسخه تولید

```bash
# ساخت فایل‌های تولید
npm run build

# یا استفاده از اسکریپت
./build.sh
```

## 📱 استفاده

### صفحات اصلی
- **داشبورد**: نمای کلی سیستم و معیارهای real-time
- **مدل‌ها**: مدیریت و کنترل مدل‌های AI
- **داده‌ها**: نظارت بر منابع داده حقوقی
- **نظارت**: مانیتورینگ پیشرفته سیستم
- **گزارش‌ها**: آنالیز و گزارش‌گیری

### ویژگی‌های کلیدی
- 🔄 به‌روزرسانی خودکار داده‌ها
- 📊 نمودارهای تعاملی و قابل تنظیم
- 🎯 کنترل دقیق پارامترهای آموزش
- 📱 طراحی واکنش‌گرا (Responsive)
- 🌙 پشتیبانی از حالت تاریک

## 🔧 تنظیمات

### تغییر interval به‌روزرسانی
در header داشبورد می‌توانید interval را بین 1، 3 و 5 ثانیه تنظیم کنید.

### تنظیم مدل‌ها
در صفحه مدل‌ها می‌توانید:
- مدل‌های جدید اضافه کنید
- پارامترهای آموزش را تنظیم کنید
- پیشرفت آموزش را مشاهده کنید

## 🤝 مشارکت

برای مشارکت در توسعه این پروژه:

1. پروژه را Fork کنید
2. شاخه جدید ایجاد کنید
3. تغییرات خود را commit کنید
4. Pull Request ارسال کنید

## 📞 پشتیبانی

برای پشتیبانی و سوالات:
- GitHub Issues
- ایمیل پشتیبانی

---

**ساخته شده با ❤️ برای جامعه AI فارسی**
EOF
    
    print_success "README.md ایجاد شد"
}

# تابع اصلی
main() {
    echo ""
    print_info "شروع نصب خودکار..."
    echo ""
    
    # بررسی Node.js
    if ! check_nodejs; then
        print_warning "Node.js موجود نیست. آیا می‌خواهید نصب شود؟ (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY]|بله|آره)$ ]]; then
            install_nodejs
        else
            print_error "Node.js برای ادامه ضروری است"
            exit 1
        fi
    fi
    
    # بررسی npm
    if ! check_npm; then
        print_error "npm موجود نیست"
        exit 1
    fi
    
    # ایجاد پروژه
    create_project_structure
    create_package_json
    
    # نصب وابستگی‌ها
    install_dependencies
    
    # راه‌اندازی Tailwind
    setup_tailwind
    
    # ایجاد فایل‌ها
    create_css_files
    create_index_js
    create_app_js
    create_index_html
    create_additional_scripts
    create_readme
    
    # آماده‌سازی کامپوننت
    copy_main_component
    
    echo ""
    print_success "🎉 نصب با موفقیت تکمیل شد!"
    echo ""
    print_info "مراحل نهایی:"
    echo "1. کامپوننت React را از artifact کپی کنید"
    echo "2. در مسیر src/components/CompletePersianAIDashboard.js قرار دهید"
    echo "3. دستور زیر را اجرا کنید:"
    echo ""
    print_step "npm start"
    echo ""
    print_info "یا:"
    echo ""
    print_step "./start.sh"
    echo ""
    print_success "داشبورد در http://localhost:3000 در دسترس خواهد بود 🚀"
}

# اجرای تابع اصلی
main "$@"