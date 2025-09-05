#!/usr/bin/env python3
"""
Enhanced Persian Legal AI Dashboard Launcher
اجرای داشبورد پیشرفته با React
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def create_enhanced_dashboard():
    """ایجاد داشبورد پیشرفته React"""
    
    # ایجاد پوشه interface جدید
    interface_dir = Path("interface_enhanced")
    interface_dir.mkdir(exist_ok=True)
    
    # ایجاد package.json
    package_json = {
        "name": "persian-legal-ai-dashboard",
        "version": "1.0.0",
        "description": "داشبورد پیشرفته سیستم هوش مصنوعی حقوقی فارسی",
        "main": "index.js",
        "scripts": {
            "start": "react-scripts start",
            "build": "react-scripts build",
            "test": "react-scripts test",
            "eject": "react-scripts eject"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-scripts": "5.0.1",
            "recharts": "^2.8.0",
            "lucide-react": "^0.294.0",
            "tailwindcss": "^3.3.0",
            "autoprefixer": "^10.4.16",
            "postcss": "^8.4.31"
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
        }
    }
    
    with open(interface_dir / "package.json", "w", encoding="utf-8") as f:
        json.dump(package_json, f, indent=2, ensure_ascii=False)
    
    # ایجاد tailwind.config.js
    tailwind_config = '''/** @type {import('tailwindcss').Config} */
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
      }
    },
  },
  plugins: [],
}'''
    
    with open(interface_dir / "tailwind.config.js", "w", encoding="utf-8") as f:
        f.write(tailwind_config)
    
    # ایجاد پوشه src
    src_dir = interface_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    # ایجاد index.css
    index_css = '''@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700&display=swap');
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: 'Vazirmatn', 'Tahoma', sans-serif;
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
}'''
    
    with open(src_dir / "index.css", "w", encoding="utf-8") as f:
        f.write(index_css)
    
    # ایجاد index.js
    index_js = '''import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import PersianLegalAIDashboard from './PersianLegalAIDashboard';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <PersianLegalAIDashboard />
  </React.StrictMode>
);'''
    
    with open(src_dir / "index.js", "w", encoding="utf-8") as f:
        f.write(index_js)
    
    # ایجاد public/index.html
    public_dir = interface_dir / "public"
    public_dir.mkdir(exist_ok=True)
    
    index_html = '''<!DOCTYPE html>
<html lang="fa" dir="rtl">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="سیستم هوش مصنوعی حقوقی فارسی - نسخه پیشرفته 2025" />
    <title>⚖️ داشبورد AI حقوقی فارسی</title>
  </head>
  <body>
    <noscript>برای اجرای این برنامه نیاز به JavaScript دارید.</noscript>
    <div id="root"></div>
  </body>
</html>'''
    
    with open(public_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    print("✅ فایل‌های داشبورد پیشرفته ایجاد شد!")
    return interface_dir

def install_dependencies(interface_dir):
    """نصب وابستگی‌های React"""
    print("📦 در حال نصب وابستگی‌ها...")
    
    try:
        subprocess.run(
            ["npm", "install"], 
            cwd=interface_dir, 
            check=True,
            capture_output=True
        )
        print("✅ وابستگی‌ها با موفقیت نصب شدند!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ خطا در نصب وابستگی‌ها: {e}")
        return False
    except FileNotFoundError:
        print("❌ Node.js یا npm یافت نشد. لطفاً Node.js را نصب کنید.")
        print("🔗 https://nodejs.org/")
        return False

def start_dashboard(interface_dir):
    """اجرای داشبورد"""
    print("🚀 در حال اجرای داشبورد پیشرفته...")
    print("📊 داشبورد در آدرس http://localhost:3000 در دسترس خواهد بود")
    print("⚖️ سیستم هوش مصنوعی حقوقی فارسی - نسخه پیشرفته")
    print("-" * 60)
    
    try:
        subprocess.run(
            ["npm", "start"], 
            cwd=interface_dir,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ خطا در اجرای داشبورد: {e}")
    except KeyboardInterrupt:
        print("\n🛑 داشبورد توسط کاربر متوقف شد")

def main():
    """تابع اصلی"""
    print("🎨 ایجاد داشبورد پیشرفته سیستم هوش مصنوعی حقوقی فارسی")
    print("=" * 60)
    
    # ایجاد فایل‌های داشبورد
    interface_dir = create_enhanced_dashboard()
    
    # کپی کردن کامپوننت React از artifact
    print("📋 لطفاً کامپوننت React را از artifact بالا کپی کرده و در فایل زیر قرار دهید:")
    print(f"📁 {interface_dir}/src/PersianLegalAIDashboard.js")
    
    # سوال از کاربر
    response = input("\n✅ آیا کامپوننت را کپی کردید؟ (y/n): ")
    
    if response.lower() in ['y', 'yes', 'بله', 'آره']:
        # نصب وابستگی‌ها
        if install_dependencies(interface_dir):
            # اجرای داشبورد
            start_dashboard(interface_dir)
        else:
            print("❌ خطا در نصب وابستگی‌ها")
    else:
        print("ℹ️  لطفاً ابتدا کامپوننت را کپی کنید سپس دوباره اجرا کنید")
        print(f"📝 دستور اجرا: cd {interface_dir} && npm start")

if __name__ == "__main__":
    main()