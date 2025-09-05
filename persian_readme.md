# ⚖️ سیستم آموزش هوش مصنوعی حقوقی فارسی

[![نسخه](https://img.shields.io/badge/نسخه-1.0.0-blue.svg)](https://github.com/your-repo/persian-legal-ai)
[![مجوز](https://img.shields.io/badge/مجوز-MIT-green.svg)](LICENSE)
[![پایتون](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![ری‌اکت](https://img.shields.io/badge/react-18.2+-blue.svg)](https://reactjs.org)

## 🎯 معرفی پروژه

**سیستم آموزش هوش مصنوعی حقوقی فارسی** یک پلتفرم جامع و آماده تولید برای آموزش، مدیریت و استقرار مدل‌های هوش مصنوعی زبان فارسی است که به‌طور خاص برای پردازش اسناد حقوقی طراحی شده است. این سیستم از تکنیک‌های پیشرفته مانند DoRA، QR-Adaptor و بهینه‌سازی‌های Intel CPU استفاده می‌کند.

### 🌟 ویژگی‌های کلیدی

- **آموزش پیشرفته AI**: پیاده‌سازی DoRA و QR-Adaptor برای آموزش کارآمد مدل‌ها
- **داشبورد زمان واقعی**: رابط مدرن React با پشتیبانی کامل RTL فارسی
- **ادغام داده‌های حقوقی فارسی**: اتصال مستقیم به پایگاه‌های داده حقوقی ایران
- **بهینه‌سازی Intel CPU**: استفاده از Intel Extension for PyTorch با پشتیبانی AMX و AVX-512
- **آماده تولید**: ابزارهای جامع نظارت، لاگ‌گیری و استقرار
- **ارتباط زمان واقعی**: به‌روزرسانی‌های زنده و اعلان‌ها بر اساس WebSocket

## 🏗️ معماری سیستم

```
persian-legal-ai/
├── backend/                    # سرور Backend با FastAPI
│   ├── main.py                # برنامه اصلی سرور
│   ├── requirements.txt       # وابستگی‌های Python
│   ├── api/                   # نقاط انتهایی API
│   ├── models/                # پیاده‌سازی مدل‌های AI
│   ├── services/              # سرویس‌های منطق کسب‌وکار
│   ├── utils/                 # توابع کمکی
│   ├── config/                # فایل‌های تنظیمات
│   └── logs/                  # لاگ‌های برنامه
├── frontend/                   # داشبورد Frontend با React
│   ├── src/
│   │   ├── components/        # کامپوننت‌های React
│   │   │   ├── CompletePersianAIDashboard.tsx  # داشبورد اصلی
│   │   │   └── AdvancedComponents.jsx          # کامپوننت‌های پیشرفته
│   │   ├── hooks/             # هوک‌های سفارشی React
│   │   │   └── usePersianAI.js                 # هوک‌های Persian AI
│   │   ├── api/               # ادغام API
│   │   │   └── persian-ai-api.js               # کلاینت API
│   │   └── styles/            # CSS و استایل‌ها
│   ├── package.json           # وابستگی‌های Node.js
│   └── public/                # منابع استاتیک
├── scripts/                    # اسکریپت‌های خودکارسازی
│   ├── setup.sh              # اسکریپت راه‌اندازی
│   └── launcher.py           # راه‌انداز داشبورد
└── docs/                      # مستندات
    ├── README.txt             # مستندات پروژه
    └── GUIDE.txt              # راهنمای کاربر
```

## 🚀 شروع سریع

### پیش‌نیازها

- **سیستم‌عامل**: Windows 10/11 Pro یا Windows Server 2019/2022
- **Python**: نسخه 3.9 یا بالاتر
- **Node.js**: نسخه 16.0 یا بالاتر
- **CPU**: Intel Xeon 24-core (توصیه شده) یا پردازنده قدرتمند مشابه
- **RAM**: 32-64GB
- **حافظه**: 500GB+ SSD
- **مجوزهای مدیریت**: ضروری برای بهینه‌سازی‌های Intel

### نصب

1. **سازماندهی فایل‌های پروژه**:
   ```bash
   # اجرای اسکریپت سازماندهی
   .\organize_files.ps1
   ```

2. **راه‌اندازی Backend**:
   ```bash
   cd persian-legal-ai/backend
   
   # ایجاد محیط مجازی
   python -m venv venv
   
   # فعال‌سازی محیط مجازی
   # در Windows:
   venv\Scripts\activate
   
   # نصب وابستگی‌ها
   pip install -r requirements.txt
   
   # نصب Intel Extension for PyTorch (اختیاری اما توصیه شده)
   pip install intel-extension-for-pytorch
   ```

3. **راه‌اندازی Frontend**:
   ```bash
   cd ../frontend
   
   # نصب وابستگی‌های Node.js
   npm install
   
   # نصب وابستگی‌های اضافی
   npm install tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```

### اجرای برنامه

#### حالت توسعه

1. **راه‌اندازی سرور Backend**:
   ```bash
   cd backend
   venv\Scripts\activate
   python main.py
   ```
   Backend در آدرس زیر در دسترس خواهد بود: `http://localhost:8000`

2. **راه‌اندازی داشبورد Frontend** (در ترمینال جدید):
   ```bash
   cd frontend
   npm start
   ```
   Frontend در آدرس زیر در دسترس خواهد بود: `http://localhost:3000`

#### حالت تولید

1. **ساخت Frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **استقرار با Docker** (اختیاری):
   ```bash
   docker-compose up -d
   ```

## 🔧 کامپوننت‌های اصلی

### کامپوننت‌های Backend

#### 1. سرور اصلی (`backend/main.py`)
```python
# برنامه FastAPI با ویژگی‌های زیر:
- نقاط انتهایی RESTful API
- ارتباط زمان واقعی WebSocket
- احراز هویت و مجوزدهی
- ادغام پایگاه داده
- مدیریت وظایف پس‌زمینه
- لاگ‌گیری جامع
```

**نقاط انتهایی کلیدی API**:
- `GET /api/status` - وضعیت سیستم
- `GET /api/metrics` - معیارهای زمان واقعی سیستم
- `GET /api/models` - مدیریت مدل‌ها
- `POST /api/training/start` - شروع آموزش مدل
- `WebSocket /ws` - به‌روزرسانی‌های زمان واقعی

#### 2. آموزش مدل‌های AI
```python
# DoRA (Weight-Decomposed Low-Rank Adaptation)
class DoRATrainer:
    - تجزیه magnitude و direction
    - نرخ‌های یادگیری جداگانه برای کامپوننت‌ها
    - پشتیبانی از مدل‌های زبان فارسی
    - ردیابی پیشرفت زمان واقعی

# QR-Adaptor (بهینه‌سازی مشترک کوانتایزیشن و رتبه)
class QRAdaptor:
    - کوانتایزیشن تطبیقی با پشتیبانی NF4
    - بهینه‌سازی مشترک bit-width و rank
    - بهینه‌سازی نسبت فشرده‌سازی
    - نظارت بر عملکرد
```

#### 3. سرویس‌های جمع‌آوری داده
```python
# منابع داده‌های حقوقی فارسی
- پیکره نعب: پیکره متنی فارسی 700 گیگابایت
- پورتال داده ایران: اسناد حقوقی دانشگاه Syracuse
- پورتال قوانین: قوانین و مقررات رسمی ایران
- وب‌سایت مجلس: اسناد مجلس شورای اسلامی

# جمع‌آوری و پردازش زمان واقعی
- ارزیابی کیفیت اسناد
- دسته‌بندی خودکار
- تشخیص تکراری
- پیش‌پردازش متن برای آموزش AI
```

### کامپوننت‌های Frontend

#### 1. داشبورد اصلی (`frontend/src/components/CompletePersianAIDashboard.tsx`)
```typescript
// داشبورد جامع با:
- نظارت زمان واقعی سیستم
- مدیریت آموزش مدل‌ها
- نظارت بر جمع‌آوری داده
- آنالیز و گزارش‌گیری پیشرفته
- رابط RTL فارسی
- پشتیبانی از حالت تاریک/روشن
```

#### 2. کامپوننت‌های پیشرفته (`frontend/src/components/AdvancedComponents.jsx`)
```typescript
// کامپوننت‌های تخصصی رابط کاربری:
- ProjectManager: مدیریت چندپروژه
- AIInsights: پیش‌بینی‌ها و توصیه‌های هوشمند
- SystemLogs: مشاهده و فیلتر لاگ‌های زمان واقعی
- ModelController: تنظیمات پیشرفته مدل
```

## 🎮 ویژگی‌های داشبورد

### 1. نظارت زمان واقعی
- **معیارهای سیستم**: استفاده از CPU، حافظه، GPU با بهینه‌سازی‌های Intel
- **پیشرفت آموزش**: خطای آموزش زنده، دقت و throughput
- **جمع‌آوری داده**: آمار پردازش اسناد زمان واقعی
- **آنالیز عملکرد**: شناسایی گلوگاه‌ها و پیشنهادهای بهینه‌سازی

### 2. مدیریت مدل‌ها
- **کنترل آموزش**: شروع، توقف، مکث و ادامه جلسات آموزش
- **تنظیم پارامترها**: تنظیم زمان واقعی نرخ یادگیری، اندازه batch، پارامترهای DoRA
- **نسخه‌بندی مدل**: مدیریت checkpoint و مقایسه مدل‌ها
- **ارزیابی عملکرد**: تست و اعتبارسنجی جامع مدل

### 3. مدیریت داده
- **ادغام منابع**: اتصالات مستقیم به پایگاه‌های داده حقوقی فارسی
- **کنترل کیفیت**: ارزیابی و فیلتر خودکار کیفیت اسناد
- **پایپ‌لاین پیش‌پردازش**: تمیزسازی متن، tokenization و آماده‌سازی
- **آنالیز مجموعه داده**: تحلیل آماری و تجسم

## 🔌 مستندات API

### نقاط انتهایی احراز هویت
```http
POST /api/auth/login          # احراز هویت کاربر
POST /api/auth/logout         # خروج کاربر
GET  /api/auth/user           # دریافت کاربر فعلی
POST /api/auth/refresh        # تازه‌سازی توکن احراز هویت
```

### نظارت سیستم
```http
GET  /api/metrics/system      # معیارهای زمان واقعی سیستم
GET  /api/metrics/training    # معیارهای جلسه آموزش
GET  /api/metrics/data        # آمار جمع‌آوری داده
WebSocket /ws/metrics         # پخش زنده معیارها
```

### مدیریت مدل‌ها
```http
GET    /api/models            # فهرست همه مدل‌ها
POST   /api/models            # ایجاد مدل جدید
GET    /api/models/{id}       # دریافت جزئیات مدل
PUT    /api/models/{id}       # به‌روزرسانی تنظیمات مدل
DELETE /api/models/{id}       # حذف مدل
POST   /api/models/{id}/train # شروع جلسه آموزش
POST   /api/models/{id}/stop  # توقف جلسه آموزش
GET    /api/models/{id}/logs  # دریافت لاگ‌های آموزش
```

### جمع‌آوری داده
```http
GET  /api/data/sources        # منابع داده در دسترس
POST /api/data/collect        # شروع جمع‌آوری داده
GET  /api/data/status         # وضعیت جمع‌آوری
GET  /api/data/quality        # معیارهای کیفیت داده
POST /api/data/process        # پردازش داده‌های جمع‌آوری شده
```

## 🛠️ نیازمندی‌های پیاده‌سازی

### 1. پیاده‌سازی آموزش DoRA
```python
# Weight-Decomposed Low-Rank Adaptation
class DoRALayer:
    def __init__(self, rank, alpha, target_modules):
        self.magnitude_adapter = MagnitudeAdapter(rank)
        self.direction_adapter = DirectionAdapter(rank)
        self.scaling_factor = alpha / rank
    
    def forward(self, x):
        # پیاده‌سازی منطق تجزیه DoRA
        magnitude = self.magnitude_adapter(x)
        direction = self.direction_adapter(x)
        return x + magnitude * direction * self.scaling_factor
```

### 2. بهینه‌سازی QR-Adaptor
```python
# بهینه‌سازی مشترک کوانتایزیشن و رتبه
class QRAdaptor:
    def __init__(self, quantization_bits, adaptive_rank):
        self.quantizer = AdaptiveQuantizer(quantization_bits)
        self.rank_optimizer = RankOptimizer(adaptive_rank)
    
    def optimize(self, model):
        # پیاده‌سازی بهینه‌سازی مشترک
        quantized_model = self.quantizer.quantize(model)
        optimized_model = self.rank_optimizer.optimize(quantized_model)
        return optimized_model
```

### 3. جمع‌آوری داده‌های فارسی
```python
# ادغام منابع داده واقعی
class PersianLegalDataCollector:
    def __init__(self):
        self.naab_connector = NaabCorpusConnector()
        self.qavanin_scraper = QavaninPortalScraper()
        self.majles_extractor = MajlesDataExtractor()
        self.iran_portal = IranDataPortalConnector()
    
    async def collect_documents(self, source, filters):
        # پیاده‌سازی جمع‌آوری داده واقعی
        documents = await source.fetch_documents(filters)
        processed_docs = self.preprocess_documents(documents)
        return self.quality_filter(processed_docs)
```

## 🔒 امنیت و ویژگی‌های تولید

### 1. احراز هویت و مجوزدهی
- احراز هویت مبتنی بر JWT
- کنترل دسترسی مبتنی بر نقش (RBAC)
- مدیریت کلید API
- مدیریت جلسه

### 2. امنیت داده
- اعتبارسنجی و پاک‌سازی ورودی
- جلوگیری از SQL injection
- محافظت XSS
- محافظت CSRF

### 3. نظارت و لاگ‌گیری
- لاگ‌گیری ساختاریافته با شناسه‌های correlation
- نظارت بر عملکرد
- ردیابی خطا و هشدار
- لاگ‌گیری audit

## 📊 بهینه‌سازی عملکرد

### 1. بهینه‌سازی‌های Backend
- async/await برای عملیات I/O
- Connection pooling برای پایگاه‌های داده
- استراتژی‌های کش (Redis)
- پردازش وظایف پس‌زمینه

### 2. بهینه‌سازی‌های Frontend
- تقسیم کد و lazy loading
- Memoization برای محاسبات گران
- Virtual scrolling برای فهرست‌های بزرگ
- بهینه‌سازی تصاویر و lazy loading

### 3. بهینه‌سازی‌های آموزش AI
- آموزش Mixed precision
- تجمیع Gradient
- Dynamic batching
- موازی‌سازی مدل

## 🧪 استراتژی تست

### 1. تست Backend
```bash
# تست‌های واحد
pytest backend/tests/unit/

# تست‌های یکپارچگی
pytest backend/tests/integration/

# تست‌های API
pytest backend/tests/api/
```

### 2. تست Frontend
```bash
# تست‌های واحد
npm test

# تست‌های یکپارچگی
npm run test:integration

# تست‌های E2E
npm run test:e2e
```

## 📈 نظارت و Observability

### 1. جمع‌آوری معیارها
- معیارهای سیستم (CPU، حافظه، دیسک، شبکه)
- معیارهای برنامه (نرخ درخواست، زمان پاسخ، نرخ خطا)
- معیارهای کسب‌وکار (دقت آموزش، کیفیت داده، فعالیت کاربر)

### 2. لاگ‌گیری
- لاگ‌گیری ساختاریافته با فرمت JSON
- تجمیع و متمرکزسازی لاگ
- تحلیل لاگ و هشدار

### 3. Tracing
- Tracing توزیع‌شده برای درخواست‌های API
- پروفایل‌گیری عملکرد
- شناسایی گلوگاه

## 🔧 مدیریت تنظیمات

### متغیرهای محیطی
```bash
# تنظیمات Backend
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your-secret-key
INTEL_OPTIMIZATION=true

# تنظیمات آموزش AI
TRAINING_BATCH_SIZE=4
LEARNING_RATE=1e-4
DORA_RANK=64
DORA_ALPHA=16.0
```

## 🚀 راهنمای استقرار

### 1. استقرار توسعه
```bash
# Backend
cd backend
python main.py

# Frontend
cd frontend
npm start
```

### 2. استقرار تولید
```bash
# ساخت frontend
cd frontend
npm run build

# استقرار با Docker
docker-compose up -d

# یا استقرار با systemd
sudo systemctl start persian-ai-backend
sudo systemctl start persian-ai-frontend
```

## 📚 منابع اضافی

### مستندات
- [مستندات API](docs/api.md)
- [راهنمای توسعه](docs/development.md)
- [راهنمای استقرار](docs/deployment.md)
- [راهنمای کاربر](docs/user-manual.md)

### وابستگی‌های خارجی
- [مستندات FastAPI](https://fastapi.tiangolo.com/)
- [مستندات React](https://reactjs.org/docs/)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Tailwind CSS](https://tailwindcss.com/docs)

## 🤝 مشارکت

1. پروژه را Fork کنید
2. شاخه ویژگی ایجاد کنید (`git checkout -b feature/amazing-feature`)
3. تغییرات خود را Commit کنید (`git commit -m 'Add amazing feature'`)
4. به شاخه Push کنید (`git push origin feature/amazing-feature`)
5. Pull Request ایجاد کنید

### راهنمای توسعه
- از PEP 8 برای کد Python پیروی کنید
- از ESLint و Prettier برای JavaScript/TypeScript استفاده کنید
- تست‌های جامع برای ویژگی‌های جدید بنویسید
- مستندات را برای تغییرات API به‌روزرسانی کنید
- از پیام‌های commit متعارف استفاده کنید

## 📝 مجوز

این پروژه تحت مجوز MIT منتشر شده است - برای جزئیات فایل [LICENSE](LICENSE) را ببینید.

## 🙏 قدردانی

- **شرکت Intel** برای Intel Extension for PyTorch
- **دانشگاه تهران** برای مدل PersianMind
- **HooshvareLab** برای مدل‌های ParsBERT
- **دانشگاه Syracuse** برای پورتال داده ایران
- **جامعه NLP فارسی** برای ابزارها و منابع مختلف

## 📞 پشتیبانی

برای پشتیبانی و سوالات:
- **مسائل**: [GitHub Issues](https://github.com/your-repo/persian-legal-ai/issues)
- **بحث‌ها**: [GitHub Discussions](https://github.com/your-repo/persian-legal-ai/discussions)
- **ایمیل**: support@persian-legal-ai.com
- **مستندات**: [ویکی پروژه](https://github.com/your-repo/persian-legal-ai/wiki)

---

**سیستم آموزش هوش مصنوعی حقوقی فارسی** - پیشرفت زبان فارسی AI با تکنیک‌های پیشرفته 2025 🚀