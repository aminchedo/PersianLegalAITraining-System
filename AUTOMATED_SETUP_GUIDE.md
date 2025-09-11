# 🚀 Persian Legal AI - Automated Setup Guide

## ⚡ One-Command Setup

After cloning the project, run this single command to set up everything:

```bash
python3 setup_persian_legal_ai.py
```

## 🎯 What the Script Does

The automated setup script (`setup_persian_legal_ai.py`) handles:

### ✅ System Checks
- Python 3.8+ verification
- Node.js 18+ verification  
- Git availability check

### 🐍 Backend Setup
- Creates Python virtual environment (or uses system Python as fallback)
- Installs all Python dependencies from:
  - `requirements.txt`
  - `backend/requirements.txt`
  - `requirements_production.txt`

### 📦 Frontend Setup
- Detects and uses available package manager (pnpm > yarn > npm)
- Installs all Node.js dependencies
- Sets up Next.js environment

### 🤖 AI Models
- Downloads Persian BERT models:
  - `HooshvareLab/bert-base-parsbert-uncased` (500MB)
  - `HooshvareLab/bert-fa-base-uncased` (450MB)
- Stores models in `./models/` directory

### 🗄️ Database
- Sets up SQLite database
- Runs any existing migrations
- Creates database files if needed

### 🌐 Services
- Starts backend server on `http://localhost:8000`
- Starts frontend dashboard on `http://localhost:3000`
- Enables hot reload for development

### 🎓 Training
- Automatically starts AI model training
- Uses DoRA (Weight-Decomposed Low-Rank Adaptation)
- Configures optimal training parameters

## 🎉 Final Result

After successful setup, you'll have:

- **📊 Dashboard**: http://localhost:3000
- **🔧 Backend API**: http://localhost:8000  
- **📚 API Docs**: http://localhost:8000/docs
- **🔍 Health Check**: http://localhost:8000/api/system/health

## 🛠️ Manual Setup (Alternative)

If you prefer manual setup:

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install  # or yarn/pnpm
npm run dev
```

## 🔧 Troubleshooting

### Virtual Environment Issues
If venv creation fails, the script automatically falls back to system Python with `--user` installs.

### Node.js Issues
The script tries multiple package managers in order:
1. pnpm (fastest)
2. yarn (reliable)
3. npm (default)

### Model Download Issues
Models are downloaded from Hugging Face. If downloads fail:
- Check internet connection
- Models will be re-downloaded on next run
- Manual download: Use Hugging Face Hub library

### Port Conflicts
Default ports:
- Backend: 8000
- Frontend: 3000

Change in environment files if needed.

## 🎯 Quick Commands

```bash
# Full setup
python3 setup_persian_legal_ai.py

# Check script syntax
python3 -c "import setup_persian_legal_ai; print('OK')"

# Manual backend start
python3 -m uvicorn backend.main:app --reload

# Manual frontend start
cd frontend && npm run dev
```

## 📋 System Requirements

- **Python**: 3.8+
- **Node.js**: 18+
- **Git**: Any recent version
- **OS**: Linux, macOS, Windows
- **RAM**: 4GB+ (8GB+ recommended for training)
- **Storage**: 2GB+ for models

## ⚠️ Notes

- First run takes longer due to model downloads
- Training starts automatically but can be stopped/restarted
- All services run with hot reload enabled
- Press `Ctrl+C` to stop all services