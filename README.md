# Persian Legal AI Training System

## 🚀 Quick Start

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../configs/requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Full System (Recommended)
```bash
./scripts/start_system.sh
```

## 📁 Project Structure
```
├── backend/          # FastAPI backend + AI models
├── frontend/         # React TypeScript dashboard
├── ai_models/        # DoRA & QR-Adaptor implementations
├── deployment/       # Docker & deployment configs
├── docs/            # Documentation
├── scripts/         # Utility scripts
├── configs/         # Configuration files
├── tests/           # Test files
└── data/            # Training and validation data
```

## 🔗 Quick Links
- Backend API: http://localhost:8000
- Frontend Dashboard: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/system/health

## 📚 Documentation
See `docs/` directory for detailed documentation.