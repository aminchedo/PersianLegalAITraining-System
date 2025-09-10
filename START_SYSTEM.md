# 🚀 Persian Legal AI - System Startup Guide

## Quick Start Commands

### 1. Start Backend Services

```bash
# Activate the virtual environment
source venv/bin/activate

# Start the main API server (Terminal 1)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start the Persian processing module (Terminal 2)
uvicorn persian_main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Verify System Health

```bash
# Run health check
python system_health_check.py

# Run integration tests
python test_integration.py
```

### 3. Test API Endpoints

```bash
# Test main API
curl http://localhost:8000/

# Test health endpoint
curl http://localhost:8000/api/system/health

# Test Persian module
curl http://localhost:8001/

# Test Persian text analysis
curl -X POST http://localhost:8001/api/persian/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "این یک متن فارسی برای تست است"}'
```

## 📊 System Status Dashboard

- **Main API:** http://localhost:8000
- **Persian Module:** http://localhost:8001
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/system/health

## 🎯 Next Development Steps

1. **AI Model Integration**
   - Add PyTorch and Transformers
   - Integrate Persian BERT model
   - Implement document classification

2. **Persian NLP Enhancement**
   - Add Hazm library for Persian text processing
   - Implement advanced text normalization
   - Add legal term extraction

3. **Database Enhancement**
   - Add more complex data models
   - Implement document storage
   - Add search capabilities

4. **Frontend Development**
   - Complete Next.js setup
   - Build user interface
   - Add document upload features

## ✅ System is READY FOR DEVELOPMENT!

The recovery process is complete and the system is fully operational.