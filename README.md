# 🚀 Persian Legal AI System

Advanced Persian Legal Document Processing and AI Classification System with full-stack implementation.

## 🎯 Project Overview

This is a production-ready Persian Legal AI system that provides:

- **AI-Powered Classification**: Persian BERT-based legal document classification
- **Full-Stack Architecture**: FastAPI backend + React frontend
- **Persian Language Support**: Native RTL support with Persian fonts
- **Database Integration**: SQLite with FTS5 for Persian text search
- **Production Ready**: Docker containerization and deployment

## 🏗️ Architecture

```
persian-legal-ai/
├── backend/                 # FastAPI Python backend
│   ├── main.py             # Main application entry point
│   ├── ai_classifier.py    # Persian BERT classifier
│   ├── config/
│   │   └── database.py     # Database configuration
│   └── requirements.txt    # Python dependencies
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx         # Main React application
│   │   └── App.css         # Persian-optimized styles
│   └── package.json        # Node.js dependencies
├── docker-compose.yml      # Production deployment
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Node.js 22+
- Docker (for production deployment)

### Development Setup

1. **Backend Setup**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

2. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

3. **Access the Application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📋 Features

### ✅ Completed Features

- **Backend Infrastructure**
  - ✅ FastAPI with async support
  - ✅ SQLite database with FTS5 search
  - ✅ Persian BERT AI classifier
  - ✅ RESTful API endpoints
  - ✅ Health monitoring
  - ✅ CORS configuration

- **Frontend Interface**
  - ✅ React with TypeScript
  - ✅ Persian RTL layout
  - ✅ Real-time AI classification
  - ✅ System health dashboard
  - ✅ Responsive design
  - ✅ Persian typography

- **Integration & Testing**
  - ✅ Frontend-backend integration
  - ✅ API proxy configuration
  - ✅ Health checks
  - ✅ Error handling

- **Production Deployment**
  - ✅ Docker containerization
  - ✅ Docker Compose orchestration
  - ✅ Nginx reverse proxy
  - ✅ Production configuration

## 🔧 API Endpoints

### System Endpoints
- `GET /` - Root endpoint
- `GET /api/system/health` - System health check
- `GET /docs` - API documentation

### AI Classification
- `POST /api/ai/classify` - Classify Persian legal text

### Document Management
- `GET /api/documents/stats` - Document statistics
- `GET /api/documents/search` - Search documents
- `POST /api/documents/upload` - Upload documents

### Training
- `POST /api/training/start` - Start model training

## 🤖 AI Classifier

The system uses Persian BERT (`HooshvareLab/bert-fa-base-uncased`) for document classification:

**Categories:**
- `legal` - General legal documents
- `contract` - Contracts and agreements  
- `regulation` - Rules and regulations
- `court_decision` - Court rulings
- `other` - Miscellaneous documents

**Example Usage:**
```bash
curl -X POST "http://localhost:8000/api/ai/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "این یک قرارداد خرید و فروش است", "include_confidence": true}'
```

## 🐳 Production Deployment

### Using Docker Compose

1. **Build and Start Services**
```bash
docker-compose up --build -d
```

2. **Access Services**
- Application: http://localhost
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

3. **Monitor Services**
```bash
docker-compose logs -f
docker-compose ps
```

### Manual Docker Build

```bash
# Build backend
docker build -f Dockerfile.backend -t persian-legal-ai-backend .

# Build frontend  
docker build -f Dockerfile.frontend -t persian-legal-ai-frontend .
```

## 🧪 Testing

### Integration Tests
```bash
chmod +x test_integration.sh
./test_integration.sh
```

### Manual Testing
```bash
# Test backend
curl http://localhost:8000/api/system/health

# Test frontend
curl http://localhost:5173/

# Test integration
curl http://localhost:5173/api/system/health
```

## 📊 System Requirements

### Development
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies and models
- **CPU**: 2+ cores recommended

### Production
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for data and models
- **CPU**: 4+ cores recommended

## 🔒 Security Features

- CORS protection
- Request rate limiting
- Input validation
- Error handling
- Health monitoring
- Security headers

## 🌐 Persian Language Support

- **RTL Layout**: Right-to-left text direction
- **Persian Fonts**: Vazirmatn font family
- **Unicode Support**: Full Persian character set
- **Text Processing**: Persian text normalization
- **AI Model**: Persian BERT for classification

## 📝 Configuration

### Environment Variables

**Backend:**
- `DATABASE_URL`: Database connection string
- `DB_ECHO`: Enable SQL logging (true/false)
- `ENVIRONMENT`: Environment mode (development/production)

**Frontend:**
- `VITE_API_URL`: Backend API URL
- `NODE_ENV`: Environment mode

## 🚨 Troubleshooting

### Common Issues

1. **Backend won't start**
   - Check Python version (3.13+)
   - Verify virtual environment activation
   - Install dependencies: `pip install -r requirements.txt`

2. **Frontend build fails**
   - Check Node.js version (22+)
   - Clear node_modules: `rm -rf node_modules && npm install`

3. **AI model loading fails**
   - Check internet connection for model download
   - Verify sufficient disk space
   - Check system memory (4GB+ recommended)

4. **Docker deployment issues**
   - Check Docker version and permissions
   - Verify ports are available (80, 8000, 3000)
   - Check Docker Compose syntax

## 📈 Performance

### Benchmarks
- **Backend Response Time**: < 1000ms
- **Frontend Load Time**: < 2000ms
- **AI Classification**: ~3-5 seconds (CPU)
- **Database Queries**: < 100ms

### Optimization
- Model caching for faster inference
- Database indexing for search performance
- Gzip compression for frontend assets
- Connection pooling for database

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace**: Persian BERT model
- **FastAPI**: Modern Python web framework
- **React**: Frontend framework
- **Tailwind CSS**: Utility-first CSS framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs`

---

**🎯 Status**: ✅ Production Ready
**🔧 Version**: 2.0.0
**📅 Last Updated**: September 2025