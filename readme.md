# ⚖️ Persian Legal AI Training System

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/your-repo/persian-legal-ai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/react-18.2+-blue.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/fastapi-0.104+-green.svg)](https://fastapi.tiangolo.com)

## 🎯 Project Overview

The **Persian Legal AI Training System** is a comprehensive, production-ready platform for training, managing, and deploying Persian language AI models specifically designed for legal document processing. This system implements cutting-edge techniques including DoRA (Weight-Decomposed Low-Rank Adaptation), QR-Adaptor optimization, and Intel CPU optimizations for maximum performance on Windows VPS environments.

### 🌟 Key Features

- **Advanced AI Training**: DoRA and QR-Adaptor implementations for efficient model training
- **Real-time Dashboard**: Modern React-based interface with Persian RTL support
- **Persian Legal Data Integration**: Direct connections to Iranian legal databases
- **Intel CPU Optimization**: Leverages Intel Extension for PyTorch with AMX and AVX-512 support
- **Production-Ready**: Comprehensive monitoring, logging, and deployment tools
- **Real-time Communication**: WebSocket-based live updates and notifications

## 🏗️ System Architecture

```
persian-legal-ai/
├── backend/                    # FastAPI Backend Server
│   ├── main.py                # Main server application
│   ├── requirements.txt       # Python dependencies
│   ├── api/                   # API endpoints
│   ├── models/                # AI model implementations
│   ├── services/              # Business logic services
│   ├── utils/                 # Utility functions
│   ├── config/                # Configuration files
│   └── logs/                  # Application logs
├── frontend/                   # React Frontend Dashboard
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── CompletePersianAIDashboard.tsx  # Main dashboard
│   │   │   └── AdvancedComponents.jsx          # Advanced UI components
│   │   ├── hooks/             # Custom React hooks
│   │   │   └── usePersianAI.js                 # Persian AI hooks
│   │   ├── api/               # API integration
│   │   │   └── persian-ai-api.js               # API client
│   │   └── styles/            # CSS and styling
│   ├── package.json           # Node.js dependencies
│   └── public/                # Static assets
├── scripts/                    # Automation scripts
│   ├── setup.sh              # Setup script
│   └── launcher.py           # Dashboard launcher
└── docs/                      # Documentation
    ├── README.txt             # Project documentation
    └── GUIDE.txt              # User guide
```

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Windows 10/11 Pro or Windows Server 2019/2022
- **Python**: 3.9 or higher
- **Node.js**: 16.0 or higher
- **CPU**: Intel Xeon 24-core (recommended) or similar high-performance processor
- **RAM**: 32-64GB
- **Storage**: 500GB+ SSD
- **Administrator Privileges**: Required for Intel optimizations

### Installation

1. **Clone or organize the project files**:
   ```bash
   # Run the organization script
   .\organize_files.ps1
   ```

2. **Backend Setup**:
   ```bash
   cd persian-legal-ai/backend
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install Intel Extension for PyTorch (optional but recommended)
   pip install intel-extension-for-pytorch
   ```

3. **Frontend Setup**:
   ```bash
   cd ../frontend
   
   # Install Node.js dependencies
   npm install
   
   # Install additional dependencies
   npm install tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```

4. **Configure Tailwind CSS**:
   Create `tailwind.config.js`:
   ```javascript
   module.exports = {
     content: ["./src/**/*.{js,jsx,ts,tsx}"],
     theme: {
       extend: {
         fontFamily: {
           'vazir': ['Vazirmatn', 'Tahoma', 'sans-serif'],
         },
       },
     },
     plugins: [],
   }
   ```

### Running the Application

#### Development Mode

1. **Start Backend Server**:
   ```bash
   cd backend
   venv\Scripts\activate
   python main.py
   ```
   Backend will be available at: `http://localhost:8000`

2. **Start Frontend Dashboard** (in new terminal):
   ```bash
   cd frontend
   npm start
   ```
   Frontend will be available at: `http://localhost:3000`

#### Production Mode

1. **Build Frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy with Docker** (optional):
   ```bash
   docker-compose up -d
   ```

## 🔧 Core Components

### Backend Components

#### 1. Main Server (`backend/main.py`)
```python
# FastAPI application with the following features:
- RESTful API endpoints
- WebSocket real-time communication
- Authentication and authorization
- Database integration
- Background task management
- Comprehensive logging
```

**Key API Endpoints**:
- `GET /api/status` - System status
- `GET /api/metrics` - Real-time system metrics
- `GET /api/models` - Model management
- `POST /api/training/start` - Start model training
- `WebSocket /ws` - Real-time updates

#### 2. AI Model Training
```python
# DoRA (Weight-Decomposed Low-Rank Adaptation)
class DoRATrainer:
    - Magnitude and direction decomposition
    - Separate learning rates for components
    - Support for Persian language models
    - Real-time progress tracking

# QR-Adaptor (Joint Quantization and Rank Optimization)
class QRAdaptor:
    - Adaptive quantization with NF4 support
    - Joint bit-width and rank optimization
    - Compression ratio optimization
    - Performance monitoring
```

#### 3. Data Collection Services
```python
# Persian Legal Data Sources
- Naab Corpus: 700GB Persian text corpus
- Iran Data Portal: Syracuse University legal documents
- Qavanin Portal: Official Iranian laws and regulations
- Majles Website: Iranian Parliament documents

# Real-time collection and processing
- Document quality assessment
- Automatic categorization
- Duplicate detection
- Text preprocessing for AI training
```

### Frontend Components

#### 1. Main Dashboard (`frontend/src/components/CompletePersianAIDashboard.tsx`)
```typescript
// Comprehensive dashboard with:
- Real-time system monitoring
- Model training management
- Data collection oversight
- Advanced analytics and reporting
- Persian RTL interface
- Dark/light mode support
```

#### 2. Advanced Components (`frontend/src/components/AdvancedComponents.jsx`)
```typescript
// Specialized UI components:
- ProjectManager: Multi-project management
- AIInsights: Intelligent predictions and recommendations
- SystemLogs: Real-time log viewing and filtering
- ModelController: Advanced model configuration
```

#### 3. Custom Hooks (`frontend/src/hooks/usePersianAI.js`)
```typescript
// React hooks for state management:
- usePersianAI: Main API integration
- useRealTimeMetrics: Live metric updates
- useModels: Model management
- useDataCollection: Data source management
- useNotifications: Alert system
- useSettings: Configuration management
```

#### 4. API Integration (`frontend/src/api/persian-ai-api.js`)
```typescript
// API client with:
- RESTful API communication
- WebSocket integration
- Error handling and retry logic
- Authentication management
- Real-time data streaming
```

## 🎮 Dashboard Features

### 1. Real-time Monitoring
- **System Metrics**: CPU, Memory, GPU usage with Intel optimizations
- **Training Progress**: Live training loss, accuracy, and throughput
- **Data Collection**: Real-time document processing statistics
- **Performance Analytics**: Bottleneck identification and optimization suggestions

### 2. Model Management
- **Training Control**: Start, stop, pause, and resume training sessions
- **Parameter Tuning**: Real-time adjustment of learning rates, batch sizes, DoRA parameters
- **Model Versioning**: Checkpoint management and model comparison
- **Performance Evaluation**: Comprehensive model testing and validation

### 3. Data Management
- **Source Integration**: Direct connections to Persian legal databases
- **Quality Control**: Automatic document quality assessment and filtering
- **Preprocessing Pipeline**: Text cleaning, tokenization, and preparation
- **Dataset Analytics**: Statistical analysis and visualization

### 4. Advanced Analytics
- **AI-Powered Insights**: Predictive analytics for training optimization
- **Performance Trends**: Historical analysis and trend prediction
- **Resource Optimization**: Intelligent resource allocation recommendations
- **Cost Analysis**: Training cost estimation and optimization

## 🔌 API Documentation

### Authentication Endpoints
```http
POST /api/auth/login          # User authentication
POST /api/auth/logout         # User logout
GET  /api/auth/user           # Get current user
POST /api/auth/refresh        # Refresh authentication token
```

### System Monitoring
```http
GET  /api/metrics/system      # Real-time system metrics
GET  /api/metrics/training    # Training session metrics
GET  /api/metrics/data        # Data collection statistics
WebSocket /ws/metrics         # Live metric streaming
```

### Model Management
```http
GET    /api/models            # List all models
POST   /api/models            # Create new model
GET    /api/models/{id}       # Get model details
PUT    /api/models/{id}       # Update model configuration
DELETE /api/models/{id}       # Delete model
POST   /api/models/{id}/train # Start training session
POST   /api/models/{id}/stop  # Stop training session
GET    /api/models/{id}/logs  # Get training logs
```

### Data Collection
```http
GET  /api/data/sources        # Available data sources
POST /api/data/collect        # Start data collection
GET  /api/data/status         # Collection status
GET  /api/data/quality        # Data quality metrics
POST /api/data/process        # Process collected data
```

### Advanced Features
```http
GET  /api/system/health       # System health check
GET  /api/system/logs         # System logs
POST /api/system/optimize     # System optimization
GET  /api/reports/generate    # Generate reports
POST /api/backup/create       # Create system backup
POST /api/backup/restore      # Restore from backup
```

## 🎯 Implementation Requirements

### 1. DoRA Training Implementation
```python
# Weight-Decomposed Low-Rank Adaptation
class DoRALayer:
    def __init__(self, rank, alpha, target_modules):
        self.magnitude_adapter = MagnitudeAdapter(rank)
        self.direction_adapter = DirectionAdapter(rank)
        self.scaling_factor = alpha / rank
    
    def forward(self, x):
        # Implement DoRA decomposition logic
        magnitude = self.magnitude_adapter(x)
        direction = self.direction_adapter(x)
        return x + magnitude * direction * self.scaling_factor
```

### 2. QR-Adaptor Optimization
```python
# Joint Quantization and Rank Optimization
class QRAdaptor:
    def __init__(self, quantization_bits, adaptive_rank):
        self.quantizer = AdaptiveQuantizer(quantization_bits)
        self.rank_optimizer = RankOptimizer(adaptive_rank)
    
    def optimize(self, model):
        # Implement joint optimization
        quantized_model = self.quantizer.quantize(model)
        optimized_model = self.rank_optimizer.optimize(quantized_model)
        return optimized_model
```

### 3. Persian Data Collection
```python
# Real data source integration
class PersianLegalDataCollector:
    def __init__(self):
        self.naab_connector = NaabCorpusConnector()
        self.qavanin_scraper = QavaninPortalScraper()
        self.majles_extractor = MajlesDataExtractor()
        self.iran_portal = IranDataPortalConnector()
    
    async def collect_documents(self, source, filters):
        # Implement real data collection
        documents = await source.fetch_documents(filters)
        processed_docs = self.preprocess_documents(documents)
        return self.quality_filter(processed_docs)
```

### 4. Intel CPU Optimization
```python
# Intel Extension for PyTorch integration
import intel_extension_for_pytorch as ipex

class IntelOptimizer:
    def __init__(self, model):
        self.model = model
        self.enable_amx = self.detect_amx_support()
        self.enable_avx512 = self.detect_avx512_support()
    
    def optimize(self):
        # Apply Intel optimizations
        optimized_model = ipex.optimize(
            self.model,
            level="O1",
            auto_kernel_selection=True
        )
        return optimized_model
```

## 🔒 Security & Production Features

### 1. Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- Session management

### 2. Data Security
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection

### 3. Monitoring & Logging
- Structured logging with correlation IDs
- Performance monitoring
- Error tracking and alerting
- Audit logging

### 4. Deployment & Scaling
- Docker containerization
- Kubernetes deployment
- Load balancing
- Auto-scaling configuration

## 📊 Performance Optimization

### 1. Backend Optimizations
- Async/await for I/O operations
- Connection pooling for databases
- Caching strategies (Redis)
- Background task processing

### 2. Frontend Optimizations
- Code splitting and lazy loading
- Memoization for expensive calculations
- Virtual scrolling for large lists
- Image optimization and lazy loading

### 3. AI Training Optimizations
- Mixed precision training
- Gradient accumulation
- Dynamic batching
- Model parallelism

## 🧪 Testing Strategy

### 1. Backend Testing
```bash
# Unit tests
pytest backend/tests/unit/

# Integration tests
pytest backend/tests/integration/

# API tests
pytest backend/tests/api/
```

### 2. Frontend Testing
```bash
# Unit tests
npm test

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e
```

## 📈 Monitoring & Observability

### 1. Metrics Collection
- System metrics (CPU, memory, disk, network)
- Application metrics (request rate, response time, error rate)
- Business metrics (training accuracy, data quality, user activity)

### 2. Logging
- Structured logging with JSON format
- Log aggregation and centralization
- Log analysis and alerting

### 3. Tracing
- Distributed tracing for API requests
- Performance profiling
- Bottleneck identification

## 🔧 Configuration Management

### Environment Variables
```bash
# Backend configuration
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your-secret-key
INTEL_OPTIMIZATION=true

# AI training configuration
TRAINING_BATCH_SIZE=4
LEARNING_RATE=1e-4
DORA_RANK=64
DORA_ALPHA=16.0
```

### Configuration Files
```yaml
# config/training.yaml
training:
  models:
    persian_mind:
      base_model: "universitytehran/PersianMind-v1.0"
      dora_config:
        rank: 64
        alpha: 16.0
        target_modules: ["q_proj", "v_proj"]
  
  data:
    sources:
      naab_corpus:
        url: "https://api.naab.ir/corpus"
        api_key: "${NAAB_API_KEY}"
      qavanin_portal:
        url: "https://qavanin.ir/api"
        rate_limit: 100
```

## 🚀 Deployment Guide

### 1. Development Deployment
```bash
# Backend
cd backend
python main.py

# Frontend
cd frontend
npm start
```

### 2. Production Deployment
```bash
# Build frontend
cd frontend
npm run build

# Deploy with Docker
docker-compose up -d

# Or deploy with systemd
sudo systemctl start persian-ai-backend
sudo systemctl start persian-ai-frontend
```

### 3. Cloud Deployment (AWS/Azure/GCP)
```bash
# Kubernetes deployment
kubectl apply -f k8s/

# Terraform infrastructure
terraform init
terraform plan
terraform apply
```

## 📚 Additional Resources

### Documentation
- [API Documentation](docs/api.md)
- [Development Guide](docs/development.md)
- [Deployment Guide](docs/deployment.md)
- [User Manual](docs/user-manual.md)

### External Dependencies
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Tailwind CSS](https://tailwindcss.com/docs)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint and Prettier for JavaScript/TypeScript
- Write comprehensive tests for new features
- Update documentation for API changes
- Use conventional commit messages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Intel Corporation** for Intel Extension for PyTorch
- **University of Tehran** for PersianMind model
- **HooshvareLab** for ParsBERT models
- **Syracuse University** for Iran Data Portal
- **Persian NLP Community** for various tools and resources

## 📞 Support

For support and questions:
- **Issues**: [GitHub Issues](https://github.com/your-repo/persian-legal-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/persian-legal-ai/discussions)
- **Email**: support@persian-legal-ai.com
- **Documentation**: [Project Wiki](https://github.com/your-repo/persian-legal-ai/wiki)

---

**Persian Legal AI Training System** - Advancing Persian language AI with cutting-edge 2025 techniques 🚀