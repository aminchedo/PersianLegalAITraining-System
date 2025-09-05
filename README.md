# 🚀 Persian Legal AI Training System

A production-grade AI model training system optimized for Persian legal document processing with maximum system utilization on 24-core Intel systems.

## 📋 System Overview

This system implements advanced AI training techniques including DoRA (Weight-Decomposed Low-Rank Adaptation), QR-Adaptor with adaptive quantization, and real-time Persian legal data processing from multiple sources.

### 🎯 Key Features

- **DoRA Training**: Advanced weight decomposition with magnitude/direction optimization
- **QR-Adaptor**: Joint quantization and rank optimization for memory efficiency
- **Intel CPU Optimization**: Maximum utilization of 24-core systems with NUMA awareness
- **Real Data Integration**: Live connections to Persian legal databases
- **Real-time Monitoring**: WebSocket-based training progress tracking
- **Production Ready**: Comprehensive error handling, checkpointing, and recovery

## 🏗️ Architecture

```
persian-legal-ai/
├── backend/
│   ├── models/
│   │   ├── dora_trainer.py          # DoRA implementation
│   │   ├── qr_adaptor.py            # Quantization & rank optimization
│   │   ├── model_manager.py         # Central model management
│   │   └── persian_models.py        # Persian model loader
│   ├── services/
│   │   ├── training_service.py      # Training orchestration
│   │   └── persian_data_processor.py # Data processing pipeline
│   ├── optimization/
│   │   ├── intel_optimizer.py       # Intel CPU optimization
│   │   ├── system_optimizer.py      # System-wide optimization
│   │   └── memory_optimizer.py      # Memory optimization
│   ├── database/
│   │   ├── models.py                # Database models
│   │   └── connection.py            # Database manager
│   ├── api/
│   │   ├── training_endpoints.py    # Training API
│   │   ├── model_endpoints.py       # Model management API
│   │   └── system_endpoints.py      # System monitoring API
│   └── main.py                      # FastAPI server
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── persian-ai-api.js    # Enhanced API client
│   │   ├── hooks/
│   │   │   └── useTrainingSession.js # Training hooks
│   │   └── components/              # React components
└── docs/                            # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- 24-core Intel CPU (recommended)
- 64GB RAM (minimum)
- Intel Extension for PyTorch

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd persian-legal-ai
```

2. **Setup Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup Frontend**
```bash
cd frontend
npm install
```

4. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the System

1. **Start Backend Server**
```bash
cd backend
source venv/bin/activate
python main.py
```

2. **Start Frontend Dashboard**
```bash
cd frontend
npm start
```

3. **Access the System**
- Dashboard: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- System Health: http://localhost:8000/api/system/health

## 🧠 Core Components

### DoRA Trainer (`backend/models/dora_trainer.py`)

Advanced weight decomposition implementation:

```python
from backend.models.dora_trainer import DoRATrainer

# Initialize DoRA trainer
trainer = DoRATrainer({
    'base_model': 'universitytehran/PersianMind-v1.0',
    'dora_rank': 64,
    'dora_alpha': 16.0,
    'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"]
})

# Load and optimize model
model, tokenizer = trainer.load_model()
trainer.setup_optimizers(learning_rate=1e-4)

# Training step
metrics = trainer.train_step(batch)
```

### QR-Adaptor (`backend/models/qr_adaptor.py`)

Joint quantization and rank optimization:

```python
from backend.models.qr_adaptor import QRAdaptor, QuantizationConfig, RankConfig

# Configure quantization and rank optimization
quantization_config = QuantizationConfig(
    quant_type=QuantizationType.NF4,
    adaptive_bits=True,
    compression_target=0.5
)

rank_config = RankConfig(
    adaptive_rank=True,
    min_rank=16,
    max_rank=256
)

# Initialize QR-Adaptor
qr_adaptor = QRAdaptor(quantization_config, rank_config)

# Apply optimizations
model = qr_adaptor.apply_quantization(model)
model = qr_adaptor.optimize_ranks(model, gradients)
```

### Intel CPU Optimizer (`backend/optimization/intel_optimizer.py`)

Maximum CPU utilization for 24-core systems:

```python
from backend.optimization.intel_optimizer import IntelCPUOptimizer, CPUConfig

# Configure CPU optimization
cpu_config = CPUConfig(
    physical_cores=24,
    logical_cores=48,
    enable_intel_extension=True,
    numa_aware=True
)

# Initialize optimizer
optimizer = IntelCPUOptimizer(cpu_config)

# Optimize model for CPU
model = optimizer.optimize_model_for_cpu(model)

# Enable mixed precision
mixed_precision_config = optimizer.enable_mixed_precision_cpu()
```

### Persian Data Processor (`backend/services/persian_data_processor.py`)

Real-time Persian legal data processing:

```python
from backend.services.persian_data_processor import PersianLegalDataProcessor

# Initialize data processor
processor = PersianLegalDataProcessor()

# Fetch legal documents
documents = await processor.fetch_legal_documents(
    sources=['naab', 'qavanin', 'majles'],
    date_range=(start_date, end_date),
    categories=['laws', 'regulations']
)

# Preprocess Persian text
processed_docs = processor.preprocess_persian_text(documents)

# Create training datasets
qa_dataset = processor.create_training_datasets(
    processed_docs, 'question_answering'
)
```

## 📊 API Endpoints

### Training Management

- `POST /api/training/sessions` - Create training session
- `GET /api/training/sessions/{session_id}/status` - Get training status
- `GET /api/training/sessions/{session_id}/metrics` - Get training metrics
- `POST /api/training/sessions/{session_id}/control` - Control training (pause/resume/cancel)
- `GET /api/training/sessions` - List training sessions

### Model Management

- `GET /api/models/` - List registered models
- `POST /api/models/register` - Register new model
- `GET /api/models/{model_id}` - Get model details
- `POST /api/models/{model_id}/evaluate` - Evaluate model

### System Monitoring

- `GET /api/system/health` - System health check
- `GET /api/system/metrics` - System metrics
- `GET /api/system/performance` - Performance overview
- `POST /api/system/optimize` - Optimize system

### WebSocket Endpoints

- `ws://localhost:8000/api/training/ws/training/{session_id}` - Real-time training monitoring

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/persian_ai
# or for SQLite
SQLITE_PATH=persian_legal_ai.db

# Data Sources
NAAB_API_KEY=your_naab_api_key
QAVANIN_BASE_URL=https://qavanin.ir
MAJLES_BASE_URL=https://majlis.ir

# System Optimization
CPU_CORES=24
MEMORY_LIMIT_GB=64
ENABLE_INTEL_EXTENSION=true
NUMA_AWARE=true

# Training
MAX_CONCURRENT_SESSIONS=2
DEFAULT_BATCH_SIZE=4
DEFAULT_LEARNING_RATE=1e-4
```

### Training Configuration

```python
training_config = {
    "model_name": "PersianMind-v1.0",
    "model_type": "dora",
    "base_model": "universitytehran/PersianMind-v1.0",
    "dataset_sources": ["naab", "qavanin"],
    "training_config": {
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2
    },
    "model_config": {
        "dora_rank": 64,
        "dora_alpha": 16.0,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
}
```

## 📈 Performance Optimization

### Intel CPU Optimization

The system automatically detects and optimizes for Intel CPUs:

- **Intel Extension for PyTorch**: Automatic optimization when available
- **NUMA Awareness**: Memory allocation optimization for multi-socket systems
- **Thread Affinity**: Optimal thread binding to physical cores
- **AMX/AVX-512**: Automatic detection and utilization of advanced instruction sets

### Memory Optimization

- **Gradient Checkpointing**: Reduces memory usage during training
- **Mixed Precision**: FP16 training for memory efficiency
- **Memory Pooling**: Efficient memory allocation and reuse
- **Automatic Cleanup**: Background memory management

### System Monitoring

Real-time monitoring of:
- CPU utilization per core
- Memory usage and optimization
- Training progress and metrics
- System health and performance

## 🗄️ Database Schema

### Training Sessions
- Session management and tracking
- Progress monitoring
- Error handling and recovery

### Model Checkpoints
- Automatic checkpointing
- Best model tracking
- Version management

### Training Metrics
- Real-time metrics collection
- Performance analysis
- Historical data

## 🔍 Monitoring and Logging

### Real-time Monitoring
- WebSocket-based live updates
- Training progress visualization
- System performance metrics

### Logging
- Comprehensive logging system
- Structured log format
- Error tracking and debugging

### Health Checks
- System component health
- Database connectivity
- Resource utilization

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Production Deployment

```bash
# Use the provided deployment script
./scripts/production_deploy.sh
```

### System Requirements

**Minimum:**
- 16-core CPU
- 32GB RAM
- 100GB storage

**Recommended:**
- 24-core Intel CPU
- 64GB RAM
- 500GB SSD storage
- Intel Extension for PyTorch

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/

# Run frontend tests
cd frontend
npm test
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Training Guide](docs/training-guide.md) - Detailed training instructions
- [System Architecture](docs/architecture.md) - System design overview
- [Performance Tuning](docs/performance.md) - Optimization guidelines

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API documentation at `/docs`

## 🎯 Roadmap

- [ ] Multi-GPU support
- [ ] Advanced data augmentation
- [ ] Model serving infrastructure
- [ ] Advanced monitoring dashboard
- [ ] Automated hyperparameter tuning

---

**Built with ❤️ for Persian Legal AI**