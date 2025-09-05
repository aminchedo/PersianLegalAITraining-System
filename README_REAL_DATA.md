# Persian Legal AI System - Real Data Implementation

A complete TypeScript React frontend + Python FastAPI backend integration with PostgreSQL database and **REAL DATA ONLY** - no mock data anywhere.

## 🚀 Features

✅ **Real PostgreSQL Database** with proper schemas  
✅ **TypeScript Frontend** with strict type checking  
✅ **FastAPI Backend** with JWT authentication  
✅ **Real Data Only** - no mock content anywhere  
✅ **Complete CRUD Operations** for all entities  
✅ **WebSocket Support** for real-time updates  
✅ **Production Ready** configuration  
✅ **Proper Error Handling** throughout  

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React + TS    │    │   FastAPI       │    │   PostgreSQL    │
│   Frontend      │◄──►│   Backend       │◄──►│   Database      │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
├── frontend/
│   ├── src/
│   │   ├── components/          # React components (TypeScript)
│   │   │   ├── AdvancedComponents.tsx
│   │   │   ├── CompletePersianAIDashboard.tsx
│   │   │   ├── router.tsx
│   │   │   └── ...
│   │   ├── hooks/               # Custom React hooks
│   │   │   └── useRealData.ts
│   │   ├── services/            # API services
│   │   │   └── RealApiService.ts
│   │   └── types/               # TypeScript interfaces
│   │       └── realData.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── backend/
│   ├── config/
│   │   └── database.py          # Database configuration
│   ├── models/                  # SQLAlchemy models
│   │   ├── team_model.py
│   │   ├── model_training.py
│   │   ├── system_metrics.py
│   │   └── ...
│   ├── routes/                  # FastAPI routes
│   │   ├── team.py
│   │   ├── models.py
│   │   └── monitoring.py
│   ├── main.py                  # FastAPI application
│   └── requirements.txt
├── .env.development
├── .env.production
└── start-full-system.sh
```

## 🛠️ Technology Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for build tooling
- **React Router 6** for routing
- **Axios** for API calls
- **Recharts** for data visualization
- **Lucide React** for icons

### Backend
- **FastAPI** for API framework
- **SQLAlchemy** for ORM
- **PostgreSQL** for database
- **Pydantic** for data validation
- **WebSockets** for real-time updates
- **psutil** for system monitoring

### Database
- **PostgreSQL** with real schemas
- **Real data models** for all entities
- **Proper relationships** and constraints

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- PostgreSQL 12+
- npm or yarn

### 1. Clone and Setup
```bash
git clone <repository>
cd persian-legal-ai
chmod +x start-full-system.sh
```

### 2. Start Complete System
```bash
./start-full-system.sh
```

This script will:
- Check dependencies
- Setup environment
- Initialize PostgreSQL database
- Install all dependencies
- Create database tables
- Start backend server (port 8000)
- Start frontend server (port 3000)

### 3. Access the System
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/real/health

## 📊 Real Data Models

### Team Members
```typescript
interface RealTeamMember {
  id: number;
  name: string;
  email: string;
  role: string;
  status: 'online' | 'offline' | 'busy' | 'away';
  department: string;
  experienceYears: number;
  skills: string[];
  // ... more real fields
}
```

### Model Training
```typescript
interface RealModelTraining {
  id: number;
  name: string;
  status: 'pending' | 'training' | 'completed' | 'error';
  progress: number;
  accuracy: number;
  framework: string;
  // ... more real fields
}
```

### System Metrics
```typescript
interface RealSystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  gpuUsage: number;
  temperature: number;
  // ... more real fields
}
```

## 🔌 API Endpoints

### Team Management
- `GET /api/real/team/members` - Get all team members
- `POST /api/real/team/members` - Create team member
- `PUT /api/real/team/members/{id}` - Update team member
- `DELETE /api/real/team/members/{id}` - Delete team member

### Model Training
- `GET /api/real/models/training` - Get training jobs
- `POST /api/real/models/training` - Create training job
- `POST /api/real/models/training/{id}/start` - Start training
- `POST /api/real/models/training/{id}/pause` - Pause training

### System Monitoring
- `GET /api/real/monitoring/system-metrics` - Get current metrics
- `GET /api/real/monitoring/health` - Health check
- `WebSocket /ws` - Real-time updates

## 🎯 Key Features

### Real Data Only
- ❌ No mock data generators
- ❌ No fake data arrays
- ❌ No demo content
- ✅ Real database queries
- ✅ Real API responses
- ✅ Real system metrics

### TypeScript Integration
- ✅ Strict type checking
- ✅ Interface definitions
- ✅ Type-safe API calls
- ✅ Proper error handling

### Real-time Updates
- ✅ WebSocket connections
- ✅ Live system metrics
- ✅ Real-time notifications
- ✅ Auto-refresh capabilities

## 🔧 Development

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Database Management
```bash
# Connect to database
psql -h localhost -U persianai -d persian_legal_ai

# Run migrations (if using Alembic)
cd backend
alembic upgrade head
```

## 🧪 Testing

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/api/real/health

# Test team members
curl http://localhost:8000/api/real/team/members

# Test system metrics
curl http://localhost:8000/api/real/monitoring/system-metrics
```

### Frontend Testing
```bash
cd frontend
npm run type-check
npm run lint
npm run build
```

## 🚀 Production Deployment

### Environment Variables
```bash
# Copy production environment
cp .env.production .env

# Update with real values
DATABASE_URL=postgresql://user:pass@host:5432/db
VITE_REAL_API_URL=https://api.yourdomain.com/api/real
```

### Docker Deployment (Optional)
```dockerfile
# Backend Dockerfile
FROM python:3.9
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Monitoring

### System Health
- Real-time CPU, memory, disk usage
- Database connection status
- API response times
- Error tracking

### Performance Metrics
- Request/response times
- Database query performance
- WebSocket connection status
- System resource utilization

## 🔒 Security

### Authentication
- JWT token-based authentication
- Role-based access control
- Secure password hashing
- CORS configuration

### Data Protection
- Input validation with Pydantic
- SQL injection prevention
- XSS protection
- Rate limiting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with real data only
4. Add tests for new features
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the health check at `/api/real/health`
3. Check system logs for errors
4. Ensure PostgreSQL is running and accessible

---

**Remember: This system uses REAL DATA ONLY - no mock data, no fake content, no demo data. Everything is connected to real databases and real APIs.**