# 🚀 Safe Deployment Guide for Persian Legal AI Training System

This guide will help you safely deploy the complete Persian Legal AI Training System to the main branch.

## 📋 Pre-Deployment Checklist

### 1. System Requirements Verification
- [ ] Python 3.9+ installed
- [ ] Node.js 16+ installed
- [ ] Git repository initialized
- [ ] All dependencies installed
- [ ] Tests passing
- [ ] Code quality checks passed

### 2. Backup Strategy
- [ ] Create backup branch before deployment
- [ ] Verify all changes are committed
- [ ] Document current system state

### 3. Testing Verification
- [ ] Backend tests passing
- [ ] Frontend tests passing
- [ ] Integration tests completed
- [ ] Performance tests validated

## 🔧 Deployment Methods

### Method 1: Automated Deployment Script (Recommended)

The safest way to deploy is using the provided deployment script:

```bash
# Make the script executable
chmod +x deploy_to_main.sh

# Run the deployment script
./deploy_to_main.sh
```

The script will:
1. ✅ Check git repository status
2. ✅ Create backup branch
3. ✅ Run all tests
4. ✅ Check code quality
5. ✅ Update dependencies
6. ✅ Create deployment documentation
7. ✅ Merge to main branch safely
8. ✅ Push to remote repository
9. ✅ Verify deployment

### Method 2: Manual Deployment

If you prefer manual control:

```bash
# 1. Check current status
git status
git branch

# 2. Create backup branch
git checkout -b backup-$(date +%Y%m%d-%H%M%S)

# 3. Return to feature branch
git checkout feature/persian-legal-ai-training-system

# 4. Run tests
cd backend && pytest tests/ && cd ..
cd frontend && npm test -- --watchAll=false && cd ..

# 5. Switch to main branch
git checkout main

# 6. Pull latest changes
git pull origin main

# 7. Merge feature branch
git merge feature/persian-legal-ai-training-system --no-ff -m "Merge: Complete Persian Legal AI Training System"

# 8. Push to remote
git push origin main

# 9. Verify deployment
git log --oneline -5
```

## 🛡️ Safety Features

### Automatic Backup
- Creates timestamped backup branch before deployment
- Preserves all changes in case of rollback needed

### Pre-deployment Checks
- Verifies git repository status
- Checks for uncommitted changes
- Runs comprehensive test suite
- Validates code quality

### Rollback Strategy
If deployment fails or issues are discovered:

```bash
# Switch to backup branch
git checkout backup-YYYYMMDD-HHMMSS

# Create new branch from backup
git checkout -b hotfix/rollback-$(date +%Y%m%d-%H%M%S)

# Fix issues and merge back
git checkout main
git merge hotfix/rollback-YYYYMMDD-HHMMSS
git push origin main
```

## 📊 Deployment Verification

After deployment, verify the system:

### 1. Backend Verification
```bash
cd backend
python main.py
# Check: http://localhost:8000/docs
```

### 2. Frontend Verification
```bash
cd frontend
npm start
# Check: http://localhost:3000
```

### 3. API Health Check
```bash
curl http://localhost:8000/api/system/health
```

### 4. Database Verification
```bash
# Check database connection
python -c "from backend.database.connection import db_manager; print(db_manager.test_connection())"
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Merge Conflicts
```bash
# Resolve conflicts manually
git status
# Edit conflicted files
git add .
git commit -m "Resolve merge conflicts"
```

#### 2. Test Failures
```bash
# Run specific test
pytest backend/tests/test_specific.py -v

# Run with verbose output
pytest backend/tests/ -v -s
```

#### 3. Dependency Issues
```bash
# Update requirements
pip install -r backend/requirements.txt --upgrade

# Clear cache
pip cache purge
```

#### 4. Database Issues
```bash
# Reset database
rm backend/persian_legal_ai.db
python -c "from backend.database.connection import db_manager; db_manager._initialize_database()"
```

## 📈 Post-Deployment Tasks

### 1. System Monitoring
- Monitor system performance
- Check training session functionality
- Verify data processing pipeline
- Test WebSocket connections

### 2. Documentation Updates
- Update API documentation
- Refresh system architecture docs
- Update deployment procedures

### 3. Team Communication
- Notify team of deployment
- Share deployment notes
- Update project status

## 🔄 Continuous Integration

For future deployments, consider setting up CI/CD:

```yaml
# .github/workflows/deploy.yml
name: Deploy to Main
on:
  push:
    branches: [ feature/persian-legal-ai-training-system ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          cd backend && pytest tests/
          cd frontend && npm test
      - name: Deploy to main
        run: ./deploy_to_main.sh
```

## 📞 Support

If you encounter issues during deployment:

1. Check the deployment logs
2. Review the backup branch
3. Consult the troubleshooting section
4. Create an issue on GitHub
5. Contact the development team

## 🎯 Success Criteria

Deployment is successful when:
- [ ] All tests pass
- [ ] Main branch is updated
- [ ] Backup branch is created
- [ ] System is accessible
- [ ] API endpoints respond correctly
- [ ] Frontend loads properly
- [ ] Training system is functional
- [ ] Documentation is updated

---

**Remember**: Always test in a development environment before deploying to production!