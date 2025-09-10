# 🚀 Persian Legal AI System - Vercel Deployment Guide

## Production-Ready Vercel Setup Complete ✅

Your Persian Legal AI System frontend is now configured for production deployment on Vercel with an external FastAPI backend.

## 📁 Configuration Files Updated

### 1. Root `vercel.json`
```json
{
  "version": 2,
  "builds": [{ "src": "persian-legal-ai-frontend/package.json", "use": "@vercel/next" }],
  "routes": [{ "src": "/(.*)", "dest": "/persian-legal-ai-frontend/$1" }],
  "env": {
    "NEXT_PUBLIC_API_URL": "https://your-backend-url.com"
  }
}
```

### 2. Root `next.config.js`
- Added `NEXT_PUBLIC_API_URL` environment variable support
- Fallback to `http://localhost:8000` for local development

### 3. Frontend `next.config.js` 
- Updated environment variable injection
- Fixed API rewrites for development
- Enhanced security headers

### 4. Frontend `vercel.json`
- Simplified for Vercel deployment
- Removed unnecessary API functions (backend is external)

### 5. API Configuration (`src/utils/api.ts`)
- ✅ Properly configured to use `NEXT_PUBLIC_API_URL`
- ✅ All API endpoints correctly formatted
- ✅ Fallback to localhost for development

## 🚀 Deployment Steps

### Step 1: Deploy Your FastAPI Backend
Deploy your FastAPI backend to one of these platforms:

**Railway:**
```bash
railway login
railway init
railway add
railway deploy
# Get your URL: https://persian-legal-ai-backend.up.railway.app
```

**Render:**
```bash
# Connect GitHub repo to Render
# Get your URL: https://persian-legal-ai-backend.onrender.com
```

**VPS/Docker:**
```bash
# Deploy with Docker
docker build -t persian-legal-ai-backend .
docker run -d -p 8000:8000 persian-legal-ai-backend
# Get your URL: https://api.your-domain.com
```

### Step 2: Deploy Frontend to Vercel

**Option A: Vercel CLI**
```bash
npm install -g vercel
cd /workspace
vercel --prod
```

**Option B: GitHub Integration**
1. Push your code to GitHub
2. Connect repository to Vercel
3. Vercel will auto-deploy

### Step 3: Configure Environment Variables in Vercel

Go to your Vercel project dashboard → Settings → Environment Variables:

```bash
# Required Variables
NEXT_PUBLIC_API_URL=https://your-actual-backend-url.com
NODE_ENV=production

# Optional Variables (see .env.production for full list)
NEXT_PUBLIC_APP_NAME=Persian Legal AI System
NEXT_PUBLIC_ENABLE_TRAINING=true
NEXT_PUBLIC_DEFAULT_LOCALE=fa
```

## 🛠️ Local Development Workflow

### Terminal 1: Start Backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2: Start Frontend
```bash
cd persian-legal-ai-frontend
cp ../.env.local.example .env.local
npm install
npm run dev
```

Frontend will automatically proxy API calls to `http://localhost:8000`

## 🔗 Production Architecture

```
┌─────────────────┐    HTTPS     ┌──────────────────┐
│   Vercel CDN    │─────────────▶│   Next.js App    │
│  (Frontend)     │              │                  │
└─────────────────┘              └──────────────────┘
                                          │
                                          │ API Calls
                                          ▼
                                 ┌──────────────────┐
                                 │  FastAPI Backend │
                                 │ (Railway/Render) │
                                 └──────────────────┘
```

## ✅ Benefits of This Setup

1. **🚀 Fast Frontend**: Vercel's global CDN for optimal performance
2. **🔧 Flexible Backend**: Deploy FastAPI anywhere (Railway, Render, VPS)
3. **💰 Cost Effective**: Vercel free tier + external backend hosting
4. **🛡️ Secure**: No serverless limitations, proper CORS handling
5. **📱 Scalable**: Independent scaling of frontend and backend

## 🧪 Testing Your Deployment

1. **Local Test**: `npm run dev` → `http://localhost:3000`
2. **Production Test**: Visit your Vercel URL
3. **API Test**: Check Network tab for API calls to your backend URL

## 🔧 Environment Variables Reference

See `.env.production` for complete list of available environment variables.

## 🚨 Important Notes

- Replace `https://your-backend-url.com` with your actual FastAPI backend URL
- All `NEXT_PUBLIC_*` variables are exposed to the browser
- Backend must have proper CORS configuration for your Vercel domain
- Test your backend URL accessibility from Vercel's servers

## 🎉 You're Ready to Deploy!

Your Persian Legal AI System is now production-ready for Vercel deployment with external FastAPI backend hosting.