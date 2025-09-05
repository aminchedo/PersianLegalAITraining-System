import React, { createContext, useContext, useState, useEffect } from 'react';
import { Brain, Database, Activity, BarChart3, FileText, Terminal, Users, Settings, Home } from 'lucide-react';

// Context for global state management
const AppContext = createContext();

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
};

// Mock data generators
const generateMetrics = (count = 50) => {
  return Array.from({ length: count }, (_, i) => ({
    time: new Date(Date.now() - (count - i) * 60000).toLocaleTimeString('fa-IR'),
    cpu: Math.random() * 80 + 10,
    memory: Math.random() * 60 + 20,
    gpu: Math.random() * 90 + 5,
    loss: Math.exp(-i/30) * (1 + Math.random() * 0.3),
    accuracy: 50 + i * 0.9 + Math.random() * 5,
    throughput: 80 + Math.random() * 40,
    temperature: 35 + Math.random() * 20,
    power: 150 + Math.random() * 100
  }));
};

// Global state provider
export const AppProvider = ({ children }) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [realTimeData, setRealTimeData] = useState(generateMetrics());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(3000);
  
  const [projects] = useState([
    { id: 'legal-2025', name: 'پروژه اسناد حقوقی 2025', status: 'active', progress: 67 },
    { id: 'criminal-ai', name: 'هوش مصنوعی حقوق جزا', status: 'completed', progress: 100 },
    { id: 'civil-qa', name: 'سیستم Q&A حقوق مدنی', status: 'pending', progress: 23 }
  ]);
  
  const [selectedProject, setSelectedProject] = useState('legal-2025');
  
  const [models, setModels] = useState([
    { 
      id: 1, 
      name: "PersianMind-v1.0", 
      status: "training", 
      progress: 67, 
      accuracy: 89.2,
      loss: 0.23,
      epochs: 8,
      timeRemaining: "2 ساعت 15 دقیقه",
      doraRank: 64,
      type: "language-model",
      description: "مدل پیشرفته زبان فارسی برای درک متون حقوقی",
      parameters: "7B",
      framework: "PyTorch",
      lastUpdated: new Date()
    },
    { 
      id: 2, 
      name: "ParsBERT-Legal", 
      status: "completed", 
      progress: 100, 
      accuracy: 92.5,
      loss: 0.15,
      epochs: 12,
      timeRemaining: "تکمیل شده",
      doraRank: 32,
      type: "bert-model",
      description: "مدل BERT تخصصی برای اسناد حقوقی",
      parameters: "110M",
      framework: "Transformers",
      lastUpdated: new Date(Date.now() - 86400000)
    },
    { 
      id: 3, 
      name: "Persian-QA-Advanced", 
      status: "pending", 
      progress: 0, 
      accuracy: 0,
      loss: 0,
      epochs: 0,
      timeRemaining: "در انتظار",
      doraRank: 128,
      type: "qa-model",
      description: "سیستم پرسش و پاسخ پیشرفته",
      parameters: "3B",
      framework: "PyTorch",
      lastUpdated: new Date()
    },
    { 
      id: 4, 
      name: "Legal-NER-v2", 
      status: "error", 
      progress: 45, 
      accuracy: 67.8,
      loss: 0.45,
      epochs: 3,
      timeRemaining: "خطا در آموزش",
      doraRank: 64,
      type: "ner-model",
      description: "شناسایی موجودیت‌های حقوقی",
      parameters: "340M",
      framework: "spaCy",
      lastUpdated: new Date()
    }
  ]);

  const [dataSources] = useState([
    { 
      id: 1,
      name: "پیکره نعب", 
      documents: 15420, 
      quality: 94, 
      status: "active", 
      speed: 125,
      type: "corpus",
      description: "پیکره جامع متون حقوقی فارسی",
      lastSync: new Date()
    },
    { 
      id: 2,
      name: "پورتال قوانین", 
      documents: 8932, 
      quality: 87, 
      status: "active", 
      speed: 89,
      type: "legal-portal",
      description: "قوانین و مقررات جمهوری اسلامی ایران",
      lastSync: new Date()
    },
    { 
      id: 3,
      name: "مجلس شورای اسلامی", 
      documents: 5673, 
      quality: 92, 
      status: "active", 
      speed: 67,
      type: "parliament",
      description: "لوایح و قوانین مصوب مجلس",
      lastSync: new Date()
    },
    { 
      id: 4,
      name: "پورتال داده ایران", 
      documents: 3241, 
      quality: 78, 
      status: "inactive", 
      speed: 0,
      type: "data-portal",
      description: "داده‌های باز دولتی",
      lastSync: new Date(Date.now() - 172800000)
    }
  ]);

  const [notifications, setNotifications] = useState([
    { 
      id: 1, 
      type: 'success', 
      title: 'آموزش تکمیل شد',
      message: 'مدل ParsBERT-Legal با دقت 92.5% آموزش داده شد', 
      time: new Date(Date.now() - 2 * 60000),
      read: false 
    },
    { 
      id: 2, 
      type: 'warning', 
      title: 'هشدار عملکرد',
      message: 'استفاده از CPU به 85% رسیده - بهینه‌سازی توصیه می‌شود', 
      time: new Date(Date.now() - 5 * 60000),
      read: false 
    },
    { 
      id: 3, 
      type: 'info', 
      title: 'جمع‌آوری داده',
      message: 'جمع‌آوری 1,247 سند جدید از پیکره نعب', 
      time: new Date(Date.now() - 10 * 60000),
      read: true 
    }
  ]);

  const [systemLogs, setSystemLogs] = useState([
    { id: 1, level: 'INFO', message: 'سیستم با موفقیت راه‌اندازی شد', timestamp: new Date(Date.now() - 1000), component: 'main' },
    { id: 2, level: 'SUCCESS', message: 'اتصال به پیکره نعب برقرار شد', timestamp: new Date(Date.now() - 2000), component: 'data-collector' },
    { id: 3, level: 'WARNING', message: 'استفاده از حافظه به 78% رسید', timestamp: new Date(Date.now() - 3000), component: 'monitor' },
    { id: 4, level: 'INFO', message: 'شروع آموزش مدل PersianMind-v1.0', timestamp: new Date(Date.now() - 4000), component: 'trainer' },
    { id: 5, level: 'ERROR', message: 'خطا در بارگیری مدل Legal-NER-v2', timestamp: new Date(Date.now() - 5000), component: 'model-loader' }
  ]);

  const [teamMembers] = useState([
    {
      id: 1,
      name: "علی احمدی",
      role: "مدیر پروژه",
      email: "ali.ahmadi@example.com",
      avatar: "AA",
      status: "online",
      projects: ["legal-2025", "criminal-ai"],
      permissions: ["admin", "model-training", "data-access"]
    },
    {
      id: 2,
      name: "فاطمه کریمی",
      role: "مهندس یادگیری ماشین",
      email: "fateme.karimi@example.com",
      avatar: "FK",
      status: "online",
      projects: ["legal-2025"],
      permissions: ["model-training", "data-access"]
    },
    {
      id: 3,
      name: "محمد رضایی",
      role: "متخصص داده",
      email: "mohammad.rezaei@example.com",
      avatar: "MR",
      status: "away",
      projects: ["civil-qa"],
      permissions: ["data-access", "data-annotation"]
    }
  ]);

  // Real-time data updates
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      setRealTimeData(prev => {
        const newPoint = {
          time: new Date().toLocaleTimeString('fa-IR'),
          cpu: Math.random() * 80 + 10,
          memory: Math.random() * 60 + 20,
          gpu: Math.random() * 90 + 5,
          loss: Math.random() * 0.5 + 0.1,
          accuracy: 85 + Math.random() * 10,
          throughput: 80 + Math.random() * 40,
          temperature: 35 + Math.random() * 20,
          power: 150 + Math.random() * 100
        };
        return [...prev.slice(1), newPoint];
      });

      // Add new logs occasionally
      if (Math.random() < 0.1) {
        const newLog = {
          id: Date.now(),
          level: ['INFO', 'WARNING', 'ERROR', 'SUCCESS'][Math.floor(Math.random() * 4)],
          message: [
            'عملیات بک‌آپ تکمیل شد',
            'بررسی سلامت سیستم انجام شد',
            'آپدیت مدل جدید اعمال شد',
            'تنظیمات امنیتی بروزرسانی شد'
          ][Math.floor(Math.random() * 4)],
          timestamp: new Date(),
          component: ['main', 'backup', 'model-manager', 'security'][Math.floor(Math.random() * 4)]
        };
        setSystemLogs(prev => [newLog, ...prev.slice(0, 49)]);
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  const value = {
    // Navigation
    activeTab,
    setActiveTab,
    sidebarCollapsed,
    setSidebarCollapsed,
    
    // Data
    realTimeData,
    setRealTimeData,
    projects,
    selectedProject,
    setSelectedProject,
    models,
    setModels,
    dataSources,
    notifications,
    setNotifications,
    systemLogs,
    setSystemLogs,
    teamMembers,
    
    // Settings
    autoRefresh,
    setAutoRefresh,
    refreshInterval,
    setRefreshInterval
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

// Simple Router Component
export const Router = ({ children }) => {
  const { activeTab } = useAppContext();
  
  const routes = {
    dashboard: children.find(child => child.props.path === '/dashboard'),
    models: children.find(child => child.props.path === '/models'),
    data: children.find(child => child.props.path === '/data'),
    monitoring: children.find(child => child.props.path === '/monitoring'),
    analytics: children.find(child => child.props.path === '/analytics'),
    reports: children.find(child => child.props.path === '/reports'),
    logs: children.find(child => child.props.path === '/logs'),
    team: children.find(child => child.props.path === '/team'),
    settings: children.find(child => child.props.path === '/settings')
  };

  return routes[activeTab] || children.find(child => child.props.path === '/dashboard');
};

export const Route = ({ path, component, children }) => {
  return component ? React.createElement(component) : children;
};