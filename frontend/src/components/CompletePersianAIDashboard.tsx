import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, ScatterChart, Scatter, RadialBarChart, RadialBar } from 'recharts';
import { 
  Play, Pause, Square, Settings, Download, Upload, Eye, EyeOff, Maximize2, RefreshCw, 
  Bell, Search, Filter, Plus, X, Menu, Home, Brain, Database, Activity, FileText, 
  Users, Zap, Shield, TrendingUp, Clock, Check, AlertTriangle, Info, AlertCircle,
  CheckCircle, Monitor, Cpu, HardDrive, Thermometer, Power, Network, Globe,
  Calendar, BarChart3, Code, Terminal, BookOpen,
  Star, Heart, MessageCircle, Send, Bookmark, Share2, MoreHorizontal,
  ChevronLeft, ChevronRight, ChevronUp, ChevronDown, ExternalLink
} from 'lucide-react';

// Mock Data Generator
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

const CompletePersianAIDashboard = () => {
  // State Management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [realTimeData, setRealTimeData] = useState(generateMetrics());
  const [refreshInterval, setRefreshInterval] = useState(3000);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [fullScreenChart, setFullScreenChart] = useState(null);
  const [showNotifications, setShowNotifications] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);
  
  // System State
  const [systemStatus, setSystemStatus] = useState({
    isConnected: true,
    trainingActive: false,
    collectionActive: false,
    systemHealth: 'excellent'
  });

  // Projects and Models
  const [projects] = useState([
    { id: 'legal-2025', name: 'پروژه اسناد حقوقی 2025', status: 'active', progress: 67 },
    { id: 'criminal-ai', name: 'هوش مصنوعی حقوق جزا', status: 'completed', progress: 100 },
    { id: 'civil-qa', name: 'سیستم Q&A حقوق مدنی', status: 'pending', progress: 23 }
  ]);
  
  const [selectedProject, setSelectedProject] = useState('legal-2025');
  
  const [models] = useState([
    { 
      id: 1, 
      name: "PersianMind-v1.0", 
      status: "training", 
      progress: 67, 
      accuracy: 89.2,
      loss: 0.23,
      epochs: 8,
      timeRemaining: "2 ساعت 15 دقیقه",
      doraRank: 64
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
      doraRank: 32
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
      doraRank: 128
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
      doraRank: 64
    }
  ]);

  const [dataSources] = useState([
    { name: "پیکره نعب", documents: 15420, quality: 94, status: "active", speed: 125 },
    { name: "پورتال قوانین", documents: 8932, quality: 87, status: "active", speed: 89 },
    { name: "مجلس شورای اسلامی", documents: 5673, quality: 92, status: "active", speed: 67 },
    { name: "پورتال داده ایران", documents: 3241, quality: 78, status: "inactive", speed: 0 }
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
    },
    { 
      id: 4, 
      type: 'error', 
      title: 'خطا در مدل',
      message: 'خطا در آموزش مدل Legal-NER-v2 - بررسی تنظیمات', 
      time: new Date(Date.now() - 15 * 60000),
      read: false 
    }
  ]);

  const [systemLogs] = useState([
    { id: 1, level: 'INFO', message: 'سیستم با موفقیت راه‌اندازی شد', timestamp: new Date(Date.now() - 1000), component: 'main' },
    { id: 2, level: 'SUCCESS', message: 'اتصال به پیکره نعب برقرار شد', timestamp: new Date(Date.now() - 2000), component: 'data-collector' },
    { id: 3, level: 'WARNING', message: 'استفاده از حافظه به 78% رسید', timestamp: new Date(Date.now() - 3000), component: 'monitor' },
    { id: 4, level: 'INFO', message: 'شروع آموزش مدل PersianMind-v1.0', timestamp: new Date(Date.now() - 4000), component: 'trainer' },
    { id: 5, level: 'ERROR', message: 'خطا در بارگیری مدل Legal-NER-v2', timestamp: new Date(Date.now() - 5000), component: 'model-loader' }
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
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  // Utility Functions
  const formatNumber = (num) => {
    return new Intl.NumberFormat('fa-IR').format(Math.round(num));
  };

  const getStatusColor = (status) => {
    const colors = {
      training: 'bg-blue-500',
      completed: 'bg-green-500',
      pending: 'bg-yellow-500',
      error: 'bg-red-500',
      active: 'bg-green-500',
      inactive: 'bg-gray-500'
    };
    return colors[status] || 'bg-gray-500';
  };

  const getStatusText = (status) => {
    const texts = {
      training: 'در حال آموزش',
      completed: 'تکمیل شده',
      pending: 'در انتظار',
      error: 'خطا',
      active: 'فعال',
      inactive: 'غیرفعال'
    };
    return texts[status] || status;
  };

  // Components
  const MenuItem = ({ icon: Icon, label, id, badge, notifications }) => (
    <div
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-3 px-4 py-3 rounded-xl cursor-pointer transition-all duration-300 group relative ${
        activeTab === id 
          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg transform scale-105' 
          : 'text-gray-700 hover:bg-gray-100 hover:transform hover:scale-105'
      }`}
    >
      <Icon className={`w-5 h-5 ${activeTab === id ? 'text-white' : 'text-gray-500 group-hover:text-blue-600'}`} />
      {!sidebarCollapsed && (
        <>
          <span className="font-medium">{label}</span>
          {badge && (
            <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full ml-auto">
              {badge}
            </span>
          )}
          {notifications && (
            <div className="absolute left-2 top-2 w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
          )}
        </>
      )}
    </div>
  );

  const MetricCard = ({ title, value, unit, icon: Icon, color, trend, subtitle, chart }) => (
    <div className={`bg-white rounded-2xl p-6 shadow-lg border-l-4 border-${color}-500 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 group`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex-1">
          <p className="text-gray-600 text-sm font-medium">{title}</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-3xl font-bold text-gray-900">{formatNumber(value)}</h3>
            <span className="text-gray-500 text-sm">{unit}</span>
          </div>
          {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
        </div>
        <div className={`bg-${color}-100 rounded-xl p-3 group-hover:scale-110 transition-transform`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
      
      {trend && (
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-green-500" />
          <span className="text-green-500 text-sm font-medium">+{trend}%</span>
          <span className="text-gray-500 text-sm">نسبت به دیروز</span>
        </div>
      )}
      
      {chart && (
        <div className="h-12 mt-3">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={realTimeData.slice(-10)}>
              <Area 
                type="monotone" 
                dataKey={chart} 
                stroke={`var(--${color}-500)`} 
                fill={`var(--${color}-500)`} 
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );

  const StatusBadge = ({ status, children }) => {
    const colors = {
      active: 'bg-green-100 text-green-800 border-green-200',
      training: 'bg-blue-100 text-blue-800 border-blue-200',
      completed: 'bg-green-100 text-green-800 border-green-200',
      pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      error: 'bg-red-100 text-red-800 border-red-200',
      inactive: 'bg-gray-100 text-gray-800 border-gray-200'
    };

    return (
      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${colors[status]}`}>
        {children}
      </span>
    );
  };

  const NotificationPanel = () => (
    <div className={`absolute top-16 left-4 w-96 bg-white rounded-2xl shadow-2xl border border-gray-200 transition-all duration-300 z-50 ${showNotifications ? 'opacity-100 visible' : 'opacity-0 invisible'}`}>
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-900">اعلان‌ها</h3>
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setNotifications(prev => prev.map(n => ({ ...n, read: true })))}
              className="text-blue-600 text-sm hover:text-blue-700"
            >
              همه را خوانده علامت بزن
            </button>
            <button 
              onClick={() => setShowNotifications(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
      <div className="max-h-96 overflow-y-auto">
        {notifications.map(notification => (
          <div key={notification.id} className={`p-4 border-b border-gray-100 hover:bg-gray-50 transition-all ${!notification.read ? 'bg-blue-50' : ''}`}>
            <div className="flex items-start gap-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                notification.type === 'success' ? 'bg-green-100 text-green-600' :
                notification.type === 'warning' ? 'bg-yellow-100 text-yellow-600' :
                notification.type === 'error' ? 'bg-red-100 text-red-600' :
                'bg-blue-100 text-blue-600'
              }`}>
                {notification.type === 'success' ? <CheckCircle className="w-4 h-4" /> :
                 notification.type === 'warning' ? <AlertTriangle className="w-4 h-4" /> :
                 notification.type === 'error' ? <AlertCircle className="w-4 h-4" /> :
                 <Info className="w-4 h-4" />}
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900">{notification.title}</h4>
                <p className="text-sm text-gray-600 leading-5">{notification.message}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {new Date(notification.time).toLocaleString('fa-IR')}
                </p>
              </div>
              {!notification.read && (
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="p-4 border-t border-gray-200">
        <button className="w-full text-center text-blue-600 text-sm hover:text-blue-700">
          مشاهده همه اعلان‌ها
        </button>
      </div>
    </div>
  );

  // Main Dashboard Content
  const renderDashboard = () => (
    <div className="space-y-6">
      {/* Enhanced Header */}
      <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl p-8 text-white relative overflow-hidden">
        <div className="absolute inset-0 bg-black opacity-10"></div>
        <div className="absolute top-0 left-0 w-full h-full">
          <div className="absolute top-4 left-4 w-20 h-20 bg-white/10 rounded-full"></div>
          <div className="absolute bottom-4 right-8 w-32 h-32 bg-white/5 rounded-full"></div>
          <div className="absolute top-1/2 right-1/4 w-16 h-16 bg-white/10 rounded-full"></div>
        </div>
        
        <div className="relative z-10">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                ⚖️ سیستم هوش مصنوعی حقوقی فارسی
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
              </h1>
              <p className="text-blue-100">پیشرفته‌ترین سیستم آموزش مدل‌های زبانی فارسی - نسخه ۲۰۲۵</p>
            </div>
            <div className="flex items-center gap-4">
              <select 
                value={selectedProject}
                onChange={(e) => setSelectedProject(e.target.value)}
                className="bg-white/20 text-white border border-white/30 rounded-xl px-4 py-2 backdrop-blur-sm"
              >
                {projects.map(project => (
                  <option key={project.id} value={project.id} className="text-gray-900">
                    {project.name}
                  </option>
                ))}
              </select>
              <button 
                onClick={() => setSystemStatus(prev => ({ ...prev, trainingActive: !prev.trainingActive }))}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 ${
                  systemStatus.trainingActive 
                    ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg' 
                    : 'bg-white text-blue-600 hover:bg-blue-50 shadow-lg'
                }`}
              >
                {systemStatus.trainingActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {systemStatus.trainingActive ? 'توقف آموزش' : 'شروع آموزش'}
              </button>
            </div>
          </div>
          
          {/* Quick Stats */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4" />
                <span className="text-sm">زمان آموزش</span>
              </div>
              <p className="text-xl font-bold">2ساعت 34دقیقه</p>
            </div>
            <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4" />
                <span className="text-sm">بهبود دقت</span>
              </div>
              <p className="text-xl font-bold">+12.5%</p>
            </div>
            <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4" />
                <span className="text-sm">سرعت پردازش</span>
              </div>
              <p className="text-xl font-bold">156 توکن/ثانیه</p>
            </div>
            <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-4 h-4" />
                <span className="text-sm">وضعیت سیستم</span>
              </div>
              <p className="text-xl font-bold text-green-300">عالی</p>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="استفاده از پردازنده" 
          value={realTimeData[realTimeData.length-1]?.cpu || 0} 
          unit="%" 
          icon={Cpu} 
          color="blue" 
          trend="12"
          subtitle="24 هسته فعال"
          chart="cpu"
        />
        <MetricCard 
          title="استفاده از حافظه" 
          value={realTimeData[realTimeData.length-1]?.memory || 0} 
          unit="%" 
          icon={HardDrive} 
          color="green" 
          trend="8"
          subtitle="از 64 گیگابایت"
          chart="memory"
        />
        <MetricCard 
          title="دقت مدل فعال" 
          value={realTimeData[realTimeData.length-1]?.accuracy || 0} 
          unit="%" 
          icon={Brain} 
          color="purple" 
          trend="5"
          subtitle="PersianMind-v1.0"
          chart="accuracy"
        />
        <MetricCard 
          title="اسناد پردازش شده" 
          value={33266} 
          unit="" 
          icon={FileText} 
          color="orange" 
          trend="23"
          subtitle="در 24 ساعت گذشته"
        />
      </div>

      {/* Advanced Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Performance Chart */}
        <div className="bg-white rounded-2xl p-6 shadow-lg relative group">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">عملکرد سیستم در زمان واقعی</h3>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-500">زنده</span>
              </div>
              <button 
                onClick={() => setFullScreenChart('system-performance')}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-2 hover:bg-gray-100 rounded-lg"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
              <button 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className={`p-2 rounded-lg transition-all ${autoRefresh ? 'text-green-600 bg-green-50' : 'text-gray-400 hover:bg-gray-100'}`}
              >
                <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={realTimeData}>
              <defs>
                <linearGradient id="cpuGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="memoryGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: 'none', 
                  borderRadius: '12px', 
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' 
                }}
              />
              <Area type="monotone" dataKey="cpu" stroke="#3B82F6" fill="url(#cpuGradient)" strokeWidth={2} />
              <Area type="monotone" dataKey="memory" stroke="#10B981" fill="url(#memoryGradient)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>CPU ({(realTimeData[realTimeData.length-1]?.cpu || 0).toFixed(1)}%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>حافظه ({(realTimeData[realTimeData.length-1]?.memory || 0).toFixed(1)}%)</span>
            </div>
          </div>
        </div>

        {/* Training Progress Chart */}
        <div className="bg-white rounded-2xl p-6 shadow-lg relative group">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">آنالیز پیشرفت آموزش</h3>
            <div className="flex items-center gap-2">
              <button 
                onClick={() => setFullScreenChart('training-progress')}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-2 hover:bg-gray-100 rounded-lg"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
              <button 
                onClick={() => setShowAdvancedMetrics(!showAdvancedMetrics)}
                className={`p-2 rounded-lg transition-all ${showAdvancedMetrics ? 'text-purple-600 bg-purple-50' : 'text-gray-400 hover:bg-gray-100'}`}
              >
                <Eye className="w-4 h-4" />
              </button>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={realTimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'white', 
                  border: 'none', 
                  borderRadius: '12px', 
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' 
                }}
              />
              <Line type="monotone" dataKey="loss" stroke="#EF4444" strokeWidth={3} dot={false} />
              <Line type="monotone" dataKey="accuracy" stroke="#8B5CF6" strokeWidth={3} dot={false} />
              {showAdvancedMetrics && (
                <Line type="monotone" dataKey="throughput" stroke="#F59E0B" strokeWidth={2} dot={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>خطا: {(realTimeData[realTimeData.length-1]?.loss || 0).toFixed(3)}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
              <span>دقت: {(realTimeData[realTimeData.length-1]?.accuracy || 0).toFixed(1)}%</span>
            </div>
            {showAdvancedMetrics && (
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span>سرعت: {(realTimeData[realTimeData.length-1]?.throughput || 0).toFixed(0)} tok/s</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Models and Data Sources Enhanced */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Models Panel */}
        <div className="lg:col-span-2 bg-white rounded-2xl p-6 shadow-lg">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">مدل‌های در حال آموزش</h3>
            <div className="flex items-center gap-2">
              <button className="bg-blue-600 text-white px-3 py-1 rounded-lg text-sm hover:bg-blue-700 transition-all">
                + جدید
              </button>
              <button className="text-gray-500 hover:text-gray-700">
                <Settings className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div className="space-y-4">
            {models.map(model => (
              <div key={model.id} className="group relative">
                <div className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-white rounded-xl border border-gray-100 hover:shadow-md transition-all duration-300">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${getStatusColor(model.status)} bg-opacity-10`}>
                      <Brain className={`w-6 h-6 ${getStatusColor(model.status).replace('bg-', 'text-')}`} />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{model.name}</p>
                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span>دقت: {model.accuracy}%</span>
                        <span>•</span>
                        <span>DoRA: {model.doraRank}</span>
                        <span>•</span>
                        <span>{model.timeRemaining}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="w-32 bg-gray-200 rounded-full h-2 mb-1">
                        <div 
                          className={`h-2 rounded-full transition-all duration-500 ${getStatusColor(model.status)}`}
                          style={{ width: `${model.progress}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">{model.progress}%</span>
                    </div>
                    <StatusBadge status={model.status}>
                      {getStatusText(model.status)}
                    </StatusBadge>
                  </div>
                </div>
                
                <div className="absolute left-4 top-1/2 transform -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="flex gap-1">
                    <button className="w-8 h-8 bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 transition-all">
                      <Eye className="w-4 h-4" />
                    </button>
                    <button className="w-8 h-8 bg-gray-600 text-white rounded-lg flex items-center justify-center hover:bg-gray-700 transition-all">
                      <Settings className="w-4 h-4" />
                    </button>
                    {model.status === 'training' && (
                      <button className="w-8 h-8 bg-red-600 text-white rounded-lg flex items-center justify-center hover:bg-red-700 transition-all">
                        <Pause className="w-4 h-4" />
                      </button>
                    )}
                    {model.status === 'pending' && (
                      <button className="w-8 h-8 bg-green-600 text-white rounded-lg flex items-center justify-center hover:bg-green-700 transition-all">
                        <Play className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Data Sources Panel */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">منابع داده</h3>
            <button 
              onClick={() => setSystemStatus(prev => ({ ...prev, collectionActive: !prev.collectionActive }))}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${
                systemStatus.collectionActive 
                  ? 'bg-red-100 text-red-600 hover:bg-red-200' 
                  : 'bg-green-100 text-green-600 hover:bg-green-200'
              }`}
            >
              {systemStatus.collectionActive ? 'توقف' : 'شروع'}
            </button>
          </div>
          
          <div className="space-y-4">
            {dataSources.map((source, index) => (
              <div key={index} className="group">
                <div className="flex items-center justify-between p-3 bg-gradient-to-r from-gray-50 to-white rounded-xl border border-gray-100 hover:shadow-md transition-all duration-300">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                      source.status === 'active' ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-600'
                    }`}>
                      <Database className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900 text-sm">{source.name}</p>
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <span>{formatNumber(source.documents)} سند</span>
                        <span>•</span>
                        <span>{source.quality}% کیفیت</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex flex-col items-end gap-1">
                    <StatusBadge status={source.status}>
                      {getStatusText(source.status)}
                    </StatusBadge>
                    {source.status === 'active' && (
                      <span className="text-xs text-gray-500">{source.speed}/ساعت</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Data Quality Distribution */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <h4 className="text-sm font-medium text-gray-900 mb-4">توزیع کیفیت</h4>
            <ResponsiveContainer width="100%" height={120}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'عالی', value: 15000, fill: '#10B981' },
                    { name: 'خوب', value: 12000, fill: '#3B82F6' },
                    { name: 'متوسط', value: 5000, fill: '#F59E0B' },
                    { name: 'ضعیف', value: 1266, fill: '#EF4444' }
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={30}
                  outerRadius={50}
                  paddingAngle={2}
                  dataKey="value"
                >
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* AI Insights Panel Enhanced */}
      <div className="bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 rounded-2xl p-8 text-white relative overflow-hidden">
        <div className="absolute inset-0 bg-black opacity-10"></div>
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-6">
            <Brain className="w-6 h-6" />
            <h3 className="text-xl font-semibold">تحلیل و پیش‌بینی هوشمند</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white/20 rounded-xl p-6 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-3">
                <TrendingUp className="w-5 h-5" />
                <span className="font-medium">پیش‌بینی عملکرد</span>
              </div>
              <p className="text-lg leading-relaxed">دقت مدل PersianMind تا 48 ساعت آینده به <strong>94.2%</strong> خواهد رسید</p>
              <div className="mt-3 flex items-center gap-2">
                <div className="text-xs opacity-75">اطمینان:</div>
                <div className="flex-1 bg-white/20 rounded-full h-1">
                  <div className="bg-white h-1 rounded-full w-4/5"></div>
                </div>
                <div className="text-xs">87%</div>
              </div>
            </div>
            
            <div className="bg-white/20 rounded-xl p-6 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-3">
                <AlertTriangle className="w-5 h-5" />
                <span className="font-medium">هشدار سیستم</span>
              </div>
              <p className="text-lg leading-relaxed">استفاده از حافظه در <strong>3 ساعت</strong> آینده به 85% خواهد رسید</p>
              <div className="mt-3 flex items-center gap-2">
                <div className="text-xs opacity-75">اطمینان:</div>
                <div className="flex-1 bg-white/20 rounded-full h-1">
                  <div className="bg-white h-1 rounded-full w-3/4"></div>
                </div>
                <div className="text-xs">78%</div>
              </div>
            </div>
            
            <div className="bg-white/20 rounded-xl p-6 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-3">
                <Zap className="w-5 h-5" />
                <span className="font-medium">توصیه بهینه‌سازی</span>
              </div>
              <p className="text-lg leading-relaxed">کاهش batch size به <strong>2</strong> برای بهبود 15% کارایی توصیه می‌شود</p>
              <div className="mt-3 flex items-center gap-2">
                <div className="text-xs opacity-75">اطمینان:</div>
                <div className="flex-1 bg-white/20 rounded-full h-1">
                  <div className="bg-white h-1 rounded-full w-5/6"></div>
                </div>
                <div className="text-xs">91%</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Full Screen Chart Modal
  const FullScreenModal = () => {
    if (!fullScreenChart) return null;

    return (
      <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl w-full max-w-7xl h-5/6 p-6 relative">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold">
              {fullScreenChart === 'system-performance' ? 'عملکرد سیستم - نمای کامل' : 'پیشرفت آموزش - نمای کامل'}
            </h2>
            <button 
              onClick={() => setFullScreenChart(null)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-all"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="h-full pb-16">
            <ResponsiveContainer width="100%" height="100%">
              {fullScreenChart === 'system-performance' ? (
                <AreaChart data={realTimeData}>
                  <defs>
                    <linearGradient id="cpuGradientFull" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                    </linearGradient>
                    <linearGradient id="memoryGradientFull" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Area type="monotone" dataKey="cpu" stroke="#3B82F6" fill="url(#cpuGradientFull)" strokeWidth={3} />
                  <Area type="monotone" dataKey="memory" stroke="#10B981" fill="url(#memoryGradientFull)" strokeWidth={3} />
                </AreaChart>
              ) : (
                <LineChart data={realTimeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="time" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip />
                  <Line type="monotone" dataKey="loss" stroke="#EF4444" strokeWidth={4} dot={false} />
                  <Line type="monotone" dataKey="accuracy" stroke="#8B5CF6" strokeWidth={4} dot={false} />
                  <Line type="monotone" dataKey="throughput" stroke="#F59E0B" strokeWidth={3} dot={false} />
                </LineChart>
              )}
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };

  // Main Return
  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`} dir="rtl">
      <FullScreenModal />
      
      {/* Enhanced Sidebar */}
      <div className={`fixed top-0 right-0 h-full bg-white shadow-2xl border-l border-gray-200 transition-all duration-300 z-40 ${
        sidebarCollapsed ? 'w-20' : 'w-72'
      }`}>
        <div className="p-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <Brain className="w-7 h-7 text-white" />
            </div>
            {!sidebarCollapsed && (
              <div>
                <h1 className="font-bold text-gray-900 text-lg">AI حقوقی فارسی</h1>
                <p className="text-xs text-gray-500">نسخه پیشرفته ۲۰۲۵</p>
              </div>
            )}
          </div>
        </div>

        <nav className="px-4 space-y-2">
          <MenuItem icon={Home} label="داشبورد" id="dashboard" />
          <MenuItem icon={Brain} label="مدل‌ها" id="models" badge="4" />
          <MenuItem icon={Database} label="داده‌ها" id="data" />
          <MenuItem icon={Activity} label="نظارت" id="monitoring" />
          <MenuItem icon={BarChart3} label="آنالیز" id="analytics" />
          <MenuItem icon={FileText} label="گزارش‌ها" id="reports" />
          <MenuItem icon={Terminal} label="لاگ‌ها" id="logs" notifications={true} />
          <MenuItem icon={Users} label="تیم" id="team" />
          <MenuItem icon={Settings} label="تنظیمات" id="settings" />
        </nav>

        <div className="absolute bottom-4 left-4 right-4">
          <div className="text-center mb-4">
            {!sidebarCollapsed && (
              <div className="bg-gradient-to-r from-green-100 to-blue-100 rounded-xl p-4">
                <div className="flex items-center gap-2 text-sm text-gray-700 mb-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  سیستم فعال و آماده
                </div>
                <div className="text-xs text-gray-500 space-y-1">
                  <p>⚡ CPU: {(realTimeData[realTimeData.length-1]?.cpu || 0).toFixed(0)}%</p>
                  <p>💾 حافظه: {(realTimeData[realTimeData.length-1]?.memory || 0).toFixed(0)}%</p>
                </div>
              </div>
            )}
          </div>
          <button 
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center justify-center py-2 text-gray-500 hover:text-gray-700 transition-all hover:bg-gray-100 rounded-lg"
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Enhanced Main Content */}
      <div className={`transition-all duration-300 ${sidebarCollapsed ? 'mr-20' : 'mr-72'}`}>
        {/* Super Enhanced Top Bar */}
        <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4 flex items-center justify-between sticky top-0 z-30">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="w-5 h-5 text-gray-400 absolute right-3 top-1/2 transform -translate-y-1/2" />
              <input 
                type="text" 
                placeholder="جستجو در مدل‌ها، داده‌ها، لاگ‌ها و گزارش‌ها..."
                className="w-96 pl-4 pr-10 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <button className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-xl text-gray-600 hover:bg-gray-200 transition-all">
              <Filter className="w-4 h-4" />
              فیلتر
            </button>
          </div>

          <div className="flex items-center gap-4">
            {/* Refresh Interval Controls */}
            <div className="flex items-center gap-2 bg-gray-100 rounded-xl p-1">
              <button 
                onClick={() => setRefreshInterval(1000)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${refreshInterval === 1000 ? 'bg-white shadow-sm' : ''}`}
              >
                1s
              </button>
              <button 
                onClick={() => setRefreshInterval(3000)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${refreshInterval === 3000 ? 'bg-white shadow-sm' : ''}`}
              >
                3s
              </button>
              <button 
                onClick={() => setRefreshInterval(5000)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${refreshInterval === 5000 ? 'bg-white shadow-sm' : ''}`}
              >
                5s
              </button>
            </div>

            {/* System Status Indicator */}
            <div className="flex items-center gap-2 px-3 py-2 bg-green-100 rounded-xl">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-green-700 text-sm font-medium">آنلاین</span>
            </div>
            
            {/* Notifications */}
            <div className="relative">
              <button 
                onClick={() => setShowNotifications(!showNotifications)}
                className="relative p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-xl transition-all"
              >
                <Bell className="w-5 h-5" />
                {notifications.filter(n => !n.read).length > 0 && (
                  <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">
                    <span className="text-white text-xs">{notifications.filter(n => !n.read).length}</span>
                  </span>
                )}
              </button>
              <NotificationPanel />
            </div>
            
            {/* User Profile */}
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center cursor-pointer hover:shadow-lg transition-all">
              <span className="text-white text-sm font-medium">ک</span>
            </div>
          </div>
        </div>

        {/* Page Content */}
        <div className="p-6">
          {activeTab === 'dashboard' && renderDashboard()}
          {activeTab === 'models' && <div className="text-center py-20 text-gray-500">صفحه مدل‌ها در حال توسعه...</div>}
          {activeTab === 'data' && <div className="text-center py-20 text-gray-500">صفحه داده‌ها در حال توسعه...</div>}
          {activeTab === 'monitoring' && <div className="text-center py-20 text-gray-500">صفحه نظارت در حال توسعه...</div>}
          {activeTab === 'analytics' && <div className="text-center py-20 text-gray-500">صفحه آنالیز در حال توسعه...</div>}
          {activeTab === 'reports' && <div className="text-center py-20 text-gray-500">صفحه گزارش‌ها در حال توسعه...</div>}
          {activeTab === 'logs' && <div className="text-center py-20 text-gray-500">صفحه لاگ‌ها در حال توسعه...</div>}
          {activeTab === 'team' && <div className="text-center py-20 text-gray-500">صفحه تیم در حال توسعه...</div>}
          {activeTab === 'settings' && <div className="text-center py-20 text-gray-500">صفحه تنظیمات در حال توسعه...</div>}
        </div>
      </div>
    </div>
  );
};

export default CompletePersianAIDashboard;