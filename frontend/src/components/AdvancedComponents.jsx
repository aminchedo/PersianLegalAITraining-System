// components/AdvancedComponents.js
/**
 * Advanced Dashboard Components
 * کامپوننت‌های پیشرفته برای داشبورد
 */

import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { 
  AlertTriangle, CheckCircle, Clock, TrendingUp, TrendingDown, 
  Zap, Activity, Brain, Database, Download, Settings, 
  Play, Pause, Square, Eye, EyeOff, Maximize2, X, 
  Bell, Info, AlertCircle 
} from 'lucide-react';

// کامپوننت مدیریت پروژه‌ها
export const ProjectManager = ({ projects, selectedProject, onProjectChange, onCreateProject }) => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');

  const handleCreateProject = () => {
    if (newProjectName.trim()) {
      onCreateProject({
        id: Date.now().toString(),
        name: newProjectName,
        status: 'active',
        progress: 0,
        created: new Date()
      });
      setNewProjectName('');
      setShowCreateModal(false);
    }
  };

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">مدیریت پروژه‌ها</h3>
        <button 
          onClick={() => setShowCreateModal(true)}
          className="bg-blue-600 text-white px-3 py-1 rounded-lg text-sm hover:bg-blue-700 transition-all"
        >
          + پروژه جدید
        </button>
      </div>

      <div className="space-y-2">
        {projects.map(project => (
          <div 
            key={project.id}
            onClick={() => onProjectChange(project.id)}
            className={`p-3 rounded-lg cursor-pointer transition-all ${
              selectedProject === project.id 
                ? 'bg-blue-100 border-2 border-blue-500' 
                : 'bg-gray-50 hover:bg-gray-100'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">{project.name}</p>
                <p className="text-sm text-gray-500">پیشرفت: {project.progress}%</p>
              </div>
              <div className={`w-3 h-3 rounded-full ${
                project.status === 'active' ? 'bg-green-500' :
                project.status === 'completed' ? 'bg-blue-500' : 'bg-yellow-500'
              }`}></div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1 mt-2">
              <div 
                className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                style={{ width: `${project.progress}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Modal ایجاد پروژه */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
          <div className="bg-white rounded-2xl p-6 w-96">
            <h3 className="text-lg font-semibold mb-4">ایجاد پروژه جدید</h3>
            <input 
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="نام پروژه..."
              className="w-full p-3 border border-gray-300 rounded-lg mb-4"
              onKeyPress={(e) => e.key === 'Enter' && handleCreateProject()}
            />
            <div className="flex gap-2">
              <button 
                onClick={handleCreateProject}
                className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-all"
              >
                ایجاد
              </button>
              <button 
                onClick={() => setShowCreateModal(false)}
                className="flex-1 bg-gray-300 text-gray-700 py-2 rounded-lg hover:bg-gray-400 transition-all"
              >
                لغو
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// کامپوننت پیش‌بینی هوشمند
export const AIInsights = ({ metrics, trends, predictions }) => {
  const [activeInsight, setActiveInsight] = useState(0);

  const insights = [
    {
      type: 'prediction',
      icon: TrendingUp,
      title: 'پیش‌بینی عملکرد',
      message: `دقت مدل تا ${Math.ceil(Math.random() * 5)} روز آینده به ${(92 + Math.random() * 5).toFixed(1)}% خواهد رسید`,
      confidence: 85 + Math.random() * 10,
      color: 'blue'
    },
    {
      type: 'warning',
      icon: AlertTriangle,
      title: 'هشدار منابع',
      message: `استفاده از ${Math.random() > 0.5 ? 'CPU' : 'حافظه'} در ${Math.ceil(Math.random() * 6)} ساعت آینده افزایش می‌یابد`,
      confidence: 75 + Math.random() * 15,
      color: 'yellow'
    },
    {
      type: 'optimization',
      icon: Zap,
      title: 'توصیه بهینه‌سازی',
      message: `کاهش batch size به ${Math.floor(Math.random() * 4) + 2} برای بهبود ${Math.floor(Math.random() * 20) + 10}% کارایی`,
      confidence: 80 + Math.random() * 15,
      color: 'green'
    }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveInsight(prev => (prev + 1) % insights.length);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl p-6 text-white">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-5 h-5" />
        <h3 className="text-lg font-semibold">تحلیل هوشمند سیستم</h3>
      </div>

      <div className="space-y-4">
        {insights.map((insight, index) => {
          const Icon = insight.icon;
          const isActive = index === activeInsight;
          
          return (
            <div 
              key={index}
              className={`bg-white/20 rounded-xl p-4 backdrop-blur-sm transition-all duration-500 ${
                isActive ? 'transform scale-105 ring-2 ring-white/50' : 'opacity-70'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                  insight.color === 'blue' ? 'bg-blue-500/30' :
                  insight.color === 'yellow' ? 'bg-yellow-500/30' : 'bg-green-500/30'
                }`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <h4 className="font-medium mb-1">{insight.title}</h4>
                  <p className="text-sm opacity-90 leading-relaxed">{insight.message}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <div className="text-xs opacity-75">اطمینان:</div>
                    <div className="flex-1 bg-white/20 rounded-full h-1">
                      <div 
                        className="bg-white h-1 rounded-full transition-all duration-1000"
                        style={{ width: `${insight.confidence}%` }}
                      ></div>
                    </div>
                    <div className="text-xs">{insight.confidence.toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* نقاط نشانگر */}
      <div className="flex justify-center gap-2 mt-4">
        {insights.map((_, index) => (
          <button
            key={index}
            onClick={() => setActiveInsight(index)}
            className={`w-2 h-2 rounded-full transition-all ${
              index === activeInsight ? 'bg-white' : 'bg-white/30'
            }`}
          />
        ))}
      </div>
    </div>
  );
};

// کامپوننت لاگ‌های سیستم
export const SystemLogs = ({ logs, autoRefresh = true }) => {
  const [filteredLogs, setFilteredLogs] = useState(logs);
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    let filtered = logs;

    if (filter !== 'all') {
      filtered = filtered.filter(log => log.level.toLowerCase() === filter);
    }

    if (searchTerm) {
      filtered = filtered.filter(log => 
        log.message.includes(searchTerm) || 
        log.component.includes(searchTerm)
      );
    }

    setFilteredLogs(filtered);
  }, [logs, filter, searchTerm]);

  const getLevelColor = (level) => {
    switch (level.toLowerCase()) {
      case 'error': return 'text-red-600 bg-red-50';
      case 'warning': return 'text-yellow-600 bg-yellow-50';
      case 'info': return 'text-blue-600 bg-blue-50';
      case 'debug': return 'text-gray-600 bg-gray-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getLevelIcon = (level) => {
    switch (level.toLowerCase()) {
      case 'error': return AlertCircle;
      case 'warning': return AlertTriangle;
      case 'info': return Info;
      default: return CheckCircle;
    }
  };

  return (
    <div className="bg-white rounded-2xl p-6 shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">لاگ‌های سیستم</h3>
        <div className="flex items-center gap-2">
          {autoRefresh && (
            <div className="flex items-center gap-1 text-sm text-gray-500">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              زنده
            </div>
          )}
          <button className="p-2 hover:bg-gray-100 rounded-lg">
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* فیلترها */}
      <div className="flex items-center gap-4 mb-4">
        <select 
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
        >
          <option value="all">همه</option>
          <option value="error">خطا</option>
          <option value="warning">هشدار</option>
          <option value="info">اطلاعات</option>
          <option value="debug">دیباگ</option>
        </select>
        
        <input 
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="جستجو در لاگ‌ها..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
        />
      </div>

      {/* لاگ‌ها */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {filteredLogs.map(log => {
          const Icon = getLevelIcon(log.level);
          return (
            <div 
              key={log.id}
              className="flex items-start gap-3 p-3 hover:bg-gray-50 rounded-lg transition-all"
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${getLevelColor(log.level)}`}>
                <Icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getLevelColor(log.level)}`}>
                    {log.level}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(log.timestamp).toLocaleString('fa-IR')}
                  </span>
                  <span className="text-xs text-gray-400">{log.component}</span>
                </div>
                <p className="text-sm text-gray-900 leading-relaxed">{log.message}</p>
              </div>
            </div>
          );
        })}
        
        {filteredLogs.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <Info className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>هیچ لاگی یافت نشد</p>
          </div>
        )}
      </div>
    </div>
  );
};

// کامپوننت کنترل پیشرفته مدل
export const ModelController = ({ model, onAction }) => {
  const [showConfig, setShowConfig] = useState(false);
  const [config, setConfig] = useState({
    learning_rate: 1e-4,
    batch_size: 4,
    dora_rank: 64,
    dora_alpha: 16.0
  });

  const getStatusColor = (status) => {
    switch (status) {
      case 'training': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'training': return 'در حال آموزش';
      case 'completed': return 'تکمیل شده';
      case 'error': return 'خطا';
      default: return 'در انتظار';
    }
  };

  return (
    <div className="bg-white rounded-2xl p-6 shadow-lg border hover:shadow-xl transition-all">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${getStatusColor(model.status)} bg-opacity-10`}>
            <Brain className={`w-6 h-6 ${getStatusColor(model.status).replace('bg-', 'text-')}`} />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">{model.name}</h3>
            <p className="text-sm text-gray-500">{getStatusText(model.status)}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={() => setShowConfig(!showConfig)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-all"
            title="تنظیمات"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* اطلاعات مدل */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-sm text-gray-500">دقت</p>
          <p className="text-lg font-semibold">{model.accuracy?.toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">خطا</p>
          <p className="text-lg font-semibold">{model.loss?.toFixed(3)}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">دوره‌ها</p>
          <p className="text-lg font-semibold">{model.epochs_completed || 0}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">زمان باقی‌مانده</p>
          <p className="text-lg font-semibold">{model.time_remaining || '--'}</p>
        </div>
      </div>

      {/* نوار پیشرفت */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-2">
          <span>پیشرفت</span>
          <span>{model.progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`${getStatusColor(model.status)} h-2 rounded-full transition-all duration-500`}
            style={{ width: `${model.progress}%` }}
          ></div>
        </div>
      </div>

      {/* دکمه‌های کنترل */}
      <div className="flex gap-2">
        {model.status === 'training' ? (
          <button 
            onClick={() => onAction('pause', model.id)}
            className="flex-1 bg-yellow-100 text-yellow-700 py-2 px-3 rounded-lg text-sm hover:bg-yellow-200 transition-all flex items-center justify-center gap-2"
          >
            <Pause className="w-4 h-4" />
            توقف
          </button>
        ) : (
          <button 
            onClick={() => onAction('start', model.id, config)}
            className="flex-1 bg-green-100 text-green-700 py-2 px-3 rounded-lg text-sm hover:bg-green-200 transition-all flex items-center justify-center gap-2"
          >
            <Play className="w-4 h-4" />
            شروع
          </button>
        )}
        
        <button 
          onClick={() => onAction('view', model.id)}
          className="flex-1 bg-blue-100 text-blue-700 py-2 px-3 rounded-lg text-sm hover:bg-blue-200 transition-all flex items-center justify-center gap-2"
        >
          <Eye className="w-4 h-4" />
          مشاهده
        </button>
      </div>

      {/* پنل تنظیمات */}
      {showConfig && (
        <div className="mt-4 p-4 bg-gray-50 rounded-xl">
          <h4 className="font-medium mb-3">تنظیمات آموزش</h4>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-600 mb-1">نرخ یادگیری</label>
              <input 
                type="number"
                step="0.0001"
                value={config.learning_rate}
                onChange={(e) => setConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600 mb-1">اندازه دسته</label>
              <select 
                value={config.batch_size}
                onChange={(e) => setConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              >
                <option value={2}>2</option>
                <option value={4}>4</option>
                <option value={8}>8</option>
                <option value={16}>16</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-600 mb-1">رتبه DoRA</label>
              <input 
                type="number"
                value={config.dora_rank}
                onChange={(e) => setConfig(prev => ({ ...prev, dora_rank: parseInt(e.target.value) }))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600 mb-1">آلفای DoRA</label>
              <input 
                type="number"
                step="0.1"
                value={config.dora_alpha}
                onChange={(e) => setConfig(prev => ({ ...prev, dora_alpha: parseFloat(e.target.value) }))}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default {
  ProjectManager,
  AIInsights,
  SystemLogs,
  ModelController
};