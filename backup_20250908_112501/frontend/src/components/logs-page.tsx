import React, { useState, useEffect } from 'react';
import { 
  Terminal, Search, Filter, Download, RefreshCw, Trash2, 
  AlertTriangle, CheckCircle, Info, AlertCircle, Clock,
  Eye, EyeOff, Settings, Play, Pause, Square, Copy,
  Calendar, FileText, Server, Database, Activity, Zap
} from 'lucide-react';
import { useAppContext } from './Router';

const LogsPage = () => {
  const { systemLogs, setSystemLogs, autoRefresh } = useAppContext();
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [levelFilter, setLevelFilter] = useState('all');
  const [componentFilter, setComponentFilter] = useState('all');
  const [timeFilter, setTimeFilter] = useState('all');
  const [selectedLogs, setSelectedLogs] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [realTimeLogs, setRealTimeLogs] = useState(true);
  const [maxLogs, setMaxLogs] = useState(1000);

  // Enhanced log data with more realistic entries
  const [allLogs, setAllLogs] = useState([
    { id: 1, level: 'INFO', message: 'سیستم با موفقیت راه‌اندازی شد', timestamp: new Date(Date.now() - 1000), component: 'main', details: 'System startup completed successfully', category: 'system' },
    { id: 2, level: 'SUCCESS', message: 'اتصال به پیکره نعب برقرار شد', timestamp: new Date(Date.now() - 2000), component: 'data-collector', details: 'Connected to Naab corpus API endpoint', category: 'data' },
    { id: 3, level: 'WARNING', message: 'استفاده از حافظه به 78% رسید', timestamp: new Date(Date.now() - 3000), component: 'monitor', details: 'Memory usage: 78.3% (50.1GB/64GB)', category: 'performance' },
    { id: 4, level: 'INFO', message: 'شروع آموزش مدل PersianMind-v1.0', timestamp: new Date(Date.now() - 4000), component: 'trainer', details: 'Model training initiated with config: batch_size=4, lr=1e-4', category: 'training' },
    { id: 5, level: 'ERROR', message: 'خطا در بارگیری مدل Legal-NER-v2', timestamp: new Date(Date.now() - 5000), component: 'model-loader', details: 'FileNotFoundError: model_weights.bin not found', category: 'model' },
    { id: 6, level: 'DEBUG', message: 'تنظیمات GPU بررسی شد', timestamp: new Date(Date.now() - 6000), component: 'gpu-manager', details: 'CUDA 12.2 detected, 16GB VRAM available', category: 'hardware' },
    { id: 7, level: 'INFO', message: 'بک‌آپ اتوماتیک انجام شد', timestamp: new Date(Date.now() - 7000), component: 'backup-service', details: 'Daily backup completed: 2.3GB saved to /backup/2025-01-15', category: 'system' },
    { id: 8, level: 'WARNING', message: 'کیفیت داده منبع "پورتال قوانین" کاهش یافت', timestamp: new Date(Date.now() - 8000), component: 'data-validator', details: 'Quality score dropped to 85% from 92%', category: 'data' },
    { id: 9, level: 'SUCCESS', message: 'مدل ParsBERT-Legal با موفقیت آموزش داده شد', timestamp: new Date(Date.now() - 9000), component: 'trainer', details: 'Training completed: 92.5% accuracy achieved', category: 'training' },
    { id: 10, level: 'ERROR', message: 'اتصال به API شکست خورد', timestamp: new Date(Date.now() - 10000), component: 'api-client', details: 'Connection timeout after 30s to legal-portal-api.ir', category: 'network' }
  ]);

  // Components list for filtering
  const components = [...new Set(allLogs.map(log => log.component))];
  const categories = [...new Set(allLogs.map(log => log.category))];

  // Real-time log generation
  useEffect(() => {
    if (!realTimeLogs || !autoRefresh) return;

    const interval = setInterval(() => {
      const newLog = {
        id: Date.now(),
        level: ['INFO', 'WARNING', 'ERROR', 'SUCCESS', 'DEBUG'][Math.floor(Math.random() * 5)],
        message: [
          'بررسی سلامت سیستم انجام شد',
          'آپدیت مدل جدید اعمال شد',
          'تنظیمات امنیتی بروزرسانی شد',
          'پردازش دسته جدید داده‌ها',
          'بهینه‌سازی عملکرد GPU',
          'همگام‌سازی با منابع خارجی',
          'اعتبارسنجی داده‌های ورودی',
          'ذخیره وضعیت مدل فعلی'
        ][Math.floor(Math.random() * 8)],
        timestamp: new Date(),
        component: components[Math.floor(Math.random() * components.length)],
        details: 'Auto-generated log entry for monitoring purposes',
        category: categories[Math.floor(Math.random() * categories.length)]
      };
      
      setAllLogs(prev => [newLog, ...prev.slice(0, maxLogs - 1)]);
    }, 3000 + Math.random() * 7000); // Random interval between 3-10 seconds

    return () => clearInterval(interval);
  }, [realTimeLogs, autoRefresh, maxLogs, components, categories]);

  // Filter logs
  useEffect(() => {
    let filtered = allLogs;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(log => 
        log.message.includes(searchTerm) || 
        log.component.includes(searchTerm) ||
        log.details.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Level filter
    if (levelFilter !== 'all') {
      filtered = filtered.filter(log => log.level.toLowerCase() === levelFilter);
    }

    // Component filter
    if (componentFilter !== 'all') {
      filtered = filtered.filter(log => log.component === componentFilter);
    }

    // Time filter
    if (timeFilter !== 'all') {
      const now = new Date();
      const timeRanges = {
        '1h': 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '24h': 24 * 60 * 60 * 1000,
        '7d': 7 * 24 * 60 * 60 * 1000
      };
      
      if (timeRanges[timeFilter]) {
        filtered = filtered.filter(log => 
          now - new Date(log.timestamp) <= timeRanges[timeFilter]
        );
      }
    }

    setFilteredLogs(filtered);
  }, [allLogs, searchTerm, levelFilter, componentFilter, timeFilter]);

  // Get log level configuration
  const getLevelConfig = (level) => {
    const configs = {
      ERROR: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertTriangle, border: 'border-red-200' },
      WARNING: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: AlertTriangle, border: 'border-yellow-200' },
      INFO: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Info, border: 'border-blue-200' },
      SUCCESS: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle, border: 'border-green-200' },
      DEBUG: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: Settings, border: 'border-gray-200' }
    };
    return configs[level] || configs.INFO;
  };

  // Get component icon
  const getComponentIcon = (component) => {
    const icons = {
      'main': Server,
      'trainer': Activity,
      'data-collector': Database,
      'model-loader': Brain,
      'api-client': Globe,
      'monitor': Eye,
      'backup-service': FileText,
      'gpu-manager': Zap,
      'data-validator': CheckCircle
    };
    return icons[component] || Terminal;
  };

  // Handle log selection
  const handleLogSelection = (logId) => {
    setSelectedLogs(prev => 
      prev.includes(logId) 
        ? prev.filter(id => id !== logId)
        : [...prev, logId]
    );
  };

  // Handle select all
  const handleSelectAll = () => {
    setSelectedLogs(
      selectedLogs.length === filteredLogs.length 
        ? [] 
        : filteredLogs.map(log => log.id)
    );
  };

  // Export logs
  const exportLogs = (format = 'json') => {
    const logsToExport = selectedLogs.length > 0 
      ? allLogs.filter(log => selectedLogs.includes(log.id))
      : filteredLogs;

    if (format === 'json') {
      const dataStr = JSON.stringify(logsToExport, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `logs_${new Date().toISOString().split('T')[0]}.json`;
      link.click();
    } else if (format === 'csv') {
      const headers = ['timestamp', 'level', 'component', 'message', 'details'];
      const csvContent = [
        headers.join(','),
        ...logsToExport.map(log => [
          log.timestamp.toISOString(),
          log.level,
          log.component,
          `"${log.message}"`,
          `"${log.details}"`
        ].join(','))
      ].join('\n');
      
      const dataBlob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `logs_${new Date().toISOString().split('T')[0]}.csv`;
      link.click();
    }
  };

  // Clear logs
  const clearLogs = () => {
    if (selectedLogs.length > 0) {
      setAllLogs(prev => prev.filter(log => !selectedLogs.includes(log.id)));
      setSelectedLogs([]);
    } else {
      setAllLogs([]);
    }
  };

  // Log statistics
  const logStats = {
    total: filteredLogs.length,
    error: filteredLogs.filter(log => log.level === 'ERROR').length,
    warning: filteredLogs.filter(log => log.level === 'WARNING').length,
    info: filteredLogs.filter(log => log.level === 'INFO').length,
    success: filteredLogs.filter(log => log.level === 'SUCCESS').length,
    debug: filteredLogs.filter(log => log.level === 'DEBUG').length
  };

  return (
    <div className="space-y-6" dir="rtl">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-800 via-gray-900 to-black rounded-2xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <Terminal className="w-8 h-8" />
              لاگ‌های سیستم
            </h1>
            <p className="text-gray-300">مانیتورینگ، تحلیل و مدیریت لاگ‌های سیستم</p>
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setRealTimeLogs(!realTimeLogs)}
              className={`px-4 py-2 rounded-xl font-medium transition-all flex items-center gap-2 ${
                realTimeLogs 
                  ? 'bg-green-600 text-white hover:bg-green-700' 
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}
            >
              {realTimeLogs ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {realTimeLogs ? 'توقف زنده' : 'شروع زنده'}
            </button>
            <button 
              onClick={() => exportLogs('json')}
              className="bg-white text-gray-900 px-4 py-2 rounded-xl hover:bg-gray-100 transition-all flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              دانلود
            </button>
          </div>
        </div>
        
        {/* Log Statistics */}
        <div className="grid grid-cols-6 gap-4 mt-6">
          <div className="bg-white/10 rounded-xl p-3 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-1">
              <Terminal className="w-4 h-4" />
              <span className="text-sm">کل</span>
            </div>
            <p className="text-xl font-bold">{logStats.total}</p>
          </div>
          <div className="bg-red-500/20 rounded-xl p-3 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-sm">خطا</span>
            </div>
            <p className="text-xl font-bold">{logStats.error}</p>
          </div>
          <div className="bg-yellow-500/20 rounded-xl p-3 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle className="w-4 h-4" />
              <span className="text-sm">هشدار</span>
            </div>
            <p className="text-xl font-bold">{logStats.warning}</p>
          </div>
          <div className="bg-blue-500/20 rounded-xl p-3 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-1">
              <Info className="w-4 h-4" />
              <span className="text-sm">اطلاعات</span>
            </div>
            <p className="text-xl font-bold">{logStats.info}</p>
          </div>
          <div className="bg-green-500/20 rounded-xl p-3 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-1">
              <CheckCircle className="w-4 h-4" />
              <span className="text-sm">موفق</span>
            </div>
            <p className="text-xl font-bold">{logStats.success}</p>
          </div>
          <div className="bg-gray-500/20 rounded-xl p-3 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-1">
              <Settings className="w-4 h-4" />
              <span className="text-sm">دیباگ</span>
            </div>
            <p className="text-xl font-bold">{logStats.debug}</p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="w-5 h-5 text-gray-400 absolute right-3 top-1/2 transform -translate-y-1/2" />
              <input 
                type="text"
                placeholder="جستجو در لاگ‌ها..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-64 pl-4 pr-10 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <button 
              onClick={() => setShowFilters(!showFilters)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
                showFilters ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Filter className="w-4 h-4" />
              فیلترها
            </button>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">
              {selectedLogs.length > 0 && `${selectedLogs.length} انتخاب شده`}
            </span>
            {selectedLogs.length > 0 && (
              <>
                <button 
                  onClick={() => exportLogs('csv')}
                  className="text-blue-600 hover:text-blue-700 text-sm"
                >
                  دانلود CSV
                </button>
                <button 
                  onClick={clearLogs}
                  className="text-red-600 hover:text-red-700 text-sm flex items-center gap-1"
                >
                  <Trash2 className="w-4 h-4" />
                  حذف
                </button>
              </>
            )}
            <button 
              onClick={() => setAllLogs([])}
              className="text-gray-600 hover:text-gray-700 text-sm"
            >
              پاک کردن همه
            </button>
          </div>
        </div>

        {/* Advanced Filters */}
        {showFilters && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-xl">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">سطح لاگ</label>
              <select 
                value={levelFilter}
                onChange={(e) => setLevelFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">همه سطوح</option>
                <option value="error">خطا</option>
                <option value="warning">هشدار</option>
                <option value="info">اطلاعات</option>
                <option value="success">موفقیت</option>
                <option value="debug">دیباگ</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">کامپوننت</label>
              <select 
                value={componentFilter}
                onChange={(e) => setComponentFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">همه کامپوننت‌ها</option>
                {components.map(component => (
                  <option key={component} value={component}>{component}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">بازه زمانی</label>
              <select 
                value={timeFilter}
                onChange={(e) => setTimeFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">همه زمان‌ها</option>
                <option value="1h">ساعت گذشته</option>
                <option value="6h">6 ساعت گذشته</option>
                <option value="24h">24 ساعت گذشته</option>
                <option value="7d">هفته گذشته</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">حداکثر تعداد</label>
              <select 
                value={maxLogs}
                onChange={(e) => setMaxLogs(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value={100}>100</option>
                <option value={500}>500</option>
                <option value={1000}>1000</option>
                <option value={5000}>5000</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Logs List */}
      <div className="bg-white rounded-2xl shadow-lg">
        {/* Table Header */}
        <div className="border-b border-gray-200 p-4">
          <div className="flex items-center gap-4">
            <input 
              type="checkbox"
              checked={selectedLogs.length === filteredLogs.length && filteredLogs.length > 0}
              onChange={handleSelectAll}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
            <div className="grid grid-cols-12 gap-4 w-full text-sm font-medium text-gray-700">
              <div className="col-span-2">زمان</div>
              <div className="col-span-1">سطح</div>
              <div className="col-span-2">کامپوننت</div>
              <div className="col-span-5">پیام</div>
              <div className="col-span-2">عملیات</div>
            </div>
          </div>
        </div>

        {/* Table Body */}
        <div className="max-h-96 overflow-y-auto">
          {filteredLogs.map(log => {
            const levelConfig = getLevelConfig(log.level);
            const LevelIcon = levelConfig.icon;
            const ComponentIcon = getComponentIcon(log.component);
            const isSelected = selectedLogs.includes(log.id);
            
            return (
              <div 
                key={log.id}
                className={`border-b border-gray-100 p-4 hover:bg-gray-50 transition-all ${
                  isSelected ? 'bg-blue-50 border-blue-200' : ''
                } ${levelConfig.border} border-r-4`}
              >
                <div className="flex items-start gap-4">
                  <input 
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => handleLogSelection(log.id)}
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 mt-1"
                  />
                  
                  <div className="grid grid-cols-12 gap-4 w-full">
                    {/* Timestamp */}
                    <div className="col-span-2">
                      <div className="text-sm text-gray-900 font-medium">
                        {new Date(log.timestamp).toLocaleTimeString('fa-IR')}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(log.timestamp).toLocaleDateString('fa-IR')}
                      </div>
                    </div>

                    {/* Level */}
                    <div className="col-span-1">
                      <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${levelConfig.bgColor} ${levelConfig.color}`}>
                        <LevelIcon className="w-3 h-3" />
                        {log.level}
                      </div>
                    </div>

                    {/* Component */}
                    <div className="col-span-2">
                      <div className="flex items-center gap-2">
                        <ComponentIcon className="w-4 h-4 text-gray-500" />
                        <span className="text-sm text-gray-900">{log.component}</span>
                      </div>
                      <div className="text-xs text-gray-500">{log.category}</div>
                    </div>

                    {/* Message */}
                    <div className="col-span-5">
                      <div className="text-sm text-gray-900 leading-relaxed">
                        {log.message}
                      </div>
                      {log.details && (
                        <div className="text-xs text-gray-500 mt-1 font-mono bg-gray-100 p-1 rounded">
                          {log.details}
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="col-span-2">
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => navigator.clipboard.writeText(JSON.stringify(log, null, 2))}
                          className="text-gray-400 hover:text-gray-600 transition-all"
                          title="کپی"
                        >
                          <Copy className="w-4 h-4" />
                        </button>
                        <button 
                          className="text-gray-400 hover:text-gray-600 transition-all"
                          title="مشاهده جزئیات"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
          
          {filteredLogs.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>هیچ لاگی یافت نشد</p>
              <p className="text-sm">فیلترها را تغییر دهید یا منتظر لاگ‌های جدید باشید</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-4">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div>
              نمایش {filteredLogs.length} لاگ از {allLogs.length} لاگ کل
            </div>
            <div className="flex items-center gap-4">
              {realTimeLogs && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span>به‌روزرسانی زنده</span>
                </div>
              )}
              <span>آخرین بروزرسانی: {new Date().toLocaleTimeString('fa-IR')}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Log Analysis Panel */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Recent Errors */}
        <div className="bg-white rounded-2xl p-6 shadow-lg border-t-4 border-red-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            خطاهای اخیر
          </h3>
          <div className="space-y-3">
            {allLogs.filter(log => log.level === 'ERROR').slice(0, 5).map(log => (
              <div key={log.id} className="p-3 bg-red-50 rounded-lg border border-red-200">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-red-700">{log.component}</span>
                  <span className="text-xs text-red-500">
                    {new Date(log.timestamp).toLocaleTimeString('fa-IR')}
                  </span>
                </div>
                <p className="text-sm text-red-800">{log.message}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Component Activity */}
        <div className="bg-white rounded-2xl p-6 shadow-lg border-t-4 border-blue-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-500" />
            فعالیت کامپوننت‌ها
          </h3>
          <div className="space-y-3">
            {Object.entries(
              allLogs.reduce((acc, log) => {
                acc[log.component] = (acc[log.component] || 0) + 1;
                return acc;
              }, {})
            ).slice(0, 5).map(([component, count]) => (
              <div key={component} className="flex items-center justify-between">
                <span className="text-sm text-gray-700">{component}</span>
                <span className="text-sm font-semibold text-blue-600">{count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* System Health */}
        <div className="bg-white rounded-2xl p-6 shadow-lg border-t-4 border-green-500">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-500" />
            وضعیت سیستم
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">آخرین خطا</span>
              <span className="text-sm text-gray-500">
                {allLogs.find(log => log.level === 'ERROR') 
                  ? new Date(allLogs.find(log => log.level === 'ERROR').timestamp).toLocaleTimeString('fa-IR')
                  : 'ندارد'
                }
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">لاگ‌های ساعت گذشته</span>
              <span className="text-sm font-semibold text-green-600">
                {allLogs.filter(log => 
                  new Date() - new Date(log.timestamp) <= 60 * 60 * 1000
                ).length}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">نرخ خطا</span>
              <span className="text-sm font-semibold text-green-600">
                {((logStats.error / logStats.total) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LogsPage;