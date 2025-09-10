import React, { useState, useEffect } from 'react';
import { Search, Filter, Download, RefreshCw, Calendar, Eye, AlertCircle, Info, CheckCircle, XCircle, Clock } from 'lucide-react';
import { usePersianAI } from '../../../hooks/usePersianAI';

const LogsPage: React.FC = () => {
  const { state } = usePersianAI();
  const [logs, setLogs] = useState<any[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterLevel, setFilterLevel] = useState('all');
  const [filterSource, setFilterSource] = useState('all');
  const [dateRange, setDateRange] = useState('today');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Generate mock logs
  useEffect(() => {
    const generateLogs = () => {
      const logLevels = ['info', 'warning', 'error', 'success'];
      const sources = ['system', 'training', 'api', 'database'];
      const messages = [
        'Training session started for model Persian-Legal-LLM',
        'Database connection established successfully',
        'API endpoint /models responded with 200 status',
        'Memory usage increased to 75%',
        'Model evaluation completed with 92% accuracy',
        'User authentication failed for admin@example.com',
        'GPU temperature reached 78°C',
        'Backup completed successfully',
        'Training checkpoint saved at epoch 15',
        'System performance optimized',
        'New data source added: Legal Documents DB',
        'Cache cleared successfully',
      ];

      const newLogs = [];
      for (let i = 0; i < 50; i++) {
        const timestamp = new Date(Date.now() - Math.random() * 86400000 * 7); // Random time within last week
        newLogs.push({
          id: i + 1,
          timestamp: timestamp.toISOString(),
          level: logLevels[Math.floor(Math.random() * logLevels.length)],
          source: sources[Math.floor(Math.random() * sources.length)],
          message: messages[Math.floor(Math.random() * messages.length)],
          details: `Additional details for log entry ${i + 1}`,
          user: i % 3 === 0 ? 'system' : `user${Math.floor(Math.random() * 5) + 1}`,
        });
      }
      setLogs(newLogs.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()));
    };

    generateLogs();
  }, []);

  // Add new logs periodically when auto-refresh is enabled
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      const newLog = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        level: ['info', 'warning'][Math.floor(Math.random() * 2)],
        source: ['system', 'training', 'api'][Math.floor(Math.random() * 3)],
        message: [
          'System heartbeat - all services running normally',
          'Training progress update: 67% complete',
          'API request processed successfully',
          'Memory garbage collection completed',
          'Model checkpoint saved',
        ][Math.floor(Math.random() * 5)],
        details: `Live log entry generated at ${new Date().toLocaleString('fa-IR')}`,
        user: 'system',
      };

      setLogs(prev => [newLog, ...prev.slice(0, 99)]);
    }, 10000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Filter logs based on search and filters
  useEffect(() => {
    let filtered = logs;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(log =>
        log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.source.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Level filter
    if (filterLevel !== 'all') {
      filtered = filtered.filter(log => log.level === filterLevel);
    }

    // Source filter
    if (filterSource !== 'all') {
      filtered = filtered.filter(log => log.source === filterSource);
    }

    // Date range filter
    const now = new Date();
    if (dateRange === 'today') {
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      filtered = filtered.filter(log => new Date(log.timestamp) >= today);
    } else if (dateRange === 'week') {
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      filtered = filtered.filter(log => new Date(log.timestamp) >= weekAgo);
    }

    setFilteredLogs(filtered);
  }, [logs, searchTerm, filterLevel, filterSource, dateRange]);

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'error': return <XCircle className="w-5 h-5 text-red-600" />;
      case 'warning': return <AlertCircle className="w-5 h-5 text-yellow-600" />;
      case 'success': return <CheckCircle className="w-5 h-5 text-green-600" />;
      default: return <Info className="w-5 h-5 text-blue-600" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error': return 'bg-red-50 text-red-700 border-red-200';
      case 'warning': return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'success': return 'bg-green-50 text-green-700 border-green-200';
      default: return 'bg-blue-50 text-blue-700 border-blue-200';
    }
  };

  const exportLogs = () => {
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Timestamp,Level,Source,Message,User\n"
      + filteredLogs.map(log => 
          `"${log.timestamp}","${log.level}","${log.source}","${log.message}","${log.user}"`
        ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `system_logs_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-600 to-slate-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">لاگ‌های سیستم</h1>
            <p className="text-gray-100">مشاهده و تحلیل لاگ‌های سیستم، آموزش و API</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="flex items-center space-x-2 space-x-reverse">
                <div className={`w-3 h-3 rounded-full ${autoRefresh ? 'bg-green-400' : 'bg-gray-400'} ${autoRefresh ? 'animate-pulse' : ''}`}></div>
                <span className="text-sm font-medium">
                  {autoRefresh ? 'به‌روزرسانی خودکار' : 'متوقف شده'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-lg ml-4">
              <Info className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">اطلاعات</p>
              <p className="text-2xl font-bold text-gray-900">
                {logs.filter(log => log.level === 'info').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-yellow-100 p-3 rounded-lg ml-4">
              <AlertCircle className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">هشدارها</p>
              <p className="text-2xl font-bold text-gray-900">
                {logs.filter(log => log.level === 'warning').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-red-100 p-3 rounded-lg ml-4">
              <XCircle className="w-6 h-6 text-red-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">خطاها</p>
              <p className="text-2xl font-bold text-gray-900">
                {logs.filter(log => log.level === 'error').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-green-100 p-3 rounded-lg ml-4">
              <CheckCircle className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">موفق</p>
              <p className="text-2xl font-bold text-gray-900">
                {logs.filter(log => log.level === 'success').length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
          <div className="lg:col-span-2">
            <div className="relative">
              <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="جستجو در لاگ‌ها..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-4 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div>
            <select 
              value={filterLevel}
              onChange={(e) => setFilterLevel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">همه سطوح</option>
              <option value="info">اطلاعات</option>
              <option value="warning">هشدار</option>
              <option value="error">خطا</option>
              <option value="success">موفق</option>
            </select>
          </div>

          <div>
            <select 
              value={filterSource}
              onChange={(e) => setFilterSource(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">همه منابع</option>
              <option value="system">سیستم</option>
              <option value="training">آموزش</option>
              <option value="api">API</option>
              <option value="database">دیتابیس</option>
            </select>
          </div>

          <div>
            <select 
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="today">امروز</option>
              <option value="week">هفته گذشته</option>
              <option value="all">همه</option>
            </select>
          </div>

          <div className="flex items-center space-x-2 space-x-reverse">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`flex-1 px-3 py-2 rounded-lg border transition-colors ${
                autoRefresh 
                  ? 'bg-green-50 border-green-300 text-green-700' 
                  : 'bg-gray-50 border-gray-300 text-gray-700'
              }`}
            >
              <RefreshCw className={`w-4 h-4 inline ml-1 ${autoRefresh ? 'animate-spin' : ''}`} />
              خودکار
            </button>
            
            <button
              onClick={exportLogs}
              className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
          <p className="text-sm text-gray-600">
            نمایش {filteredLogs.length} لاگ از {logs.length} مورد
          </p>
          <div className="flex items-center space-x-2 space-x-reverse">
            <span className="text-sm text-gray-600">آخرین به‌روزرسانی:</span>
            <span className="text-sm font-medium text-gray-900">
              {new Date().toLocaleTimeString('fa-IR')}
            </span>
          </div>
        </div>
      </div>

      {/* Logs List */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-900">لاگ‌های سیستم</h2>
        </div>
        
        <div className="divide-y divide-gray-200">
          {filteredLogs.length > 0 ? (
            filteredLogs.map((log) => (
              <div key={log.id} className="p-6 hover:bg-gray-50 transition-colors">
                <div className="flex items-start space-x-4 space-x-reverse">
                  <div className="flex-shrink-0 mt-1">
                    {getLevelIcon(log.level)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3 space-x-reverse">
                        <span className={`px-2 py-1 text-xs rounded-full font-medium ${getLevelColor(log.level)}`}>
                          {log.level.toUpperCase()}
                        </span>
                        <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-700 font-medium">
                          {log.source}
                        </span>
                        {log.user !== 'system' && (
                          <span className="px-2 py-1 text-xs rounded-full bg-purple-100 text-purple-700 font-medium">
                            {log.user}
                          </span>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-2 space-x-reverse text-sm text-gray-500">
                        <Clock className="w-4 h-4" />
                        <span>{new Date(log.timestamp).toLocaleString('fa-IR')}</span>
                      </div>
                    </div>
                    
                    <p className="text-gray-900 mb-2">{log.message}</p>
                    
                    {log.details && (
                      <details className="text-sm text-gray-600">
                        <summary className="cursor-pointer hover:text-gray-800 transition-colors">
                          جزئیات بیشتر
                        </summary>
                        <p className="mt-2 p-3 bg-gray-50 rounded border border-gray-200">
                          {log.details}
                        </p>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="p-12 text-center">
              <Eye className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">هیچ لاگی یافت نشد</h3>
              <p className="text-gray-600">فیلترهای خود را تغییر دهید یا مدت زمان بیشتری انتخاب کنید</p>
            </div>
          )}
        </div>
        
        {/* Load more button */}
        {filteredLogs.length > 0 && (
          <div className="p-6 border-t border-gray-200 text-center">
            <button className="bg-gray-100 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-200 transition-colors">
              بارگذاری لاگ‌های بیشتر
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default LogsPage;