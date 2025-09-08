import React, { useState, useEffect } from 'react';
import { 
  Monitor, Cpu, HardDrive, Thermometer, Power, Network, Activity, 
  AlertTriangle, CheckCircle, Clock, Zap, Server, BarChart3,
  RefreshCw, Settings, Download, Eye, EyeOff, Maximize2, X,
  TrendingUp, TrendingDown, Wifi, Globe, Database, Memory
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, PieChart, Pie, Cell } from 'recharts';
import { useAppContext } from './router';

const MonitoringPage = () => {
  const { realTimeData, autoRefresh, setAutoRefresh, refreshInterval } = useAppContext();
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [alertsFilter, setAlertsFilter] = useState('all');
  const [timeRange, setTimeRange] = useState('1h');
  const [fullScreenChart, setFullScreenChart] = useState(null);

  // System health metrics
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: { usage: 0, temperature: 0, cores: 24, threads: 48 },
    memory: { usage: 0, total: 64, available: 0 },
    gpu: { usage: 0, temperature: 0, memory: 16, vram: 0 },
    disk: { usage: 85, total: 2000, free: 300, iops: 1240 },
    network: { upload: 125, download: 89, latency: 12 },
    power: { consumption: 0, efficiency: 0 }
  });

  // System alerts
  const [alerts, setAlerts] = useState([
    {
      id: 1,
      type: 'warning',
      title: 'استفاده بالای CPU',
      message: 'استفاده از CPU برای 10 دقیقه گذشته بالای 85% بوده است',
      timestamp: new Date(Date.now() - 300000),
      severity: 'medium',
      component: 'cpu',
      status: 'active'
    },
    {
      id: 2,
      type: 'error',
      title: 'دمای بالای GPU',
      message: 'دمای GPU به 83°C رسیده - کولینگ بررسی شود',
      timestamp: new Date(Date.now() - 180000),
      severity: 'high',
      component: 'gpu',
      status: 'active'
    },
    {
      id: 3,
      type: 'info',
      title: 'بروزرسانی سیستم',
      message: 'بروزرسانی جدید سیستم در دسترس است',
      timestamp: new Date(Date.now() - 900000),
      severity: 'low',
      component: 'system',
      status: 'resolved'
    },
    {
      id: 4,
      type: 'warning',
      title: 'فضای دیسک کم',
      message: 'فضای باقی‌مانده دیسک کمتر از 15% است',
      timestamp: new Date(Date.now() - 1200000),
      severity: 'medium',
      component: 'disk',
      status: 'active'
    }
  ]);

  // Service status
  const [services, setServices] = useState([
    { name: 'Model Trainer', status: 'running', uptime: '5d 12h', cpu: 45, memory: 2.1 },
    { name: 'Data Collector', status: 'running', uptime: '3d 8h', cpu: 12, memory: 0.8 },
    { name: 'API Server', status: 'running', uptime: '7d 15h', cpu: 8, memory: 1.2 },
    { name: 'Database', status: 'running', uptime: '15d 3h', cpu: 15, memory: 4.5 },
    { name: 'Cache Service', status: 'running', uptime: '12d 9h', cpu: 5, memory: 0.9 },
    { name: 'Backup Service', status: 'stopped', uptime: '0h', cpu: 0, memory: 0 }
  ]);

  // Update metrics from realTimeData
  useEffect(() => {
    if (realTimeData.length > 0) {
      const latest = realTimeData[realTimeData.length - 1];
      setSystemMetrics(prev => ({
        ...prev,
        cpu: { 
          ...prev.cpu, 
          usage: latest.cpu,
          temperature: latest.temperature 
        },
        memory: { 
          ...prev.memory, 
          usage: latest.memory,
          available: prev.memory.total * (1 - latest.memory / 100) 
        },
        gpu: { 
          ...prev.gpu, 
          usage: latest.gpu,
          temperature: latest.temperature + 10,
          vram: (latest.gpu / 100) * prev.gpu.memory
        },
        power: {
          ...prev.power,
          consumption: latest.power,
          efficiency: 92 + Math.random() * 6
        }
      }));
    }
  }, [realTimeData]);

  // Filter alerts
  const filteredAlerts = alerts.filter(alert => {
    if (alertsFilter === 'all') return true;
    if (alertsFilter === 'active') return alert.status === 'active';
    if (alertsFilter === 'resolved') return alert.status === 'resolved';
    return alert.severity === alertsFilter;
  });

  // Get time range data
  const getTimeRangeData = () => {
    const ranges = {
      '15m': realTimeData.slice(-15),
      '1h': realTimeData.slice(-20),
      '6h': realTimeData.slice(-30),
      '24h': realTimeData.slice(-50)
    };
    return ranges[timeRange] || realTimeData.slice(-20);
  };

  // Alert severity configs
  const getAlertConfig = (type, severity) => {
    const configs = {
      error: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertTriangle },
      warning: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: AlertTriangle },
      info: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: CheckCircle }
    };
    return configs[type] || configs.info;
  };

  // Service status configs
  const getServiceConfig = (status) => {
    const configs = {
      running: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle },
      stopped: { color: 'text-red-700', bgColor: 'bg-red-100', icon: X },
      error: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertTriangle }
    };
    return configs[status] || configs.stopped;
  };

  // Metric cards data
  const metricCards = [
    {
      title: 'استفاده از CPU',
      value: systemMetrics.cpu.usage,
      unit: '%',
      icon: Cpu,
      color: 'blue',
      details: `${systemMetrics.cpu.cores} هسته / ${systemMetrics.cpu.threads} رشته`,
      chart: 'cpu'
    },
    {
      title: 'استفاده از حافظه',
      value: systemMetrics.memory.usage,
      unit: '%',
      icon: HardDrive,
      color: 'green',
      details: `${systemMetrics.memory.available.toFixed(1)} GB در دسترس`,
      chart: 'memory'
    },
    {
      title: 'استفاده از GPU',
      value: systemMetrics.gpu.usage,
      unit: '%',
      icon: Activity,
      color: 'purple',
      details: `${systemMetrics.gpu.vram.toFixed(1)} GB VRAM`,
      chart: 'gpu'
    },
    {
      title: 'مصرف برق',
      value: systemMetrics.power.consumption,
      unit: 'W',
      icon: Power,
      color: 'orange',
      details: `${systemMetrics.power.efficiency.toFixed(1)}% بازدهی`,
      chart: 'power'
    }
  ];

  // System health score
  const calculateHealthScore = () => {
    const weights = {
      cpu: 0.25,
      memory: 0.25,
      gpu: 0.20,
      temperature: 0.20,
      alerts: 0.10
    };
    
    const cpuScore = Math.max(0, 100 - systemMetrics.cpu.usage);
    const memoryScore = Math.max(0, 100 - systemMetrics.memory.usage);
    const gpuScore = Math.max(0, 100 - systemMetrics.gpu.usage);
    const tempScore = Math.max(0, 100 - ((systemMetrics.cpu.temperature - 30) / 50) * 100);
    const alertScore = Math.max(0, 100 - (alerts.filter(a => a.status === 'active').length * 10));
    
    return Math.round(
      cpuScore * weights.cpu +
      memoryScore * weights.memory +
      gpuScore * weights.gpu +
      tempScore * weights.temperature +
      alertScore * weights.alerts
    );
  };

  const healthScore = calculateHealthScore();

  return (
    <div className="space-y-6" dir="rtl">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 via-blue-600 to-indigo-600 rounded-2xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <Monitor className="w-8 h-8" />
              نظارت سیستم
            </h1>
            <p className="text-blue-100">مانیتورینگ عملکرد، سلامت و منابع سیستم</p>
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-xl font-medium transition-all flex items-center gap-2 ${
                autoRefresh 
                  ? 'bg-white/20 text-white hover:bg-white/30' 
                  : 'bg-white text-blue-600 hover:bg-blue-50'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${autoRefresh ? 'animate-spin' : ''}`} />
              {autoRefresh ? 'زنده' : 'متوقف'}
            </button>
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="bg-white/20 text-white border border-white/30 rounded-xl px-4 py-2 backdrop-blur-sm"
            >
              <option value="15m" className="text-gray-900">15 دقیقه</option>
              <option value="1h" className="text-gray-900">1 ساعت</option>
              <option value="6h" className="text-gray-900">6 ساعت</option>
              <option value="24h" className="text-gray-900">24 ساعت</option>
            </select>
          </div>
        </div>
        
        {/* System Health Score */}
        <div className="mt-6 bg-white/20 rounded-xl p-4 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold mb-1">امتیاز سلامت سیستم</h3>
              <p className="text-sm opacity-75">بر اساس تجمیع تمام متریک‌ها</p>
            </div>
            <div className="text-center">
              <div className={`text-4xl font-bold ${
                healthScore >= 90 ? 'text-green-300' :
                healthScore >= 70 ? 'text-yellow-300' : 'text-red-300'
              }`}>
                {healthScore}
              </div>
              <div className="text-sm opacity-75">از 100</div>
            </div>
          </div>
          <div className="w-full bg-white/20 rounded-full h-2 mt-3">
            <div 
              className={`h-2 rounded-full transition-all duration-1000 ${
                healthScore >= 90 ? 'bg-green-400' :
                healthScore >= 70 ? 'bg-yellow-400' : 'bg-red-400'
              }`}
              style={{ width: `${healthScore}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metricCards.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <div 
              key={index}
              className="bg-white rounded-2xl p-6 shadow-lg border-l-4 border-gray-200 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 group cursor-pointer"
              style={{ borderLeftColor: `var(--${metric.color}-500)` }}
              onClick={() => setSelectedMetric(metric)}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex-1">
                  <p className="text-gray-600 text-sm font-medium">{metric.title}</p>
                  <div className="flex items-baseline gap-2">
                    <h3 className="text-3xl font-bold text-gray-900">
                      {metric.value.toFixed(metric.unit === 'W' ? 0 : 1)}
                    </h3>
                    <span className="text-gray-500 text-sm">{metric.unit}</span>
                  </div>
                  <p className="text-gray-500 text-xs mt-1">{metric.details}</p>
                </div>
                <div className={`bg-${metric.color}-100 rounded-xl p-3 group-hover:scale-110 transition-transform`}>
                  <Icon className={`w-6 h-6 text-${metric.color}-600`} />
                </div>
              </div>
              
              {/* Mini chart */}
              <div className="h-12">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={getTimeRangeData().slice(-10)}>
                    <Area 
                      type="monotone" 
                      dataKey={metric.chart} 
                      stroke={`var(--${metric.color}-500)`} 
                      fill={`var(--${metric.color}-500)`} 
                      fillOpacity={0.2}
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              
              {/* Status indicator */}
              <div className="flex items-center justify-between mt-3">
                <div className={`w-2 h-2 rounded-full ${
                  metric.value > 85 ? 'bg-red-500' :
                  metric.value > 70 ? 'bg-yellow-500' : 'bg-green-500'
                }`}></div>
                <span className="text-xs text-gray-500">
                  {metric.value > 85 ? 'بالا' : metric.value > 70 ? 'متوسط' : 'نرمال'}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Resources Chart */}
        <div className="bg-white rounded-2xl p-6 shadow-lg relative group">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">منابع سیستم</h3>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-500">زنده</span>
              </div>
              <button 
                onClick={() => setFullScreenChart('resources')}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-2 hover:bg-gray-100 rounded-lg"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={getTimeRangeData()}>
              <defs>
                <linearGradient id="cpuGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="memoryGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="gpuGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip />
              <Area type="monotone" dataKey="cpu" stroke="#3B82F6" fill="url(#cpuGradient)" strokeWidth={2} />
              <Area type="monotone" dataKey="memory" stroke="#10B981" fill="url(#memoryGradient)" strokeWidth={2} />
              <Area type="monotone" dataKey="gpu" stroke="#8B5CF6" fill="url(#gpuGradient)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>CPU</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>حافظه</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
              <span>GPU</span>
            </div>
          </div>
        </div>

        {/* Temperature and Power */}
        <div className="bg-white rounded-2xl p-6 shadow-lg relative group">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">دما و مصرف برق</h3>
            <button 
              onClick={() => setFullScreenChart('thermal')}
              className="opacity-0 group-hover:opacity-100 transition-opacity p-2 hover:bg-gray-100 rounded-lg"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={getTimeRangeData()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
              <YAxis yAxisId="temp" stroke="#6b7280" fontSize={12} />
              <YAxis yAxisId="power" orientation="left" stroke="#6b7280" fontSize={12} />
              <Tooltip />
              <Line yAxisId="temp" type="monotone" dataKey="temperature" stroke="#EF4444" strokeWidth={3} dot={false} />
              <Line yAxisId="power" type="monotone" dataKey="power" stroke="#F59E0B" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <span>دما (°C)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <span>برق (W)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Services and Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Services */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">سرویس‌های سیستم</h3>
            <button className="text-gray-500 hover:text-gray-700">
              <Settings className="w-4 h-4" />
            </button>
          </div>
          
          <div className="space-y-4">
            {services.map((service, index) => {
              const statusConfig = getServiceConfig(service.status);
              const StatusIcon = statusConfig.icon;
              
              return (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition-all">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${statusConfig.bgColor}`}>
                      <StatusIcon className={`w-5 h-5 ${statusConfig.color}`} />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{service.name}</p>
                      <p className="text-sm text-gray-500">Uptime: {service.uptime}</p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span>CPU: {service.cpu}%</span>
                      <span>RAM: {service.memory}GB</span>
                    </div>
                    <span className={`text-xs font-medium ${statusConfig.color}`}>
                      {service.status === 'running' ? 'در حال اجرا' : 
                       service.status === 'stopped' ? 'متوقف' : 'خطا'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* System Alerts */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">هشدارهای سیستم</h3>
            <select 
              value={alertsFilter}
              onChange={(e) => setAlertsFilter(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
            >
              <option value="all">همه</option>
              <option value="active">فعال</option>
              <option value="resolved">حل شده</option>
              <option value="high">بحرانی</option>
              <option value="medium">متوسط</option>
              <option value="low">کم</option>
            </select>
          </div>
          
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {filteredAlerts.map(alert => {
              const alertConfig = getAlertConfig(alert.type, alert.severity);
              const AlertIcon = alertConfig.icon;
              
              return (
                <div 
                  key={alert.id}
                  className={`p-3 rounded-xl border-l-4 ${
                    alert.status === 'active' ? 'bg-red-50 border-red-500' : 'bg-gray-50 border-gray-300'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${alertConfig.bgColor}`}>
                      <AlertIcon className={`w-4 h-4 ${alertConfig.color}`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-medium text-gray-900">{alert.title}</h4>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          alert.severity === 'high' ? 'bg-red-100 text-red-700' :
                          alert.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-blue-100 text-blue-700'
                        }`}>
                          {alert.severity === 'high' ? 'بحرانی' :
                           alert.severity === 'medium' ? 'متوسط' : 'کم'}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 leading-5">{alert.message}</p>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs text-gray-500">
                          {new Date(alert.timestamp).toLocaleString('fa-IR')}
                        </span>
                        <span className={`text-xs font-medium ${
                          alert.status === 'active' ? 'text-red-600' : 'text-green-600'
                        }`}>
                          {alert.status === 'active' ? 'فعال' : 'حل شده'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
            
            {filteredAlerts.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>هیچ هشداری یافت نشد</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Network and Storage */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Network Stats */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">شبکه</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-500" />
                <span className="text-sm">آپلود</span>
              </div>
              <span className="font-semibold">{systemMetrics.network.upload} MB/s</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingDown className="w-4 h-4 text-blue-500" />
                <span className="text-sm">دانلود</span>
              </div>
              <span className="font-semibold">{systemMetrics.network.download} MB/s</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-orange-500" />
                <span className="text-sm">تاخیر</span>
              </div>
              <span className="font-semibold">{systemMetrics.network.latency} ms</span>
            </div>
          </div>
        </div>

        {/* Storage Stats */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">ذخیره‌سازی</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>استفاده شده</span>
                <span>{systemMetrics.disk.usage}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${
                    systemMetrics.disk.usage > 90 ? 'bg-red-500' :
                    systemMetrics.disk.usage > 80 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${systemMetrics.disk.usage}%` }}
                ></div>
              </div>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>آزاد</span>
              <span>{systemMetrics.disk.free} GB</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>IOPS</span>
              <span>{systemMetrics.disk.iops}</span>
            </div>
          </div>
        </div>

        {/* System Info */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">اطلاعات سیستم</h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span>OS:</span>
              <span>Ubuntu 22.04 LTS</span>
            </div>
            <div className="flex justify-between">
              <span>Kernel:</span>
              <span>5.15.0-91</span>
            </div>
            <div className="flex justify-between">
              <span>Python:</span>
              <span>3.11.5</span>
            </div>
            <div className="flex justify-between">
              <span>PyTorch:</span>
              <span>2.1.2</span>
            </div>
            <div className="flex justify-between">
              <span>CUDA:</span>
              <span>12.2</span>
            </div>
          </div>
        </div>
      </div>

      {/* Full Screen Chart Modal */}
      {fullScreenChart && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-7xl h-5/6 p-6 relative">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold">
                {fullScreenChart === 'resources' ? 'منابع سیستم - نمای کامل' : 'دما و مصرف برق - نمای کامل'}
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
                {fullScreenChart === 'resources' ? (
                  <AreaChart data={getTimeRangeData()}>
                    <defs>
                      <linearGradient id="cpuGradientFull" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="memoryGradientFull" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="gpuGradientFull" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip />
                    <Area type="monotone" dataKey="cpu" stroke="#3B82F6" fill="url(#cpuGradientFull)" strokeWidth={3} />
                    <Area type="monotone" dataKey="memory" stroke="#10B981" fill="url(#memoryGradientFull)" strokeWidth={3} />
                    <Area type="monotone" dataKey="gpu" stroke="#8B5CF6" fill="url(#gpuGradientFull)" strokeWidth={3} />
                  </AreaChart>
                ) : (
                  <LineChart data={getTimeRangeData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" stroke="#6b7280" />
                    <YAxis yAxisId="temp" stroke="#6b7280" />
                    <YAxis yAxisId="power" orientation="left" stroke="#6b7280" />
                    <Tooltip />
                    <Line yAxisId="temp" type="monotone" dataKey="temperature" stroke="#EF4444" strokeWidth={4} dot={false} />
                    <Line yAxisId="power" type="monotone" dataKey="power" stroke="#F59E0B" strokeWidth={4} dot={false} />
                  </LineChart>
                )}
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Metric Detail Modal */}
      {selectedMetric && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-auto p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">جزئیات: {selectedMetric.title}</h3>
              <button 
                onClick={() => setSelectedMetric(null)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Current Stats */}
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">وضعیت فعلی</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>مقدار فعلی:</span>
                      <span className="font-semibold">{selectedMetric.value.toFixed(1)}{selectedMetric.unit}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>وضعیت:</span>
                      <span className={`font-semibold ${
                        selectedMetric.value > 85 ? 'text-red-600' :
                        selectedMetric.value > 70 ? 'text-yellow-600' : 'text-green-600'
                      }`}>
                        {selectedMetric.value > 85 ? 'بالا' : selectedMetric.value > 70 ? 'متوسط' : 'نرمال'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>جزئیات:</span>
                      <span>{selectedMetric.details}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Detailed Chart */}
              <div>
                <h4 className="font-semibold mb-3">نمودار تفصیلی</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={getTimeRangeData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" stroke="#6b7280" fontSize={12} />
                    <YAxis stroke="#6b7280" fontSize={12} />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey={selectedMetric.chart} 
                      stroke={`var(--${selectedMetric.color}-500)`} 
                      strokeWidth={3} 
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MonitoringPage;