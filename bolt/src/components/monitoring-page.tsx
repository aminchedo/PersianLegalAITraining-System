import React, { useState, useEffect } from 'react';
import { Monitor, Cpu, HardDrive, Zap, Thermometer, Wifi, AlertTriangle, CheckCircle, Activity, Server, Database } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, RadialBarChart, RadialBar, Cell } from 'recharts';
import { usePersianAI } from '../hooks/usePersianAI';

const MonitoringPage: React.FC = () => {
  const { state } = usePersianAI();
  const [realTimeData, setRealTimeData] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [systemHealth, setSystemHealth] = useState<number>(95);

  useEffect(() => {
    // Generate real-time monitoring data
    const interval = setInterval(() => {
      const now = new Date();
      const newDataPoint = {
        time: now.toLocaleTimeString('fa-IR', { hour12: false }),
        cpu: Math.random() * 20 + 40,
        memory: Math.random() * 15 + 60,
        gpu: Math.random() * 25 + 65,
        network: Math.random() * 100 + 50,
        disk: Math.random() * 5 + 30,
        temperature: Math.random() * 10 + 55,
      };
      setRealTimeData(prev => [...prev.slice(-19), newDataPoint]);
    }, 3000);

    // Generate random alerts
    const alertInterval = setInterval(() => {
      if (Math.random() > 0.8) {
        const alertTypes = [
          { type: 'warning', message: 'استفاده از CPU بالا رفته است', icon: AlertTriangle, color: 'text-yellow-600' },
          { type: 'info', message: 'سیستم به‌روزرسانی شد', icon: CheckCircle, color: 'text-blue-600' },
          { type: 'success', message: 'آموزش مدل با موفقیت تکمیل شد', icon: CheckCircle, color: 'text-green-600' },
        ];
        const randomAlert = alertTypes[Math.floor(Math.random() * alertTypes.length)];
        setAlerts(prev => [{
          id: Date.now(),
          ...randomAlert,
          timestamp: new Date().toLocaleTimeString('fa-IR')
        }, ...prev.slice(0, 9)]);
      }
    }, 15000);

    return () => {
      clearInterval(interval);
      clearInterval(alertInterval);
    };
  }, []);

  const systemComponents = [
    { name: 'CPU', value: state.systemMetrics?.cpu_usage || 0, max: 100, color: '#3b82f6', icon: Cpu },
    { name: 'Memory', value: state.systemMetrics?.memory_usage || 0, max: 100, color: '#10b981', icon: HardDrive },
    { name: 'GPU', value: state.systemMetrics?.gpu_usage || 0, max: 100, color: '#8b5cf6', icon: Zap },
    { name: 'Disk', value: state.systemMetrics?.disk_usage || 0, max: 100, color: '#f59e0b', icon: Database },
  ];

  const serverStatus = [
    { name: 'API Server', status: 'online', uptime: '99.9%', load: 45, color: 'text-green-600' },
    { name: 'Database', status: 'online', uptime: '99.8%', load: 23, color: 'text-green-600' },
    { name: 'Training Worker', status: 'online', uptime: '98.5%', load: 78, color: 'text-green-600' },
    { name: 'File Storage', status: 'warning', uptime: '97.2%', load: 89, color: 'text-yellow-600' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">نظارت سیستم</h1>
            <p className="text-indigo-100">مانیتورینگ زنده وضعیت سیستم و منابع</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="flex items-center space-x-2 space-x-reverse">
                <div className={`w-3 h-3 rounded-full ${state.isConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
                <span className="text-sm font-medium">
                  {state.isConnected ? 'متصل' : 'قطع شده'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {systemComponents.map((component) => (
          <div key={component.name} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3 space-x-reverse">
                <div className="p-2 rounded-lg" style={{ backgroundColor: `${component.color}15` }}>
                  <component.icon className="w-6 h-6" style={{ color: component.color }} />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{component.name}</h3>
                  <p className="text-2xl font-bold text-gray-900">{component.value}%</p>
                </div>
              </div>
            </div>
            
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="h-2 rounded-full transition-all duration-300"
                style={{ 
                  width: `${component.value}%`, 
                  backgroundColor: component.color 
                }}
              ></div>
            </div>
            
            <div className="flex items-center justify-between mt-2">
              <span className="text-xs text-gray-600">استفاده فعلی</span>
              <span className={`text-xs font-medium ${
                component.value > 80 ? 'text-red-600' : 
                component.value > 60 ? 'text-yellow-600' : 'text-green-600'
              }`}>
                {component.value > 80 ? 'بالا' : component.value > 60 ? 'متوسط' : 'نرمال'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Real-time Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900">عملکرد سیستم (زنده)</h2>
            <div className="flex items-center space-x-2 space-x-reverse text-sm text-gray-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>به‌روزرسانی هر 3 ثانیه</span>
            </div>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={realTimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="cpu" stroke="#3b82f6" name="CPU" strokeWidth={2} />
                <Line type="monotone" dataKey="memory" stroke="#10b981" name="حافظه" strokeWidth={2} />
                <Line type="monotone" dataKey="gpu" stroke="#8b5cf6" name="GPU" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6">شبکه و دما</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={realTimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="network" 
                  stackId="1" 
                  stroke="#06b6d4" 
                  fill="#06b6d4" 
                  name="شبکه (MB/s)"
                />
                <Area 
                  type="monotone" 
                  dataKey="temperature" 
                  stackId="2" 
                  stroke="#f97316" 
                  fill="#f97316" 
                  name="دما (°C)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* System Health & Alerts */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">وضعیت سرویس‌ها</h2>
            
            <div className="space-y-4">
              {serverStatus.map((server, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-4 space-x-reverse">
                    <div className={`w-3 h-3 rounded-full ${
                      server.status === 'online' ? 'bg-green-500' : 
                      server.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}></div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{server.name}</h3>
                      <p className="text-sm text-gray-600">Uptime: {server.uptime}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4 space-x-reverse">
                    <div className="text-left">
                      <p className="text-sm text-gray-600">Load</p>
                      <p className="font-medium text-gray-900">{server.load}%</p>
                    </div>
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-300 ${
                          server.load > 80 ? 'bg-red-500' : 
                          server.load > 60 ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${server.load}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900">هشدارها</h2>
            <button className="text-blue-600 hover:text-blue-700 transition-colors text-sm">
              مشاهده همه
            </button>
          </div>
          
          <div className="space-y-3">
            {alerts.length > 0 ? (
              alerts.map((alert) => (
                <div key={alert.id} className="flex items-center space-x-3 space-x-reverse p-3 bg-gray-50 rounded-lg">
                  <alert.icon className={`w-5 h-5 ${alert.color}`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900">{alert.message}</p>
                    <p className="text-xs text-gray-600">{alert.timestamp}</p>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <CheckCircle className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500">هیچ هشداری وجود ندارد</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Resource Usage Details */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">جزئیات استفاده از منابع</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* CPU Details */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-3 space-x-reverse mb-4">
              <Cpu className="w-6 h-6 text-blue-600" />
              <h3 className="font-semibold text-blue-900">پردازنده (CPU)</h3>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-blue-700">استفاده فعلی:</span>
                <span className="font-medium text-blue-900">{state.systemMetrics?.cpu_usage || 0}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-blue-700">میانگین 5 دقیقه:</span>
                <span className="font-medium text-blue-900">42%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-blue-700">تعداد هسته:</span>
                <span className="font-medium text-blue-900">8</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-blue-700">فرکانس:</span>
                <span className="font-medium text-blue-900">2.8 GHz</span>
              </div>
            </div>
          </div>

          {/* Memory Details */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center space-x-3 space-x-reverse mb-4">
              <HardDrive className="w-6 h-6 text-green-600" />
              <h3 className="font-semibold text-green-900">حافظه (RAM)</h3>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-green-700">استفاده:</span>
                <span className="font-medium text-green-900">{state.systemMetrics?.memory_usage || 0}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-green-700">مقدار استفاده:</span>
                <span className="font-medium text-green-900">12.8 GB</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-green-700">کل حافظه:</span>
                <span className="font-medium text-green-900">32 GB</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-green-700">آزاد:</span>
                <span className="font-medium text-green-900">19.2 GB</span>
              </div>
            </div>
          </div>

          {/* GPU Details */}
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="flex items-center space-x-3 space-x-reverse mb-4">
              <Zap className="w-6 h-6 text-purple-600" />
              <h3 className="font-semibold text-purple-900">کارت گرافیک (GPU)</h3>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-purple-700">استفاده:</span>
                <span className="font-medium text-purple-900">{state.systemMetrics?.gpu_usage || 0}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-purple-700">حافظه GPU:</span>
                <span className="font-medium text-purple-900">{state.systemMetrics?.gpu_memory || 0}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-purple-700">دما:</span>
                <span className="font-medium text-purple-900">{state.systemMetrics?.temperature || 0}°C</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-purple-700">مدل:</span>
                <span className="font-medium text-purple-900">RTX 4090</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Recommendations */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">پیشنهادات بهینه‌سازی</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 space-x-reverse mb-2">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
              <h3 className="font-semibold text-yellow-800">استفاده از حافظه بالا</h3>
            </div>
            <p className="text-sm text-yellow-700">
              استفاده از حافظه به 67% رسیده است. پیشنهاد می‌شود فرآیندهای غیرضروری را متوقف کنید.
            </p>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 space-x-reverse mb-2">
              <Thermometer className="w-5 h-5 text-blue-600" />
              <h3 className="font-semibold text-blue-800">کنترل دما</h3>
            </div>
            <p className="text-sm text-blue-700">
              دمای GPU در حد نرمال است. سیستم خنک‌کاری به خوبی کار می‌کند.
            </p>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 space-x-reverse mb-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <h3 className="font-semibold text-green-800">عملکرد بهینه</h3>
            </div>
            <p className="text-sm text-green-700">
              سیستم در بهترین حالت کارایی قرار دارد. تمام سرویس‌ها عملکرد مطلوبی دارند.
            </p>
          </div>

          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 space-x-reverse mb-2">
              <Server className="w-5 h-5 text-purple-600" />
              <h3 className="font-semibold text-purple-800">ظرفیت آموزش</h3>
            </div>
            <p className="text-sm text-purple-700">
              سیستم ظرفیت آموزش 2 مدل همزمان دیگر را دارد. منابع کافی در دسترس هستند.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonitoringPage;