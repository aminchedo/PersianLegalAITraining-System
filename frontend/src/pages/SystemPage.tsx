import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, Server, Database, Zap, HardDrive, Cpu, MemoryStick, Gauge } from 'lucide-react';

interface SystemMetrics {
  timestamp: string;
  system: {
    cpu_percent: number;
    memory_percent: number;
    memory_used_gb: number;
    memory_total_gb: number;
    disk_percent: number;
    disk_free_gb: number;
    disk_total_gb: number;
  };
  gpu: {
    gpu_available: boolean;
    gpu_count?: number;
    gpu_memory_allocated?: number;
    gpu_memory_cached?: number;
    gpu_utilization?: number;
  };
  models: {
    models_loaded: boolean;
    model_size_mb: number;
    pytorch_version: string;
    device: string;
    precision: string;
  };
  performance: {
    api_response_time: number;
    database_query_time: number;
    classification_speed: number;
  };
  health_score: number;
}

interface PerformanceSummary {
  period_hours: number;
  samples_count: number;
  averages: {
    cpu_percent: number;
    memory_percent: number;
    api_response_time_ms: number;
    health_score: number;
  };
  current_status: 'healthy' | 'degraded' | 'critical';
  recommendations: string[];
}

const SystemPage: React.FC = () => {
  // Fetch real-time metrics
  const { data: metrics } = useQuery<SystemMetrics>({
    queryKey: ['systemMetrics'],
    queryFn: async () => {
      const response = await fetch('/api/system/metrics');
      return response.json();
    },
    refetchInterval: 5000, // Update every 5 seconds
  });

  // Fetch performance summary
  const { data: summary } = useQuery<PerformanceSummary>({
    queryKey: ['performanceSummary'],
    queryFn: async () => {
      const response = await fetch('/api/system/performance-summary?hours=24');
      return response.json();
    },
    refetchInterval: 60000, // Update every minute
  });

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'degraded': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'healthy': return 'سالم';
      case 'degraded': return 'کاهش عملکرد';
      case 'critical': return 'بحرانی';
      default: return 'نامشخص';
    }
  };

  const formatBytes = (bytes: number) => {
    return `${bytes.toFixed(1)} GB`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              مانیتورینگ سیستم
            </h1>
            <p className="text-gray-600">
              نظارت بر عملکرد سیستم و منابع سخت‌افزاری
            </p>
          </div>
          
          {metrics && (
            <div className="text-left">
              <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${getHealthColor(metrics.health_score)}`}>
                <Activity className="h-4 w-4 ml-2" />
                امتیاز سلامت: {metrics.health_score}
              </div>
              <p className="text-xs text-gray-500 mt-1">
                آخرین بروزرسانی: {new Date(metrics.timestamp).toLocaleString('fa-IR')}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* System Overview Cards */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">پردازنده (CPU)</h3>
              <Cpu className="h-5 w-5 text-blue-600" />
            </div>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-gray-900">
                {metrics.system.cpu_percent.toFixed(1)}%
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${metrics.system.cpu_percent}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">حافظه (RAM)</h3>
              <MemoryStick className="h-5 w-5 text-green-600" />
            </div>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-gray-900">
                {metrics.system.memory_percent.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">
                {formatBytes(metrics.system.memory_used_gb)} / {formatBytes(metrics.system.memory_total_gb)}
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${metrics.system.memory_percent}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">فضای ذخیره</h3>
              <HardDrive className="h-5 w-5 text-purple-600" />
            </div>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-gray-900">
                {metrics.system.disk_percent.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">
                {formatBytes(metrics.system.disk_free_gb)} آزاد از {formatBytes(metrics.system.disk_total_gb)}
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${metrics.system.disk_percent}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">پردازنده گرافیکی</h3>
              <Zap className="h-5 w-5 text-yellow-600" />
            </div>
            <div className="space-y-3">
              <div className="text-2xl font-bold text-gray-900">
                {metrics.gpu.gpu_available ? 'فعال' : 'غیرفعال'}
              </div>
              {metrics.gpu.gpu_available && (
                <>
                  <div className="text-sm text-gray-600">
                    تعداد: {metrics.gpu.gpu_count} | استفاده: {metrics.gpu.gpu_utilization?.toFixed(1)}%
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${metrics.gpu.gpu_utilization || 0}%` }}
                    ></div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            عملکرد سیستم
          </h2>
          
          {metrics && (
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600">زمان پاسخ API</span>
                <span className="font-medium">
                  {metrics.performance.api_response_time > 0 ? `${metrics.performance.api_response_time}ms` : 'N/A'}
                </span>
              </div>
              
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600">زمان پرس‌وجو دیتابیس</span>
                <span className="font-medium">
                  {metrics.performance.database_query_time > 0 ? `${metrics.performance.database_query_time}ms` : 'N/A'}
                </span>
              </div>
              
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600">سرعت طبقه‌بندی</span>
                <span className="font-medium">
                  {metrics.performance.classification_speed > 0 ? `${metrics.performance.classification_speed}ms` : 'N/A'}
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            اطلاعات مدل
          </h2>
          
          {metrics && (
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600">وضعیت مدل</span>
                <span className={`font-medium ${metrics.models.models_loaded ? 'text-green-600' : 'text-red-600'}`}>
                  {metrics.models.models_loaded ? 'بارگذاری شده' : 'بارگذاری نشده'}
                </span>
              </div>
              
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600">حجم مدل</span>
                <span className="font-medium">{metrics.models.model_size_mb} MB</span>
              </div>
              
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600">دستگاه پردازش</span>
                <span className="font-medium">{metrics.models.device.toUpperCase()}</span>
              </div>
              
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-gray-600">نسخه PyTorch</span>
                <span className="font-medium">{metrics.models.pytorch_version}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Performance Summary */}
      {summary && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">
              خلاصه عملکرد (24 ساعت گذشته)
            </h2>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(summary.current_status)}`}>
              {getStatusText(summary.current_status)}
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 mb-1">
                {summary.averages.cpu_percent.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">میانگین CPU</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 mb-1">
                {summary.averages.memory_percent.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">میانگین حافظه</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 mb-1">
                {summary.averages.api_response_time_ms.toFixed(0)}ms
              </div>
              <div className="text-sm text-gray-600">میانگین پاسخ API</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 mb-1">
                {summary.averages.health_score.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">میانگین امتیاز سلامت</div>
            </div>
          </div>
          
          {summary.recommendations.length > 0 && (
            <div className="border-t border-gray-200 pt-6">
              <h3 className="text-md font-semibold text-gray-900 mb-3">
                پیشنهادات بهینه‌سازی
              </h3>
              <div className="space-y-2">
                {summary.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start p-3 bg-yellow-50 rounded-lg">
                    <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 ml-3 flex-shrink-0"></div>
                    <p className="text-sm text-gray-700">{rec}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SystemPage;