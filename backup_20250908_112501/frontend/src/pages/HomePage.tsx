import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { BarChart3, FileText, Brain, Database, Activity, TrendingUp } from 'lucide-react';

const HomePage: React.FC = () => {
  // Fetch system health data
  const { data: systemHealth } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: async () => {
      const response = await fetch('/api/system/health');
      return response.json();
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch database statistics
  const { data: dbStats } = useQuery({
    queryKey: ['databaseStats'],
    queryFn: async () => {
      const response = await fetch('/api/database/statistics');
      return response.json();
    },
  });

  const statsCards = [
    {
      title: 'اسناد حقوقی',
      value: dbStats?.total_documents || '0',
      icon: FileText,
      color: 'bg-blue-500',
      change: '+12%'
    },
    {
      title: 'مدل‌های آموزش‌دیده',
      value: '3',
      icon: Brain,
      color: 'bg-green-500',
      change: '+1'
    },
    {
      title: 'دقت طبقه‌بندی',
      value: '94.2%',
      icon: TrendingUp,
      color: 'bg-purple-500',
      change: '+2.1%'
    },
    {
      title: 'وضعیت سیستم',
      value: systemHealth?.status === 'healthy' ? 'سالم' : 'نامشخص',
      icon: Activity,
      color: systemHealth?.status === 'healthy' ? 'bg-green-500' : 'bg-yellow-500',
      change: systemHealth?.uptime ? `${Math.floor(systemHealth.uptime / 3600)}h` : '0h'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          خانه سیستم هوش مصنوعی حقوقی فارسی
        </h1>
        <p className="text-gray-600">
          سیستم پیشرفته آموزش و طبقه‌بندی اسناد حقوقی با استفاده از هوش مصنوعی
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsCards.map((card, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{card.title}</p>
                <p className="text-2xl font-bold text-gray-900">{card.value}</p>
                <p className="text-sm text-green-600 mt-1">
                  {card.change} از ماه گذشته
                </p>
              </div>
              <div className={`${card.color} rounded-full p-3`}>
                <card.icon className="h-6 w-6 text-white" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            اقدامات سریع
          </h2>
          <div className="space-y-3">
            <button className="w-full text-right bg-blue-50 hover:bg-blue-100 rounded-lg p-4 transition-colors">
              <div className="flex items-center">
                <FileText className="h-5 w-5 text-blue-600 ml-3" />
                <div>
                  <p className="font-medium text-gray-900">آپلود سند جدید</p>
                  <p className="text-sm text-gray-600">اضافه کردن سند حقوقی جدید</p>
                </div>
              </div>
            </button>
            
            <button className="w-full text-right bg-green-50 hover:bg-green-100 rounded-lg p-4 transition-colors">
              <div className="flex items-center">
                <Brain className="h-5 w-5 text-green-600 ml-3" />
                <div>
                  <p className="font-medium text-gray-900">شروع آموزش جدید</p>
                  <p className="text-sm text-gray-600">آموزش مدل با داده‌های جدید</p>
                </div>
              </div>
            </button>
            
            <button className="w-full text-right bg-purple-50 hover:bg-purple-100 rounded-lg p-4 transition-colors">
              <div className="flex items-center">
                <Database className="h-5 w-5 text-purple-600 ml-3" />
                <div>
                  <p className="font-medium text-gray-900">طبقه‌بندی متن</p>
                  <p className="text-sm text-gray-600">تست طبقه‌بندی متن فارسی</p>
                </div>
              </div>
            </button>
          </div>
        </div>

        {/* System Overview */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            نمای کلی سیستم
          </h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">استفاده از CPU</span>
              <span className="font-medium">{systemHealth?.cpu_usage || '0'}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${systemHealth?.cpu_usage || 0}%` }}
              ></div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-600">استفاده از حافظه</span>
              <span className="font-medium">{systemHealth?.memory_usage || '0'}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${systemHealth?.memory_usage || 0}%` }}
              ></div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-600">وضعیت GPU</span>
              <span className={`font-medium ${systemHealth?.gpu_available ? 'text-green-600' : 'text-red-600'}`}>
                {systemHealth?.gpu_available ? 'فعال' : 'غیرفعال'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          فعالیت‌های اخیر
        </h2>
        <div className="space-y-3">
          <div className="flex items-center p-3 bg-gray-50 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full ml-3"></div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">
                مدل BERT فارسی با موفقیت بارگذاری شد
              </p>
              <p className="text-xs text-gray-500">2 دقیقه پیش</p>
            </div>
          </div>
          
          <div className="flex items-center p-3 bg-gray-50 rounded-lg">
            <div className="w-2 h-2 bg-blue-500 rounded-full ml-3"></div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">
                5 سند حقوقی جدید اضافه شد
              </p>
              <p className="text-xs text-gray-500">15 دقیقه پیش</p>
            </div>
          </div>
          
          <div className="flex items-center p-3 bg-gray-50 rounded-lg">
            <div className="w-2 h-2 bg-purple-500 rounded-full ml-3"></div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">
                جلسه آموزش DoRA کامل شد
              </p>
              <p className="text-xs text-gray-500">1 ساعت پیش</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;