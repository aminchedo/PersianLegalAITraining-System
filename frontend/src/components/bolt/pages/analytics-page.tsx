import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, TrendingDown, Download, Filter, Calendar, BarChart3, Activity, Target, Award, Clock } from 'lucide-react';
import { usePersianAI } from '../../../hooks/usePersianAI';

const AnalyticsPage: React.FC = () => {
  const { state } = usePersianAI();
  const [timeRange, setTimeRange] = useState('7d');
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [performanceData, setPerformanceData] = useState<any[]>([]);

  useEffect(() => {
    // Generate mock analytics data
    generateAnalyticsData();
  }, [timeRange]);

  const generateAnalyticsData = () => {
    const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
    const data = [];
    const perfData = [];
    
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      data.push({
        date: date.toLocaleDateString('fa-IR'),
        accuracy: Math.random() * 0.1 + 0.85,
        loss: Math.random() * 0.2 + 0.1,
        speed: Math.random() * 50 + 100,
        memory: Math.random() * 20 + 60,
        cpu: Math.random() * 30 + 40,
      });

      perfData.push({
        model: `Model ${i + 1}`,
        accuracy: Math.random() * 0.2 + 0.8,
        speed: Math.random() * 100 + 50,
        efficiency: Math.random() * 0.3 + 0.7,
        cost: Math.random() * 1000 + 500,
      });
    }
    
    setComparisonData(data.reverse());
    setPerformanceData(perfData.slice(0, 8));
  };

  const radarData = [
    { subject: 'دقت', A: 92, B: 87, fullMark: 100 },
    { subject: 'سرعت', A: 85, B: 78, fullMark: 100 },
    { subject: 'کارایی', A: 88, B: 82, fullMark: 100 },
    { subject: 'پایداری', A: 90, B: 85, fullMark: 100 },
    { subject: 'مصرف حافظه', A: 82, B: 75, fullMark: 100 },
    { subject: 'قابلیت اطمینان', A: 95, B: 88, fullMark: 100 },
  ];

  const modelComparisonData = [
    { name: 'Persian Legal LLM v1.0', accuracy: 87, speed: 145, memory: 2.3 },
    { name: 'Document Classifier', accuracy: 92, speed: 234, memory: 1.8 },
    { name: 'Contract Analyzer', accuracy: 89, speed: 178, memory: 2.1 },
    { name: 'Legal QA System', accuracy: 85, speed: 198, memory: 2.5 },
  ];

  const trainingTrends = [
    { month: 'فروردین', success: 23, failed: 2, canceled: 1 },
    { month: 'اردیبهشت', success: 28, failed: 1, canceled: 2 },
    { month: 'خرداد', success: 31, failed: 3, canceled: 0 },
    { month: 'تیر', success: 27, failed: 2, canceled: 1 },
    { month: 'مرداد', success: 33, failed: 1, canceled: 1 },
    { month: 'شهریور', success: 29, failed: 2, canceled: 0 },
  ];

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">آنالیز و گزارش‌گیری</h1>
            <p className="text-blue-100">تحلیل عملکرد مدل‌ها و نمودارهای پیشرفته</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-white/50"
            >
              <option value="7d">7 روز گذشته</option>
              <option value="30d">30 روز گذشته</option>
              <option value="90d">90 روز گذشته</option>
            </select>
            <button className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white hover:bg-white/20 transition-colors">
              <Download className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">میانگین دقت</p>
              <p className="text-2xl font-bold text-gray-900">89.4%</p>
              <div className="flex items-center mt-2">
                <TrendingUp className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-600 ml-1">+2.3%</span>
              </div>
            </div>
            <div className="bg-green-100 p-3 rounded-lg">
              <Target className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">آموزش‌های موفق</p>
              <p className="text-2xl font-bold text-gray-900">164</p>
              <div className="flex items-center mt-2">
                <TrendingUp className="w-4 h-4 text-blue-500" />
                <span className="text-sm text-blue-600 ml-1">+15</span>
              </div>
            </div>
            <div className="bg-blue-100 p-3 rounded-lg">
              <Award className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">میانگین سرعت</p>
              <p className="text-2xl font-bold text-gray-900">187 tokens/s</p>
              <div className="flex items-center mt-2">
                <TrendingDown className="w-4 h-4 text-red-500" />
                <span className="text-sm text-red-600 ml-1">-5.2%</span>
              </div>
            </div>
            <div className="bg-orange-100 p-3 rounded-lg">
              <Activity className="w-6 h-6 text-orange-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">زمان آموزش</p>
              <p className="text-2xl font-bold text-gray-900">4.2 ساعت</p>
              <div className="flex items-center mt-2">
                <TrendingUp className="w-4 h-4 text-purple-500" />
                <span className="text-sm text-purple-600 ml-1">-12 دقیقه</span>
              </div>
            </div>
            <div className="bg-purple-100 p-3 rounded-lg">
              <Clock className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900">روند عملکرد مدل‌ها</h2>
            <select 
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            >
              <option value="accuracy">دقت</option>
              <option value="loss">تابع هزینه</option>
              <option value="speed">سرعت</option>
            </select>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey={selectedMetric} 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6">مقایسه مدل‌ها</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis />
                <Radar name="مدل A" dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Radar name="مدل B" dataKey="B" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Training Success Rate & Resource Usage */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6">نرخ موفقیت آموزش</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={trainingTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="success" fill="#10b981" name="موفق" />
                <Bar dataKey="failed" fill="#ef4444" name="ناموفق" />
                <Bar dataKey="canceled" fill="#f59e0b" name="لغو شده" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6">استفاده از منابع سیستم</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Area 
                  type="monotone" 
                  dataKey="cpu" 
                  stackId="1" 
                  stroke="#3b82f6" 
                  fill="#3b82f6" 
                  name="CPU"
                />
                <Area 
                  type="monotone" 
                  dataKey="memory" 
                  stackId="1" 
                  stroke="#10b981" 
                  fill="#10b981" 
                  name="حافظه"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Model Performance Comparison */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">مقایسه تفصیلی مدل‌ها</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-right py-3 text-sm font-semibold text-gray-700">نام مدل</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">دقت (%)</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">سرعت (tokens/s)</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">مصرف حافظه (GB)</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">رتبه کلی</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">وضعیت</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {modelComparisonData.map((model, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="py-4 text-sm text-gray-900">{model.name}</td>
                  <td className="py-4">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 rounded-full h-2 ml-2">
                        <div 
                          className="bg-green-600 h-2 rounded-full"
                          style={{ width: `${model.accuracy}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-900">{model.accuracy}%</span>
                    </div>
                  </td>
                  <td className="py-4 text-sm text-gray-900">{model.speed}</td>
                  <td className="py-4 text-sm text-gray-900">{model.memory}</td>
                  <td className="py-4">
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <span 
                          key={i} 
                          className={`text-lg ${i < 4 - index ? 'text-yellow-400' : 'text-gray-300'}`}
                        >
                          ★
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="py-4">
                    <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-green-100 text-green-700">
                      فعال
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Export & Reports */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">صدور گزارش‌ها</h2>
          <button className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors">
            تولید گزارش کامل
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors text-right">
            <BarChart3 className="w-8 h-8 text-blue-600 mb-2" />
            <h3 className="font-semibold text-gray-900">گزارش عملکرد</h3>
            <p className="text-sm text-gray-600 mt-1">تحلیل کامل عملکرد مدل‌ها</p>
          </button>
          
          <button className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors text-right">
            <Activity className="w-8 h-8 text-green-600 mb-2" />
            <h3 className="font-semibold text-gray-900">گزارش منابع</h3>
            <p className="text-sm text-gray-600 mt-1">استفاده از منابع سیستم</p>
          </button>
          
          <button className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors text-right">
            <TrendingUp className="w-8 h-8 text-purple-600 mb-2" />
            <h3 className="font-semibold text-gray-900">گزارش روند</h3>
            <p className="text-sm text-gray-600 mt-1">تحلیل روند بهبود عملکرد</p>
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;