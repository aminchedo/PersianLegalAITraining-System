import React, { useState } from 'react';
import { Zap, Download, Upload, Settings, Play, Pause, Square, Eye, Edit, Trash2, Copy, MoreVertical, Star, TrendingUp } from 'lucide-react';
import { usePersianAI } from '../hooks/usePersianAI';

const ModelsPage: React.FC = () => {
  const { state } = usePersianAI();
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [showDetails, setShowDetails] = useState<string>('');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const modelCategories = [
    { id: 'llm', name: 'مدل‌های زبانی', count: 3, color: 'bg-blue-100 text-blue-700' },
    { id: 'classification', name: 'طبقه‌بندی', count: 2, color: 'bg-green-100 text-green-700' },
    { id: 'embedding', name: 'Embedding', count: 1, color: 'bg-purple-100 text-purple-700' },
  ];

  const mockModels = [
    {
      id: '1',
      name: 'Persian Legal LLM v2.0',
      type: 'llm',
      status: 'deployed',
      accuracy: 0.91,
      size: '7B',
      version: '2.0.1',
      downloads: 1250,
      created_at: '2024-01-20',
      description: 'مدل پیشرفته زبان فارسی برای متون حقوقی',
      performance: {
        speed: 185,
        memory: 2.1,
        cost: 0.02,
      }
    },
    {
      id: '2', 
      name: 'Contract Analyzer Pro',
      type: 'classification',
      status: 'training',
      accuracy: 0.87,
      size: '1.2B',
      version: '1.5.0',
      downloads: 890,
      created_at: '2024-01-18',
      description: 'تحلیل و طبقه‌بندی قراردادها',
      performance: {
        speed: 312,
        memory: 1.8,
        cost: 0.015,
      }
    },
    {
      id: '3',
      name: 'Legal Document Embedder',
      type: 'embedding',
      status: 'deployed',
      accuracy: 0.93,
      size: '400M',
      version: '3.1.2',
      downloads: 2150,
      created_at: '2024-01-15',
      description: 'تبدیل اسناد حقوقی به بردار',
      performance: {
        speed: 450,
        memory: 0.9,
        cost: 0.008,
      }
    }
  ];

  const filteredModels = mockModels.filter(model => 
    filterStatus === 'all' || model.status === filterStatus
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'bg-green-100 text-green-700';
      case 'training': return 'bg-blue-100 text-blue-700';
      case 'error': return 'bg-red-100 text-red-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'deployed': return 'آماده';
      case 'training': return 'در حال آموزش';
      case 'error': return 'خطا';
      default: return 'در انتظار';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">مدیریت مدل‌های هوش مصنوعی</h1>
            <p className="text-purple-100">مشاهده، مدیریت و استقرار مدل‌های آموزش‌دیده</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <button className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white hover:bg-white/20 transition-colors">
              <Upload className="w-5 h-5" />
            </button>
            <button className="bg-white text-purple-600 px-6 py-2 rounded-lg hover:bg-gray-100 transition-colors font-semibold">
              مدل جدید
            </button>
          </div>
        </div>
      </div>

      {/* Model Categories */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {modelCategories.map((category) => (
          <div key={category.id} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">{category.name}</h3>
                <p className="text-2xl font-bold text-gray-900">{category.count}</p>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${category.color}`}>
                فعال
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 space-x-reverse">
            <h2 className="text-xl font-bold text-gray-900">مدل‌های موجود</h2>
            <select 
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            >
              <option value="all">همه وضعیت‌ها</option>
              <option value="deployed">آماده</option>
              <option value="training">در حال آموزش</option>
              <option value="error">خطا</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2 space-x-reverse">
            <button className="text-gray-600 hover:text-gray-900 transition-colors">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredModels.map((model) => (
          <div key={model.id} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3 space-x-reverse">
                <div className="bg-purple-100 p-2 rounded-lg">
                  <Zap className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{model.name}</h3>
                  <p className="text-sm text-gray-600">v{model.version}</p>
                </div>
              </div>
              
              <div className="relative">
                <button className="text-gray-600 hover:text-gray-900 transition-colors">
                  <MoreVertical className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="space-y-3 mb-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">وضعیت</span>
                <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(model.status)}`}>
                  {getStatusText(model.status)}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">دقت</span>
                <span className="text-sm font-medium text-gray-900">{(model.accuracy * 100).toFixed(1)}%</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">اندازه</span>
                <span className="text-sm font-medium text-gray-900">{model.size}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">دانلود</span>
                <div className="flex items-center space-x-1 space-x-reverse">
                  <Download className="w-3 h-3 text-gray-500" />
                  <span className="text-sm font-medium text-gray-900">{model.downloads.toLocaleString()}</span>
                </div>
              </div>
            </div>

            <div className="border-t border-gray-200 pt-4">
              <p className="text-sm text-gray-600 mb-4">{model.description}</p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 space-x-reverse">
                  <button 
                    onClick={() => setShowDetails(showDetails === model.id ? '' : model.id)}
                    className="text-purple-600 hover:text-purple-700 transition-colors text-sm"
                  >
                    <Eye className="w-4 h-4 ml-1 inline" />
                    جزئیات
                  </button>
                  
                  {model.status === 'deployed' && (
                    <button className="bg-green-600 text-white px-3 py-1 text-sm rounded hover:bg-green-700 transition-colors">
                      <Play className="w-3 h-3 ml-1 inline" />
                      اجرا
                    </button>
                  )}
                  
                  {model.status === 'training' && (
                    <button className="bg-yellow-600 text-white px-3 py-1 text-sm rounded hover:bg-yellow-700 transition-colors">
                      <Pause className="w-3 h-3 ml-1 inline" />
                      توقف
                    </button>
                  )}
                </div>
                
                <div className="flex items-center space-x-1 space-x-reverse">
                  <button className="text-gray-600 hover:text-gray-900 transition-colors">
                    <Star className="w-4 h-4" />
                  </button>
                  <button className="text-gray-600 hover:text-gray-900 transition-colors">
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Model Details Expansion */}
            {showDetails === model.id && (
              <div className="mt-4 pt-4 border-t border-gray-200 space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-gray-600">سرعت</p>
                    <p className="text-sm font-medium text-gray-900">{model.performance.speed} tokens/s</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">مصرف حافظه</p>
                    <p className="text-sm font-medium text-gray-900">{model.performance.memory} GB</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">هزینه</p>
                    <p className="text-sm font-medium text-gray-900">${model.performance.cost}/1K tokens</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">تاریخ ایجاد</p>
                    <p className="text-sm font-medium text-gray-900">
                      {new Date(model.created_at).toLocaleDateString('fa-IR')}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2 space-x-reverse pt-2">
                  <button className="bg-purple-600 text-white px-3 py-1 text-sm rounded hover:bg-purple-700 transition-colors">
                    <Edit className="w-3 h-3 ml-1 inline" />
                    ویرایش
                  </button>
                  <button className="border border-gray-300 text-gray-700 px-3 py-1 text-sm rounded hover:bg-gray-50 transition-colors">
                    <Copy className="w-3 h-3 ml-1 inline" />
                    کپی
                  </button>
                  <button className="text-red-600 hover:text-red-700 transition-colors px-3 py-1 text-sm">
                    <Trash2 className="w-3 h-3 ml-1 inline" />
                    حذف
                  </button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Model Performance Comparison */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">مقایسه عملکرد مدل‌ها</h2>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-right py-3 text-sm font-semibold text-gray-700">نام مدل</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">نوع</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">دقت</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">سرعت</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">حافظه</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">هزینه</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">امتیاز</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">عملیات</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {mockModels.map((model) => (
                <tr key={model.id} className="hover:bg-gray-50">
                  <td className="py-4">
                    <div className="flex items-center space-x-3 space-x-reverse">
                      <div className="bg-purple-100 p-1 rounded">
                        <Zap className="w-4 h-4 text-purple-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">{model.name}</p>
                        <p className="text-xs text-gray-600">v{model.version}</p>
                      </div>
                    </div>
                  </td>
                  <td className="py-4">
                    <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-700">
                      {model.type === 'llm' ? 'زبانی' : 
                       model.type === 'classification' ? 'طبقه‌بندی' : 'Embedding'}
                    </span>
                  </td>
                  <td className="py-4">
                    <div className="flex items-center space-x-2 space-x-reverse">
                      <div className="w-12 bg-gray-200 rounded-full h-1.5">
                        <div 
                          className="bg-green-600 h-1.5 rounded-full"
                          style={{ width: `${model.accuracy * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-900">{(model.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="py-4 text-sm text-gray-900">{model.performance.speed} t/s</td>
                  <td className="py-4 text-sm text-gray-900">{model.performance.memory} GB</td>
                  <td className="py-4 text-sm text-gray-900">${model.performance.cost}/1K</td>
                  <td className="py-4">
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <Star 
                          key={i} 
                          className={`w-3 h-3 ${i < 4 ? 'text-yellow-400 fill-current' : 'text-gray-300'}`}
                        />
                      ))}
                    </div>
                  </td>
                  <td className="py-4">
                    <div className="flex items-center space-x-2 space-x-reverse">
                      <button className="text-purple-600 hover:text-purple-700 transition-colors">
                        <Eye className="w-4 h-4" />
                      </button>
                      <button className="text-blue-600 hover:text-blue-700 transition-colors">
                        <Download className="w-4 h-4" />
                      </button>
                      <button className="text-green-600 hover:text-green-700 transition-colors">
                        <Play className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">عملیات سریع</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors text-right">
            <Upload className="w-8 h-8 text-blue-600 mb-2" />
            <h3 className="font-semibold text-gray-900">بارگذاری مدل</h3>
            <p className="text-sm text-gray-600 mt-1">مدل آموزش‌دیده خود را بارگذاری کنید</p>
          </button>
          
          <button className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors text-right">
            <TrendingUp className="w-8 h-8 text-green-600 mb-2" />
            <h3 className="font-semibold text-gray-900">بهینه‌سازی</h3>
            <p className="text-sm text-gray-600 mt-1">عملکرد مدل‌ها را بهینه کنید</p>
          </button>
          
          <button className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors text-right">
            <Download className="w-8 h-8 text-purple-600 mb-2" />
            <h3 className="font-semibold text-gray-900">صدور مدل</h3>
            <p className="text-sm text-gray-600 mt-1">مدل‌ها را برای استقرار آماده کنید</p>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelsPage;