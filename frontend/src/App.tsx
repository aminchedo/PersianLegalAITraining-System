import React, { useState, useEffect } from 'react';
import { Search, Upload, BarChart3, Settings, FileText, Brain } from 'lucide-react';
import './App.css';

interface SystemHealth {
  status: string;
  database_connected: boolean;
  ai_model_loaded: boolean;
  version: string;
}

interface ClassificationResult {
  text: string;
  classification: Record<string, number>;
  confidence: number;
  predicted_class: string;
  timestamp: string;
}

export default function App() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [classificationText, setClassificationText] = useState('');
  const [classificationResult, setClassificationResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await fetch('/api/system/health');
      const data = await response.json();
      setSystemHealth(data);
    } catch (error) {
      console.error('Failed to check system health:', error);
    }
  };

  const handleClassification = async () => {
    if (!classificationText.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/ai/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: classificationText,
          include_confidence: true
        }),
      });
      
      const data = await response.json();
      setClassificationResult(data);
    } catch (error) {
      console.error('Classification failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'degraded': return 'text-yellow-500';
      default: return 'text-red-500';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100" dir="rtl">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3 space-x-reverse">
              <Brain className="h-8 w-8 text-indigo-600" />
              <h1 className="text-2xl font-bold text-gray-900">
                سامانه هوش مصنوعی حقوقی فارسی
              </h1>
            </div>
            
            {systemHealth && (
              <div className="flex items-center space-x-2 space-x-reverse">
                <div className={`w-3 h-3 rounded-full ${
                  systemHealth.status === 'healthy' ? 'bg-green-500' : 
                  systemHealth.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
                <span className={`text-sm font-medium ${getStatusColor(systemHealth.status)}`}>
                  {systemHealth.status.toUpperCase()}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-gray-900">پایگاه داده</h3>
                <p className="text-sm text-gray-500">
                  {systemHealth?.database_connected ? 'متصل' : 'قطع'}
                </p>
              </div>
              <FileText className={`h-8 w-8 ${
                systemHealth?.database_connected ? 'text-green-500' : 'text-red-500'
              }`} />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-gray-900">مدل هوش مصنوعی</h3>
                <p className="text-sm text-gray-500">
                  {systemHealth?.ai_model_loaded ? 'بارگذاری شده' : 'غیرفعال'}
                </p>
              </div>
              <Brain className={`h-8 w-8 ${
                systemHealth?.ai_model_loaded ? 'text-green-500' : 'text-red-500'
              }`} />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-gray-900">نسخه سیستم</h3>
                <p className="text-sm text-gray-500">
                  {systemHealth?.version || 'نامشخص'}
                </p>
              </div>
              <Settings className="h-8 w-8 text-indigo-500" />
            </div>
          </div>
        </div>

        {/* Classification Interface */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            طبقه‌بندی متن حقوقی
          </h2>
          
          <div className="space-y-4">
            <textarea
              value={classificationText}
              onChange={(e) => setClassificationText(e.target.value)}
              placeholder="متن حقوقی خود را برای طبقه‌بندی وارد کنید..."
              className="w-full h-32 p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
              dir="rtl"
            />
            
            <button
              onClick={handleClassification}
              disabled={loading || !classificationText.trim()}
              className="bg-indigo-600 text-white px-6 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 space-x-reverse"
            >
              {loading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              ) : (
                <Search className="h-4 w-4" />
              )}
              <span>{loading ? 'در حال پردازش...' : 'طبقه‌بندی'}</span>
            </button>
          </div>

          {/* Classification Results */}
          {classificationResult && (
            <div className="mt-6 p-4 bg-gray-50 rounded-md">
              <h3 className="text-lg font-medium text-gray-900 mb-3">نتایج طبقه‌بندی</h3>
              
              <div className="space-y-2">
                <p className="text-sm text-gray-600">
                  <strong>دسته پیش‌بینی شده:</strong> {classificationResult.predicted_class}
                </p>
                <p className="text-sm text-gray-600">
                  <strong>اطمینان:</strong> {(classificationResult.confidence * 100).toFixed(1)}%
                </p>
                
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">توزیع احتمالات:</h4>
                  <div className="space-y-1">
                    {Object.entries(classificationResult.classification).map(([category, score]) => (
                      <div key={category} className="flex items-center space-x-2 space-x-reverse">
                        <span className="text-sm w-20 text-gray-600">{category}:</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-indigo-600 h-2 rounded-full"
                            style={{ width: `${score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-600">
                          {(score * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer">
            <Upload className="h-8 w-8 text-indigo-600 mb-3" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">آپلود اسناد</h3>
            <p className="text-sm text-gray-500">بارگذاری و پردازش اسناد حقوقی جدید</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer">
            <Search className="h-8 w-8 text-indigo-600 mb-3" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">جستجوی اسناد</h3>
            <p className="text-sm text-gray-500">جستجوی پیشرفته در بانک اطلاعات حقوقی</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer">
            <BarChart3 className="h-8 w-8 text-indigo-600 mb-3" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">آمار و گزارشات</h3>
            <p className="text-sm text-gray-500">مشاهده آمار و گزارش‌های سیستم</p>
          </div>
        </div>
      </main>
    </div>
  );
}