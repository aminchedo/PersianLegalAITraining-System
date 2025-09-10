import React, { useState } from 'react';
import { Upload, Download, RefreshCw, Database, Globe, FileText, CheckCircle, AlertCircle, Clock, Search, Filter, Eye, Plus } from 'lucide-react';
import { usePersianAI } from '../../../hooks/usePersianAI';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const DataPage: React.FC = () => {
  const { state } = usePersianAI();
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');

  const dataQualityData = [
    { name: 'عالی', value: 45, color: '#10b981' },
    { name: 'خوب', value: 32, color: '#3b82f6' },
    { name: 'متوسط', value: 18, color: '#f59e0b' },
    { name: 'ضعیف', value: 5, color: '#ef4444' },
  ];

  const dataVolumeData = [
    { month: 'فروردین', legal_docs: 1200, case_studies: 800, regulations: 500 },
    { month: 'اردیبهشت', legal_docs: 1500, case_studies: 950, regulations: 620 },
    { month: 'خرداد', legal_docs: 1800, case_studies: 1100, regulations: 750 },
    { month: 'تیر', legal_docs: 2100, case_studies: 1300, regulations: 880 },
    { month: 'مرداد', legal_docs: 2400, case_studies: 1450, regulations: 920 },
    { month: 'شهریور', legal_docs: 2200, case_studies: 1520, regulations: 1100 },
  ];

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setIsUploading(true);
      setUploadProgress(0);
      
      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsUploading(false);
            return 100;
          }
          return prev + 10;
        });
      }, 200);
    }
  };

  const filteredDataSources = state.dataSources.filter(source => {
    const matchesSearch = source.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || source.type === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">مدیریت منابع داده</h1>
            <p className="text-green-100">مدیریت، بارگذاری و نظارت بر کیفیت منابع داده</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <button className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white hover:bg-white/20 transition-colors">
              <RefreshCw className="w-5 h-5" />
            </button>
            <button className="bg-white text-green-600 px-6 py-2 rounded-lg hover:bg-gray-100 transition-colors font-semibold">
              <Plus className="w-5 h-5 ml-2 inline" />
              افزودن منبع جدید
            </button>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-lg ml-4">
              <Database className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">کل رکوردها</p>
              <p className="text-2xl font-bold text-gray-900">
                {state.dataSources.reduce((sum, source) => sum + source.records_count, 0).toLocaleString()}
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
              <p className="text-sm font-medium text-gray-600">منابع فعال</p>
              <p className="text-2xl font-bold text-gray-900">
                {state.dataSources.filter(s => s.status === 'connected').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-yellow-100 p-3 rounded-lg ml-4">
              <Clock className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">در حال جمع‌آوری</p>
              <p className="text-2xl font-bold text-gray-900">
                {state.dataSources.filter(s => s.status === 'collecting').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center">
            <div className="bg-purple-100 p-3 rounded-lg ml-4">
              <Globe className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">میانگین کیفیت</p>
              <p className="text-2xl font-bold text-gray-900">
                {(state.dataSources.reduce((sum, source) => sum + source.quality_score, 0) / state.dataSources.length * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Data Upload Section */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">بارگذاری داده‌های جدید</h2>
        
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">فایل‌های خود را اینجا رها کنید</h3>
          <p className="text-gray-600 mb-4">یا روی دکمه زیر کلیک کنید تا فایل‌ها را انتخاب کنید</p>
          
          <input
            type="file"
            multiple
            accept=".json,.csv,.txt,.pdf"
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors cursor-pointer inline-block"
          >
            انتخاب فایل‌ها
          </label>
          
          {isUploading && (
            <div className="mt-4">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-600 mt-2">در حال بارگذاری... {uploadProgress}%</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <FileText className="w-8 h-8 text-blue-600 mb-2" />
            <h4 className="font-semibold text-blue-900">اسناد حقوقی</h4>
            <p className="text-sm text-blue-700 mt-1">فرمت‌های PDF, DOC, TXT</p>
          </div>
          
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <Database className="w-8 h-8 text-green-600 mb-2" />
            <h4 className="font-semibold text-green-900">داده‌های ساختاری</h4>
            <p className="text-sm text-green-700 mt-1">فرمت‌های JSON, CSV, XML</p>
          </div>
          
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <Globe className="w-8 h-8 text-purple-600 mb-2" />
            <h4 className="font-semibold text-purple-900">منابع آنلاین</h4>
            <p className="text-sm text-purple-700 mt-1">API و وب‌سایت‌ها</p>
          </div>
        </div>
      </div>

      {/* Data Sources Management */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900">منابع داده</h2>
              <div className="flex items-center space-x-4 space-x-reverse">
                <div className="relative">
                  <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <input
                    type="text"
                    placeholder="جستجو..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-4 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <select 
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="all">همه انواع</option>
                  <option value="legal_docs">اسناد حقوقی</option>
                  <option value="case_studies">مطالعات موردی</option>
                  <option value="regulations">مقررات</option>
                  <option value="custom">سفارشی</option>
                </select>
              </div>
            </div>

            <div className="space-y-4">
              {filteredDataSources.map((source) => (
                <div key={source.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3 space-x-reverse">
                      <div className={`p-2 rounded-lg ${
                        source.type === 'legal_docs' ? 'bg-blue-100' :
                        source.type === 'case_studies' ? 'bg-green-100' :
                        source.type === 'regulations' ? 'bg-purple-100' :
                        'bg-gray-100'
                      }`}>
                        {source.type === 'legal_docs' ? <FileText className="w-5 h-5 text-blue-600" /> :
                         source.type === 'case_studies' ? <Database className="w-5 h-5 text-green-600" /> :
                         source.type === 'regulations' ? <Globe className="w-5 h-5 text-purple-600" /> :
                         <Database className="w-5 h-5 text-gray-600" />}
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900">{source.name}</h3>
                        <p className="text-sm text-gray-600">{source.records_count.toLocaleString()} رکورد</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-3 space-x-reverse">
                      <div className={`flex items-center space-x-1 space-x-reverse text-xs px-2 py-1 rounded-full ${
                        source.status === 'connected' ? 'bg-green-100 text-green-700' :
                        source.status === 'collecting' ? 'bg-blue-100 text-blue-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {source.status === 'connected' ? <CheckCircle className="w-3 h-3" /> :
                         source.status === 'collecting' ? <Clock className="w-3 h-3" /> :
                         <AlertCircle className="w-3 h-3" />}
                        <span>
                          {source.status === 'connected' ? 'متصل' :
                           source.status === 'collecting' ? 'در حال جمع‌آوری' : 'قطع شده'}
                        </span>
                      </div>
                      
                      <button className="text-gray-600 hover:text-gray-900 transition-colors">
                        <Eye className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 mt-3 pt-3 border-t border-gray-200">
                    <div>
                      <p className="text-xs text-gray-600">کیفیت داده</p>
                      <div className="flex items-center mt-1">
                        <div className="w-16 bg-gray-200 rounded-full h-2 ml-2">
                          <div 
                            className={`h-2 rounded-full ${
                              source.quality_score > 0.8 ? 'bg-green-600' :
                              source.quality_score > 0.6 ? 'bg-yellow-600' :
                              'bg-red-600'
                            }`}
                            style={{ width: `${source.quality_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-xs font-medium text-gray-900">
                          {(source.quality_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-xs text-gray-600">آخرین بروزرسانی</p>
                      <p className="text-xs text-gray-900 mt-1">
                        {new Date(source.last_updated).toLocaleDateString('fa-IR')}
                      </p>
                    </div>
                    
                    <div>
                      <p className="text-xs text-gray-600">عملیات</p>
                      <div className="flex items-center space-x-2 space-x-reverse mt-1">
                        <button className="bg-blue-600 text-white px-2 py-1 text-xs rounded hover:bg-blue-700 transition-colors">
                          همگام‌سازی
                        </button>
                        <button className="text-gray-600 hover:text-gray-900 transition-colors">
                          <Download className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Data Quality & Analytics */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">توزیع کیفیت داده</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={dataQualityData}
                    cx="50%"
                    cy="50%"
                    innerRadius={30}
                    outerRadius={70}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {dataQualityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-2 mt-4">
              {dataQualityData.map((item, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span className="text-sm text-gray-600">{item.name}</span>
                  </div>
                  <span className="text-sm font-semibold text-gray-900">{item.value}%</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">پیشنهادات بهبود</h3>
            <div className="space-y-3">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                <div className="flex items-center space-x-2 space-x-reverse mb-2">
                  <AlertCircle className="w-4 h-4 text-yellow-600" />
                  <p className="text-sm font-medium text-yellow-800">نیاز به تمیزسازی</p>
                </div>
                <p className="text-xs text-yellow-700">
                  منبع "آرای دیوان" حاوی داده‌های ناقص است
                </p>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-center space-x-2 space-x-reverse mb-2">
                  <Database className="w-4 h-4 text-blue-600" />
                  <p className="text-sm font-medium text-blue-800">افزودن منبع</p>
                </div>
                <p className="text-xs text-blue-700">
                  اضافه کردن قوانین تجاری پیشنهاد می‌شود
                </p>
              </div>

              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <div className="flex items-center space-x-2 space-x-reverse mb-2">
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  <p className="text-sm font-medium text-green-800">به‌روزرسانی</p>
                </div>
                <p className="text-xs text-green-700">
                  همه منابع به‌روز و آماده استفاده هستند
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Data Volume Trends */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">روند حجم داده‌ها</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={dataVolumeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="legal_docs" fill="#3b82f6" name="اسناد حقوقی" />
              <Bar dataKey="case_studies" fill="#10b981" name="مطالعات موردی" />
              <Bar dataKey="regulations" fill="#8b5cf6" name="مقررات" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default DataPage;