import React, { useState, useEffect } from 'react';
import { 
  Database, Plus, X, Edit3, Trash2, RefreshCw, Download, Upload, 
  Search, Filter, Eye, Settings, Play, Pause, AlertTriangle, 
  CheckCircle, Clock, Activity, BarChart3, FileText, Globe,
  TrendingUp, TrendingDown, Server, Zap, Shield
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { useAppContext } from './Router';

const DataPage = () => {
  const { dataSources, realTimeData } = useAppContext();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');
  const [selectedSource, setSelectedSource] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [viewMode, setViewMode] = useState('grid');
  const [refreshing, setRefreshing] = useState(false);

  const [newSource, setNewSource] = useState({
    name: '',
    type: 'corpus',
    description: '',
    url: '',
    apiKey: '',
    syncInterval: 60
  });

  const [collectionStats, setCollectionStats] = useState({
    totalDocuments: 33266,
    totalSize: '45.6 GB',
    dailyGrowth: 1247,
    qualityScore: 89.2,
    activeConnections: 3,
    errorRate: 0.02
  });

  // Filter data sources
  const filteredSources = dataSources.filter(source => {
    const matchesSearch = source.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         source.description?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || source.status === filterStatus;
    const matchesType = filterType === 'all' || source.type === filterType;
    return matchesSearch && matchesStatus && matchesType;
  });

  // Collection performance data
  const collectionData = realTimeData.slice(-24).map((point, index) => ({
    hour: `${23 - index}:00`,
    documents: Math.floor(Math.random() * 200) + 50,
    speed: Math.floor(Math.random() * 150) + 50,
    quality: 85 + Math.random() * 10,
    errors: Math.floor(Math.random() * 5)
  })).reverse();

  // Quality distribution data
  const qualityData = [
    { name: 'عالی', value: 15000, fill: '#10B981' },
    { name: 'خوب', value: 12000, fill: '#3B82F6' },
    { name: 'متوسط', value: 5000, fill: '#F59E0B' },
    { name: 'ضعیف', value: 1266, fill: '#EF4444' }
  ];

  // Type distribution data
  const typeData = [
    { name: 'پیکره متنی', value: 18000, documents: 18000 },
    { name: 'اسناد حقوقی', value: 9000, documents: 9000 },
    { name: 'قوانین', value: 4000, documents: 4000 },
    { name: 'مقررات', value: 2266, documents: 2266 }
  ];

  // Status configurations
  const getStatusConfig = (status) => {
    const configs = {
      active: { color: 'bg-green-500', textColor: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle },
      inactive: { color: 'bg-gray-500', textColor: 'text-gray-700', bgColor: 'bg-gray-100', icon: Clock },
      syncing: { color: 'bg-blue-500', textColor: 'text-blue-700', bgColor: 'bg-blue-100', icon: RefreshCw },
      error: { color: 'bg-red-500', textColor: 'text-red-700', bgColor: 'bg-red-100', icon: AlertTriangle }
    };
    return configs[status] || configs.inactive;
  };

  const getStatusText = (status) => {
    const texts = {
      active: 'فعال',
      inactive: 'غیرفعال',
      syncing: 'در حال همگام‌سازی',
      error: 'خطا'
    };
    return texts[status] || status;
  };

  // Type configurations
  const getTypeConfig = (type) => {
    const configs = {
      corpus: { label: 'پیکره متنی', icon: FileText, color: 'purple' },
      'legal-portal': { label: 'پورتال حقوقی', icon: Globe, color: 'blue' },
      parliament: { label: 'مجلس', icon: FileText, color: 'green' },
      'data-portal': { label: 'پورتال داده', icon: Database, color: 'orange' }
    };
    return configs[type] || configs.corpus;
  };

  // Handle refresh
  const handleRefresh = async () => {
    setRefreshing(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setRefreshing(false);
  };

  // Handle add new source
  const handleAddSource = () => {
    const source = {
      id: Date.now(),
      ...newSource,
      documents: 0,
      quality: 0,
      status: 'inactive',
      speed: 0,
      lastSync: new Date()
    };
    
    // In real app, this would update the global state
    console.log('Adding new source:', source);
    
    setNewSource({
      name: '',
      type: 'corpus',
      description: '',
      url: '',
      apiKey: '',
      syncInterval: 60
    });
    setShowAddModal(false);
  };

  // Handle source actions
  const handleSourceAction = (action, sourceId) => {
    console.log(`Action ${action} on source ${sourceId}`);
    // In real app, this would update the source status
  };

  return (
    <div className="space-y-6" dir="rtl">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 rounded-2xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <Database className="w-8 h-8" />
              مدیریت منابع داده
            </h1>
            <p className="text-blue-100">جمع‌آوری، پردازش و مدیریت منابع داده‌های آموزشی</p>
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={handleRefresh}
              disabled={refreshing}
              className="bg-white/20 text-white px-4 py-2 rounded-xl hover:bg-white/30 transition-all flex items-center gap-2 backdrop-blur-sm"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              بروزرسانی
            </button>
            <button 
              onClick={() => setShowAddModal(true)}
              className="bg-white text-blue-600 px-6 py-3 rounded-xl font-medium hover:bg-blue-50 transition-all duration-300 flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              منبع جدید
            </button>
          </div>
        </div>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <FileText className="w-4 h-4" />
              <span className="text-sm">کل اسناد</span>
            </div>
            <p className="text-2xl font-bold">{collectionStats.totalDocuments.toLocaleString('fa-IR')}</p>
            <p className="text-xs opacity-75">حجم: {collectionStats.totalSize}</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm">رشد روزانه</span>
            </div>
            <p className="text-2xl font-bold">{collectionStats.dailyGrowth.toLocaleString('fa-IR')}</p>
            <p className="text-xs opacity-75">سند جدید</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Shield className="w-4 h-4" />
              <span className="text-sm">کیفیت کلی</span>
            </div>
            <p className="text-2xl font-bold">{collectionStats.qualityScore}%</p>
            <p className="text-xs opacity-75">میانگین وزنی</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4" />
              <span className="text-sm">اتصالات فعال</span>
            </div>
            <p className="text-2xl font-bold">{collectionStats.activeConnections}</p>
            <p className="text-xs opacity-75">از {dataSources.length} منبع</p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="w-5 h-5 text-gray-400 absolute right-3 top-1/2 transform -translate-y-1/2" />
              <input 
                type="text"
                placeholder="جستجو در منابع داده..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-64 pl-4 pr-10 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <select 
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">همه وضعیت‌ها</option>
              <option value="active">فعال</option>
              <option value="inactive">غیرفعال</option>
              <option value="syncing">در حال همگام‌سازی</option>
              <option value="error">خطا</option>
            </select>
            <select 
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">همه انواع</option>
              <option value="corpus">پیکره متنی</option>
              <option value="legal-portal">پورتال حقوقی</option>
              <option value="parliament">مجلس</option>
              <option value="data-portal">پورتال داده</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-lg transition-all ${viewMode === 'grid' ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:bg-gray-100'}`}
            >
              <BarChart3 className="w-4 h-4" />
            </button>
            <button 
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-lg transition-all ${viewMode === 'list' ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:bg-gray-100'}`}
            >
              <Database className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Data Sources Grid/List */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredSources.map(source => {
              const statusConfig = getStatusConfig(source.status);
              const typeConfig = getTypeConfig(source.type);
              const StatusIcon = statusConfig.icon;
              const TypeIcon = typeConfig.icon;
              
              return (
                <div key={source.id} className="group relative">
                  <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl p-6 border border-gray-200 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center bg-${typeConfig.color}-100`}>
                        <TypeIcon className={`w-6 h-6 text-${typeConfig.color}-600`} />
                      </div>
                      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button 
                          onClick={() => setSelectedSource(source)}
                          className="w-8 h-8 bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 transition-all"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="w-8 h-8 bg-gray-600 text-white rounded-lg flex items-center justify-center hover:bg-gray-700 transition-all">
                          <Settings className="w-4 h-4" />
                        </button>
                        <button className="w-8 h-8 bg-red-600 text-white rounded-lg flex items-center justify-center hover:bg-red-700 transition-all">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {/* Content */}
                    <div>
                      <h3 className="font-bold text-gray-900 mb-2">{source.name}</h3>
                      <p className="text-sm text-gray-600 mb-3">{source.description || typeConfig.label}</p>
                      
                      {/* Stats */}
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <p className="text-xs text-gray-500">اسناد</p>
                          <p className="font-semibold">{source.documents.toLocaleString('fa-IR')}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">کیفیت</p>
                          <p className="font-semibold">{source.quality}%</p>
                        </div>
                      </div>

                      {/* Quality Bar */}
                      <div className="mb-4">
                        <div className="flex justify-between text-xs mb-1">
                          <span>کیفیت داده</span>
                          <span>{source.quality}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full transition-all duration-500 ${
                              source.quality >= 90 ? 'bg-green-500' :
                              source.quality >= 80 ? 'bg-blue-500' :
                              source.quality >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${source.quality}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* Status and Speed */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <StatusIcon className={`w-4 h-4 ${statusConfig.textColor}`} />
                          <span className={`text-xs font-medium ${statusConfig.textColor}`}>
                            {getStatusText(source.status)}
                          </span>
                        </div>
                        {source.status === 'active' && (
                          <span className="text-xs text-gray-500">{source.speed}/ساعت</span>
                        )}
                      </div>

                      {/* Action Buttons */}
                      <div className="flex gap-2 mt-4">
                        {source.status === 'active' ? (
                          <button 
                            onClick={() => handleSourceAction('pause', source.id)}
                            className="flex-1 bg-yellow-100 text-yellow-700 py-2 px-3 rounded-lg text-sm hover:bg-yellow-200 transition-all flex items-center justify-center gap-2"
                          >
                            <Pause className="w-4 h-4" />
                            توقف
                          </button>
                        ) : source.status === 'inactive' ? (
                          <button 
                            onClick={() => handleSourceAction('start', source.id)}
                            className="flex-1 bg-green-100 text-green-700 py-2 px-3 rounded-lg text-sm hover:bg-green-200 transition-all flex items-center justify-center gap-2"
                          >
                            <Play className="w-4 h-4" />
                            شروع
                          </button>
                        ) : null}
                        
                        <button 
                          onClick={() => setSelectedSource(source)}
                          className="flex-1 bg-blue-100 text-blue-700 py-2 px-3 rounded-lg text-sm hover:bg-blue-200 transition-all flex items-center justify-center gap-2"
                        >
                          <Eye className="w-4 h-4" />
                          مشاهده
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="space-y-4">
            {filteredSources.map(source => {
              const statusConfig = getStatusConfig(source.status);
              const typeConfig = getTypeConfig(source.type);
              const StatusIcon = statusConfig.icon;
              const TypeIcon = typeConfig.icon;
              
              return (
                <div key={source.id} className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-all">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center bg-${typeConfig.color}-100`}>
                        <TypeIcon className={`w-6 h-6 text-${typeConfig.color}-600`} />
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-900">{source.name}</h3>
                        <p className="text-sm text-gray-600">{source.description || typeConfig.label}</p>
                        <div className="flex items-center gap-4 text-sm text-gray-500 mt-1">
                          <span>اسناد: {source.documents.toLocaleString('fa-IR')}</span>
                          <span>•</span>
                          <span>کیفیت: {source.quality}%</span>
                          {source.status === 'active' && (
                            <>
                              <span>•</span>
                              <span>سرعت: {source.speed}/ساعت</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="w-32 bg-gray-200 rounded-full h-2 mb-1">
                          <div 
                            className={`h-2 rounded-full transition-all duration-500 ${
                              source.quality >= 90 ? 'bg-green-500' :
                              source.quality >= 80 ? 'bg-blue-500' :
                              source.quality >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${source.quality}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-gray-500">کیفیت: {source.quality}%</span>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <StatusIcon className={`w-4 h-4 ${statusConfig.textColor}`} />
                        <span className={`text-xs font-medium ${statusConfig.textColor}`}>
                          {getStatusText(source.status)}
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => setSelectedSource(source)}
                          className="w-8 h-8 bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 transition-all"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="w-8 h-8 bg-gray-600 text-white rounded-lg flex items-center justify-center hover:bg-gray-700 transition-all">
                          <Settings className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Analytics Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Collection Performance */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-6">عملکرد جمع‌آوری داده</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={collectionData}>
              <defs>
                <linearGradient id="documentsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                </linearGradient>
                <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="hour" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip />
              <Area type="monotone" dataKey="documents" stroke="#3B82F6" fill="url(#documentsGradient)" strokeWidth={2} />
              <Area type="monotone" dataKey="speed" stroke="#10B981" fill="url(#speedGradient)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>اسناد جمع‌آوری شده</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>سرعت جمع‌آوری</span>
            </div>
          </div>
        </div>

        {/* Data Quality Distribution */}
        <div className="bg-white rounded-2xl p-6 shadow-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-6">توزیع کیفیت داده</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={qualityData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={5}
                dataKey="value"
              >
                {qualityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-4 mt-4">
            {qualityData.map((item, index) => (
              <div key={index} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.fill }}></div>
                <span className="text-sm">{item.name}: {item.value.toLocaleString('fa-IR')}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Data Type Analysis */}
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">تحلیل انواع داده</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={typeData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
            <YAxis stroke="#6b7280" fontSize={12} />
            <Tooltip />
            <Bar dataKey="documents" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Add New Source Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-2xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">افزودن منبع داده جدید</h3>
              <button 
                onClick={() => setShowAddModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">نام منبع</label>
                <input 
                  type="text"
                  value={newSource.name}
                  onChange={(e) => setNewSource(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  placeholder="نام منبع داده..."
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">نوع منبع</label>
                  <select 
                    value={newSource.type}
                    onChange={(e) => setNewSource(prev => ({ ...prev, type: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="corpus">پیکره متنی</option>
                    <option value="legal-portal">پورتال حقوقی</option>
                    <option value="parliament">مجلس</option>
                    <option value="data-portal">پورتال داده</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">فاصله همگام‌سازی (دقیقه)</label>
                  <input 
                    type="number"
                    value={newSource.syncInterval}
                    onChange={(e) => setNewSource(prev => ({ ...prev, syncInterval: parseInt(e.target.value) }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                    min="5"
                    max="1440"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">آدرس URL</label>
                <input 
                  type="url"
                  value={newSource.url}
                  onChange={(e) => setNewSource(prev => ({ ...prev, url: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  placeholder="https://example.com/api/data"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">کلید API (اختیاری)</label>
                <input 
                  type="password"
                  value={newSource.apiKey}
                  onChange={(e) => setNewSource(prev => ({ ...prev, apiKey: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  placeholder="کلید API..."
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">توضیحات</label>
                <textarea 
                  value={newSource.description}
                  onChange={(e) => setNewSource(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="توضیحات منبع داده..."
                />
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button 
                onClick={handleAddSource}
                className="flex-1 bg-blue-600 text-white py-3 rounded-xl hover:bg-blue-700 transition-all font-medium"
              >
                افزودن منبع
              </button>
              <button 
                onClick={() => setShowAddModal(false)}
                className="flex-1 bg-gray-300 text-gray-700 py-3 rounded-xl hover:bg-gray-400 transition-all font-medium"
              >
                لغو
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Source Detail Modal */}
      {selectedSource && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-auto p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">جزئیات منبع: {selectedSource.name}</h3>
              <button 
                onClick={() => setSelectedSource(null)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Source Info */}
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">اطلاعات منبع</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>نوع:</span>
                      <span>{getTypeConfig(selectedSource.type).label}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>وضعیت:</span>
                      <span>{getStatusText(selectedSource.status)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>آخرین همگام‌سازی:</span>
                      <span>{new Date(selectedSource.lastSync).toLocaleString('fa-IR')}</span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">آمار عملکرد</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>تعداد اسناد:</span>
                      <span className="font-semibold">{selectedSource.documents.toLocaleString('fa-IR')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>کیفیت:</span>
                      <span className="font-semibold text-green-600">{selectedSource.quality}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>سرعت جمع‌آوری:</span>
                      <span>{selectedSource.speed}/ساعت</span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Performance Chart */}
              <div>
                <h4 className="font-semibold mb-3">نمودار عملکرد</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={collectionData.slice(-12)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="hour" stroke="#6b7280" fontSize={12} />
                    <YAxis stroke="#6b7280" fontSize={12} />
                    <Tooltip />
                    <Line type="monotone" dataKey="documents" stroke="#3B82F6" strokeWidth={2} />
                    <Line type="monotone" dataKey="quality" stroke="#10B981" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              {selectedSource.status === 'active' ? (
                <button 
                  onClick={() => {
                    handleSourceAction('pause', selectedSource.id);
                    setSelectedSource(null);
                  }}
                  className="bg-yellow-600 text-white px-6 py-2 rounded-xl hover:bg-yellow-700 transition-all flex items-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  توقف جمع‌آوری
                </button>
              ) : selectedSource.status === 'inactive' ? (
                <button 
                  onClick={() => {
                    handleSourceAction('start', selectedSource.id);
                    setSelectedSource(null);
                  }}
                  className="bg-green-600 text-white px-6 py-2 rounded-xl hover:bg-green-700 transition-all flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  شروع جمع‌آوری
                </button>
              ) : null}
              
              <button className="bg-blue-600 text-white px-6 py-2 rounded-xl hover:bg-blue-700 transition-all flex items-center gap-2">
                <Download className="w-4 h-4" />
                دانلود داده‌ها
              </button>
              
              <button className="bg-gray-600 text-white px-6 py-2 rounded-xl hover:bg-gray-700 transition-all flex items-center gap-2">
                <RefreshCw className="w-4 h-4" />
                همگام‌سازی فوری
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataPage;