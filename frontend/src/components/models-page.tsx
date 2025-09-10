import React, { useState } from 'react';
import { 
  Brain, Play, Pause, Square, Settings, Download, Upload, Eye, EyeOff, 
  Maximize2, RefreshCw, Plus, X, Edit3, Trash2, Copy, Share2,
  Activity, Zap, Clock, CheckCircle, AlertTriangle, AlertCircle,
  TrendingUp, TrendingDown, BarChart3, PieChart
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RePieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { useAppContext } from './router';
import { ModelTraining } from '../types/dashboard';

const ModelsPage = () => {
  const { models, setModels, realTimeData } = useAppContext();
  const [selectedModel, setSelectedModel] = useState<ModelTraining | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [viewMode, setViewMode] = useState('grid'); // grid, list, detail
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('lastUpdated');
  const [searchTerm, setSearchTerm] = useState('');

  const [newModel, setNewModel] = useState({
    name: '',
    type: 'language-model',
    description: '',
    parameters: '1B',
    framework: 'PyTorch',
    doraRank: 64
  });

  // Filter and sort models
  const filteredModels = models
    .filter(model => {
      const matchesSearch = model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           model.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = filterStatus === 'all' || model.status === filterStatus;
      return matchesSearch && matchesStatus;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'progress':
          return b.progress - a.progress;
        case 'accuracy':
          return b.accuracy - a.accuracy;
        case 'lastUpdated':
        default:
          return new Date(b.lastUpdated) - new Date(a.lastUpdated);
      }
    });

  // Model action handlers
  const handleModelAction = (action: string, modelId: any, config: any = null) => {
    setModels(prev => prev.map(model => {
      if (model.id === modelId) {
        switch (action) {
          case 'start':
            return { ...model, status: 'training' };
          case 'pause':
            return { ...model, status: 'pending' };
          case 'stop':
            return { ...model, status: 'pending', progress: 0 };
          case 'delete':
            return null;
          default:
            return model;
        }
      }
      return model;
    }).filter(Boolean));
  };

  const handleCreateModel = () => {
    const model = {
      id: Date.now(),
      ...newModel,
      status: 'pending',
      progress: 0,
      accuracy: 0,
      loss: 0,
      epochs: 0,
      timeRemaining: 'در انتظار',
      lastUpdated: new Date()
    };
    
    setModels(prev => [...prev, model]);
    setNewModel({
      name: '',
      type: 'language-model',
      description: '',
      parameters: '1B',
      framework: 'PyTorch',
      doraRank: 64
    });
    setShowCreateModal(false);
  };

  // Status colors and icons
  const getStatusConfig = (status: string) => {
    const configs = {
      training: { color: 'bg-blue-500', textColor: 'text-blue-700', bgColor: 'bg-blue-100', icon: Activity },
      completed: { color: 'bg-green-500', textColor: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle },
      pending: { color: 'bg-yellow-500', textColor: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: Clock },
      error: { color: 'bg-red-500', textColor: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle }
    };
    return configs[status] || configs.pending;
  };

  const getStatusText = (status: string) => {
    const texts = {
      training: 'در حال آموزش',
      completed: 'تکمیل شده',
      pending: 'در انتظار',
      error: 'خطا'
    };
    return texts[status] || status;
  };

  // Model type configurations
  const getModelTypeConfig = (type: string) => {
    const configs = {
      'language-model': { label: 'مدل زبانی', color: 'purple', icon: Brain },
      'bert-model': { label: 'مدل BERT', color: 'blue', icon: Brain },
      'qa-model': { label: 'پرسش و پاسخ', color: 'green', icon: Brain },
      'ner-model': { label: 'شناسایی موجودیت', color: 'orange', icon: Brain }
    };
    return configs[type] || configs['language-model'];
  };

  // Performance chart data
  const performanceData = realTimeData.slice(-20).map((point, index) => ({
    epoch: index + 1,
    accuracy: 60 + index * 2 + Math.random() * 5,
    loss: Math.exp(-index/10) * (0.5 + Math.random() * 0.2),
    validationLoss: Math.exp(-index/8) * (0.6 + Math.random() * 0.2)
  }));

  // Statistics
  const modelStats = {
    total: models.length,
    training: models.filter(m => m.status === 'training').length,
    completed: models.filter(m => m.status === 'completed').length,
    pending: models.filter(m => m.status === 'pending').length,
    error: models.filter(m => m.status === 'error').length
  };

  const avgAccuracy = models.reduce((sum, model) => sum + model.accuracy, 0) / models.length;

  return (
    <div className="space-y-6" dir="rtl">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 rounded-2xl p-8 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
              <Brain className="w-8 h-8" />
              مدیریت مدل‌های هوش مصنوعی
            </h1>
            <p className="text-blue-100">آموزش، نظارت و مدیریت مدل‌های یادگیری ماشین</p>
          </div>
          <button 
            onClick={() => setShowCreateModal(true)}
            className="bg-white text-blue-600 px-6 py-3 rounded-xl font-medium hover:bg-blue-50 transition-all duration-300 flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            مدل جدید
          </button>
        </div>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-4 h-4" />
              <span className="text-sm">کل مدل‌ها</span>
            </div>
            <p className="text-2xl font-bold">{modelStats.total}</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4" />
              <span className="text-sm">در حال آموزش</span>
            </div>
            <p className="text-2xl font-bold">{modelStats.training}</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-4 h-4" />
              <span className="text-sm">تکمیل شده</span>
            </div>
            <p className="text-2xl font-bold">{modelStats.completed}</p>
          </div>
          <div className="bg-white/20 rounded-xl p-4 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm">میانگین دقت</span>
            </div>
            <p className="text-2xl font-bold">{avgAccuracy.toFixed(1)}%</p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <input 
              type="text"
              placeholder="جستجو در مدل‌ها..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-64 px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <select 
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">همه وضعیت‌ها</option>
              <option value="training">در حال آموزش</option>
              <option value="completed">تکمیل شده</option>
              <option value="pending">در انتظار</option>
              <option value="error">خطا</option>
            </select>
            <select 
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
            >
              <option value="lastUpdated">آخرین بروزرسانی</option>
              <option value="name">نام</option>
              <option value="progress">پیشرفت</option>
              <option value="accuracy">دقت</option>
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
              <Brain className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Models Grid/List */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredModels.map(model => {
              const statusConfig = getStatusConfig(model.status);
              const typeConfig = getModelTypeConfig(model.type);
              const StatusIcon = statusConfig.icon;
              
              return (
                <div key={model.id} className="group relative">
                  <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl p-6 border border-gray-200 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${statusConfig.bgColor}`}>
                        <StatusIcon className={`w-6 h-6 ${statusConfig.textColor}`} />
                      </div>
                      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button 
                          onClick={() => setSelectedModel(model)}
                          className="w-8 h-8 bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 transition-all"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button className="w-8 h-8 bg-gray-600 text-white rounded-lg flex items-center justify-center hover:bg-gray-700 transition-all">
                          <Settings className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => handleModelAction('delete', model.id)}
                          className="w-8 h-8 bg-red-600 text-white rounded-lg flex items-center justify-center hover:bg-red-700 transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {/* Content */}
                    <div>
                      <h3 className="font-bold text-gray-900 mb-2">{model.name}</h3>
                      <p className="text-sm text-gray-600 mb-3 line-clamp-2">{model.description}</p>
                      
                      <div className="flex items-center gap-4 text-sm text-gray-500 mb-4">
                        <span>{typeConfig.label}</span>
                        <span>•</span>
                        <span>{model.parameters}</span>
                        <span>•</span>
                        <span>{model.framework}</span>
                      </div>

                      {/* Progress */}
                      <div className="mb-4">
                        <div className="flex justify-between text-sm mb-2">
                          <span>پیشرفت</span>
                          <span>{model.progress}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className={`${statusConfig.color} h-2 rounded-full transition-all duration-500`}
                            style={{ width: `${model.progress}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* Stats */}
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <p className="text-xs text-gray-500">دقت</p>
                          <p className="font-semibold">{model.accuracy}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">خطا</p>
                          <p className="font-semibold">{model.loss.toFixed(3)}</p>
                        </div>
                      </div>

                      {/* Status Badge */}
                      <div className="flex items-center justify-between">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.textColor}`}>
                          {getStatusText(model.status)}
                        </span>
                        <span className="text-xs text-gray-500">{model.timeRemaining}</span>
                      </div>

                      {/* Action Buttons */}
                      <div className="flex gap-2 mt-4">
                        {model.status === 'training' ? (
                          <button 
                            onClick={() => handleModelAction('pause', model.id)}
                            className="flex-1 bg-yellow-100 text-yellow-700 py-2 px-3 rounded-lg text-sm hover:bg-yellow-200 transition-all flex items-center justify-center gap-2"
                          >
                            <Pause className="w-4 h-4" />
                            توقف
                          </button>
                        ) : model.status === 'pending' ? (
                          <button 
                            onClick={() => handleModelAction('start', model.id)}
                            className="flex-1 bg-green-100 text-green-700 py-2 px-3 rounded-lg text-sm hover:bg-green-200 transition-all flex items-center justify-center gap-2"
                          >
                            <Play className="w-4 h-4" />
                            شروع
                          </button>
                        ) : null}
                        
                        <button 
                          onClick={() => setSelectedModel(model)}
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
            {filteredModels.map(model => {
              const statusConfig = getStatusConfig(model.status);
              const typeConfig = getModelTypeConfig(model.type);
              const StatusIcon = statusConfig.icon;
              
              return (
                <div key={model.id} className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-all">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${statusConfig.bgColor}`}>
                        <StatusIcon className={`w-6 h-6 ${statusConfig.textColor}`} />
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-900">{model.name}</h3>
                        <p className="text-sm text-gray-600">{model.description}</p>
                        <div className="flex items-center gap-4 text-sm text-gray-500 mt-1">
                          <span>{typeConfig.label}</span>
                          <span>•</span>
                          <span>دقت: {model.accuracy}%</span>
                          <span>•</span>
                          <span>DoRA: {model.doraRank}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="w-32 bg-gray-200 rounded-full h-2 mb-1">
                          <div 
                            className={`h-2 rounded-full transition-all duration-500 ${statusConfig.color}`}
                            style={{ width: `${model.progress}%` }}
                          ></div>
                        </div>
                        <span className="text-xs text-gray-500">{model.progress}%</span>
                      </div>
                      
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.textColor}`}>
                        {getStatusText(model.status)}
                      </span>
                      
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => setSelectedModel(model)}
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

      {/* Performance Chart */}
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">تحلیل عملکرد آموزش</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="epoch" stroke="#6b7280" />
            <YAxis stroke="#6b7280" />
            <Tooltip />
            <Line type="monotone" dataKey="accuracy" stroke="#10B981" strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="loss" stroke="#EF4444" strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="validationLoss" stroke="#F59E0B" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Create Model Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-2xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">ایجاد مدل جدید</h3>
              <button 
                onClick={() => setShowCreateModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">نام مدل</label>
                <input 
                  type="text"
                  value={newModel.name}
                  onChange={(e) => setNewModel(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  placeholder="نام مدل را وارد کنید..."
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">نوع مدل</label>
                  <select 
                    value={newModel.type}
                    onChange={(e) => setNewModel(prev => ({ ...prev, type: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="language-model">مدل زبانی</option>
                    <option value="bert-model">مدل BERT</option>
                    <option value="qa-model">پرسش و پاسخ</option>
                    <option value="ner-model">شناسایی موجودیت</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">فریمورک</label>
                  <select 
                    value={newModel.framework}
                    onChange={(e) => setNewModel(prev => ({ ...prev, framework: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="PyTorch">PyTorch</option>
                    <option value="Transformers">Transformers</option>
                    <option value="TensorFlow">TensorFlow</option>
                    <option value="spaCy">spaCy</option>
                  </select>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">توضیحات</label>
                <textarea 
                  value={newModel.description}
                  onChange={(e) => setNewModel(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="توضیحات مدل..."
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">تعداد پارامترها</label>
                  <select 
                    value={newModel.parameters}
                    onChange={(e) => setNewModel(prev => ({ ...prev, parameters: e.target.value }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="110M">110M</option>
                    <option value="340M">340M</option>
                    <option value="1B">1B</option>
                    <option value="3B">3B</option>
                    <option value="7B">7B</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">رتبه DoRA</label>
                  <input 
                    type="number"
                    value={newModel.doraRank}
                    onChange={(e) => setNewModel(prev => ({ ...prev, doraRank: parseInt(e.target.value) }))}
                    className="w-full px-4 py-2 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                    min="16"
                    max="256"
                    step="16"
                  />
                </div>
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button 
                onClick={handleCreateModel}
                className="flex-1 bg-blue-600 text-white py-3 rounded-xl hover:bg-blue-700 transition-all font-medium"
              >
                ایجاد مدل
              </button>
              <button 
                onClick={() => setShowCreateModal(false)}
                className="flex-1 bg-gray-300 text-gray-700 py-3 rounded-xl hover:bg-gray-400 transition-all font-medium"
              >
                لغو
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Model Detail Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-auto p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">جزئیات مدل: {selectedModel.name}</h3>
              <button 
                onClick={() => setSelectedModel(null)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Model Info */}
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">اطلاعات مدل</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>نوع:</span>
                      <span>{getModelTypeConfig(selectedModel.type).label}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>فریمورک:</span>
                      <span>{selectedModel.framework}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>پارامترها:</span>
                      <span>{selectedModel.parameters}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>رتبه DoRA:</span>
                      <span>{selectedModel.doraRank}</span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-xl p-4">
                  <h4 className="font-semibold mb-3">آمار عملکرد</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>دقت:</span>
                      <span className="font-semibold text-green-600">{selectedModel.accuracy}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>خطا:</span>
                      <span className="font-semibold text-red-600">{selectedModel.loss.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>دوره‌ها:</span>
                      <span>{selectedModel.epochs}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>پیشرفت:</span>
                      <span>{selectedModel.progress}%</span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Performance Chart */}
              <div>
                <h4 className="font-semibold mb-3">نمودار عملکرد</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="epoch" stroke="#6b7280" fontSize={12} />
                    <YAxis stroke="#6b7280" fontSize={12} />
                    <Tooltip />
                    <Line type="monotone" dataKey="accuracy" stroke="#10B981" strokeWidth={2} />
                    <Line type="monotone" dataKey="loss" stroke="#EF4444" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              {selectedModel.status === 'training' ? (
                <button 
                  onClick={() => {
                    handleModelAction('pause', selectedModel.id);
                    setSelectedModel(null);
                  }}
                  className="bg-yellow-600 text-white px-6 py-2 rounded-xl hover:bg-yellow-700 transition-all flex items-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  توقف آموزش
                </button>
              ) : selectedModel.status === 'pending' ? (
                <button 
                  onClick={() => {
                    handleModelAction('start', selectedModel.id);
                    setSelectedModel(null);
                  }}
                  className="bg-green-600 text-white px-6 py-2 rounded-xl hover:bg-green-700 transition-all flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  شروع آموزش
                </button>
              ) : null}
              
              <button className="bg-blue-600 text-white px-6 py-2 rounded-xl hover:bg-blue-700 transition-all flex items-center gap-2">
                <Download className="w-4 h-4" />
                دانلود مدل
              </button>
              
              <button className="bg-gray-600 text-white px-6 py-2 rounded-xl hover:bg-gray-700 transition-all flex items-center gap-2">
                <Share2 className="w-4 h-4" />
                اشتراک‌گذاری
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelsPage;