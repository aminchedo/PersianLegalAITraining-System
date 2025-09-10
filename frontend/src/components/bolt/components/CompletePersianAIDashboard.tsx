import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell } from 'recharts';
import { Play, Pause, Square, AlertCircle, CheckCircle, Clock, Activity, Cpu, HardDrive, Zap, TrendingUp, Users, Database, Settings, Upload, Download, RefreshCw, Target, Award, Layers } from 'lucide-react';
import { usePersianAI } from '../../../hooks/usePersianAI';
import { Link } from 'react-router-dom';

const CompletePersianAIDashboard: React.FC = () => {
  const { state } = usePersianAI();
  const [realTimeData, setRealTimeData] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    rank: 16,
    alpha: 32,
    target_modules: ['q_proj', 'v_proj'],
    quantization_bits: 8,
    adaptive_rank: true,
  });

  // Generate real-time training data
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const newDataPoint = {
        time: now.toLocaleTimeString('fa-IR', { hour12: false }),
        loss: Math.random() * 0.5 + 0.1,
        accuracy: Math.random() * 0.2 + 0.8,
        cpu: Math.random() * 20 + 40,
        memory: Math.random() * 15 + 60,
        gpu: Math.random() * 10 + 75,
      };
      setRealTimeData(prev => [...prev.slice(-19), newDataPoint]);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const pieData = [
    { name: 'در حال آموزش', value: 2, color: '#3b82f6' },
    { name: 'آماده', value: 5, color: '#10b981' },
    { name: 'خطا', value: 1, color: '#ef4444' },
  ];

  const currentTraining = state.trainingSessions.find(s => s.status === 'running');

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 rounded-xl p-6 text-white">
        <h1 className="text-3xl font-bold mb-2">داشبورد آموزش هوش مصنوعی حقوقی</h1>
        <p className="text-purple-100">مدیریت و نظارت بر فرآیند آموزش مدل‌های هوش مصنوعی</p>
        
        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm">کل مدل‌ها</p>
                <p className="text-2xl font-bold">{state.models.length}</p>
              </div>
              <Zap className="w-8 h-8 text-purple-200" />
            </div>
          </div>
          
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm">در حال آموزش</p>
                <p className="text-2xl font-bold">{state.trainingSessions.filter(s => s.status === 'running').length}</p>
              </div>
              <Activity className="w-8 h-8 text-purple-200" />
            </div>
          </div>
          
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm">منابع داده</p>
                <p className="text-2xl font-bold">{state.dataSources.length}</p>
              </div>
              <Database className="w-8 h-8 text-purple-200" />
            </div>
          </div>
          
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm">میانگین دقت</p>
                <p className="text-2xl font-bold">87%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-purple-200" />
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Training Control Section - 70% of page focus */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Model Selection Panel */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">انتخاب و پیکربندی مدل</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {state.models.map((model) => (
                <div
                  key={model.id}
                  className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                    selectedModel === model.id 
                      ? 'border-purple-500 bg-purple-50' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedModel(model.id)}
                >
                  <div className="flex items-center space-x-3 space-x-reverse mb-3">
                    <div className="bg-purple-100 p-2 rounded-lg">
                      <Layers className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{model.name}</h3>
                      <p className="text-sm text-gray-600">{model.type}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      model.status === 'deployed' ? 'bg-green-100 text-green-700' :
                      model.status === 'training' ? 'bg-blue-100 text-blue-700' :
                      model.status === 'error' ? 'bg-red-100 text-red-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {model.status === 'deployed' ? 'آماده' :
                       model.status === 'training' ? 'در حال آموزش' :
                       model.status === 'error' ? 'خطا' : 'در انتظار'}
                    </span>
                    <span className="text-sm font-medium text-gray-700">
                      دقت: {(model.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* DoRA/QR-Adaptor Configuration */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">تنظیمات پیشرفته آموزش</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Basic Parameters */}
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">پارامترهای اصلی</h4>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Epochs</label>
                    <input
                      type="number"
                      value={trainingConfig.epochs}
                      onChange={(e) => setTrainingConfig(prev => ({...prev, epochs: parseInt(e.target.value)}))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Batch Size</label>
                    <input
                      type="number"
                      value={trainingConfig.batch_size}
                      onChange={(e) => setTrainingConfig(prev => ({...prev, batch_size: parseInt(e.target.value)}))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div className="col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">Learning Rate</label>
                    <input
                      type="number"
                      step="0.0001"
                      value={trainingConfig.learning_rate}
                      onChange={(e) => setTrainingConfig(prev => ({...prev, learning_rate: parseFloat(e.target.value)}))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>

              {/* DoRA Configuration */}
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">تنظیمات DoRA</h4>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Rank</label>
                    <input
                      type="number"
                      value={trainingConfig.rank}
                      onChange={(e) => setTrainingConfig(prev => ({...prev, rank: parseInt(e.target.value)}))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Alpha</label>
                    <input
                      type="number"
                      value={trainingConfig.alpha}
                      onChange={(e) => setTrainingConfig(prev => ({...prev, alpha: parseInt(e.target.value)}))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Quantization</label>
                    <select
                      value={trainingConfig.quantization_bits}
                      onChange={(e) => setTrainingConfig(prev => ({...prev, quantization_bits: parseInt(e.target.value)}))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    >
                      <option value={4}>4-bit</option>
                      <option value={8}>8-bit</option>
                      <option value={16}>16-bit</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center">
                    <label className="flex items-center space-x-2 space-x-reverse">
                      <input
                        type="checkbox"
                        checked={trainingConfig.adaptive_rank}
                        onChange={(e) => setTrainingConfig(prev => ({...prev, adaptive_rank: e.target.checked}))}
                        className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                      />
                      <span className="text-sm font-medium text-gray-700">Adaptive Rank</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* Training Controls */}
            <div className="flex items-center justify-between mt-6 pt-6 border-t border-gray-200">
              <div className="flex items-center space-x-4 space-x-reverse">
                <button
                  disabled={!selectedModel}
                  className="flex items-center bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:bg-gray-300 transition-colors"
                >
                  <Play className="w-5 h-5 ml-2" />
                  شروع آموزش
                </button>
                
                <button className="flex items-center bg-yellow-600 text-white px-4 py-2 rounded-lg hover:bg-yellow-700 transition-colors">
                  <Pause className="w-4 h-4 ml-2" />
                  مکث
                </button>
                
                <button className="flex items-center bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors">
                  <Square className="w-4 h-4 ml-2" />
                  توقف
                </button>
              </div>
              
              <div className="flex items-center space-x-2 space-x-reverse">
                <button className="flex items-center border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
                  <Upload className="w-4 h-4 ml-2" />
                  بارگذاری
                </button>
                
                <button className="flex items-center border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
                  <Download className="w-4 h-4 ml-2" />
                  دانلود
                </button>
              </div>
            </div>
          </div>

          {/* Main Training Control Panel */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-900">کنترل آموزش مدل‌ها</h2>
              <Link
                to="/"
                className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
              >
                مدیریت کامل آموزش
              </Link>
            </div>

            {currentTraining ? (
              <div className="space-y-6">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-blue-900">{currentTraining.model_name}</h3>
                      <p className="text-blue-700 text-sm">epoch {currentTraining.current_epoch} از {currentTraining.total_epochs}</p>
                    </div>
                    <div className="flex space-x-2 space-x-reverse">
                      <button className="bg-yellow-500 text-white p-2 rounded hover:bg-yellow-600">
                        <Pause className="w-4 h-4" />
                      </button>
                      <button className="bg-red-500 text-white p-2 rounded hover:bg-red-600">
                        <Square className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="text-center">
                      <p className="text-sm text-blue-700">پیشرفت</p>
                      <p className="text-lg font-bold text-blue-900">{currentTraining.progress}%</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-blue-700">تابع هزینه</p>
                      <p className="text-lg font-bold text-blue-900">{currentTraining.loss.toFixed(3)}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-blue-700">دقت</p>
                      <p className="text-lg font-bold text-blue-900">{(currentTraining.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-blue-700">زمان باقیمانده</p>
                      <p className="text-lg font-bold text-blue-900">2ساعت 15دقیقه</p>
                    </div>
                  </div>

                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                      style={{ width: `${currentTraining.progress}%` }}
                    ></div>
                  </div>
                </div>

                {/* Real-time Training Chart */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold mb-4">نمودار زنده آموزش</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={realTimeData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="loss" stroke="#ef4444" name="تابع هزینه" />
                        <Line type="monotone" dataKey="accuracy" stroke="#10b981" name="دقت" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">هیچ آموزشی در حال اجرا نیست</h3>
                <p className="text-gray-600 mb-4">برای شروع آموزش مدل جدید، روی دکمه زیر کلیک کنید</p>
                <Link
                  to="/"
                  className="inline-flex items-center bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-colors"
                >
                  <Play className="w-5 h-5 ml-2" />
                  شروع آموزش جدید
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Training Start & Model Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">شروع سریع آموزش</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {state.models.filter(m => m.status !== 'training').map((model) => (
              <div key={model.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-gray-900">{model.name}</h4>
                  <div className={`flex items-center space-x-1 space-x-reverse text-xs px-2 py-1 rounded-full ${
                    model.status === 'deployed' ? 'bg-green-100 text-green-700' :
                    model.status === 'error' ? 'bg-red-100 text-red-700' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>
                    {model.status === 'deployed' ? <CheckCircle className="w-3 h-3" /> :
                     model.status === 'error' ? <AlertCircle className="w-3 h-3" /> :
                     <Clock className="w-3 h-3" />}
                    <span>{
                      model.status === 'deployed' ? 'آماده' :
                      model.status === 'error' ? 'خطا' : 'در انتظار'
                    }</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">دقت: {(model.accuracy * 100).toFixed(1)}%</span>
                  <button className="bg-purple-600 text-white px-3 py-1 text-sm rounded hover:bg-purple-700 transition-colors">
                    آموزش مجدد
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">وضعیت مدل‌ها</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2 mt-4">
            {pieData.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-2 space-x-reverse">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  ></div>
                  <span className="text-sm text-gray-600">{item.name}</span>
                </div>
                <span className="text-sm font-semibold text-gray-900">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System Monitoring & Training Queue */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">نظارت سیستم</h3>
          <div className="space-y-4">
            {state.systemMetrics && (
              <>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <Cpu className="w-5 h-5 text-blue-500" />
                    <span className="text-gray-700">CPU</span>
                  </div>
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${state.systemMetrics.cpu_usage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">{state.systemMetrics.cpu_usage}%</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <HardDrive className="w-5 h-5 text-green-500" />
                    <span className="text-gray-700">RAM</span>
                  </div>
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${state.systemMetrics.memory_usage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">{state.systemMetrics.memory_usage}%</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <Zap className="w-5 h-5 text-purple-500" />
                    <span className="text-gray-700">GPU</span>
                  </div>
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${state.systemMetrics.gpu_usage}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-gray-900">{state.systemMetrics.gpu_usage}%</span>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">صف آموزش</h3>
          <div className="space-y-3">
            {state.trainingSessions.length > 0 ? (
              state.trainingSessions.map((session) => (
                <div key={session.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">{session.model_name}</p>
                    <p className="text-sm text-gray-600">
                      {session.status === 'running' ? 'در حال اجرا' :
                       session.status === 'completed' ? 'تکمیل شده' :
                       session.status === 'failed' ? 'خطا' : 'متوقف شده'}
                    </p>
                  </div>
                  <div className="text-left">
                    <p className="text-sm font-medium text-gray-900">{session.progress}%</p>
                    <div className={`w-2 h-2 rounded-full ${
                      session.status === 'running' ? 'bg-blue-500' :
                      session.status === 'completed' ? 'bg-green-500' :
                      session.status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
                    }`}></div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8">
                <Clock className="w-12 h-12 text-gray-300 mx-auto mb-2" />
                <p className="text-gray-500">صف آموزش خالی است</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Training History */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">تاریخچه آموزش اخیر</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-right py-3 text-sm font-semibold text-gray-700">نام مدل</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">وضعیت</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">دقت نهایی</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">مدت آموزش</th>
                <th className="text-right py-3 text-sm font-semibold text-gray-700">تاریخ</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {state.trainingSessions.map((session) => (
                <tr key={session.id} className="hover:bg-gray-50">
                  <td className="py-3 text-sm text-gray-900">{session.model_name}</td>
                  <td className="py-3">
                    <span className={`inline-flex items-center px-2 py-1 text-xs rounded-full ${
                      session.status === 'running' ? 'bg-blue-100 text-blue-700' :
                      session.status === 'completed' ? 'bg-green-100 text-green-700' :
                      session.status === 'failed' ? 'bg-red-100 text-red-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {session.status === 'running' ? 'در حال اجرا' :
                       session.status === 'completed' ? 'تکمیل شده' :
                       session.status === 'failed' ? 'خطا' : 'متوقف شده'}
                    </span>
                  </td>
                  <td className="py-3 text-sm text-gray-900">{(session.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-3 text-sm text-gray-900">3 ساعت 45 دقیقه</td>
                  <td className="py-3 text-sm text-gray-500">
                    {new Date(session.started_at).toLocaleDateString('fa-IR')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default CompletePersianAIDashboard;