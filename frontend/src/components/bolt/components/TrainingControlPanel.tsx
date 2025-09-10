import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { 
  Play, Pause, Square, Settings, Save, Upload, Download, RefreshCw, 
  AlertTriangle, CheckCircle, Clock, Activity, Cpu, HardDrive, Zap,
  Filter, Search, Eye, Edit, Trash2, Copy, MoreVertical, Calendar,
  TrendingUp, TrendingDown, Target, Layers, Database
} from 'lucide-react';
import { usePersianAI } from '../../../hooks/usePersianAI';

const TrainingControlPanel: React.FC = () => {
  const { state, dispatch } = usePersianAI();
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.001,
    // DoRA Configuration
    rank: 16,
    alpha: 32,
    target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    dropout: 0.1,
    // QR-Adaptor settings
    quantization_bits: 8,
    adaptive_rank: true,
    compression_ratio: 0.5,
    // Advanced settings
    warmup_steps: 100,
    weight_decay: 0.01,
    gradient_clipping: 1.0,
    scheduler: 'cosine',
  });

  const [realTimeMetrics, setRealTimeMetrics] = useState<any[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [logFilter, setLogFilter] = useState<string>('all');
  const [isAdvancedMode, setIsAdvancedMode] = useState(false);
  const [scheduledSessions, setScheduledSessions] = useState<any[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  // Real-time metrics simulation
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const newMetric = {
        time: now.toLocaleTimeString('fa-IR', { hour12: false }),
        loss: Math.random() * 0.3 + 0.1,
        accuracy: Math.random() * 0.15 + 0.8,
        learning_rate: trainingConfig.learning_rate * (0.8 + Math.random() * 0.4),
        gpu_memory: Math.random() * 10 + 70,
        throughput: Math.random() * 50 + 100,
      };
      setRealTimeMetrics(prev => [...prev.slice(-29), newMetric]);

      // Add random log entry
      if (Math.random() > 0.7) {
        const logMessages = [
          'Epoch 5/10 - Batch 120/500 - Loss: 0.234',
          'Learning rate adjusted to 0.0008',
          'Validation accuracy improved: 0.876',
          'Checkpoint saved at epoch 5',
          'GPU memory usage: 7.2GB/8GB',
        ];
        const newLog = `[${now.toLocaleTimeString('fa-IR')}] ${logMessages[Math.floor(Math.random() * logMessages.length)]}`;
        setTrainingLogs(prev => [newLog, ...prev.slice(0, 99)]);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [trainingConfig.learning_rate]);

  const handleStartTraining = async () => {
    if (!selectedModel) {
      alert('لطفاً مدلی را انتخاب کنید');
      return;
    }

    setIsTraining(true);
    try {
      // Real API call to start training
      const response = await fetch('http://localhost:8000/api/training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          config: trainingConfig,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start training');
      }

      const result = await response.json();
      console.log('Training started:', result);
      
      // Mock training session creation
      const newSession = {
        id: result.session_id || `ts_${Date.now()}`,
        model_id: selectedModel,
        model_name: state.models.find(m => m.id === selectedModel)?.name || 'Unknown Model',
        status: 'running' as const,
        progress: 0,
        current_epoch: 0,
        total_epochs: trainingConfig.epochs,
        loss: 0.5,
        accuracy: 0.5,
        started_at: new Date().toISOString(),
        estimated_completion: new Date(Date.now() + trainingConfig.epochs * 30 * 60 * 1000).toISOString(),
      };

      dispatch({ type: 'UPDATE_TRAINING_SESSION', payload: newSession });
    } catch (error) {
      console.error('Training start failed:', error);
      alert('خطا در شروع آموزش: ' + error.message);
    } finally {
      setIsTraining(false);
    }
  };

  const handleStopTraining = async () => {
    const currentSession = state.trainingSessions.find(s => s.status === 'running');
    if (!currentSession) return;

    try {
      const response = await fetch('http://localhost:8000/api/training/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: currentSession.id,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to stop training');
      }

      // Update session status
      const updatedSession = { ...currentSession, status: 'paused' as const };
      dispatch({ type: 'UPDATE_TRAINING_SESSION', payload: updatedSession });
    } catch (error) {
      console.error('Training stop failed:', error);
      alert('خطا در توقف آموزش: ' + error.message);
    }
  };

  const handleConfigChange = (key: string, value: any) => {
    setTrainingConfig(prev => ({ ...prev, [key]: value }));
  };

  const currentSession = state.trainingSessions.find(s => s.status === 'running');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">کنترل کامل آموزش مدل‌ها</h1>
            <p className="text-indigo-100">تنظیمات پیشرفته و نظارت دقیق بر فرآیند آموزش</p>
          </div>
          <div className="flex items-center space-x-4 space-x-reverse">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-center">
                <p className="text-2xl font-bold">{state.trainingSessions.filter(s => s.status === 'running').length}</p>
                <p className="text-sm text-indigo-100">آموزش فعال</p>
              </div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-center">
                <p className="text-2xl font-bold">{state.models.length}</p>
                <p className="text-sm text-indigo-100">مدل موجود</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Panel - Training Configuration */}
        <div className="xl:col-span-2 space-y-6">
          {/* Model Selection */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">انتخاب و تنظیم مدل</h2>
            
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

            {/* Advanced Mode Toggle */}
            <div className="flex items-center justify-between mb-4">
              <label className="flex items-center space-x-2 space-x-reverse">
                <input
                  type="checkbox"
                  checked={isAdvancedMode}
                  onChange={(e) => setIsAdvancedMode(e.target.checked)}
                  className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                />
                <span className="text-sm font-medium text-gray-700">حالت پیشرفته</span>
              </label>
            </div>
          </div>

          {/* Basic Training Parameters */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">پارامترهای اصلی آموزش</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">تعداد Epochs</label>
                <input
                  type="number"
                  value={trainingConfig.epochs}
                  onChange={(e) => handleConfigChange('epochs', parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">اندازه Batch</label>
                <input
                  type="number"
                  value={trainingConfig.batch_size}
                  onChange={(e) => handleConfigChange('batch_size', parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">نرخ یادگیری</label>
                <input
                  type="number"
                  step="0.0001"
                  value={trainingConfig.learning_rate}
                  onChange={(e) => handleConfigChange('learning_rate', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Dropout</label>
                <input
                  type="number"
                  step="0.01"
                  value={trainingConfig.dropout}
                  onChange={(e) => handleConfigChange('dropout', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
            </div>
          </div>

          {/* DoRA Configuration */}
          {isAdvancedMode && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">تنظیمات DoRA</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Rank</label>
                  <input
                    type="number"
                    value={trainingConfig.rank}
                    onChange={(e) => handleConfigChange('rank', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Alpha</label>
                  <input
                    type="number"
                    value={trainingConfig.alpha}
                    onChange={(e) => handleConfigChange('alpha', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
              </div>
              
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">Target Modules</label>
                <div className="grid grid-cols-2 gap-2">
                  {['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].map((module) => (
                    <label key={module} className="flex items-center space-x-2 space-x-reverse">
                      <input
                        type="checkbox"
                        checked={trainingConfig.target_modules.includes(module)}
                        onChange={(e) => {
                          const modules = e.target.checked 
                            ? [...trainingConfig.target_modules, module]
                            : trainingConfig.target_modules.filter(m => m !== module);
                          handleConfigChange('target_modules', modules);
                        }}
                        className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                      />
                      <span className="text-sm text-gray-700">{module}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* QR-Adaptor Settings */}
          {isAdvancedMode && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">تنظیمات QR-Adaptor</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Quantization Bits</label>
                  <select
                    value={trainingConfig.quantization_bits}
                    onChange={(e) => handleConfigChange('quantization_bits', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  >
                    <option value={4}>4-bit</option>
                    <option value={8}>8-bit</option>
                    <option value={16}>16-bit</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Compression Ratio</label>
                  <input
                    type="number"
                    step="0.1"
                    value={trainingConfig.compression_ratio}
                    onChange={(e) => handleConfigChange('compression_ratio', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
              </div>
              
              <div className="mt-4">
                <label className="flex items-center space-x-2 space-x-reverse">
                  <input
                    type="checkbox"
                    checked={trainingConfig.adaptive_rank}
                    onChange={(e) => handleConfigChange('adaptive_rank', e.target.checked)}
                    className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Adaptive Rank</span>
                </label>
              </div>
            </div>
          )}

          {/* Training Controls */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">کنترل‌های آموزش</h3>
            
            <div className="flex items-center space-x-4 space-x-reverse mb-6">
              <button
                onClick={handleStartTraining}
                disabled={!selectedModel || isTraining}
                className="flex items-center bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:bg-gray-300 transition-colors"
              >
                {isTraining && <div className="spinner ml-2"></div>}
                <Play className="w-5 h-5 ml-2" />
                {isTraining ? 'در حال شروع...' : 'شروع آموزش'}
              </button>
              
              {currentSession && (
                <>
                  <button 
                    onClick={handleStopTraining}
                    className="flex items-center bg-yellow-600 text-white px-4 py-2 rounded-lg hover:bg-yellow-700 transition-colors"
                  >
                    <Pause className="w-4 h-4 ml-2" />
                    مکث
                  </button>
                  
                  <button 
                    onClick={handleStopTraining}
                    className="flex items-center bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
                  >
                    <Square className="w-4 h-4 ml-2" />
                    توقف
                  </button>
                </>
              )}
              
              <button className="flex items-center border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
                <Save className="w-4 h-4 ml-2" />
                ذخیره تنظیمات
              </button>
              
              <button className="flex items-center border border-gray-300 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
                <Upload className="w-4 h-4 ml-2" />
                بارگذاری تنظیمات
              </button>
            </div>

            {/* Preset Configurations */}
            <div className="border-t border-gray-200 pt-4">
              <h4 className="text-sm font-semibold text-gray-700 mb-3">تنظیمات از پیش تعریف شده</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <button 
                  onClick={() => setTrainingConfig({
                    ...trainingConfig,
                    epochs: 5,
                    batch_size: 16,
                    learning_rate: 0.0001,
                  })}
                  className="p-3 border border-gray-200 rounded-lg text-left hover:bg-gray-50 transition-colors"
                >
                  <p className="font-medium text-sm">سریع</p>
                  <p className="text-xs text-gray-600">5 epoch, batch 16</p>
                </button>
                
                <button 
                  onClick={() => setTrainingConfig({
                    ...trainingConfig,
                    epochs: 10,
                    batch_size: 32,
                    learning_rate: 0.001,
                  })}
                  className="p-3 border border-gray-200 rounded-lg text-left hover:bg-gray-50 transition-colors"
                >
                  <p className="font-medium text-sm">متعادل</p>
                  <p className="text-xs text-gray-600">10 epoch, batch 32</p>
                </button>
                
                <button 
                  onClick={() => setTrainingConfig({
                    ...trainingConfig,
                    epochs: 20,
                    batch_size: 64,
                    learning_rate: 0.0005,
                  })}
                  className="p-3 border border-gray-200 rounded-lg text-left hover:bg-gray-50 transition-colors"
                >
                  <p className="font-medium text-sm">دقیق</p>
                  <p className="text-xs text-gray-600">20 epoch, batch 64</p>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Monitoring & Logs */}
        <div className="space-y-6">
          {/* Current Training Status */}
          {currentSession ? (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">آموزش جاری</h3>
              
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 mb-2">{currentSession.model_name}</h4>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-blue-700">Epoch</p>
                      <p className="font-semibold text-blue-900">
                        {currentSession.current_epoch}/{currentSession.total_epochs}
                      </p>
                    </div>
                    <div>
                      <p className="text-blue-700">پیشرفت</p>
                      <p className="font-semibold text-blue-900">{currentSession.progress}%</p>
                    </div>
                    <div>
                      <p className="text-blue-700">تابع هزینه</p>
                      <p className="font-semibold text-blue-900">{currentSession.loss.toFixed(3)}</p>
                    </div>
                    <div>
                      <p className="text-blue-700">دقت</p>
                      <p className="font-semibold text-blue-900">{(currentSession.accuracy * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  
                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${currentSession.progress}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="text-center py-8">
                <Activity className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">آموزش فعالی وجود ندارد</h3>
                <p className="text-gray-600">مدل و تنظیمات را انتخاب کنید و آموزش را شروع کنید</p>
              </div>
            </div>
          )}

          {/* System Resources */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">منابع سیستم</h3>
            
            {state.systemMetrics && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <Cpu className="w-4 h-4 text-blue-500" />
                    <span className="text-sm text-gray-700">CPU</span>
                  </div>
                  <span className="text-sm font-medium">{state.systemMetrics.cpu_usage}%</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <HardDrive className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-gray-700">RAM</span>
                  </div>
                  <span className="text-sm font-medium">{state.systemMetrics.memory_usage}%</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 space-x-reverse">
                    <Zap className="w-4 h-4 text-purple-500" />
                    <span className="text-sm text-gray-700">GPU</span>
                  </div>
                  <span className="text-sm font-medium">{state.systemMetrics.gpu_usage}%</span>
                </div>
                
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-xs text-gray-600">دما</p>
                  <p className="text-lg font-bold text-gray-900">{state.systemMetrics.temperature}°C</p>
                </div>
              </div>
            )}
          </div>

          {/* Training Queue */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">صف آموزش</h3>
              <button className="text-purple-600 hover:text-purple-700 transition-colors">
                <Calendar className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-2">
              {state.trainingSessions.length > 0 ? (
                state.trainingSessions.map((session) => (
                  <div key={session.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">{session.model_name}</p>
                      <p className="text-xs text-gray-600">{session.progress}% تکمیل</p>
                    </div>
                    <button className="text-gray-400 hover:text-gray-600">
                      <MoreVertical className="w-4 h-4" />
                    </button>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500 text-center py-4">صف خالی است</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Metrics Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">نمودار عملکرد زنده</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={realTimeMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="loss" stroke="#ef4444" name="تابع هزینه" strokeWidth={2} />
                <Line type="monotone" dataKey="accuracy" stroke="#10b981" name="دقت" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">منابع سیستم</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={realTimeMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="gpu_memory" stackId="1" stroke="#8b5cf6" fill="#8b5cf6" name="GPU Memory" />
                <Area type="monotone" dataKey="throughput" stackId="2" stroke="#3b82f6" fill="#3b82f6" name="Throughput" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Live Training Logs */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">لاگ‌های زنده آموزش</h3>
          <div className="flex items-center space-x-4 space-x-reverse">
            <select 
              value={logFilter}
              onChange={(e) => setLogFilter(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            >
              <option value="all">همه</option>
              <option value="error">خطاها</option>
              <option value="warning">هشدارها</option>
              <option value="info">اطلاعات</option>
            </select>
            <button className="text-gray-600 hover:text-gray-900">
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        <div className="bg-gray-900 text-green-400 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
          {trainingLogs.length > 0 ? (
            trainingLogs.map((log, index) => (
              <div key={index} className="mb-1 hover:bg-gray-800 px-2 py-1 rounded">
                {log}
              </div>
            ))
          ) : (
            <div className="text-gray-500 text-center">
              هیچ لاگی در دسترس نیست
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingControlPanel;