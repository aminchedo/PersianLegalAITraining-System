import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Play, Pause, Square, Settings, BarChart3, Clock, Zap } from 'lucide-react';

interface TrainingSession {
  session_id: string;
  status: 'initializing' | 'training' | 'completed' | 'failed';
  current_epoch: number;
  total_epochs: number;
  current_loss: number;
  best_accuracy: number;
  start_time: string;
  progress_percent: number;
  logs: string[];
}

const TrainingPage: React.FC = () => {
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 3,
    batch_size: 8,
    learning_rate: 0.0002,
    model_type: 'dora'
  });
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  
  const queryClient = useQueryClient();

  // Fetch training sessions
  const { data: sessions } = useQuery<TrainingSession[]>({
    queryKey: ['trainingSessions'],
    queryFn: async () => {
      const response = await fetch('/api/training/sessions');
      return response.json();
    },
    refetchInterval: 5000, // Refresh every 5 seconds during training
  });

  // Fetch active session details
  const { data: activeSession } = useQuery<TrainingSession>({
    queryKey: ['activeSession', activeSessionId],
    queryFn: async () => {
      if (!activeSessionId) return null;
      const response = await fetch(`/api/training/sessions/${activeSessionId}/status`);
      return response.json();
    },
    enabled: !!activeSessionId,
    refetchInterval: 2000, // More frequent updates for active session
  });

  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: async (config: typeof trainingConfig) => {
      const response = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      return response.json();
    },
    onSuccess: (data) => {
      setActiveSessionId(data.session_id);
      queryClient.invalidateQueries({ queryKey: ['trainingSessions'] });
    },
  });

  // Stop training mutation
  const stopTrainingMutation = useMutation({
    mutationFn: async (sessionId: string) => {
      const response = await fetch(`/api/training/sessions/${sessionId}/stop`, {
        method: 'POST',
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trainingSessions'] });
    },
  });

  const handleStartTraining = () => {
    startTrainingMutation.mutate(trainingConfig);
  };

  const handleStopTraining = () => {
    if (activeSessionId) {
      stopTrainingMutation.mutate(activeSessionId);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training': return 'text-blue-600 bg-blue-100';
      case 'completed': return 'text-green-600 bg-green-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'initializing': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'training': return 'در حال آموزش';
      case 'completed': return 'تکمیل شده';
      case 'failed': return 'ناموفق';
      case 'initializing': return 'مقداردهی اولیه';
      default: return 'نامشخص';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          آموزش مدل هوش مصنوعی
        </h1>
        <p className="text-gray-600">
          آموزش و بهینه‌سازی مدل‌های BERT فارسی با تکنیک DoRA
        </p>
      </div>

      {/* Training Control Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            تنظیمات آموزش
          </h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                تعداد اپوک‌ها
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={trainingConfig.epochs}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                اندازه دسته (Batch Size)
              </label>
              <select
                value={trainingConfig.batch_size}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value={4}>4</option>
                <option value={8}>8</option>
                <option value={16}>16</option>
                <option value={32}>32</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                نرخ یادگیری (Learning Rate)
              </label>
              <select
                value={trainingConfig.learning_rate}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value={0.0001}>0.0001</option>
                <option value={0.0002}>0.0002</option>
                <option value={0.0005}>0.0005</option>
                <option value={0.001}>0.001</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                نوع مدل
              </label>
              <select
                value={trainingConfig.model_type}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, model_type: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="dora">DoRA (Weight-Decomposed Low-Rank Adaptation)</option>
                <option value="lora">LoRA (Low-Rank Adaptation)</option>
                <option value="full">Fine-tuning کامل</option>
              </select>
            </div>

            <div className="flex gap-3 pt-4">
              <button
                onClick={handleStartTraining}
                disabled={startTrainingMutation.isPending || (activeSession && activeSession.status === 'training')}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg flex items-center justify-center"
              >
                <Play className="h-4 w-4 ml-2" />
                شروع آموزش
              </button>
              
              <button
                onClick={handleStopTraining}
                disabled={!activeSession || activeSession.status !== 'training'}
                className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg flex items-center justify-center"
              >
                <Square className="h-4 w-4 ml-2" />
                توقف آموزش
              </button>
            </div>
          </div>
        </div>

        {/* Active Training Status */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            وضعیت آموزش فعال
          </h2>
          
          {activeSession ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">وضعیت:</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(activeSession.status)}`}>
                  {getStatusText(activeSession.status)}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-600">پیشرفت:</span>
                <span className="font-medium">{activeSession.progress_percent.toFixed(1)}%</span>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${activeSession.progress_percent}%` }}
                ></div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-600">اپوک فعلی:</span>
                <span className="font-medium">{activeSession.current_epoch}/{activeSession.total_epochs}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-600">بهترین دقت:</span>
                <span className="font-medium">{(activeSession.best_accuracy * 100).toFixed(2)}%</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-600">تلفات فعلی:</span>
                <span className="font-medium">{activeSession.current_loss.toFixed(4)}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-gray-600">زمان شروع:</span>
                <span className="font-medium">
                  {new Date(activeSession.start_time).toLocaleString('fa-IR')}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <Settings className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p>هیچ آموزش فعالی در جریان نیست</p>
            </div>
          )}
        </div>
      </div>

      {/* Training History */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          تاریخچه آموزش‌ها
        </h2>
        
        {sessions && sessions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-right py-3 px-4 font-medium text-gray-900">شناسه جلسه</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900">وضعیت</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900">پیشرفت</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900">بهترین دقت</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900">زمان شروع</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-900">عملیات</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((session) => (
                  <tr key={session.session_id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 font-mono text-xs">{session.session_id.substring(0, 16)}...</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(session.status)}`}>
                        {getStatusText(session.status)}
                      </span>
                    </td>
                    <td className="py-3 px-4">{session.progress_percent.toFixed(1)}%</td>
                    <td className="py-3 px-4">{(session.best_accuracy * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4">{new Date(session.start_time).toLocaleDateString('fa-IR')}</td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => setActiveSessionId(session.session_id)}
                        className="text-blue-600 hover:text-blue-800 text-sm"
                      >
                        مشاهده جزئیات
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>هیچ آموزشی تاکنون انجام نشده است</p>
          </div>
        )}
      </div>

      {/* Training Logs */}
      {activeSession && activeSession.logs.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            لاگ‌های آموزش
          </h2>
          <div className="bg-gray-900 text-green-400 rounded-lg p-4 max-h-64 overflow-y-auto font-mono text-sm">
            {activeSession.logs.map((log, index) => (
              <div key={index} className="mb-1">
                {log}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingPage;