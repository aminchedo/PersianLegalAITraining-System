/**
 * React Hook for Training Session Management
 * هوک React برای مدیریت جلسات آموزش
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import PersianLegalAIAPI from '../api/persian-ai-api';

const api = new PersianLegalAIAPI();

export const useTrainingSession = (sessionId) => {
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [progress, setProgress] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  
  const websocketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // اتصال به WebSocket برای نظارت real-time
  const connectWebSocket = useCallback(() => {
    if (!sessionId) return;

    try {
      websocketRef.current = api.createTrainingWebSocket(sessionId);
      setIsConnected(true);
      setError(null);
    } catch (err) {
      console.error('خطا در اتصال WebSocket:', err);
      setError('خطا در اتصال real-time');
      setIsConnected(false);
    }
  }, [sessionId]);

  // قطع اتصال WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (websocketRef.current) {
      api.closeTrainingWebSocket(sessionId);
      websocketRef.current = null;
      setIsConnected(false);
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, [sessionId]);

  // دریافت وضعیت آموزش
  const fetchStatus = useCallback(async () => {
    if (!sessionId) return;

    try {
      setIsLoading(true);
      const response = await api.getTrainingStatus(sessionId);
      
      if (response) {
        setStatus(response.status);
        setProgress(response.progress);
        setMetrics(response.metrics);
        setSystemInfo(response.system_info);
        setError(null);
      }
    } catch (err) {
      console.error('خطا در دریافت وضعیت آموزش:', err);
      setError('خطا در دریافت وضعیت آموزش');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  // دریافت معیارهای آموزش
  const fetchMetrics = useCallback(async (limit = 100) => {
    if (!sessionId) return;

    try {
      const response = await api.getTrainingMetrics(sessionId, limit);
      if (response && response.metrics) {
        setMetrics(response.metrics);
      }
    } catch (err) {
      console.error('خطا در دریافت معیارهای آموزش:', err);
    }
  }, [sessionId]);

  // کنترل آموزش (توقف، مکث، ادامه)
  const controlTraining = useCallback(async (action) => {
    if (!sessionId) return { success: false, error: 'شناسه جلسه نامعتبر' };

    try {
      setIsLoading(true);
      const response = await api.controlTraining(sessionId, action);
      
      if (response.success) {
        // به‌روزرسانی وضعیت پس از کنترل
        setTimeout(() => fetchStatus(), 1000);
      }
      
      return response;
    } catch (err) {
      console.error(`خطا در ${action} آموزش:`, err);
      return { success: false, error: err.message };
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, fetchStatus]);

  // تنظیم event handlers برای WebSocket
  useEffect(() => {
    if (!sessionId) return;

    // Event handler برای به‌روزرسانی وضعیت
    const handleStatusUpdate = (sessionId, data) => {
      if (data.status) setStatus(data.status);
      if (data.progress) setProgress(data.progress);
      if (data.metrics) setMetrics(data.metrics);
      if (data.system_info) setSystemInfo(data.system_info);
    };

    // Event handler برای به‌روزرسانی آموزش
    const handleTrainingUpdate = (sessionId, data) => {
      if (data.status) setStatus(data.status);
      if (data.metrics) {
        setMetrics(prev => {
          if (prev && Array.isArray(prev)) {
            return [data.metrics, ...prev.slice(0, 99)]; // نگه داشتن 100 رکورد آخر
          }
          return [data.metrics];
        });
      }
    };

    // ثبت event handlers
    api.on('training_status_update', handleStatusUpdate);
    api.on('training_training_update', handleTrainingUpdate);

    // اتصال WebSocket
    connectWebSocket();

    // دریافت اولیه وضعیت
    fetchStatus();

    // Cleanup
    return () => {
      api.off('training_status_update', handleStatusUpdate);
      api.off('training_training_update', handleTrainingUpdate);
      disconnectWebSocket();
    };
  }, [sessionId, connectWebSocket, disconnectWebSocket, fetchStatus]);

  // اتصال مجدد خودکار در صورت قطع اتصال
  useEffect(() => {
    if (!isConnected && sessionId && status !== 'completed' && status !== 'failed') {
      reconnectTimeoutRef.current = setTimeout(() => {
        connectWebSocket();
      }, 5000);
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [isConnected, sessionId, status, connectWebSocket]);

  return {
    // State
    status,
    metrics,
    progress,
    systemInfo,
    error,
    isLoading,
    isConnected,
    
    // Actions
    fetchStatus,
    fetchMetrics,
    controlTraining,
    connectWebSocket,
    disconnectWebSocket,
    
    // Computed
    isTraining: status === 'running',
    isPaused: status === 'paused',
    isCompleted: status === 'completed',
    isFailed: status === 'failed',
    progressPercentage: progress?.progress_percentage || 0,
    currentEpoch: progress?.current_epoch || 0,
    totalEpochs: progress?.total_epochs || 0,
    currentStep: progress?.current_step || 0,
    totalSteps: progress?.total_steps || 0
  };
};

export const useTrainingSessions = (statusFilter = null) => {
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSessions = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await api.getTrainingSessions(statusFilter);
      
      if (response && response.sessions) {
        setSessions(response.sessions);
        setError(null);
      }
    } catch (err) {
      console.error('خطا در دریافت جلسات آموزش:', err);
      setError('خطا در دریافت جلسات آموزش');
    } finally {
      setIsLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    fetchSessions();
    
    // به‌روزرسانی دوره‌ای
    const interval = setInterval(fetchSessions, 30000); // هر 30 ثانیه
    
    return () => clearInterval(interval);
  }, [fetchSessions]);

  return {
    sessions,
    isLoading,
    error,
    refetch: fetchSessions
  };
};

export const useAvailableModels = () => {
  const [models, setModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchModels = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await api.getAvailableModels();
      
      if (response && response.models) {
        setModels(response.models);
        setError(null);
      }
    } catch (err) {
      console.error('خطا در دریافت مدل‌های موجود:', err);
      setError('خطا در دریافت مدل‌های موجود');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return {
    models,
    isLoading,
    error,
    refetch: fetchModels
  };
};

export const useTrainingRecommendations = (modelName, taskType) => {
  const [recommendations, setRecommendations] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRecommendations = useCallback(async () => {
    if (!modelName) return;

    try {
      setIsLoading(true);
      const response = await api.getTrainingRecommendations(modelName, taskType);
      
      if (response && response.recommendations) {
        setRecommendations(response.recommendations);
        setError(null);
      }
    } catch (err) {
      console.error('خطا در دریافت توصیه‌های آموزش:', err);
      setError('خطا در دریافت توصیه‌های آموزش');
    } finally {
      setIsLoading(false);
    }
  }, [modelName, taskType]);

  useEffect(() => {
    fetchRecommendations();
  }, [fetchRecommendations]);

  return {
    recommendations,
    isLoading,
    error,
    refetch: fetchRecommendations
  };
};

export const useDataPreparation = () => {
  const [dataInfo, setDataInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const prepareData = useCallback(async (sources, taskType, maxDocuments) => {
    try {
      setIsLoading(true);
      const response = await api.prepareTrainingData(sources, taskType, maxDocuments);
      
      if (response) {
        setDataInfo(response);
        setError(null);
      }
    } catch (err) {
      console.error('خطا در آماده‌سازی داده:', err);
      setError('خطا در آماده‌سازی داده');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    dataInfo,
    isLoading,
    error,
    prepareData
  };
};

export default useTrainingSession;