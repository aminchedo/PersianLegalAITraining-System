// hooks/usePersianAI.js
/**
 * Custom React Hooks for Persian Legal AI Dashboard
 * Hook های سفارشی برای مدیریت داده‌های داشبورد
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import PersianLegalAIAPI from '../api/persian-ai-api';

// Hook اصلی برای مدیریت سیستم AI
export const usePersianAI = () => {
  const [api] = useState(() => new PersianLegalAIAPI());
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initializeAPI = async () => {
      try {
        const connected = await api.connect();
        setIsConnected(connected);
        setError(null);
      } catch (err) {
        setError(err.message);
        console.error('خطا در اتصال به API:', err);
      } finally {
        setLoading(false);
      }
    };

    initializeAPI();
  }, [api]);

  return {
    api,
    isConnected,
    loading,
    error,
    setError
  };
};

// Hook برای مدیریت معیارهای real-time
export const useRealTimeMetrics = (refreshInterval = 3000) => {
  const { api, isConnected } = usePersianAI();
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const intervalRef = useRef(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const newMetrics = await api.getSystemMetrics();
      setMetrics(newMetrics);
      
      // اضافه کردن به تاریخچه
      setHistory(prev => {
        const updated = [...prev, {
          ...newMetrics,
          time: new Date().toLocaleTimeString('fa-IR')
        }];
        // نگه داشتن فقط 50 نقطه آخر
        return updated.slice(-50);
      });
      
      setLoading(false);
    } catch (error) {
      console.error('خطا در دریافت معیارها:', error);
    }
  }, [api]);

  useEffect(() => {
    fetchMetrics(); // دریافت اولیه

    intervalRef.current = setInterval(fetchMetrics, refreshInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchMetrics, refreshInterval]);

  const updateRefreshInterval = (newInterval) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(fetchMetrics, newInterval);
  };

  return {
    metrics,
    history,
    loading,
    refresh: fetchMetrics,
    updateRefreshInterval
  };
};

// Hook برای مدیریت مدل‌ها
export const useModels = () => {
  const { api } = usePersianAI();
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [trainingActive, setTrainingActive] = useState(false);

  const fetchModels = useCallback(async () => {
    try {
      const modelsData = await api.getModelsStatus();
      setModels(modelsData);
      
      // بررسی اینکه آیا مدلی در حال آموزش است
      const hasTraining = modelsData.some(model => model.status === 'training');
      setTrainingActive(hasTraining);
      
      setLoading(false);
    } catch (error) {
      console.error('خطا در دریافت مدل‌ها:', error);
    }
  }, [api]);

  useEffect(() => {
    fetchModels();
    
    // به‌روزرسانی هر 10 ثانیه
    const interval = setInterval(fetchModels, 10000);
    return () => clearInterval(interval);
  }, [fetchModels]);

  const startTraining = async (modelConfig) => {
    try {
      setTrainingActive(true);
      const result = await api.startTraining(modelConfig);
      await fetchModels(); // به‌روزرسانی فوری
      return result;
    } catch (error) {
      setTrainingActive(false);
      throw error;
    }
  };

  const stopTraining = async () => {
    try {
      const result = await api.stopTraining();
      setTrainingActive(false);
      await fetchModels(); // به‌روزرسانی فوری
      return result;
    } catch (error) {
      throw error;
    }
  };

  return {
    models,
    loading,
    trainingActive,
    startTraining,
    stopTraining,
    refresh: fetchModels
  };
};

// Hook برای مدیریت جمع‌آوری داده
export const useDataCollection = () => {
  const { api } = usePersianAI();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [collectionActive, setCollectionActive] = useState(false);

  const fetchStats = useCallback(async () => {
    try {
      const statsData = await api.getDataCollectionStats();
      setStats(statsData);
      setLoading(false);
    } catch (error) {
      console.error('خطا در دریافت آمار داده:', error);
    }
  }, [api]);

  useEffect(() => {
    fetchStats();
    
    // به‌روزرسانی هر 30 ثانیه
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const startCollection = async () => {
    try {
      setCollectionActive(true);
      const result = await api.startDataCollection();
      await fetchStats(); // به‌روزرسانی فوری
      return result;
    } catch (error) {
      setCollectionActive(false);
      throw error;
    }
  };

  const stopCollection = async () => {
    try {
      // فرض می‌کنیم API متد stop دارد
      setCollectionActive(false);
      await fetchStats();
      return { success: true };
    } catch (error) {
      throw error;
    }
  };

  return {
    stats,
    loading,
    collectionActive,
    startCollection,
    stopCollection,
    refresh: fetchStats
  };
};

// Hook برای مدیریت اعلان‌ها
export const useNotifications = () => {
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      type: 'success',
      title: 'آموزش تکمیل شد',
      message: 'مدل PersianMind با موفقیت آموزش داده شد',
      time: new Date(Date.now() - 2 * 60000),
      read: false
    },
    {
      id: 2,
      type: 'warning',
      title: 'هشدار عملکرد',
      message: 'استفاده از CPU به 85% رسیده است',
      time: new Date(Date.now() - 5 * 60000),
      read: false
    },
    {
      id: 3,
      type: 'info',
      title: 'جمع‌آوری داده',
      message: 'جمع‌آوری 1000 سند جدید تکمیل شد',
      time: new Date(Date.now() - 10 * 60000),
      read: true
    }
  ]);

  const addNotification = (notification) => {
    const newNotification = {
      id: Date.now(),
      time: new Date(),
      read: false,
      ...notification
    };
    setNotifications(prev => [newNotification, ...prev]);
  };

  const markAsRead = (id) => {
    setNotifications(prev =>
      prev.map(notif =>
        notif.id === id ? { ...notif, read: true } : notif
      )
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev =>
      prev.map(notif => ({ ...notif, read: true }))
    );
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  return {
    notifications,
    unreadCount,
    addNotification,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAll
  };
};

// Hook برای مدیریت تنظیمات
export const useSettings = () => {
  const { api } = usePersianAI();
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const loadSettings = async () => {
      try {
        const settingsData = await api.getSettings();
        setSettings(settingsData);
        setLoading(false);
      } catch (error) {
        console.error('خطا در بارگیری تنظیمات:', error);
        setLoading(false);
      }
    };

    loadSettings();
  }, [api]);

  const updateSettings = async (newSettings) => {
    setSaving(true);
    try {
      const result = await api.saveSettings(newSettings);
      if (result.success) {
        setSettings(newSettings);
      }
      setSaving(false);
      return result;
    } catch (error) {
      setSaving(false);
      throw error;
    }
  };

  const resetToDefaults = async () => {
    const defaults = api.getDefaultSettings();
    return await updateSettings(defaults);
  };

  return {
    settings,
    loading,
    saving,
    updateSettings,
    resetToDefaults
  };
};

// Hook برای مدیریت WebSocket
export const useWebSocket = () => {
  const { api, isConnected } = usePersianAI();
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!isConnected) return;

    wsRef.current = api.createWebSocket();
    
    // شنود اتصال WebSocket
    api.on('connected', () => setWsConnected(true));
    api.on('disconnected', () => setWsConnected(false));

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [api, isConnected]);

  const subscribe = (event, handler) => {
    api.on(event, handler);
    
    // برگرداندن تابع unsubscribe
    return () => api.off(event, handler);
  };

  return {
    connected: wsConnected,
    subscribe
  };
};

// Hook برای مدیریت UI State
export const useUI = () => {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('persian-ai-dark-mode');
    return saved ? JSON.parse(saved) : false;
  });

  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    const saved = localStorage.getItem('persian-ai-sidebar-collapsed');
    return saved ? JSON.parse(saved) : false;
  });

  const [fullScreenChart, setFullScreenChart] = useState(null);
  const [showNotifications, setShowNotifications] = useState(false);

  useEffect(() => {
    localStorage.setItem('persian-ai-dark-mode', JSON.stringify(darkMode));
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem('persian-ai-sidebar-collapsed', JSON.stringify(sidebarCollapsed));
  }, [sidebarCollapsed]);

  const toggleDarkMode = () => setDarkMode(prev => !prev);
  const toggleSidebar = () => setSidebarCollapsed(prev => !prev);

  return {
    darkMode,
    sidebarCollapsed,
    fullScreenChart,
    showNotifications,
    setDarkMode,
    setSidebarCollapsed,
    setFullScreenChart,
    setShowNotifications,
    toggleDarkMode,
    toggleSidebar
  };
};

// Hook برای عملکردهای آمار پیشرفته
export const useAdvancedAnalytics = () => {
  const { history } = useRealTimeMetrics();
  
  const calculateTrends = useCallback(() => {
    if (history.length < 2) return {};

    const recent = history.slice(-10);
    const older = history.slice(-20, -10);

    const calculateAverage = (data, key) => {
      return data.reduce((sum, item) => sum + (item[key] || 0), 0) / data.length;
    };

    const recentAvg = {
      cpu: calculateAverage(recent, 'cpu_usage'),
      memory: calculateAverage(recent, 'memory_usage'),
      accuracy: calculateAverage(recent, 'accuracy')
    };

    const olderAvg = {
      cpu: calculateAverage(older, 'cpu_usage'),
      memory: calculateAverage(older, 'memory_usage'),
      accuracy: calculateAverage(older, 'accuracy')
    };

    return {
      cpu_trend: ((recentAvg.cpu - olderAvg.cpu) / olderAvg.cpu * 100).toFixed(1),
      memory_trend: ((recentAvg.memory - olderAvg.memory) / olderAvg.memory * 100).toFixed(1),
      accuracy_trend: ((recentAvg.accuracy - olderAvg.accuracy) / olderAvg.accuracy * 100).toFixed(1)
    };
  }, [history]);

  const predictNextHour = useCallback(() => {
    if (history.length < 10) return null;

    const recent = history.slice(-10);
    const cpuValues = recent.map(item => item.cpu_usage || 0);
    
    // پیش‌بینی ساده بر اساس روند
    const trend = (cpuValues[cpuValues.length - 1] - cpuValues[0]) / cpuValues.length;
    const predicted = cpuValues[cpuValues.length - 1] + (trend * 20); // 20 نقطه آینده

    return {
      cpu_prediction: Math.max(0, Math.min(100, predicted)),
      confidence: Math.max(50, 90 - Math.abs(trend) * 10)
    };
  }, [history]);

  return {
    trends: calculateTrends(),
    prediction: predictNextHour(),
    dataPoints: history.length
  };
};