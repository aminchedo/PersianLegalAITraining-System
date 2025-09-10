/**
 * Performance Monitor Component
 * کامپوننت نظارت بر عملکرد سیستم
 */

import React, { useEffect, useState, useCallback } from 'react';
import { Activity, Zap, Clock, TrendingUp, AlertTriangle } from 'lucide-react';
import { formatPersianNumber, formatPersianFileSize, formatPersianPercentage } from '../utils/persian-utils';

interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  memoryUsage: number;
  networkSpeed: number;
  fps: number;
  bundleSize: number;
  componentCount: number;
  errorCount: number;
}

interface PerformanceMonitorProps {
  enabled?: boolean;
  showDetails?: boolean;
  onMetricsUpdate?: (metrics: PerformanceMetrics) => void;
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  enabled = true,
  showDetails = false,
  onMetricsUpdate
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    loadTime: 0,
    renderTime: 0,
    memoryUsage: 0,
    networkSpeed: 0,
    fps: 0,
    bundleSize: 0,
    componentCount: 0,
    errorCount: 0
  });

  const [isVisible, setIsVisible] = useState(false);
  const [performanceObserver, setPerformanceObserver] = useState<PerformanceObserver | null>(null);

  // Measure page load performance
  const measureLoadPerformance = useCallback(() => {
    if (typeof window === 'undefined') return;

    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    if (navigation) {
      const loadTime = navigation.loadEventEnd - navigation.navigationStart;
      const renderTime = navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart;
      
      setMetrics(prev => ({
        ...prev,
        loadTime: Math.round(loadTime),
        renderTime: Math.round(renderTime)
      }));
    }
  }, []);

  // Measure memory usage
  const measureMemoryUsage = useCallback(() => {
    if (typeof window === 'undefined' || !('memory' in performance)) return;

    const memory = (performance as any).memory;
    if (memory) {
      const memoryUsage = memory.usedJSHeapSize;
      setMetrics(prev => ({
        ...prev,
        memoryUsage
      }));
    }
  }, []);

  // Measure FPS
  const measureFPS = useCallback(() => {
    let frames = 0;
    let lastTime = performance.now();
    let fps = 0;

    const countFrames = () => {
      frames++;
      const currentTime = performance.now();
      
      if (currentTime >= lastTime + 1000) {
        fps = Math.round((frames * 1000) / (currentTime - lastTime));
        frames = 0;
        lastTime = currentTime;
        
        setMetrics(prev => ({
          ...prev,
          fps
        }));
      }
      
      if (enabled) {
        requestAnimationFrame(countFrames);
      }
    };

    if (enabled) {
      requestAnimationFrame(countFrames);
    }
  }, [enabled]);

  // Measure network performance
  const measureNetworkPerformance = useCallback(() => {
    if (typeof navigator === 'undefined' || !('connection' in navigator)) return;

    const connection = (navigator as any).connection;
    if (connection) {
      setMetrics(prev => ({
        ...prev,
        networkSpeed: connection.downlink || 0
      }));
    }
  }, []);

  // Count React components
  const countComponents = useCallback(() => {
    if (typeof window === 'undefined') return;

    // Estimate component count based on DOM elements with React fiber properties
    const elements = document.querySelectorAll('[data-reactroot], [data-react-*]');
    setMetrics(prev => ({
      ...prev,
      componentCount: elements.length
    }));
  }, []);

  // Setup performance observer
  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return;

    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry) => {
        if (entry.entryType === 'measure') {
          // Handle custom measurements
        } else if (entry.entryType === 'navigation') {
          measureLoadPerformance();
        }
      });
    });

    observer.observe({ entryTypes: ['navigation', 'measure', 'paint'] });
    setPerformanceObserver(observer);

    return () => {
      observer.disconnect();
    };
  }, [enabled, measureLoadPerformance]);

  // Initialize measurements
  useEffect(() => {
    if (!enabled) return;

    measureLoadPerformance();
    measureMemoryUsage();
    measureNetworkPerformance();
    countComponents();
    measureFPS();

    // Update metrics periodically
    const interval = setInterval(() => {
      measureMemoryUsage();
      measureNetworkPerformance();
      countComponents();
    }, 5000);

    return () => clearInterval(interval);
  }, [enabled, measureLoadPerformance, measureMemoryUsage, measureNetworkPerformance, countComponents, measureFPS]);

  // Notify parent component of metrics updates
  useEffect(() => {
    if (onMetricsUpdate) {
      onMetricsUpdate(metrics);
    }
  }, [metrics, onMetricsUpdate]);

  // Performance score calculation
  const calculatePerformanceScore = (): number => {
    let score = 100;
    
    // Deduct points for slow load time
    if (metrics.loadTime > 3000) score -= 20;
    else if (metrics.loadTime > 1500) score -= 10;
    
    // Deduct points for high memory usage (> 50MB)
    if (metrics.memoryUsage > 50 * 1024 * 1024) score -= 15;
    else if (metrics.memoryUsage > 25 * 1024 * 1024) score -= 5;
    
    // Deduct points for low FPS
    if (metrics.fps < 30) score -= 20;
    else if (metrics.fps < 50) score -= 10;
    
    // Deduct points for slow network
    if (metrics.networkSpeed < 1) score -= 10;
    
    return Math.max(0, score);
  };

  const performanceScore = calculatePerformanceScore();
  const getScoreColor = (score: number): string => {
    if (score >= 90) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreLabel = (score: number): string => {
    if (score >= 90) return 'عالی';
    if (score >= 70) return 'خوب';
    if (score >= 50) return 'متوسط';
    return 'ضعیف';
  };

  if (!enabled) return null;

  return (
    <>
      {/* Performance Toggle Button */}
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="fixed bottom-4 left-4 z-50 bg-slate-800 text-white p-3 rounded-full shadow-lg hover:bg-slate-700 transition-colors"
        title="نمایش معیارهای عملکرد"
      >
        <Activity className="w-5 h-5" />
      </button>

      {/* Performance Panel */}
      {isVisible && (
        <div className="fixed bottom-20 left-4 z-50 bg-white rounded-lg shadow-xl border border-slate-200 p-4 w-80 max-h-96 overflow-y-auto" dir="rtl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              نظارت بر عملکرد
            </h3>
            <button
              onClick={() => setIsVisible(false)}
              className="text-slate-400 hover:text-slate-600"
            >
              ✕
            </button>
          </div>

          {/* Performance Score */}
          <div className="mb-4 p-3 bg-slate-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-600">امتیاز عملکرد کلی</span>
              <span className={`font-bold text-lg ${getScoreColor(performanceScore)}`}>
                {formatPersianNumber(performanceScore)}
              </span>
            </div>
            <div className="text-xs text-slate-500">
              {getScoreLabel(performanceScore)}
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="space-y-3">
            {/* Load Time */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600 flex items-center gap-2">
                <Clock className="w-3 h-3" />
                زمان بارگذاری
              </span>
              <span className="font-mono">
                {formatPersianNumber(metrics.loadTime)} ms
              </span>
            </div>

            {/* Render Time */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">زمان رندر</span>
              <span className="font-mono">
                {formatPersianNumber(metrics.renderTime)} ms
              </span>
            </div>

            {/* Memory Usage */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">استفاده از حافظه</span>
              <span className="font-mono">
                {formatPersianFileSize(metrics.memoryUsage)}
              </span>
            </div>

            {/* FPS */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">نرخ فریم</span>
              <span className="font-mono">
                {formatPersianNumber(metrics.fps)} FPS
              </span>
            </div>

            {/* Network Speed */}
            {metrics.networkSpeed > 0 && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-600">سرعت شبکه</span>
                <span className="font-mono">
                  {formatPersianNumber(metrics.networkSpeed)} Mbps
                </span>
              </div>
            )}

            {/* Component Count */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">تعداد کامپوننت</span>
              <span className="font-mono">
                {formatPersianNumber(metrics.componentCount)}
              </span>
            </div>
          </div>

          {/* Performance Tips */}
          {performanceScore < 70 && (
            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-yellow-800">
                  <div className="font-medium mb-1">پیشنهادات بهبود عملکرد:</div>
                  <ul className="space-y-1 text-yellow-700">
                    {metrics.loadTime > 3000 && (
                      <li>• بهینه‌سازی اندازه باندل</li>
                    )}
                    {metrics.memoryUsage > 50 * 1024 * 1024 && (
                      <li>• کاهش استفاده از حافظه</li>
                    )}
                    {metrics.fps < 30 && (
                      <li>• بهینه‌سازی انیمیشن‌ها</li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Development Mode Indicator */}
          {process.env.NODE_ENV === 'development' && (
            <div className="mt-3 text-xs text-slate-500 text-center">
              حالت توسعه - آمار دقیق نیست
            </div>
          )}
        </div>
      )}
    </>
  );
};

export default PerformanceMonitor;