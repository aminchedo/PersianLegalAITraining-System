// api/persian-ai-api.js
/**
 * Persian Legal AI API Integration
 * اتصال داشبورد به سیستم هوش مصنوعی
 */

class PersianLegalAIAPI {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.isConnected = false;
    this.eventHandlers = {};
  }

  // اتصال به سیستم
  async connect() {
    try {
      const response = await fetch(`${this.baseURL}/api/status`);
      if (response.ok) {
        this.isConnected = true;
        console.log('✅ اتصال به سیستم AI برقرار شد');
        return true;
      }
    } catch (error) {
      console.log('⚠️ اتصال مستقیم به سیستم برقرار نشد - حالت نمایشی فعال');
      this.isConnected = false;
    }
    return false;
  }

  // دریافت معیارهای سیستم
  async getSystemMetrics() {
    if (!this.isConnected) {
      // داده‌های نمایشی
      return this.generateMockMetrics();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/metrics`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت معیارها:', error);
      return this.generateMockMetrics();
    }
  }

  // دریافت وضعیت مدل‌ها
  async getModelsStatus() {
    if (!this.isConnected) {
      return this.generateMockModels();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/models`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت وضعیت مدل‌ها:', error);
      return this.generateMockModels();
    }
  }

  // شروع آموزش مدل
  async startTraining(modelConfig) {
    if (!this.isConnected) {
      console.log('🎭 حالت نمایشی: آموزش شروع شد');
      return { success: true, message: 'آموزش در حالت نمایشی شروع شد' };
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modelConfig)
      });
      return await response.json();
    } catch (error) {
      console.error('خطا در شروع آموزش:', error);
      return { success: false, error: error.message };
    }
  }

  // توقف آموزش
  async stopTraining() {
    if (!this.isConnected) {
      console.log('🎭 حالت نمایشی: آموزش متوقف شد');
      return { success: true, message: 'آموزش در حالت نمایشی متوقف شد' };
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/stop`, {
        method: 'POST'
      });
      return await response.json();
    } catch (error) {
      console.error('خطا در توقف آموزش:', error);
      return { success: false, error: error.message };
    }
  }

  // دریافت آمار جمع‌آوری داده
  async getDataCollectionStats() {
    if (!this.isConnected) {
      return this.generateMockDataStats();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/data/stats`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت آمار داده:', error);
      return this.generateMockDataStats();
    }
  }

  // شروع جمع‌آوری داده
  async startDataCollection() {
    if (!this.isConnected) {
      console.log('🎭 حالت نمایشی: جمع‌آوری داده شروع شد');
      return { success: true, message: 'جمع‌آوری داده در حالت نمایشی شروع شد' };
    }

    try {
      const response = await fetch(`${this.baseURL}/api/data/collection/start`, {
        method: 'POST'
      });
      return await response.json();
    } catch (error) {
      console.error('خطا در شروع جمع‌آوری:', error);
      return { success: false, error: error.message };
    }
  }

  // دریافت لاگ‌های سیستم
  async getSystemLogs(limit = 100) {
    if (!this.isConnected) {
      return this.generateMockLogs(limit);
    }

    try {
      const response = await fetch(`${this.baseURL}/api/logs?limit=${limit}`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت لاگ‌ها:', error);
      return this.generateMockLogs(limit);
    }
  }

  // ذخیره تنظیمات
  async saveSettings(settings) {
    if (!this.isConnected) {
      console.log('🎭 حالت نمایشی: تنظیمات ذخیره شد');
      localStorage.setItem('persian-ai-settings', JSON.stringify(settings));
      return { success: true, message: 'تنظیمات در حالت نمایشی ذخیره شد' };
    }

    try {
      const response = await fetch(`${this.baseURL}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      return await response.json();
    } catch (error) {
      console.error('خطا در ذخیره تنظیمات:', error);
      return { success: false, error: error.message };
    }
  }

  // دریافت تنظیمات
  async getSettings() {
    if (!this.isConnected) {
      const saved = localStorage.getItem('persian-ai-settings');
      return saved ? JSON.parse(saved) : this.getDefaultSettings();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/settings`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت تنظیمات:', error);
      return this.getDefaultSettings();
    }
  }

  // ایجاد WebSocket برای به‌روزرسانی real-time
  createWebSocket() {
    if (!this.isConnected) {
      // شبیه‌سازی WebSocket با setInterval
      return this.simulateWebSocket();
    }

    const ws = new WebSocket(`ws://localhost:8000/ws`);
    
    ws.onopen = () => {
      console.log('🔗 اتصال WebSocket برقرار شد');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleWebSocketMessage(data);
    };

    ws.onclose = () => {
      console.log('🔌 اتصال WebSocket قطع شد');
      setTimeout(() => this.createWebSocket(), 5000); // اتصال مجدد
    };

    return ws;
  }

  // شبیه‌سازی WebSocket
  simulateWebSocket() {
    const interval = setInterval(() => {
      this.handleWebSocketMessage({
        type: 'metrics_update',
        data: this.generateMockMetrics()
      });
    }, 3000);

    return {
      close: () => clearInterval(interval)
    };
  }

  // مدیریت پیام‌های WebSocket
  handleWebSocketMessage(message) {
    const { type, data } = message;
    
    if (this.eventHandlers[type]) {
      this.eventHandlers[type].forEach(handler => handler(data));
    }
  }

  // ثبت event handler
  on(event, handler) {
    if (!this.eventHandlers[event]) {
      this.eventHandlers[event] = [];
    }
    this.eventHandlers[event].push(handler);
  }

  // حذف event handler
  off(event, handler) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
    }
  }

  // تولید داده‌های نمایشی
  generateMockMetrics() {
    return {
      cpu_usage: Math.random() * 80 + 10,
      memory_usage: Math.random() * 60 + 20,
      gpu_usage: Math.random() * 90 + 5,
      training_loss: Math.random() * 0.5 + 0.1,
      accuracy: 85 + Math.random() * 10,
      throughput: 80 + Math.random() * 40,
      temperature: 45 + Math.random() * 15,
      power_consumption: 150 + Math.random() * 100,
      timestamp: new Date().toISOString()
    };
  }

  generateMockModels() {
    const statuses = ['training', 'completed', 'pending', 'error'];
    return [
      {
        id: 1,
        name: "PersianMind-v1.0",
        status: statuses[Math.floor(Math.random() * statuses.length)],
        progress: Math.floor(Math.random() * 100),
        accuracy: 85 + Math.random() * 10,
        loss: Math.random() * 0.5,
        epochs_completed: Math.floor(Math.random() * 10),
        time_remaining: "2 ساعت",
        dora_rank: 64,
        learning_rate: 1e-4
      },
      {
        id: 2,
        name: "ParsBERT-Legal",
        status: statuses[Math.floor(Math.random() * statuses.length)],
        progress: Math.floor(Math.random() * 100),
        accuracy: 88 + Math.random() * 8,
        loss: Math.random() * 0.3,
        epochs_completed: Math.floor(Math.random() * 15),
        time_remaining: "45 دقیقه",
        dora_rank: 32,
        learning_rate: 5e-5
      }
    ];
  }

  generateMockDataStats() {
    return {
      total_documents: 33266 + Math.floor(Math.random() * 1000),
      quality_documents: 29000 + Math.floor(Math.random() * 1000),
      sources: {
        naab_corpus: 15420 + Math.floor(Math.random() * 100),
        qavanin_portal: 8932 + Math.floor(Math.random() * 50),
        majles_website: 5673 + Math.floor(Math.random() * 30),
        iran_data_portal: 3241 + Math.floor(Math.random() * 20)
      },
      quality_distribution: {
        excellent: 15000 + Math.floor(Math.random() * 500),
        good: 12000 + Math.floor(Math.random() * 400),
        fair: 5000 + Math.floor(Math.random() * 200),
        poor: 1266 + Math.floor(Math.random() * 100)
      },
      collection_rate: 120 + Math.floor(Math.random() * 40) // documents per hour
    };
  }

  generateMockLogs(limit) {
    const levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG'];
    const messages = [
      'مدل PersianMind آموزش داده شد',
      'جمع‌آوری 500 سند جدید',
      'استفاده از CPU به 85% رسید',
      'ذخیره checkpoint انجام شد',
      'اتصال به منبع داده برقرار شد',
      'بهینه‌سازی DoRA تکمیل شد'
    ];

    return Array.from({ length: limit }, (_, i) => ({
      id: i + 1,
      timestamp: new Date(Date.now() - i * 60000).toISOString(),
      level: levels[Math.floor(Math.random() * levels.length)],
      message: messages[Math.floor(Math.random() * messages.length)],
      component: 'persian-ai-system'
    }));
  }

  getDefaultSettings() {
    return {
      training: {
        learning_rate: 1e-4,
        batch_size: 4,
        num_epochs: 3,
        dora_rank: 64,
        dora_alpha: 16.0,
        enable_decomposition: true
      },
      data_collection: {
        max_workers: 8,
        quality_threshold: 0.7,
        enable_caching: true,
        collection_interval: 3600
      },
      system: {
        cpu_threads: 24,
        monitoring_interval: 5,
        auto_backup: true,
        log_level: 'INFO'
      },
      ui: {
        theme: 'light',
        language: 'fa',
        refresh_interval: 3000,
        notifications_enabled: true
      }
    };
  }
}

// Export for use in React components
export default PersianLegalAIAPI;