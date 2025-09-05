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
    this.websockets = {};
  }

  // اتصال به سیستم
  async connect() {
    try {
      const response = await fetch(`${this.baseURL}/api/system/health`);
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
      const response = await fetch(`${this.baseURL}/api/system/metrics`);
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
      const response = await fetch(`${this.baseURL}/api/models/`);
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
      const response = await fetch(`${this.baseURL}/api/training/sessions`, {
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

  // دریافت وضعیت آموزش
  async getTrainingStatus(sessionId) {
    if (!this.isConnected) {
      return this.generateMockTrainingStatus();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/sessions/${sessionId}/status`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت وضعیت آموزش:', error);
      return this.generateMockTrainingStatus();
    }
  }

  // دریافت معیارهای آموزش
  async getTrainingMetrics(sessionId, limit = 100) {
    if (!this.isConnected) {
      return this.generateMockTrainingMetrics();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/sessions/${sessionId}/metrics?limit=${limit}`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت معیارهای آموزش:', error);
      return this.generateMockTrainingMetrics();
    }
  }

  // کنترل آموزش (توقف، مکث، ادامه)
  async controlTraining(sessionId, action) {
    if (!this.isConnected) {
      console.log(`🎭 حالت نمایشی: آموزش ${action} شد`);
      return { success: true, message: `آموزش در حالت نمایشی ${action} شد` };
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/sessions/${sessionId}/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      return await response.json();
    } catch (error) {
      console.error(`خطا در ${action} آموزش:`, error);
      return { success: false, error: error.message };
    }
  }

  // دریافت لیست جلسات آموزش
  async getTrainingSessions(status = null) {
    if (!this.isConnected) {
      return this.generateMockTrainingSessions();
    }

    try {
      const url = status ? 
        `${this.baseURL}/api/training/sessions?status=${status}` : 
        `${this.baseURL}/api/training/sessions`;
      const response = await fetch(url);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت جلسات آموزش:', error);
      return this.generateMockTrainingSessions();
    }
  }

  // دریافت مدل‌های موجود
  async getAvailableModels() {
    if (!this.isConnected) {
      return this.generateMockAvailableModels();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/models/available`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت مدل‌های موجود:', error);
      return this.generateMockAvailableModels();
    }
  }

  // دریافت توصیه‌های آموزش
  async getTrainingRecommendations(modelName, taskType = 'text_generation') {
    if (!this.isConnected) {
      return this.generateMockRecommendations();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/models/${modelName}/recommendations?task_type=${taskType}`);
      return await response.json();
    } catch (error) {
      console.error('خطا در دریافت توصیه‌های آموزش:', error);
      return this.generateMockRecommendations();
    }
  }

  // آماده‌سازی داده‌های آموزش
  async prepareTrainingData(sources, taskType = 'text_generation', maxDocuments = 1000) {
    if (!this.isConnected) {
      console.log('🎭 حالت نمایشی: آماده‌سازی داده شروع شد');
      return this.generateMockDataPreparation();
    }

    try {
      const response = await fetch(`${this.baseURL}/api/training/data/prepare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sources, task_type: taskType, max_documents: maxDocuments })
      });
      return await response.json();
    } catch (error) {
      console.error('خطا در آماده‌سازی داده:', error);
      return this.generateMockDataPreparation();
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

  // ایجاد WebSocket برای نظارت بر آموزش
  createTrainingWebSocket(sessionId) {
    if (!this.isConnected) {
      // شبیه‌سازی WebSocket برای آموزش
      return this.simulateTrainingWebSocket(sessionId);
    }

    const ws = new WebSocket(`ws://localhost:8000/api/training/ws/training/${sessionId}`);
    
    ws.onopen = () => {
      console.log(`🔗 اتصال WebSocket آموزش برای جلسه ${sessionId} برقرار شد`);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleTrainingWebSocketMessage(sessionId, data);
    };

    ws.onclose = () => {
      console.log(`🔌 اتصال WebSocket آموزش برای جلسه ${sessionId} قطع شد`);
      // اتصال مجدد پس از 5 ثانیه
      setTimeout(() => {
        if (this.websockets[sessionId]) {
          this.websockets[sessionId] = this.createTrainingWebSocket(sessionId);
        }
      }, 5000);
    };

    ws.onerror = (error) => {
      console.error(`خطا در WebSocket آموزش برای جلسه ${sessionId}:`, error);
    };

    // ذخیره WebSocket
    this.websockets[sessionId] = ws;
    return ws;
  }

  // بستن WebSocket آموزش
  closeTrainingWebSocket(sessionId) {
    if (this.websockets[sessionId]) {
      this.websockets[sessionId].close();
      delete this.websockets[sessionId];
    }
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

  // مدیریت پیام‌های WebSocket آموزش
  handleTrainingWebSocketMessage(sessionId, message) {
    const { type, data } = message;
    
    if (this.eventHandlers[`training_${type}`]) {
      this.eventHandlers[`training_${type}`].forEach(handler => handler(sessionId, data));
    }
  }

  // شبیه‌سازی WebSocket آموزش
  simulateTrainingWebSocket(sessionId) {
    const interval = setInterval(() => {
      this.handleTrainingWebSocketMessage(sessionId, {
        type: 'status_update',
        data: this.generateMockTrainingStatus()
      });
    }, 5000);

    return {
      close: () => clearInterval(interval)
    };
  }

  // تولید داده‌های نمایشی برای وضعیت آموزش
  generateMockTrainingStatus() {
    return {
      session_id: 'mock-session-' + Math.random().toString(36).substr(2, 9),
      status: 'running',
      progress: {
        current_epoch: Math.floor(Math.random() * 10),
        total_epochs: 10,
        current_step: Math.floor(Math.random() * 1000),
        total_steps: 1000,
        progress_percentage: Math.floor(Math.random() * 100)
      },
      metrics: {
        current_loss: Math.random() * 0.5 + 0.1,
        best_loss: Math.random() * 0.3 + 0.05,
        current_accuracy: 85 + Math.random() * 10,
        best_accuracy: 90 + Math.random() * 5,
        learning_rate: 1e-4
      },
      system_info: {
        cpu_usage: Math.random() * 80 + 10,
        memory_usage: Math.random() * 60 + 20
      }
    };
  }

  // تولید داده‌های نمایشی برای معیارهای آموزش
  generateMockTrainingMetrics() {
    return {
      session_id: 'mock-session-' + Math.random().toString(36).substr(2, 9),
      metrics: Array.from({ length: 50 }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        epoch: Math.floor(i / 10),
        step: i * 20,
        loss: Math.random() * 0.5 + 0.1,
        accuracy: 85 + Math.random() * 10,
        learning_rate: 1e-4,
        cpu_usage: Math.random() * 80 + 10,
        memory_usage_mb: Math.random() * 1000 + 500
      }))
    };
  }

  // تولید داده‌های نمایشی برای جلسات آموزش
  generateMockTrainingSessions() {
    const statuses = ['running', 'completed', 'failed', 'paused', 'pending'];
    return {
      sessions: Array.from({ length: 5 }, (_, i) => ({
        id: 'session-' + (i + 1),
        model_name: ['PersianMind-v1.0', 'ParsBERT-Legal', 'PersianGPT-2'][i % 3],
        model_type: ['dora', 'qr_adaptor', 'hybrid'][i % 3],
        status: statuses[i % statuses.length],
        created_at: new Date(Date.now() - i * 3600000).toISOString(),
        current_epoch: Math.floor(Math.random() * 10),
        total_epochs: 10,
        current_loss: Math.random() * 0.5 + 0.1,
        best_loss: Math.random() * 0.3 + 0.05
      })),
      total: 5
    };
  }

  // تولید داده‌های نمایشی برای مدل‌های موجود
  generateMockAvailableModels() {
    return {
      models: [
        {
          name: 'PersianMind-v1.0',
          type: 'causal_lm',
          base_model: 'universitytehran/PersianMind-v1.0',
          description: 'مدل زبان فارسی برای تولید متن حقوقی',
          supported_tasks: ['text_generation', 'question_answering']
        },
        {
          name: 'ParsBERT-Legal',
          type: 'bert',
          base_model: 'HooshvareLab/bert-base-parsbert-uncased',
          description: 'مدل BERT فارسی برای درک متن حقوقی',
          supported_tasks: ['text_classification', 'named_entity_recognition']
        }
      ]
    };
  }

  // تولید داده‌های نمایشی برای توصیه‌ها
  generateMockRecommendations() {
    return {
      model_name: 'PersianMind-v1.0',
      task_type: 'text_generation',
      recommendations: {
        model_config: {
          dora_rank: 64,
          dora_alpha: 16.0,
          target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
        },
        training_config: {
          batch_size: 4,
          learning_rate: 1e-4,
          epochs: 10,
          gradient_accumulation_steps: 2
        },
        data_config: {
          max_length: 2048,
          min_length: 100,
          sources: ['naab', 'qavanin']
        }
      }
    };
  }

  // تولید داده‌های نمایشی برای آماده‌سازی داده
  generateMockDataPreparation() {
    return {
      total_documents: 1000 + Math.floor(Math.random() * 500),
      processed_documents: 950 + Math.floor(Math.random() * 50),
      high_quality_documents: 800 + Math.floor(Math.random() * 100),
      training_dataset: {
        task_type: 'text_generation',
        dataset: Array.from({ length: 100 }, (_, i) => ({
          prompt: `سوال ${i + 1}: قانون مدنی چه می‌گوید؟`,
          completion: `پاسخ ${i + 1}: قانون مدنی مجموعه‌ای از قوانین...`,
          source: 'naab'
        })),
        size: 100
      },
      quality_metrics: {
        avg_quality_score: 0.8 + Math.random() * 0.1,
        high_quality_ratio: 0.7 + Math.random() * 0.2
      }
    };
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