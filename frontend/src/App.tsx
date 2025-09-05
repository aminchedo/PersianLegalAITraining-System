import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// TypeScript interfaces
interface SystemHealth {
  status: string;
  timestamp: string;
  system_metrics: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_gb: number;
    disk_percent: number;
    disk_free_gb: number;
    active_processes: number;
  };
  gpu_info: {
    gpu_available: boolean;
    gpu_count?: number;
    gpu_name?: string;
    gpu_memory_total?: number;
    gpu_memory_used?: number;
  };
  platform_info: {
    os: string;
    os_version: string;
    python_version: string;
    architecture: string;
  };
}

interface TrainingSession {
  session_id: string;
  status: string;
  progress: {
    data_loaded: boolean;
    model_initialized: boolean;
    training_started: boolean;
    training_completed: boolean;
    train_samples: number;
    eval_samples: number;
    current_epoch: number;
    total_epochs: number;
    current_step: number;
    total_steps: number;
  };
  metrics: {
    total_steps?: number;
    total_epochs?: number;
    total_loss?: number;
    best_loss?: number;
    training_time?: number;
    avg_loss?: number;
  };
  created_at: string;
  updated_at: string;
}

interface TrainingSessionRequest {
  model_type: 'dora' | 'qr_adaptor';
  model_name: string;
  config: {
    base_model: string;
    dora_rank?: number;
    dora_alpha?: number;
    quantization_bits?: number;
    rank?: number;
    alpha?: number;
    num_epochs: number;
    batch_size: number;
    learning_rate: number;
  };
  data_source: string;
  task_type: string;
}

const API_BASE_URL = 'http://localhost:8000';

const App: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [trainingSessions, setTrainingSessions] = useState<TrainingSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newSession, setNewSession] = useState<TrainingSessionRequest>({
    model_type: 'dora',
    model_name: 'Test Model',
    config: {
      base_model: 'HooshvareLab/bert-base-parsbert-uncased',
      dora_rank: 8,
      dora_alpha: 16,
      num_epochs: 3,
      batch_size: 8,
      learning_rate: 2e-4
    },
    data_source: 'sample',
    task_type: 'text_classification'
  });

  // Fetch system health
  const fetchSystemHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/system/health`);
      setSystemHealth(response.data);
    } catch (err) {
      console.error('Failed to fetch system health:', err);
      setError('Failed to fetch system health');
    }
  };

  // Fetch training sessions
  const fetchTrainingSessions = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/training/sessions`);
      setTrainingSessions(response.data);
    } catch (err) {
      console.error('Failed to fetch training sessions:', err);
      setError('Failed to fetch training sessions');
    }
  };

  // Create new training session
  const createTrainingSession = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/training/sessions`, newSession);
      console.log('Training session created:', response.data);
      
      // Refresh training sessions
      await fetchTrainingSessions();
      
      // Reset form
      setNewSession({
        model_type: 'dora',
        model_name: 'Test Model',
        config: {
          base_model: 'HooshvareLab/bert-base-parsbert-uncased',
          dora_rank: 8,
          dora_alpha: 16,
          num_epochs: 3,
          batch_size: 8,
          learning_rate: 2e-4
        },
        data_source: 'sample',
        task_type: 'text_classification'
      });
      
    } catch (err) {
      console.error('Failed to create training session:', err);
      setError('Failed to create training session');
    } finally {
      setLoading(false);
    }
  };

  // Load data on component mount
  useEffect(() => {
    fetchSystemHealth();
    fetchTrainingSessions();
    
    // Set up auto-refresh
    const interval = setInterval(() => {
      fetchSystemHealth();
      fetchTrainingSessions();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Persian Legal AI Training System</h1>
        <p>سیستم آموزش هوش مصنوعی حقوقی فارسی</p>
      </header>

      <main className="App-main">
        {/* System Health Section */}
        <section className="system-health">
          <h2>System Health</h2>
          {systemHealth ? (
            <div className="health-grid">
              <div className="health-card">
                <h3>CPU Usage</h3>
                <div className="metric-value">{systemHealth.system_metrics.cpu_percent.toFixed(1)}%</div>
              </div>
              <div className="health-card">
                <h3>Memory Usage</h3>
                <div className="metric-value">{systemHealth.system_metrics.memory_percent.toFixed(1)}%</div>
              </div>
              <div className="health-card">
                <h3>Available Memory</h3>
                <div className="metric-value">{systemHealth.system_metrics.memory_available_gb.toFixed(1)} GB</div>
              </div>
              <div className="health-card">
                <h3>Disk Usage</h3>
                <div className="metric-value">{systemHealth.system_metrics.disk_percent.toFixed(1)}%</div>
              </div>
              <div className="health-card">
                <h3>Active Processes</h3>
                <div className="metric-value">{systemHealth.system_metrics.active_processes}</div>
              </div>
              <div className="health-card">
                <h3>GPU Status</h3>
                <div className="metric-value">
                  {systemHealth.gpu_info.gpu_available ? 
                    `Available (${systemHealth.gpu_info.gpu_count || 0})` : 
                    'Not Available'
                  }
                </div>
              </div>
            </div>
          ) : (
            <div className="loading">Loading system health...</div>
          )}
        </section>

        {/* Training Sessions Section */}
        <section className="training-sessions">
          <h2>Training Sessions</h2>
          
          {/* Create New Session Form */}
          <div className="create-session-form">
            <h3>Create New Training Session</h3>
            <div className="form-grid">
              <div className="form-group">
                <label>Model Type:</label>
                <select 
                  value={newSession.model_type} 
                  onChange={(e) => setNewSession({
                    ...newSession,
                    model_type: e.target.value as 'dora' | 'qr_adaptor'
                  })}
                >
                  <option value="dora">DoRA</option>
                  <option value="qr_adaptor">QR-Adaptor</option>
                </select>
              </div>
              
              <div className="form-group">
                <label>Model Name:</label>
                <input 
                  type="text" 
                  value={newSession.model_name}
                  onChange={(e) => setNewSession({
                    ...newSession,
                    model_name: e.target.value
                  })}
                />
              </div>
              
              <div className="form-group">
                <label>Data Source:</label>
                <select 
                  value={newSession.data_source}
                  onChange={(e) => setNewSession({
                    ...newSession,
                    data_source: e.target.value
                  })}
                >
                  <option value="sample">Sample Data</option>
                  <option value="qavanin">Qavanin.ir</option>
                  <option value="majlis">Majlis.ir</option>
                </select>
              </div>
              
              <div className="form-group">
                <label>Task Type:</label>
                <select 
                  value={newSession.task_type}
                  onChange={(e) => setNewSession({
                    ...newSession,
                    task_type: e.target.value
                  })}
                >
                  <option value="text_classification">Text Classification</option>
                  <option value="question_answering">Question Answering</option>
                  <option value="text_generation">Text Generation</option>
                </select>
              </div>
              
              <div className="form-group">
                <label>Epochs:</label>
                <input 
                  type="number" 
                  value={newSession.config.num_epochs}
                  onChange={(e) => setNewSession({
                    ...newSession,
                    config: {
                      ...newSession.config,
                      num_epochs: parseInt(e.target.value)
                    }
                  })}
                />
              </div>
              
              <div className="form-group">
                <label>Batch Size:</label>
                <input 
                  type="number" 
                  value={newSession.config.batch_size}
                  onChange={(e) => setNewSession({
                    ...newSession,
                    config: {
                      ...newSession.config,
                      batch_size: parseInt(e.target.value)
                    }
                  })}
                />
              </div>
            </div>
            
            <button 
              onClick={createTrainingSession} 
              disabled={loading}
              className="create-button"
            >
              {loading ? 'Creating...' : 'Create Training Session'}
            </button>
          </div>

          {/* Training Sessions List */}
          <div className="sessions-list">
            <h3>Active Training Sessions</h3>
            {trainingSessions.length > 0 ? (
              <div className="sessions-grid">
                {trainingSessions.map((session) => (
                  <div key={session.session_id} className="session-card">
                    <div className="session-header">
                      <h4>{session.session_id.substring(0, 8)}...</h4>
                      <span className={`status ${session.status}`}>{session.status}</span>
                    </div>
                    
                    <div className="session-progress">
                      <div className="progress-item">
                        <span>Data Loaded:</span>
                        <span className={session.progress.data_loaded ? 'success' : 'pending'}>
                          {session.progress.data_loaded ? '✓' : '○'}
                        </span>
                      </div>
                      <div className="progress-item">
                        <span>Model Initialized:</span>
                        <span className={session.progress.model_initialized ? 'success' : 'pending'}>
                          {session.progress.model_initialized ? '✓' : '○'}
                        </span>
                      </div>
                      <div className="progress-item">
                        <span>Training Started:</span>
                        <span className={session.progress.training_started ? 'success' : 'pending'}>
                          {session.progress.training_started ? '✓' : '○'}
                        </span>
                      </div>
                      <div className="progress-item">
                        <span>Training Completed:</span>
                        <span className={session.progress.training_completed ? 'success' : 'pending'}>
                          {session.progress.training_completed ? '✓' : '○'}
                        </span>
                      </div>
                    </div>
                    
                    {session.metrics && Object.keys(session.metrics).length > 0 && (
                      <div className="session-metrics">
                        <h5>Metrics:</h5>
                        <div className="metrics-grid">
                          {session.metrics.avg_loss && (
                            <div className="metric">
                              <span>Avg Loss:</span>
                              <span>{session.metrics.avg_loss.toFixed(4)}</span>
                            </div>
                          )}
                          {session.metrics.training_time && (
                            <div className="metric">
                              <span>Training Time:</span>
                              <span>{session.metrics.training_time.toFixed(2)}s</span>
                            </div>
                          )}
                          {session.metrics.total_steps && (
                            <div className="metric">
                              <span>Total Steps:</span>
                              <span>{session.metrics.total_steps}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    <div className="session-info">
                      <small>Created: {new Date(session.created_at).toLocaleString()}</small>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-sessions">No training sessions found</div>
            )}
          </div>
        </section>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            <h3>Error:</h3>
            <p>{error}</p>
            <button onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;