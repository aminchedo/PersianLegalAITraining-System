import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';

// Types
export interface Model {
  id: string;
  name: string;
  type: 'llm' | 'embedding' | 'classification';
  status: 'idle' | 'training' | 'deployed' | 'error';
  accuracy: number;
  created_at: string;
  updated_at: string;
  parameters: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    rank?: number;
    alpha?: number;
    target_modules?: string[];
    quantization_bits?: number;
    adaptive_rank?: boolean;
  };
}

export interface TrainingSession {
  id: string;
  model_id: string;
  model_name: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  current_epoch: number;
  total_epochs: number;
  loss: number;
  accuracy: number;
  started_at: string;
  estimated_completion: string;
}

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  gpu_usage: number;
  gpu_memory: number;
  disk_usage: number;
  network_io: number;
  temperature: number;
}

export interface DataSource {
  id: string;
  name: string;
  type: 'legal_docs' | 'case_studies' | 'regulations' | 'custom';
  status: 'connected' | 'disconnected' | 'collecting';
  url?: string;
  records_count: number;
  last_updated: string;
  quality_score: number;
}

interface PersianAIState {
  models: Model[];
  trainingSessions: TrainingSession[];
  systemMetrics: SystemMetrics | null;
  dataSources: DataSource[];
  logs: any[];
  isConnected: boolean;
  loading: boolean;
  error: string | null;
}

type PersianAIAction =
  | { type: 'SET_MODELS'; payload: Model[] }
  | { type: 'SET_TRAINING_SESSIONS'; payload: TrainingSession[] }
  | { type: 'SET_SYSTEM_METRICS'; payload: SystemMetrics }
  | { type: 'SET_DATA_SOURCES'; payload: DataSource[] }
  | { type: 'SET_LOGS'; payload: any[] }
  | { type: 'SET_CONNECTED'; payload: boolean }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'UPDATE_TRAINING_SESSION'; payload: TrainingSession }
  | { type: 'ADD_LOG'; payload: any };

const initialState: PersianAIState = {
  models: [],
  trainingSessions: [],
  systemMetrics: null,
  dataSources: [],
  logs: [],
  isConnected: false,
  loading: false,
  error: null,
};

function persianAIReducer(state: PersianAIState, action: PersianAIAction): PersianAIState {
  switch (action.type) {
    case 'SET_MODELS':
      return { ...state, models: action.payload };
    case 'SET_TRAINING_SESSIONS':
      return { ...state, trainingSessions: action.payload };
    case 'SET_SYSTEM_METRICS':
      return { ...state, systemMetrics: action.payload };
    case 'SET_DATA_SOURCES':
      return { ...state, dataSources: action.payload };
    case 'SET_LOGS':
      return { ...state, logs: action.payload };
    case 'SET_CONNECTED':
      return { ...state, isConnected: action.payload };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'UPDATE_TRAINING_SESSION':
      return {
        ...state,
        trainingSessions: state.trainingSessions.map(session =>
          session.id === action.payload.id ? action.payload : session
        ),
      };
    case 'ADD_LOG':
      return { ...state, logs: [action.payload, ...state.logs.slice(0, 99)] };
    default:
      return state;
  }
}

const PersianAIContext = createContext<{
  state: PersianAIState;
  dispatch: React.Dispatch<PersianAIAction>;
} | null>(null);

export function PersianAIProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(persianAIReducer, initialState);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/metrics');
    
    ws.onopen = () => {
      dispatch({ type: 'SET_CONNECTED', payload: true });
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'metrics') {
        dispatch({ type: 'SET_SYSTEM_METRICS', payload: data.payload });
      } else if (data.type === 'training_update') {
        dispatch({ type: 'UPDATE_TRAINING_SESSION', payload: data.payload });
      } else if (data.type === 'log') {
        dispatch({ type: 'ADD_LOG', payload: data.payload });
      }
    };

    ws.onclose = () => {
      dispatch({ type: 'SET_CONNECTED', payload: false });
    };

    ws.onerror = () => {
      dispatch({ type: 'SET_ERROR', payload: 'اتصال WebSocket قطع شد' });
    };

    return () => {
      ws.close();
    };
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      // Simulate API calls
      const mockData = generateMockData();
      dispatch({ type: 'SET_MODELS', payload: mockData.models });
      dispatch({ type: 'SET_TRAINING_SESSIONS', payload: mockData.trainingSessions });
      dispatch({ type: 'SET_DATA_SOURCES', payload: mockData.dataSources });
      dispatch({ type: 'SET_SYSTEM_METRICS', payload: mockData.systemMetrics });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: 'خطا در بارگذاری اطلاعات' });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  return (
    <PersianAIContext.Provider value={{ state, dispatch }}>
      {children}
    </PersianAIContext.Provider>
  );
}

export function usePersianAI() {
  const context = useContext(PersianAIContext);
  if (!context) {
    throw new Error('usePersianAI must be used within PersianAIProvider');
  }
  return context;
}

// Mock data generator
function generateMockData() {
  return {
    models: [
      {
        id: '1',
        name: 'Persian Legal LLM v1.0',
        type: 'llm' as const,
        status: 'training' as const,
        accuracy: 0.87,
        created_at: '2024-01-15T10:30:00Z',
        updated_at: '2024-01-20T14:20:00Z',
        parameters: {
          epochs: 10,
          batch_size: 32,
          learning_rate: 0.001,
          rank: 16,
          alpha: 32,
          target_modules: ['q_proj', 'v_proj'],
        },
      },
      {
        id: '2',
        name: 'Document Classifier',
        type: 'classification' as const,
        status: 'deployed' as const,
        accuracy: 0.92,
        created_at: '2024-01-10T09:15:00Z',
        updated_at: '2024-01-18T11:45:00Z',
        parameters: {
          epochs: 15,
          batch_size: 64,
          learning_rate: 0.0005,
        },
      },
    ],
    trainingSessions: [
      {
        id: 'ts1',
        model_id: '1',
        model_name: 'Persian Legal LLM v1.0',
        status: 'running' as const,
        progress: 65,
        current_epoch: 6,
        total_epochs: 10,
        loss: 0.234,
        accuracy: 0.876,
        started_at: '2024-01-20T09:00:00Z',
        estimated_completion: '2024-01-20T16:30:00Z',
      },
    ],
    dataSources: [
      {
        id: 'ds1',
        name: 'قوانین مدنی ایران',
        type: 'legal_docs' as const,
        status: 'connected' as const,
        records_count: 15420,
        last_updated: '2024-01-20T08:30:00Z',
        quality_score: 0.94,
      },
      {
        id: 'ds2',
        name: 'آرای دیوان عدالت اداری',
        type: 'case_studies' as const,
        status: 'collecting' as const,
        records_count: 8750,
        last_updated: '2024-01-19T15:22:00Z',
        quality_score: 0.88,
      },
    ],
    systemMetrics: {
      cpu_usage: 45,
      memory_usage: 67,
      gpu_usage: 82,
      gpu_memory: 78,
      disk_usage: 34,
      network_io: 15,
      temperature: 65,
    },
  };
}