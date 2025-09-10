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
  estimated_completion?: string;
  logs: string[];
}

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  gpu_usage?: number;
  disk_usage: number;
  temperature?: number;
  power_consumption?: number;
}

export interface PersianAIState {
  models: Model[];
  trainingSessions: TrainingSession[];
  systemMetrics: SystemMetrics | null;
  isLoading: boolean;
  error: string | null;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
}

type PersianAIAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_CONNECTION_STATUS'; payload: 'connected' | 'disconnected' | 'connecting' }
  | { type: 'SET_MODELS'; payload: Model[] }
  | { type: 'SET_TRAINING_SESSIONS'; payload: TrainingSession[] }
  | { type: 'SET_SYSTEM_METRICS'; payload: SystemMetrics }
  | { type: 'UPDATE_TRAINING_SESSION'; payload: { id: string; updates: Partial<TrainingSession> } }
  | { type: 'ADD_TRAINING_SESSION'; payload: TrainingSession }
  | { type: 'REMOVE_TRAINING_SESSION'; payload: string };

const initialState: PersianAIState = {
  models: [],
  trainingSessions: [],
  systemMetrics: null,
  isLoading: false,
  error: null,
  connectionStatus: 'disconnected',
};

function persianAIReducer(state: PersianAIState, action: PersianAIAction): PersianAIState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'SET_CONNECTION_STATUS':
      return { ...state, connectionStatus: action.payload };
    case 'SET_MODELS':
      return { ...state, models: action.payload };
    case 'SET_TRAINING_SESSIONS':
      return { ...state, trainingSessions: action.payload };
    case 'SET_SYSTEM_METRICS':
      return { ...state, systemMetrics: action.payload };
    case 'UPDATE_TRAINING_SESSION':
      return {
        ...state,
        trainingSessions: state.trainingSessions.map(session =>
          session.id === action.payload.id 
            ? { ...session, ...action.payload.updates }
            : session
        )
      };
    case 'ADD_TRAINING_SESSION':
      return {
        ...state,
        trainingSessions: [...state.trainingSessions, action.payload]
      };
    case 'REMOVE_TRAINING_SESSION':
      return {
        ...state,
        trainingSessions: state.trainingSessions.filter(session => session.id !== action.payload)
      };
    default:
      return state;
  }
}

const PersianAIContext = createContext<{
  state: PersianAIState;
  dispatch: React.Dispatch<PersianAIAction>;
} | null>(null);

export interface PersianAIProviderProps {
  children: ReactNode;
}

export function PersianAIProvider({ children }: PersianAIProviderProps) {
  const [state, dispatch] = useReducer(persianAIReducer, initialState);

  useEffect(() => {
    // Initialize connection and load initial data
    const initializeData = async () => {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'SET_CONNECTION_STATUS', payload: 'connecting' });

      try {
        // Simulate API calls - replace with actual API calls
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Mock data - replace with real API calls
        const mockModels: Model[] = [
          {
            id: '1',
            name: 'Persian Legal Classifier',
            type: 'classification',
            status: 'deployed',
            accuracy: 0.92,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            parameters: {
              epochs: 10,
              batch_size: 32,
              learning_rate: 0.001,
            }
          }
        ];

        const mockSessions: TrainingSession[] = [];

        const mockMetrics: SystemMetrics = {
          cpu_usage: 45.2,
          memory_usage: 67.8,
          disk_usage: 23.1,
          gpu_usage: 78.5,
        };

        dispatch({ type: 'SET_MODELS', payload: mockModels });
        dispatch({ type: 'SET_TRAINING_SESSIONS', payload: mockSessions });
        dispatch({ type: 'SET_SYSTEM_METRICS', payload: mockMetrics });
        dispatch({ type: 'SET_CONNECTION_STATUS', payload: 'connected' });

      } catch (error) {
        dispatch({ type: 'SET_ERROR', payload: 'خطا در بارگذاری اطلاعات' });
        dispatch({ type: 'SET_CONNECTION_STATUS', payload: 'disconnected' });
      } finally {
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    };

    initializeData();
  }, []);

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

// Helper hooks
export function useModels() {
  const { state } = usePersianAI();
  return state.models;
}

export function useTrainingSessions() {
  const { state } = usePersianAI();
  return state.trainingSessions;
}

export function useSystemMetrics() {
  const { state } = usePersianAI();
  return state.systemMetrics;
}

export function useConnectionStatus() {
  const { state } = usePersianAI();
  return state.connectionStatus;
}

// Action creators
export function useTrainingActions() {
  const { dispatch } = usePersianAI();

  const startTraining = (session: TrainingSession) => {
    dispatch({ type: 'ADD_TRAINING_SESSION', payload: session });
  };

  const updateTraining = (id: string, updates: Partial<TrainingSession>) => {
    dispatch({ type: 'UPDATE_TRAINING_SESSION', payload: { id, updates } });
  };

  const stopTraining = (id: string) => {
    dispatch({ type: 'REMOVE_TRAINING_SESSION', payload: id });
  };

  return {
    startTraining,
    updateTraining,
    stopTraining,
  };
}

export function useSystemActions() {
  const { dispatch } = usePersianAI();

  const updateMetrics = (metrics: SystemMetrics) => {
    dispatch({ type: 'SET_SYSTEM_METRICS', payload: metrics });
  };

  const setError = (error: string | null) => {
    dispatch({ type: 'SET_ERROR', payload: error });
  };

  const setLoading = (loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading });
  };

  return {
    updateMetrics,
    setError,
    setLoading,
  };
}