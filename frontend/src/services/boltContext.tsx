import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { Bolt } from '../types/bolt';

interface BoltState {
  trainingSessions: Bolt.TrainingSession[];
  models: Bolt.Model[];
  documents: Bolt.Document[];
  analytics: Bolt.Analytics | null;
  isLoading: boolean;
  error: string | null;
}

type BoltAction = 
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_TRAINING_SESSIONS'; payload: Bolt.TrainingSession[] }
  | { type: 'SET_MODELS'; payload: Bolt.Model[] }
  | { type: 'SET_DOCUMENTS'; payload: Bolt.Document[] }
  | { type: 'SET_ANALYTICS'; payload: Bolt.Analytics }
  | { type: 'UPDATE_TRAINING_SESSION'; payload: { id: string; updates: Partial<Bolt.TrainingSession> } };

const initialState: BoltState = {
  trainingSessions: [],
  models: [],
  documents: [],
  analytics: null,
  isLoading: false,
  error: null,
};

function boltReducer(state: BoltState, action: BoltAction): BoltState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'SET_TRAINING_SESSIONS':
      return { ...state, trainingSessions: action.payload };
    case 'SET_MODELS':
      return { ...state, models: action.payload };
    case 'SET_DOCUMENTS':
      return { ...state, documents: action.payload };
    case 'SET_ANALYTICS':
      return { ...state, analytics: action.payload };
    case 'UPDATE_TRAINING_SESSION':
      return {
        ...state,
        trainingSessions: state.trainingSessions.map(session =>
          session.id === action.payload.id 
            ? { ...session, ...action.payload.updates }
            : session
        )
      };
    default:
      return state;
  }
}

const BoltContext = createContext<{
  state: BoltState;
  dispatch: React.Dispatch<BoltAction>;
} | null>(null);

export const BoltProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(boltReducer, initialState);

  return (
    <BoltContext.Provider value={{ state, dispatch }}>
      {children}
    </BoltContext.Provider>
  );
};

export const useBolt = () => {
  const context = useContext(BoltContext);
  if (!context) {
    throw new Error('useBolt must be used within a BoltProvider');
  }
  return context;
};
