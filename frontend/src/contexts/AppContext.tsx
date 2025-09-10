'use client'

import React, { createContext, useContext, useReducer, ReactNode } from 'react'

// Types
interface AppState {
  user: any | null
  isAuthenticated: boolean
  theme: 'light' | 'dark'
  language: 'fa' | 'en'
  loading: boolean
  error: string | null
}

type AppAction = 
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_USER'; payload: any }
  | { type: 'LOGOUT' }
  | { type: 'SET_THEME'; payload: 'light' | 'dark' }
  | { type: 'SET_LANGUAGE'; payload: 'fa' | 'en' }

interface AppContextType {
  state: AppState
  dispatch: React.Dispatch<AppAction>
}

// Initial State
const initialState: AppState = {
  user: null,
  isAuthenticated: false,
  theme: 'light',
  language: 'fa',
  loading: false,
  error: null
}

// Reducer
const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload }
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false }
    case 'SET_USER':
      return { ...state, user: action.payload, isAuthenticated: !!action.payload }
    case 'LOGOUT':
      return { ...state, user: null, isAuthenticated: false }
    case 'SET_THEME':
      return { ...state, theme: action.payload }
    case 'SET_LANGUAGE':
      return { ...state, language: action.payload }
    default:
      return state
  }
}

// Context
const AppContext = createContext<AppContextType | undefined>(undefined)

// Provider
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState)

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  )
}

// Hook
export const useAppContext = () => {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider')
  }
  return context
}

export default AppContext