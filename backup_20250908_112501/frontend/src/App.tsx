import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import './App.css';

// Persian UI Components
import PersianLayout from './components/layout/PersianLayout';
import HomePage from './pages/HomePage';
import DocumentsPage from './pages/DocumentsPage';
import TrainingPage from './pages/TrainingPage';
import ClassificationPage from './pages/ClassificationPage';
import SystemPage from './pages/SystemPage';

// Create query client with Persian-optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: 3,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="App" dir="rtl" lang="fa">
          <PersianLayout>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/training" element={<TrainingPage />} />
              <Route path="/classification" element={<ClassificationPage />} />
              <Route path="/system" element={<SystemPage />} />
            </Routes>
          </PersianLayout>
        </div>
      </Router>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;