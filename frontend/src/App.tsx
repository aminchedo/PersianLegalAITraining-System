import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import './App.css';

// Persian UI Components
import PersianLayout from './components/layout/PersianLayout';
import ErrorBoundary from './components/ErrorBoundary';

// Lazy load pages for better performance
const HomePage = lazy(() => import('./pages/HomePage'));
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'));
const TrainingPage = lazy(() => import('./pages/TrainingPage'));
const ClassificationPage = lazy(() => import('./pages/ClassificationPage'));
const SystemPage = lazy(() => import('./pages/SystemPage'));


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
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className="App font-vazir" dir="rtl" lang="fa">
            <PersianLayout>
              <Suspense fallback={
                <div className="flex items-center justify-center min-h-screen">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                  <span className="mr-3 text-gray-600">در حال بارگذاری...</span>
                </div>
              }>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/documents" element={<DocumentsPage />} />
                  <Route path="/training" element={<TrainingPage />} />
                  <Route path="/classification" element={<ClassificationPage />} />
                  <Route path="/system" element={<SystemPage />} />
                </Routes>
              </Suspense>
            </PersianLayout>
          </div>
        </Router>
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;