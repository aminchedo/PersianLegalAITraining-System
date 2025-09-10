import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import Router from './components/router';
import { PersianAIProvider } from './hooks/usePersianAI';
import './index.css';

function App() {
  return (
    <div dir="rtl" className="min-h-screen bg-gray-50">
      <PersianAIProvider>
        <BrowserRouter>
          <Router />
        </BrowserRouter>
      </PersianAIProvider>
    </div>
  );
}

export default App;