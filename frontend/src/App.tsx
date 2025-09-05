import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

// Import components
import LoginForm from './components/auth/LoginForm';
import SystemHealthCard from './components/dashboard/SystemHealthCard';
import TrainingControlPanel from './components/dashboard/TrainingControlPanel';

// Import services
import authService from './services/authService';
import systemService from './services/systemService';

// Import types
import { User } from './types/auth';
import { SystemHealth } from './types/system';

// Dashboard component
const Dashboard: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initializeDashboard = async () => {
      try {
        // Get current user
        const currentUser = authService.getUser();
        if (!currentUser) {
          // Try to get user from API
          const userData = await authService.getCurrentUser();
          setUser(userData);
        } else {
          setUser(currentUser);
        }

        // Get system health
        const health = await systemService.getSystemHealth();
        setSystemHealth(health);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    initializeDashboard();
  }, []);

  const handleLogout = async () => {
    try {
      await authService.logout();
      window.location.href = '/login';
    } catch (err) {
      console.error('Logout failed:', err);
      // Force redirect even if logout API fails
      window.location.href = '/login';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold text-gray-900">
                Persian Legal AI Dashboard
              </h1>
              <span className="ml-2 text-sm text-gray-500">
                سیستم هوش مصنوعی حقوقی فارسی
              </span>
            </div>
            <div className="flex items-center space-x-4">
              {user && (
                <div className="flex items-center space-x-3">
                  <div className="text-sm">
                    <div className="font-medium text-gray-900">{user.full_name}</div>
                    <div className="text-gray-500">{user.username}</div>
                  </div>
                  <div className="flex space-x-1">
                    {user.permissions.map((permission) => (
                      <span
                        key={permission}
                        className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                      >
                        {permission}
                      </span>
                    ))}
                  </div>
                  <button
                    onClick={handleLogout}
                    className="text-sm text-gray-500 hover:text-gray-700"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="space-y-6">
            {/* System Health */}
            <SystemHealthCard />

            {/* Training Control Panel */}
            {authService.hasPermission('training') && <TrainingControlPanel />}

            {/* Additional dashboard content */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Quick Stats */}
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Stats</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-indigo-600">
                        {systemHealth?.checks.system_metrics.cpu.count || 0}
                      </div>
                      <div className="text-sm text-gray-500">CPU Cores</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {systemHealth?.checks.system_metrics.memory.total_gb.toFixed(1) || 0}GB
                      </div>
                      <div className="text-sm text-gray-500">Total Memory</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {systemHealth?.checks.gpu_info.count || 0}
                      </div>
                      <div className="text-sm text-gray-500">GPU Devices</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-600">
                        {systemHealth?.checks.database.active_connections || 0}
                      </div>
                      <div className="text-sm text-gray-500">DB Connections</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* System Information */}
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">System Information</h3>
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="text-sm font-medium text-gray-500">OS</dt>
                      <dd className="text-sm text-gray-900">
                        {systemHealth?.checks.system_metrics.cpu.count || 'Unknown'} cores
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm font-medium text-gray-500">Memory</dt>
                      <dd className="text-sm text-gray-900">
                        {systemHealth?.checks.system_metrics.memory.total_gb.toFixed(1) || 'Unknown'} GB
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm font-medium text-gray-500">GPU</dt>
                      <dd className="text-sm text-gray-900">
                        {systemHealth?.checks.gpu_info.available 
                          ? `${systemHealth.checks.gpu_info.count} devices` 
                          : 'Not available'
                        }
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm font-medium text-gray-500">Database</dt>
                      <dd className="text-sm text-gray-900">
                        {systemHealth?.checks.database.status || 'Unknown'}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm font-medium text-gray-500">SSL</dt>
                      <dd className="text-sm text-gray-900">
                        {systemHealth?.checks.security.ssl.enabled ? 'Enabled' : 'Disabled'}
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>
            </div>

            {/* Security Status */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Security Status</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-3 ${
                      systemHealth?.checks.security.authentication.enabled ? 'bg-green-400' : 'bg-red-400'
                    }`}></div>
                    <span className="text-sm text-gray-700">Authentication</span>
                  </div>
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-3 ${
                      systemHealth?.checks.security.ssl.enabled ? 'bg-green-400' : 'bg-red-400'
                    }`}></div>
                    <span className="text-sm text-gray-700">SSL/TLS</span>
                  </div>
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-3 ${
                      systemHealth?.checks.security.authentication.rate_limiting ? 'bg-green-400' : 'bg-red-400'
                    }`}></div>
                    <span className="text-sm text-gray-700">Rate Limiting</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

// Protected Route component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);

  useEffect(() => {
    const checkAuth = () => {
      const authenticated = authService.isAuthenticated();
      setIsAuthenticated(authenticated);
    };

    checkAuth();
  }, []);

  if (isAuthenticated === null) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
};

// Main App component
const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);

  useEffect(() => {
    const checkAuth = () => {
      const authenticated = authService.isAuthenticated();
      setIsAuthenticated(authenticated);
    };

    checkAuth();
  }, []);

  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
  };

  if (isAuthenticated === null) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route 
          path="/login" 
          element={
            isAuthenticated ? (
              <Navigate to="/dashboard" replace />
            ) : (
              <LoginForm onLoginSuccess={handleLoginSuccess} />
            )
          } 
        />
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/" 
          element={<Navigate to="/dashboard" replace />} 
        />
        <Route 
          path="*" 
          element={<Navigate to="/dashboard" replace />} 
        />
      </Routes>
    </Router>
  );
};

export default App;