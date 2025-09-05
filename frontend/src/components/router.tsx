import React, { createContext, useContext, useState, useEffect } from 'react';
import { Brain, Database, Activity, BarChart3, FileText, Terminal, Users, Settings, Home } from 'lucide-react';
import { useRealTeamData, useRealModelData, useRealSystemStats, useApiConnection } from '../hooks/useRealData';

// Context for global state management
interface AppContextType {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  autoRefresh: boolean;
  setAutoRefresh: (refresh: boolean) => void;
  refreshInterval: number;
  setRefreshInterval: (interval: number) => void;
  isConnected: boolean | null;
  connectionTesting: boolean;
  testConnection: () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
};

// Global state provider with real data
export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(3000);
  
  // Real data hooks
  const { isConnected, testing: connectionTesting, testConnection } = useApiConnection();
  
  const contextValue: AppContextType = {
    activeTab,
    setActiveTab,
    sidebarCollapsed,
    setSidebarCollapsed,
    autoRefresh,
    setAutoRefresh,
    refreshInterval,
    setRefreshInterval,
    isConnected,
    connectionTesting,
    testConnection
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

// Navigation items
const navigationItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Home },
  { id: 'team', label: 'Team', icon: Users },
  { id: 'models', label: 'Models', icon: Brain },
  { id: 'monitoring', label: 'Monitoring', icon: Activity },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'data', label: 'Data', icon: Database },
  { id: 'logs', label: 'Logs', icon: Terminal },
  { id: 'settings', label: 'Settings', icon: Settings },
];

// Sidebar component
export const Sidebar: React.FC = () => {
  const { activeTab, setActiveTab, sidebarCollapsed, setSidebarCollapsed, isConnected } = useAppContext();

  return (
    <div className={`bg-gray-900 text-white transition-all duration-300 ${
      sidebarCollapsed ? 'w-16' : 'w-64'
    }`}>
      <div className="p-4">
        <div className="flex items-center justify-between">
          {!sidebarCollapsed && (
            <h1 className="text-xl font-bold">Persian Legal AI</h1>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <Settings className="h-5 w-5" />
          </button>
        </div>
        
        {!sidebarCollapsed && (
          <div className="mt-4 flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              isConnected === true ? 'bg-green-500' : 
              isConnected === false ? 'bg-red-500' : 'bg-yellow-500'
            }`}></div>
            <span className="text-sm text-gray-300">
              {isConnected === true ? 'Connected' : 
               isConnected === false ? 'Disconnected' : 'Connecting...'}
            </span>
          </div>
        )}
      </div>

      <nav className="mt-6">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center px-4 py-3 text-left hover:bg-gray-800 transition-colors ${
                activeTab === item.id ? 'bg-gray-800 border-r-2 border-blue-500' : ''
              }`}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              {!sidebarCollapsed && (
                <span className="ml-3">{item.label}</span>
              )}
            </button>
          );
        })}
      </nav>
    </div>
  );
};

// Header component
export const Header: React.FC = () => {
  const { autoRefresh, setAutoRefresh, refreshInterval, setRefreshInterval, testConnection, connectionTesting } = useAppContext();

  return (
    <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-gray-900">Persian Legal AI System</h2>
          <p className="text-sm text-gray-600">Real-time monitoring and management</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-600">Auto Refresh:</label>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
          </div>
          
          {autoRefresh && (
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-600">Interval:</label>
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                className="border border-gray-300 rounded px-2 py-1 text-sm"
              >
                <option value={1000}>1s</option>
                <option value={3000}>3s</option>
                <option value={5000}>5s</option>
                <option value={10000}>10s</option>
                <option value={30000}>30s</option>
              </select>
            </div>
          )}
          
          <button
            onClick={testConnection}
            disabled={connectionTesting}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {connectionTesting ? 'Testing...' : 'Test Connection'}
          </button>
        </div>
      </div>
    </header>
  );
};

// Main content area
export const MainContent: React.FC = () => {
  const { activeTab } = useAppContext();

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardContent />;
      case 'team':
        return <TeamContent />;
      case 'models':
        return <ModelsContent />;
      case 'monitoring':
        return <MonitoringContent />;
      case 'analytics':
        return <AnalyticsContent />;
      case 'data':
        return <DataContent />;
      case 'logs':
        return <LogsContent />;
      case 'settings':
        return <SettingsContent />;
      default:
        return <DashboardContent />;
    }
  };

  return (
    <main className="flex-1 overflow-auto">
      {renderContent()}
    </main>
  );
};

// Content components
const DashboardContent: React.FC = () => {
  const { data: stats, loading, error } = useRealSystemStats();

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">Error loading dashboard: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Dashboard</h1>
      
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <Users className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Team Members</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.teamMembers}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-purple-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Models</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.totalModels}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Active Models</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.activeModels}</p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">System Overview</h2>
        <p className="text-gray-600">
          Welcome to the Persian Legal AI System. This dashboard provides real-time monitoring 
          and management of your AI training infrastructure.
        </p>
      </div>
    </div>
  );
};

const TeamContent: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Team Management</h1>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-gray-600">Team management content will be loaded here with real data.</p>
      </div>
    </div>
  );
};

const ModelsContent: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Model Training</h1>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-gray-600">Model training content will be loaded here with real data.</p>
      </div>
    </div>
  );
};

const MonitoringContent: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">System Monitoring</h1>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-gray-600">System monitoring content will be loaded here with real data.</p>
      </div>
    </div>
  );
};

const AnalyticsContent: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Analytics</h1>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-gray-600">Analytics content will be loaded here with real data.</p>
      </div>
    </div>
  );
};

const DataContent: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Data Management</h1>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-gray-600">Data management content will be loaded here with real data.</p>
      </div>
    </div>
  );
};

const LogsContent: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">System Logs</h1>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-gray-600">System logs content will be loaded here with real data.</p>
      </div>
    </div>
  );
};

const SettingsContent: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Settings</h1>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-gray-600">Settings content will be loaded here with real data.</p>
      </div>
    </div>
  );
};

// Main App Layout
export const AppLayout: React.FC = () => {
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header />
        <MainContent />
      </div>
    </div>
  );
};

// Main App component
export const App: React.FC = () => {
  return (
    <AppProvider>
      <AppLayout />
    </AppProvider>
  );
};

export default App;