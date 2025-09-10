import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { 
  Play, Pause, Square, Settings, Download, Upload, Eye, EyeOff, Maximize2, RefreshCw, 
  Bell, Search, Filter, Plus, X, Menu, Home, Brain, Database, Activity, FileText, 
  Users, Zap, Shield, TrendingUp, Clock, Check, AlertTriangle, Info, AlertCircle,
  CheckCircle, Monitor, Cpu, HardDrive, Thermometer, Power, Network, Globe,
  Calendar, BarChart3, Code, Terminal, BookOpen
} from 'lucide-react';
import { useRealTeamData, useRealModelData, useRealSystemMetrics, useRealSystemStats } from '../hooks/useRealData';
import { RealTeamMember, RealModelTraining, RealSystemMetrics } from '../types/realData';
// Bolt Components
import BoltAnalyticsPage from './bolt/pages/analytics-page';
import BoltDataPage from './bolt/pages/data-page';
import BoltModelsPage from './bolt/pages/models-page';
import BoltMonitoringPage from './bolt/pages/monitoring-page';
import BoltSettingsPage from './bolt/pages/settings-page';
import BoltLogsPage from './bolt/pages/logs-page';
import BoltTeam from './bolt/components/team';
import { BoltProvider } from '../services/boltContext';
import { boltApi } from '../api/boltApi';


// Loading component
const LoadingSpinner: React.FC<{ message?: string }> = ({ message = "Loading..." }) => (
  <div className="flex items-center justify-center p-8">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    <span className="ml-2 text-gray-600">{message}</span>
  </div>
);

// Error component
const ErrorDisplay: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
    <div className="flex items-center">
      <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
      <span className="text-red-800">{message}</span>
    </div>
    {onRetry && (
      <button 
        onClick={onRetry}
        className="mt-2 bg-red-600 text-white px-3 py-1 rounded text-sm hover:bg-red-700"
      >
        Retry
      </button>
    )}
  </div>
);

const CompletePersianAIDashboard: React.FC = () => {
  // State Management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(3000);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [fullScreenChart, setFullScreenChart] = useState<string | null>(null);
  const [showNotifications, setShowNotifications] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);
  
  // Real data hooks
  const { data: teamMembers, loading: teamLoading, error: teamError, refetch: refetchTeam } = useRealTeamData();
  const { data: models, loading: modelsLoading, error: modelsError, refetch: refetchModels } = useRealModelData();
  const { data: systemMetrics, loading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useRealSystemMetrics(autoRefresh, refreshInterval);
  const { data: systemStats, loading: statsLoading, error: statsError, refetch: refetchStats } = useRealSystemStats(autoRefresh, 30000);

  // System State
  const [systemStatus, setSystemStatus] = useState({
    isConnected: true,
    trainingActive: false,
    collectionActive: false,
    systemHealth: 'excellent'
  });

  // Update system status based on real data
  useEffect(() => {
    if (systemMetrics) {
      const health = systemMetrics.isHealthy ? 'excellent' : 'warning';
      const trainingActive = models?.some(m => m.status === 'training') || false;
      
      setSystemStatus({
        isConnected: true,
        trainingActive,
        collectionActive: false,
        systemHealth: health
      });
    }
  }, [systemMetrics, models]);

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
  const Sidebar = () => (
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
              systemStatus.isConnected ? 'bg-green-500' : 'bg-red-500'
            }`}></div>
            <span className="text-sm text-gray-300">
              {systemStatus.isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        )}
      </div>

      <nav className="mt-6">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          return (
    <BoltProvider>

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
          
    </BoltProvider>
);
        })}
      </nav>
    </div>
  );

  // Header component
  const Header = () => (
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
            onClick={() => {
              refetchTeam();
              refetchModels();
              refetchMetrics();
              refetchStats();
            }}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700"
          >
            <RefreshCw className="h-4 w-4 inline mr-1" />
            Refresh
          </button>
        </div>
      </div>
    </header>
  );

  // Dashboard content
  const DashboardContent = () => {
    if (statsLoading) return <LoadingSpinner message="Loading system stats..." />;
    if (statsError) return <ErrorDisplay message={statsError} onRetry={refetchStats} />;
    if (!systemStats) return <div>No stats data available</div>;

    return (
      <div className="p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Dashboard</h1>
        
        {/* System Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <Users className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Team Members</p>
                <p className="text-2xl font-semibold text-gray-900">{systemStats.teamMembers}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-purple-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Models</p>
                <p className="text-2xl font-semibold text-gray-900">{systemStats.totalModels}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Active Models</p>
                <p className="text-2xl font-semibold text-gray-900">{systemStats.activeModels}</p>
              </div>
            </div>
          </div>
        </div>

        {/* System Metrics */}
        {systemMetrics && (
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">System Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <Cpu className="h-5 w-5 text-blue-600 mr-1" />
                  <span className="text-sm font-medium">CPU</span>
                </div>
                <div className="text-2xl font-bold text-blue-600">
                  {systemMetrics.cpuUsage.toFixed(1)}%
                </div>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <Database className="h-5 w-5 text-purple-600 mr-1" />
                  <span className="text-sm font-medium">Memory</span>
                </div>
                <div className="text-2xl font-bold text-purple-600">
                  {systemMetrics.memoryUsage.toFixed(1)}%
                </div>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <HardDrive className="h-5 w-5 text-orange-600 mr-1" />
                  <span className="text-sm font-medium">Disk</span>
                </div>
                <div className="text-2xl font-bold text-orange-600">
                  {systemMetrics.diskUsage.toFixed(1)}%
                </div>
              </div>

              <div className="text-center">
                <div className="flex items-center justify-center mb-2">
                  <Network className="h-5 w-5 text-green-600 mr-1" />
                  <span className="text-sm font-medium">Network</span>
                </div>
                <div className="text-2xl font-bold text-green-600">
                  {systemMetrics.networkIn.toFixed(1)} MB/s
                </div>
              </div>
            </div>
          </div>
        )}

        {/* System Overview */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">System Overview</h2>
          <p className="text-gray-600">
            Welcome to the Persian Legal AI System. This dashboard provides real-time monitoring 
            and management of your AI training infrastructure using real data from the database.
          </p>
          <div className="mt-4 flex items-center space-x-4">
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus.systemHealth === 'excellent' ? 'bg-green-500' : 'bg-yellow-500'
              }`}></div>
              <span className="text-sm text-gray-600">
                System Health: {systemStatus.systemHealth}
              </span>
            </div>
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus.trainingActive ? 'bg-blue-500' : 'bg-gray-400'
              }`}></div>
              <span className="text-sm text-gray-600">
                Training: {systemStatus.trainingActive ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Main content area
  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <DashboardContent />;
      case 'team':
        return (
          <div className="p-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">Team Management</h1>
            {teamLoading ? <LoadingSpinner message="Loading team data..." /> :
             teamError ? <ErrorDisplay message={teamError} onRetry={refetchTeam} /> :
             <div className="bg-white p-6 rounded-lg shadow">
               <p className="text-gray-600">Team management content with real data.</p>
               <p className="text-sm text-gray-500 mt-2">Team members: {teamMembers?.length || 0}</p>
             </div>}
          </div>
        );
      case 'models':
        return (
          <div className="p-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">Model Training</h1>
            {modelsLoading ? <LoadingSpinner message="Loading model data..." /> :
             modelsError ? <ErrorDisplay message={modelsError} onRetry={refetchModels} /> :
             <div className="bg-white p-6 rounded-lg shadow">
               <p className="text-gray-600">Model training content with real data.</p>
               <p className="text-sm text-gray-500 mt-2">Training jobs: {models?.length || 0}</p>
             </div>}
          </div>
        );
      
      // Bolt Routes
      case 'bolt-analytics':
        return <BoltAnalyticsPage />;
      case 'bolt-data':
        return <BoltDataPage />;
      case 'bolt-models':
        return <BoltModelsPage />;
      case 'bolt-monitoring':
        return <BoltMonitoringPage />;
      case 'bolt-settings':
        return <BoltSettingsPage />;
      case 'bolt-logs':
        return <BoltLogsPage />;
      case 'bolt-team':
        return <BoltTeam />;
      default:
        return (
          <div className="p-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}</h1>
            <div className="bg-white p-6 rounded-lg shadow">
              <p className="text-gray-600">Content for {activeTab} with real data integration.</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header />
        <main className="flex-1 overflow-auto">
          {renderContent()}
        </main>
      </div>
    </div>
  );
};

export default CompletePersianAIDashboard;