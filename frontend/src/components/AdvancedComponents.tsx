// components/AdvancedComponents.tsx
/**
 * Advanced Dashboard Components - Real Data Implementation
 * کامپوننت‌های پیشرفته برای داشبورد با داده‌های واقعی
 */

import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { 
  AlertTriangle, CheckCircle, Clock, TrendingUp, TrendingDown, 
  Zap, Activity, Brain, Database, Download, Settings, 
  Play, Pause, Square, Eye, EyeOff, Maximize2, X, 
  Bell, Info, AlertCircle, Users, Cpu, HardDrive, Wifi
} from 'lucide-react';
import { useRealTeamData, useRealModelData, useRealSystemMetrics, useRealSystemStats } from '../hooks/useRealData';
import { RealTeamMember, RealModelTraining, RealSystemMetrics } from '../types/realData';

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

// Real Team Management Component
export const RealTeamManager: React.FC = () => {
  const { data: teamMembers, loading, error, refetch } = useRealTeamData();
  const [selectedMember, setSelectedMember] = useState<RealTeamMember | null>(null);

  if (loading) return <LoadingSpinner message="Loading team data..." />;
  if (error) return <ErrorDisplay message={error} onRetry={refetch} />;

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg">Team Management</h3>
        <div className="flex items-center space-x-2">
          <Users className="h-5 w-5 text-blue-600" />
          <span className="text-sm text-gray-600">{teamMembers.length} members</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {teamMembers.map(member => (
          <div 
            key={member.id}
            className={`p-3 rounded-lg border cursor-pointer transition-all ${
              selectedMember?.id === member.id 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => setSelectedMember(member)}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">{member.name}</span>
              <span className={`px-2 py-1 rounded-full text-xs ${
                member.status === 'online' ? 'bg-green-100 text-green-800' :
                member.status === 'busy' ? 'bg-yellow-100 text-yellow-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {member.status}
              </span>
            </div>
            <div className="text-sm text-gray-600">
              <div>{member.role}</div>
              <div>{member.department}</div>
              <div className="mt-1">
                <span className="text-xs">Tasks: {member.completedTasks}/{member.totalTasks}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {selectedMember && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium mb-2">Member Details</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><strong>Email:</strong> {selectedMember.email}</div>
            <div><strong>Experience:</strong> {selectedMember.experienceYears} years</div>
            <div><strong>Location:</strong> {selectedMember.location}</div>
            <div><strong>Performance:</strong> {selectedMember.performanceScore}%</div>
            <div className="col-span-2">
              <strong>Skills:</strong> {selectedMember.skills.join(', ')}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Real Model Training Component
export const RealModelTraining: React.FC = () => {
  const { data: models, loading, error, refetch } = useRealModelData();
  const [selectedModel, setSelectedModel] = useState<RealModelTraining | null>(null);

  if (loading) return <LoadingSpinner message="Loading model data..." />;
  if (error) return <ErrorDisplay message={error} onRetry={refetch} />;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training': return 'bg-blue-100 text-blue-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg">Model Training</h3>
        <div className="flex items-center space-x-2">
          <Brain className="h-5 w-5 text-purple-600" />
          <span className="text-sm text-gray-600">{models.length} models</span>
        </div>
      </div>

      <div className="space-y-3">
        {models.map(model => (
          <div 
            key={model.id}
            className={`p-3 rounded-lg border cursor-pointer transition-all ${
              selectedModel?.id === model.id 
                ? 'border-purple-500 bg-purple-50' 
                : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => setSelectedModel(model)}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">{model.name}</span>
              <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(model.status)}`}>
                {model.status}
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-sm text-gray-600 mb-2">
              <div>Framework: {model.framework}</div>
              <div>Progress: {model.progress}%</div>
              <div>Accuracy: {model.accuracy}%</div>
              <div>Epochs: {model.currentEpoch}/{model.epochs}</div>
            </div>

            {model.status === 'training' && (
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${model.progress}%` }}
                ></div>
              </div>
            )}
          </div>
        ))}
      </div>

      {selectedModel && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium mb-2">Model Details</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div><strong>Description:</strong> {selectedModel.description || 'N/A'}</div>
            <div><strong>Loss:</strong> {selectedModel.loss}</div>
            <div><strong>Dataset Size:</strong> {selectedModel.datasetSize.toLocaleString()}</div>
            <div><strong>Model Size:</strong> {selectedModel.modelSize} MB</div>
            <div><strong>GPU Usage:</strong> {selectedModel.gpuUsage}%</div>
            <div><strong>Memory Usage:</strong> {selectedModel.memoryUsage} GB</div>
            {selectedModel.timeRemaining && (
              <div className="col-span-2">
                <strong>Time Remaining:</strong> {selectedModel.timeRemaining}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Real System Metrics Component
export const RealSystemMetrics: React.FC = () => {
  const { data: metrics, loading, error, refetch } = useRealSystemMetrics(true, 5000);

  if (loading) return <LoadingSpinner message="Loading system metrics..." />;
  if (error) return <ErrorDisplay message={error} onRetry={refetch} />;
  if (!metrics) return <div>No metrics data available</div>;

  const getHealthColor = (value: number, threshold: number = 80) => {
    if (value >= threshold) return 'text-red-600';
    if (value >= threshold * 0.7) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg">System Metrics</h3>
        <div className="flex items-center space-x-2">
          <Activity className="h-5 w-5 text-green-600" />
          <span className={`text-sm ${metrics.isHealthy ? 'text-green-600' : 'text-red-600'}`}>
            {metrics.isHealthy ? 'Healthy' : 'Warning'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="flex items-center justify-center mb-2">
            <Cpu className="h-5 w-5 text-blue-600 mr-1" />
            <span className="text-sm font-medium">CPU</span>
          </div>
          <div className={`text-2xl font-bold ${getHealthColor(metrics.cpuUsage)}`}>
            {metrics.cpuUsage.toFixed(1)}%
          </div>
        </div>

        <div className="text-center">
          <div className="flex items-center justify-center mb-2">
            <Database className="h-5 w-5 text-purple-600 mr-1" />
            <span className="text-sm font-medium">Memory</span>
          </div>
          <div className={`text-2xl font-bold ${getHealthColor(metrics.memoryUsage)}`}>
            {metrics.memoryUsage.toFixed(1)}%
          </div>
        </div>

        <div className="text-center">
          <div className="flex items-center justify-center mb-2">
            <HardDrive className="h-5 w-5 text-orange-600 mr-1" />
            <span className="text-sm font-medium">Disk</span>
          </div>
          <div className={`text-2xl font-bold ${getHealthColor(metrics.diskUsage)}`}>
            {metrics.diskUsage.toFixed(1)}%
          </div>
        </div>

        <div className="text-center">
          <div className="flex items-center justify-center mb-2">
            <Wifi className="h-5 w-5 text-green-600 mr-1" />
            <span className="text-sm font-medium">Network</span>
          </div>
          <div className="text-2xl font-bold text-gray-700">
            {metrics.networkIn.toFixed(1)} MB/s
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <strong>Temperature:</strong> {metrics.temperature.toFixed(1)}°C
        </div>
        <div>
          <strong>Power:</strong> {metrics.powerConsumption.toFixed(0)}W
        </div>
        <div>
          <strong>Active Connections:</strong> {metrics.activeConnections}
        </div>
        <div>
          <strong>Queue Size:</strong> {metrics.queueSize}
        </div>
      </div>
    </div>
  );
};

// Real System Statistics Component
export const RealSystemStats: React.FC = () => {
  const { data: stats, loading, error, refetch } = useRealSystemStats(true, 30000);

  if (loading) return <LoadingSpinner message="Loading system stats..." />;
  if (error) return <ErrorDisplay message={error} onRetry={refetch} />;
  if (!stats) return <div>No stats data available</div>;

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg">System Statistics</h3>
        <div className="text-sm text-gray-500">
          Last updated: {new Date(stats.timestamp).toLocaleTimeString()}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-3 bg-blue-50 rounded-lg">
          <Users className="h-8 w-8 text-blue-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-blue-600">{stats.teamMembers}</div>
          <div className="text-sm text-gray-600">Team Members</div>
        </div>

        <div className="text-center p-3 bg-purple-50 rounded-lg">
          <Brain className="h-8 w-8 text-purple-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-purple-600">{stats.totalModels}</div>
          <div className="text-sm text-gray-600">Total Models</div>
        </div>

        <div className="text-center p-3 bg-green-50 rounded-lg">
          <Activity className="h-8 w-8 text-green-600 mx-auto mb-2" />
          <div className="text-2xl font-bold text-green-600">{stats.activeModels}</div>
          <div className="text-sm text-gray-600">Active Models</div>
        </div>
      </div>
    </div>
  );
};

// Main Advanced Components Container
export const AdvancedComponents: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RealSystemStats />
        <RealSystemMetrics />
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RealTeamManager />
        <RealModelTraining />
      </div>
    </div>
  );
};

export default AdvancedComponents;