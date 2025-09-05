import React, { useState, useEffect } from 'react';
import { SystemHealth } from '../../types/system';
import systemService from '../../services/systemService';

const SystemHealthCard: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const healthData = await systemService.getSystemHealth();
        setHealth(healthData);
        setError(null);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchHealth();
    
    // Set up periodic updates
    const interval = setInterval(fetchHealth, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'degraded':
        return 'text-yellow-600 bg-yellow-100';
      case 'unhealthy':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return '✓';
      case 'degraded':
        return '⚠';
      case 'unhealthy':
        return '✗';
      default:
        return '?';
    }
  };

  if (loading) {
    return (
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
            <div className="h-3 bg-gray-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="text-red-600">
            <h3 className="text-lg font-medium">System Health</h3>
            <p className="text-sm">Error: {error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!health) {
    return null;
  }

  const { checks } = health;
  const { system_metrics, gpu_info, database, services } = checks;

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">System Health</h3>
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(health.overall_health)}`}>
            <span className="mr-1">{getStatusIcon(health.overall_health)}</span>
            {health.overall_health}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* CPU Usage */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-500">CPU</span>
              <span className="text-sm text-gray-900">
                {system_metrics.cpu.usage_percent.toFixed(1)}%
              </span>
            </div>
            <div className="mt-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${system_metrics.cpu.usage_percent}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* Memory Usage */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-500">Memory</span>
              <span className="text-sm text-gray-900">
                {system_metrics.memory.usage_percent.toFixed(1)}%
              </span>
            </div>
            <div className="mt-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full"
                  style={{ width: `${system_metrics.memory.usage_percent}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* GPU Status */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-500">GPU</span>
              <span className="text-sm text-gray-900">
                {gpu_info.available ? `${gpu_info.count} devices` : 'N/A'}
              </span>
            </div>
            {gpu_info.available && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full"
                    style={{ width: `${gpu_info.utilization}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {gpu_info.memory_used.toFixed(1)}GB / {gpu_info.memory_total.toFixed(1)}GB
                </div>
              </div>
            )}
          </div>

          {/* Database Status */}
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-500">Database</span>
              <span className={`text-xs px-2 py-1 rounded-full ${getStatusColor(database.status)}`}>
                {database.status}
              </span>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {database.connection_time.toFixed(3)}s
            </div>
          </div>
        </div>

        {/* Services Status */}
        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Services</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(services).map(([service, status]) => (
              <span
                key={service}
                className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                  status === 'healthy' || status === 'active' || status === 'available'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-red-100 text-red-800'
                }`}
              >
                {service}: {status}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-4 text-xs text-gray-500">
          Last updated: {new Date(health.timestamp).toLocaleString()}
        </div>
      </div>
    </div>
  );
};

export default SystemHealthCard;