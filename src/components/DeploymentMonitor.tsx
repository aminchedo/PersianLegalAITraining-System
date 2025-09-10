import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  CheckCircle, 
  AlertCircle, 
  XCircle, 
  RefreshCw, 
  Server, 
  Database, 
  Settings,
  Activity
} from 'lucide-react';
import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { Badge } from './ui/Badge';
import { Loading } from './ui/Loading';

// Types following existing patterns
interface DeploymentStatus {
  status: string;
  timestamp: string;
  services_healthy: boolean;
  docker_available: boolean;
  configuration_valid: boolean;
  resource_sufficient: boolean;
  deployment_ready: boolean;
}

interface DeploymentHealth {
  status: string;
  timestamp: string;
  base_system_health: any;
  deployment_checks: Record<string, boolean>;
  deployment_ready: boolean;
  recommendations: string[];
}

// API service following existing patterns
const deploymentApi = {
  getStatus: async (): Promise<DeploymentStatus> => {
    const response = await fetch('/api/system/deployment/status');
    if (!response.ok) throw new Error('Failed to fetch deployment status');
    return response.json();
  },
  
  getHealth: async (): Promise<DeploymentHealth> => {
    const response = await fetch('/api/system/deployment/health');
    if (!response.ok) throw new Error('Failed to fetch deployment health');
    return response.json();
  },
  
  getRecommendations: async () => {
    const response = await fetch('/api/system/deployment/recommendations');
    if (!response.ok) throw new Error('Failed to fetch recommendations');
    return response.json();
  }
};

const DeploymentMonitor: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'status' | 'health' | 'recommendations'>('status');
  
  // Status query
  const { 
    data: status, 
    isLoading: statusLoading, 
    error: statusError,
    refetch: refetchStatus 
  } = useQuery({
    queryKey: ['deployment-status'],
    queryFn: deploymentApi.getStatus,
    refetchInterval: 30000, // Refetch every 30 seconds
    retry: 2
  });
  
  // Health query
  const { 
    data: health, 
    isLoading: healthLoading,
    error: healthError,
    refetch: refetchHealth 
  } = useQuery({
    queryKey: ['deployment-health'],
    queryFn: deploymentApi.getHealth,
    refetchInterval: 30000,
    retry: 2
  });
  
  // Recommendations query
  const { 
    data: recommendations, 
    isLoading: recommendationsLoading,
    refetch: refetchRecommendations 
  } = useQuery({
    queryKey: ['deployment-recommendations'],
    queryFn: deploymentApi.getRecommendations,
    retry: 2
  });
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'issues_detected':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };
  
  const getStatusBadge = (ready: boolean) => {
    return (
      <Badge 
        variant={ready ? 'success' : 'warning'}
        className="mr-2"
      >
        {ready ? 'آماده استقرار' : 'نیاز به بررسی'}
      </Badge>
    );
  };
  
  const handleRefresh = () => {
    refetchStatus();
    refetchHealth();
    refetchRecommendations();
  };
  
  const renderStatusTab = () => {
    if (statusLoading) return <Loading />;
    if (statusError) return <div className="text-red-500">خطا در دریافت وضعیت</div>;
    if (!status) return null;
    
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getStatusIcon(status.status)}
            <h3 className="text-lg font-semibold">وضعیت استقرار</h3>
            {getStatusBadge(status.deployment_ready)}
          </div>
          <div className="text-sm text-gray-500">
            آخرین بروزرسانی: {new Date(status.timestamp).toLocaleString('fa-IR')}
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4" />
              <span className="font-medium">Docker</span>
              {status.docker_available ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-red-500" />
              )}
            </div>
            <p className="text-sm text-gray-600 mt-1">
              {status.docker_available ? 'در دسترس' : 'نصب نشده'}
            </p>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Settings className="w-4 h-4" />
              <span className="font-medium">پیکربندی</span>
              {status.configuration_valid ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-red-500" />
              )}
            </div>
            <p className="text-sm text-gray-600 mt-1">
              {status.configuration_valid ? 'معتبر' : 'نیاز به بررسی'}
            </p>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              <span className="font-medium">منابع سیستم</span>
              {status.resource_sufficient ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-red-500" />
              )}
            </div>
            <p className="text-sm text-gray-600 mt-1">
              {status.resource_sufficient ? 'کافی' : 'ناکافی'}
            </p>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              <span className="font-medium">سرویس‌ها</span>
              {status.services_healthy ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-red-500" />
              )}
            </div>
            <p className="text-sm text-gray-600 mt-1">
              {status.services_healthy ? 'سالم' : 'نیاز به بررسی'}
            </p>
          </Card>
        </div>
      </div>
    );
  };
  
  const renderHealthTab = () => {
    if (healthLoading) return <Loading />;
    if (healthError) return <div className="text-red-500">خطا در دریافت وضعیت سلامت</div>;
    if (!health) return null;
    
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          {getStatusIcon(health.status)}
          <h3 className="text-lg font-semibold">بررسی سلامت سیستم</h3>
        </div>
        
        <Card className="p-4">
          <h4 className="font-medium mb-3">بررسی‌های استقرار</h4>
          <div className="space-y-2">
            {Object.entries(health.deployment_checks).map(([check, passed]) => (
              <div key={check} className="flex items-center justify-between">
                <span className="text-sm">{check.replace(/_/g, ' ')}</span>
                {passed ? (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-500" />
                )}
              </div>
            ))}
          </div>
        </Card>
        
        {health.recommendations.length > 0 && (
          <Card className="p-4">
            <h4 className="font-medium mb-3">توصیه‌ها</h4>
            <ul className="space-y-1">
              {health.recommendations.map((rec, index) => (
                <li key={index} className="text-sm text-gray-600">
                  • {rec}
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    );
  };
  
  const renderRecommendationsTab = () => {
    if (recommendationsLoading) return <Loading />;
    if (!recommendations) return null;
    
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">راهنمای استقرار</h3>
        
        {recommendations.recommendations?.immediate_actions?.length > 0 && (
          <Card className="p-4">
            <h4 className="font-medium mb-3 text-red-600">اقدامات فوری</h4>
            <div className="space-y-2">
              {recommendations.recommendations.immediate_actions.map((action: any, index: number) => (
                <div key={index} className="border-r-2 border-red-200 pr-3">
                  <div className="font-medium text-sm">{action.action}</div>
                  <div className="text-xs text-gray-600">{action.description}</div>
                  {action.command && (
                    <code className="text-xs bg-gray-100 px-2 py-1 rounded mt-1 block">
                      {action.command}
                    </code>
                  )}
                </div>
              ))}
            </div>
          </Card>
        )}
        
        {recommendations.next_steps?.length > 0 && (
          <Card className="p-4">
            <h4 className="font-medium mb-3">مراحل بعدی</h4>
            <ol className="space-y-1">
              {recommendations.next_steps.map((step: string, index: number) => (
                <li key={index} className="text-sm text-gray-600">
                  {step}
                </li>
              ))}
            </ol>
          </Card>
        )}
      </div>
    );
  };
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">مانیتور استقرار</h2>
        <Button onClick={handleRefresh} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          بروزرسانی
        </Button>
      </div>
      
      {/* Tabs */}
      <div className="flex space-x-1 space-x-reverse bg-gray-100 p-1 rounded-lg">
        {[
          { id: 'status', name: 'وضعیت', icon: Server },
          { id: 'health', name: 'سلامت', icon: Activity },
          { id: 'recommendations', name: 'راهنما', icon: Settings }
        ].map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.name}
            </button>
          );
        })}
      </div>
      
      {/* Tab Content */}
      <div className="min-h-96">
        {activeTab === 'status' && renderStatusTab()}
        {activeTab === 'health' && renderHealthTab()}
        {activeTab === 'recommendations' && renderRecommendationsTab()}
      </div>
    </div>
  );
};

export default DeploymentMonitor;