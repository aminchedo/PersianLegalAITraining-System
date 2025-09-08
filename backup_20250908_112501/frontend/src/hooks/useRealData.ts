import { useState, useEffect, useCallback } from 'react';
import { realApiService } from '../services/RealApiService';
import { 
  RealTeamMember, 
  RealModelTraining, 
  RealSystemMetrics,
  RealAnalyticsData,
  RealLegalDocument,
  RealTrainingJob,
  ApiError
} from '../types/realData';

// Custom hook for team data
export const useRealTeamData = (params?: {
  skip?: number;
  limit?: number;
  activeOnly?: boolean;
  department?: string;
}) => {
  const [data, setData] = useState<RealTeamMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const teamData = await realApiService.getTeamMembers(params);
      setData(teamData);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch team data');
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for model training data
export const useRealModelData = (params?: {
  skip?: number;
  limit?: number;
  status?: string;
  framework?: string;
}) => {
  const [data, setData] = useState<RealModelTraining[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const modelData = await realApiService.getTrainingJobs(params);
      setData(modelData);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch model data');
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for system metrics
export const useRealSystemMetrics = (autoRefresh: boolean = true, interval: number = 5000) => {
  const [data, setData] = useState<RealSystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const metrics = await realApiService.getCurrentSystemMetrics();
      setData(metrics);
      setLoading(false);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch system metrics');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    
    if (autoRefresh) {
      const intervalId = setInterval(fetchData, interval);
      return () => clearInterval(intervalId);
    }
  }, [fetchData, autoRefresh, interval]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for system metrics history
export const useRealSystemMetricsHistory = (hours: number = 24) => {
  const [data, setData] = useState<RealSystemMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const history = await realApiService.getSystemMetricsHistory(hours);
      setData(history);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch metrics history');
    } finally {
      setLoading(false);
    }
  }, [hours]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for analytics data
export const useRealAnalyticsData = (params?: {
  metricName?: string;
  hours?: number;
  limit?: number;
}) => {
  const [data, setData] = useState<RealAnalyticsData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const analytics = await realApiService.getAnalyticsData(params);
      setData(analytics);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch analytics data');
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for legal documents
export const useRealLegalDocuments = (params?: {
  skip?: number;
  limit?: number;
  documentType?: string;
  category?: string;
  language?: string;
}) => {
  const [data, setData] = useState<RealLegalDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const documents = await realApiService.getLegalDocuments(params);
      setData(documents);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch legal documents');
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for training jobs
export const useRealTrainingJobs = (params?: {
  skip?: number;
  limit?: number;
  status?: string;
  assignedTo?: number;
}) => {
  const [data, setData] = useState<RealTrainingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const jobs = await realApiService.getTrainingJobsList(params);
      setData(jobs);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch training jobs');
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for system statistics
export const useRealSystemStats = (autoRefresh: boolean = true, interval: number = 30000) => {
  const [data, setData] = useState<{
    teamMembers: number;
    totalModels: number;
    activeModels: number;
    timestamp: string;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const stats = await realApiService.getSystemStats();
      setData(stats);
      setLoading(false);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch system stats');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    
    if (autoRefresh) {
      const intervalId = setInterval(fetchData, interval);
      return () => clearInterval(intervalId);
    }
  }, [fetchData, autoRefresh, interval]);

  return { data, loading, error, refetch: fetchData };
};

// Custom hook for health check
export const useRealHealthCheck = (autoRefresh: boolean = true, interval: number = 60000) => {
  const [data, setData] = useState<{
    status: string;
    timestamp: string;
    system?: Record<string, any>;
    error?: string;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const health = await realApiService.getSystemHealth();
      setData(health);
      setLoading(false);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.detail || 'Failed to fetch health status');
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    
    if (autoRefresh) {
      const intervalId = setInterval(fetchData, interval);
      return () => clearInterval(intervalId);
    }
  }, [fetchData, autoRefresh, interval]);

  return { data, loading, error, refetch: fetchData };
};

// Utility hook for API connection testing
export const useApiConnection = () => {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [testing, setTesting] = useState(false);

  const testConnection = useCallback(async () => {
    setTesting(true);
    try {
      const connected = await realApiService.testConnection();
      setIsConnected(connected);
    } catch (error) {
      setIsConnected(false);
    } finally {
      setTesting(false);
    }
  }, []);

  useEffect(() => {
    testConnection();
  }, [testConnection]);

  return { isConnected, testing, testConnection };
};