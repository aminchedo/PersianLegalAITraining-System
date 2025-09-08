/**
 * React hooks for Persian Legal AI API
 * هوک‌های React برای API سیستم هوش مصنوعی حقوقی فارسی
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  persianApiService, 
  DocumentSearchRequest, 
  ClassificationRequest, 
  TrainingStartRequest 
} from '../services/persianApiService';

// System hooks
export const useSystemHealth = () => {
  return useQuery({
    queryKey: ['systemHealth'],
    queryFn: () => persianApiService.getSystemHealth(),
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 20000, // Consider data stale after 20 seconds
  });
};

export const useSystemStatus = () => {
  return useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => persianApiService.getSystemStatus(),
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 5000, // Consider data stale after 5 seconds
  });
};

// Document hooks
export const useSearchDocuments = () => {
  return useMutation({
    mutationFn: (request: DocumentSearchRequest) => 
      persianApiService.searchDocuments(request),
  });
};

export const useInsertDocument = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (document: {
      title: string;
      content: string;
      source_url?: string;
      document_type?: string;
      category?: string;
      subcategory?: string;
      persian_date?: string;
    }) => persianApiService.insertDocument(document),
    onSuccess: () => {
      // Invalidate document stats to refresh the UI
      queryClient.invalidateQueries({ queryKey: ['documentStats'] });
      queryClient.invalidateQueries({ queryKey: ['systemStatus'] });
    },
  });
};

export const useDocumentStats = () => {
  return useQuery({
    queryKey: ['documentStats'],
    queryFn: () => persianApiService.getDocumentStats(),
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000, // Consider data stale after 30 seconds
  });
};

// AI hooks
export const useClassifyDocument = () => {
  return useMutation({
    mutationFn: (request: ClassificationRequest) => 
      persianApiService.classifyDocument(request),
  });
};

export const useModelInfo = () => {
  return useQuery({
    queryKey: ['modelInfo'],
    queryFn: () => persianApiService.getModelInfo(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 2 * 60 * 1000, // Refresh every 2 minutes
  });
};

// Training hooks
export const useStartTraining = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (request: TrainingStartRequest) => 
      persianApiService.startTraining(request),
    onSuccess: () => {
      // Refresh training sessions and system status
      queryClient.invalidateQueries({ queryKey: ['trainingSessions'] });
      queryClient.invalidateQueries({ queryKey: ['systemStatus'] });
    },
  });
};

export const useTrainingStatus = (sessionId: string, enabled: boolean = true) => {
  return useQuery({
    queryKey: ['trainingStatus', sessionId],
    queryFn: () => persianApiService.getTrainingStatus(sessionId),
    enabled: enabled && !!sessionId,
    refetchInterval: (data) => {
      // Refresh more frequently for active sessions
      if (data?.status === 'running') return 2000; // 2 seconds
      if (data?.status === 'starting') return 5000; // 5 seconds
      return 30000; // 30 seconds for completed/failed sessions
    },
    staleTime: 1000, // Always consider training data stale
  });
};

export const useTrainingSessions = () => {
  return useQuery({
    queryKey: ['trainingSessions'],
    queryFn: () => persianApiService.listTrainingSessions(),
    refetchInterval: 15000, // Refresh every 15 seconds
    staleTime: 10000, // Consider data stale after 10 seconds
  });
};

// Connection test hook
export const useTestConnection = () => {
  return useMutation({
    mutationFn: () => persianApiService.testConnection(),
  });
};

// Custom hook for real-time training monitoring
export const useTrainingMonitor = (sessionIds: string[]) => {
  const queryClient = useQueryClient();
  
  const queries = sessionIds.map(sessionId => ({
    queryKey: ['trainingStatus', sessionId],
    queryFn: () => persianApiService.getTrainingStatus(sessionId),
    refetchInterval: 3000, // 3 seconds
    staleTime: 1000,
  }));
  
  return useQuery({
    queryKey: ['trainingMonitor', sessionIds],
    queryFn: async () => {
      const results = await Promise.allSettled(
        sessionIds.map(id => persianApiService.getTrainingStatus(id))
      );
      
      return results.map((result, index) => ({
        sessionId: sessionIds[index],
        status: result.status === 'fulfilled' ? result.value : null,
        error: result.status === 'rejected' ? result.reason : null,
      }));
    },
    enabled: sessionIds.length > 0,
    refetchInterval: 3000,
    staleTime: 1000,
  });
};

// Helper hook for managing search state
export const useSearchState = () => {
  const searchMutation = useSearchDocuments();
  
  const search = async (query: string, filters?: {
    category?: string;
    document_type?: string;
    limit?: number;
    offset?: number;
  }) => {
    return searchMutation.mutateAsync({
      query,
      ...filters,
    });
  };
  
  return {
    search,
    isSearching: searchMutation.isPending,
    searchResults: searchMutation.data,
    searchError: searchMutation.error,
    resetSearch: searchMutation.reset,
  };
};

// Helper hook for managing classification state
export const useClassificationState = () => {
  const classifyMutation = useClassifyDocument();
  
  const classify = async (text: string, returnProbabilities: boolean = true) => {
    return classifyMutation.mutateAsync({
      text,
      return_probabilities: returnProbabilities,
    });
  };
  
  return {
    classify,
    isClassifying: classifyMutation.isPending,
    classificationResult: classifyMutation.data,
    classificationError: classifyMutation.error,
    resetClassification: classifyMutation.reset,
  };
};