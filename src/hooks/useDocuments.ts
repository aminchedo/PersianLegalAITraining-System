import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiService from '../services/apiService';
// Types are used in function return types and parameters

// Main documents hooks
export const useDocuments = (query?: string, category?: string) => {
  return useQuery({
    queryKey: ['documents', query, category],
    queryFn: () => apiService.searchDocuments(query || '', category),
    enabled: !!query,
  });
};

export const useDocument = (id: string) => {
  return useQuery({
    queryKey: ['document', id],
    queryFn: () => apiService.getDocument(id),
    enabled: !!id,
  });
};

export const useDocumentsByCategory = (category: string) => {
  return useQuery({
    queryKey: ['documents', 'category', category],
    queryFn: () => apiService.getDocumentsByCategory(category),
    enabled: !!category,
  });
};

// Scraping status hooks - CRITICAL EXPORTS
export const useScrapingStatus = () => {
  return useQuery({
    queryKey: ['scrapingStatus'],
    queryFn: () => apiService.getScrapingStatus(),
    refetchInterval: 2000, // Real-time updates every 2 seconds
    refetchIntervalInBackground: true,
  });
};

export const useStartScraping = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sources: string[]) => apiService.startScraping(sources),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scrapingStatus'] });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
};

export const useStopScraping = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => apiService.stopScraping(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scrapingStatus'] });
    },
  });
};

// Classification hooks
export const useClassifyDocument = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => apiService.classifyDocument(documentId),
    onSuccess: (_, documentId) => {
      queryClient.invalidateQueries({ queryKey: ['document', documentId] });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
};