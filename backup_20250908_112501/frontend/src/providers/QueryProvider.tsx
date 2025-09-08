/**
 * React Query Provider for Persian Legal AI
 * ارائه‌دهنده React Query برای سیستم هوش مصنوعی حقوقی فارسی
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ReactNode } from 'react';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Global query options
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: (failureCount, error) => {
        // Don't retry on 4xx errors
        if (error instanceof Error && error.message.includes('4')) {
          return false;
        }
        // Retry up to 3 times for other errors
        return failureCount < 3;
      },
      refetchOnWindowFocus: false, // Don't refetch on window focus
    },
    mutations: {
      // Global mutation options
      retry: 1, // Retry failed mutations once
    },
  },
});

interface QueryProviderProps {
  children: ReactNode;
}

export function QueryProvider({ children }: QueryProviderProps) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* Show React Query DevTools in development */}
      {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  );
}

export { queryClient };