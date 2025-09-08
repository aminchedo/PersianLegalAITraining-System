// System service for Persian Legal AI Frontend
import axios, { AxiosResponse } from 'axios';
import { SystemHealth, SystemInfo } from '../types/system';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class SystemService {
  async getSystemHealth(): Promise<SystemHealth> {
    try {
      const response: AxiosResponse<SystemHealth> = await axios.get(
        `${API_BASE_URL}/api/system/health`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to get system health:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get system health');
    }
  }

  async getSimpleHealth(): Promise<{ status: string; timestamp: string; database: string }> {
    try {
      const response: AxiosResponse<{ status: string; timestamp: string; database: string }> = await axios.get(
        `${API_BASE_URL}/api/system/health/simple`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to get simple health:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get simple health');
    }
  }

  async getSystemInfo(): Promise<SystemInfo> {
    try {
      const response: AxiosResponse<SystemInfo> = await axios.get(
        `${API_BASE_URL}/api/system/info`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to get system info:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get system info');
    }
  }

  async getSystemMetrics(): Promise<any> {
    try {
      const response: AxiosResponse<any> = await axios.get(
        `${API_BASE_URL}/api/system/metrics`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to get system metrics:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get system metrics');
    }
  }

  // WebSocket connection for real-time system updates
  connectToSystemUpdates(onUpdate: (data: any) => void): WebSocket {
    const wsUrl = `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Connected to system updates');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onUpdate(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from system updates');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return ws;
  }

  // Health check with retry logic
  async healthCheckWithRetry(maxRetries: number = 3, delay: number = 1000): Promise<SystemHealth> {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await this.getSystemHealth();
      } catch (error) {
        console.warn(`Health check attempt ${attempt} failed:`, error);
        
        if (attempt === maxRetries) {
          throw error;
        }
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, delay * attempt));
      }
    }
    
    throw new Error('Health check failed after all retries');
  }

  // Monitor system health with periodic updates
  startHealthMonitoring(
    interval: number = 30000, // 30 seconds
    onHealthUpdate: (health: SystemHealth) => void,
    onError: (error: Error) => void
  ): () => void {
    let isMonitoring = true;
    let timeoutId: NodeJS.Timeout;

    const checkHealth = async () => {
      if (!isMonitoring) return;

      try {
        const health = await this.getSystemHealth();
        onHealthUpdate(health);
      } catch (error) {
        onError(error as Error);
      }

      if (isMonitoring) {
        timeoutId = setTimeout(checkHealth, interval);
      }
    };

    // Start monitoring
    checkHealth();

    // Return stop function
    return () => {
      isMonitoring = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }
}

// Create singleton instance
const systemService = new SystemService();

export default systemService;