// Training service for Persian Legal AI Frontend
import axios, { AxiosResponse } from 'axios';
import {
  TrainingSessionRequest,
  TrainingSessionResponse,
  TrainingSessionStatus,
  TrainingMetrics,
  TrainingLog,
  VerifiedTrainingSession,
  DatasetInfo
} from '../types/training';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class TrainingService {
  async createTrainingSession(request: TrainingSessionRequest): Promise<TrainingSessionResponse> {
    try {
      const response: AxiosResponse<TrainingSessionResponse> = await axios.post(
        `${API_BASE_URL}/api/training/sessions`,
        request
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to create training session:', error);
      throw new Error(error.response?.data?.detail || 'Failed to create training session');
    }
  }

  async getTrainingSessions(): Promise<TrainingSessionStatus[]> {
    try {
      const response: AxiosResponse<TrainingSessionStatus[]> = await axios.get(
        `${API_BASE_URL}/api/training/sessions`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to get training sessions:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get training sessions');
    }
  }

  async getTrainingSession(sessionId: string): Promise<TrainingSessionStatus> {
    try {
      const response: AxiosResponse<TrainingSessionStatus> = await axios.get(
        `${API_BASE_URL}/api/training/sessions/${sessionId}`
      );
      return response.data;
    } catch (error: any) {
      console.error(`Failed to get training session ${sessionId}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to get training session');
    }
  }

  async deleteTrainingSession(sessionId: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE_URL}/api/training/sessions/${sessionId}`);
    } catch (error: any) {
      console.error(`Failed to delete training session ${sessionId}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to delete training session');
    }
  }

  async getTrainingMetrics(sessionId: string): Promise<TrainingMetrics> {
    try {
      const response: AxiosResponse<TrainingMetrics> = await axios.get(
        `${API_BASE_URL}/api/training/sessions/${sessionId}/metrics`
      );
      return response.data;
    } catch (error: any) {
      console.error(`Failed to get training metrics for session ${sessionId}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to get training metrics');
    }
  }

  async getTrainingLogs(sessionId: string): Promise<TrainingLog[]> {
    try {
      const response: AxiosResponse<TrainingLog[]> = await axios.get(
        `${API_BASE_URL}/api/training/sessions/${sessionId}/logs`
      );
      return response.data;
    } catch (error: any) {
      console.error(`Failed to get training logs for session ${sessionId}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to get training logs');
    }
  }

  async stopTrainingSession(sessionId: string): Promise<void> {
    try {
      await axios.post(`${API_BASE_URL}/api/training/sessions/${sessionId}/stop`);
    } catch (error: any) {
      console.error(`Failed to stop training session ${sessionId}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to stop training session');
    }
  }

  // Verified training endpoints
  async startVerifiedTraining(request: TrainingSessionRequest): Promise<TrainingSessionResponse> {
    try {
      const response: AxiosResponse<TrainingSessionResponse> = await axios.post(
        `${API_BASE_URL}/api/training/sessions/verified`,
        request
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to start verified training:', error);
      throw new Error(error.response?.data?.detail || 'Failed to start verified training');
    }
  }

  async getVerifiedTrainingStatus(sessionId: string): Promise<VerifiedTrainingSession> {
    try {
      const response: AxiosResponse<VerifiedTrainingSession> = await axios.get(
        `${API_BASE_URL}/api/training/sessions/verified/${sessionId}/status`
      );
      return response.data;
    } catch (error: any) {
      console.error(`Failed to get verified training status for session ${sessionId}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to get verified training status');
    }
  }

  async getVerifiedDatasets(): Promise<DatasetInfo[]> {
    try {
      const response: AxiosResponse<DatasetInfo[]> = await axios.get(
        `${API_BASE_URL}/api/training/datasets/verified`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to get verified datasets:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get verified datasets');
    }
  }

  async verifyDataset(datasetName: string): Promise<{ verified: boolean; quality_score: number }> {
    try {
      const response: AxiosResponse<{ verified: boolean; quality_score: number }> = await axios.post(
        `${API_BASE_URL}/api/training/datasets/verify`,
        { dataset_name: datasetName }
      );
      return response.data;
    } catch (error: any) {
      console.error(`Failed to verify dataset ${datasetName}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to verify dataset');
    }
  }

  async getVerifiedTrainingSessions(): Promise<VerifiedTrainingSession[]> {
    try {
      const response: AxiosResponse<VerifiedTrainingSession[]> = await axios.get(
        `${API_BASE_URL}/api/training/sessions/verified`
      );
      return response.data;
    } catch (error: any) {
      console.error('Failed to get verified training sessions:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get verified training sessions');
    }
  }

  async cancelVerifiedTrainingSession(sessionId: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE_URL}/api/training/sessions/verified/${sessionId}`);
    } catch (error: any) {
      console.error(`Failed to cancel verified training session ${sessionId}:`, error);
      throw new Error(error.response?.data?.detail || 'Failed to cancel verified training session');
    }
  }

  // WebSocket connection for real-time updates
  connectToTrainingUpdates(sessionId: string, onUpdate: (data: any) => void): WebSocket {
    const wsUrl = `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/ws/training/${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`Connected to training updates for session ${sessionId}`);
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
      console.log(`Disconnected from training updates for session ${sessionId}`);
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error for session ${sessionId}:`, error);
    };

    return ws;
  }
}

// Create singleton instance
const trainingService = new TrainingService();

export default trainingService;