import axios from 'axios';
import type { Document, ScrapingStatus, ClassificationResult, SearchResponse } from '../types';

// API client configuration
const api = axios.create({
  baseURL: process.env.NODE_ENV === 'production' 
    ? 'https://your-api-domain.com/api' 
    : 'http://localhost:8000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for authentication if needed
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Document search and retrieval
const searchDocuments = async (query: string, category?: string, limit: number = 10, offset: number = 0): Promise<SearchResponse> => {
  const params = new URLSearchParams({
    query,
    limit: limit.toString(),
    offset: offset.toString(),
  });
  
  if (category) {
    params.append('category', category);
  }
  
  const response = await api.get(`/documents/search?${params}`);
  return response.data;
};

const getDocument = async (id: string): Promise<Document> => {
  const response = await api.get(`/documents/${id}`);
  return response.data;
};

const getDocumentsByCategory = async (category: string, limit: number = 10): Promise<Document[]> => {
  const response = await api.get(`/documents/category/${category}?limit=${limit}`);
  return response.data;
};

// Scraping operations - CRITICAL METHODS
const getScrapingStatus = async (): Promise<ScrapingStatus> => {
  const response = await api.get('/scraping/status');
  return response.data;
};

const startScraping = async (sources: string[]): Promise<void> => {
  await api.post('/scraping/start', { sources });
};

const stopScraping = async (): Promise<void> => {
  await api.post('/scraping/stop');
};

// AI Classification
const classifyDocument = async (documentId: string): Promise<ClassificationResult> => {
  const response = await api.post(`/documents/${documentId}/classify`);
  return response.data;
};

// Statistics and analytics
const getDocumentStats = async () => {
  const response = await api.get('/documents/stats');
  return response.data;
};

const getCategoryStats = async () => {
  const response = await api.get('/documents/categories/stats');
  return response.data;
};

// Export all methods
const apiService = {
  searchDocuments,
  getDocument,
  getDocumentsByCategory,
  getScrapingStatus,
  startScraping,
  stopScraping,
  classifyDocument,
  getDocumentStats,
  getCategoryStats,
};

export default apiService;