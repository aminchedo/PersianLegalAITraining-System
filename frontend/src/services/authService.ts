// Authentication service for Persian Legal AI Frontend
import axios, { AxiosResponse } from 'axios';
import { LoginRequest, LoginResponse, User, RefreshTokenResponse } from '../types/auth';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class AuthService {
  private token: string | null = null;
  private user: User | null = null;

  constructor() {
    // Load token from localStorage on initialization
    this.token = localStorage.getItem('auth_token');
    this.user = this.getStoredUser();
  }

  private getStoredUser(): User | null {
    const userStr = localStorage.getItem('user');
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch (error) {
        console.error('Error parsing stored user:', error);
        return null;
      }
    }
    return null;
  }

  private setStoredUser(user: User): void {
    localStorage.setItem('user', JSON.stringify(user));
  }

  private clearStoredData(): void {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
  }

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    try {
      // Mock authentication for development
      const mockUsers = {
        'admin': { password: 'admin123', user: { username: 'admin', email: 'admin@persian-legal-ai.com', full_name: 'Administrator', is_active: true, permissions: ['admin', 'training', 'data_access'] } },
        'trainer': { password: 'trainer123', user: { username: 'trainer', email: 'trainer@persian-legal-ai.com', full_name: 'AI Trainer', is_active: true, permissions: ['training', 'data_access'] } },
        'viewer': { password: 'viewer123', user: { username: 'viewer', email: 'viewer@persian-legal-ai.com', full_name: 'Data Viewer', is_active: true, permissions: ['data_access'] } }
      };

      const userData = mockUsers[credentials.username as keyof typeof mockUsers];
      
      if (!userData || userData.password !== credentials.password) {
        throw new Error('Invalid credentials');
      }

      const mockResponse: LoginResponse = {
        access_token: 'mock-jwt-token-' + Date.now(),
        token_type: 'bearer',
        expires_in: 3600,
        user: userData.user
      };

      const { access_token, user } = mockResponse;
      
      // Store token and user data
      this.token = access_token;
      this.user = user;
      localStorage.setItem('auth_token', access_token);
      this.setStoredUser(user);

      // Set default authorization header
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

      return mockResponse;
    } catch (error: any) {
      console.error('Login failed:', error);
      throw new Error(error.message || 'Login failed');
    }
  }

  async logout(): Promise<void> {
    try {
      if (this.token) {
        await axios.post(
          `${API_BASE_URL}/api/auth/logout`,
          {},
          {
            headers: {
              Authorization: `Bearer ${this.token}`
            }
          }
        );
      }
    } catch (error) {
      console.error('Logout request failed:', error);
    } finally {
      // Clear local data regardless of API call success
      this.clearStoredData();
      this.token = null;
      this.user = null;
      delete axios.defaults.headers.common['Authorization'];
    }
  }

  async refreshToken(): Promise<RefreshTokenResponse> {
    try {
      if (!this.token) {
        throw new Error('No token to refresh');
      }

      const response: AxiosResponse<RefreshTokenResponse> = await axios.post(
        `${API_BASE_URL}/api/auth/refresh`,
        {},
        {
          headers: {
            Authorization: `Bearer ${this.token}`
          }
        }
      );

      const { access_token } = response.data;
      this.token = access_token;
      localStorage.setItem('auth_token', access_token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;

      return response.data;
    } catch (error: any) {
      console.error('Token refresh failed:', error);
      this.logout(); // Clear invalid token
      throw new Error(error.response?.data?.detail || 'Token refresh failed');
    }
  }

  async getCurrentUser(): Promise<User> {
    try {
      if (!this.token) {
        throw new Error('No authentication token');
      }

      const response: AxiosResponse<User> = await axios.get(
        `${API_BASE_URL}/api/auth/me`,
        {
          headers: {
            Authorization: `Bearer ${this.token}`
          }
        }
      );

      this.user = response.data;
      this.setStoredUser(response.data);
      return response.data;
    } catch (error: any) {
      console.error('Get current user failed:', error);
      throw new Error(error.response?.data?.detail || 'Failed to get user information');
    }
  }

  isAuthenticated(): boolean {
    return !!this.token && !!this.user;
  }

  getToken(): string | null {
    return this.token;
  }

  getUser(): User | null {
    return this.user;
  }

  hasPermission(permission: string): boolean {
    return this.user?.permissions?.includes(permission) || false;
  }

  hasAnyPermission(permissions: string[]): boolean {
    return permissions.some(permission => this.hasPermission(permission));
  }

  hasAllPermissions(permissions: string[]): boolean {
    return permissions.every(permission => this.hasPermission(permission));
  }

  // Initialize axios interceptor for automatic token refresh
  setupAxiosInterceptors(): void {
    // Request interceptor to add auth header
    axios.interceptors.request.use(
      (config) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor to handle token refresh
    axios.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            await this.refreshToken();
            originalRequest.headers.Authorization = `Bearer ${this.token}`;
            return axios(originalRequest);
          } catch (refreshError) {
            this.logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }
}

// Create singleton instance
const authService = new AuthService();

// Setup axios interceptors
authService.setupAxiosInterceptors();

export default authService;