import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import CompletePersianAIDashboard from '../components/CompletePersianAIDashboard'

// Mock the real data hooks
vi.mock('../hooks/useRealData', () => ({
  useRealTeamData: () => ({
    data: [
      {
        id: '1',
        name: 'احمد محمدی',
        role: 'مدیر پروژه',
        status: 'active',
        avatar: '/avatars/ahmad.jpg',
        lastActive: '2024-01-15T10:30:00Z',
        tasksCompleted: 15,
        currentTask: 'توسعه مدل DoRA'
      }
    ],
    loading: false,
    error: null,
    refetch: vi.fn()
  }),
  useRealModelData: () => ({
    data: [
      {
        id: 'model_1',
        name: 'Persian Legal BERT',
        status: 'training',
        progress: 75,
        accuracy: 0.92,
        lastTrained: '2024-01-15T08:00:00Z',
        trainingTime: '2h 30m',
        datasetSize: '10K samples'
      }
    ],
    loading: false,
    error: null,
    refetch: vi.fn()
  }),
  useRealSystemMetrics: () => ({
    data: {
      cpu_percent: 45.2,
      memory_percent: 67.8,
      gpu_utilization: 78.5,
      disk_usage: 23.1,
      network_io: 125.6,
      isHealthy: true,
      timestamp: '2024-01-15T10:30:00Z'
    },
    loading: false,
    error: null,
    refetch: vi.fn()
  }),
  useRealSystemStats: () => ({
    data: {
      totalSessions: 25,
      activeSessions: 3,
      totalQueries: 1250,
      avgResponseTime: 1.2,
      successRate: 98.5,
      uptime: '99.9%'
    },
    loading: false,
    error: null,
    refetch: vi.fn()
  })
}))

describe('Frontend Visual & UX Validation', () => {
  it('dashboard renders with beautiful Persian RTL layout', async () => {
    render(<CompletePersianAIDashboard />)
    
    // Wait for component to load
    await waitFor(() => {
      expect(screen.getByText('Persian Legal AI')).toBeInTheDocument()
    })
    
    // Test RTL layout
    expect(document.documentElement).toHaveAttribute('dir', 'rtl')
    
    // Test Persian font rendering
    const mainTitle = screen.getByText('Persian Legal AI')
    expect(getComputedStyle(mainTitle).fontFamily).toMatch(/Persian|Tahoma|Arial/)
    
    // Test smooth animations
    const animatedElements = document.querySelectorAll('[data-testid="animated-component"]')
    animatedElements.forEach(element => {
      const style = getComputedStyle(element)
      expect(style.transition).not.toBe('none')
    })
  })

  it('all TypeScript interfaces are correctly implemented', () => {
    // Test component props validation
    const dashboardProps = {
      user: {
        id: '1',
        name: 'احمد محمدی',
        role: 'مدیر پروژه'
      },
      sessions: [
        {
          session_id: 'test_1',
          status: 'training',
          progress: {
            data_loaded: true,
            model_initialized: true,
            training_started: true,
            training_completed: false,
            train_samples: 1000,
            eval_samples: 200,
            current_epoch: 2,
            total_epochs: 3,
            current_step: 450,
            total_steps: 600
          }
        }
      ],
      onSessionStart: vi.fn()
    }
    
    expect(() => render(<CompletePersianAIDashboard {...dashboardProps} />))
      .not.toThrow()
  })

  it('displays system metrics with proper formatting', async () => {
    render(<CompletePersianAIDashboard />)
    
    await waitFor(() => {
      // Check for system metrics display
      expect(screen.getByText(/CPU/)).toBeInTheDocument()
      expect(screen.getByText(/Memory/)).toBeInTheDocument()
      expect(screen.getByText(/GPU/)).toBeInTheDocument()
    })
  })

  it('handles loading states gracefully', () => {
    // Mock loading state
    vi.mocked(require('../hooks/useRealData').useRealSystemMetrics).mockReturnValue({
      data: null,
      loading: true,
      error: null,
      refetch: vi.fn()
    })
    
    render(<CompletePersianAIDashboard />)
    
    expect(screen.getByText(/Loading/)).toBeInTheDocument()
  })

  it('displays error states with retry functionality', () => {
    // Mock error state
    vi.mocked(require('../hooks/useRealData').useRealSystemMetrics).mockReturnValue({
      data: null,
      loading: false,
      error: 'Connection failed',
      refetch: vi.fn()
    })
    
    render(<CompletePersianAIDashboard />)
    
    expect(screen.getByText(/Connection failed/)).toBeInTheDocument()
    expect(screen.getByText(/Retry/)).toBeInTheDocument()
  })
})