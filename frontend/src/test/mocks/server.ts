import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'

export const server = setupServer(
  // System health endpoint
  http.get('/api/system/health', () => {
    return HttpResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      system_metrics: {
        cpu_percent: 45.2,
        memory_percent: 67.8,
        memory_available_gb: 8.5,
        disk_percent: 23.1,
        disk_free_gb: 156.7,
        active_processes: 127
      },
      gpu_info: {
        gpu_available: true,
        gpu_count: 1,
        gpu_name: 'NVIDIA RTX 4090',
        gpu_memory_total: 24576,
        gpu_memory_used: 8192
      },
      platform_info: {
        os: 'Linux',
        os_version: '6.12.8+',
        python_version: '3.11.0',
        architecture: 'x86_64'
      }
    })
  }),

  // Training sessions endpoint
  http.get('/api/training/sessions', () => {
    return HttpResponse.json([
      {
        session_id: 'test_session_1',
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
        },
        metrics: {
          total_steps: 600,
          total_epochs: 3,
          total_loss: 2.1234,
          learning_rate: 0.0002,
          train_loss: 2.1234,
          eval_loss: 2.4567,
          perplexity: 8.35
        }
      }
    ])
  }),

  // Legal query processing
  http.post('/api/legal/process', async ({ request }) => {
    const body = await request.json()
    return HttpResponse.json({
      analysis: 'تحلیل حقوقی کامل برای: ' + body.query,
      legal_references: [
        'ماده ۱۰ قانون مدنی ایران',
        'اصل بیست و دوم قانون اساسی'
      ],
      confidence: 0.95,
      processing_time: 1.2
    })
  })
)