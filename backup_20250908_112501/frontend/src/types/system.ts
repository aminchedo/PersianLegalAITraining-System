// System types for Persian Legal AI Frontend
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  checks: {
    system_metrics: SystemMetrics;
    gpu_info: GPUInfo;
    database: DatabaseHealth;
    services: Record<string, string>;
    performance: PerformanceMetrics;
    security: SecurityStatus;
    response_time: number;
  };
  overall_health: string;
  message: string;
}

export interface SystemMetrics {
  cpu: {
    usage_percent: number;
    count: number;
    frequency_mhz?: number;
    load_average: number[];
  };
  memory: {
    total_gb: number;
    available_gb: number;
    used_gb: number;
    usage_percent: number;
    swap_total_gb: number;
    swap_used_gb: number;
    swap_usage_percent: number;
  };
  disk: {
    total_gb: number;
    used_gb: number;
    free_gb: number;
    usage_percent: number;
    read_bytes: number;
    write_bytes: number;
  };
  network: {
    bytes_sent: number;
    bytes_recv: number;
    packets_sent: number;
    packets_recv: number;
  };
  processes: {
    total: number;
    top_cpu: ProcessInfo[];
    top_memory: ProcessInfo[];
  };
}

export interface ProcessInfo {
  pid: number;
  name: string;
  cpu_percent: number;
  memory_percent: number;
}

export interface GPUInfo {
  available: boolean;
  count: number;
  devices: GPUDevice[];
  memory_total: number;
  memory_used: number;
  memory_free: number;
  utilization: number;
}

export interface GPUDevice {
  id: number;
  name: string;
  memory_total: number;
  memory_used: number;
  memory_free: number;
  utilization: number;
}

export interface DatabaseHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  connection_time: number;
  query_time: number;
  active_connections: number;
  max_connections: number;
  database_size: string;
  last_backup?: string;
}

export interface PerformanceMetrics {
  optimization: {
    active: boolean;
    optimal_batch_size: number;
    optimal_workers: number;
    report: Record<string, any>;
  };
  response_times: {
    health_check: number;
    database_query: number;
    api_endpoint: number;
  };
}

export interface SecurityStatus {
  authentication: {
    enabled: boolean;
    jwt_enabled: boolean;
    rate_limiting: boolean;
  };
  ssl: {
    enabled: boolean;
    certificate_valid: boolean;
  };
  permissions: {
    current_user?: string;
    user_permissions: string[];
  };
}

export interface SystemInfo {
  cpu_cores: number;
  memory_gb: number;
  disk_space_gb: number;
  os_info: string;
  python_version: string;
  torch_available: boolean;
  cuda_available: boolean;
}