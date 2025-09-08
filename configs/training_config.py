"""
Training Configuration for Persian Legal AI System
Comprehensive configuration management for all system components
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
from loguru import logger


@dataclass
class DoRAConfig:
    """Configuration for DoRA (Weight-Decomposed Low-Rank Adaptation)"""
    
    # Core DoRA parameters
    rank: int = 64
    alpha: float = 16.0
    dropout: float = 0.1
    
    # Weight decomposition settings
    enable_decomposition: bool = True
    magnitude_learning_rate: float = 1e-4
    direction_learning_rate: float = 1e-3
    
    # Target modules for adaptation
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Optimization settings
    use_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    
    # Monitoring
    log_decomposition_metrics: bool = True
    save_decomposition_plots: bool = True


@dataclass
class QRAdaptorConfig:
    """Configuration for QR-Adaptor joint optimization"""
    
    # Quantization settings
    quantization_bits: Dict[str, int] = field(default_factory=lambda: {
        "critical_layers": 4,
        "standard_layers": 8,
        "less_critical_layers": 16
    })
    
    # Adaptive rank settings
    adaptive_rank: bool = True
    min_rank: int = 8
    max_rank: int = 128
    rank_adjustment_frequency: int = 100
    
    # Joint optimization
    bit_rank_balance_factor: float = 0.5
    importance_threshold: float = 0.01
    
    # NF4 quantization
    use_nf4: bool = True
    double_quantization: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class WindowsOptimizationConfig:
    """Configuration for Windows VPS CPU optimization"""
    
    # CPU configuration
    cpu_cores: int = 24
    use_physical_cores_only: bool = True
    numa_aware: bool = True
    
    # Intel Extension settings
    enable_ipex: bool = True
    ipex_optimization_level: str = "O1"
    enable_amx: bool = True
    enable_avx512: bool = True
    
    # Memory optimization
    use_mimalloc: bool = True
    enable_large_pages: bool = True
    memory_pool_size: str = "32GB"
    
    # Threading
    omp_num_threads: int = 24
    mkl_num_threads: int = 24
    
    # Process optimization
    set_process_priority: str = "high"  # high, realtime, normal
    enable_cpu_affinity: bool = True


@dataclass
class PersianNLPConfig:
    """Configuration for Persian NLP processing"""
    
    # Model preferences
    preferred_models: List[str] = field(default_factory=lambda: [
        "universitytehran/PersianMind-v1.0",
        "HooshvareLab/bert-base-parsbert-uncased",
        "myrkur/sentence-transformer-parsbert-fa-2.0"
    ])
    
    # Text processing
    max_sequence_length: int = 512
    batch_size: int = 16
    enable_text_normalization: bool = True
    
    # Persian-specific settings
    handle_zwnj: bool = True
    arabic_to_persian: bool = True
    normalize_numbers: bool = True
    
    # Legal domain settings
    legal_entity_extraction: bool = True
    legal_reference_parsing: bool = True
    temporal_expression_extraction: bool = True


@dataclass
class DataCollectionConfig:
    """Configuration for Persian legal data collection"""
    
    # Data sources
    data_sources: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "naab_corpus": {
            "enabled": True,
            "url": "https://www.naab.ir/corpus",
            "priority": 1,
            "max_documents": 100000
        },
        "iran_data_portal": {
            "enabled": True,
            "url": "https://irandataportal.syr.edu/laws-and-regulations",
            "priority": 2,
            "max_documents": 50000
        },
        "qavanin_portal": {
            "enabled": True,
            "url": "https://qavanin.ir/",
            "priority": 3,
            "max_documents": 25000
        },
        "majles_website": {
            "enabled": True,
            "url": "https://majles.ir/",
            "priority": 4,
            "max_documents": 15000
        }
    })
    
    # Collection settings
    max_workers: int = 8
    request_timeout: int = 30
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0
    
    # Quality filters
    min_document_length: int = 100
    max_document_length: int = 50000
    quality_threshold: float = 0.7
    
    # Storage
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours
    compression_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for system monitoring"""
    
    # Monitoring intervals
    system_metrics_interval: int = 5  # seconds
    training_metrics_interval: int = 10
    model_checkpoint_interval: int = 3600  # 1 hour
    
    # Alerting
    enable_alerts: bool = True
    cpu_usage_threshold: float = 90.0
    memory_usage_threshold: float = 85.0
    disk_usage_threshold: float = 90.0
    
    # Logging
    log_level: str = "INFO"
    enable_windows_event_log: bool = True
    max_log_file_size: str = "100MB"
    log_retention_days: int = 30
    
    # Performance tracking
    track_inference_time: bool = True
    track_memory_usage: bool = True
    track_cpu_utilization: bool = True


@dataclass
class WebInterfaceConfig:
    """Configuration for Streamlit web interface"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8501
    enable_cors: bool = True
    
    # UI settings
    theme: str = "dark"
    enable_rtl: bool = True
    language: str = "persian"
    
    # Features
    enable_real_time_monitoring: bool = True
    enable_training_control: bool = True
    enable_data_visualization: bool = True
    
    # Security
    enable_authentication: bool = False
    session_timeout: int = 3600


class TrainingConfig:
    """Main configuration class for the Persian Legal AI system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        
        # Component configurations
        self.dora = DoRAConfig()
        self.qr_adaptor = QRAdaptorConfig()
        self.windows_optimization = WindowsOptimizationConfig()
        self.persian_nlp = PersianNLPConfig()
        self.data_collection = DataCollectionConfig()
        self.monitoring = MonitoringConfig()
        self.web_interface = WebInterfaceConfig()
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Set derived properties
        self._set_derived_properties()
        
        logger.info("Training configuration initialized")
    
    def _set_derived_properties(self) -> None:
        """Set properties derived from configuration"""
        
        # CPU threading configuration
        self.cpu_threads = self.windows_optimization.omp_num_threads
        self.interop_threads = max(1, self.cpu_threads // 4)
        
        # Model configuration
        self.base_model_name = self.persian_nlp.preferred_models[0]
        self.model_cache_dir = Path("./models/cache")
        self.data_cache_dir = Path("./data/cache")
        
        # DoRA configuration
        self.dora_rank = self.dora.rank
        self.dora_alpha = self.dora.alpha
        self.target_modules = self.dora.target_modules
        
        # QR-Adaptor configuration
        self.quantization_bits = self.qr_adaptor.quantization_bits["standard_layers"]
        
        # Data collection
        self.data_collection_workers = self.data_collection.max_workers
        
        # Monitoring
        self.monitoring_interval = self.monitoring.system_metrics_interval
        
        # Training parameters
        self.training_params = {
            "learning_rate": 1e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "warmup_steps": 100,
            "save_steps": 1000,
            "logging_steps": 10,
            "max_grad_norm": 1.0,
            "dataloader_num_workers": 4,
            "fp16": True,
            "remove_unused_columns": False,
            "label_names": ["labels"],
        }
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML or JSON file"""
        try:
            config_path = Path(config_path)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Update configurations
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section_config = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to file"""
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config_data = {
                "dora": asdict(self.dora),
                "qr_adaptor": asdict(self.qr_adaptor),
                "windows_optimization": asdict(self.windows_optimization),
                "persian_nlp": asdict(self.persian_nlp),
                "data_collection": asdict(self.data_collection),
                "monitoring": asdict(self.monitoring),
                "web_interface": asdict(self.web_interface)
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Validate DoRA configuration
            assert self.dora.rank > 0, "DoRA rank must be positive"
            assert self.dora.alpha > 0, "DoRA alpha must be positive"
            assert 0 <= self.dora.dropout <= 1, "DoRA dropout must be between 0 and 1"
            
            # Validate QR-Adaptor configuration
            assert self.qr_adaptor.min_rank <= self.qr_adaptor.max_rank, "Min rank must be <= max rank"
            
            # Validate Windows optimization
            assert self.windows_optimization.cpu_cores > 0, "CPU cores must be positive"
            
            # Validate data collection
            assert self.data_collection.max_workers > 0, "Max workers must be positive"
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        base_config = {
            "torch_dtype": "float32",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "cache_dir": str(self.model_cache_dir)
        }
        
        # Model-specific optimizations
        if "parsbert" in model_name.lower():
            base_config.update({
                "max_position_embeddings": 512,
                "attention_dropout": 0.1,
                "hidden_dropout": 0.1
            })
        
        return base_config
    
    def get_training_config(self, model_name: str) -> Dict[str, Any]:
        """Get training-specific configuration for a model"""
        config = self.training_params.copy()
        
        # Adjust batch size based on model size
        if "7b" in model_name.lower() or "large" in model_name.lower():
            config["batch_size"] = 2
            config["gradient_accumulation_steps"] = 16
        elif "base" in model_name.lower():
            config["batch_size"] = 4
            config["gradient_accumulation_steps"] = 8
        
        return config
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""Persian Legal AI Configuration:
  Base Model: {self.base_model_name}
  DoRA Rank: {self.dora_rank}
  CPU Threads: {self.cpu_threads}
  Data Workers: {self.data_collection_workers}
  Monitoring Interval: {self.monitoring_interval}s"""


# Global configuration instance
_config_instance: Optional[TrainingConfig] = None


def get_config() -> TrainingConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = TrainingConfig()
    return _config_instance


def set_config(config: TrainingConfig) -> None:
    """Set global configuration instance"""
    global _config_instance
    _config_instance = config