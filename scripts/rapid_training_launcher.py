#!/usr/bin/env python3
"""
One-Click Rapid Training Launcher for Persian Legal AI
راه‌انداز آموزش سریع یک‌کلیکه برای هوش مصنوعی حقوقی فارسی
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.services.rapid_trainer import RapidTrainingOrchestrator
from backend.data.dataset_integration import PersianLegalDataIntegrator
from backend.validation.dataset_validator import DatasetQualityValidator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rapid_training.log', encoding='utf-8')
        ]
    )

def display_banner():
    """Display application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 Persian Legal AI - Rapid Training Launcher                              ║
║  راه‌انداز آموزش سریع هوش مصنوعی حقوقی فارسی                                ║
║                                                                              ║
║  ⚡ Accelerated Training with Premium Persian Legal Datasets                 ║
║  🚀 Maximum Speed • 🏆 Maximum Quality • 📊 Professional Results            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def display_available_datasets():
    """Display information about available datasets"""
    print("\n📊 Available Premium Persian Legal Datasets:")
    print("=" * 60)
    
    try:
        integrator = PersianLegalDataIntegrator()
        stats = integrator.get_dataset_stats()
        
        if not stats or 'total_samples' not in stats:
            print("⚠️  No dataset statistics available")
            return
        
        for key, data in stats.items():
            if key in ['total_samples', 'accessible_datasets', 'recommended_datasets']:
                continue
            
            status = "✅" if data.get('recommended', False) else "📋"
            print(f"  {status} {data['name']}")
            print(f"     📈 Samples: {data['samples']:,}")
            print(f"     💾 Size: {data['size_mb']} MB")
            print(f"     🏷️  Type: {data['type']}")
            print(f"     📝 {data.get('description', 'No description available')}")
            print()
        
        print(f"📈 Total Training Samples: {stats['total_samples']:,}")
        print(f"✅ Accessible Datasets: {stats.get('accessible_datasets', 0)}")
        print(f"🏆 Recommended Datasets: {stats.get('recommended_datasets', 0)}")
        
    except Exception as e:
        print(f"⚠️  Error loading dataset information: {e}")

def get_training_configuration() -> dict:
    """Get training configuration from user or use defaults"""
    print("\n⚙️  Training Configuration:")
    print("=" * 40)
    
    # Default configuration
    config = {
        'model_config': {
            'base_model': 'HooshvareLab/bert-fa-base-uncased',
            'dora_rank': 64,
            'dora_alpha': 16.0,
            'accelerated_mode': True,
            'batch_size': 16,
            'gradient_accumulation_steps': 4,
            'max_length': 512
        },
        'training_config': {
            'epochs': 15,
            'learning_rate': 2e-5,
            'save_every_n_epochs': 2,
            'max_training_hours': 24,
            'early_stopping_patience': 3
        },
        'dataset_config': {
            'max_samples_per_dataset': 50000,
            'validation_split': 0.1,
            'quality_threshold': 70
        },
        'output_config': {
            'checkpoint_dir': './checkpoints',
            'report_dir': './reports',
            'model_output_dir': './models'
        }
    }
    
    print("Using optimized default configuration:")
    print(f"  🎯 Model: {config['model_config']['base_model']}")
    print(f"  📚 Epochs: {config['training_config']['epochs']}")
    print(f"  🎓 Learning Rate: {config['training_config']['learning_rate']}")
    print(f"  📦 Batch Size: {config['model_config']['batch_size']}")
    print(f"  🔍 Quality Threshold: {config['dataset_config']['quality_threshold']}%")
    
    return config

def validate_system_requirements():
    """Validate system requirements"""
    print("\n🔍 System Requirements Check:")
    print("=" * 40)
    
    import psutil
    
    # Check memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    
    print(f"💾 Memory: {memory_gb:.1f}GB total, {available_gb:.1f}GB available")
    if available_gb < 4:
        print("⚠️  Warning: Less than 4GB available memory")
    
    # Check disk space
    disk = psutil.disk_usage('/')
    disk_free_gb = disk.free / (1024**3)
    print(f"💿 Disk Space: {disk_free_gb:.1f}GB free")
    if disk_free_gb < 10:
        print("⚠️  Warning: Less than 10GB free disk space")
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    print(f"🖥️  CPU Cores: {cpu_count}")
    if cpu_count < 2:
        print("⚠️  Warning: Less than 2 CPU cores")
    
    print("✅ System requirements check completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Persian Legal AI Rapid Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rapid_training_launcher.py                    # Start with default config
  python rapid_training_launcher.py --verbose          # Verbose logging
  python rapid_training_launcher.py --config config.json  # Custom config
  python rapid_training_launcher.py --datasets-only    # Show datasets only
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to custom configuration file')
    parser.add_argument('--datasets-only', action='store_true',
                       help='Show available datasets and exit')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal data')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Display banner
    display_banner()
    
    try:
        # Show available datasets
        display_available_datasets()
        
        if args.datasets_only:
            print("\n✅ Dataset information displayed. Exiting.")
            return 0
        
        # Validate system requirements
        validate_system_requirements()
        
        # Get training configuration
        if args.config:
            print(f"\n📄 Loading configuration from {args.config}")
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = get_training_configuration()
        
        # Modify config for quick test
        if args.quick_test:
            print("\n🧪 Quick test mode enabled")
            config['training_config']['epochs'] = 2
            config['dataset_config']['max_samples_per_dataset'] = 1000
            config['model_config']['batch_size'] = 8
        
        # Create output directories
        for dir_path in config['output_config'].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize orchestrator
        print("\n🚀 Initializing Rapid Training Orchestrator...")
        orchestrator = RapidTrainingOrchestrator(config)
        
        # Start training
        print("\n⚡ Launching rapid training with optimized configuration...")
        print("=" * 60)
        
        start_time = datetime.now()
        results = orchestrator.execute_rapid_training()
        end_time = datetime.now()
        
        # Display results
        print("\n" + "=" * 60)
        if results['success']:
            print("🎉 Training completed successfully!")
            print(f"⏱️  Total Time: {(end_time - start_time).total_seconds() / 60:.2f} minutes")
            
            if 'training_results' in results:
                training_results = results['training_results']
                if 'training_results' in training_results:
                    tr = training_results['training_results']
                    print(f"📊 Final Loss: {tr.get('final_loss', 'N/A')}")
                    print(f"🚀 Samples/sec: {tr.get('samples_per_second', 'N/A')}")
                    print(f"📈 Total Samples: {tr.get('total_samples', 'N/A'):,}")
            
            print(f"📦 Datasets Used: {', '.join(results.get('datasets_used', []))}")
            print(f"📄 Report: {results.get('report_path', 'N/A')}")
            
            # Show recommendations
            if 'validation_results' in results:
                print("\n💡 Recommendations:")
                validator = DatasetQualityValidator()
                recommendations = validator.get_recommendations(
                    results['validation_results']
                )
                for rec in recommendations[:5]:  # Show first 5 recommendations
                    print(f"  • {rec}")
            
        else:
            print("❌ Training failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
            if 'system_status' in results:
                system_status = results['system_status']
                if not system_status.get('ready', False):
                    print("\n🔧 System Issues:")
                    for issue in system_status.get('issues', []):
                        print(f"  • {issue}")
        
        print("\n" + "=" * 60)
        print("🏁 Rapid Training Session Completed")
        
        return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)