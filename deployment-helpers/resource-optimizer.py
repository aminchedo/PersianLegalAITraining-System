#!/usr/bin/env python3
"""
Resource Optimizer for Persian Legal AI
Suggests optimizations without making changes
"""

import os
import psutil
import json
from pathlib import Path

class ResourceOptimizer:
    def __init__(self):
        self.suggestions = []
        self.current_config = {}
        
    def analyze_current_setup(self):
        """Analyze without modifying"""
        print("🔍 Analyzing Current Resource Setup...")
        
        # Get system info
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        disk = psutil.disk_usage('/')
        
        self.current_config = {
            'system_memory_gb': round(memory.total / (1024**3), 2),
            'available_memory_gb': round(memory.available / (1024**3), 2),
            'cpu_cores': cpu_count,
            'disk_free_gb': round(disk.free / (1024**3), 2)
        }
        
        print(f"💾 System Memory: {self.current_config['system_memory_gb']}GB total, {self.current_config['available_memory_gb']}GB available")
        print(f"🖥️ CPU Cores: {self.current_config['cpu_cores']}")
        print(f"💽 Disk Space: {self.current_config['disk_free_gb']}GB available")
        
        return self.current_config
    
    def suggest_memory_optimizations(self):
        """Suggest memory improvements"""
        suggestions = []
        
        available_memory = self.current_config.get('available_memory_gb', 0)
        
        if available_memory < 4:
            suggestions.extend([
                "⚠️ Low memory detected - consider batch size reduction for AI models",
                "💡 Enable model quantization to reduce memory usage",
                "🔄 Implement lazy loading for heavy dependencies",
                "📁 Use memory-mapped files for large datasets"
            ])
        elif available_memory < 8:
            suggestions.extend([
                "💡 Consider enabling model quantization for better performance",
                "🔄 Optimize batch processing for AI models",
                "📊 Monitor memory usage during peak loads"
            ])
        else:
            suggestions.extend([
                "✅ Sufficient memory available",
                "🚀 Consider enabling larger batch sizes for better throughput",
                "💾 Memory caching can be increased for better performance"
            ])
        
        return suggestions
    
    def suggest_deployment_improvements(self):
        """Suggest deployment enhancements"""
        return [
            "🔍 Add comprehensive health checks to all services",
            "🔄 Implement graceful shutdown handlers",
            "📊 Add resource limits to containers to prevent OOM kills",
            "🔁 Enable auto-restart on failures",
            "📝 Add structured logging for better debugging",
            "⏱️ Configure proper timeout values",
            "🔐 Add security headers and rate limiting",
            "📈 Implement metrics collection",
            "🚨 Set up alerting for critical failures",
            "🔄 Add rolling deployment strategy"
        ]
    
    def suggest_docker_optimizations(self):
        """Suggest Docker-specific optimizations"""
        return [
            "🐳 Use multi-stage builds to reduce image size",
            "📦 Optimize layer caching in Dockerfiles",
            "🔒 Run containers as non-root users",
            "💾 Use named volumes for persistent data",
            "🌐 Configure proper network isolation",
            "🔄 Set up container health checks",
            "📊 Add resource limits (memory, CPU)",
            "🚀 Use init system in containers for proper signal handling",
            "📝 Add labels for better container management",
            "🔧 Optimize container startup time"
        ]
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        print("\n📊 Generating Resource Optimization Report...")
        
        report = {
            'timestamp': psutil.boot_time(),
            'system_info': self.current_config,
            'memory_optimizations': self.suggest_memory_optimizations(),
            'deployment_improvements': self.suggest_deployment_improvements(),
            'docker_optimizations': self.suggest_docker_optimizations()
        }
        
        # Save report
        report_path = Path('deployment-helpers/resource_optimization_report.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved to: {report_path}")
        
        # Display summary
        print("\n📋 OPTIMIZATION SUMMARY:")
        print("\n💾 Memory Optimizations:")
        for suggestion in report['memory_optimizations'][:3]:
            print(f"  {suggestion}")
        
        print("\n🚀 Deployment Improvements:")
        for suggestion in report['deployment_improvements'][:3]:
            print(f"  {suggestion}")
        
        print("\n🐳 Docker Optimizations:")
        for suggestion in report['docker_optimizations'][:3]:
            print(f"  {suggestion}")
        
        return report

def main():
    optimizer = ResourceOptimizer()
    optimizer.analyze_current_setup()
    optimizer.generate_optimization_report()

if __name__ == '__main__':
    main()
