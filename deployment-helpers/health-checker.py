#!/usr/bin/env python3
"""
Health Checker for Persian Legal AI
Monitors service health without interfering with operations
"""

import time
import requests
import subprocess
from datetime import datetime
from pathlib import Path

class HealthChecker:
    def __init__(self):
        self.services = {
            'backend': 'http://localhost:8000/api/system/health',
            'frontend': 'http://localhost:3000',
            'redis': None  # Will check with redis-cli
        }
        self.results = {}
    
    def check_backend(self):
        """Check backend health"""
        try:
            response = requests.get(self.services['backend'], timeout=10)
            if response.status_code == 200:
                return {'status': 'healthy', 'details': response.json()}
            else:
                return {'status': 'unhealthy', 'details': f'HTTP {response.status_code}'}
        except requests.exceptions.RequestException as e:
            return {'status': 'unreachable', 'details': str(e)}
    
    def check_frontend(self):
        """Check frontend health"""
        try:
            response = requests.get(self.services['frontend'], timeout=10)
            if response.status_code == 200:
                return {'status': 'healthy', 'details': 'Frontend accessible'}
            else:
                return {'status': 'unhealthy', 'details': f'HTTP {response.status_code}'}
        except requests.exceptions.RequestException as e:
            return {'status': 'unreachable', 'details': str(e)}
    
    def check_redis(self):
        """Check Redis health"""
        try:
            # Try to ping Redis
            result = subprocess.run(
                ['docker', 'exec', 'persian-legal-redis', 'redis-cli', 'ping'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and 'PONG' in result.stdout:
                return {'status': 'healthy', 'details': 'Redis responding'}
            else:
                return {'status': 'unhealthy', 'details': result.stderr or 'No PONG response'}
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            return {'status': 'unreachable', 'details': str(e)}
    
    def check_all_services(self):
        """Check all services and return results"""
        print(f"🔍 Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # Check each service
        self.results['backend'] = self.check_backend()
        self.results['frontend'] = self.check_frontend()
        self.results['redis'] = self.check_redis()
        
        # Display results
        for service, result in self.results.items():
            status = result['status']
            emoji = '✅' if status == 'healthy' else '❌' if status == 'unhealthy' else '⚠️'
            print(f"{emoji} {service.capitalize()}: {status}")
            if result['details']:
                print(f"   Details: {result['details']}")
        
        # Overall health
        healthy_count = sum(1 for r in self.results.values() if r['status'] == 'healthy')
        total_count = len(self.results)
        
        print("=" * 50)
        if healthy_count == total_count:
            print("🎉 All services healthy!")
        else:
            print(f"⚠️ {healthy_count}/{total_count} services healthy")
        
        return self.results
    
    def monitor_continuously(self, interval=30):
        """Monitor services continuously"""
        print(f"🔄 Starting continuous health monitoring (every {interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.check_all_services()
                print(f"\n⏰ Next check in {interval} seconds...\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 Health monitoring stopped")

def main():
    checker = HealthChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        checker.monitor_continuously()
    else:
        checker.check_all_services()

if __name__ == '__main__':
    import sys
    main()
