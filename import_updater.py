#!/usr/bin/env python3
"""
Persian Legal AI - Import Path Updater
Advanced import path updating for Python and JavaScript/TypeScript files
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImportUpdater:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.updates_count = 0
        self.files_processed = 0
        
        # Define import mapping rules
        self.python_mappings = {
            r'from backend\.': 'from src.backend.',
            r'import backend\.': 'import src.backend.',
            r'from frontend\.': 'from src.frontend.',
            r'import frontend\.': 'import src.frontend.',
            r'from models\.': 'from models.',  # Keep models at root
            r'import models\.': 'import models.',
            r'from config\.': 'from configs.shared.',
            r'import config\.': 'import configs.shared.',
            r'from configs\.': 'from configs.shared.',
            r'import configs\.': 'import configs.shared.',
            r'from data\.': 'from data.',
            r'import data\.': 'import data.',
            r'from tests\.': 'from tests.',
            r'import tests\.': 'import tests.',
            r'from scripts\.': 'from scripts.',
            r'import scripts\.': 'import scripts.',
        }
        
        self.js_ts_mappings = {
            # Update relative imports based on new structure
            r"from ['\"]\.\.\/backend\/": "from '../../../src/backend/",
            r"import.*from ['\"]\.\.\/backend\/": "import from '../../../src/backend/",
            r"from ['\"]\.\.\/frontend\/": "from '../src/frontend/",
            r"import.*from ['\"]\.\.\/frontend\/": "import from '../src/frontend/",
            r"from ['\"]\.\.\/models\/": "from '../../../models/",
            r"import.*from ['\"]\.\.\/models\/": "import from '../../../models/",
        }

    def update_python_file(self, file_path: Path) -> bool:
        """Update import statements in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply each mapping
            for pattern, replacement in self.python_mappings.items():
                content = re.sub(pattern, replacement, content)
            
            # Handle relative imports within src structure
            if 'src/' in str(file_path):
                # Calculate relative path adjustments
                depth = len([p for p in file_path.parts if p != 'src']) - len(file_path.parts[file_path.parts.index('src'):])
                
                # Update relative imports for files moved into src/
                if depth > 0:
                    relative_prefix = '../' * depth
                    content = re.sub(r'from \.\./', f'from {relative_prefix}', content)
            
            # Special handling for common import patterns
            content = self._fix_common_patterns(content, file_path)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.updates_count += 1
                logger.info(f"Updated Python imports in: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating Python file {file_path}: {e}")
            return False

    def update_js_ts_file(self, file_path: Path) -> bool:
        """Update import statements in JavaScript/TypeScript files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply each mapping
            for pattern, replacement in self.js_ts_mappings.items():
                content = re.sub(pattern, replacement, content)
            
            # Handle specific frontend reorganization
            if 'src/frontend' in str(file_path):
                # Update imports within frontend structure
                content = self._fix_frontend_imports(content, file_path)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.updates_count += 1
                logger.info(f"Updated JS/TS imports in: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating JS/TS file {file_path}: {e}")
            return False

    def _fix_common_patterns(self, content: str, file_path: Path) -> str:
        """Fix common import patterns after reorganization"""
        
        # Fix FastAPI imports for moved backend files
        if 'src/backend' in str(file_path):
            # Update database imports
            content = re.sub(
                r'from database\.',
                'from .database.',
                content
            )
            
            # Update model imports
            content = re.sub(
                r'from models\.',
                'from .models.',
                content
            )
            
            # Update service imports
            content = re.sub(
                r'from services\.',
                'from .services.',
                content
            )
        
        # Fix model imports
        if 'models/' in str(file_path):
            content = re.sub(
                r'from \.\.',
                'from ..',
                content
            )
        
        return content

    def _fix_frontend_imports(self, content: str, file_path: Path) -> str:
        """Fix frontend-specific import patterns"""
        
        # Update component imports
        content = re.sub(
            r"from ['\"]\.\.\/components\/",
            "from '../components/",
            content
        )
        
        # Update service imports
        content = re.sub(
            r"from ['\"]\.\.\/services\/",
            "from '../services/",
            content
        )
        
        # Update utility imports
        content = re.sub(
            r"from ['\"]\.\.\/utils\/",
            "from '../utils/",
            content
        )
        
        return content

    def update_config_files(self) -> int:
        """Update paths in configuration files"""
        config_files_updated = 0
        
        # Update docker-compose files
        for config_file in self.project_root.rglob('docker-compose*.yml'):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Update volume and build paths
                content = re.sub(r'\./backend:', './src/backend:', content)
                content = re.sub(r'\./frontend:', './src/frontend:', content)
                content = re.sub(r'backend/', 'src/backend/', content)
                content = re.sub(r'frontend/', 'src/frontend/', content)
                
                if content != original_content:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    config_files_updated += 1
                    logger.info(f"Updated config file: {config_file}")
                    
            except Exception as e:
                logger.error(f"Error updating config file {config_file}: {e}")
        
        # Update package.json files
        for package_file in self.project_root.rglob('package.json'):
            try:
                import json
                
                with open(package_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                updated = False
                
                # Update scripts section
                if 'scripts' in data:
                    for script_name, script_value in data['scripts'].items():
                        new_value = script_value
                        new_value = re.sub(r'frontend/', 'src/frontend/', new_value)
                        new_value = re.sub(r'backend/', 'src/backend/', new_value)
                        
                        if new_value != script_value:
                            data['scripts'][script_name] = new_value
                            updated = True
                
                if updated:
                    with open(package_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    config_files_updated += 1
                    logger.info(f"Updated package.json: {package_file}")
                    
            except Exception as e:
                logger.error(f"Error updating package.json {package_file}: {e}")
        
        return config_files_updated

    def process_directory(self, directory: Path) -> Dict[str, int]:
        """Process all files in a directory"""
        stats = {
            'python_files': 0,
            'js_ts_files': 0,
            'config_files': 0,
            'total_updates': 0
        }
        
        # Process Python files
        for py_file in directory.rglob('*.py'):
            if self._should_process_file(py_file):
                self.files_processed += 1
                if self.update_python_file(py_file):
                    stats['python_files'] += 1
        
        # Process JavaScript/TypeScript files
        for ext in ['*.js', '*.ts', '*.jsx', '*.tsx']:
            for js_file in directory.rglob(ext):
                if self._should_process_file(js_file):
                    self.files_processed += 1
                    if self.update_js_ts_file(js_file):
                        stats['js_ts_files'] += 1
        
        # Process config files
        stats['config_files'] = self.update_config_files()
        
        stats['total_updates'] = self.updates_count
        return stats

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        # Skip files in certain directories
        skip_dirs = {'.git', 'node_modules', '__pycache__', '.pytest_cache', '.venv', 'venv'}
        
        for part in file_path.parts:
            if part in skip_dirs:
                return False
        
        # Skip backup files
        if file_path.suffix in {'.bak', '.backup', '.old'}:
            return False
            
        return True

    def generate_report(self) -> str:
        """Generate a report of the import updates"""
        report = f"""
# Import Path Update Report

## Summary
- Files processed: {self.files_processed}
- Total updates made: {self.updates_count}

## Mapping Rules Applied

### Python Files
"""
        for pattern, replacement in self.python_mappings.items():
            report += f"- `{pattern}` → `{replacement}`\n"
        
        report += "\n### JavaScript/TypeScript Files\n"
        for pattern, replacement in self.js_ts_mappings.items():
            report += f"- `{pattern}` → `{replacement}`\n"
        
        report += f"\n## Completion Time
- Updated on: {os.popen('date').read().strip()}
"
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Update import paths after project reorganization')
    parser.add_argument('project_root', nargs='?', default='.', 
                       help='Project root directory (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    parser.add_argument('--report', action='store_true',
                       help='Generate a detailed report')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        sys.exit(1)
    
    logger.info(f"Starting import path updates in: {project_root}")
    
    updater = ImportUpdater(str(project_root))
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
        # In dry run, we would analyze but not modify
        return
    
    stats = updater.process_directory(project_root)
    
    logger.info("Import update completed!")
    logger.info(f"Python files updated: {stats['python_files']}")
    logger.info(f"JS/TS files updated: {stats['js_ts_files']}")
    logger.info(f"Config files updated: {stats['config_files']}")
    logger.info(f"Total updates: {stats['total_updates']}")
    
    if args.report:
        report = updater.generate_report()
        report_file = project_root / f"import_update_report_{os.popen('date +%Y%m%d_%H%M%S').read().strip()}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report generated: {report_file}")

if __name__ == '__main__':
    main()