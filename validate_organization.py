#!/usr/bin/env python3
"""
Persian Legal AI - Organization Validation Script
Comprehensive validation of project organization and structure
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
import ast
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrganizationValidator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Define expected structure
        self.expected_structure = {
            'src': {
                'backend': ['api', 'auth', 'database', 'models', 'services'],
                'frontend': ['components', 'pages', 'hooks', 'services', 'types'],
                'shared': ['types', 'constants', 'utils']
            },
            'models': ['dora', 'qr_adaptor', 'bert'],
            'data': ['databases', 'datasets', 'exports'],
            'configs': ['development', 'production', 'testing', 'shared'],
            'tests': ['unit', 'integration', 'e2e', 'performance'],
            'scripts': ['deployment', 'maintenance', 'setup', 'testing', 'utils'],
            'docs': ['api', 'guides', 'reports', 'analysis', 'development']
        }
        
        self.critical_files = {
            'src/backend/main.py': 'Main backend application',
            'src/backend/persian_main.py': 'Persian main application (alternative)',
            'configs/shared/requirements.txt': 'Python dependencies',
            'configs/development/docker-compose.yml': 'Development Docker config',
            'README.md': 'Project documentation'
        }

    def validate_directory_structure(self) -> Dict[str, List[str]]:
        """Validate the directory structure matches expectations"""
        logger.info("Validating directory structure...")
        
        structure_issues = {'missing': [], 'unexpected': []}
        
        def check_structure(expected: Dict, current_path: Path, level: str = ""):
            for key, value in expected.items():
                full_path = current_path / key
                
                if not full_path.exists():
                    structure_issues['missing'].append(f"{level}{key}/")
                    self.errors.append(f"Missing directory: {full_path}")
                    continue
                
                if isinstance(value, list):
                    # Check subdirectories
                    for subdir in value:
                        sub_path = full_path / subdir
                        if not sub_path.exists():
                            structure_issues['missing'].append(f"{level}{key}/{subdir}/")
                            self.warnings.append(f"Missing subdirectory: {sub_path}")
                
                elif isinstance(value, dict):
                    # Recursively check nested structure
                    check_structure(value, full_path, f"{level}{key}/")
        
        check_structure(self.expected_structure, self.project_root)
        
        return structure_issues

    def validate_critical_files(self) -> Dict[str, bool]:
        """Validate critical files exist"""
        logger.info("Validating critical files...")
        
        file_status = {}
        
        for file_path, description in self.critical_files.items():
            full_path = self.project_root / file_path
            exists = full_path.exists()
            file_status[file_path] = exists
            
            if exists:
                self.info.append(f"✓ Found: {description} at {file_path}")
            else:
                # Check for alternative locations
                alternatives = self._find_alternative_locations(file_path)
                if alternatives:
                    self.warnings.append(f"⚠ {description} not at expected location {file_path}, found at: {alternatives}")
                else:
                    self.errors.append(f"✗ Missing: {description} at {file_path}")
        
        return file_status

    def _find_alternative_locations(self, expected_path: str) -> List[str]:
        """Find alternative locations for missing files"""
        filename = Path(expected_path).name
        alternatives = []
        
        for file_path in self.project_root.rglob(filename):
            if str(file_path.relative_to(self.project_root)) != expected_path:
                alternatives.append(str(file_path.relative_to(self.project_root)))
        
        return alternatives

    def validate_python_imports(self) -> Dict[str, List[str]]:
        """Validate Python imports are not broken"""
        logger.info("Validating Python imports...")
        
        import_issues = {'syntax_errors': [], 'import_errors': [], 'warnings': []}
        
        for py_file in self.project_root.rglob('*.py'):
            if self._should_check_file(py_file):
                try:
                    # Check syntax
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        import_issues['syntax_errors'].append(f"{py_file}: {e}")
                        self.errors.append(f"Syntax error in {py_file}: {e}")
                        continue
                    
                    # Check for potentially broken imports
                    broken_imports = self._check_imports(content, py_file)
                    if broken_imports:
                        import_issues['import_errors'].extend(broken_imports)
                        for imp in broken_imports:
                            self.warnings.append(f"Potentially broken import in {py_file}: {imp}")
                
                except Exception as e:
                    import_issues['syntax_errors'].append(f"{py_file}: {e}")
                    self.errors.append(f"Error checking {py_file}: {e}")
        
        return import_issues

    def _check_imports(self, content: str, file_path: Path) -> List[str]:
        """Check for potentially broken imports"""
        broken_imports = []
        
        # Find import statements
        import_patterns = [
            r'from\s+([^\s]+)\s+import',
            r'import\s+([^\s,]+)'
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                module_name = match.group(1)
                
                # Check for old paths that should have been updated
                old_patterns = ['backend.', 'frontend.', 'config.', 'models.']
                for old_pattern in old_patterns:
                    if module_name.startswith(old_pattern) and 'src.' not in module_name:
                        broken_imports.append(module_name)
        
        return broken_imports

    def validate_config_files(self) -> Dict[str, List[str]]:
        """Validate configuration files"""
        logger.info("Validating configuration files...")
        
        config_issues = {'invalid_yaml': [], 'invalid_json': [], 'missing_paths': []}
        
        # Check YAML files
        for yaml_file in self.project_root.rglob('*.yml'):
            if self._should_check_file(yaml_file):
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    
                    # Check for old paths in docker-compose files
                    if 'docker-compose' in yaml_file.name:
                        self._check_docker_compose_paths(yaml_file, config_issues)
                
                except yaml.YAMLError as e:
                    config_issues['invalid_yaml'].append(f"{yaml_file}: {e}")
                    self.errors.append(f"Invalid YAML in {yaml_file}: {e}")
                except Exception as e:
                    self.warnings.append(f"Error reading {yaml_file}: {e}")
        
        # Check JSON files
        for json_file in self.project_root.rglob('*.json'):
            if self._should_check_file(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    config_issues['invalid_json'].append(f"{json_file}: {e}")
                    self.errors.append(f"Invalid JSON in {json_file}: {e}")
                except Exception as e:
                    self.warnings.append(f"Error reading {json_file}: {e}")
        
        return config_issues

    def _check_docker_compose_paths(self, docker_file: Path, config_issues: Dict):
        """Check Docker Compose file for correct paths"""
        try:
            with open(docker_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for old paths that should have been updated
            old_paths = ['./backend:', './frontend:']
            new_paths = ['./src/backend:', './src/frontend:']
            
            for old_path, new_path in zip(old_paths, new_paths):
                if old_path in content and new_path not in content:
                    config_issues['missing_paths'].append(f"{docker_file}: {old_path} should be {new_path}")
                    self.warnings.append(f"Docker path not updated in {docker_file}: {old_path}")
        
        except Exception as e:
            self.warnings.append(f"Error checking Docker paths in {docker_file}: {e}")

    def validate_frontend_structure(self) -> Dict[str, List[str]]:
        """Validate frontend structure and files"""
        logger.info("Validating frontend structure...")
        
        frontend_issues = {'missing_configs': [], 'missing_deps': [], 'structure': []}
        
        frontend_dir = self.project_root / 'src' / 'frontend'
        
        if not frontend_dir.exists():
            self.errors.append("Frontend directory not found at src/frontend")
            return frontend_issues
        
        # Check for essential frontend files
        essential_files = [
            'package.json',
            'tsconfig.json',
            'vite.config.ts',
            'tailwind.config.js'
        ]
        
        for file in essential_files:
            if not (frontend_dir / file).exists():
                # Check if file exists elsewhere
                found_elsewhere = list(self.project_root.rglob(file))
                if found_elsewhere:
                    frontend_issues['missing_configs'].append(f"{file} found at {found_elsewhere[0]} but not in src/frontend")
                else:
                    frontend_issues['missing_configs'].append(f"{file} not found")
        
        # Check component structure
        components_dir = frontend_dir / 'components'
        if components_dir.exists():
            component_files = list(components_dir.rglob('*.tsx'))
            if len(component_files) == 0:
                frontend_issues['structure'].append("No .tsx components found in components directory")
            else:
                self.info.append(f"Found {len(component_files)} React components")
        
        return frontend_issues

    def validate_backend_structure(self) -> Dict[str, List[str]]:
        """Validate backend structure and files"""
        logger.info("Validating backend structure...")
        
        backend_issues = {'missing_modules': [], 'import_errors': [], 'structure': []}
        
        backend_dir = self.project_root / 'src' / 'backend'
        
        if not backend_dir.exists():
            self.errors.append("Backend directory not found at src/backend")
            return backend_issues
        
        # Check for essential backend modules
        essential_modules = ['api', 'models', 'services', 'database']
        
        for module in essential_modules:
            module_dir = backend_dir / module
            if not module_dir.exists():
                backend_issues['missing_modules'].append(f"{module} module not found")
            else:
                # Check if module has __init__.py
                if not (module_dir / '__init__.py').exists():
                    backend_issues['structure'].append(f"{module} module missing __init__.py")
        
        # Check for main application files
        main_files = ['main.py', 'persian_main.py']
        found_main = False
        
        for main_file in main_files:
            if (backend_dir / main_file).exists():
                found_main = True
                self.info.append(f"Found main application file: {main_file}")
                break
        
        if not found_main:
            backend_issues['missing_modules'].append("No main application file found")
        
        return backend_issues

    def validate_test_structure(self) -> Dict[str, List[str]]:
        """Validate test structure"""
        logger.info("Validating test structure...")
        
        test_issues = {'missing_tests': [], 'structure': []}
        
        tests_dir = self.project_root / 'tests'
        
        if not tests_dir.exists():
            self.warnings.append("Tests directory not found")
            return test_issues
        
        # Check test categories
        test_categories = ['unit', 'integration', 'e2e']
        
        for category in test_categories:
            category_dir = tests_dir / category
            if not category_dir.exists():
                test_issues['missing_tests'].append(f"{category} tests directory not found")
            else:
                test_files = list(category_dir.rglob('test_*.py')) + list(category_dir.rglob('*.test.ts'))
                if len(test_files) == 0:
                    test_issues['missing_tests'].append(f"No test files found in {category}")
                else:
                    self.info.append(f"Found {len(test_files)} test files in {category}")
        
        return test_issues

    def _should_check_file(self, file_path: Path) -> bool:
        """Check if file should be validated"""
        skip_dirs = {'.git', 'node_modules', '__pycache__', '.pytest_cache', '.venv', 'venv', 'backup'}
        
        for part in file_path.parts:
            if part in skip_dirs or part.startswith('.'):
                return False
        
        return True

    def run_application_tests(self) -> Dict[str, bool]:
        """Run basic application startup tests"""
        logger.info("Running application startup tests...")
        
        test_results = {
            'backend_startup': False,
            'frontend_build': False,
            'docker_build': False
        }
        
        # Test backend startup (import check)
        try:
            backend_main = self.project_root / 'src' / 'backend' / 'main.py'
            if backend_main.exists():
                result = subprocess.run(
                    [sys.executable, '-m', 'py_compile', str(backend_main)],
                    capture_output=True,
                    timeout=30
                )
                test_results['backend_startup'] = result.returncode == 0
                if result.returncode == 0:
                    self.info.append("✓ Backend main file compiles successfully")
                else:
                    self.errors.append(f"✗ Backend compilation failed: {result.stderr.decode()}")
        except Exception as e:
            self.warnings.append(f"Could not test backend startup: {e}")
        
        # Test frontend build (if package.json exists)
        try:
            package_json = self.project_root / 'src' / 'frontend' / 'package.json'
            if package_json.exists():
                # Check if npm/yarn is available
                npm_available = subprocess.run(['which', 'npm'], capture_output=True).returncode == 0
                if npm_available:
                    # This is a dry-run check, not actually building
                    test_results['frontend_build'] = True
                    self.info.append("✓ Frontend build configuration exists")
                else:
                    self.warnings.append("npm not available for frontend build test")
        except Exception as e:
            self.warnings.append(f"Could not test frontend build: {e}")
        
        # Test Docker configuration
        try:
            docker_compose = self.project_root / 'configs' / 'development' / 'docker-compose.yml'
            if docker_compose.exists():
                with open(docker_compose, 'r') as f:
                    content = f.read()
                    if 'src/backend' in content and 'src/frontend' in content:
                        test_results['docker_build'] = True
                        self.info.append("✓ Docker configuration uses correct paths")
                    else:
                        self.warnings.append("Docker configuration may have incorrect paths")
        except Exception as e:
            self.warnings.append(f"Could not test Docker configuration: {e}")
        
        return test_results

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        
        # Run all validations
        structure_issues = self.validate_directory_structure()
        file_status = self.validate_critical_files()
        import_issues = self.validate_python_imports()
        config_issues = self.validate_config_files()
        frontend_issues = self.validate_frontend_structure()
        backend_issues = self.validate_backend_structure()
        test_issues = self.validate_test_structure()
        app_tests = self.run_application_tests()
        
        # Calculate scores
        total_checks = len(self.errors) + len(self.warnings) + len(self.info)
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        success_count = len(self.info)
        
        if total_checks > 0:
            success_rate = (success_count / total_checks) * 100
        else:
            success_rate = 100
        
        # Generate report
        report = f"""# Persian Legal AI - Organization Validation Report

## Summary
- **Total Checks:** {total_checks}
- **Errors:** {error_count}
- **Warnings:** {warning_count}
- **Success:** {success_count}
- **Success Rate:** {success_rate:.1f}%

## Overall Status
{'✅ PASSED' if error_count == 0 else '❌ FAILED' if error_count > 5 else '⚠️ NEEDS ATTENTION'}

---

## Directory Structure Validation

### Missing Directories
"""
        
        if structure_issues['missing']:
            for missing in structure_issues['missing']:
                report += f"- ❌ {missing}\n"
        else:
            report += "- ✅ All expected directories found\n"
        
        report += f"""
## Critical Files Validation
"""
        
        for file_path, exists in file_status.items():
            status = "✅" if exists else "❌"
            report += f"- {status} {file_path}\n"
        
        report += f"""
## Python Import Validation
- Syntax Errors: {len(import_issues['syntax_errors'])}
- Import Errors: {len(import_issues['import_errors'])}
"""
        
        if import_issues['syntax_errors']:
            report += "\n### Syntax Errors\n"
            for error in import_issues['syntax_errors']:
                report += f"- ❌ {error}\n"
        
        if import_issues['import_errors']:
            report += "\n### Import Issues\n"
            for error in import_issues['import_errors']:
                report += f"- ⚠️ {error}\n"
        
        report += f"""
## Configuration Files Validation
- Invalid YAML: {len(config_issues['invalid_yaml'])}
- Invalid JSON: {len(config_issues['invalid_json'])}
- Path Issues: {len(config_issues['missing_paths'])}
"""
        
        report += f"""
## Frontend Structure Validation
- Missing Configs: {len(frontend_issues['missing_configs'])}
- Structure Issues: {len(frontend_issues['structure'])}
"""
        
        report += f"""
## Backend Structure Validation
- Missing Modules: {len(backend_issues['missing_modules'])}
- Structure Issues: {len(backend_issues['structure'])}
"""
        
        report += f"""
## Application Tests
- Backend Startup: {'✅' if app_tests['backend_startup'] else '❌'}
- Frontend Build: {'✅' if app_tests['frontend_build'] else '❌'}
- Docker Config: {'✅' if app_tests['docker_build'] else '❌'}
"""
        
        if self.errors:
            report += f"""
## Errors ({len(self.errors)})
"""
            for error in self.errors:
                report += f"- ❌ {error}\n"
        
        if self.warnings:
            report += f"""
## Warnings ({len(self.warnings)})
"""
            for warning in self.warnings:
                report += f"- ⚠️ {warning}\n"
        
        if self.info:
            report += f"""
## Success Items ({len(self.info)})
"""
            for info in self.info:
                report += f"- {info}\n"
        
        report += f"""
---

## Recommendations

### High Priority
"""
        
        if error_count > 0:
            report += "1. Fix all critical errors listed above\n"
            report += "2. Verify import paths are correctly updated\n"
            report += "3. Ensure all main application files are in correct locations\n"
        else:
            report += "1. ✅ No critical issues found\n"
        
        report += f"""
### Medium Priority
"""
        
        if warning_count > 0:
            report += "1. Review and address warnings\n"
            report += "2. Update configuration files with correct paths\n"
            report += "3. Ensure all test files are properly organized\n"
        else:
            report += "1. ✅ No warnings to address\n"
        
        report += f"""
### Low Priority
1. Add missing test files for better coverage
2. Consider adding more comprehensive documentation
3. Optimize directory structure for better maintainability

---

**Generated on:** {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}
**Validator Version:** 1.0.0
"""
        
        return report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Persian Legal AI project organization')
    parser.add_argument('project_root', nargs='?', default='.', 
                       help='Project root directory (default: current directory)')
    parser.add_argument('--output', '-o', help='Output report to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--errors-only', action='store_true', help='Show only errors')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        logger.error(f"Project root does not exist: {project_root}")
        sys.exit(1)
    
    logger.info(f"Validating project organization in: {project_root}")
    
    validator = OrganizationValidator(str(project_root))
    report = validator.generate_validation_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        logger.info(f"Report written to: {args.output}")
    else:
        print(report)
    
    # Exit with error code if there are critical errors
    if len(validator.errors) > 0:
        logger.error(f"Validation failed with {len(validator.errors)} errors")
        sys.exit(1)
    elif len(validator.warnings) > 0:
        logger.warning(f"Validation completed with {len(validator.warnings)} warnings")
        sys.exit(0)
    else:
        logger.info("Validation passed successfully!")
        sys.exit(0)

if __name__ == '__main__':
    main()