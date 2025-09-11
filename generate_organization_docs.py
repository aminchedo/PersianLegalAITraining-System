#!/usr/bin/env python3
"""
Persian Legal AI - Organization Documentation Generator
Generates comprehensive documentation for the organized project structure
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set
import subprocess
import json
from datetime import datetime

class DocumentationGenerator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_structure_tree(self, max_depth: int = 4) -> str:
        """Generate a visual tree structure of the project"""
        try:
            result = subprocess.run(
                ['tree', '-a', '-I', '.git|node_modules|__pycache__|*.pyc|.pytest_cache|.venv', 
                 '-L', str(max_depth), str(self.project_root)],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                # Fallback to find command
                return self._generate_tree_fallback(max_depth)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._generate_tree_fallback(max_depth)
    
    def _generate_tree_fallback(self, max_depth: int) -> str:
        """Fallback tree generation using find"""
        tree_lines = []
        
        def add_directory(path: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return
            
            items = sorted([p for p in path.iterdir() 
                           if not p.name.startswith('.') and p.name not in 
                           {'node_modules', '__pycache__', '.pytest_cache', '.venv'}])
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    add_directory(item, prefix + extension, depth + 1)
        
        tree_lines.append(str(self.project_root.name))
        add_directory(self.project_root)
        
        return "\n".join(tree_lines)
    
    def count_files_by_type(self) -> Dict[str, int]:
        """Count files by extension"""
        file_counts = {}
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file() and not any(skip in str(file_path) 
                                              for skip in ['.git', 'node_modules', '__pycache__']):
                ext = file_path.suffix.lower()
                if not ext:
                    ext = 'no_extension'
                file_counts[ext] = file_counts.get(ext, 0) + 1
        
        return file_counts
    
    def analyze_directory_sizes(self) -> Dict[str, Dict]:
        """Analyze directory sizes and file counts"""
        directory_stats = {}
        
        for item in self.project_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                file_count = len([f for f in item.rglob('*') if f.is_file()])
                
                try:
                    # Get directory size
                    result = subprocess.run(['du', '-sh', str(item)], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        size = result.stdout.split('\t')[0]
                    else:
                        size = "unknown"
                except:
                    size = "unknown"
                
                directory_stats[item.name] = {
                    'file_count': file_count,
                    'size': size
                }
        
        return directory_stats
    
    def get_git_info(self) -> Dict[str, str]:
        """Get git repository information"""
        git_info = {}
        
        try:
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            git_info['branch'] = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Get last commit
            result = subprocess.run(['git', 'log', '-1', '--oneline'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            git_info['last_commit'] = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Get commit count
            result = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            git_info['commit_count'] = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            git_info['has_changes'] = len(result.stdout.strip()) > 0 if result.returncode == 0 else False
            
        except:
            git_info = {'branch': 'unknown', 'last_commit': 'unknown', 
                       'commit_count': 'unknown', 'has_changes': False}
        
        return git_info
    
    def get_dependencies_info(self) -> Dict[str, List[str]]:
        """Get information about project dependencies"""
        deps_info = {'python': [], 'node': [], 'docker': []}
        
        # Python dependencies
        requirements_files = list(self.project_root.rglob('requirements*.txt'))
        for req_file in requirements_files:
            try:
                with open(req_file, 'r') as f:
                    deps_info['python'].extend([line.strip() for line in f 
                                              if line.strip() and not line.startswith('#')])
            except:
                pass
        
        # Node dependencies
        package_files = list(self.project_root.rglob('package.json'))
        for pkg_file in package_files:
            try:
                with open(pkg_file, 'r') as f:
                    data = json.load(f)
                    if 'dependencies' in data:
                        deps_info['node'].extend(list(data['dependencies'].keys()))
                    if 'devDependencies' in data:
                        deps_info['node'].extend(list(data['devDependencies'].keys()))
            except:
                pass
        
        # Docker services
        docker_files = list(self.project_root.rglob('docker-compose*.yml'))
        for docker_file in docker_files:
            try:
                with open(docker_file, 'r') as f:
                    content = f.read()
                    # Simple extraction of service names
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith(' ') and ':' in line and 'version' not in line:
                            service = line.split(':')[0].strip()
                            if service not in ['services', 'version', 'volumes', 'networks']:
                                deps_info['docker'].append(service)
            except:
                pass
        
        # Remove duplicates
        for key in deps_info:
            deps_info[key] = list(set(deps_info[key]))
        
        return deps_info
    
    def generate_readme(self) -> str:
        """Generate comprehensive README.md"""
        
        structure_tree = self.generate_structure_tree()
        file_counts = self.count_files_by_type()
        dir_stats = self.analyze_directory_sizes()
        git_info = self.get_git_info()
        deps_info = self.get_dependencies_info()
        
        readme_content = f"""# Persian Legal AI Training System

## 🚀 Project Overview

The Persian Legal AI Training System is a sophisticated artificial intelligence platform designed for Persian legal document processing and analysis. This system implements advanced AI models including DoRA (Weight-Decomposed Low-Rank Adaptation) and QR-Adaptor for joint quantization and rank optimization.

### 🎯 Key Features

- **Advanced AI Models**: DoRA and QR-Adaptor implementations
- **Persian Language Processing**: Specialized for Persian/Farsi legal documents
- **Microservices Architecture**: FastAPI backend with React TypeScript frontend
- **Production Ready**: Docker containerization with comprehensive monitoring
- **Comprehensive Testing**: Unit, integration, and end-to-end tests
- **Real-time Analytics**: Performance monitoring and optimization

---

## 📁 Project Structure

```
{structure_tree}
```

### 📊 Directory Statistics

| Directory | Files | Size |
|-----------|--------|------|"""

        for dir_name, stats in sorted(dir_stats.items()):
            readme_content += f"\n| {dir_name} | {stats['file_count']} | {stats['size']} |"

        readme_content += f"""

### 📈 File Distribution

| Extension | Count |
|-----------|-------|"""

        for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            readme_content += f"\n| {ext if ext != 'no_extension' else '(no ext)'} | {count} |"

        readme_content += f"""

---

## 🏗️ Architecture

### Backend (`src/backend/`)
- **FastAPI Framework**: High-performance async API
- **Authentication**: JWT-based security system
- **Database**: Multi-database support (PostgreSQL, SQLite)
- **AI Models**: Integrated DoRA and QR-Adaptor models
- **Monitoring**: Real-time performance tracking

### Frontend (`src/frontend/`)
- **React + TypeScript**: Modern type-safe frontend
- **Vite Build System**: Fast development and building
- **Tailwind CSS**: Utility-first styling
- **Component Architecture**: Reusable UI components

### AI Models (`models/`)
- **DoRA Implementation**: Weight-decomposed low-rank adaptation
- **QR-Adaptor**: Joint quantization and rank optimization
- **Persian BERT**: Specialized Persian language model
- **Training Pipeline**: Automated model training and evaluation

### Data Management (`data/`)
- **Databases**: Structured data storage with FTS5 search
- **Datasets**: Legal document datasets and training data
- **Exports**: Report generation and data export utilities

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10.12 or higher
- **Node.js**: 16.x or higher
- **Docker**: Latest version
- **Git**: For version control

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd persian-legal-ai-system
   ```

2. **Set up the backend**
   ```bash
   cd src/backend
   pip install -r ../../configs/shared/requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd src/frontend
   npm install
   ```

4. **Start with Docker**
   ```bash
   docker-compose -f configs/development/docker-compose.yml up -d
   ```

### Development Scripts

Located in `scripts/` directory:

- **Setup**: `scripts/setup/setup.sh` - Initial project setup
- **Start System**: `scripts/setup/start_system.sh` - Start all services
- **Testing**: `scripts/testing/run_exhaustive_tests.sh` - Run all tests
- **Deployment**: `scripts/deployment/deploy_docker.sh` - Deploy to production

---

## 🧪 Testing

### Test Structure

- **Unit Tests** (`tests/unit/`): Component-level testing
- **Integration Tests** (`tests/integration/`): Service integration testing
- **E2E Tests** (`tests/e2e/`): End-to-end user flow testing
- **Performance Tests** (`tests/performance/`): Load and performance testing

### Running Tests

```bash
# Run all tests
./scripts/testing/comprehensive-test.sh

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/
```

---

## 📚 Documentation

### Available Documentation

- **API Documentation** (`docs/api/`): REST API endpoints and schemas
- **User Guides** (`docs/guides/`): Setup, deployment, and usage guides
- **Development Docs** (`docs/development/`): Coding standards and contribution guide
- **Technical Reports** (`docs/reports/`): System analysis and audit reports
- **Architecture Analysis** (`docs/analysis/`): Technical architecture documentation

### Key Documents

- [Deployment Guide](docs/guides/DEPLOYMENT_GUIDE.md)
- [System Start Guide](docs/guides/START_SYSTEM.md)
- [Backend Implementation](docs/guides/backend-implementation-guide.md)
- [Rapid Training Guide](docs/guides/RAPID_TRAINING_GUIDE.md)

---

## 🔧 Configuration

### Environment Configuration

Configuration files are organized in `configs/` by environment:

- **Development** (`configs/development/`): Local development settings
- **Production** (`configs/production/`): Production deployment settings
- **Testing** (`configs/testing/`): Test environment configuration
- **Shared** (`configs/shared/`): Common configuration files

### Key Configuration Files

- `configs/development/docker-compose.yml` - Development services
- `configs/production/docker-compose.production.yml` - Production services
- `configs/shared/requirements.txt` - Python dependencies
- `src/frontend/package.json` - Node.js dependencies

---

## 🚀 Deployment

### Docker Deployment

```bash
# Development
docker-compose -f configs/development/docker-compose.yml up -d

# Production
docker-compose -f configs/production/docker-compose.production.yml up -d
```

### Manual Deployment

```bash
# Backend
cd src/backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd src/frontend
npm run build
npm run preview
```

---

## 📊 Dependencies

### Python Dependencies ({len(deps_info['python'])} packages)
"""

        if deps_info['python']:
            for dep in sorted(deps_info['python'])[:10]:
                readme_content += f"- {dep}\n"
            if len(deps_info['python']) > 10:
                readme_content += f"- ... and {len(deps_info['python']) - 10} more\n"

        readme_content += f"""
### Node.js Dependencies ({len(deps_info['node'])} packages)
"""

        if deps_info['node']:
            for dep in sorted(deps_info['node'])[:10]:
                readme_content += f"- {dep}\n"
            if len(deps_info['node']) > 10:
                readme_content += f"- ... and {len(deps_info['node']) - 10} more\n"

        readme_content += f"""
### Docker Services
"""
        
        if deps_info['docker']:
            for service in sorted(deps_info['docker']):
                readme_content += f"- {service}\n"

        readme_content += f"""
---

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Standards

- **Python**: Follow PEP 8 style guide
- **TypeScript**: Use ESLint and Prettier configurations
- **Documentation**: Update docs for any API changes
- **Testing**: Add tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

For support and questions:

- **Documentation**: Check the `docs/` directory
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions

---

## 📊 Project Statistics

- **Repository**: {git_info['branch']} branch
- **Last Commit**: {git_info['last_commit']}
- **Total Commits**: {git_info['commit_count']}
- **Uncommitted Changes**: {'Yes' if git_info['has_changes'] else 'No'}
- **Total Files**: {sum(file_counts.values())}
- **Documentation Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

**Persian Legal AI Training System** - Advanced AI for Persian Legal Document Processing
"""

        return readme_content
    
    def generate_project_overview(self) -> str:
        """Generate project overview document"""
        
        overview = f"""# Persian Legal AI - Project Overview

## Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Organization Summary

This document provides a comprehensive overview of the Persian Legal AI Training System project structure after the major reorganization completed on {datetime.now().strftime('%Y-%m-%d')}.

### Organization Principles

The project has been reorganized following modern software architecture principles:

1. **Separation of Concerns**: Clear separation between frontend, backend, and shared components
2. **Environment-based Configuration**: Separate configurations for development, production, and testing
3. **Test-Driven Structure**: Comprehensive test organization by type and scope
4. **Documentation-First**: Extensive documentation organized by purpose and audience
5. **Script Automation**: All automation scripts organized by function

### Directory Structure Rationale

#### `src/` - Source Code
All application source code is centralized under the `src/` directory:

- **`src/backend/`**: FastAPI backend application
  - Modular architecture with clear API, database, and service layers
  - Authentication and authorization systems
  - AI model integration and training services
  - Monitoring and optimization components

- **`src/frontend/`**: React TypeScript frontend application
  - Component-based architecture
  - Type-safe development with TypeScript
  - Modern build tools (Vite) and styling (Tailwind CSS)
  - Comprehensive testing setup

- **`src/shared/`**: Shared utilities and types
  - Common TypeScript types and interfaces
  - Shared constants and configuration
  - Utility functions used across frontend and backend

#### `models/` - AI Models
Centralized AI model storage and management:

- **DoRA Models**: Weight-decomposed low-rank adaptation implementations
- **QR-Adaptor**: Joint quantization and rank optimization
- **Persian BERT**: Specialized Persian language processing models
- **Shared Utilities**: Common model utilities and base classes

#### `data/` - Data Management
Consolidated data storage and management:

- **Databases**: All database files and schemas
- **Datasets**: Training and validation data
- **Exports**: Generated reports and data exports
- **Model Artifacts**: Trained model checkpoints and weights

#### `configs/` - Configuration Management
Environment-based configuration organization:

- **Development**: Local development settings and Docker configurations
- **Production**: Production deployment configurations
- **Testing**: Test environment settings
- **Shared**: Common configuration files and templates

#### `tests/` - Testing Infrastructure
Comprehensive testing organization:

- **Unit Tests**: Component-level testing for backend and frontend
- **Integration Tests**: Service integration and API testing
- **E2E Tests**: End-to-end user workflow testing
- **Performance Tests**: Load testing and performance benchmarks

#### `scripts/` - Automation Scripts
Organized automation and utility scripts:

- **Deployment**: Production deployment and CI/CD scripts
- **Maintenance**: System maintenance and health check scripts
- **Setup**: Installation and environment setup scripts
- **Testing**: Test execution and validation scripts
- **Utils**: General utility and development helper scripts

#### `docs/` - Documentation
Comprehensive documentation organization:

- **API Documentation**: REST API specifications and schemas
- **User Guides**: Setup, deployment, and usage documentation
- **Development Docs**: Coding standards and contribution guidelines
- **Technical Reports**: System analysis and audit reports
- **Architecture Analysis**: Technical architecture documentation

### Key Benefits of This Organization

1. **Developer Experience**: Clear navigation and logical file placement
2. **Maintainability**: Easy to understand and modify project structure
3. **Scalability**: Structure supports project growth and team expansion
4. **CI/CD Ready**: Clear separation enables efficient build and deployment pipelines
5. **Testing**: Comprehensive test organization supports quality assurance
6. **Documentation**: Extensive documentation supports onboarding and maintenance

### Migration Impact

The reorganization process:

1. **Preserved Functionality**: All existing functionality maintained
2. **Updated Import Paths**: All import statements updated to reflect new structure
3. **Configuration Updates**: Docker and build configurations updated
4. **Documentation Updates**: All documentation reflects new structure
5. **Backup Created**: Complete backup created before reorganization

### Next Steps

1. **Validation**: Run comprehensive validation to ensure all systems work correctly
2. **Testing**: Execute full test suite to verify functionality
3. **Documentation Review**: Review and update any remaining documentation
4. **Team Onboarding**: Update team documentation with new structure
5. **CI/CD Updates**: Update continuous integration and deployment pipelines

---

This organization provides a solid foundation for the continued development and scaling of the Persian Legal AI Training System.
"""
        
        return overview

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate organization documentation')
    parser.add_argument('project_root', nargs='?', default='.', 
                       help='Project root directory (default: current directory)')
    parser.add_argument('--readme', action='store_true', help='Generate README.md')
    parser.add_argument('--overview', action='store_true', help='Generate project overview')
    parser.add_argument('--all', action='store_true', help='Generate all documentation')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        print(f"Error: Project root does not exist: {project_root}")
        sys.exit(1)
    
    generator = DocumentationGenerator(str(project_root))
    
    if args.all or args.readme:
        readme_content = generator.generate_readme()
        readme_file = project_root / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ Generated README.md: {readme_file}")
    
    if args.all or args.overview:
        overview_content = generator.generate_project_overview()
        overview_file = project_root / f'PROJECT_OVERVIEW_{generator.timestamp}.md'
        with open(overview_file, 'w', encoding='utf-8') as f:
            f.write(overview_content)
        print(f"✅ Generated project overview: {overview_file}")
    
    if not (args.readme or args.overview or args.all):
        print("No documentation type specified. Use --readme, --overview, or --all")
        sys.exit(1)
    
    print("✅ Documentation generation completed!")

if __name__ == '__main__':
    main()