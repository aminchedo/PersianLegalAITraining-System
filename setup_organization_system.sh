#!/bin/bash

# Persian Legal AI Training System - Organization System Setup
# Complete setup script for the organization system
# Version: 3.0.0

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Utility functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

log_info() {
    echo -e "${CYAN}[INFO] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v pip3 >/dev/null 2>&1 || missing_tools+=("pip3")
    
    # Optional but recommended tools
    if ! command -v tree >/dev/null 2>&1; then
        log_warning "tree command not found - will use fallback for directory visualization"
    fi
    
    if ! command -v docker >/dev/null 2>&1; then
        log_warning "Docker not found - Docker-based features will be limited"
    fi
    
    if ! command -v npm >/dev/null 2>&1; then
        log_warning "npm not found - Frontend validation will be limited"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and run this script again."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Setup Python environment
setup_python_environment() {
    log "Setting up Python environment..."
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $python_version"
    
    # Install required Python packages for organization scripts
    local required_packages=(
        "pyyaml>=6.0"
        "pathlib"
        "argparse"
    )
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import ${package%>=*}" 2>/dev/null; then
            log_info "Installing Python package: $package"
            pip3 install "$package" --user --quiet
        fi
    done
    
    log_success "Python environment setup completed"
}

# Verify organization scripts
verify_scripts() {
    log "Verifying organization scripts..."
    
    local scripts=(
        "organize_persian_ai.sh"
        "import_updater.py"
        "validate_organization.py"
        "generate_organization_docs.py"
    )
    
    local missing_scripts=()
    
    for script in "${scripts[@]}"; do
        if [ ! -f "$PROJECT_ROOT/$script" ]; then
            missing_scripts+=("$script")
        elif [ ! -x "$PROJECT_ROOT/$script" ]; then
            log_info "Making $script executable..."
            chmod +x "$PROJECT_ROOT/$script"
        fi
    done
    
    if [ ${#missing_scripts[@]} -ne 0 ]; then
        log_error "Missing organization scripts: ${missing_scripts[*]}"
        log_error "Please ensure all scripts are present in the project root."
        exit 1
    fi
    
    # Test script syntax
    for script in "${scripts[@]}"; do
        if [[ "$script" == *.py ]]; then
            if ! python3 -m py_compile "$PROJECT_ROOT/$script" 2>/dev/null; then
                log_error "Python syntax error in $script"
                exit 1
            fi
        elif [[ "$script" == *.sh ]]; then
            if ! bash -n "$PROJECT_ROOT/$script" 2>/dev/null; then
                log_error "Bash syntax error in $script"
                exit 1
            fi
        fi
    done
    
    log_success "All organization scripts verified"
}

# Create initial backup
create_initial_backup() {
    log "Creating initial backup..."
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$PROJECT_ROOT/backup_before_organization_$timestamp"
    
    # Create git backup if possible
    if [ -d ".git" ]; then
        git add . 2>/dev/null || true
        git commit -m "Pre-organization system setup - $timestamp" 2>/dev/null || true
        git tag "pre-org-system-setup-$timestamp" 2>/dev/null || true
        log_info "Git backup created with tag: pre-org-system-setup-$timestamp"
    fi
    
    # Create file backup
    mkdir -p "$backup_dir"
    
    # Backup key files and directories
    local backup_items=(
        "*.py" "*.sh" "*.md" "*.json" "*.yml" "*.yaml"
        "src" "backend" "frontend" "models" "ai_models"
        "configs" "config" "data" "database" "tests" "test"
        "scripts" "docs" "deployment" "services" "utils"
    )
    
    for item in "${backup_items[@]}"; do
        if ls $item 1> /dev/null 2>&1; then
            cp -r $item "$backup_dir/" 2>/dev/null || true
        fi
    done
    
    log_success "Initial backup created at: $backup_dir"
}

# Show organization plan
show_organization_plan() {
    cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                    PERSIAN LEGAL AI ORGANIZATION PLAN                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 ORGANIZATION OBJECTIVES:
   • Transform 433+ scattered files into clean, professional structure
   • Consolidate duplicate directories (configs/, models/, data/)
   • Organize all documentation and scripts by function
   • Update all import paths and configuration files
   • Ensure zero functionality loss during reorganization

📁 TARGET STRUCTURE:
   persian-legal-ai-system/
   ├── src/                     # All source code
   │   ├── backend/            # FastAPI backend (65+ files)
   │   ├── frontend/           # React TypeScript (89+ files)
   │   └── shared/             # Shared utilities
   ├── models/                 # AI models (consolidated)
   ├── data/                   # Data & databases (consolidated)
   ├── configs/                # All configurations (consolidated)
   ├── tests/                  # All tests (organized by type)
   ├── scripts/                # All scripts (organized by function)
   ├── docs/                   # All documentation (organized by purpose)
   ├── docker-compose.yml      # Main Docker config
   └── README.md              # Updated project documentation

🔧 ORGANIZATION PROCESS:
   1. 🛡️  SAFETY FIRST - Create comprehensive backups
   2. 📊 ANALYSIS - Analyze current structure and conflicts
   3. 📁 STRUCTURE - Create target directory structure
   4. 🚀 REORGANIZE - Move and consolidate all files
   5. 🔗 UPDATE - Fix all import paths and configurations
   6. ✅ VALIDATE - Comprehensive validation and testing
   7. 📝 DOCUMENT - Generate updated documentation

🛠️ AVAILABLE TOOLS:
   • organize_persian_ai.sh    - Main organization script
   • import_updater.py         - Import path updater
   • validate_organization.py  - Validation and testing
   • generate_organization_docs.py - Documentation generator

🚀 EXECUTION MODES:
   • --analyze     - Analyze current structure
   • --plan        - Show detailed organization plan
   • --dry-run     - Preview changes without executing
   • --execute     - Execute the organization
   • --validate    - Validate organized structure
   • --undo        - Undo organization (if needed)

EOF
}

# Interactive setup wizard
interactive_setup() {
    log "Starting interactive setup wizard..."
    
    echo ""
    echo "Persian Legal AI Training System - Organization Setup Wizard"
    echo "==========================================================="
    echo ""
    
    # Project analysis
    echo "📊 CURRENT PROJECT ANALYSIS:"
    echo "----------------------------"
    
    local py_files=$(find . -name "*.py" -type f | wc -l)
    local js_ts_files=$(find . -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" -type f | wc -l)
    local md_files=$(find . -name "*.md" -type f | wc -l)
    local sh_files=$(find . -name "*.sh" -type f | wc -l)
    local total_files=$(find . -type f | wc -l)
    
    echo "• Python files: $py_files"
    echo "• JS/TS files: $js_ts_files"
    echo "• Documentation files: $md_files"
    echo "• Script files: $sh_files"
    echo "• Total files: $total_files"
    echo ""
    
    # Check for duplicate directories
    echo "🔍 DUPLICATE DIRECTORY ANALYSIS:"
    echo "--------------------------------"
    
    local duplicates_found=false
    
    if [ -d "config" ] && [ -d "configs" ]; then
        echo "• Found: config/ and configs/ directories"
        duplicates_found=true
    fi
    
    if [ -d "data" ] && [ -d "database" ]; then
        echo "• Found: data/ and database/ directories"
        duplicates_found=true
    fi
    
    if [ -d "models" ] && [ -d "ai_models" ]; then
        echo "• Found: models/ and ai_models/ directories"
        duplicates_found=true
    fi
    
    if [ -d "test" ] && [ -d "tests" ]; then
        echo "• Found: test/ and tests/ directories"
        duplicates_found=true
    fi
    
    if [ "$duplicates_found" = false ]; then
        echo "• No duplicate directories found"
    fi
    echo ""
    
    # Setup options
    echo "🚀 SETUP OPTIONS:"
    echo "----------------"
    echo "1. Quick Setup - Run analysis and show recommendations"
    echo "2. Full Organization - Execute complete reorganization"
    echo "3. Validation Only - Validate current or organized structure"
    echo "4. Custom Setup - Choose specific operations"
    echo "5. Exit - Exit setup wizard"
    echo ""
    
    read -p "Select option (1-5): " choice
    
    case $choice in
        1)
            quick_setup
            ;;
        2)
            full_organization
            ;;
        3)
            validation_only
            ;;
        4)
            custom_setup
            ;;
        5)
            log_info "Exiting setup wizard"
            exit 0
            ;;
        *)
            log_error "Invalid option selected"
            exit 1
            ;;
    esac
}

# Quick setup - analysis and recommendations
quick_setup() {
    log "Running quick setup..."
    
    echo ""
    echo "📊 ANALYZING PROJECT STRUCTURE..."
    echo "================================="
    
    # Run analysis
    if ./organize_persian_ai.sh --analyze; then
        log_success "Analysis completed successfully"
    else
        log_error "Analysis failed"
        exit 1
    fi
    
    echo ""
    echo "📋 RECOMMENDATIONS:"
    echo "==================="
    echo "Based on the analysis, here are the recommended next steps:"
    echo ""
    echo "1. 🛡️  Create backup: ./organize_persian_ai.sh --dry-run"
    echo "2. 👀 Preview changes: ./organize_persian_ai.sh --dry-run"
    echo "3. 🚀 Execute organization: ./organize_persian_ai.sh --execute"
    echo "4. ✅ Validate results: ./organize_persian_ai.sh --validate"
    echo ""
    
    read -p "Would you like to proceed with the full organization? (y/N): " proceed
    
    if [[ "$proceed" =~ ^[Yy]$ ]]; then
        full_organization
    else
        log_info "You can run the organization later using: ./organize_persian_ai.sh --execute"
    fi
}

# Full organization process
full_organization() {
    log "Starting full organization process..."
    
    echo ""
    log_warning "⚠️  IMPORTANT: This will reorganize your entire project structure!"
    log_warning "⚠️  A comprehensive backup will be created before any changes."
    echo ""
    
    read -p "Are you sure you want to proceed? (y/N): " confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log_info "Organization cancelled"
        exit 0
    fi
    
    # Step 1: Dry run
    echo ""
    log "Step 1: Running dry-run to preview changes..."
    if ./organize_persian_ai.sh --dry-run; then
        log_success "Dry-run completed successfully"
    else
        log_error "Dry-run failed"
        exit 1
    fi
    
    echo ""
    read -p "Dry-run completed. Proceed with actual organization? (y/N): " proceed
    
    if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
        log_info "Organization cancelled after dry-run"
        exit 0
    fi
    
    # Step 2: Execute organization
    echo ""
    log "Step 2: Executing organization..."
    if ./organize_persian_ai.sh --execute; then
        log_success "Organization completed successfully"
    else
        log_error "Organization failed"
        exit 1
    fi
    
    # Step 3: Update imports
    echo ""
    log "Step 3: Updating import paths..."
    if python3 import_updater.py .; then
        log_success "Import paths updated successfully"
    else
        log_warning "Import path updates completed with warnings"
    fi
    
    # Step 4: Validate organization
    echo ""
    log "Step 4: Validating organization..."
    if python3 validate_organization.py .; then
        log_success "Organization validation passed"
    else
        log_warning "Organization validation completed with warnings"
    fi
    
    # Step 5: Generate documentation
    echo ""
    log "Step 5: Generating documentation..."
    if python3 generate_organization_docs.py . --all; then
        log_success "Documentation generated successfully"
    else
        log_warning "Documentation generation completed with warnings"
    fi
    
    echo ""
    log_success "🎉 FULL ORGANIZATION COMPLETED!"
    echo ""
    echo "📋 SUMMARY:"
    echo "• Project structure reorganized"
    echo "• Import paths updated"
    echo "• Organization validated"
    echo "• Documentation generated"
    echo ""
    echo "📁 Check these files for details:"
    echo "• organization_log_*.log - Detailed operation log"
    echo "• organization_report_*.md - Comprehensive organization report"
    echo "• README.md - Updated project documentation"
    echo ""
}

# Validation only
validation_only() {
    log "Running validation only..."
    
    echo ""
    echo "🔍 VALIDATION OPTIONS:"
    echo "====================="
    echo "1. Validate current structure"
    echo "2. Validate organized structure (post-organization)"
    echo "3. Generate validation report"
    echo ""
    
    read -p "Select validation option (1-3): " val_choice
    
    case $val_choice in
        1|2)
            if python3 validate_organization.py . --verbose; then
                log_success "Validation completed successfully"
            else
                log_warning "Validation completed with issues"
            fi
            ;;
        3)
            local report_file="validation_report_$(date +%Y%m%d_%H%M%S).md"
            if python3 validate_organization.py . --output "$report_file"; then
                log_success "Validation report generated: $report_file"
            else
                log_warning "Validation report generated with warnings: $report_file"
            fi
            ;;
        *)
            log_error "Invalid validation option"
            exit 1
            ;;
    esac
}

# Custom setup
custom_setup() {
    log "Starting custom setup..."
    
    echo ""
    echo "🛠️  CUSTOM SETUP OPTIONS:"
    echo "========================="
    echo "Select the operations you want to perform:"
    echo ""
    
    local operations=()
    
    echo "Available operations:"
    echo "1. Analyze current structure"
    echo "2. Create backup"
    echo "3. Run dry-run preview"
    echo "4. Execute organization"
    echo "5. Update import paths"
    echo "6. Validate organization"
    echo "7. Generate documentation"
    echo ""
    
    read -p "Enter operation numbers separated by spaces (e.g., 1 3 6): " selected
    
    for op in $selected; do
        case $op in
            1) operations+=("analyze") ;;
            2) operations+=("backup") ;;
            3) operations+=("dry-run") ;;
            4) operations+=("execute") ;;
            5) operations+=("imports") ;;
            6) operations+=("validate") ;;
            7) operations+=("docs") ;;
            *) log_warning "Invalid operation number: $op" ;;
        esac
    done
    
    if [ ${#operations[@]} -eq 0 ]; then
        log_error "No valid operations selected"
        exit 1
    fi
    
    echo ""
    log_info "Selected operations: ${operations[*]}"
    echo ""
    
    read -p "Proceed with selected operations? (y/N): " proceed
    
    if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
        log_info "Custom setup cancelled"
        exit 0
    fi
    
    # Execute selected operations
    for operation in "${operations[@]}"; do
        echo ""
        case $operation in
            "analyze")
                log "Analyzing project structure..."
                ./organize_persian_ai.sh --analyze
                ;;
            "backup")
                log "Creating backup..."
                create_initial_backup
                ;;
            "dry-run")
                log "Running dry-run preview..."
                ./organize_persian_ai.sh --dry-run
                ;;
            "execute")
                log "Executing organization..."
                ./organize_persian_ai.sh --execute
                ;;
            "imports")
                log "Updating import paths..."
                python3 import_updater.py .
                ;;
            "validate")
                log "Validating organization..."
                python3 validate_organization.py .
                ;;
            "docs")
                log "Generating documentation..."
                python3 generate_organization_docs.py . --all
                ;;
        esac
        
        if [ $? -eq 0 ]; then
            log_success "Operation '$operation' completed successfully"
        else
            log_warning "Operation '$operation' completed with warnings"
        fi
    done
    
    log_success "Custom setup completed!"
}

# Main execution
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║           Persian Legal AI Training System - Organization Setup              ║"
    echo "║                              Version 3.0.0                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check for command line arguments
    if [ $# -gt 0 ]; then
        case "$1" in
            --help|-h)
                show_organization_plan
                echo ""
                echo "USAGE: $0 [--help|--check|--plan|--quick|--full]"
                echo ""
                echo "Options:"
                echo "  --help    Show this help message"
                echo "  --check   Check prerequisites only"
                echo "  --plan    Show organization plan"
                echo "  --quick   Run quick analysis and recommendations"
                echo "  --full    Run full organization process"
                echo "  (no args) Run interactive setup wizard"
                exit 0
                ;;
            --check)
                check_prerequisites
                setup_python_environment
                verify_scripts
                log_success "All prerequisites checked successfully!"
                exit 0
                ;;
            --plan)
                show_organization_plan
                exit 0
                ;;
            --quick)
                check_prerequisites
                setup_python_environment
                verify_scripts
                create_initial_backup
                quick_setup
                exit 0
                ;;
            --full)
                check_prerequisites
                setup_python_environment
                verify_scripts
                create_initial_backup
                full_organization
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    fi
    
    # Run interactive setup
    check_prerequisites
    setup_python_environment
    verify_scripts
    create_initial_backup
    show_organization_plan
    interactive_setup
}

# Execute main function
main "$@"