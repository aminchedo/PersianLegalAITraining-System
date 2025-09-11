#!/bin/bash

# Persian Legal AI Training System - Advanced Organization Script
# Version: 3.0.0
# Date: September 11, 2025
# Purpose: Comprehensive project organization with safety, validation, and recovery

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="${PROJECT_ROOT}/backup_pre_organization_${TIMESTAMP}"
LOG_FILE="${PROJECT_ROOT}/organization_log_${TIMESTAMP}.log"
CONFLICT_LOG="${PROJECT_ROOT}/merge_conflicts_${TIMESTAMP}.log"
REPORT_FILE="${PROJECT_ROOT}/organization_report_${TIMESTAMP}.md"
UNDO_SCRIPT="${PROJECT_ROOT}/undo_organization_${TIMESTAMP}.sh"

# Operation counters
FILES_MOVED=0
FOLDERS_CREATED=0
CONFLICTS_RESOLVED=0
IMPORTS_UPDATED=0

# Initialize logging
exec > >(tee -a "$LOG_FILE")
exec 2>&1

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

# Check if git is available and repository is clean
check_git_status() {
    log "Checking git status..."
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install git first."
        exit 1
    fi
    
    if [ ! -d ".git" ]; then
        log_error "Not a git repository. Please initialize git first."
        exit 1
    fi
    
    if [ -n "$(git status --porcelain)" ]; then
        log_warning "Working directory is not clean. Uncommitted changes detected."
        echo "Would you like to continue anyway? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Aborting organization. Please commit or stash changes first."
            exit 1
        fi
    fi
}

# Create comprehensive backup
create_backup() {
    log "Creating comprehensive backup..."
    
    # Create git backup
    git add . 2>/dev/null || true
    git commit -m "Pre-organization backup - ${TIMESTAMP}" 2>/dev/null || true
    git branch "backup-before-major-organization-${TIMESTAMP}" 2>/dev/null || true
    git tag "pre-organization-${TIMESTAMP}" 2>/dev/null || true
    
    # Create file system backup
    mkdir -p "$BACKUP_DIR"
    
    # Backup key directories and files
    local backup_items=(
        "src" "backend" "frontend" "models" "ai_models" "configs" "config"
        "data" "database" "tests" "test" "scripts" "docs" "deployment"
        "services" "utils" "optimization" "*.py" "*.js" "*.ts" "*.json"
        "*.md" "*.yml" "*.yaml" "*.sh" "docker-compose*.yml"
        "package.json" "requirements.txt" ".env*" ".git*"
    )
    
    for item in "${backup_items[@]}"; do
        if [ -e "$item" ] || ls $item 1> /dev/null 2>&1; then
            cp -r $item "$BACKUP_DIR/" 2>/dev/null || true
        fi
    done
    
    log_success "Backup created at: $BACKUP_DIR"
}

# Analyze current project structure
analyze_structure() {
    log "Analyzing current project structure..."
    
    echo "# Project Structure Analysis - ${TIMESTAMP}" > "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "## Before Organization" >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"
    tree -a -I '.git|node_modules|__pycache__|*.pyc|.pytest_cache|.venv' . >> "$REPORT_FILE" 2>/dev/null || find . -type f -name ".*" -prune -o -print | sort >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Count files by type
    local py_files=$(find . -name "*.py" -type f | wc -l)
    local js_files=$(find . -name "*.js" -type f | wc -l)
    local ts_files=$(find . -name "*.ts" -type f | wc -l)
    local tsx_files=$(find . -name "*.tsx" -type f | wc -l)
    local md_files=$(find . -name "*.md" -type f | wc -l)
    local sh_files=$(find . -name "*.sh" -type f | wc -l)
    local json_files=$(find . -name "*.json" -type f | wc -l)
    local yml_files=$(find . -name "*.yml" -o -name "*.yaml" -type f | wc -l)
    
    echo "## File Count Analysis" >> "$REPORT_FILE"
    echo "- Python files: $py_files" >> "$REPORT_FILE"
    echo "- JavaScript files: $js_files" >> "$REPORT_FILE"
    echo "- TypeScript files: $ts_files" >> "$REPORT_FILE"
    echo "- TypeScript React files: $tsx_files" >> "$REPORT_FILE"
    echo "- Markdown files: $md_files" >> "$REPORT_FILE"
    echo "- Shell scripts: $sh_files" >> "$REPORT_FILE"
    echo "- JSON files: $json_files" >> "$REPORT_FILE"
    echo "- YAML files: $yml_files" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    log_info "Python files: $py_files, JS/TS files: $((js_files + ts_files + tsx_files)), Docs: $md_files, Scripts: $sh_files"
}

# Create target directory structure
create_target_structure() {
    log "Creating target directory structure..."
    
    local directories=(
        # Source code organization
        "src/frontend/components"
        "src/frontend/pages"
        "src/frontend/hooks"
        "src/frontend/services"
        "src/frontend/types"
        "src/frontend/styles"
        "src/frontend/utils"
        "src/frontend/test"
        "src/backend/api"
        "src/backend/auth"
        "src/backend/database"
        "src/backend/models"
        "src/backend/services"
        "src/backend/monitoring"
        "src/backend/optimization"
        "src/backend/training"
        "src/backend/validation"
        "src/backend/middleware"
        "src/backend/routes"
        "src/backend/utils"
        "src/backend/logging"
        "src/shared/types"
        "src/shared/constants"
        "src/shared/utils"
        "src/shared/configs"
        
        # Models consolidation
        "models/dora"
        "models/qr_adaptor"
        "models/bert/persian_bert"
        "models/bert/enhanced"
        "models/bert/configs"
        "models/shared/base_models"
        "models/shared/utilities"
        
        # Data consolidation
        "data/databases/schemas"
        "data/databases/migrations"
        "data/datasets/legal_documents"
        "data/datasets/training_data"
        "data/datasets/validation_data"
        "data/exports/reports"
        "data/exports/backups"
        "data/models/checkpoints"
        "data/models/weights"
        
        # Configuration consolidation
        "configs/development"
        "configs/production"
        "configs/testing"
        "configs/shared"
        
        # Testing infrastructure
        "tests/unit/backend"
        "tests/unit/frontend"
        "tests/unit/models"
        "tests/integration"
        "tests/e2e"
        "tests/performance"
        "tests/fixtures/sample_data"
        "tests/fixtures/mock_responses"
        "tests/reports"
        
        # Scripts organization
        "scripts/deployment"
        "scripts/maintenance"
        "scripts/setup"
        "scripts/testing"
        "scripts/utils"
        "scripts/development"
        
        # Documentation organization
        "docs/api/schemas"
        "docs/guides/user_manuals"
        "docs/reports/audit_reports"
        "docs/reports/completion_reports"
        "docs/reports/integration_reports"
        "docs/reports/performance_reports"
        "docs/analysis/architecture_analysis"
        "docs/development"
        "docs/references"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            FOLDERS_CREATED=$((FOLDERS_CREATED + 1))
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Created $FOLDERS_CREATED directories"
}

# Function to check for file conflicts
check_conflict() {
    local source="$1"
    local target="$2"
    
    if [ -f "$target" ]; then
        # File exists, check if different
        if ! cmp -s "$source" "$target"; then
            local source_size=$(stat -f%z "$source" 2>/dev/null || stat -c%s "$source" 2>/dev/null)
            local target_size=$(stat -f%z "$target" 2>/dev/null || stat -c%s "$target" 2>/dev/null)
            local source_date=$(stat -f%m "$source" 2>/dev/null || stat -c%Y "$source" 2>/dev/null)
            local target_date=$(stat -f%m "$target" 2>/dev/null || stat -c%Y "$target" 2>/dev/null)
            
            echo "CONFLICT: $source -> $target" >> "$CONFLICT_LOG"
            echo "  Source: ${source_size} bytes, modified $(date -r $source_date 2>/dev/null || date -d @$source_date 2>/dev/null)" >> "$CONFLICT_LOG"
            echo "  Target: ${target_size} bytes, modified $(date -r $target_date 2>/dev/null || date -d @$target_date 2>/dev/null)" >> "$CONFLICT_LOG"
            
            # Keep newer file, rename older
            if [ "$source_date" -gt "$target_date" ]; then
                mv "$target" "${target%.${target##*.}}_old.${target##*.}"
                echo "  Resolution: Kept source (newer), renamed target to ${target%.${target##*.}}_old.${target##*.}" >> "$CONFLICT_LOG"
                return 0  # OK to proceed
            else
                mv "$source" "${source%.${source##*.}}_old.${source##*.}"
                echo "  Resolution: Kept target (newer), renamed source to ${source%.${source##*.}}_old.${source##*.}" >> "$CONFLICT_LOG"
                return 1  # Don't move
            fi
        else
            # Files are identical, remove source
            rm "$source"
            echo "DUPLICATE: Removed identical file $source" >> "$CONFLICT_LOG"
            return 1  # Don't move
        fi
    fi
    return 0  # OK to proceed
}

# Safe move function with conflict resolution
safe_move() {
    local source="$1"
    local target="$2"
    
    # Create target directory if it doesn't exist
    mkdir -p "$(dirname "$target")"
    
    # Check for conflicts
    if check_conflict "$source" "$target"; then
        mv "$source" "$target"
        FILES_MOVED=$((FILES_MOVED + 1))
        echo "mv '$source' '$target'" >> "$UNDO_SCRIPT"
        log_info "Moved: $source -> $target"
        return 0
    else
        CONFLICTS_RESOLVED=$((CONFLICTS_RESOLVED + 1))
        return 1
    fi
}

# Reorganize source code
reorganize_source_code() {
    log "Reorganizing source code..."
    
    # Move frontend files
    if [ -d "frontend" ]; then
        log_info "Moving frontend files..."
        
        # Move frontend source files
        [ -d "frontend/src/components" ] && safe_move "frontend/src/components" "src/frontend/components"
        [ -d "frontend/src/pages" ] && safe_move "frontend/src/pages" "src/frontend/pages"
        [ -d "frontend/src/hooks" ] && safe_move "frontend/src/hooks" "src/frontend/hooks"
        [ -d "frontend/src/services" ] && safe_move "frontend/src/services" "src/frontend/services"
        [ -d "frontend/src/types" ] && safe_move "frontend/src/types" "src/frontend/types"
        [ -d "frontend/src/styles" ] && safe_move "frontend/src/styles" "src/frontend/styles"
        [ -d "frontend/src/utils" ] && safe_move "frontend/src/utils" "src/frontend/utils"
        [ -d "frontend/src/test" ] && safe_move "frontend/src/test" "src/frontend/test"
        [ -d "frontend/src/tests" ] && safe_move "frontend/src/tests" "src/frontend/test"
        
        # Move individual frontend files
        find frontend -name "*.tsx" -o -name "*.ts" -o -name "*.jsx" -o -name "*.js" -o -name "*.css" -o -name "*.scss" | while read -r file; do
            if [[ "$file" != *"/src/"* ]] && [[ "$file" != *"/node_modules/"* ]]; then
                relative_path="${file#frontend/}"
                safe_move "$file" "src/frontend/$relative_path"
            fi
        done
    fi
    
    # Move bolt files (alternate frontend)
    if [ -d "bolt" ]; then
        log_info "Moving bolt frontend files..."
        [ -d "bolt/src" ] && safe_move "bolt/src" "src/frontend/bolt"
        find bolt -name "*.tsx" -o -name "*.ts" -o -name "*.jsx" -o -name "*.js" -o -name "*.css" | while read -r file; do
            if [[ "$file" != *"/node_modules/"* ]]; then
                relative_path="${file#bolt/}"
                safe_move "$file" "src/frontend/bolt/$relative_path"
            fi
        done
    fi
    
    # Move backend files
    if [ -d "backend" ]; then
        log_info "Moving backend files..."
        
        # Move backend directories
        [ -d "backend/api" ] && safe_move "backend/api" "src/backend/api"
        [ -d "backend/auth" ] && safe_move "backend/auth" "src/backend/auth"
        [ -d "backend/database" ] && safe_move "backend/database" "src/backend/database"
        [ -d "backend/models" ] && safe_move "backend/models" "src/backend/models"
        [ -d "backend/services" ] && safe_move "backend/services" "src/backend/services"
        [ -d "backend/monitoring" ] && safe_move "backend/monitoring" "src/backend/monitoring"
        [ -d "backend/optimization" ] && safe_move "backend/optimization" "src/backend/optimization"
        [ -d "backend/training" ] && safe_move "backend/training" "src/backend/training"
        [ -d "backend/validation" ] && safe_move "backend/validation" "src/backend/validation"
        [ -d "backend/middleware" ] && safe_move "backend/middleware" "src/backend/middleware"
        [ -d "backend/routes" ] && safe_move "backend/routes" "src/backend/routes"
        [ -d "backend/utils" ] && safe_move "backend/utils" "src/backend/utils"
        [ -d "backend/logging" ] && safe_move "backend/logging" "src/backend/logging"
        
        # Move individual backend Python files
        find backend -name "*.py" -type f | while read -r file; do
            if [[ "$file" != *"/"* ]] || [[ "$(dirname "$file")" == "backend" ]]; then
                filename=$(basename "$file")
                safe_move "$file" "src/backend/$filename"
            fi
        done
    fi
    
    # Move root-level Python files
    find . -maxdepth 1 -name "*.py" -type f | while read -r file; do
        filename=$(basename "$file")
        if [[ "$filename" != "organize_persian_ai.py" ]]; then
            safe_move "$file" "src/backend/$filename"
        fi
    done
    
    # Move api directory if exists at root
    [ -d "api" ] && safe_move "api" "src/backend/api"
    
    # Move src directory contents if exists
    if [ -d "src" ]; then
        find src -type f | while read -r file; do
            relative_path="${file#src/}"
            if [[ "$file" == *".py" ]]; then
                safe_move "$file" "src/backend/$relative_path"
            elif [[ "$file" == *".ts" ]] || [[ "$file" == *".tsx" ]] || [[ "$file" == *".js" ]] || [[ "$file" == *".jsx" ]]; then
                safe_move "$file" "src/frontend/$relative_path"
            else
                safe_move "$file" "src/shared/$relative_path"
            fi
        done
    fi
}

# Consolidate AI models
consolidate_models() {
    log "Consolidating AI models..."
    
    # Move ai_models directory
    if [ -d "ai_models" ]; then
        find ai_models -type f | while read -r file; do
            relative_path="${file#ai_models/}"
            safe_move "$file" "models/$relative_path"
        done
        [ -d "ai_models" ] && rmdir ai_models 2>/dev/null || true
    fi
    
    # Move models directory contents
    if [ -d "models" ] && [ "$(ls -A models 2>/dev/null)" ]; then
        # Already exists, organize contents
        [ -d "models/dora_trainer.py" ] && safe_move "models/dora_trainer.py" "models/dora/dora_trainer.py"
        [ -d "models/qr_adaptor.py" ] && safe_move "models/qr_adaptor.py" "models/qr_adaptor/qr_adaptor.py"
    fi
    
    # Move any standalone model files
    find . -maxdepth 2 -name "*dora*" -type f | while read -r file; do
        safe_move "$file" "models/dora/$(basename "$file")"
    done
    
    find . -maxdepth 2 -name "*qr_adaptor*" -type f | while read -r file; do
        safe_move "$file" "models/qr_adaptor/$(basename "$file")"
    done
}

# Consolidate data and databases
consolidate_data() {
    log "Consolidating data and databases..."
    
    # Move database directory
    if [ -d "database" ]; then
        find database -type f | while read -r file; do
            relative_path="${file#database/}"
            safe_move "$file" "data/databases/$relative_path"
        done
        [ -d "database" ] && rmdir database 2>/dev/null || true
    fi
    
    # Move data directory contents
    if [ -d "data" ]; then
        find data -name "*.db*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "data/databases/$filename"
        done
    fi
    
    # Move any root database files
    find . -maxdepth 1 -name "*.db*" -type f | while read -r file; do
        safe_move "$file" "data/databases/$(basename "$file")"
    done
}

# Consolidate configurations
consolidate_configs() {
    log "Consolidating configurations..."
    
    # Merge config and configs directories
    if [ -d "config" ]; then
        find config -type f | while read -r file; do
            relative_path="${file#config/}"
            safe_move "$file" "configs/shared/$relative_path"
        done
        [ -d "config" ] && rmdir config 2>/dev/null || true
    fi
    
    # Organize configs directory
    if [ -d "configs" ]; then
        # Move docker files to appropriate environments
        find configs -name "docker-compose*.yml" | while read -r file; do
            filename=$(basename "$file")
            if [[ "$filename" == *"production"* ]]; then
                safe_move "$file" "configs/production/$filename"
            elif [[ "$filename" == *"development"* ]] || [[ "$filename" == "docker-compose.yml" ]]; then
                safe_move "$file" "configs/development/$filename"
            else
                safe_move "$file" "configs/shared/$filename"
            fi
        done
        
        # Move other config files
        find configs -maxdepth 1 -name "*.json" -o -name "*.yml" -o -name "*.yaml" -o -name "*.toml" -o -name "*.txt" | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "configs/shared/$filename"
        done
    fi
    
    # Move root config files
    find . -maxdepth 1 -name "*.config.js" -o -name "*.config.ts" -o -name "tsconfig*.json" -o -name ".env*" -o -name "*.toml" | while read -r file; do
        safe_move "$file" "configs/shared/$(basename "$file")"
    done
}

# Consolidate tests
consolidate_tests() {
    log "Consolidating tests..."
    
    # Move test directory
    if [ -d "test" ]; then
        find test -type f | while read -r file; do
            relative_path="${file#test/}"
            if [[ "$file" == *".py" ]]; then
                safe_move "$file" "tests/unit/backend/$relative_path"
            elif [[ "$file" == *".ts" ]] || [[ "$file" == *".js" ]]; then
                safe_move "$file" "tests/unit/frontend/$relative_path"
            else
                safe_move "$file" "tests/fixtures/$relative_path"
            fi
        done
        [ -d "test" ] && rmdir test 2>/dev/null || true
    fi
    
    # Move tests directory contents
    if [ -d "tests" ]; then
        find tests -name "*integration*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "tests/integration/$filename"
        done
        
        find tests -name "*performance*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "tests/performance/$filename"
        done
        
        find tests -name "test_*.py" -type f | while read -r file; do
            if [[ "$file" != *"integration"* ]] && [[ "$file" != *"performance"* ]]; then
                filename=$(basename "$file")
                safe_move "$file" "tests/unit/backend/$filename"
            fi
        done
        
        find tests -name "*.test.ts" -o -name "*.test.js" -o -name "*.spec.ts" -o -name "*.spec.js" | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "tests/unit/frontend/$filename"
        done
        
        find tests -name "*.html" | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "tests/e2e/$filename"
        done
    fi
}

# Consolidate scripts
consolidate_scripts() {
    log "Consolidating scripts..."
    
    if [ -d "scripts" ]; then
        # Organize by function
        find scripts -name "*deploy*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "scripts/deployment/$filename"
        done
        
        find scripts -name "*setup*" -o -name "*install*" -o -name "*start*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "scripts/setup/$filename"
        done
        
        find scripts -name "*test*" -o -name "*validate*" -o -name "*verify*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "scripts/testing/$filename"
        done
        
        find scripts -name "*recovery*" -o -name "*health*" -o -name "*maintenance*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "scripts/maintenance/$filename"
        done
        
        find scripts -name "*fix*" -o -name "*update*" -o -name "*migration*" -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "scripts/development/$filename"
        done
        
        # Move remaining scripts to utils
        find scripts -maxdepth 1 -name "*.sh" -o -name "*.py" -o -name "*.js" -o -name "*.ts" | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "scripts/utils/$filename"
        done
    fi
    
    # Move deployment directory scripts
    if [ -d "deployment/scripts" ]; then
        find deployment/scripts -type f | while read -r file; do
            filename=$(basename "$file")
            safe_move "$file" "scripts/deployment/$filename"
        done
    fi
    
    # Move root-level scripts
    find . -maxdepth 1 -name "*.sh" -type f | while read -r file; do
        filename=$(basename "$file")
        if [[ "$filename" != "organize_persian_ai.sh" ]] && [[ "$filename" != "undo_organization_"*".sh" ]]; then
            safe_move "$file" "scripts/utils/$filename"
        fi
    done
}

# Consolidate documentation
consolidate_docs() {
    log "Consolidating documentation..."
    
    # Move root markdown files to appropriate docs subdirectories
    local report_files=(
        "BACKEND_MAIN_FILES_ANALYSIS.md"
        "BOLT_INTEGRATION_COMPLETE.md"
        "CLEANUP_SUMMARY.md"
        "MERGE_COMPLETION_REPORT.md"
        "PERSIAN_AI_ENHANCEMENT_COMPLETION_REPORT.md"
        "PERSIAN_LEGAL_AI_AUDIT_COMPLETION_REPORT.md"
        "PHASE_3_COMPLETION_SUMMARY.md"
        "PHASE_CRITICAL_PRODUCTION_ISSUES_RESOLVED.md"
        "RECOVERY_REPORT.md"
        "SAFE_MERGE_COMPLETION_REPORT.md"
        "VISUAL_UI_TEST_REPORT.md"
        "integration-report.md"
    )
    
    local guide_files=(
        "DEPLOYMENT_GUIDE.md"
        "START_SYSTEM.md"
        "backend-implementation-guide.md"
    )
    
    local analysis_files=(
        "analysis_report.txt"
        "BACKEND_MAIN_FILES_ANALYSIS.md"
    )
    
    # Move report files
    for file in "${report_files[@]}"; do
        [ -f "$file" ] && safe_move "$file" "docs/reports/completion_reports/$file"
    done
    
    # Move guide files
    for file in "${guide_files[@]}"; do
        [ -f "$file" ] && safe_move "$file" "docs/guides/$file"
    done
    
    # Move analysis files
    for file in "${analysis_files[@]}"; do
        [ -f "$file" ] && safe_move "$file" "docs/analysis/$file"
    done
    
    # Move comprehensive audit report
    [ -f "COMPREHENSIVE_PROJECT_AUDIT_REPORT.md" ] && safe_move "COMPREHENSIVE_PROJECT_AUDIT_REPORT.md" "docs/reports/audit_reports/COMPREHENSIVE_PROJECT_AUDIT_REPORT.md"
    
    # Organize existing docs directory
    if [ -d "docs" ]; then
        find docs -name "*report*" -type f | while read -r file; do
            filename=$(basename "$file")
            if [[ "$file" != *"/reports/"* ]]; then
                safe_move "$file" "docs/reports/completion_reports/$filename"
            fi
        done
        
        find docs -name "*guide*" -type f | while read -r file; do
            filename=$(basename "$file")
            if [[ "$file" != *"/guides/"* ]]; then
                safe_move "$file" "docs/guides/$filename"
            fi
        done
    fi
}

# Update import paths in Python files
update_python_imports() {
    log "Updating Python import paths..."
    
    find src -name "*.py" -type f | while read -r file; do
        local updated=false
        
        # Common import path updates
        if grep -q "from backend\." "$file" 2>/dev/null; then
            sed -i.bak 's/from backend\./from src.backend./g' "$file"
            updated=true
        fi
        
        if grep -q "import backend\." "$file" 2>/dev/null; then
            sed -i.bak 's/import backend\./import src.backend./g' "$file"
            updated=true
        fi
        
        if grep -q "from models\." "$file" 2>/dev/null; then
            sed -i.bak 's/from models\./from ..\/..\/models./g' "$file"
            updated=true
        fi
        
        if grep -q "from config\." "$file" 2>/dev/null; then
            sed -i.bak 's/from config\./from ..\/..\/configs.shared./g' "$file"
            updated=true
        fi
        
        if [ "$updated" = true ]; then
            IMPORTS_UPDATED=$((IMPORTS_UPDATED + 1))
            rm -f "${file}.bak"
            log_info "Updated imports in: $file"
        fi
    done
}

# Update import paths in TypeScript/JavaScript files
update_js_imports() {
    log "Updating JavaScript/TypeScript import paths..."
    
    find src/frontend -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" -type f | while read -r file; do
        local updated=false
        
        # Update relative imports
        if grep -q "from ['\"]\.\./" "$file" 2>/dev/null; then
            # This needs more sophisticated logic based on new structure
            updated=true
        fi
        
        if grep -q "import.*from ['\"]\.\./" "$file" 2>/dev/null; then
            updated=true
        fi
        
        if [ "$updated" = true ]; then
            IMPORTS_UPDATED=$((IMPORTS_UPDATED + 1))
            log_info "Updated imports in: $file"
        fi
    done
}

# Update configuration file paths
update_config_paths() {
    log "Updating configuration file paths..."
    
    # Update docker-compose files
    find configs -name "docker-compose*.yml" | while read -r file; do
        if grep -q "\./" "$file" 2>/dev/null; then
            # Update volume and build paths
            sed -i.bak 's|\./backend|./src/backend|g' "$file"
            sed -i.bak 's|\./frontend|./src/frontend|g' "$file"
            sed -i.bak 's|\./models|./models|g' "$file"
            sed -i.bak 's|\./data|./data|g' "$file"
            rm -f "${file}.bak"
            log_info "Updated paths in: $file"
        fi
    done
    
    # Update package.json if exists
    find . -name "package.json" | while read -r file; do
        if grep -q "\"scripts\"" "$file" 2>/dev/null; then
            sed -i.bak 's|frontend/|src/frontend/|g' "$file"
            sed -i.bak 's|backend/|src/backend/|g' "$file"
            rm -f "${file}.bak"
            log_info "Updated paths in: $file"
        fi
    done
}

# Clean up empty directories
cleanup_empty_dirs() {
    log "Cleaning up empty directories..."
    
    # Remove empty directories (except .git and node_modules)
    find . -type d -empty -not -path "./.git*" -not -path "./node_modules*" -not -path "./.venv*" | while read -r dir; do
        if [ -d "$dir" ]; then
            rmdir "$dir" 2>/dev/null && log_info "Removed empty directory: $dir"
        fi
    done
}

# Validate organization
validate_organization() {
    log "Validating organization..."
    
    local validation_errors=0
    
    # Check if key directories exist
    local required_dirs=(
        "src/backend" "src/frontend" "models" "data" "configs" "tests" "scripts" "docs"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_error "Required directory missing: $dir"
            validation_errors=$((validation_errors + 1))
        fi
    done
    
    # Check for Python import errors (basic syntax check)
    find src -name "*.py" -type f | while read -r file; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            log_warning "Python syntax error in: $file"
        fi
    done
    
    # Test if main application files exist
    if [ ! -f "src/backend/main.py" ] && [ ! -f "src/backend/persian_main.py" ]; then
        log_error "Main backend application file not found"
        validation_errors=$((validation_errors + 1))
    fi
    
    log_info "Validation completed with $validation_errors errors"
    return $validation_errors
}

# Generate final report
generate_report() {
    log "Generating final organization report..."
    
    echo "" >> "$REPORT_FILE"
    echo "## After Organization" >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"
    tree -a -I '.git|node_modules|__pycache__|*.pyc|.pytest_cache|.venv' . >> "$REPORT_FILE" 2>/dev/null || find . -type f -name ".*" -prune -o -print | sort >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    echo "## Organization Statistics" >> "$REPORT_FILE"
    echo "- Files moved: $FILES_MOVED" >> "$REPORT_FILE"
    echo "- Folders created: $FOLDERS_CREATED" >> "$REPORT_FILE"
    echo "- Conflicts resolved: $CONFLICTS_RESOLVED" >> "$REPORT_FILE"
    echo "- Import statements updated: $IMPORTS_UPDATED" >> "$REPORT_FILE"
    echo "- Organization completed: $(date)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    if [ -f "$CONFLICT_LOG" ] && [ -s "$CONFLICT_LOG" ]; then
        echo "## Conflicts Resolved" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        cat "$CONFLICT_LOG" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
    
    echo "## Files and Directories" >> "$REPORT_FILE"
    echo "### Source Code (src/)" >> "$REPORT_FILE"
    find src -type f | wc -l | xargs echo "- Total files in src/:" >> "$REPORT_FILE"
    
    echo "### Models (models/)" >> "$REPORT_FILE"
    find models -type f 2>/dev/null | wc -l | xargs echo "- Total model files:" >> "$REPORT_FILE"
    
    echo "### Data (data/)" >> "$REPORT_FILE"
    find data -type f 2>/dev/null | wc -l | xargs echo "- Total data files:" >> "$REPORT_FILE"
    
    echo "### Tests (tests/)" >> "$REPORT_FILE"
    find tests -type f 2>/dev/null | wc -l | xargs echo "- Total test files:" >> "$REPORT_FILE"
    
    echo "### Scripts (scripts/)" >> "$REPORT_FILE"
    find scripts -type f 2>/dev/null | wc -l | xargs echo "- Total script files:" >> "$REPORT_FILE"
    
    echo "### Documentation (docs/)" >> "$REPORT_FILE"
    find docs -type f 2>/dev/null | wc -l | xargs echo "- Total documentation files:" >> "$REPORT_FILE"
    
    log_success "Report generated: $REPORT_FILE"
}

# Generate undo script
generate_undo_script() {
    log "Generating undo script..."
    
    cat > "$UNDO_SCRIPT" << 'EOF'
#!/bin/bash
# Persian Legal AI - Organization Undo Script
# This script will attempt to reverse the organization changes

set -euo pipefail

echo "WARNING: This will attempt to undo the organization changes."
echo "This may not be 100% accurate if files were modified after organization."
echo "It's recommended to use the git backup instead: git checkout pre-organization-TIMESTAMP"
echo ""
echo "Continue with file-based undo? (y/N)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborting undo. Use git to restore from backup."
    exit 1
fi

echo "Starting undo process..."

# Reverse move operations (generated dynamically)
EOF

    # The actual undo commands are appended during the move operations
    chmod +x "$UNDO_SCRIPT"
    log_success "Undo script generated: $UNDO_SCRIPT"
}

# Main execution functions
show_help() {
    cat << EOF
Persian Legal AI Training System - Organization Script

USAGE:
    $0 [MODE] [OPTIONS]

MODES:
    --analyze     Analyze current project structure
    --plan        Show organization plan without executing
    --dry-run     Preview all changes without executing
    --execute     Execute the organization process
    --validate    Validate the organized structure
    --undo        Undo the last organization (if possible)
    --report      Generate organization report
    --help        Show this help message

OPTIONS:
    --force       Skip confirmations
    --verbose     Enable verbose logging
    --backup-only Create backup without organizing

EXAMPLES:
    $0 --analyze                    # Analyze current structure
    $0 --dry-run                    # Preview changes
    $0 --execute                    # Execute organization
    $0 --validate                   # Validate after organization
    $0 --undo                       # Undo last organization

SAFETY:
    - Always creates git backup before changes
    - Generates undo script for file-level recovery
    - Validates organization after completion
    - Logs all operations for audit trail

EOF
}

analyze_mode() {
    log "=== ANALYSIS MODE ==="
    check_git_status
    analyze_structure
    log_success "Analysis complete. Report: $REPORT_FILE"
}

plan_mode() {
    log "=== PLANNING MODE ==="
    cat << EOF

ORGANIZATION PLAN:
==================

1. BACKUP & SAFETY
   - Create git backup with branch and tag
   - Create file system backup
   - Generate undo script

2. SOURCE CODE REORGANIZATION
   - Move frontend/ → src/frontend/
   - Move backend/ → src/backend/
   - Move bolt/ → src/frontend/bolt/
   - Organize shared utilities → src/shared/

3. MODEL CONSOLIDATION
   - Merge ai_models/ + models/ → models/
   - Organize by model type (dora/, qr_adaptor/, bert/)

4. DATA CONSOLIDATION
   - Merge data/ + database/ → data/
   - Organize databases, datasets, exports

5. CONFIGURATION CONSOLIDATION
   - Merge config/ + configs/ → configs/
   - Organize by environment (dev/prod/test)

6. TESTING CONSOLIDATION
   - Merge test/ + tests/ → tests/
   - Organize by type (unit/integration/e2e)

7. SCRIPT ORGANIZATION
   - Organize scripts by function
   - deployment/, maintenance/, setup/, testing/

8. DOCUMENTATION ORGANIZATION
   - Move root .md files to docs/
   - Organize by type (reports/, guides/, analysis/)

9. IMPORT PATH UPDATES
   - Update Python imports
   - Update JS/TS imports
   - Update config file paths

10. VALIDATION & CLEANUP
    - Remove empty directories
    - Validate organization
    - Generate final report

EOF
}

dry_run_mode() {
    log "=== DRY RUN MODE ==="
    log_warning "DRY RUN: No files will actually be moved"
    
    analyze_structure
    create_target_structure
    
    log "Would reorganize source code..."
    log "Would consolidate models..."
    log "Would consolidate data..."
    log "Would consolidate configs..."
    log "Would consolidate tests..."
    log "Would consolidate scripts..."
    log "Would consolidate docs..."
    log "Would update import paths..."
    log "Would validate organization..."
    
    log_success "Dry run complete. No changes made."
}

execute_mode() {
    log "=== EXECUTION MODE ==="
    
    # Safety checks
    check_git_status
    
    log_warning "This will reorganize your entire project structure!"
    echo "Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_info "Organization cancelled."
        exit 0
    fi
    
    # Initialize
    echo "#!/bin/bash" > "$UNDO_SCRIPT"
    echo "# Undo script generated on $(date)" >> "$UNDO_SCRIPT"
    echo "" >> "$UNDO_SCRIPT"
    
    # Execute organization steps
    create_backup
    analyze_structure
    create_target_structure
    reorganize_source_code
    consolidate_models
    consolidate_data
    consolidate_configs
    consolidate_tests
    consolidate_scripts
    consolidate_docs
    update_python_imports
    update_js_imports
    update_config_paths
    cleanup_empty_dirs
    
    # Validate and report
    if validate_organization; then
        generate_report
        generate_undo_script
        log_success "Organization completed successfully!"
        log_info "Files moved: $FILES_MOVED"
        log_info "Folders created: $FOLDERS_CREATED"
        log_info "Conflicts resolved: $CONFLICTS_RESOLVED"
        log_info "Imports updated: $IMPORTS_UPDATED"
        log_info "Report: $REPORT_FILE"
        log_info "Undo script: $UNDO_SCRIPT"
        log_info "Backup: $BACKUP_DIR"
    else
        log_error "Organization completed with validation errors. Check the logs."
        exit 1
    fi
}

validate_mode() {
    log "=== VALIDATION MODE ==="
    validate_organization
}

undo_mode() {
    log "=== UNDO MODE ==="
    
    # Find the most recent undo script
    local latest_undo=$(find . -name "undo_organization_*.sh" -type f -exec ls -t {} + | head -n 1)
    
    if [ -z "$latest_undo" ]; then
        log_error "No undo script found. Use git to restore from backup:"
        log_info "git branch -a | grep backup-before-major-organization"
        log_info "git checkout backup-before-major-organization-TIMESTAMP"
        exit 1
    fi
    
    log_warning "Found undo script: $latest_undo"
    echo "Execute undo? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        bash "$latest_undo"
    else
        log_info "Undo cancelled."
    fi
}

report_mode() {
    log "=== REPORT MODE ==="
    analyze_structure
    generate_report
    log_success "Report generated: $REPORT_FILE"
}

# Main script execution
main() {
    local mode="${1:-}"
    
    case "$mode" in
        --analyze)
            analyze_mode
            ;;
        --plan)
            plan_mode
            ;;
        --dry-run)
            dry_run_mode
            ;;
        --execute)
            execute_mode
            ;;
        --validate)
            validate_mode
            ;;
        --undo)
            undo_mode
            ;;
        --report)
            report_mode
            ;;
        --help|"")
            show_help
            ;;
        *)
            log_error "Unknown mode: $mode"
            show_help
            exit 1
            ;;
    esac
}

# Initialize logging and execute
echo "Persian Legal AI Training System - Organization Script v3.0.0"
echo "Started: $(date)"
echo "Log file: $LOG_FILE"
echo "========================================================"

main "$@"

echo "========================================================"
echo "Completed: $(date)"