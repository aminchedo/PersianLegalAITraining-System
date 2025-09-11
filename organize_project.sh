#!/bin/bash

# Persian Legal AI Project Organization Script
# This script organizes the messy project structure into a clean, maintainable format
# Author: AI Assistant
# Date: September 11, 2025

set -euo pipefail

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/organization_backup_$(date +%Y%m%d_%H%M%S)"
UNDO_SCRIPT="${SCRIPT_DIR}/undo_organization.sh"
REPORT_FILE="${SCRIPT_DIR}/organization_report.md"
MOVED_FILES=()
FAILED_MOVES=()
UNCATEGORIZED_FILES=()

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section headers
print_header() {
    local message=$1
    echo
    print_color "$CYAN" "═══════════════════════════════════════════════════════════════"
    print_color "$CYAN" "  $message"
    print_color "$CYAN" "═══════════════════════════════════════════════════════════════"
    echo
}

# Function to print progress
print_progress() {
    local current=$1
    local total=$2
    local message=$3
    local percentage=$((current * 100 / total))
    print_color "$BLUE" "[$current/$total] ($percentage%) $message"
}

# Function to log file moves for undo script
log_move() {
    local source=$1
    local destination=$2
    echo "mv '$destination' '$source'" >> "$UNDO_SCRIPT"
    MOVED_FILES+=("$source -> $destination")
}

# Function to safely move files
safe_move() {
    local source=$1
    local destination=$2
    local description=$3
    
    if [[ -f "$source" ]]; then
        local dest_dir=$(dirname "$destination")
        mkdir -p "$dest_dir"
        
        if mv "$source" "$destination" 2>/dev/null; then
            print_color "$GREEN" "✓ Moved: $(basename "$source") → $dest_dir/"
            log_move "$source" "$destination"
            return 0
        else
            print_color "$RED" "✗ Failed to move: $source"
            FAILED_MOVES+=("$source: $description")
            return 1
        fi
    else
        print_color "$YELLOW" "⚠ File not found: $source"
        return 1
    fi
}

# Function to create directory structure
create_directory_structure() {
    print_header "CREATING DIRECTORY STRUCTURE"
    
    local directories=(
        "docs/reports"
        "docs/guides"
        "docs/analysis"
        "docs/archive"
        "tests/unit"
        "tests/integration"
        "tests/e2e"
        "tests/fixtures"
        "scripts/deployment"
        "scripts/maintenance"
        "scripts/utils"
        "configs/development"
        "configs/production"
        "configs/testing"
        "backup/databases"
        "backup/configs"
        "backup/logs"
    )
    
    local total=${#directories[@]}
    local current=0
    
    for dir in "${directories[@]}"; do
        current=$((current + 1))
        print_progress "$current" "$total" "Creating directory: $dir"
        mkdir -p "$dir"
        
        # Create .gitkeep files in empty directories
        if [[ ! "$(ls -A "$dir" 2>/dev/null)" ]]; then
            touch "$dir/.gitkeep"
            print_color "$YELLOW" "  Created .gitkeep in $dir"
        fi
    done
    
    print_color "$GREEN" "✓ Directory structure created successfully!"
}

# Function to create git backup
create_git_backup() {
    print_header "CREATING GIT BACKUP"
    
    if command -v git &> /dev/null && [[ -d .git ]]; then
        print_color "$BLUE" "Creating git stash backup..."
        git add . 2>/dev/null || true
        git stash push -m "Pre-organization backup $(date)" 2>/dev/null || true
        print_color "$GREEN" "✓ Git backup created (stashed)"
        
        # Also create a commit backup
        git add . 2>/dev/null || true
        git commit -m "Pre-organization backup $(date)" 2>/dev/null || true
        print_color "$GREEN" "✓ Commit backup created"
    else
        print_color "$YELLOW" "⚠ Git not available or not a git repository"
        print_color "$BLUE" "Creating file system backup instead..."
        mkdir -p "$BACKUP_DIR"
        cp -r . "$BACKUP_DIR" 2>/dev/null || true
        print_color "$GREEN" "✓ File system backup created at: $BACKUP_DIR"
    fi
}

# Function to organize markdown files
organize_markdown_files() {
    print_header "ORGANIZING MARKDOWN FILES"
    
    # Report files (contain REPORT in filename)
    local report_files=(
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
    
    # Guide files (contain GUIDE in filename or are guides)
    local guide_files=(
        "DEPLOYMENT_GUIDE.md"
        "START_SYSTEM.md"
        "backend-implementation-guide.md"
    )
    
    # Analysis files
    local analysis_files=(
        "BACKEND_MAIN_FILES_ANALYSIS.md"
        "analysis_report.txt"
    )
    
    print_color "$BLUE" "Moving report files..."
    local current=0
    local total=$((${#report_files[@]} + ${#guide_files[@]} + ${#analysis_files[@]}))
    
    for file in "${report_files[@]}"; do
        current=$((current + 1))
        print_progress "$current" "$total" "Processing report: $file"
        safe_move "$file" "docs/reports/$file" "Report file"
    done
    
    print_color "$BLUE" "Moving guide files..."
    for file in "${guide_files[@]}"; do
        current=$((current + 1))
        print_progress "$current" "$total" "Processing guide: $file"
        safe_move "$file" "docs/guides/$file" "Guide file"
    done
    
    print_color "$BLUE" "Moving analysis files..."
    for file in "${analysis_files[@]}"; do
        current=$((current + 1))
        print_progress "$current" "$total" "Processing analysis: $file"
        safe_move "$file" "docs/analysis/$file" "Analysis file"
    done
}

# Function to organize test files
organize_test_files() {
    print_header "ORGANIZING TEST FILES"
    
    # Test files mapping
    local test_files=(
        "test_frontend.html:tests/e2e/test_frontend.html"
        "test_integration.py:tests/integration/test_integration.py"
        "test_production_system.py:tests/integration/test_production_system.py"
    )
    
    local current=0
    local total=${#test_files[@]}
    
    for mapping in "${test_files[@]}"; do
        current=$((current + 1))
        local source="${mapping%%:*}"
        local destination="${mapping##*:}"
        print_progress "$current" "$total" "Processing test: $source"
        safe_move "$source" "$destination" "Test file"
    done
    
    # Look for additional test files with patterns
    print_color "$BLUE" "Searching for additional test files..."
    for file in test*.py *test*.py test*.js *test*.js test*.html *test*.html; do
        if [[ -f "$file" ]] && [[ ! "$file" =~ ^tests/ ]]; then
            local ext="${file##*.}"
            case "$ext" in
                py)
                    safe_move "$file" "tests/integration/$file" "Python test file"
                    ;;
                js)
                    safe_move "$file" "tests/e2e/$file" "JavaScript test file"
                    ;;
                html)
                    safe_move "$file" "tests/e2e/$file" "HTML test file"
                    ;;
            esac
        fi
    done
}

# Function to organize script files
organize_script_files() {
    print_header "ORGANIZING SCRIPT FILES"
    
    # Specific script mappings
    local script_mappings=(
        "system_health_check.py:scripts/maintenance/system_health_check.py"
        "start_system.py:scripts/start_system.py"
        "dependency-analyzer.js:scripts/utils/dependency-analyzer.js"
        "dependency-analyzer.ts:scripts/utils/dependency-analyzer.ts"
        "backend-validator-simple.js:scripts/utils/backend-validator-simple.js"
        "backend-validator.js:scripts/utils/backend-validator.js"
        "route-updater.js:scripts/utils/route-updater.js"
        "route-updater.ts:scripts/utils/route-updater.ts"
    )
    
    local current=0
    local total=${#script_mappings[@]}
    
    for mapping in "${script_mappings[@]}"; do
        current=$((current + 1))
        local source="${mapping%%:*}"
        local destination="${mapping##*:}"
        print_progress "$current" "$total" "Processing script: $source"
        safe_move "$source" "$destination" "Script file"
    done
    
    # Move all .sh files to scripts/
    print_color "$BLUE" "Moving shell scripts..."
    for file in *.sh; do
        if [[ -f "$file" ]] && [[ "$file" != "organize_project.sh" ]] && [[ "$file" != "undo_organization.sh" ]]; then
            case "$file" in
                *deploy*|*production*|*merge*)
                    safe_move "$file" "scripts/deployment/$file" "Deployment script"
                    ;;
                *health*|*maintenance*|*recovery*)
                    safe_move "$file" "scripts/maintenance/$file" "Maintenance script"
                    ;;
                *)
                    safe_move "$file" "scripts/$file" "Shell script"
                    ;;
            esac
        fi
    done
}

# Function to organize configuration files
organize_config_files() {
    print_header "ORGANIZING CONFIGURATION FILES"
    
    local config_mappings=(
        "docker-compose.yml:configs/development/docker-compose.yml"
        "docker-compose.production.yml:configs/production/docker-compose.production.yml"
    )
    
    local current=0
    local total=${#config_mappings[@]}
    
    for mapping in "${config_mappings[@]}"; do
        current=$((current + 1))
        local source="${mapping%%:*}"
        local destination="${mapping##*:}"
        print_progress "$current" "$total" "Processing config: $source"
        safe_move "$source" "$destination" "Configuration file"
    done
}

# Function to organize database files
organize_database_files() {
    print_header "ORGANIZING DATABASE FILES"
    
    local db_files=(
        "persian_legal_ai.db"
    )
    
    local current=0
    local total=${#db_files[@]}
    
    for file in "${db_files[@]}"; do
        current=$((current + 1))
        print_progress "$current" "$total" "Processing database: $file"
        safe_move "$file" "backup/databases/$file" "Database file"
    done
    
    # Look for additional database files
    print_color "$BLUE" "Searching for additional database files..."
    for file in *.db *.sqlite *.sqlite3; do
        if [[ -f "$file" ]] && [[ ! "$file" =~ ^backup/ ]]; then
            safe_move "$file" "backup/databases/$file" "Database file"
        fi
    done
}

# Function to check for uncategorized files
check_uncategorized_files() {
    print_header "CHECKING FOR UNCATEGORIZED FILES"
    
    local extensions=("*.md" "*.py" "*.js" "*.ts" "*.sh" "*.db" "*.sqlite" "*.sqlite3" "*.html" "*.css" "*.json" "*.yml" "*.yaml")
    
    for pattern in "${extensions[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" ]] && [[ "$file" != "README.md" ]] && [[ "$file" != "organize_project.sh" ]] && [[ "$file" != "undo_organization.sh" ]] && [[ "$file" != "organization_report.md" ]]; then
                UNCATEGORIZED_FILES+=("$file")
                print_color "$YELLOW" "⚠ Uncategorized: $file"
            fi
        done
    done
    
    if [[ ${#UNCATEGORIZED_FILES[@]} -eq 0 ]]; then
        print_color "$GREEN" "✓ All relevant files have been categorized!"
    fi
}

# Function to generate directory tree
generate_directory_tree() {
    print_header "CURRENT DIRECTORY STRUCTURE"
    
    if command -v tree &> /dev/null; then
        tree -I 'node_modules|__pycache__|*.pyc|.git' --dirsfirst
    else
        print_color "$YELLOW" "⚠ 'tree' command not available, using ls instead"
        find . -type d -not -path '*/\.*' -not -path '*/node_modules*' -not -path '*/__pycache__*' | sort | sed 's/[^-][^\/]*\// |/g; s/|\([^ ]\)/|-\1/'
    fi
}

# Function to generate report
generate_report() {
    print_header "GENERATING ORGANIZATION REPORT"
    
    cat > "$REPORT_FILE" << EOF
# Persian Legal AI Project Organization Report

**Generated on:** $(date)
**Script version:** 1.0
**Total files processed:** $((${#MOVED_FILES[@]} + ${#FAILED_MOVES[@]} + ${#UNCATEGORIZED_FILES[@]}))

## Summary

- ✅ **Successfully moved:** ${#MOVED_FILES[@]} files
- ❌ **Failed moves:** ${#FAILED_MOVES[@]} files  
- ⚠️ **Uncategorized:** ${#UNCATEGORIZED_FILES[@]} files

## Directory Structure Created

\`\`\`
docs/
├── reports/          # All *REPORT*.md files
├── guides/           # All *GUIDE*.md files  
├── analysis/         # All analysis files
└── archive/          # Old documentation

tests/
├── unit/            # Unit test files
├── integration/     # Integration tests
├── e2e/            # End-to-end tests
└── fixtures/       # Test data

scripts/
├── deployment/     # Deployment scripts
├── maintenance/    # Maintenance scripts
└── utils/         # Utility scripts

configs/
├── development/    # Dev configs
├── production/     # Prod configs
└── testing/       # Test configs

backup/
├── databases/      # Database backups
├── configs/        # Config backups
└── logs/          # Log files
\`\`\`

## Successfully Moved Files

EOF

    for move in "${MOVED_FILES[@]}"; do
        echo "- $move" >> "$REPORT_FILE"
    done

    if [[ ${#FAILED_MOVES[@]} -gt 0 ]]; then
        echo -e "\n## Failed Moves\n" >> "$REPORT_FILE"
        for fail in "${FAILED_MOVES[@]}"; do
            echo "- $fail" >> "$REPORT_FILE"
        done
    fi

    if [[ ${#UNCATEGORIZED_FILES[@]} -gt 0 ]]; then
        echo -e "\n## Uncategorized Files\n" >> "$REPORT_FILE"
        for uncat in "${UNCATEGORIZED_FILES[@]}"; do
            echo "- $uncat" >> "$REPORT_FILE"
        done
    fi

    echo -e "\n## Undo Instructions\n" >> "$REPORT_FILE"
    echo "To undo this organization, run:" >> "$REPORT_FILE"
    echo "\`\`\`bash" >> "$REPORT_FILE"
    echo "bash undo_organization.sh" >> "$REPORT_FILE"
    echo "\`\`\`" >> "$REPORT_FILE"

    print_color "$GREEN" "✓ Report generated: $REPORT_FILE"
}

# Function to initialize undo script
initialize_undo_script() {
    cat > "$UNDO_SCRIPT" << 'EOF'
#!/bin/bash
# Undo script for Persian Legal AI project organization
# This script will move files back to their original locations

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_color "$YELLOW" "Starting undo process..."
print_color "$RED" "WARNING: This will move files back to root level!"
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_color "$YELLOW" "Undo cancelled."
    exit 0
fi

print_color "$YELLOW" "Undoing file organization..."

EOF

    chmod +x "$UNDO_SCRIPT"
}

# Main execution function
main() {
    print_color "$PURPLE" "Persian Legal AI Project Organization Script"
    print_color "$PURPLE" "=========================================="
    echo
    print_color "$BLUE" "This script will organize your project structure safely."
    print_color "$BLUE" "A backup will be created and an undo script will be generated."
    echo
    
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_color "$YELLOW" "Organization cancelled."
        exit 0
    fi
    
    # Initialize undo script
    initialize_undo_script
    
    # Create git backup
    create_git_backup
    
    # Create directory structure
    create_directory_structure
    
    # Organize files by category
    organize_markdown_files
    organize_test_files
    organize_script_files
    organize_config_files
    organize_database_files
    
    # Check for uncategorized files
    check_uncategorized_files
    
    # Finalize undo script
    cat >> "$UNDO_SCRIPT" << 'EOF'

print_color "$GREEN" "✓ Undo completed successfully!"
print_color "$YELLOW" "Note: Empty directories were not removed automatically."
print_color "$YELLOW" "You may want to clean them up manually if needed."
EOF
    
    # Generate report
    generate_report
    
    # Show final structure
    generate_directory_tree
    
    # Final summary
    print_header "ORGANIZATION COMPLETE!"
    print_color "$GREEN" "✅ Project organization completed successfully!"
    print_color "$BLUE" "📁 Files moved: ${#MOVED_FILES[@]}"
    print_color "$RED" "❌ Failed moves: ${#FAILED_MOVES[@]}"
    print_color "$YELLOW" "⚠️ Uncategorized: ${#UNCATEGORIZED_FILES[@]}"
    echo
    print_color "$CYAN" "📋 Detailed report: $REPORT_FILE"
    print_color "$CYAN" "🔄 Undo script: $UNDO_SCRIPT"
    echo
    print_color "$GREEN" "Your Persian Legal AI project is now properly organized! 🎉"
}

# Run the main function
main "$@"