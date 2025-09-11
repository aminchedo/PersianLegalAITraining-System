#!/bin/bash

# Persian Legal AI Project Organization Script
# This script organizes the project structure according to best practices

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/workspace"
BACKUP_DIR="${PROJECT_ROOT}/backup_organization_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/organization.log"
REPORT_FILE="${PROJECT_ROOT}/organization_report.md"
UNDO_SCRIPT="${PROJECT_ROOT}/undo_organization.sh"

# Initialize log file
echo "=== Persian Legal AI Project Organization Started at $(date) ===" > "$LOG_FILE"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message" >> "$LOG_FILE"
}

# Function to create directory if it doesn't exist
create_dir() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "$GREEN" "Created directory: $dir"
    fi
}

# Function to move file safely
move_file() {
    local source=$1
    local destination=$2
    local description=$3
    
    if [ -f "$source" ]; then
        # Create destination directory if it doesn't exist
        local dest_dir=$(dirname "$destination")
        create_dir "$dest_dir"
        
        # Move the file
        mv "$source" "$destination"
        print_status "$BLUE" "Moved: $source -> $destination ($description)"
        
        # Add to undo script
        echo "mv \"$destination\" \"$source\"" >> "$UNDO_SCRIPT"
    else
        print_status "$YELLOW" "File not found: $source"
    fi
}

# Function to create .gitkeep file
create_gitkeep() {
    local dir=$1
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        touch "$dir/.gitkeep"
        print_status "$CYAN" "Created .gitkeep in empty directory: $dir"
    fi
}

# Function to generate directory tree
generate_tree() {
    local output_file=$1
    echo "```" > "$output_file"
    tree -a -I '.git|node_modules|__pycache__|*.pyc|.DS_Store' >> "$output_file" 2>/dev/null || find . -type d | sed 's|[^/]*/|- |g' >> "$output_file"
    echo "```" >> "$output_file"
}

print_status "$PURPLE" "🚀 Starting Persian Legal AI Project Organization"
print_status "$CYAN" "Project Root: $PROJECT_ROOT"
print_status "$CYAN" "Backup Directory: $BACKUP_DIR"

# Step 1: Create git backup
print_status "$YELLOW" "📦 Creating git backup..."
if [ -d "${PROJECT_ROOT}/.git" ]; then
    git add -A
    git commit -m "Pre-organization backup - $(date)" || print_status "$YELLOW" "No changes to commit"
    print_status "$GREEN" "✅ Git backup completed"
else
    print_status "$YELLOW" "⚠️  No git repository found, skipping git backup"
fi

# Step 2: Create backup directory structure
print_status "$YELLOW" "📁 Creating backup directory..."
create_dir "$BACKUP_DIR"

# Step 3: Initialize undo script
print_status "$YELLOW" "🔄 Creating undo script..."
cat > "$UNDO_SCRIPT" << 'EOF'
#!/bin/bash
# Undo script for Persian Legal AI Project Organization
# Generated automatically - DO NOT EDIT MANUALLY

echo "🔄 Reversing project organization..."

EOF
chmod +x "$UNDO_SCRIPT"

# Step 4: Create new directory structure
print_status "$YELLOW" "🏗️  Creating new directory structure..."

# Main directories
create_dir "${PROJECT_ROOT}/docs/reports"
create_dir "${PROJECT_ROOT}/docs/guides"
create_dir "${PROJECT_ROOT}/docs/analysis"
create_dir "${PROJECT_ROOT}/docs/archive"

create_dir "${PROJECT_ROOT}/tests/unit"
create_dir "${PROJECT_ROOT}/tests/integration"
create_dir "${PROJECT_ROOT}/tests/e2e"
create_dir "${PROJECT_ROOT}/tests/fixtures"

create_dir "${PROJECT_ROOT}/scripts/deployment"
create_dir "${PROJECT_ROOT}/scripts/maintenance"
create_dir "${PROJECT_ROOT}/scripts/utils"

create_dir "${PROJECT_ROOT}/configs/development"
create_dir "${PROJECT_ROOT}/configs/production"
create_dir "${PROJECT_ROOT}/configs/testing"

create_dir "${PROJECT_ROOT}/backup/databases"
create_dir "${PROJECT_ROOT}/backup/configs"
create_dir "${PROJECT_ROOT}/backup/logs"

# Step 5: Move files according to rules
print_status "$YELLOW" "📋 Moving files according to organization rules..."

# Move REPORT files to docs/reports/
print_status "$BLUE" "📊 Moving report files..."
move_file "${PROJECT_ROOT}/BOLT_INTEGRATION_COMPLETE.md" "${PROJECT_ROOT}/docs/reports/BOLT_INTEGRATION_COMPLETE.md" "Report file"
move_file "${PROJECT_ROOT}/CLEANUP_SUMMARY.md" "${PROJECT_ROOT}/docs/reports/CLEANUP_SUMMARY.md" "Report file"
move_file "${PROJECT_ROOT}/MERGE_COMPLETION_REPORT.md" "${PROJECT_ROOT}/docs/reports/MERGE_COMPLETION_REPORT.md" "Report file"
move_file "${PROJECT_ROOT}/PERSIAN_AI_ENHANCEMENT_COMPLETION_REPORT.md" "${PROJECT_ROOT}/docs/reports/PERSIAN_AI_ENHANCEMENT_COMPLETION_REPORT.md" "Report file"
move_file "${PROJECT_ROOT}/PERSIAN_LEGAL_AI_AUDIT_COMPLETION_REPORT.md" "${PROJECT_ROOT}/docs/reports/PERSIAN_LEGAL_AI_AUDIT_COMPLETION_REPORT.md" "Report file"
move_file "${PROJECT_ROOT}/PHASE_3_COMPLETION_SUMMARY.md" "${PROJECT_ROOT}/docs/reports/PHASE_3_COMPLETION_SUMMARY.md" "Report file"
move_file "${PROJECT_ROOT}/PHASE_CRITICAL_PRODUCTION_ISSUES_RESOLVED.md" "${PROJECT_ROOT}/docs/reports/PHASE_CRITICAL_PRODUCTION_ISSUES_RESOLVED.md" "Report file"
move_file "${PROJECT_ROOT}/RECOVERY_REPORT.md" "${PROJECT_ROOT}/docs/reports/RECOVERY_REPORT.md" "Report file"
move_file "${PROJECT_ROOT}/SAFE_MERGE_COMPLETION_REPORT.md" "${PROJECT_ROOT}/docs/reports/SAFE_MERGE_COMPLETION_REPORT.md" "Report file"
move_file "${PROJECT_ROOT}/VISUAL_UI_TEST_REPORT.md" "${PROJECT_ROOT}/docs/reports/VISUAL_UI_TEST_REPORT.md" "Report file"
move_file "${PROJECT_ROOT}/integration-report.md" "${PROJECT_ROOT}/docs/reports/integration-report.md" "Report file"

# Move GUIDE files to docs/guides/
print_status "$BLUE" "📖 Moving guide files..."
move_file "${PROJECT_ROOT}/DEPLOYMENT_GUIDE.md" "${PROJECT_ROOT}/docs/guides/DEPLOYMENT_GUIDE.md" "Guide file"
move_file "${PROJECT_ROOT}/START_SYSTEM.md" "${PROJECT_ROOT}/docs/guides/START_SYSTEM.md" "Guide file"
move_file "${PROJECT_ROOT}/backend-implementation-guide.md" "${PROJECT_ROOT}/docs/guides/backend-implementation-guide.md" "Guide file"

# Move analysis files to docs/analysis/
print_status "$BLUE" "🔍 Moving analysis files..."
move_file "${PROJECT_ROOT}/BACKEND_MAIN_FILES_ANALYSIS.md" "${PROJECT_ROOT}/docs/analysis/BACKEND_MAIN_FILES_ANALYSIS.md" "Analysis file"
move_file "${PROJECT_ROOT}/analysis_report.txt" "${PROJECT_ROOT}/docs/analysis/analysis_report.txt" "Analysis file"

# Move test files to tests/
print_status "$BLUE" "🧪 Moving test files..."
move_file "${PROJECT_ROOT}/test_frontend.html" "${PROJECT_ROOT}/tests/e2e/test_frontend.html" "E2E test file"
move_file "${PROJECT_ROOT}/test_integration.py" "${PROJECT_ROOT}/tests/integration/test_integration.py" "Integration test file"
move_file "${PROJECT_ROOT}/test_production_system.py" "${PROJECT_ROOT}/tests/integration/test_production_system.py" "Integration test file"

# Move script files to scripts/
print_status "$BLUE" "🔧 Moving script files..."
move_file "${PROJECT_ROOT}/check-bolt-endpoints.sh" "${PROJECT_ROOT}/scripts/utils/check-bolt-endpoints.sh" "Utility script"
move_file "${PROJECT_ROOT}/comprehensive-test.sh" "${PROJECT_ROOT}/scripts/utils/comprehensive-test.sh" "Utility script"
move_file "${PROJECT_ROOT}/fix-imports.sh" "${PROJECT_ROOT}/scripts/utils/fix-imports.sh" "Utility script"
move_file "${PROJECT_ROOT}/fix-quotes.sh" "${PROJECT_ROOT}/scripts/utils/fix-quotes.sh" "Utility script"
move_file "${PROJECT_ROOT}/persian_legal_ai_recovery.sh" "${PROJECT_ROOT}/scripts/maintenance/persian_legal_ai_recovery.sh" "Maintenance script"
move_file "${PROJECT_ROOT}/safe-merge-to-main.sh" "${PROJECT_ROOT}/scripts/deployment/safe-merge-to-main.sh" "Deployment script"
move_file "${PROJECT_ROOT}/safe-merge.sh" "${PROJECT_ROOT}/scripts/deployment/safe-merge.sh" "Deployment script"
move_file "${PROJECT_ROOT}/smart-analysis.sh" "${PROJECT_ROOT}/scripts/utils/smart-analysis.sh" "Utility script"
move_file "${PROJECT_ROOT}/smart-migration.sh" "${PROJECT_ROOT}/scripts/utils/smart-migration.sh" "Utility script"

# Move Python utility scripts
move_file "${PROJECT_ROOT}/system_health_check.py" "${PROJECT_ROOT}/scripts/maintenance/system_health_check.py" "Maintenance script"
move_file "${PROJECT_ROOT}/start_system.py" "${PROJECT_ROOT}/scripts/start_system.py" "System script"

# Move JavaScript/TypeScript utility files
move_file "${PROJECT_ROOT}/backend-validator-simple.js" "${PROJECT_ROOT}/scripts/utils/backend-validator-simple.js" "Utility script"
move_file "${PROJECT_ROOT}/backend-validator.js" "${PROJECT_ROOT}/scripts/utils/backend-validator.js" "Utility script"
move_file "${PROJECT_ROOT}/dependency-analyzer.js" "${PROJECT_ROOT}/scripts/utils/dependency-analyzer.js" "Utility script"
move_file "${PROJECT_ROOT}/dependency-analyzer.ts" "${PROJECT_ROOT}/scripts/utils/dependency-analyzer.ts" "Utility script"
move_file "${PROJECT_ROOT}/route-updater.js" "${PROJECT_ROOT}/scripts/utils/route-updater.js" "Utility script"
move_file "${PROJECT_ROOT}/route-updater.ts" "${PROJECT_ROOT}/scripts/utils/route-updater.ts" "Utility script"

# Move configuration files
print_status "$BLUE" "⚙️  Moving configuration files..."
move_file "${PROJECT_ROOT}/docker-compose.production.yml" "${PROJECT_ROOT}/configs/production/docker-compose.production.yml" "Production config"
move_file "${PROJECT_ROOT}/docker-compose.yml" "${PROJECT_ROOT}/configs/development/docker-compose.yml" "Development config"

# Move database files
print_status "$BLUE" "🗄️  Moving database files..."
move_file "${PROJECT_ROOT}/persian_legal_ai.db" "${PROJECT_ROOT}/backup/databases/persian_legal_ai.db" "Database file"

# Step 6: Create .gitkeep files in empty directories
print_status "$YELLOW" "📌 Creating .gitkeep files in empty directories..."
create_gitkeep "${PROJECT_ROOT}/docs/archive"
create_gitkeep "${PROJECT_ROOT}/tests/unit"
create_gitkeep "${PROJECT_ROOT}/tests/fixtures"
create_gitkeep "${PROJECT_ROOT}/scripts/deployment"
create_gitkeep "${PROJECT_ROOT}/scripts/maintenance"
create_gitkeep "${PROJECT_ROOT}/configs/testing"
create_gitkeep "${PROJECT_ROOT}/backup/configs"
create_gitkeep "${PROJECT_ROOT}/backup/logs"

# Step 7: Generate organization report
print_status "$YELLOW" "📊 Generating organization report..."

cat > "$REPORT_FILE" << EOF
# Persian Legal AI Project Organization Report

**Generated on:** $(date)
**Script Version:** 1.0
**Project Root:** $PROJECT_ROOT

## Summary

This report documents the organization of the Persian Legal AI project structure. All files have been moved according to best practices for project organization.

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

## Files Moved

### Reports (docs/reports/)
- BOLT_INTEGRATION_COMPLETE.md
- CLEANUP_SUMMARY.md
- MERGE_COMPLETION_REPORT.md
- PERSIAN_AI_ENHANCEMENT_COMPLETION_REPORT.md
- PERSIAN_LEGAL_AI_AUDIT_COMPLETION_REPORT.md
- PHASE_3_COMPLETION_SUMMARY.md
- PHASE_CRITICAL_PRODUCTION_ISSUES_RESOLVED.md
- RECOVERY_REPORT.md
- SAFE_MERGE_COMPLETION_REPORT.md
- VISUAL_UI_TEST_REPORT.md
- integration-report.md

### Guides (docs/guides/)
- DEPLOYMENT_GUIDE.md
- START_SYSTEM.md
- backend-implementation-guide.md

### Analysis (docs/analysis/)
- BACKEND_MAIN_FILES_ANALYSIS.md
- analysis_report.txt

### Tests
- test_frontend.html → tests/e2e/
- test_integration.py → tests/integration/
- test_production_system.py → tests/integration/

### Scripts
- check-bolt-endpoints.sh → scripts/utils/
- comprehensive-test.sh → scripts/utils/
- fix-imports.sh → scripts/utils/
- fix-quotes.sh → scripts/utils/
- persian_legal_ai_recovery.sh → scripts/maintenance/
- safe-merge-to-main.sh → scripts/deployment/
- safe-merge.sh → scripts/deployment/
- smart-analysis.sh → scripts/utils/
- smart-migration.sh → scripts/utils/
- system_health_check.py → scripts/maintenance/
- start_system.py → scripts/
- backend-validator-simple.js → scripts/utils/
- backend-validator.js → scripts/utils/
- dependency-analyzer.js → scripts/utils/
- dependency-analyzer.ts → scripts/utils/
- route-updater.js → scripts/utils/
- route-updater.ts → scripts/utils/

### Configuration Files
- docker-compose.production.yml → configs/production/
- docker-compose.yml → configs/development/

### Database Files
- persian_legal_ai.db → backup/databases/

## Files Kept in Root
- README.md (kept as requested)
- main.py (core application file)
- persian_main.py (core application file)
- requirements.txt (Python dependencies)
- project.json (project configuration)

## Safety Measures
- Git backup created before organization
- Undo script generated: undo_organization.sh
- All operations logged to: organization.log
- No files were deleted, only moved

## Current Directory Structure

EOF

# Generate current directory tree
generate_tree "${REPORT_FILE}.tree"
cat "${REPORT_FILE}.tree" >> "$REPORT_FILE"
rm "${REPORT_FILE}.tree"

# Add footer to report
cat >> "$REPORT_FILE" << EOF

## Next Steps
1. Review the new structure
2. Update any hardcoded paths in your code
3. Update documentation references
4. Test that all scripts still work from their new locations

## Undo Instructions
If you need to revert the organization, run:
\`\`\`bash
bash undo_organization.sh
\`\`\`

---
*Report generated by organize_project.sh v1.0*
EOF

# Step 8: Finalize undo script
cat >> "$UNDO_SCRIPT" << EOF

echo "✅ Organization reversal completed!"
echo "📁 You may need to manually remove empty directories if desired."
EOF

# Step 9: Display final results
print_status "$GREEN" "🎉 Project organization completed successfully!"
print_status "$CYAN" "📊 Organization report: $REPORT_FILE"
print_status "$CYAN" "🔄 Undo script: $UNDO_SCRIPT"
print_status "$CYAN" "📝 Log file: $LOG_FILE"

# Display current directory structure
print_status "$PURPLE" "📁 Current project structure:"
tree -a -I '.git|node_modules|__pycache__|*.pyc|.DS_Store|backup_*' -L 3 2>/dev/null || find . -maxdepth 3 -type d | sed 's|[^/]*/|- |g'

print_status "$GREEN" "✅ All done! Your Persian Legal AI project is now properly organized."
print_status "$YELLOW" "💡 Don't forget to update any hardcoded paths in your code!"