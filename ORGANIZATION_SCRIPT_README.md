# Persian Legal AI Project Organization Script

## Overview

This comprehensive bash script organizes your messy Persian Legal AI project structure into a clean, maintainable format. It safely moves files to appropriate directories while creating backups and undo capabilities.

## Features

### ✅ **Safety First**
- **Git Backup**: Creates git stash and commit backup before any changes
- **File System Backup**: Creates full backup if git is not available  
- **Undo Script**: Generates `undo_organization.sh` to reverse all changes
- **Non-destructive**: Only moves files, never deletes them

### 🎨 **Beautiful Output**
- **Colored Progress**: Real-time colored output with progress indicators
- **Section Headers**: Clear visual separation of different phases
- **Progress Tracking**: Shows percentage completion for each phase
- **Status Icons**: ✓ ❌ ⚠️ icons for different outcomes

### 📁 **Smart Organization**
- **Report Files** → `docs/reports/` (all *REPORT*.md files)
- **Guide Files** → `docs/guides/` (all *GUIDE*.md files)
- **Analysis Files** → `docs/analysis/` (analysis and audit files)
- **Test Files** → `tests/` (unit, integration, e2e)
- **Scripts** → `scripts/` (deployment, maintenance, utils)
- **Configs** → `configs/` (development, production, testing)
- **Databases** → `backup/databases/` (all .db files)

### 📊 **Comprehensive Reporting**
- **Detailed Report**: `organization_report.md` with complete summary
- **File Tracking**: Lists all moved files and their destinations
- **Error Reporting**: Documents any files that couldn't be moved
- **Directory Tree**: Shows final project structure

## Directory Structure Created

```
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
```

## Usage

### Basic Usage
```bash
bash organize_project.sh
```

### Interactive Mode
The script will:
1. Ask for confirmation before proceeding
2. Show progress for each phase
3. Display colored output with status updates
4. Generate comprehensive reports

## File Organization Rules

### Markdown Files
- `*REPORT*.md` → `docs/reports/`
- `*GUIDE*.md` → `docs/guides/`
- `*ANALYSIS*.md` → `docs/analysis/`
- `README.md` → **Stays in root**

### Test Files
- `test*.py`, `*test*.py` → `tests/integration/`
- `test*.js`, `*test*.js` → `tests/e2e/`
- `test*.html`, `*test*.html` → `tests/e2e/`

### Script Files
- `*.sh` deployment/merge scripts → `scripts/deployment/`
- `*.sh` health/maintenance scripts → `scripts/maintenance/`
- `*.js`, `*.ts` utility scripts → `scripts/utils/`
- Other `*.sh` scripts → `scripts/`

### Configuration Files
- `docker-compose.yml` → `configs/development/`
- `docker-compose.production.yml` → `configs/production/`

### Database Files
- `*.db`, `*.sqlite`, `*.sqlite3` → `backup/databases/`

## Specific File Mappings

The script handles these specific files from your project:

### Reports → `docs/reports/`
- `BOLT_INTEGRATION_COMPLETE.md`
- `CLEANUP_SUMMARY.md`
- `MERGE_COMPLETION_REPORT.md`
- `PERSIAN_AI_ENHANCEMENT_COMPLETION_REPORT.md`
- `PERSIAN_LEGAL_AI_AUDIT_COMPLETION_REPORT.md`
- `PHASE_3_COMPLETION_SUMMARY.md`
- `PHASE_CRITICAL_PRODUCTION_ISSUES_RESOLVED.md`
- `RECOVERY_REPORT.md`
- `SAFE_MERGE_COMPLETION_REPORT.md`
- `VISUAL_UI_TEST_REPORT.md`
- `integration-report.md`

### Guides → `docs/guides/`
- `DEPLOYMENT_GUIDE.md`
- `START_SYSTEM.md`
- `backend-implementation-guide.md`

### Analysis → `docs/analysis/`
- `BACKEND_MAIN_FILES_ANALYSIS.md`
- `analysis_report.txt`

### Tests → `tests/`
- `test_frontend.html` → `tests/e2e/`
- `test_integration.py` → `tests/integration/`
- `test_production_system.py` → `tests/integration/`

### Scripts → `scripts/`
- `system_health_check.py` → `scripts/maintenance/`
- `start_system.py` → `scripts/`
- `dependency-analyzer.js` → `scripts/utils/`
- `backend-validator.js` → `scripts/utils/`
- All `.sh` files → `scripts/` (categorized by type)

## Safety Features

### Git Backup
```bash
# Creates git stash
git stash push -m "Pre-organization backup $(date)"

# Creates commit backup  
git commit -m "Pre-organization backup $(date)"
```

### File System Backup
If git is not available, creates full directory backup:
```
organization_backup_YYYYMMDD_HHMMSS/
```

### Undo Capability
Generates `undo_organization.sh` that can reverse all changes:
```bash
bash undo_organization.sh
```

## Generated Files

After running, you'll have:

1. **`organization_report.md`** - Detailed report of all changes
2. **`undo_organization.sh`** - Script to reverse organization
3. **`.gitkeep` files** - In all empty directories

## Error Handling

The script handles various error conditions:

- **Missing Files**: Reports files that don't exist
- **Permission Issues**: Reports files that can't be moved
- **Duplicate Names**: Handles conflicts gracefully
- **Invalid Paths**: Creates necessary directories

## Requirements

- **Bash 4.0+** (uses associative arrays and modern features)
- **Basic Unix tools**: `mv`, `mkdir`, `find`, `sed`
- **Optional**: `tree` command for better directory display
- **Optional**: `git` for backup functionality

## Troubleshooting

### Common Issues

**Script won't run:**
```bash
chmod +x organize_project.sh
```

**Permission denied:**
```bash
# Check file permissions
ls -la organize_project.sh

# Make executable
chmod +x organize_project.sh
```

**Files not moving:**
- Check if files exist in root directory
- Verify you have write permissions
- Look at the generated report for specific errors

### Recovery

If something goes wrong:

1. **Use undo script:**
   ```bash
   bash undo_organization.sh
   ```

2. **Restore from git backup:**
   ```bash
   git stash pop
   ```

3. **Restore from file backup:**
   ```bash
   cp -r organization_backup_*/. .
   ```

## Script Architecture

### Functions Overview
- `create_directory_structure()` - Creates all target directories
- `organize_*_files()` - Specialized functions for each file type
- `safe_move()` - Safely moves files with error handling
- `generate_report()` - Creates comprehensive report
- `create_git_backup()` - Creates git-based backups
- `check_uncategorized_files()` - Finds files that weren't categorized

### Color Coding
- 🔵 **Blue**: Progress and informational messages
- 🟢 **Green**: Success messages and completions
- 🟡 **Yellow**: Warnings and non-critical issues
- 🔴 **Red**: Errors and critical issues
- 🟣 **Purple**: Headers and titles
- 🔷 **Cyan**: Section dividers and final summaries

## Contributing

If you need to modify the script:

1. **Test syntax**: `bash -n organize_project.sh`
2. **Add new file types**: Extend the appropriate `organize_*_files()` function
3. **Update mappings**: Modify the arrays in each organization function
4. **Test thoroughly**: Always test with a backup first

## License

This script is part of the Persian Legal AI project and follows the same licensing terms.

---

**Generated by**: AI Assistant  
**Date**: September 11, 2025  
**Version**: 1.0

🎉 **Happy organizing!** Your Persian Legal AI project will be much cleaner and more maintainable after running this script.