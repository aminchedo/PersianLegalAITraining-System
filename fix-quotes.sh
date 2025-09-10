#!/bin/bash
# fix-quotes.sh - Fix mixed quotes in import statements

echo "🔧 Fixing mixed quotes in import statements..."

# Fix mixed quotes in all Bolt components
find /workspace/frontend/src/components/bolt -name "*.tsx" -exec sed -i.bak \
    -e "s|from \"../../../hooks/usePersianAI'|from '../../../hooks/usePersianAI'|g" \
    -e "s|from '../../../hooks/usePersianAI\"|from '../../../hooks/usePersianAI'|g" \
    -e "s|from \"../../../|from '../../../|g" \
    -e "s|../../../.*'|&|g" \
    -e "s|../../../.*\"|&|g" \
    {} \;

# Fix any remaining quote issues
find /workspace/frontend/src/components/bolt -name "*.tsx" -exec sed -i.bak2 \
    -e "s|from \"\\.\\.\\./\\.\\.\\./\\.\\.\\./|from '../../../|g" \
    -e "s|';\$|';|g" \
    -e "s|\";$|\";|g" \
    {} \;

# Remove backup files
find /workspace/frontend/src/components/bolt -name "*.bak*" -delete

echo "✅ Fixed quote issues in import statements"