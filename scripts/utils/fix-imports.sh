#!/bin/bash
# fix-imports.sh - Fix import paths in Bolt components

echo "🔧 Fixing import paths in Bolt components..."

# Fix imports in all Bolt components
find /workspace/frontend/src/components/bolt -name "*.tsx" -exec sed -i.bak \
    -e "s|from '../hooks/usePersianAI'|from '../../../hooks/usePersianAI'|g" \
    -e "s|from \"../hooks/usePersianAI\"|from \"../../../hooks/usePersianAI\"|g" \
    -e "s|from '../hooks/|from '../../../hooks/|g" \
    -e "s|from \"../hooks/|from \"../../../hooks/|g" \
    -e "s|from '../api/|from '../../../api/|g" \
    -e "s|from \"../api/|from \"../../../api/|g" \
    -e "s|from '../types/|from '../../../types/|g" \
    -e "s|from \"../types/|from \"../../../types/|g" \
    -e "s|from '../services/|from '../../../services/|g" \
    -e "s|from \"../services/|from \"../../../services/|g" \
    -e "s|from '../lib/|from '../../../lib/|g" \
    -e "s|from \"../lib/|from \"../../../lib/|g" \
    {} \;

# Remove backup files
find /workspace/frontend/src/components/bolt -name "*.bak" -delete

echo "✅ Fixed import paths in Bolt components"

# List files that were processed
echo "📋 Processed files:"
find /workspace/frontend/src/components/bolt -name "*.tsx" | while read file; do
    echo "   - $file"
done