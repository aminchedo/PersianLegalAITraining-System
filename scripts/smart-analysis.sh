#!/bin/bash
# smart-analysis.sh - Intelligent File Analysis for Bolt Integration

echo "🔍 Starting intelligent file analysis..."

# Function to compare file complexity and completeness
compare_files() {
    local source_file="$1"
    local target_file="$2"
    
    if [ ! -f "$source_file" ]; then
        echo "❌ Source file not found: $source_file"
        return 1
    fi
    
    if [ ! -f "$target_file" ]; then
        echo "✅ Target file doesn't exist - safe to create: $target_file"
        return 0
    fi
    
    # File size comparison
    source_size=$(wc -l < "$source_file")
    target_size=$(wc -l < "$target_file")
    
    # Function count comparison
    source_functions=$(grep -c "function\|const.*=.*=>\|async.*(" "$source_file" || echo 0)
    target_functions=$(grep -c "function\|const.*=.*=>\|async.*(" "$target_file" || echo 0)
    
    # Import count comparison
    source_imports=$(grep -c "^import" "$source_file" || echo 0)
    target_imports=$(grep -c "^import" "$target_file" || echo 0)
    
    # Type definitions count
    source_types=$(grep -c "interface\|type.*=" "$source_file" || echo 0)
    target_types=$(grep -c "interface\|type.*=" "$target_file" || echo 0)
    
    echo "📊 File Analysis: $(basename $source_file)"
    echo "   Source: $source_size lines, $source_functions functions, $source_imports imports, $source_types types"
    echo "   Target: $target_size lines, $target_functions functions, $target_imports imports, $target_types types"
    
    # Decision logic
    source_score=$((source_size + source_functions * 10 + source_imports * 5 + source_types * 8))
    target_score=$((target_size + target_functions * 10 + target_imports * 5 + target_types * 8))
    
    if [ $source_score -gt $target_score ]; then
        echo "✅ Source is more complete - recommend replacement"
        return 0
    else
        echo "⚠️  Target is more complete - recommend manual merge"
        return 2
    fi
}

# Create analysis log
echo "📝 Creating analysis log..."
cat > analysis.log << EOF
File Analysis Log - $(date)
============================
EOF

# Analyze all potential conflicts
echo "🔎 Analyzing potential file conflicts..."

# Check components
if [ -d "bolt/src/components" ]; then
    echo "🧩 Analyzing components..."
    find bolt/src/components -name "*.tsx" -o -name "*.ts" | while read source_file; do
        relative_path=$(echo "$source_file" | sed 's|bolt/src/components/||')
        target_file="frontend/src/components/$relative_path"
        
        echo "Analyzing: $source_file vs $target_file" >> analysis.log
        compare_files "$source_file" "$target_file" >> analysis.log 2>&1
        echo "---" >> analysis.log
    done
fi

# Check API files
if [ -d "bolt/src/api" ]; then
    echo "📡 Analyzing API files..."
    find bolt/src/api -name "*.ts" | while read source_file; do
        echo "📡 API File: $source_file"
        lines=$(wc -l < "$source_file")
        functions=$(grep -c "async\|function\|static.*(" "$source_file" || echo 0)
        echo "   $lines lines, $functions functions - will consolidate into boltApi.ts"
        echo "API Analysis: $source_file ($lines lines, $functions functions)" >> analysis.log
    done
fi

# Check hooks
if [ -d "bolt/src/hooks" ]; then
    echo "🪝 Analyzing hooks..."
    find bolt/src/hooks -name "*.ts" | while read source_file; do
        relative_path=$(echo "$source_file" | sed 's|bolt/src/hooks/||')
        target_file="frontend/src/hooks/$relative_path"
        
        echo "Hook Analysis: $source_file vs $target_file" >> analysis.log
        compare_files "$source_file" "$target_file" >> analysis.log 2>&1
        echo "---" >> analysis.log
    done
fi

echo "✅ Analysis completed. Check analysis.log for detailed results."

# Summary
echo "📊 ANALYSIS SUMMARY:"
echo "==================="

# Count files to be migrated
bolt_components=$(find bolt/src/components -name "*.tsx" -o -name "*.ts" | wc -l)
bolt_api_files=$(find bolt/src/api -name "*.ts" | wc -l)
bolt_hooks=$(find bolt/src/hooks -name "*.ts" | wc -l)

echo "📁 Files to migrate:"
echo "   - Components: $bolt_components files"
echo "   - API files: $bolt_api_files files"
echo "   - Hooks: $bolt_hooks files"

# Check for conflicts
conflicts=0
if [ -f "frontend/src/components/CompletePersianAIDashboard.tsx" ] && [ -f "bolt/src/components/CompletePersianAIDashboard.tsx" ]; then
    conflicts=$((conflicts + 1))
    echo "⚠️  CONFLICT: CompletePersianAIDashboard.tsx exists in both locations"
fi

if [ -f "frontend/src/api/persian-ai-api.js" ] && [ -f "bolt/src/api/persian-ai-api.ts" ]; then
    conflicts=$((conflicts + 1))
    echo "⚠️  CONFLICT: API files exist in both locations (JS vs TS)"
fi

echo "🔍 Total potential conflicts: $conflicts"

if [ $conflicts -eq 0 ]; then
    echo "✅ No major conflicts detected - safe to proceed with migration"
else
    echo "⚠️  Conflicts detected - will require careful merging"
fi