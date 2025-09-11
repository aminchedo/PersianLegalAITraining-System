#!/bin/bash
# comprehensive-test.sh - Comprehensive Testing & Validation System

echo "🧪 Starting comprehensive testing..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNINGS=0

# Function to log test results
log_test_result() {
    local test_name="$1"
    local result="$2"
    local message="$3"
    
    case $result in
        "PASS")
            echo -e "${GREEN}✅ $test_name: PASSED${NC} - $message"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            ;;
        "FAIL")
            echo -e "${RED}❌ $test_name: FAILED${NC} - $message"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            ;;
        "WARN")
            echo -e "${YELLOW}⚠️  $test_name: WARNING${NC} - $message"
            TESTS_WARNINGS=$((TESTS_WARNINGS + 1))
            ;;
    esac
}

# Function to test TypeScript compilation
test_typescript() {
    echo -e "${BLUE}📝 Testing TypeScript compilation...${NC}"
    
    cd /workspace/frontend
    
    if npm run type-check 2>/dev/null; then
        log_test_result "TypeScript" "PASS" "All types compile successfully"
        typescript_status="✅"
    else
        log_test_result "TypeScript" "FAIL" "TypeScript compilation errors found"
        typescript_status="❌"
        
        # Show specific errors
        echo "TypeScript errors:"
        npm run type-check 2>&1 | tail -20
    fi
    
    cd /workspace
}

# Function to test ESLint
test_linting() {
    echo -e "${BLUE}🔍 Testing code quality (ESLint)...${NC}"
    
    cd /workspace/frontend
    
    # Run ESLint and capture output
    if npm run lint 2>/dev/null; then
        log_test_result "ESLint" "PASS" "No linting errors found"
        linting_status="✅"
    else
        # Check if it's just warnings or actual errors
        lint_output=$(npm run lint 2>&1)
        if echo "$lint_output" | grep -q "error"; then
            log_test_result "ESLint" "FAIL" "ESLint errors found"
            linting_status="❌"
        else
            log_test_result "ESLint" "WARN" "ESLint warnings found"
            linting_status="⚠️"
        fi
        
        echo "ESLint output (last 20 lines):"
        echo "$lint_output" | tail -20
    fi
    
    cd /workspace
}

# Function to test build process
test_build_process() {
    echo -e "${BLUE}🏗️  Testing build process...${NC}"
    
    cd /workspace/frontend
    
    # Clean previous builds
    rm -rf build/ dist/
    
    # Run build
    if timeout 300 npm run build 2>/dev/null; then
        log_test_result "Build" "PASS" "Build completed successfully"
        build_status="✅"
        
        # Check bundle sizes
        if [ -d "build" ]; then
            echo "📊 Bundle size analysis:"
            find build -name "*.js" -exec ls -lh {} \; | head -10
            
            # Check for large bundles (>5MB)
            large_files=$(find build -name "*.js" -size +5M)
            if [ -n "$large_files" ]; then
                log_test_result "Bundle Size" "WARN" "Large bundles detected"
                echo "$large_files"
            else
                log_test_result "Bundle Size" "PASS" "Bundle sizes are acceptable"
            fi
        fi
        
    else
        log_test_result "Build" "FAIL" "Build process failed or timed out"
        build_status="❌"
        
        # Show build errors
        echo "Build errors:"
        npm run build 2>&1 | tail -30
    fi
    
    cd /workspace
}

# Function to validate Bolt functionality
test_bolt_functionality() {
    echo -e "${BLUE}⚡ Testing Bolt-specific functionality...${NC}"
    
    cd /workspace/frontend
    
    # Check if Bolt components exist
    bolt_components=(
        "src/components/bolt/pages/analytics-page.tsx"
        "src/components/bolt/pages/data-page.tsx"
        "src/components/bolt/components/CompletePersianAIDashboard.tsx"
        "src/api/boltApi.ts"
        "src/services/boltContext.tsx"
        "src/types/bolt.ts"
    )
    
    missing_components=0
    for component in "${bolt_components[@]}"; do
        if [ -f "$component" ]; then
            log_test_result "Bolt File" "PASS" "Found: $component"
        else
            log_test_result "Bolt File" "FAIL" "Missing: $component"
            missing_components=$((missing_components + 1))
        fi
    done
    
    if [ $missing_components -eq 0 ]; then
        bolt_status="✅"
    else
        bolt_status="❌"
    fi
    
    # Test import statements
    echo "🔍 Validating import statements..."
    problematic_imports=$(grep -r "from.*@/" src/components/bolt/ 2>/dev/null || true)
    if [ -n "$problematic_imports" ]; then
        log_test_result "Bolt Imports" "WARN" "Found @ imports in Bolt components"
        echo "$problematic_imports" | head -5
    else
        log_test_result "Bolt Imports" "PASS" "No problematic @ imports found"
    fi
    
    # Check for circular dependencies (basic check)
    echo "🔄 Checking for potential circular dependencies..."
    if find src/components/bolt -name "*.tsx" -exec grep -l "import.*CompletePersianAIDashboard" {} \; | grep -v CompletePersianAIDashboard.tsx; then
        log_test_result "Circular Deps" "WARN" "Potential circular dependency detected"
    else
        log_test_result "Circular Deps" "PASS" "No obvious circular dependencies"
    fi
    
    cd /workspace
}

# Function to test API integration
test_api_integration() {
    echo -e "${BLUE}📡 Testing API integration...${NC}"
    
    cd /workspace/frontend
    
    # Check if boltApi.ts is properly structured
    if [ -f "src/api/boltApi.ts" ]; then
        # Check for required methods
        required_methods=("healthCheck" "axiosInstance" "setupInterceptors")
        missing_methods=0
        
        for method in "${required_methods[@]}"; do
            if grep -q "$method" src/api/boltApi.ts; then
                log_test_result "API Method" "PASS" "Found method: $method"
            else
                log_test_result "API Method" "FAIL" "Missing method: $method"
                missing_methods=$((missing_methods + 1))
            fi
        done
        
        if [ $missing_methods -eq 0 ]; then
            api_status="✅"
        else
            api_status="❌"
        fi
    else
        log_test_result "API File" "FAIL" "boltApi.ts not found"
        api_status="❌"
    fi
    
    cd /workspace
}

# Function to test dashboard integration
test_dashboard_integration() {
    echo -e "${BLUE}🎛️  Testing dashboard integration...${NC}"
    
    cd /workspace/frontend
    
    # Check if dashboard has been updated with Bolt routes
    if grep -q "bolt-analytics" src/components/CompletePersianAIDashboard.tsx; then
        log_test_result "Dashboard Routes" "PASS" "Bolt routes found in dashboard"
    else
        log_test_result "Dashboard Routes" "FAIL" "Bolt routes not found in dashboard"
    fi
    
    # Check if BoltProvider is imported
    if grep -q "BoltProvider" src/components/CompletePersianAIDashboard.tsx; then
        log_test_result "Bolt Provider" "PASS" "BoltProvider imported in dashboard"
    else
        log_test_result "Bolt Provider" "FAIL" "BoltProvider not imported"
    fi
    
    # Check if error boundary exists
    if [ -f "src/components/bolt/BoltErrorBoundary.tsx" ]; then
        log_test_result "Error Boundary" "PASS" "Bolt error boundary exists"
    else
        log_test_result "Error Boundary" "FAIL" "Bolt error boundary missing"
    fi
    
    cd /workspace
}

# Function to generate integration report
generate_integration_report() {
    echo -e "${BLUE}📊 Generating integration report...${NC}"
    
    cat > /workspace/integration-report.md << EOF
# Bolt Integration Report
Generated: $(date)

## Test Results Summary
- TypeScript Compilation: $typescript_status
- ESLint: $linting_status  
- Build Process: $build_status
- Bolt Functionality: $bolt_status
- API Integration: $api_status
- Dashboard Integration: Dashboard tests completed

## Test Statistics
- ✅ Tests Passed: $TESTS_PASSED
- ❌ Tests Failed: $TESTS_FAILED
- ⚠️  Warnings: $TESTS_WARNINGS
- 📊 Total Tests: $((TESTS_PASSED + TESTS_FAILED + TESTS_WARNINGS))

## File Migration Status
$(cat /workspace/migration.log 2>/dev/null || echo "No migration log found")

## Bundle Analysis
$(cd /workspace/frontend && find build -name "*.js" -exec ls -lh {} \; 2>/dev/null | head -5 || echo "No build files found")

## Next Steps
1. Review any failed tests above
2. Fix TypeScript/ESLint errors if any
3. Test manually in browser
4. Verify all Bolt features work correctly
5. Ready for merge if all critical tests pass

## Integration Status
EOF

    # Determine overall status
    if [ $TESTS_FAILED -eq 0 ] && [ "$typescript_status" = "✅" ] && [ "$build_status" = "✅" ]; then
        echo "🎉 **INTEGRATION SUCCESSFUL** - Ready for production" >> /workspace/integration-report.md
        overall_status="SUCCESS"
    elif [ $TESTS_FAILED -gt 0 ] || [ "$typescript_status" = "❌" ] || [ "$build_status" = "❌" ]; then
        echo "❌ **INTEGRATION FAILED** - Critical issues need resolution" >> /workspace/integration-report.md
        overall_status="FAILED"
    else
        echo "⚠️  **INTEGRATION PARTIAL** - Some warnings, review recommended" >> /workspace/integration-report.md
        overall_status="PARTIAL"
    fi
    
    echo "📋 Integration report saved: integration-report.md"
}

# Main execution
echo "🚀 Running comprehensive test suite..."
echo "======================================"

test_typescript
test_linting
test_build_process
test_bolt_functionality
test_api_integration
test_dashboard_integration

generate_integration_report

# Final summary
echo ""
echo "🏁 TESTING COMPLETED"
echo "===================="
echo -e "📊 Results: ${GREEN}$TESTS_PASSED passed${NC}, ${RED}$TESTS_FAILED failed${NC}, ${YELLOW}$TESTS_WARNINGS warnings${NC}"

case $overall_status in
    "SUCCESS")
        echo -e "${GREEN}🎉 Integration validation successful!${NC}"
        echo -e "${GREEN}✅ Ready for safe merge to main branch${NC}"
        exit 0
        ;;
    "FAILED")
        echo -e "${RED}❌ Integration validation failed${NC}"
        echo -e "${RED}🔧 Please fix critical issues before merging${NC}"
        exit 1
        ;;
    "PARTIAL")
        echo -e "${YELLOW}⚠️  Integration partially successful${NC}"
        echo -e "${YELLOW}📋 Review warnings and proceed with caution${NC}"
        exit 0
        ;;
esac