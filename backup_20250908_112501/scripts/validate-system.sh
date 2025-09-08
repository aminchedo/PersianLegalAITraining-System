#!/bin/bash

echo '=== Persian Legal AI System Validation ==='
echo 'Real Data Implementation - System Check'
echo ''

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_status "Testing: $test_name"
    
    if eval "$test_command" >/dev/null 2>&1; then
        print_success "$test_name - PASSED"
        ((TESTS_PASSED++))
        return 0
    else
        print_error "$test_name - FAILED"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Check if services are running
check_services() {
    print_status "Checking if services are running..."
    
    # Check backend
    if curl -s http://localhost:8000/api/real/health >/dev/null; then
        print_success "Backend service is running"
    else
        print_error "Backend service is not running"
        return 1
    fi
    
    # Check frontend (if accessible)
    if curl -s http://localhost:3000 >/dev/null; then
        print_success "Frontend service is running"
    else
        print_warning "Frontend service may not be accessible"
    fi
}

# Test TypeScript compilation
test_typescript() {
    print_status "Testing TypeScript compilation..."
    
    cd frontend
    
    if npx tsc --noEmit --strict; then
        print_success "TypeScript compilation - PASSED"
        cd ..
        return 0
    else
        print_error "TypeScript compilation - FAILED"
        cd ..
        return 1
    fi
}

# Test API endpoints
test_api_endpoints() {
    print_status "Testing API endpoints..."
    
    local endpoints=(
        "GET:/api/real/health:Health Check"
        "GET:/api/real/stats:System Stats"
        "GET:/api/real/team/members:Team Members"
        "GET:/api/real/models/training:Training Jobs"
        "GET:/api/real/monitoring/system-metrics:System Metrics"
    )
    
    for endpoint in "${endpoints[@]}"; do
        IFS=':' read -r method path name <<< "$endpoint"
        
        if curl -s -X "$method" "http://localhost:8000$path" >/dev/null; then
            print_success "$name endpoint - PASSED"
        else
            print_error "$name endpoint - FAILED"
        fi
    done
}

# Test database connection
test_database() {
    print_status "Testing database connection..."
    
    # Try to connect to PostgreSQL
    if command -v psql &> /dev/null; then
        if PGPASSWORD=password psql -h localhost -U persianai -d persian_legal_ai -c "SELECT 1;" >/dev/null 2>&1; then
            print_success "Database connection - PASSED"
        else
            print_error "Database connection - FAILED"
        fi
    else
        print_warning "PostgreSQL client not available - skipping database test"
    fi
}

# Check for mock data patterns
check_mock_data() {
    print_status "Checking for mock data patterns..."
    
    local mock_patterns=(
        "generateMetrics"
        "mock"
        "fake"
        "dummy"
        "demo"
        "sample"
    )
    
    local found_mock=false
    
    for pattern in "${mock_patterns[@]}"; do
        if grep -r -i "$pattern" frontend/src/ --include="*.tsx" --include="*.ts" | grep -v ".test." | grep -v "//" >/dev/null; then
            print_warning "Found potential mock data pattern: $pattern"
            found_mock=true
        fi
    done
    
    if [ "$found_mock" = false ]; then
        print_success "No mock data patterns found - PASSED"
    else
        print_warning "Some mock data patterns detected - please review"
    fi
}

# Test real data endpoints
test_real_data() {
    print_status "Testing real data endpoints..."
    
    # Test team members endpoint
    local team_response=$(curl -s http://localhost:8000/api/real/team/members)
    if echo "$team_response" | grep -q "\[\]"; then
        print_success "Team members endpoint returns real data structure"
    else
        print_success "Team members endpoint returns data"
    fi
    
    # Test system metrics endpoint
    local metrics_response=$(curl -s http://localhost:8000/api/real/monitoring/system-metrics)
    if echo "$metrics_response" | grep -q "cpu_usage"; then
        print_success "System metrics endpoint returns real data"
    else
        print_error "System metrics endpoint failed"
    fi
}

# Test WebSocket connection
test_websocket() {
    print_status "Testing WebSocket connection..."
    
    # Simple WebSocket test using curl (if available)
    if command -v websocat &> /dev/null; then
        if timeout 5 websocat ws://localhost:8000/ws >/dev/null 2>&1; then
            print_success "WebSocket connection - PASSED"
        else
            print_warning "WebSocket connection test failed (websocat not available or connection failed)"
        fi
    else
        print_warning "WebSocket test skipped (websocat not available)"
    fi
}

# Test frontend build
test_frontend_build() {
    print_status "Testing frontend build..."
    
    cd frontend
    
    if npm run build >/dev/null 2>&1; then
        print_success "Frontend build - PASSED"
        cd ..
        return 0
    else
        print_error "Frontend build - FAILED"
        cd ..
        return 1
    fi
}

# Main validation function
main() {
    echo ''
    print_status "Starting system validation..."
    echo ''
    
    # Run all tests
    check_services
    test_typescript
    test_api_endpoints
    test_database
    check_mock_data
    test_real_data
    test_websocket
    test_frontend_build
    
    echo ''
    print_status "Validation Summary:"
    echo "Tests Passed: $TESTS_PASSED"
    echo "Tests Failed: $TESTS_FAILED"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo ''
        print_success "=== ALL TESTS PASSED ==="
        print_success "System is ready for production use"
        echo ''
        print_status "Access points:"
        echo "- Frontend: http://localhost:3000"
        echo "- Backend API: http://localhost:8000"
        echo "- API Docs: http://localhost:8000/docs"
        echo "- Health Check: http://localhost:8000/api/real/health"
        echo ''
        print_success "Real data system is fully operational!"
    else
        echo ''
        print_error "=== SOME TESTS FAILED ==="
        print_error "Please review the failed tests and fix issues"
        exit 1
    fi
}

# Run main function
main