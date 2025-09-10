#!/bin/bash
set -e

echo "🧪 Running Persian Legal AI Integration Tests..."
echo "================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to log test results
log_test() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ $2${NC}"
        ((TESTS_FAILED++))
    fi
}

# Function to test HTTP endpoint
test_endpoint() {
    local url=$1
    local expected_status=$2
    local description=$3
    
    echo -n "Testing $description... "
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response.json "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        log_test 0 "$description"
        return 0
    else
        log_test 1 "$description (got $response, expected $expected_status)"
        return 1
    fi
}

# Function to test POST endpoint
test_post_endpoint() {
    local url=$1
    local data=$2
    local expected_status=$3
    local description=$4
    
    echo -n "Testing $description... "
    
    response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$data" \
        "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        log_test 0 "$description"
        return 0
    else
        log_test 1 "$description (got $response, expected $expected_status)"
        return 1
    fi
}

echo -e "${YELLOW}Phase 1: Backend API Tests${NC}"
echo "----------------------------"

# Test backend root endpoint
test_endpoint "http://localhost:8000/" "200" "Backend root endpoint"

# Test health endpoint
test_endpoint "http://localhost:8000/api/system/health" "200" "System health endpoint"

# Test classification endpoint
test_post_endpoint "http://localhost:8000/api/ai/classify" \
    '{"text": "این یک قرارداد حقوقی است", "include_confidence": true}' \
    "200" "AI classification endpoint"

# Test document stats endpoint
test_endpoint "http://localhost:8000/api/documents/stats" "200" "Document statistics endpoint"

# Test search endpoint
test_endpoint "http://localhost:8000/api/documents/search?q=قانون&limit=5" "200" "Document search endpoint"

echo ""
echo -e "${YELLOW}Phase 2: Frontend Tests${NC}"
echo "------------------------"

# Test frontend accessibility
test_endpoint "http://localhost:5173/" "200" "Frontend home page"

# Test that frontend can reach backend through proxy
test_endpoint "http://localhost:5173/api/system/health" "200" "Frontend-to-backend proxy"

echo ""
echo -e "${YELLOW}Phase 3: Integration Tests${NC}"
echo "------------------------------"

# Test full integration flow
echo -n "Testing full classification flow... "
response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"text": "این یک سند حقوقی مربوط به قرارداد خرید و فروش است", "include_confidence": true}' \
    "http://localhost:5173/api/ai/classify" 2>/dev/null)

if echo "$response" | grep -q "classification"; then
    log_test 0 "Full classification integration"
else
    log_test 1 "Full classification integration"
fi

# Test system health through frontend
echo -n "Testing system health through frontend... "
response=$(curl -s "http://localhost:5173/api/system/health" 2>/dev/null)

if echo "$response" | grep -q "database_connected"; then
    log_test 0 "System health integration"
else
    log_test 1 "System health integration"
fi

echo ""
echo -e "${YELLOW}Phase 4: Performance Tests${NC}"
echo "------------------------------"

# Test response times
echo -n "Testing backend response time... "
start_time=$(date +%s%N)
curl -s "http://localhost:8000/api/system/health" > /dev/null
end_time=$(date +%s%N)
response_time=$(((end_time - start_time) / 1000000))

if [ $response_time -lt 1000 ]; then
    log_test 0 "Backend response time (${response_time}ms)"
else
    log_test 1 "Backend response time (${response_time}ms - too slow)"
fi

echo -n "Testing frontend response time... "
start_time=$(date +%s%N)
curl -s "http://localhost:5173/" > /dev/null
end_time=$(date +%s%N)
response_time=$(((end_time - start_time) / 1000000))

if [ $response_time -lt 2000 ]; then
    log_test 0 "Frontend response time (${response_time}ms)"
else
    log_test 1 "Frontend response time (${response_time}ms - too slow)"
fi

echo ""
echo "================================================"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All integration tests passed!${NC}"
    echo -e "${GREEN}Persian Legal AI System is fully operational!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Check the output above.${NC}"
    exit 1
fi