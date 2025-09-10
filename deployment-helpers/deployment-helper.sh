#!/bin/bash
# 🛡️ Safe Deployment Helper for Persian Legal AI
# This script assists with deployment without modifying existing files

set -e

echo "🚀 Persian Legal AI Deployment Helper"
echo "🛡️ SAFETY: This script preserves all existing functionality"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "${RED}❌ Error: docker-compose.yml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "✅ Found docker-compose.yml"

# Check if backup exists
BACKUP_COUNT=$(ls persian-legal-ai-backup-*.tar.gz 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt 0 ]; then
    echo "✅ Backup verified ($BACKUP_COUNT backup files found)"
else
    echo "${YELLOW}⚠️ No backup found - creating one now...${NC}"
    tar -czf "persian-legal-ai-backup-$(date +%Y%m%d_%H%M%S).tar.gz" \
        --exclude=node_modules \
        --exclude=__pycache__ \
        --exclude=.git \
        --exclude="*.tar.gz" \
        .
    echo "✅ Backup created"
fi

# Function to check Docker installation
check_docker() {
    if command -v docker >/dev/null 2>&1; then
        echo "✅ Docker is installed"
        docker --version
    else
        echo "${RED}❌ Docker is not installed${NC}"
        echo "Please install Docker first:"
        echo "  Ubuntu/Debian: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
        echo "  Or visit: https://docs.docker.com/get-docker/"
        return 1
    fi
}

# Function to check Docker Compose
check_docker_compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        echo "✅ Docker Compose is installed"
        docker-compose --version
    elif docker compose version >/dev/null 2>&1; then
        echo "✅ Docker Compose (plugin) is available"
        docker compose version
        echo "Note: Use 'docker compose' instead of 'docker-compose'"
    else
        echo "${RED}❌ Docker Compose is not available${NC}"
        echo "Please install Docker Compose:"
        echo "  pip install docker-compose"
        echo "  Or install Docker Compose plugin"
        return 1
    fi
}

# Function to validate configuration
validate_config() {
    echo "🔍 Validating configuration..."
    
    if docker-compose config >/dev/null 2>&1; then
        echo "✅ Docker Compose configuration is valid"
    elif docker compose config >/dev/null 2>&1; then
        echo "✅ Docker Compose configuration is valid"
    else
        echo "${RED}❌ Docker Compose configuration has errors${NC}"
        echo "Run 'docker-compose config' or 'docker compose config' for details"
        return 1
    fi
}

# Function to check environment setup
check_environment() {
    echo "🌍 Checking environment setup..."
    
    if [ -f ".env" ]; then
        echo "✅ .env file found"
    elif [ -f ".env.production" ]; then
        echo "✅ .env.production file found"
    else
        echo "${YELLOW}⚠️ No .env file found${NC}"
        echo "Consider copying .env.production.example to .env.production"
        echo "and updating the values for your environment"
    fi
}

# Function to check system resources
check_resources() {
    echo "💾 Checking system resources..."
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
        if [ "$AVAILABLE_MEM" -ge 4 ]; then
            echo "✅ Sufficient memory available (${AVAILABLE_MEM}GB)"
        else
            echo "${YELLOW}⚠️ Low memory available (${AVAILABLE_MEM}GB)${NC}"
            echo "Consider closing other applications or adding more RAM"
        fi
    fi
    
    # Check disk space
    AVAILABLE_DISK=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "${AVAILABLE_DISK%.*}" -ge 5 ]; then
        echo "✅ Sufficient disk space available (${AVAILABLE_DISK})"
    else
        echo "${YELLOW}⚠️ Low disk space available (${AVAILABLE_DISK})${NC}"
        echo "Consider freeing up disk space"
    fi
}

# Main deployment check
main() {
    echo "🔍 Running deployment checks..."
    echo ""
    
    check_docker || exit 1
    echo ""
    
    check_docker_compose || exit 1
    echo ""
    
    validate_config || exit 1
    echo ""
    
    check_environment
    echo ""
    
    check_resources
    echo ""
    
    echo "${GREEN}🎉 All checks passed!${NC}"
    echo ""
    echo "🚀 Ready to deploy. You can now run:"
    echo "  docker-compose up -d"
    echo "  OR"
    echo "  docker compose up -d"
    echo ""
    echo "🔍 Monitor with:"
    echo "  docker-compose logs -f"
    echo "  OR"
    echo "  docker compose logs -f"
    echo ""
    echo "🛡️ SAFETY: All original files preserved"
}

# Run main function
main
