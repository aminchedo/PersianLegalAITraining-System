#!/bin/bash
# Version Verification Script for Persian Legal AI Training System

echo "🔍 Persian Legal AI Training System - Version Verification Report"
echo "=================================================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}📋 VERSION INVENTORY${NC}"
echo "===================="

echo -e "${YELLOW}System Versions:${NC}"
echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
echo "Node.js: $(node --version 2>/dev/null || echo 'Not found')"
echo "npm: $(npm --version 2>/dev/null || echo 'Not found')"
echo ""

echo -e "${YELLOW}Version Management Files:${NC}"
echo ".python-version: $(cat .python-version 2>/dev/null || echo 'Missing')"
echo ".nvmrc: $(cat .nvmrc 2>/dev/null || echo 'Missing')"
echo "runtime.txt: $(cat runtime.txt 2>/dev/null || echo 'Missing')"
echo ""

echo -e "${YELLOW}Package.json Engines:${NC}"
echo "Root package.json:"
grep -A 4 '"engines"' package.json 2>/dev/null || echo "Not found"
echo ""
echo "Frontend package.json:"
grep -A 3 '"engines"' frontend/package.json 2>/dev/null || echo "Not found"
echo ""

echo -e "${YELLOW}Docker Configurations:${NC}"
echo "Backend Dockerfile Python version:"
grep "python3\." backend/Dockerfile 2>/dev/null || echo "Not found"
echo "Frontend Dockerfile Node version:"
grep "FROM node:" frontend/Dockerfile 2>/dev/null || echo "Not found"
echo ""

echo -e "${BLUE}⚠️  COMPATIBILITY CHECK${NC}"
echo "========================"

# Check Python version compatibility
PYTHON_VERSION=$(cat .python-version 2>/dev/null)
REQUIRED_PYTHON=$(grep 'requires-python' pyproject.toml 2>/dev/null | cut -d'"' -f2)
echo -e "${YELLOW}Python Compatibility:${NC}"
echo "Required: $REQUIRED_PYTHON"
echo "Specified: $PYTHON_VERSION"

# Check Node.js version compatibility  
NODE_VERSION=$(cat .nvmrc 2>/dev/null)
NODE_REQUIRED=$(grep '"node"' package.json | cut -d'"' -f4)
echo -e "${YELLOW}Node.js Compatibility:${NC}"
echo "Required: $NODE_REQUIRED"
echo "Specified: $NODE_VERSION"
echo ""

echo -e "${BLUE}🎯 RESOLUTION STATUS${NC}"
echo "===================="

# Check if all files exist
FILES=(".python-version" ".nvmrc" "runtime.txt" "pyproject.toml" "requirements-compatible.txt" "install-versions.sh")
ALL_EXIST=true

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file${NC}"
        ALL_EXIST=false
    fi
done

echo ""

if [ "$ALL_EXIST" = true ]; then
    echo -e "${GREEN}🎉 All version management files created successfully!${NC}"
    echo -e "${GREEN}🚀 System is ready for consistent version management.${NC}"
else
    echo -e "${RED}⚠️  Some files are missing. Please run the installation script.${NC}"
fi

echo ""
echo -e "${BLUE}📝 NEXT STEPS${NC}"
echo "=============="
echo "1. Run: ./install-versions.sh (to install correct versions)"
echo "2. Run: cd frontend && npm install (to install frontend dependencies)"
echo "3. Run: pip install -r requirements-compatible.txt (for Python packages)"
echo "4. Verify: python -c 'import torch; print(torch.__version__)'"