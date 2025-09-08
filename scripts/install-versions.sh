#!/bin/bash
# Persian Legal AI Training System - Version Installation Script
# This script installs the correct Python and Node.js versions

set -e

echo "🚀 Persian Legal AI Training System - Version Installation"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo -e "${YELLOW}Installing pyenv...${NC}"
    curl https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
fi

# Check if nvm is installed
if ! command -v nvm &> /dev/null; then
    echo -e "${YELLOW}Installing nvm...${NC}"
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
fi

# Install correct Python version
echo -e "${BLUE}Installing Python 3.10.12...${NC}"
pyenv install 3.10.12 -s
pyenv local 3.10.12

# Install correct Node.js version
echo -e "${BLUE}Installing Node.js 18.17.0...${NC}"
nvm install 18.17.0
nvm use 18.17.0

# Verify installations
echo -e "${GREEN}Verifying installations...${NC}"
python --version
node --version
npm --version

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements-compatible.txt

# Install frontend dependencies
echo -e "${BLUE}Installing frontend dependencies...${NC}"
cd frontend
npm install
cd ..

echo -e "${GREEN}✅ All versions installed successfully!${NC}"
echo -e "${GREEN}Python: $(python --version)${NC}"
echo -e "${GREEN}Node.js: $(node --version)${NC}"
echo -e "${GREEN}npm: $(npm --version)${NC}"

# Compatibility check
echo -e "${BLUE}Running compatibility check...${NC}"
python -c "
import torch
import transformers
import fastapi
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ FastAPI: {fastapi.__version__}')
print('🎉 All AI/ML libraries are compatible!')
"

echo -e "${GREEN}🎯 Installation complete! System is ready for Persian Legal AI Training.${NC}"