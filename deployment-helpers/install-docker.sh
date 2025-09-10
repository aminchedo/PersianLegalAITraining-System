#!/bin/bash
# 🐳 Docker Installation Helper for Persian Legal AI
# Helps install Docker and Docker Compose on various systems

set -e

echo "🐳 Docker Installation Helper"
echo "🛡️ This script helps install Docker without modifying existing files"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VER=$(lsb_release -sr)
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi
    
    echo "🖥️ Detected OS: $OS $VER"
}

# Check if Docker is already installed
check_docker() {
    if command -v docker >/dev/null 2>&1; then
        echo "${GREEN}✅ Docker is already installed${NC}"
        docker --version
        return 0
    else
        echo "${YELLOW}⚠️ Docker is not installed${NC}"
        return 1
    fi
}

# Check if Docker Compose is available
check_docker_compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        echo "${GREEN}✅ Docker Compose is already installed${NC}"
        docker-compose --version
        return 0
    elif docker compose version >/dev/null 2>&1; then
        echo "${GREEN}✅ Docker Compose (plugin) is available${NC}"
        docker compose version
        return 0
    else
        echo "${YELLOW}⚠️ Docker Compose is not available${NC}"
        return 1
    fi
}

# Install Docker on Ubuntu/Debian
install_docker_ubuntu() {
    echo "${BLUE}📦 Installing Docker on Ubuntu/Debian...${NC}"
    
    # Update package index
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo \
      "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Update package index again
    sudo apt-get update
    
    # Install Docker
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    echo "${GREEN}✅ Docker installed successfully${NC}"
    echo "${YELLOW}⚠️ Please log out and log back in for group changes to take effect${NC}"
}

# Install Docker on CentOS/RHEL
install_docker_centos() {
    echo "${BLUE}📦 Installing Docker on CentOS/RHEL...${NC}"
    
    # Install required packages
    sudo yum install -y yum-utils
    
    # Add Docker repository
    sudo yum-config-manager \
        --add-repo \
        https://download.docker.com/linux/centos/docker-ce.repo
    
    # Install Docker
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    echo "${GREEN}✅ Docker installed successfully${NC}"
    echo "${YELLOW}⚠️ Please log out and log back in for group changes to take effect${NC}"
}

# Install Docker using convenience script
install_docker_generic() {
    echo "${BLUE}📦 Installing Docker using convenience script...${NC}"
    
    # Download and run Docker installation script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Clean up
    rm get-docker.sh
    
    echo "${GREEN}✅ Docker installed successfully${NC}"
    echo "${YELLOW}⚠️ Please log out and log back in for group changes to take effect${NC}"
}

# Install Docker Compose (standalone)
install_docker_compose_standalone() {
    echo "${BLUE}📦 Installing Docker Compose (standalone)...${NC}"
    
    # Get latest version
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)
    
    # Download Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    
    # Make executable
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Create symlink
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    echo "${GREEN}✅ Docker Compose installed successfully${NC}"
}

# Main installation function
main() {
    echo "🔍 Checking current Docker installation..."
    
    detect_os
    
    # Check if Docker is already installed
    if check_docker; then
        echo ""
        check_docker_compose
        echo ""
        echo "${GREEN}🎉 Docker setup is complete!${NC}"
        return 0
    fi
    
    echo ""
    echo "🚀 Docker installation options:"
    echo "1) Ubuntu/Debian automatic installation"
    echo "2) CentOS/RHEL automatic installation"
    echo "3) Generic installation (convenience script)"
    echo "4) Manual installation instructions"
    echo "5) Exit"
    echo ""
    
    read -p "Choose an option (1-5): " choice
    
    case $choice in
        1)
            install_docker_ubuntu
            ;;
        2)
            install_docker_centos
            ;;
        3)
            install_docker_generic
            ;;
        4)
            show_manual_instructions
            ;;
        5)
            echo "👋 Exiting..."
            exit 0
            ;;
        *)
            echo "${RED}❌ Invalid option${NC}"
            exit 1
            ;;
    esac
    
    echo ""
    echo "🔍 Verifying installation..."
    
    # Check Docker installation
    if check_docker; then
        echo ""
        # Check Docker Compose
        if ! check_docker_compose; then
            echo ""
            read -p "Install Docker Compose standalone? (y/n): " install_compose
            if [ "$install_compose" = "y" ] || [ "$install_compose" = "Y" ]; then
                install_docker_compose_standalone
            fi
        fi
        
        echo ""
        echo "${GREEN}🎉 Docker installation complete!${NC}"
        echo ""
        echo "📝 Next steps:"
        echo "1. Log out and log back in (or restart your terminal)"
        echo "2. Run: docker --version"
        echo "3. Run: docker-compose --version (or docker compose version)"
        echo "4. Navigate to your project directory"
        echo "5. Run: ./deployment-helpers/deployment-helper.sh"
        
    else
        echo "${RED}❌ Docker installation failed${NC}"
        echo "Please check the error messages above and try manual installation"
        show_manual_instructions
    fi
}

# Show manual installation instructions
show_manual_instructions() {
    echo ""
    echo "${BLUE}📋 Manual Installation Instructions:${NC}"
    echo ""
    echo "🐳 Docker:"
    echo "  Visit: https://docs.docker.com/get-docker/"
    echo "  Follow instructions for your operating system"
    echo ""
    echo "🔧 Docker Compose:"
    echo "  Visit: https://docs.docker.com/compose/install/"
    echo "  Or use pip: pip install docker-compose"
    echo ""
    echo "🍎 macOS:"
    echo "  Download Docker Desktop from docker.com"
    echo ""
    echo "🪟 Windows:"
    echo "  Download Docker Desktop from docker.com"
    echo "  Enable WSL2 backend for better performance"
    echo ""
    echo "🐧 Linux:"
    echo "  Use your distribution's package manager"
    echo "  Or use the convenience script: curl -fsSL https://get.docker.com | sh"
}

# Run main function
main
