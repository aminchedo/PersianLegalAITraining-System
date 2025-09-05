#!/bin/bash

# Persian Legal AI - Frontend Update Script
# اسکریپت به‌روزرسانی Frontend برای سیستم هوش مصنوعی حقوقی فارسی

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
check_node() {
    log_info "Checking Node.js installation..."
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js first."
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed. Please install npm first."
        exit 1
    fi
    
    local node_version=$(node --version)
    log_success "Node.js $node_version is installed"
}

# Install frontend dependencies
install_dependencies() {
    log_info "Installing frontend dependencies..."
    
    cd frontend
    
    if [ ! -f package.json ]; then
        log_error "package.json not found in frontend directory"
        exit 1
    fi
    
    # Clean install
    rm -rf node_modules package-lock.json
    npm install
    
    log_success "Dependencies installed successfully"
    cd ..
}

# Build frontend
build_frontend() {
    log_info "Building frontend..."
    
    cd frontend
    
    # Run linting
    log_info "Running ESLint..."
    npm run lint || log_warning "Linting issues found, but continuing..."
    
    # Run tests
    log_info "Running tests..."
    npm run test:run || log_warning "Some tests failed, but continuing..."
    
    # Build production bundle
    log_info "Building production bundle..."
    npm run build
    
    log_success "Frontend built successfully"
    cd ..
}

# Update Docker frontend container
update_docker_frontend() {
    log_info "Updating Docker frontend container..."
    
    # Rebuild and restart frontend container
    docker-compose build frontend
    docker-compose up -d frontend
    
    # Wait for container to be healthy
    log_info "Waiting for frontend container to be healthy..."
    timeout 60 bash -c 'until curl -f http://localhost:3000/; do sleep 5; done'
    
    log_success "Docker frontend container updated successfully"
}

# Run development server
start_dev_server() {
    log_info "Starting development server..."
    
    cd frontend
    npm run dev &
    local dev_pid=$!
    
    log_success "Development server started (PID: $dev_pid)"
    log_info "Frontend available at http://localhost:5173"
    log_info "Press Ctrl+C to stop the server"
    
    # Wait for user to stop
    wait $dev_pid
    cd ..
}

# Deploy to production
deploy_production() {
    log_info "Deploying to production..."
    
    # Build and update Docker container
    build_frontend
    update_docker_frontend
    
    log_success "Production deployment completed"
}

# Run tests
run_tests() {
    log_info "Running frontend tests..."
    
    cd frontend
    
    # Unit tests
    log_info "Running unit tests..."
    npm run test:run
    
    # Coverage report
    log_info "Generating coverage report..."
    npm run coverage
    
    log_success "Tests completed"
    cd ..
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    
    cd frontend
    rm -rf dist node_modules package-lock.json
    log_success "Build artifacts cleaned"
    cd ..
}

# Show help
show_help() {
    echo "Persian Legal AI Frontend Update Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install       Install frontend dependencies"
    echo "  build         Build frontend for production"
    echo "  dev           Start development server"
    echo "  deploy        Deploy to production (Docker)"
    echo "  test          Run tests and generate coverage"
    echo "  clean         Clean build artifacts"
    echo "  docker        Update Docker frontend container only"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install    # Install dependencies"
    echo "  $0 build      # Build for production"
    echo "  $0 dev        # Start development server"
    echo "  $0 deploy     # Deploy to production"
}

# Main function
main() {
    local command=${1:-help}
    
    case $command in
        install)
            check_node
            install_dependencies
            ;;
        build)
            check_node
            install_dependencies
            build_frontend
            ;;
        dev)
            check_node
            install_dependencies
            start_dev_server
            ;;
        deploy)
            check_node
            deploy_production
            ;;
        test)
            check_node
            install_dependencies
            run_tests
            ;;
        clean)
            clean_build
            ;;
        docker)
            update_docker_frontend
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"