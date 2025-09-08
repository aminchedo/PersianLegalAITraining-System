#!/bin/bash

# Persian Legal AI - Docker Deployment Script
# اسکریپت استقرار Docker برای سیستم هوش مصنوعی حقوقی فارسی

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

# Check if Docker is installed
check_docker() {
    log_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are installed"
}

# Check if NVIDIA Docker is available
check_nvidia_docker() {
    log_info "Checking NVIDIA Docker support..."
    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        log_success "NVIDIA Docker support detected"
        return 0
    else
        log_warning "NVIDIA Docker support not detected. GPU acceleration may not be available."
        return 1
    fi
}

# Generate SSL certificates
generate_certificates() {
    log_info "Generating SSL certificates..."
    
    mkdir -p backend/certificates
    mkdir -p nginx/ssl
    
    # Generate self-signed certificate for development
    if [ ! -f backend/certificates/server.crt ]; then
        openssl req -x509 -newkey rsa:4096 -keyout backend/certificates/server.key \
            -out backend/certificates/server.crt -days 365 -nodes \
            -subj "/C=IR/ST=Tehran/L=Tehran/O=Persian Legal AI/OU=IT Department/CN=localhost"
        
        # Copy to nginx directory
        cp backend/certificates/server.crt nginx/ssl/
        cp backend/certificates/server.key nginx/ssl/
        
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p backend/logs
    mkdir -p backend/models
    mkdir -p backend/data
    mkdir -p backend/certificates
    mkdir -p nginx/ssl
    
    log_success "Directories created"
}

# Build and start services
deploy_services() {
    log_info "Building and starting services..."
    
    # Stop existing containers
    docker-compose down --remove-orphans
    
    # Build and start services
    docker-compose up --build -d
    
    log_success "Services deployed successfully"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    # Wait for database
    log_info "Waiting for database..."
    timeout 60 bash -c 'until docker-compose exec -T database pg_isready -U persian_ai_user -d persian_legal_ai; do sleep 2; done'
    
    # Wait for backend
    log_info "Waiting for backend..."
    timeout 120 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    
    # Wait for frontend
    log_info "Waiting for frontend..."
    timeout 60 bash -c 'until curl -f http://localhost:3000/; do sleep 5; done'
    
    log_success "All services are healthy"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    docker-compose exec backend python -c "
from database.connection import init_database
init_database()
print('Database initialized successfully')
"
    
    log_success "Database migrations completed"
}

# Display service status
show_status() {
    log_info "Service Status:"
    echo ""
    docker-compose ps
    echo ""
    
    log_info "Service URLs:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  Backend Health: http://localhost:8000/health"
    echo "  Database: localhost:5432"
    echo "  Redis: localhost:6379"
    echo ""
    
    log_info "Logs can be viewed with:"
    echo "  docker-compose logs -f [service_name]"
    echo ""
}

# Main deployment function
main() {
    log_info "Starting Persian Legal AI Docker deployment..."
    echo ""
    
    # Parse command line arguments
    SKIP_CERTIFICATES=false
    SKIP_MIGRATIONS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-certificates)
                SKIP_CERTIFICATES=true
                shift
                ;;
            --skip-migrations)
                SKIP_MIGRATIONS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-certificates    Skip SSL certificate generation"
                echo "  --skip-migrations     Skip database migrations"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_docker
    check_nvidia_docker
    create_directories
    
    if [ "$SKIP_CERTIFICATES" = false ]; then
        generate_certificates
    fi
    
    deploy_services
    wait_for_services
    
    if [ "$SKIP_MIGRATIONS" = false ]; then
        run_migrations
    fi
    
    show_status
    
    log_success "Persian Legal AI deployment completed successfully!"
    log_info "You can now access the system at http://localhost:3000"
}

# Run main function
main "$@"