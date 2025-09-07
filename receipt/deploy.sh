#!/bin/bash

# AI Platform Deployment Script
# This script handles git pull, build, and deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists git; then
    print_error "Git is not installed!"
    exit 1
fi

if ! command_exists docker; then
    print_error "Docker is not installed!"
    exit 1
fi

if ! command_exists docker-compose; then
    print_error "Docker Compose is not installed!"
    exit 1
fi

print_success "All prerequisites are available"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "Working in directory: $SCRIPT_DIR"

# Git operations
print_status "Updating code from repository..."

# Check if we're in a git repository
if [ ! -d ".git" ] && [ ! -d "../.git" ]; then
    print_warning "Not in a git repository. Skipping git pull."
else
    # Stash any local changes
    if ! git diff --quiet; then
        print_warning "Local changes detected. Stashing them..."
        git stash push -m "Auto-stash before deployment $(date)"
    fi

    # Pull latest changes
    git pull origin main || {
        print_error "Git pull failed. Please resolve conflicts manually."
        exit 1
    }
    
    print_success "Code updated successfully"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p uploads logs
print_success "Directories created"

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose down --remove-orphans || print_warning "No containers to stop"

# Remove old images (optional - uncomment if you want to always rebuild)
# print_status "Removing old images..."
# docker-compose down --rmi all || print_warning "No images to remove"

# Build and start services
print_status "Building and starting services..."
docker-compose up --build -d

# Wait for service to be ready
print_status "Waiting for service to be ready..."
sleep 10

# Check if the service is healthy
print_status "Checking service health..."
max_attempts=12
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
        print_success "Service is healthy and ready!"
        break
    else
        print_status "Attempt $attempt/$max_attempts - Service not ready yet, waiting..."
        sleep 5
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    print_error "Service failed to become healthy within the expected time"
    print_status "Checking logs..."
    docker-compose logs ai-platform
    exit 1
fi

# Show final status
print_success "Deployment completed successfully!"
print_status "Service is running at: http://localhost:8000"
print_status "Use 'docker-compose logs -f ai-platform' to view logs"
print_status "Use 'docker-compose down' to stop the service"

# Show running containers
print_status "Running containers:"
docker-compose ps
