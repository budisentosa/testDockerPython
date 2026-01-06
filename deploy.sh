#!/bin/bash

# Deployment script for Jupyter Lab on VM
# Usage: ./deploy.sh [build|start|stop|restart|logs|status]

set -e

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="testdockerpython"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function check_requirements() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
}

function build() {
    print_info "Building Docker image..."
    docker-compose build
    print_info "Build complete!"
}

function start() {
    print_info "Starting Jupyter Lab..."
    docker-compose up -d

    sleep 3

    print_info "Jupyter Lab is running!"
    print_info "Fetching access token..."
    echo ""
    docker-compose logs | grep -E "http://.*:8888.*token=" || print_warn "Token not found in logs. Run './deploy.sh logs' to view full logs."
    echo ""
    print_info "Access Jupyter at: http://localhost:8888"
}

function stop() {
    print_info "Stopping Jupyter Lab..."
    docker-compose down
    print_info "Stopped!"
}

function restart() {
    print_info "Restarting Jupyter Lab..."
    docker-compose restart
    print_info "Restarted!"
}

function logs() {
    print_info "Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

function status() {
    print_info "Container status:"
    docker-compose ps
}

function show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  start      Start Jupyter Lab"
    echo "  stop       Stop Jupyter Lab"
    echo "  restart    Restart Jupyter Lab"
    echo "  logs       Show and follow logs"
    echo "  status     Show container status"
    echo "  help       Show this help message"
    echo ""
    echo "If no command is provided, 'start' will be executed."
}

# Main logic
check_requirements

COMMAND=${1:-start}

case $COMMAND in
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    help)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
