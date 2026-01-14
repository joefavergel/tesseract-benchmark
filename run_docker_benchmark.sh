#!/bin/bash
# Docker Benchmark Runner Script
# Builds and runs OCR benchmark containers with configurable input/output paths

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
DATASET_VERSION="v0.0.1"
ITERATIONS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <image_type> [options]"
    echo ""
    echo "Arguments:"
    echo "  image_type    Docker image to build and run (pytesseract, native, or any folder in docker/)"
    echo ""
    echo "Options:"
    echo "  -d, --dataset VERSION   Dataset version to use (default: $DATASET_VERSION)"
    echo "  -i, --iterations N      Number of iterations per file (default: $ITERATIONS)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 pytesseract"
    echo "  $0 native -d v0.1.0"
    echo "  $0 pytesseract -d v0.0.1 -i 5"
    exit 1
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
if [ $# -lt 1 ]; then
    usage
fi

IMAGE_TYPE="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_VERSION="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate image type exists
DOCKER_DIR="$PROJECT_ROOT/docker/$IMAGE_TYPE"
if [ ! -d "$DOCKER_DIR" ]; then
    log_error "Docker folder not found: $DOCKER_DIR"
    echo "Available image types:"
    ls -1 "$PROJECT_ROOT/docker/"
    exit 1
fi

# Validate dataset exists
DATASET_DIR="$PROJECT_ROOT/datalake/datasets/$DATASET_VERSION"
if [ ! -d "$DATASET_DIR" ]; then
    log_error "Dataset not found: $DATASET_DIR"
    echo "Available datasets:"
    ls -1 "$PROJECT_ROOT/datalake/datasets/"
    exit 1
fi

# Define paths
IMAGE_NAME="tesseract-benchmark-$IMAGE_TYPE"
OUTPUT_DIR="$PROJECT_ROOT/results/$DATASET_VERSION/$IMAGE_TYPE"

log_info "Configuration:"
echo "  Image type:      $IMAGE_TYPE"
echo "  Dataset version: $DATASET_VERSION"
echo "  Iterations:      $ITERATIONS"
echo "  Input dir:       $DATASET_DIR"
echo "  Output dir:      $OUTPUT_DIR"
echo ""

# Step 1: Build Docker image
log_info "Building Docker image: $IMAGE_NAME"
cd "$DOCKER_DIR"

if [ -f "Dockerfile" ]; then
    docker build -t "$IMAGE_NAME" .
else
    log_error "Dockerfile not found in $DOCKER_DIR"
    exit 1
fi

# Step 2: Return to project root and cleanup containers
cd "$PROJECT_ROOT"

log_info "Stopping all running containers..."
RUNNING_CONTAINERS=$(docker ps -a -q 2>/dev/null || true)
if [ -n "$RUNNING_CONTAINERS" ]; then
    docker stop $RUNNING_CONTAINERS 2>/dev/null || true
    log_info "Removing stopped containers..."
    docker rm $RUNNING_CONTAINERS 2>/dev/null || true
else
    log_warn "No containers to stop/remove"
fi

# Step 3: Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 4: Run the benchmark container
log_info "Running benchmark container..."
docker run \
    -v "$DATASET_DIR:/app/input:ro" \
    -v "$OUTPUT_DIR:/app/output" \
    -e "ITERATIONS=$ITERATIONS" \
    "$IMAGE_NAME"

log_info "Benchmark complete!"
log_info "Results saved to: $OUTPUT_DIR"

# List generated result files
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  No JSON files found"
