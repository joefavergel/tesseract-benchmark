# Dockerfile for native Tesseract benchmark (no Python overhead)
# Using Debian slim for glibc compatibility (same as pytesseract for fair comparison)

FROM debian:bookworm-slim

# Install Tesseract, Poppler (for PDF processing), and monitoring tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    imagemagick \
    poppler-utils \
    procps \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Create working directories
WORKDIR /app
RUN mkdir -p /app/input /app/output

# Copy benchmark script
COPY benchmark.sh /app/
RUN chmod +x /app/benchmark.sh

# Default command
CMD ["/app/benchmark.sh"]
