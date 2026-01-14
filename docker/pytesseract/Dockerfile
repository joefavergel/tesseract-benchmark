FROM python:3.11-slim-bookworm


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYDEVD_DISABLE_FILE_VALIDATION=1

# System dependencies
RUN apt-get update \
  && apt-get install -y build-essential \
  && apt-get install -y libpq-dev \
  && apt-get install -y telnet netcat-traditional \
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*

# Tesseract and PDF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --all-packages --frozen
ENV PATH=".venv/bin:$PATH" PYTHONUNBUFFERED=1

WORKDIR /app
RUN mkdir -p /app/input /app/output /app/metrics

COPY benchmark.py /app/

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

CMD ["uv", "run", "python", "benchmark.py"]