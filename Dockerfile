# Dockerfile
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
# Primero solo lo que define dependencias: aprovecha la cache de capas
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install .
FROM base AS dev
RUN pip install ".[dev]"
COPY tests/ tests/
CMD ["pytest", "-v"]
FROM base AS runtime
ENTRYPOINT ["psi-demo"]
CMD ["--salida", "/out/demo.png"]