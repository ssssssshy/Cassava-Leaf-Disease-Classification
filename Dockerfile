FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


ENV MODEL_CONFIG="/configs/yolo12.yaml"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

RUN uv pip uninstall opencv-python --python .venv

COPY src/ ./src/
COPY configs/ ./configs/


EXPOSE 8000

CMD [".venv/bin/uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]