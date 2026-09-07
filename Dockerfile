FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt .

# Install CPU-only PyTorch
RUN uv pip install --system --no-cache \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN uv pip install --system --no-cache -r requirements.txt

COPY src ./src
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]