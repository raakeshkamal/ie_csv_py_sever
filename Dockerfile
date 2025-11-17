# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Install uv (from official docs: https://docs.astral.sh/uv/guides/integration/docker/)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy the pyproject.toml and uv.lock files into the container
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8800 (since docker-compose.yml maps to 8800)
EXPOSE 8800

# Run the application with uvicorn
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8800"]
