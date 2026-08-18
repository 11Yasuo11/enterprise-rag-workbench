FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE alembic.ini ./
COPY migrations ./migrations
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .
COPY services/api ./services/api
EXPOSE 8000
