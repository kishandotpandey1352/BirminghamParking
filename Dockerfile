FROM node:22-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY parkpulse parkpulse
COPY frontend frontend
COPY --from=frontend-build /frontend/dist frontend/dist
COPY run.py .
COPY parking_birmingham parking_birmingham
COPY artifacts artifacts
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
ENV PORT=8080
EXPOSE 8080
CMD ["python", "run.py"]