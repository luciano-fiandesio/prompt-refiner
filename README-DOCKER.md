# Docker Setup for Prompt Refinement Application

## Prerequisites

- Docker and Docker Compose installed on your system
- A Google API key for Gemini models

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

2. **Build and run the services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the applications:**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

## Docker Commands

### Start services
```bash
# Build and start
docker-compose up --build

# Start in background
docker-compose up -d

# Rebuild after code changes
docker-compose up --build --force-recreate
```

### Stop services
```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove everything including volumes
docker-compose down -v
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f streamlit
```

### Access container shell
```bash
# API container
docker exec -it prompt-refinement-api /bin/bash

# Streamlit container
docker exec -it prompt-refinement-ui /bin/bash
```

## Development

### Using docker-compose.override.yml

Create a `docker-compose.override.yml` for development settings:

```yaml
version: '3.8'

services:
  api:
    volumes:
      - ./:/app
    command: ["uv", "run", "python", "run_api.py"]

  streamlit:
    volumes:
      - ./:/app
```

This allows hot-reloading during development.

## Troubleshooting

### Port already in use
If you get port conflicts, you can change the ports in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change host port to 8001
```

### API connection issues
If the Streamlit app can't connect to the API, ensure both services are on the same network and the API URL in `streamlit_client.py` is set to `http://api:8000` when running in Docker.

### Memory issues
If you encounter memory issues, you can limit container resources:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

## Production Considerations

For production deployment:

1. Use specific image tags instead of `latest`
2. Set up proper secrets management (don't use .env files)
3. Configure health checks and restart policies
4. Use a reverse proxy (nginx/traefik) for SSL
5. Set up logging aggregation
6. Configure resource limits
7. Use multi-stage builds to minimize image size