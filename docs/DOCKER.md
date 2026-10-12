# Docker Deployment Guide

Run Banko AI Assistant using Docker or Podman - no Python or pip installation required!

## Prerequisites

- **Docker** or **Podman** installed
- **AI Provider API Key** (OpenAI, AWS Bedrock, IBM Watsonx, or Google Gemini)
- **CockroachDB** (included in docker-compose, or use external database)

## Quick Start

### Option 1: Docker Compose with Built-in Database (Easiest)

This starts both CockroachDB and Banko AI in containers - perfect for testing and demos.

**This will start:**
- CockroachDB (database) on port 26257
- CockroachDB Admin UI on port 8080
- Banko AI web app on port 5000

1. **Create environment file:**

```bash
# Copy the example and edit with your credentials
cat > .env << 'EOF'
# Choose your AI provider
AI_SERVICE=openai

# OpenAI credentials
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini

# IBM Watsonx credentials (if using Watsonx)
# WATSONX_API_KEY=your_api_key_here
# WATSONX_PROJECT_ID=your_project_id_here
# WATSONX_MODEL_ID=openai/gpt-oss-120b

# AWS Bedrock credentials (if using AWS)
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_REGION=us-east-1

# Google Gemini credentials (if using Google)
# GOOGLE_PROJECT_ID=your_project_id
# GOOGLE_API_KEY=your_gemini_key

# Cache configuration (optional)
CACHE_SIMILARITY_THRESHOLD=0.75
CACHE_STRICT_MODE=true
CACHE_TTL_HOURS=24

# Generate a random secret key
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
EOF
```

2. **Start the services:**

```bash
# Start both CockroachDB and Banko AI
docker-compose up -d

# View logs
docker-compose logs -f banko-ai

# Stop services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

3. **Access the application:**
   - Web Interface: http://localhost:5000
   - CockroachDB Admin UI: http://localhost:8080

### Option 2: Standalone Mode (Use External CockroachDB)

Use this when you already have a CockroachDB instance running (local or cloud).

**Using docker-compose (recommended):**

```bash
# Use the standalone compose file
docker-compose -f docker-compose.standalone.yml up -d

# Set your database URL
export DATABASE_URL="cockroachdb://root@localhost:26257/defaultdb?sslmode=disable"

# Restart to apply
docker-compose -f docker-compose.standalone.yml restart
```

**Or pull and run directly:**

```bash
docker pull virag/banko-ai-assistant:latest
```

**Run with external CockroachDB:**

```bash
# Start CockroachDB separately (if not already running)
docker run -d \
  --name cockroachdb \
  -p 26257:26257 -p 8080:8080 \
  cockroachdb/cockroach:v25.4.6 \
  start-single-node --insecure

# Wait for CockroachDB to be ready
sleep 10

# Run Banko AI (connected to external CockroachDB)
docker run -d \
  --name banko-ai \
  -p 5000:5000 \
  -e DATABASE_URL="cockroachdb://root@host.docker.internal:26257/defaultdb?sslmode=disable" \
  -e AI_SERVICE="openai" \
  -e OPENAI_API_KEY="your_api_key" \
  -e SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" \
  --add-host=host.docker.internal:host-gateway \
  virag/banko-ai-assistant:latest
```

**Connect to Cloud CockroachDB:**

```bash
docker run -d \
  --name banko-ai \
  -p 5000:5000 \
  -e DATABASE_URL="cockroachdb://user:password@your-cluster.cockroachlabs.cloud:26257/banko_ai?sslmode=verify-full" \
  -e AI_SERVICE="openai" \
  -e OPENAI_API_KEY="your_api_key" \
  -e SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" \
  virag/banko-ai-assistant:latest
```

**Access the application:** http://localhost:5000

### Option 3: Build Multi-Architecture Image

Build for both AMD64 and ARM64 (for production deployment).

```bash
# Setup buildx (one-time)
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t virag/banko-ai-assistant:latest \
  --push \
  .

# Or build locally without pushing
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t banko-ai-assistant:local \
  --load \
  .
```

**Test locally before pushing:**

```bash
# Build for your architecture
docker build -t banko-ai-assistant:test .

# Test with docker-compose
docker-compose up -d
```

## Using with Podman

Podman is a Docker alternative that doesn't require root privileges.

```bash
# Using podman-compose (similar to docker-compose)
podman-compose up -d

# Or using podman directly
podman pull virag/banko-ai-assistant:latest
podman run -d \
  --name banko-ai \
  -p 5000:5000 \
  -e DATABASE_URL="your_database_url" \
  -e AI_SERVICE="watsonx" \
  -e WATSONX_API_KEY="your_api_key" \
  virag/banko-ai-assistant:latest
```

## Configuration

### Environment Variables

All configuration is done via environment variables. See the main [README.md](https://github.com/cockroachlabs-field/banko-ai-assistant/blob/main/README.md) for complete documentation.

**Required:**
- `DATABASE_URL` - CockroachDB connection string
- `AI_SERVICE` - AI provider (watsonx, openai, aws, gemini)
- Provider-specific API keys

**Optional:**
- `CACHE_SIMILARITY_THRESHOLD` - Cache threshold (default: 0.75)
- `CACHE_STRICT_MODE` - Strict cache matching (default: true)
- `CACHE_TTL_HOURS` - Cache TTL (default: 24)
- `DB_POOL_SIZE` - Connection pool size (default: 100)
- `FLASK_ENV` - Flask environment (default: production)

### Volume Mounts (Optional)

Mount volumes for persistent cache or custom configurations:

```bash
docker run -d \
  -v ./cache:/home/bankoai/.cache \
  -v ./config:/app/config \
  virag/banko-ai-assistant:latest
```

## Multi-Architecture Support

Images are built for both amd64 and arm64:

```bash
# Docker will automatically pull the correct architecture
docker pull virag/banko-ai-assistant:latest

# Explicitly specify platform
docker pull --platform linux/amd64 virag/banko-ai-assistant:latest
docker pull --platform linux/arm64 virag/banko-ai-assistant:latest
```

## Production Deployment

### Using External CockroachDB Cluster

```bash
docker run -d \
  --name banko-ai \
  -p 5000:5000 \
  -e DATABASE_URL="cockroachdb://user:password@load-balancer:26257/banko_ai?sslmode=verify-full" \
  -e AI_SERVICE="watsonx" \
  -e WATSONX_API_KEY="${WATSONX_API_KEY}" \
  -e WATSONX_PROJECT_ID="${WATSONX_PROJECT_ID}" \
  -e CACHE_STRICT_MODE="true" \
  -e FLASK_ENV="production" \
  -e SECRET_KEY="${SECRET_KEY}" \
  --restart unless-stopped \
  virag/banko-ai-assistant:latest
```

### Behind a Load Balancer

```bash
# Run multiple instances
docker run -d --name banko-ai-1 -p 5001:5000 ... virag/banko-ai-assistant:latest
docker run -d --name banko-ai-2 -p 5002:5000 ... virag/banko-ai-assistant:latest
docker run -d --name banko-ai-3 -p 5003:5000 ... virag/banko-ai-assistant:latest

# Configure your load balancer (nginx, haproxy, etc.) to distribute traffic
```

### Health Checks

The container health check hits `/api/health/live`, which answers as soon
as the server is listening, even while warm-up is still running.
`/api/health/ready` returns 200 once the schema and the demo personas are in
place, and `/api/health` shows every warm-up step:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' banko-ai

# Manual checks
curl http://localhost:5000/api/health/live
curl http://localhost:5000/api/health/ready
curl http://localhost:5000/api/health
```

### Reverse Proxy Configuration

Socket.IO defaults to same-origin connections. Standard reverse proxies (nginx, haproxy, Caddy) work without extra config because the underlying engine.io honors `X-Forwarded-Host` and `X-Forwarded-Proto`. If your proxy rewrites the `Host` header without forwarding the original, set `SOCKETIO_CORS_ORIGINS` to the external origin clients will use (for example, `https://banko.example.com`).

## Troubleshooting

### View Logs

```bash
# Docker Compose
docker-compose logs -f banko-ai

# Docker
docker logs -f banko-ai

# Podman
podman logs -f banko-ai
```

### Common Issues

**Container fails to start:**
```bash
# Check logs for errors
docker logs banko-ai

# Verify environment variables
docker exec banko-ai env | grep -E "DATABASE_URL|AI_SERVICE|WATSONX"
```

**Database connection errors:**
```bash
# Check if CockroachDB is running
docker ps | grep cockroach

# Test database connection from container
docker exec banko-ai curl -f cockroachdb:26257
```

**AI provider connection errors:**
```bash
# Verify API keys are set correctly
docker exec banko-ai env | grep API_KEY

# Check provider-specific logs
docker logs banko-ai | grep -i "watsonx\|openai\|bedrock\|gemini"
```

### Rebuilding the Image

```bash
# Force rebuild without cache
docker-compose build --no-cache

# Or with docker directly
docker build --no-cache -t banko-ai-assistant:local .
```

## Building and Publishing Images

For maintainers - the repository includes a build script for creating multi-architecture images.

### Using the Build Script (Recommended)

```bash
# Build locally for both amd64 and arm64
./build-docker.sh

# Build and push to Docker Hub with latest + timestamp tags
./build-docker.sh --push

# Build and push with custom tag
./build-docker.sh --push --tag v1.0.33

# Show help
./build-docker.sh --help
```

The build script automatically:
- Creates multi-architecture images (linux/amd64, linux/arm64)
- Tags with both `latest` and timestamp (e.g., `2025.11.04-1430`)
- Sets up buildx builder if needed
- Provides colorful output and error handling

### Manual Multi-Architecture Build

```bash
# Create and use buildx builder
docker buildx create --name multiarch-builder --use
docker buildx inspect --bootstrap

# Build multi-architecture image (local test)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t virag/banko-ai-assistant:latest \
  --load .

# Build and push to Docker Hub
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t virag/banko-ai-assistant:latest \
  -t virag/banko-ai-assistant:$(date +%Y.%m.%d-%H%M) \
  --push .
```

**Note**: When using `--load` (local build), Docker can only load one platform at a time (your current architecture). Use `--push` to publish both architectures to Docker Hub.

## Support

- **Issues**: https://github.com/cockroachlabs-field/banko-ai-assistant/issues
- **Documentation**: [README.md](https://github.com/cockroachlabs-field/banko-ai-assistant/blob/main/README.md)
- **Docker Hub**: https://hub.docker.com/r/virag/banko-ai-assistant

## Security Notes

1. **Never commit `.env` files** with sensitive credentials
2. **Use secrets management** in production (Docker secrets, Kubernetes secrets, etc.)
3. **Run containers as non-root** (built-in to the Dockerfile)
4. **Use SSL/TLS** for database connections in production (`sslmode=verify-full`)
5. **Generate random SECRET_KEY** for each deployment
6. **Keep images updated** for security patches

## License

MIT License - See [LICENSE](https://github.com/cockroachlabs-field/banko-ai-assistant/blob/main/LICENSE) file for details.
