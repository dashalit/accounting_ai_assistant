# Docker Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- At least 2GB RAM available for Docker

## Quick Start

### 1. Clone and Configure

```bash
# Copy environment file
cp .env.example .env

# Edit configuration if needed
nano .env  # or use your preferred editor
```

### 2. Start the service

```bash
# Basic setup (API only)
docker-compose up -d

# With nginx reverse proxy
docker-compose --profile with-nginx up -d

# With all services (nginx + redis)
docker-compose --profile with-nginx --profile with-redis up -d
```

### 3. Check status

```bash
docker-compose ps
docker-compose logs -f api
```

### 4. Access the API

- API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- Nginx (if enabled): http://localhost:80

## API Usage Examples

### Upload and process a document

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

### Search documents

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 5,
    "use_mmr": true,
    "mmr_lambda": 0.7
  }'
```

### List processed documents

```bash
curl "http://localhost:8000/documents"
```

### Get statistics

```bash
curl "http://localhost:8000/stats"
```

### Delete a document

```bash
curl -X DELETE "http://localhost:8000/documents/{document_id}"
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `mock` | Embedding model to use |
| `CHUNK_SIZE` | `500` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `USE_FAISS` | `false` | Use FAISS for faster search |
| `VECTOR_DIMENSION` | `384` | Vector dimension for FAISS |

## Using Real Embedding Models

For production use with real embeddings:

1. Edit `.env`:
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
USE_FAISS=true
VECTOR_DIMENSION=384
```

2. Restart:
```bash
docker-compose down
docker-compose up -d --build
```

## Data Persistence

All data is stored in the `./data` directory:
- `./data/uploads/` - Uploaded documents
- `./data/store/` - Vector store index

## Stopping the service

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Troubleshooting

### Check logs
```bash
docker-compose logs api
docker-compose logs nginx  # if using nginx
```

### Restart service
```bash
docker-compose restart api
```

### Rebuild image
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Low memory error
Reduce memory limits in `docker-compose.yml` or increase Docker memory allocation.

## Production Deployment

For production:

1. Use a real embedding model
2. Enable FAISS for better performance
3. Use nginx reverse proxy
4. Set up SSL/TLS termination
5. Configure proper logging
6. Set up monitoring and alerts
7. Use secrets management for sensitive data
