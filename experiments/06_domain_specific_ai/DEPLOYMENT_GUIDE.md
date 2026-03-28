# Deployment Guide - Standalone Web Application

## Overview

The Game Dev AI Assistant is now available as a **standalone web application** with:
- ✅ Beautiful web UI
- ✅ REST API
- ✅ Drag-and-drop file upload
- ✅ Real-time search (text and images)
- ✅ Docker deployment
- ✅ Multi-user ready

## Quick Start (Local)

### 1. Install Dependencies

```bash
pip install -r requirements_app.txt
```

### 2. Run Server

```bash
# Simple run
python app.py

# Or with uvicorn (recommended)
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access Web UI

Open browser: **http://localhost:8000**

That's it! 🎉

---

## Features

### 🔍 Search Tab
- **Text search**: Type any query to search across docs and images
- **Modality filter**: Search text-only, images-only, or both
- **Results**: Shows matching documents and images with similarity scores
- **Real-time**: Instant results with CLIP embeddings

### 📤 Upload Tab
- **Drag and drop**: Drag files directly to upload
- **Supported formats**:
  - Documents: `.md`, `.txt`
  - Images: `.png`, `.jpg`, `.jpeg`
- **Auto-indexing**: Files are immediately added to vector store

### 📚 Browse Tab
- **View all items**: See everything in your knowledge base
- **Preview**: Thumbnails for images, snippets for text
- **Stats**: Total items loaded

---

## Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# 1. Build and start
docker-compose up -d

# 2. View logs
docker-compose logs -f

# 3. Stop
docker-compose down
```

Access at: **http://localhost:8000**

### Option 2: Docker Only

```bash
# Build image
docker build -t gamedev-ai .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/vector_store_web:/app/vector_store_web \
  -v $(pwd)/uploads:/app/uploads \
  --name gamedev-ai-assistant \
  gamedev-ai

# View logs
docker logs -f gamedev-ai-assistant

# Stop
docker stop gamedev-ai-assistant
docker rm gamedev-ai-assistant
```

---

## Production Deployment

### Deploy to Cloud (AWS/GCP/Azure)

#### AWS (EC2 + Docker)

```bash
# 1. Launch EC2 instance (t3.medium or larger)
# 2. SSH into instance
# 3. Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose

# 4. Clone repository
git clone <your-repo>
cd experiments/06_domain_specific_ai

# 5. Run with docker-compose
sudo docker-compose up -d

# 6. Configure security group to allow port 8000
```

Access at: `http://<your-ec2-public-ip>:8000`

#### Azure (Container Instances)

```bash
# 1. Build and push to Azure Container Registry
az acr create --resource-group myResourceGroup --name gamedevai --sku Basic
az acr build --registry gamedevai --image gamedev-ai:v1 .

# 2. Deploy container
az container create \
  --resource-group myResourceGroup \
  --name gamedev-ai-assistant \
  --image gamedevai.azurecr.io/gamedev-ai:v1 \
  --cpu 2 \
  --memory 4 \
  --port 8000 \
  --dns-name-label gamedev-ai
```

Access at: `http://gamedev-ai.<region>.azurecontainer.io:8000`

#### Google Cloud (Cloud Run)

```bash
# 1. Build and push
gcloud builds submit --tag gcr.io/<project-id>/gamedev-ai

# 2. Deploy
gcloud run deploy gamedev-ai-assistant \
  --image gcr.io/<project-id>/gamedev-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

Access at: URL provided by Cloud Run

---

## API Endpoints

### GET /api/status
Get system status

**Response**:
```json
{
  "status": "ready",
  "vector_store_loaded": true,
  "num_items": 150,
  "message": "Ready with 150 items"
}
```

### POST /api/query
Text-based search

**Request**:
```json
{
  "query": "combat UI health bars",
  "k": 5,
  "modality": null  // or "text" or "image"
}
```

**Response**:
```json
{
  "query": "combat UI health bars",
  "results": [
    {
      "item": {
        "type": "image",
        "path": "screenshots/combat_ui.png",
        "caption": "Combat UI with health bars",
        "image_data": "data:image/png;base64,..."
      },
      "score": 0.892
    }
  ],
  "timestamp": "2024-03-26T10:30:00"
}
```

### POST /api/query-image
Image-based search (find similar images)

**Request**: multipart/form-data
- `file`: Image file
- `k`: Number of results (default: 5)
- `modality`: Filter ("text", "image", or null)

### POST /api/upload-document
Upload document or image

**Request**: multipart/form-data
- `file`: Document/image file
- `caption`: Caption for images (optional)

**Response**:
```json
{
  "status": "success",
  "filename": "combat_ui.png",
  "type": "image",
  "total_items": 151
}
```

### POST /api/build-from-directory
Build vector store from directory

**Request**:
```json
{
  "docs_dir": "/path/to/game_docs"
}
```

### GET /api/items?skip=0&limit=50
List items in vector store

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Vector Store
VECTOR_STORE_PATH=vector_store_web

# Upload
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE=10485760  # 10MB

# Optional: OpenAI API
OPENAI_API_KEY=sk-...
```

### Custom Configuration

Edit `app.py`:

```python
# Change vector store path
VECTOR_STORE_PATH = "my_custom_vector_store"

# Change upload directory
UPLOAD_DIR = Path("my_uploads")

# Change port (or use uvicorn --port flag)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## Security Considerations

### Production Checklist

- [ ] **Authentication**: Add API keys or OAuth
- [ ] **HTTPS**: Use SSL/TLS certificates
- [ ] **Rate limiting**: Prevent abuse
- [ ] **File validation**: Strict file type checking
- [ ] **CORS**: Restrict origins in production
- [ ] **Firewall**: Only expose necessary ports
- [ ] **Backups**: Regular vector store backups

### Adding Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/query")
async def query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate token
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")

    # ... rest of function
```

### Adding HTTPS

```bash
# Generate self-signed certificate (dev only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with HTTPS
uvicorn app:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

For production, use Let's Encrypt:

```bash
# Install certbot
sudo apt-get install certbot

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com

# Certificates will be in /etc/letsencrypt/live/yourdomain.com/
```

---

## Performance Tuning

### For Large Knowledge Bases (10,000+ items)

```python
# Use FAISS GPU index
import faiss

# In multimodal_assistant.py, change:
self.index = faiss.IndexFlatIP(dimension)

# To:
res = faiss.StandardGpuResources()
self.index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(dimension))
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed_text(text: str):
    return assistant.vector_store.embed_text(text)
```

### Multi-worker Deployment

```bash
# Run with multiple workers
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# Or with gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Monitoring & Logging

### Add Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in endpoints
@app.post("/api/query")
async def query(request: QueryRequest):
    logger.info(f"Query received: {request.query}")
    # ...
```

### Metrics with Prometheus

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics at: `http://localhost:8000/metrics`

---

## Backup & Restore

### Backup Vector Store

```bash
# Backup
tar -czf vector_store_backup_$(date +%Y%m%d).tar.gz vector_store_web/

# Restore
tar -xzf vector_store_backup_20240326.tar.gz
```

### Automated Backups (cron)

```bash
# Edit crontab
crontab -e

# Add daily backup at 2am
0 2 * * * cd /path/to/app && tar -czf backups/vector_store_$(date +\%Y\%m\%d).tar.gz vector_store_web/
```

---

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Out of memory
```bash
# Increase Docker memory
docker run -m 4g ...

# Or in docker-compose.yml:
services:
  gamedev-ai:
    mem_limit: 4g
```

### Slow queries
- Use FAISS GPU index
- Reduce embedding model size (CLIP-base vs CLIP-large)
- Enable caching
- Index fewer items initially

### Images not loading
- Check file permissions
- Ensure absolute paths
- Check CORS settings
- Verify image formats

---

## Next Steps

1. **Add your documentation**: Upload via web UI or mount directory
2. **Customize UI**: Edit HTML in `app.py` `get_html_ui()`
3. **Add authentication**: Implement API keys or OAuth
4. **Deploy to cloud**: Follow cloud deployment instructions
5. **Monitor**: Set up logging and metrics
6. **Backup**: Schedule automated backups

---

## Support

- **Documentation**: See `README.md` and `MULTIMODAL_GUIDE.md`
- **API Reference**: Visit `http://localhost:8000/docs` (auto-generated)
- **Issues**: Report bugs or request features

---

**Your standalone AI assistant is ready! 🚀**
