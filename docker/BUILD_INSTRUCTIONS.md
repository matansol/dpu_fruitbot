# Quick Build & Deploy Instructions

## Step 1: Analyze Your Models (Optional but Recommended)

```powershell
python .\docker\analyze_models.py
```

This will show you the size of each model. Keep only the model(s) you need for production.

## Step 2: Update .dockerignore

Edit `.dockerignore` and uncomment the model directories you DON'T need. For example:

```
# Keep only the latest/production model
models/fruitbot/20251118-115646_easy/
models/fruitbot/20251118-164559_easy/
# ... uncomment all except your production model
```

## Step 3: Build and Run with Docker Compose

```powershell
# Build and run
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

## Step 4: Test the Application

Open your browser to: `http://localhost:8000`

Check health: `http://localhost:8000/health`

## Step 5: Stop the Application

```powershell
docker-compose down
```

## Alternative: Build with Docker directly

```powershell
# Build
docker build -t fruitbot-app .

# Run
docker run -p 8000:8000 fruitbot-app

# Run with environment variable
docker run -p 8000:8000 -e AZURE_DATABASE_URI="your-db-uri" fruitbot-app
```

## What Files Are Included in the Docker Image?

The Docker image includes:
- `app.py` - Main application
- `dpu_clf.py` - DPU classifier functions
- `static/` - CSS, JS, images
- `templates/` - HTML templates
- `models/` - Your trained models (filtered by .dockerignore)
- `procgen/` - Procgen environment (with fruitbot game)

## What Files Are Excluded?

The `.dockerignore` file excludes:
- Training scripts (`train.py`, `evaluate.py`, `grid_search.py`)
- Jupyter notebooks (`.ipynb` files)
- Database folder (`DataBase/`)
- Tests (`tests/`)
- Documentation (most `.md` files)
- Build artifacts
- Virtual environments
- Unnecessary procgen games (all except fruitbot)

## Next Steps for Production

1. **Choose a Python version**: Currently using Python 3.10. Update in Dockerfile if needed.

2. **Minimize models**: Keep only 1-2 production models to reduce image size.

3. **Database configuration**: 
   - Create a `.env` file with your Azure Database URI
   - Or set environment variable in `docker-compose.yml`

4. **Deploy to Azure**:
   - Azure Container Apps
   - Azure App Service (Web App for Containers)
   - Azure Container Instances

5. **Security**:
   - Don't commit `.env` files with secrets
   - Use Azure Key Vault for production secrets
   - Consider using managed identities

## Troubleshooting

### "Port already in use"
Change the port in `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"
```

### "Module not found"
Add missing package to `docker/requirements.txt` and rebuild.

### "Out of disk space"
Remove old Docker images: `docker system prune -a`

### Build is slow
The first build takes longer. Subsequent builds use cache.
