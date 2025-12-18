# Docker Setup for Fruitbot App

This directory contains Docker configuration files for containerizing the Fruitbot RL application.

## Files

- `Dockerfile` - Main Docker image definition
- `docker-compose.yml` - Docker Compose configuration for easy deployment
- `.dockerignore` - Excludes unnecessary files from Docker build
- `docker/requirements.txt` - Minimal Python dependencies for production

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```powershell
# Build and run the application
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the application
docker-compose down
```

The app will be available at `http://localhost:8000`

### Option 2: Using Docker directly

```powershell
# Build the image
docker build -t fruitbot-app .

# Run the container
docker run -p 8000:8000 fruitbot-app

# Run with environment variables
docker run -p 8000:8000 -e AZURE_DATABASE_URI="your-database-uri" fruitbot-app

# Run with volume mounts for models
docker run -p 8000:8000 -v ${PWD}/models:/app/models fruitbot-app
```

## Environment Variables

Create a `.env` file in the root directory with:

```env
AZURE_DATABASE_URI=your-database-connection-string
```

Or use SQLite (default):
```env
AZURE_DATABASE_URI=sqlite:///test.db
```

## Reducing Image Size

### Remove Unnecessary Models

The `.dockerignore` file has commented-out model directories. To reduce image size:

1. Identify which model(s) you need for production
2. Edit `.dockerignore` and uncomment the models you DON'T need
3. Rebuild the image

Example - keep only the latest model:
```dockerignore
# In .dockerignore, uncomment all except your production model:
models/fruitbot/20251118-115646_easy/
models/fruitbot/20251118-164559_easy/
# ... keep only one model uncommented
```

### Multi-stage Build (Advanced)

For even smaller images, consider using a multi-stage build to separate build dependencies from runtime dependencies.

## Troubleshooting

### OpenMP Library Error
The Dockerfile sets `KMP_DUPLICATE_LIB_OK=TRUE` to prevent OpenMP conflicts.

### Missing Dependencies
If you encounter import errors, check `docker/requirements.txt` and add any missing packages.

### Build Fails
- Ensure you have enough disk space
- Check that all required files exist (app.py, dpu_clf.py, templates/, static/, models/)
- Try building without cache: `docker-compose build --no-cache`

### Port Already in Use
If port 8000 is already in use, modify the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Use port 8080 on host
```

## Development vs Production

### Development
- Use volume mounts to avoid rebuilding on code changes
- Keep more models for testing
- Use SQLite for simplicity

### Production
- Remove unused models
- Use Azure Database URI
- Consider adding health checks and monitoring
- Use proper secrets management (Azure Key Vault, etc.)

## Health Check

The Dockerfile includes a health check. Customize it in `app.py` by adding:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## Next Steps

1. Test the Docker build locally
2. Optimize model selection in `.dockerignore`
3. Configure production database URI
4. Consider deploying to Azure Container Apps or Azure App Service
