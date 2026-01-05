# testDockerPython

A containerized Jupyter Lab environment with Python 3.11.9 and data science libraries (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Plotly).

## Features

- Python 3.11.9
- Jupyter Lab
- Pre-installed data science stack:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Plotly
  - Yellowbrick
  - And more...

## Quick Start

### Using Docker

#### Build the image locally:
```bash
docker build -t testdockerpython:latest .
```

#### Run the container:
```bash
docker run -p 8888:8888 -v $(pwd)/notebooks:/home/jupyter/work testdockerpython:latest
```

Then open http://localhost:8888 in your browser.

### Using Docker Hub image:
```bash
docker run -p 8888:8888 -v $(pwd)/notebooks:/home/jupyter/work budisentosa/testdockerpython:latest
```

### Using GitHub Packages image:
```bash
docker run -p 8888:8888 -v $(pwd)/notebooks:/home/jupyter/work ghcr.io/budisentosa/testdockerpython:latest
```

## Configuration

### Environment Variables

- `JUPYTER_ENABLE_LAB=1`: Enable Jupyter Lab (enabled by default)

### Jupyter Token

The Jupyter Lab instance is configured without authentication for development. For production use, modify `jupyter_notebook_config.py` to add password protection.

## Volume Mounting

Mount your local notebooks directory to persist your work:
```bash
docker run -p 8888:8888 -v /path/to/notebooks:/home/jupyter/work testdockerpython:latest
```

## Publishing Images

### Pushing to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag testdockerpython:latest budisentosa/testdockerpython:latest
docker tag testdockerpython:latest budisentosa/testdockerpython:1.0.0

# Push to Docker Hub
docker push budisentosa/testdockerpython:latest
docker push budisentosa/testdockerpython:1.0.0
```

### Pushing to GitHub Packages (GitHub Container Registry)

1. **Create a Personal Access Token**:
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `write:packages`, `read:packages`
   - Copy the token

2. **Login to GitHub Container Registry**:
```bash
docker login ghcr.io -u budisentosa
# When prompted for password, paste your Personal Access Token
```

3. **Tag and push the image**:
```bash
# Tag the image
docker tag testdockerpython:latest ghcr.io/budisentosa/testdockerpython:latest
docker tag testdockerpython:latest ghcr.io/budisentosa/testdockerpython:1.0.0

# Push to GitHub Container Registry
docker push ghcr.io/budisentosa/testdockerpython:latest
docker push ghcr.io/budisentosa/testdockerpython:1.0.0
```

4. **Make the package public** (optional):
   - Go to https://github.com/budisentosa?tab=packages
   - Click on the package `testdockerpython`
   - Go to "Package settings"
   - Scroll down and click "Change visibility"
   - Select "Public"

## Development

### Project Structure

```
testDockerPython/
├── Dockerfile                      # Container specification
├── requirements.txt                # Python dependencies
├── jupyter_notebook_config.py      # Jupyter configuration
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
└── README.md                       # This file
```

### Requirements

- Docker >= 20.10
- Docker Hub account (for pushing images)
- GitHub account (for version control)

## License

MIT
