docker compose down && docker compose up -d && docker compose logs -f


docker build -t testdockerpython:latest .
docker tag testdockerpython:latest budisentosa/testdockerpython:latest
docker tag testdockerpython:latest ghcr.io/budisentosa/testdockerpython:latest
docker push budisentosa/testdockerpython:latest
docker push ghcr.io/budisentosa/testdockerpython:latest

docker pull budisentosa/testdockerpython:latest

docker pull ghcr.io/budisentosa/testdockerpython:latest



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

### Using Docker Compose (Recommended)

```bash
# Start Jupyter Lab
docker-compose up -d

# View logs and get the access token
docker-compose logs

# Stop Jupyter Lab
docker-compose down

# Restart Jupyter Lab
docker-compose restart
```

Then open http://localhost:8888 in your browser.

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
├── docker-compose.yml              # Docker Compose configuration
├── deploy.sh                       # Deployment script for VM
├── requirements.txt                # Python dependencies
├── jupyter_notebook_config.py      # Jupyter configuration
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
├── notebooks/                      # Your Jupyter notebooks (created on first run)
└── README.md                       # This file
```

### Requirements

- Docker >= 20.10
- Docker Compose >= 1.29
- Docker Hub account (for pushing images)
- GitHub account (for version control)

## Deployment to Remote VM

### Step 1: Copy project to VM

From your Windows machine:
```bash
scp -r C:\Users\ThinkPad\Documents\testDockerPython gdt:~/
```

### Step 2: Connect to VM with port forwarding

```bash
ssh gdt
```

Your SSH config already includes `LocalForward 8888 localhost:8888`, which forwards the VM's port 8888 to your local machine.

### Step 3: Deploy with Docker Compose

On the VM:
```bash
cd ~/testDockerPython

# Make deploy script executable
chmod +x deploy.sh

# Start Jupyter Lab (builds image if needed)
./deploy.sh start

# Or use docker-compose directly
docker-compose up -d

# View logs to get the access token
docker-compose logs
```

### Step 4: Access from Windows

Open your browser on Windows and go to:
```
http://localhost:8888
```

Use the token from the logs to access Jupyter Lab.

### Useful Commands on VM

Using the deploy script:
```bash
./deploy.sh start    # Start Jupyter Lab
./deploy.sh stop     # Stop Jupyter Lab
./deploy.sh restart  # Restart Jupyter Lab
./deploy.sh logs     # View logs
./deploy.sh status   # Show container status
./deploy.sh build    # Build the image
```

Or using docker-compose directly:
```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build

# View running containers
docker-compose ps
```

## License

MIT
