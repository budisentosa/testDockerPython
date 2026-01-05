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
docker run -p 8888:8888 -v $(pwd)/notebooks:/home/jupyter/work <docker_username>/testdockerpython:latest
```

Replace `<docker_username>` with the actual Docker Hub username.

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

## Building for Docker Hub

```bash
# Login to Docker Hub
docker login

# Build with tag
docker build -t <docker_username>/testdockerpython:1.0.0 .

# Tag latest version
docker tag <docker_username>/testdockerpython:1.0.0 <docker_username>/testdockerpython:latest

# Push to Docker Hub
docker push <docker_username>/testdockerpython:1.0.0
docker push <docker_username>/testdockerpython:latest
```

Replace `<docker_username>` with your Docker Hub username.

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
