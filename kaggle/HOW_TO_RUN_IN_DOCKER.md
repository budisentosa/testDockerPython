# How to Run Jupyter Notebooks in Docker

This guide explains how to run the Jupyter notebooks in the `kaggle` folder using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed (usually comes with Docker Desktop)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

#### 1. Update docker-compose.yml to mount the kaggle folder

Edit the `docker-compose.yml` file in the project root and change the volumes section:

```yaml
volumes:
  # Change from ./notebooks to ./kaggle
  - ./kaggle:/home/jupyter/work
```

#### 2. Start Jupyter Lab

From the project root directory:

```bash
# Start the container
docker-compose up -d

# View logs to see the access URL
docker-compose logs
```

#### 3. Access Jupyter Lab

Open your browser and go to:
```
http://localhost:8888
```

The Jupyter Lab interface will open with all your kaggle notebooks ready to use.

#### 4. Stop Jupyter Lab

When you're done working:

```bash
docker-compose down
```

### Option 2: Using Docker directly

#### Pull the pre-built image:

```bash
docker pull ghcr.io/budisentosa/testdockerpython:latest
```

#### Run the container with kaggle folder mounted:

**On Linux/Mac:**
```bash
docker run -p 8888:8888 -v $(pwd)/kaggle:/home/jupyter/work ghcr.io/budisentosa/testdockerpython:latest
```

**On Windows (PowerShell):**
```powershell
docker run -p 8888:8888 -v ${PWD}/kaggle:/home/jupyter/work ghcr.io/budisentosa/testdockerpython:latest
```

**On Windows (Command Prompt):**
```cmd
docker run -p 8888:8888 -v %cd%/kaggle:/home/jupyter/work ghcr.io/budisentosa/testdockerpython:latest
```

## Project Structure Inside Container

When you access Jupyter Lab, you'll see this structure:

```
/home/jupyter/work/
├── customer-clustring/
│   ├── customer-clustring-using-pca.ipynb
│   ├── Customer DataSet.csv
│   ├── models/
│   └── ...
├── customer-segmentation/
│   ├── customer-segmentation-FIXED-COMBINED.ipynb
│   ├── customer-segmentation-eda-k-means-pca.ipynb
│   ├── bank_transactions.csv
│   ├── models/
│   └── ...
└── customer-segmentation-clustering/
    ├── customer-segmentation-clustering.ipynb
    ├── marketing_campaign.csv
    └── ...
```

## Included Libraries

The Docker image comes pre-installed with:

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Yellowbrick
- And more data science libraries

Check `requirements.txt` in the project root for the complete list.

## Working with Notebooks

### Creating New Notebooks

1. In Jupyter Lab, click the "+" button or File > New > Notebook
2. Select the Python 3 kernel
3. Your new notebook will be saved in the kaggle folder on your local machine

### Saving Work

All changes you make in Jupyter Lab are automatically saved to your local `kaggle` folder because of the Docker volume mount. This means:
- Your work persists even after stopping the container
- You can edit files both in Jupyter Lab and on your local machine
- Git can track all changes

## Troubleshooting

### Cannot access Jupyter Lab

Check if the container is running:
```bash
docker-compose ps
```

View container logs:
```bash
docker-compose logs -f
```

### Permission Issues

If you encounter permission issues on Linux/Mac:

```bash
# Give appropriate permissions to kaggle folder
chmod -R 755 kaggle
```

On Windows, the docker-compose.yml uses `user: "0:0"` to avoid permission issues.

### Port 8888 Already in Use

If port 8888 is already in use, you can change it in `docker-compose.yml`:

```yaml
ports:
  - "8889:8888"  # Use port 8889 on host instead
```

Then access Jupyter at `http://localhost:8889`

### Container Keeps Restarting

Check the logs for errors:
```bash
docker-compose logs
```

Verify that the kaggle folder exists and is accessible.

## Advanced Usage

### Custom Jupyter Configuration

The Jupyter configuration is stored in `jupyter_notebook_config.py`. You can modify:
- Authentication settings
- Port configuration
- File upload limits
- And more

### Installing Additional Python Packages

#### Temporary Installation (lost when container is recreated):

In a Jupyter notebook cell:
```python
!pip install package-name
```

#### Permanent Installation:

1. Add the package to `requirements.txt` in the project root
2. Rebuild the Docker image:
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

### Resource Limits

The docker-compose.yml sets resource limits:
- CPU: 2 cores (limit), 1 core (reservation)
- Memory: 4GB (limit), 2GB (reservation)

You can adjust these in the `docker-compose.yml` file if needed.

## Tips

1. Use Ctrl+S or Cmd+S to save your notebooks frequently
2. The Jupyter Lab interface supports drag-and-drop file uploads
3. You can use the built-in terminal in Jupyter Lab for command-line operations
4. All files in the kaggle folder are accessible from both Jupyter Lab and your local file system
5. Git changes are tracked in real-time, so you can commit your work normally

## Remote Development (Optional)

If you're running Docker on a remote VM and want to access it from your local machine:

1. Set up SSH port forwarding:
   ```bash
   ssh -L 8888:localhost:8888 user@remote-host
   ```

2. Start Jupyter Lab on the remote VM:
   ```bash
   docker-compose up -d
   ```

3. Access from your local browser at `http://localhost:8888`

## Additional Resources

- Main project README: `../README.md`
- Dockerfile: `../Dockerfile`
- Docker Compose config: `../docker-compose.yml`
- Python dependencies: `../requirements.txt`
