# Jupyter Notebook Configuration

# IP configuration
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

# Allow connections from all interfaces
c.NotebookApp.allow_origin = '*'
c.NotebookApp.allow_remote_access = True

# Token-based authentication (no password required in container)
# Token will be generated automatically and shown in logs
c.NotebookApp.token = ''
c.NotebookApp.password = ''

# Trust notebooks (for cached outputs)
c.NotebookApp.trust_xheaders = True

# Increase max request size for large file uploads (100MB)
c.NotebookApp.max_buffer_size = 100 * 1024 * 1024
