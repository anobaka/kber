FROM python:3.11-slim

# Install git (needed for GitPython and repo cloning)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git openssh-client && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create default repos directory
RUN mkdir -p /data/repos

EXPOSE 8080

CMD ["python", "main.py"]
