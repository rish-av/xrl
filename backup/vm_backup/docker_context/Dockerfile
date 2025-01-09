# Base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    vim \
    nginx \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files
ADD configs.tar.gz /

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Expose necessary ports
EXPOSE 80

# Default command
CMD ["/bin/bash"]
