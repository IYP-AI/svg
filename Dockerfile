FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# ------------------------------------------------------------
# Dependencies for scientific computing libraries (numpy / scipy / pandas / scikit-learn)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    curl \
    memcached \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------------------------------
# Install all pip dependencies
# ------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ------------------------------------------------------------
# Copy project files into the image
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# Expose port for visdom
# ------------------------------------------------------------
EXPOSE 3000

# ------------------------------------------------------------
# Container startup: run visdom + Jupyter + application
# ------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
