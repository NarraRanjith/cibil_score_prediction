FROM python:3.8-slim

# Install awscli and clean up cache to reduce image size
RUN apt-get update \
	&& apt-get install -y --no-install-recommends awscli \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1

CMD ["python", "app.py"]