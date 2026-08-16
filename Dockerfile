# Dockerfile cho Shopee Rating Prediction Demo
# Base: Python 3.11 slim (nhe, on dinh, tuong thich tot voi underthesea)

FROM python:3.11-slim

# Metadata
LABEL maintainer="Shopee Rating Prediction Team"
LABEL description="Demo Streamlit cho he thong du doan rating Shopee bang Linear/Ridge Regression"

# Tranh tao bytecode .pyc va buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Cai dependencies he thong (cho matplotlib, scikit-learn build neu can)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        fonts-dejavu-core \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Tao thu muc lam viec
WORKDIR /app

# Copy requirements truoc de tan dung Docker cache layer
COPY requirements.txt .

# Cai Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code (.dockerignore loai CSV, reports/, docs/, ...)
COPY . .

# Model artifacts (models/rating_model.pkl, rating_vectorizer.pkl, rating_metadata.json)
# duoc train san o host va commit vao repo, hoac mount qua volume trong docker-compose.yml.
# Khong train trong image -> build nhanh + image gon.

# Expose port Streamlit
EXPOSE 8501

# Healthcheck de docker biet app san sang
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Chay Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
