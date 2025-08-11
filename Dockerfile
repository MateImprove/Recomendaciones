FROM python:3.9-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Cloud Run usa PORT (por defecto 8080)
ENV PORT=8080
EXPOSE 8080

# Arranca Streamlit escuchando en 0.0.0.0 y en $PORT
CMD ["bash","-lc","streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true"]
