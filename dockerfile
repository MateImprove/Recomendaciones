# Utiliza una imagen base de Python 3.9 ligera para el contenedor
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos de tu carpeta actual (incluyendo app.py, requirements.txt, etc.)
# al directorio de trabajo en el contenedor
COPY . .

# Instala todas las librerías listadas en requirements.txt
RUN pip install -r requirements.txt

# Expone el puerto que Streamlit usará
EXPOSE 8501

# Define el comando para iniciar Streamlit usando la variable de entorno PORT
# Esto corrige el error de "misconfigured port"
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]