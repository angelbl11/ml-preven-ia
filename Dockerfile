# Imagen base: Python slim para reducir el tamaño de la imagen final
FROM python:3.11-slim-buster

# Instalar dependencias del sistema necesarias para LightGBM y otras bibliotecas ML
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar solo el archivo de requisitos para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar las dependencias de producción sin caché para reducir el tamaño de la imagen
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que se ejecuta la aplicación (8080 es el puerto estándar para Cloud Run)
EXPOSE 8080

# Comando para iniciar la aplicación usando Uvicorn (ajustado para usar el módulo correcto)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]