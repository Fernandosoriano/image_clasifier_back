# Usa una imagen base de Python
FROM python:3.10


# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install -r requirements.txt

# Copia el código de la aplicación Flask
COPY . .

# Expone el puerto 5000 para la API
EXPOSE 5000

# Comando para ejecutar la aplicación Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
