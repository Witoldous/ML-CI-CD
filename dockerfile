FROM python:3.10

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj pliki do kontenera
COPY . /app

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# Domyślna komenda uruchamiająca API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
