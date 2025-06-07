# Dockerfile für PDF-Verarbeitung mit unstructured und GPU-Support
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    tesseract-ocr-deu \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir psutil pymupdf requests

# Setze CUDA-Umgebungsvariablen
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Volumen für PDF-Dateien und die ChromaDB
VOLUME /app/Test_PDF
VOLUME /app/chroma_db_store

# Skript kopieren
COPY parse_pdf.py .

# Ausführen des PDF-Parser-Skripts
CMD ["python", "parse_pdf.py"] 