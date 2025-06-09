"""
Konfigurationsvariablen für das Parse_Index System

Zentrale Stelle für alle Konfigurationsparameter des PDF-Parsing 
und Indexierungs-Systems.
"""

# Verzeichnis-Konfiguration
PDF_FOLDER = "./PDFs"  # Ordner für PDF-Dateien
PERSIST_DIR = "./chroma_db_store"  # Persistente Speicherung der Vektordatenbank
COLLECTION_NAME = "test_collection"  # Name der ChromaDB Collection

# Embedding-Model Konfiguration
# Bewährtes multilinguales Embedding-Modell für deutsche Inhalte
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Parser-Konfiguration für optimale Chunk-Größen
# Diese Werte wurden für semantische Qualität und Retrieval-Performance optimiert
CHUNK_SIZES_CONFIG = {
    "window_size": 8,           # Anzahl Kontext-Sätze für Sentence Window
    "chunk_size": 1024,         # Standard-Chunk-Größe (Zeichen)
    "chunk_overlap": 200,       # Überlappung zwischen Chunks (Zeichen)
    "chunk_size_small": 512,    # Kleine Chunks für detaillierte Suche
    "chunk_size_large": 2048,   # Große Chunks für umfassenden Kontext
}

# Unstructured-Konfiguration
UNSTRUCTURED_STRATEGIES = {
    "primary": "auto",      # Hauptstrategie: Automatische Wahl
    "fallback": "hi_res",   # Fallback bei Problemen: High-Resolution OCR
    "fast": "fast"          # Schnelle Alternative falls verfügbar
}

# ChromaDB-Konfiguration
CHROMA_CONFIG = {
    "collection_name": COLLECTION_NAME,
    "embedding_function": "sentence-transformers",  # Kompatibilität mit rag_api.py
    "distance_metric": "cosine",  # Kosinus-Ähnlichkeit für semantische Suche
    "metadata_fields": [
        "file_path", "filename", "page_number", 
        "section_h1", "section_h2", "section_h3", 
        "current_section", "element_type"
    ]
}

# Performance-Konfiguration
PERFORMANCE_CONFIG = {
    "enable_gpu": True,                 # GPU-Beschleunigung aktivieren
    "batch_size_embedding": 32,         # Batch-Größe für Embedding-Berechnung
    "max_workers": 4,                   # Maximale Worker für parallele Verarbeitung
    "memory_limit_mb": 8192,            # Speicherlimit in MB
    "timeout_per_pdf_seconds": 3600,    # Timeout pro PDF in Sekunden
}

# Logging-Konfiguration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    "show_progress": True,
    "detailed_stats": True
} 