"""
Zentrale Konfiguration für Docling Adapter
------------------------------------------
Alle Konfigurationsparameter für PDF-Verarbeitung, Chunking und Qualitätsanalyse.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


# Basis-Pfade
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

# Erstelle Verzeichnisse falls sie nicht existieren
for directory in [CACHE_DIR, LOGS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Haupt-Konfiguration für Docling
DOCLING_CONFIG = {
    # === Docling Core Settings ===
    "ocr_enabled": True,
    "table_extraction": True,
    "image_extraction": True,
    "formula_extraction": True,
    "layout_analysis": True,
    "reading_order": True,
    "preserve_formatting": True,
    "extract_metadata": True,
    
    # === Export-Einstellungen ===
    "export_format": "markdown",  # "markdown", "html", "json", "text"
    
    # === Chunking-Strategie ===
    "chunk_strategy": "hybrid",  # "by_heading", "by_page", "hybrid", "by_element"
    "chunk_by_page": False,
    "max_chunk_size": 1024,  # Optimiert für Embedding-Modelle
    "chunk_overlap": 100,
    "min_chunk_size": 50,
    
    # === Performance & Timeouts ===
    "timeout_seconds": 300,  # 5 Minuten pro PDF
    "max_workers": 4,  # Parallele Verarbeitung
    "enable_caching": True,
    "cache_ttl_hours": 24,
    
    # === Qualitäts-Schwellwerte ===
    "high_quality_threshold": 0.8,
    "low_quality_threshold": 0.3,
    "max_ocr_ratio": 0.7,
    
    # === Datei-Limits ===
    "max_file_size_mb": 500,
    "max_pages": 1000,
    
    # === Logging ===
    "log_level": "INFO",
    "log_to_file": True,
    "log_file": str(LOGS_DIR / "docling_adapter.log"),
    
    # === Concurrency & Locking ===
    "enable_file_locking": True,
    "lock_timeout_seconds": 60,
    "lock_retry_attempts": 3,
    "lock_retry_delay": 1.0,
}


# Entwicklungs-Konfiguration (schneller, weniger robust)
DEVELOPMENT_CONFIG = {
    **DOCLING_CONFIG,
    "timeout_seconds": 60,
    "max_workers": 2,
    "enable_caching": False,
    "log_level": "DEBUG",
    "ocr_enabled": False,  # Schneller für Tests
    "max_chunk_size": 512,
}


# Produktions-Konfiguration (robust, optimiert)
PRODUCTION_CONFIG = {
    **DOCLING_CONFIG,
    "timeout_seconds": 600,  # 10 Minuten
    "max_workers": 8,
    "enable_caching": True,
    "cache_ttl_hours": 72,
    "log_level": "INFO",
    "max_chunk_size": 1024,
    "chunk_overlap": 150,  # Mehr Overlap für besseren Recall
}


# Test-Konfiguration (minimal, deterministisch)
TEST_CONFIG = {
    **DOCLING_CONFIG,
    "timeout_seconds": 30,
    "max_workers": 1,
    "enable_caching": False,
    "log_level": "WARNING",
    "ocr_enabled": False,
    "table_extraction": False,
    "image_extraction": False,
    "max_chunk_size": 256,
    "chunk_overlap": 50,
}


def get_config(profile: str = "default") -> Dict[str, Any]:
    """
    Gibt Konfiguration für das angegebene Profil zurück.
    
    Args:
        profile: Konfigurations-Profil ("default", "development", "production", "test")
        
    Returns:
        Konfigurationsdictionary
    """
    configs = {
        "default": DOCLING_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "test": TEST_CONFIG,
    }
    
    if profile not in configs:
        raise ValueError(f"Unbekanntes Profil: {profile}. Verfügbar: {list(configs.keys())}")
    
    return configs[profile].copy()


def get_config_from_env() -> Dict[str, Any]:
    """
    Lädt Konfiguration basierend auf Umgebungsvariablen.
    
    Environment Variables:
        DOCLING_PROFILE: Konfigurations-Profil
        DOCLING_LOG_LEVEL: Log-Level override
        DOCLING_TIMEOUT: Timeout override
        DOCLING_MAX_WORKERS: Worker-Anzahl override
        
    Returns:
        Konfigurationsdictionary
    """
    profile = os.getenv("DOCLING_PROFILE", "default")
    config = get_config(profile)
    
    # Environment-Overrides
    if os.getenv("DOCLING_LOG_LEVEL"):
        config["log_level"] = os.getenv("DOCLING_LOG_LEVEL")
    
    if os.getenv("DOCLING_TIMEOUT"):
        try:
            config["timeout_seconds"] = int(os.getenv("DOCLING_TIMEOUT"))
        except ValueError:
            pass
    
    if os.getenv("DOCLING_MAX_WORKERS"):
        try:
            config["max_workers"] = int(os.getenv("DOCLING_MAX_WORKERS"))
        except ValueError:
            pass
    
    if os.getenv("DOCLING_CACHE_DIR"):
        global CACHE_DIR
        CACHE_DIR = Path(os.getenv("DOCLING_CACHE_DIR"))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    return config


# Chunk-Konfigurationen für verschiedene Embedding-Modelle
EMBEDDING_CONFIGS = {
    "sentence-transformers": {
        "max_chunk_size": 512,
        "chunk_overlap": 50,
        "optimal_size": 384,
    },
    "openai": {
        "max_chunk_size": 8192,
        "chunk_overlap": 200,
        "optimal_size": 1536,
    },
    "huggingface": {
        "max_chunk_size": 1024,
        "chunk_overlap": 100,
        "optimal_size": 768,
    },
}


def get_embedding_config(model_type: str = "sentence-transformers") -> Dict[str, int]:
    """
    Gibt optimale Chunk-Konfiguration für Embedding-Modell zurück.
    
    Args:
        model_type: Typ des Embedding-Modells
        
    Returns:
        Chunk-Konfiguration
    """
    return EMBEDDING_CONFIGS.get(model_type, EMBEDDING_CONFIGS["sentence-transformers"])


# Qualitäts-Profile
QUALITY_PROFILES = {
    "fast": {
        "enable_ocr_analysis": False,
        "enable_structure_analysis": True,
        "enable_content_analysis": False,
        "confidence_threshold": 0.5,
    },
    "balanced": {
        "enable_ocr_analysis": True,
        "enable_structure_analysis": True,
        "enable_content_analysis": True,
        "confidence_threshold": 0.7,
    },
    "thorough": {
        "enable_ocr_analysis": True,
        "enable_structure_analysis": True,
        "enable_content_analysis": True,
        "enable_technical_analysis": True,
        "confidence_threshold": 0.8,
    },
}


# Metadaten-Extraktions-Konfiguration
METADATA_CONFIG = {
    "extract_pdf_metadata": True,
    "extract_file_metadata": True,
    "extract_content_metadata": True,
    "extract_heuristic_metadata": True,
    
    # Sprach-Erkennung
    "enable_language_detection": True,
    "language_confidence_threshold": 0.8,
    
    # Norm-Erkennung (DIN, ISO, etc.)
    "enable_standard_detection": True,
    "standard_patterns": [
        r"DIN\s+(?:EN\s+)?(?:ISO\s+)?\d+(?:[-:]\d+)*(?::\d{4})?",
        r"ISO\s+\d+(?:[-:]\d+)*(?::\d{4})?",
        r"EN\s+\d+(?:[-:]\d+)*(?::\d{4})?",
    ],
    
    # DOI/ISBN-Erkennung
    "enable_identifier_detection": True,
    "doi_pattern": r"10\.\d{4,}/[^\s]+",
    "isbn_pattern": r"ISBN[-\s]?(?:97[89][-\s]?)?(?:\d[-\s]?){9}\d",
}


# Export der wichtigsten Konfigurationen
__all__ = [
    "DOCLING_CONFIG",
    "DEVELOPMENT_CONFIG", 
    "PRODUCTION_CONFIG",
    "TEST_CONFIG",
    "get_config",
    "get_config_from_env",
    "get_embedding_config",
    "QUALITY_PROFILES",
    "METADATA_CONFIG",
    "BASE_DIR",
    "CACHE_DIR",
    "LOGS_DIR",
    "TEMP_DIR",
] 