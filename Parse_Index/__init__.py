"""
Parse_Index - Modulare PDF-Parsing und Indexierung für RAG-System

Dieses Paket bietet eine saubere, modulare Struktur für die PDF-Verarbeitung
mit Unstructured und LlamaIndex für semantische Retrieval-Systeme.

Hauptmodule:
- config: Konfigurationsvariablen
- utils: Hilfsfunktionen (GPU-Check, Requirements, etc.)
- unstructured_wrapper: Unstructured-Integration
- document_enhancer: Semantische Dokumentverarbeitung
- node_parsers: LlamaIndex Parser-Systeme
- main_parser: Hauptorchestrierung
"""

from .main_parser import main, load_and_process_pdfs
from .config import (
    PDF_FOLDER, 
    PERSIST_DIR, 
    COLLECTION_NAME, 
    EMBED_MODEL_NAME,
    CHUNK_SIZES_CONFIG
)
from .utils import check_embedding_requirements, check_gpu_status
from .node_parsers import create_hybrid_parser_system
from .document_enhancer import create_enhanced_documents_from_pdf
from .unstructured_wrapper import process_pdf_with_local_unstructured

__version__ = "1.0.0"
__author__ = "DIN_BOT Team"

# Hauptfunktionen für externe Nutzung
__all__ = [
    'main',
    'load_and_process_pdfs',
    'check_embedding_requirements',
    'check_gpu_status',
    'create_hybrid_parser_system',
    'create_enhanced_documents_from_pdf',
    'process_pdf_with_local_unstructured',
    'PDF_FOLDER',
    'PERSIST_DIR',
    'COLLECTION_NAME',
    'EMBED_MODEL_NAME',
    'CHUNK_SIZES_CONFIG'
] 