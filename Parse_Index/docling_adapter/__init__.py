"""
Docling Adapter für Parse_Index System

Moderne PDF-Verarbeitung mit Docling anstelle von img2table und unstructured.
Bietet erweiterte Layout-Analyse, Tabellen-Extraktion und OCR-Unterstützung.

Hauptkomponenten:
- DoclingAdapter: Zentrale Klasse für PDF → LlamaIndex Document Konvertierung
- QualityAnalyzer: Bewertung der PDF-Qualität und OCR-Bedarf
- SectionExtractor: Semantische Abschnittsbildung
- MetadataExtractor: Extraktion von Titel, Autor, DOI etc.
"""

from .adapter import DoclingAdapter
from .quality import QualityAnalyzer
from .sectionizer import SectionExtractor
from .metadata import MetadataExtractor
from .utils import setup_logging, validate_docling_config

__version__ = "1.0.0"
__author__ = "DIN_BOT Team"

# Hauptexporte für einfache Verwendung
__all__ = [
    "DoclingAdapter",
    "QualityAnalyzer", 
    "SectionExtractor",
    "MetadataExtractor",
    "setup_logging",
    "validate_docling_config"
]

# Convenience-Funktion für schnelle PDF-Verarbeitung
def parse_pdf_to_documents(pdf_path: str, **kwargs):
    """
    Schnelle PDF-zu-LlamaIndex-Documents Konvertierung.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        **kwargs: Zusätzliche Konfigurationsparameter
        
    Returns:
        List[Document]: Liste von LlamaIndex Document-Objekten
    """
    adapter = DoclingAdapter(**kwargs)
    return adapter.parse_pdf(pdf_path) 