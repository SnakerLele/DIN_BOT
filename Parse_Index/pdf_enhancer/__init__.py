"""
PDF Enhancer Modul für das Parse_Index System

Dieses Modul bietet erweiterte PDF-Verarbeitung mit:
- Semantischer Abschnitts-Gruppierung  
- Header-Hierarchie-Analyse
- Tabellen-Erkennung und -Strukturierung
- Metadaten-Anreicherung
- Qualitäts-Validierung

Hauptfunktionen:
    semantic_enhanced_pdf: Erstellt semantisch sinnvolle Dokument-Chunks
    validate_document_quality: Validiert die Qualität der erstellten Dokumente
"""

from .enhancer import semantic_enhanced_pdf
from .quality import validate_document_quality, calculate_quality_score
from .header_utils import classify_header_level, analyze_hierarchical_headers
from .table_utils import convert_table_to_text_and_meta
from .semantic_sections import group_elements_into_semantic_sections, convert_semantic_chunks_to_documents

# Hauptexporte für einfache Nutzung
__all__ = [
    'semantic_enhanced_pdf',
    'validate_document_quality',
    'calculate_quality_score',
    'classify_header_level',
    'analyze_hierarchical_headers',
    'convert_table_to_text_and_meta',
    'group_elements_into_semantic_sections',
    'convert_semantic_chunks_to_documents'
]

# Version
__version__ = "1.0.0" 