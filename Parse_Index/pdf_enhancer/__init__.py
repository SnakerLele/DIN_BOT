"""
PDF Enhancer Modul für das Parse_Index System

Dieses Modul bietet erweiterte PDF-Verarbeitung mit:
- Semantischer Abschnitts-Gruppierung  
- Header-Hierarchie-Analyse
- Tabellen-Erkennung und -Strukturierung (Unstructured + img2table)
- Metadaten-Anreicherung
- Qualitäts-Validierung

Hauptfunktionen:
    semantic_enhanced_pdf: Erstellt semantisch sinnvolle Dokument-Chunks
    validate_document_quality: Validiert die Qualität der erstellten Dokumente
    enhanced_table_extraction: Robuste Tabellen-Extraktion mit img2table
"""

from .enhancer import semantic_enhanced_pdf
from .quality import validate_document_quality, calculate_quality_score
from .header_utils import classify_header_level, analyze_hierarchical_headers
from .table_utils import (
    convert_table_to_text_and_meta,
    extract_tables_with_img2table, 
    create_table_nodes_from_img2table,
    enhanced_table_extraction
)
from .semantic_sections import group_elements_into_semantic_sections, convert_semantic_chunks_to_documents

# img2table Import (optional, falls verfügbar)
try:
    from .img2table_utils import Img2TableExtractor, create_img2table_extractor
    IMG2TABLE_AVAILABLE = True
except ImportError:
    IMG2TABLE_AVAILABLE = False

# Hauptexporte für einfache Nutzung
__all__ = [
    'semantic_enhanced_pdf',
    'validate_document_quality',
    'calculate_quality_score',
    'classify_header_level',
    'analyze_hierarchical_headers',
    'convert_table_to_text_and_meta',
    'extract_tables_with_img2table',
    'create_table_nodes_from_img2table', 
    'enhanced_table_extraction',
    'group_elements_into_semantic_sections',
    'convert_semantic_chunks_to_documents'
]

# Füge img2table Exporte hinzu falls verfügbar
if IMG2TABLE_AVAILABLE:
    __all__.extend(['Img2TableExtractor', 'create_img2table_extractor'])

# Version
__version__ = "1.1.0"  # Version erhöht für img2table Integration 