"""
Print-Utilities für PDF-Verarbeitung

Dieses Modul enthält Funktionen zur:
- Zusammenfassungs-Ausgaben
- Statistik-Displays
- Debug- und Status-Informationen
"""

from typing import List, Dict, Any
from llama_index.core import Document

def print_semantic_analysis_summary(semantic_chunks: List[Dict[str, Any]], documents: List[Document]):
    """
    Gibt eine Zusammenfassung der semantischen Analyse aus.
    Inkludiert Tabellen-Statistiken.
    
    Args:
        semantic_chunks: Liste der semantischen Chunks
        documents: Liste der erstellten Documents
    """
    # Basis-Statistiken
    total_chunks = len(semantic_chunks)
    table_chunks = [chunk for chunk in semantic_chunks if chunk.get('is_table', False)]
    text_chunks = [chunk for chunk in semantic_chunks if not chunk.get('is_table', False)]
    
    print(f"[SEMANTIC] {total_chunks} semantische Abschnitte erstellt:")
    print(f"  - Text-Abschnitte: {len(text_chunks)}")
    print(f"  - Tabellen-Abschnitte: {len(table_chunks)}")
    
    # Text-Chunks
    for i, chunk in enumerate(text_chunks[:3]):  # Nur erste 3 anzeigen
        print(f"    Text {i+1}: '{chunk['section']}' ({chunk['length']} Zeichen)")
    if len(text_chunks) > 3:
        print(f"    ... und {len(text_chunks)-3} weitere Text-Abschnitte")
    
    # Tabellen-Chunks
    if table_chunks:
        print(f"  [TABELLEN] Erkannte Tabellen:")
        for i, table in enumerate(table_chunks):
            quality = table.get('table_quality', 'unknown')
            rows = table.get('table_rows', '?')
            cols = table.get('table_cols', '?')
            method = table.get('table_extraction_method', 'unknown')
            print(f"    Tabelle {i+1}: {rows}x{cols} Zellen - {quality} Qualität ({method})")
    
    # Header-Integration Statistiken
    header_enriched_chunks = [d for d in documents if d.metadata.get('contains_headers')]
    h1_headers = set(d.metadata.get('section_h1') for d in documents if d.metadata.get('section_h1'))
    h2_headers = set(d.metadata.get('section_h2') for d in documents if d.metadata.get('section_h2'))
    h3_headers = set(d.metadata.get('section_h3') for d in documents if d.metadata.get('section_h3'))
    
    print(f"[HEADER] Header-Integration erfolgreich:")
    print(f"  - Chunks mit Header-Metadaten: {len(header_enriched_chunks)}/{len(documents)}")
    print(f"  - H1-Überschriften gefunden: {len(h1_headers)}")
    print(f"  - H2-Überschriften gefunden: {len(h2_headers)}")
    print(f"  - H3-Überschriften gefunden: {len(h3_headers)}")

def print_quality_summary(quality_metrics: Dict[str, Any]):
    """
    Gibt eine Zusammenfassung der Qualitätsvalidierung aus.
    
    Args:
        quality_metrics: Qualitäts-Metriken Dictionary
    """
    print(f"\n[QUALITÄT] Dokumenten-Qualitäts-Analyse:")
    print(f"  - Gesamt-Dokumente: {quality_metrics['total_documents']}")
    print(f"  - Text-Dokumente: {quality_metrics['text_documents']}")
    print(f"  - Tabellen-Dokumente: {quality_metrics['table_documents']}")
    print(f"  - Durchschnittliche Textlänge: {quality_metrics['avg_text_length']:.0f} Zeichen")
    print(f"  - Dokumente mit Seitennummern: {quality_metrics['documents_with_pages']}")
    print(f"  - Dokumente mit Headern: {quality_metrics['documents_with_headers']}")
    print(f"  - Einzigartige Abschnitte: {quality_metrics['unique_sections']}")
    print(f"  - Leere Dokumente: {quality_metrics['empty_documents']}")
    
    # Tabellen-spezifische Metriken
    if quality_metrics['table_documents'] > 0:
        print(f"  [TABELLEN-QUALITÄT]:")
        print(f"    - Gute Qualität: {quality_metrics['tables_good_quality']}")
        print(f"    - Mittlere Qualität: {quality_metrics['tables_medium_quality']}")
        print(f"    - Schlechte Qualität: {quality_metrics['tables_poor_quality']}")
        print(f"    - Erfolgsrate: {quality_metrics['table_extraction_success_rate']:.1%}")
        print(f"    - Ø Zeilen pro Tabelle: {quality_metrics['avg_table_rows']:.1f}")
        print(f"    - Ø Spalten pro Tabelle: {quality_metrics['avg_table_cols']:.1f}")
    
    # Qualitäts-Score
    print(f"  - Qualitäts-Score: {quality_metrics['quality_score']:.1f}/100")
    
    # Warnungen
    if quality_metrics['warnings']:
        print(f"  [WARNUNGEN]:")
        for warning in quality_metrics['warnings']:
            print(f"    ⚠️  {warning}")
    else:
        print(f"  ✅ Keine Qualitätsprobleme gefunden")

def print_processing_status(step: str, message: str, success: bool = True):
    """
    Gibt eine formatierte Status-Nachricht aus.
    
    Args:
        step: Verarbeitungsschritt
        message: Status-Nachricht
        success: True für Erfolg, False für Fehler
    """
    status_symbol = "✅" if success else "❌"
    print(f"{status_symbol} [{step}] {message}")

def print_element_statistics(elements: list):
    """
    Gibt Statistiken über PDF-Elemente aus.
    
    Args:
        elements: Liste von Unstructured-Elementen
    """
    element_types = {}
    for element in elements:
        element_type = type(element).__name__
        element_types[element_type] = element_types.get(element_type, 0) + 1
    
    print(f"[ELEMENT-STATISTIK] {len(elements)} PDF-Elemente gefunden:")
    for element_type, count in sorted(element_types.items()):
        print(f"  - {element_type}: {count}")

def print_header_distribution(documents: List[Document]):
    """
    Gibt die Verteilung der Header-Ebenen aus.
    
    Args:
        documents: Liste der Documents
    """
    h1_count = len([d for d in documents if d.metadata.get('section_h1')])
    h2_count = len([d for d in documents if d.metadata.get('section_h2')])
    h3_count = len([d for d in documents if d.metadata.get('section_h3')])
    
    print(f"[HEADER-VERTEILUNG]:")
    print(f"  - Dokumente mit H1-Zuordnung: {h1_count}")
    print(f"  - Dokumente mit H2-Zuordnung: {h2_count}")
    print(f"  - Dokumente mit H3-Zuordnung: {h3_count}") 