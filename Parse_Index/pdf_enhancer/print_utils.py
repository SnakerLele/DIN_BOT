"""
Print-Utilities für PDF-Verarbeitung

Dieses Modul enthält Funktionen zur:
- Zusammenfassungs-Ausgaben
- Statistik-Displays
- Debug- und Status-Informationen
"""

from typing import List, Dict, Any
from llama_index.core import Document

def print_semantic_analysis_summary(semantic_chunks: List[Dict[str, Any]], documents: List[Document], img2table_count: int = 0):
    """
    Gibt eine Zusammenfassung der semantischen Analyse aus.
    **ERWEITERT für img2table-Integration.**
    
    Args:
        semantic_chunks: Liste der semantischen Chunks
        documents: Liste der erstellten Documents
        img2table_count: Anzahl der img2table-Tabellen (neue Pipeline)  
    """
    # Pipeline-Aufteilung
    img2table_docs = [d for d in documents if d.metadata.get('extraction_method') == 'img2table']
    unstructured_docs = [d for d in documents if d.metadata.get('extraction_method') != 'img2table']
    
    # Basis-Statistiken für unstructured-Chunks
    total_chunks = len(semantic_chunks)
    table_chunks = [chunk for chunk in semantic_chunks if chunk.get('is_table', False)]
    text_chunks = [chunk for chunk in semantic_chunks if not chunk.get('is_table', False)]
    
    print(f"[ENHANCED PIPELINE] 🎯 Zusammenfassung der img2table-priorisierten Verarbeitung:")
    print(f"")
    print(f"📊 IMG2TABLE-TABELLEN (PRIORITÄT):")
    print(f"  - Extrahierte Tabellen: {len(img2table_docs)}")
    if img2table_docs:
        # img2table Qualitäts-Statistiken
        quality_stats = {}
        total_img2table_cells = 0
        for doc in img2table_docs:
            quality = doc.metadata.get('table_quality', 'unknown')
            quality_stats[quality] = quality_stats.get(quality, 0) + 1
            rows = doc.metadata.get('table_rows', 0)
            cols = doc.metadata.get('table_columns', 0)
            total_img2table_cells += (rows * cols)
        
        print(f"  - Qualitäts-Verteilung:")
        for quality, count in quality_stats.items():
            print(f"    • {quality}: {count} Tabellen")
        print(f"  - Gesamt-Zellen erkannt: {total_img2table_cells}")
        
        # Erste 3 img2table-Tabellen anzeigen
        for i, doc in enumerate(img2table_docs[:3]):
            page = doc.metadata.get('page_number', '?')
            rows = doc.metadata.get('table_rows', 0)
            cols = doc.metadata.get('table_columns', 0)
            quality = doc.metadata.get('table_quality', 'unknown')
            print(f"    📊 img2table-Tabelle {i+1}: Seite {page}, {rows}x{cols}, {quality} Qualität")
        if len(img2table_docs) > 3:
            print(f"    ... und {len(img2table_docs)-3} weitere img2table-Tabellen")
    else:
        print(f"  - Keine Tabellen mit img2table gefunden")
    
    print(f"")
    print(f"📄 UNSTRUCTURED-CHUNKS (SEMANTISCHE ABSCHNITTE + FALLBACK):")
    print(f"  - Semantische Abschnitte: {total_chunks}")
    print(f"  - Text-Abschnitte: {len(text_chunks)}")
    print(f"  - Fallback-Tabellen: {len(table_chunks)}")
    
    # Text-Chunks
    for i, chunk in enumerate(text_chunks[:3]):  # Nur erste 3 anzeigen
        print(f"    📋 Text {i+1}: '{chunk['section']}' ({chunk['length']} Zeichen)")
    if len(text_chunks) > 3:
        print(f"    ... und {len(text_chunks)-3} weitere Text-Abschnitte")
    
    # Unstructured-Tabellen (Fallback)
    if table_chunks:
        print(f"  [FALLBACK-TABELLEN] Unstructured-Tabellen als Backup:")
        for i, table in enumerate(table_chunks):
            quality = table.get('table_quality', 'unknown')
            rows = table.get('table_rows', '?')
            cols = table.get('table_cols', '?')
            method = table.get('table_extraction_method', 'unknown')
            print(f"    📊 Fallback-Tabelle {i+1}: {rows}x{cols} Zellen - {quality} Qualität ({method})")
    
    # Header-Integration Statistiken (nur für unstructured-Dokumente)
    header_enriched_chunks = [d for d in unstructured_docs if d.metadata.get('contains_headers')]
    h1_headers = set(d.metadata.get('section_h1') for d in unstructured_docs if d.metadata.get('section_h1'))
    h2_headers = set(d.metadata.get('section_h2') for d in unstructured_docs if d.metadata.get('section_h2'))
    h3_headers = set(d.metadata.get('section_h3') for d in unstructured_docs if d.metadata.get('section_h3'))
    
    print(f"")
    print(f"[HEADER-INTEGRATION] Hierarchische Header-Analyse (unstructured):")
    print(f"  - Chunks mit Header-Metadaten: {len(header_enriched_chunks)}/{len(unstructured_docs)}")
    print(f"  - H1-Überschriften: {len(h1_headers)}")
    print(f"  - H2-Überschriften: {len(h2_headers)}")
    print(f"  - H3-Überschriften: {len(h3_headers)}")
    
    # Gesamt-Statistiken
    total_docs = len(documents)
    print(f"")
    print(f"🎯 PIPELINE-ERFOLG:")
    print(f"  - img2table-Chunks: {len(img2table_docs)} hochwertige Tabellen")
    print(f"  - Unstructured-Chunks: {len(unstructured_docs)} semantische Abschnitte")
    print(f"  - Gesamt-Documents: {total_docs} bereite für RAG-Indexierung")
    print(f"  - Tabellen-Priorität: img2table > unstructured (Fallback)")
    print(f"  - Erfolgsrate: {((len(img2table_docs) + len(unstructured_docs)) / max(total_docs, 1)) * 100:.1f}%")

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