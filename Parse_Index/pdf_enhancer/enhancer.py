"""
Hauptmodul für semantische PDF-Verbesserung

Dieses Modul orchestriert die semantische Verarbeitung von PDF-Dokumenten
durch Koordination aller Teilmodule:
- Header-Analyse
- Tabellen-Verarbeitung  
- Semantische Abschnitts-Gruppierung
- Qualitäts-Validierung
"""

import os
from typing import List
from llama_index.core import Document

from .header_utils import analyze_hierarchical_headers
from .semantic_sections import group_elements_into_semantic_sections, convert_semantic_chunks_to_documents
from .print_utils import print_semantic_analysis_summary

def semantic_enhanced_pdf(pdf_path: str) -> List[Document]:
    """
    Erstellt semantisch sinnvolle Dokument-Chunks durch intelligente Gruppierung.
    
    Kombiniert zusammengehörige PDF-Elemente zu kohärenten Abschnitten,
    integriert hierarchische Header-Analyse und erkennt Tabellen für
    reichere Metadaten und bessere Retrieval-Qualität.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Liste von Document-Objekten mit semantischen Abschnitten und Tabellen
    """
    print(f"[SEMANTIC] Verarbeite {pdf_path} mit semantischer Abschnitts-Gruppierung und Tabellen-Erkennung")
    
    # 1. Lade PDF-Elemente mit Unstructured (hi_res für bessere Header- und Tabellen-Erkennung)
    try:
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # Geändert von "auto" zu "hi_res" für bessere Layout-Erkennung
            include_page_breaks=True,
            include_metadata=True,  # Zusätzliche Metadaten für bessere Header-Erkennung
            combine_text_under_n_chars=0,
            infer_table_structure=True  # Aktiviert Tabellen-Struktur-Erkennung
        )
        
        print(f"[SEMANTIC] {len(elements)} PDF-Elemente geladen (hi_res Strategie mit Tabellen-Erkennung)")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der PDF-Elemente: {str(e)}")
        return []
    
    # 2. Analysiere hierarchische Headers für alle Elemente
    print(f"[SEMANTIC] Starte hierarchische Header-Analyse für bessere Metadaten...")
    header_mapping = analyze_hierarchical_headers(elements)
    
    # 3. Gruppiere Elemente zu semantischen Abschnitten (inkl. Tabellen-Behandlung)
    semantic_chunks = group_elements_into_semantic_sections(elements)
    
    # 4. Konvertiere zu LlamaIndex Documents mit erweiterten Header-Metadaten
    documents = convert_semantic_chunks_to_documents(
        semantic_chunks, 
        header_mapping, 
        elements, 
        pdf_path
    )
    
    print_semantic_analysis_summary(semantic_chunks, documents)
    
    return documents