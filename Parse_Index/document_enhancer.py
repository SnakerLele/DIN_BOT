"""
Document Enhancer für das Parse_Index System

Dieses Modul enthält Funktionen zur semantischen Anreicherung und Verbesserung
von PDF-Dokumenten durch:
- Hierarchische Header-Analyse
- Semantische Abschnitts-Gruppierung  
- Metadaten-Anreicherung
- Strukturelle Dokumentenverbesserung
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from llama_index.core import Document

def semantic_enhanced_pdf(pdf_path: str) -> List[Document]:
    """
    Erstellt semantisch sinnvolle Dokument-Chunks durch intelligente Gruppierung.
    
    Kombiniert zusammengehörige PDF-Elemente zu kohärenten Abschnitten und
    integriert hierarchische Header-Analyse für reichere Metadaten.
    
    Args:
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Liste von Document-Objekten mit semantischen Abschnitten
    """
    print(f"[SEMANTIC] Verarbeite {pdf_path} mit semantischer Abschnitts-Gruppierung")
    
    # 1. Lade PDF-Elemente mit Unstructured (hi_res für bessere Header-Erkennung)
    try:
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # Geändert von "auto" zu "hi_res" für bessere Layout-Erkennung
            include_page_breaks=True,
            include_metadata=True,  # Zusätzliche Metadaten für bessere Header-Erkennung
            combine_text_under_n_chars=0
        )
        
        print(f"[SEMANTIC] {len(elements)} PDF-Elemente geladen (hi_res Strategie)")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der PDF-Elemente: {str(e)}")
        return []
    
    # 2. Analysiere hierarchische Headers für alle Elemente
    print(f"[SEMANTIC] Starte hierarchische Header-Analyse für bessere Metadaten...")
    header_mapping = analyze_hierarchical_headers(elements)
    
    # 3. Gruppiere Elemente zu semantischen Abschnitten
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

def group_elements_into_semantic_sections(elements: list) -> List[Dict[str, Any]]:
    """
    Gruppiert PDF-Elemente zu semantisch zusammengehörigen Abschnitten.
    
    Args:
        elements: Liste von Unstructured-PDF-Elementen
        
    Returns:
        Liste von semantischen Chunk-Dictionaries
    """
    semantic_chunks = []
    current_chunk = ""
    current_section = None
    current_page = None
    current_element_indices = []
    
    for i, element in enumerate(elements):
        element_text = str(element).strip()
        element_type = type(element).__name__
        
        # Extrahiere Seitennummer
        page_number = None
        if hasattr(element, 'metadata') and element.metadata and hasattr(element.metadata, 'page_number'):
            page_number = element.metadata.page_number
        
        # Erkenne Abschnittswechsel (Title-Elemente mit ausreichender Länge)
        is_new_section = (element_type == "Title" and len(element_text) > 3)
        
        if is_new_section:
            # Speichere den vorherigen Chunk wenn er Inhalt hat
            if current_chunk.strip() and len(current_chunk.strip()) > 50:
                semantic_chunks.append({
                    'text': current_chunk.strip(),
                    'section': current_section or "Unnamed Section",
                    'page_number': current_page,
                    'length': len(current_chunk.strip()),
                    'element_indices': current_element_indices.copy()
                })
            
            # Starte neuen Chunk
            current_section = element_text
            current_page = page_number
            current_chunk = element_text + "\n\n"
            current_element_indices = [i]
        else:
            # Füge zum aktuellen Chunk hinzu
            if element_text and len(element_text.strip()) > 2:
                current_chunk += element_text + " "
                current_element_indices.append(i)
                # Update page number if available
                if page_number and not current_page:
                    current_page = page_number
    
    # Letzten Chunk speichern
    if current_chunk.strip() and len(current_chunk.strip()) > 50:
        semantic_chunks.append({
            'text': current_chunk.strip(),
            'section': current_section or "Final Section",
            'page_number': current_page,
            'length': len(current_chunk.strip()),
            'element_indices': current_element_indices.copy()
        })
    
    return semantic_chunks

def convert_semantic_chunks_to_documents(
    semantic_chunks: List[Dict[str, Any]], 
    header_mapping: Dict[int, Dict[str, Any]], 
    elements: list, 
    pdf_path: str
) -> List[Document]:
    """
    Konvertiert semantische Chunks zu LlamaIndex Documents mit Header-Metadaten.
    
    Args:
        semantic_chunks: Liste von semantischen Chunk-Dictionaries
        header_mapping: Mapping von Element-Index zu Header-Metadaten
        elements: Ursprüngliche Unstructured-Elemente
        pdf_path: Pfad zur PDF-Datei
        
    Returns:
        Liste von Document-Objekten mit angereicherten Metadaten
    """
    documents = []
    filename = os.path.basename(pdf_path)
    
    for i, chunk in enumerate(semantic_chunks):
        # Basis-Metadaten
        metadata = {
            "file_path": pdf_path,
            "file_directory": os.path.dirname(pdf_path),
            "filename": filename,
            "section_title": chunk['section'],
            "chunk_index": i,
            "semantic_chunk": True
        }
        
        # Füge Seitennummer hinzu wenn verfügbar
        if chunk['page_number'] is not None:
            metadata["page_number"] = chunk['page_number']
        
        # Integriere hierarchische Header-Metadaten
        chunk_header_info = extract_header_info_for_chunk(
            chunk['element_indices'], 
            header_mapping, 
            elements
        )
        
        # Füge Header-Metadaten zu den Dokument-Metadaten hinzu
        metadata.update(chunk_header_info)
        
        document = Document(
            text=chunk['text'],
            metadata=metadata
        )
        documents.append(document)
    
    return documents

def extract_header_info_for_chunk(
    element_indices: List[int], 
    header_mapping: Dict[int, Dict[str, Any]], 
    elements: list
) -> Dict[str, Any]:
    """
    Extrahiert Header-Informationen für einen spezifischen Chunk.
    
    Args:
        element_indices: Liste der Element-Indices im Chunk
        header_mapping: Mapping von Element-Index zu Header-Metadaten
        elements: Ursprüngliche Unstructured-Elemente
        
    Returns:
        Dictionary mit Header-Metadaten für den Chunk
    """
    chunk_header_info = {
        'section_h1': None,
        'section_h2': None, 
        'section_h3': None,
        'current_section': None,
        'header_elements': []
    }
    
    for element_idx in element_indices:
        if element_idx in header_mapping:
            element_headers = header_mapping[element_idx]
            
            # Sammle Header-Informationen (verwende die letzten gefundenen)
            for key in ['section_h1', 'section_h2', 'section_h3', 'current_section']:
                if element_headers.get(key):
                    chunk_header_info[key] = element_headers[key]
            
            # Sammle Header-Elemente
            if element_headers.get('is_header'):
                chunk_header_info['header_elements'].append({
                    'text': str(elements[element_idx]).strip(),
                    'type': element_headers.get('element_type'),
                    'index': element_idx
                })
    
    # Zusätzliche Header-Statistiken
    header_count = len(chunk_header_info['header_elements'])
    metadata_additions = {
        'header_count': header_count,
        'contains_headers': header_count > 0
    }
    
    if chunk_header_info['header_elements']:
        metadata_additions['first_header'] = chunk_header_info['header_elements'][0]['text']
    
    # Entferne komplexe Datenstrukturen und None-Werte (ChromaDB kompatibel)
    clean_header_info = {}
    for k, v in chunk_header_info.items():
        if k == 'header_elements':
            # Überspringe komplexe Datenstrukturen
            continue
        elif v is not None and isinstance(v, (str, int, float, bool)):
            clean_header_info[k] = v
    
    clean_header_info.update(metadata_additions)
    
    return clean_header_info

def analyze_hierarchical_headers(elements: list) -> Dict[int, Dict[str, Any]]:
    """
    Analysiert Unstructured-Elemente und erstellt eine Header-Zuordnung.
    
    Da ElementMetadata read-only ist, geben wir eine Mapping-Tabelle zurück.
    
    Args:
        elements: Liste von Unstructured-Elementen
        
    Returns:
        dict: Mapping von Element-Index zu Header-Metadaten
    """
    print("[HEADER-ANALYSE] Starte hierarchische Header-Analyse...")
    
    current_headers = {
        'h1': None,
        'h2': None, 
        'h3': None,
        'current_section': None
    }
    
    header_stats = {
        'titles_found': 0,
        'headers_found': 0,
        'elements_with_headers': 0
    }
    
    # Mapping von Element-Index zu Header-Metadaten
    header_mapping = {}
    
    for i, element in enumerate(elements):
        element_text = str(element).strip()
        element_type = type(element).__name__
        
        # Prüfe ob Element eine Überschrift ist
        is_header = element_type in ['Title', 'Header'] and len(element_text) > 0
        
        if is_header:
            header_level = classify_header_level(element_text, element_type)
            
            if header_level == 'h1':
                # Neue Hauptüberschrift - Reset aller Sub-Header
                current_headers['h1'] = element_text
                current_headers['h2'] = None
                current_headers['h3'] = None
                current_headers['current_section'] = element_text
                header_stats['titles_found'] += 1
                print(f"  [H1] Neue Hauptüberschrift: '{element_text}'")
                
            elif header_level == 'h2':
                # Neue Unterüberschrift - Reset nur H3
                current_headers['h2'] = element_text
                current_headers['h3'] = None
                current_headers['current_section'] = element_text
                header_stats['headers_found'] += 1
                print(f"  [H2] Neue Unterüberschrift: '{element_text}'")
                
            elif header_level == 'h3':
                # Neue Detail-Überschrift
                current_headers['h3'] = element_text
                current_headers['current_section'] = element_text
                header_stats['headers_found'] += 1
                print(f"  [H3] Neue Detail-Überschrift: '{element_text}'")
        
        # Erstelle Header-Metadaten für dieses Element
        element_header_metadata = {
            'element_type': element_type,
            'is_header': is_header
        }
        
        # Füge aktuelle Header hinzu
        for key in ['h1', 'h2', 'h3', 'current_section']:
            if current_headers[key]:
                element_header_metadata[f'section_{key}'] = current_headers[key]
        
        # Speichere Metadaten für diesen Element-Index
        header_mapping[i] = element_header_metadata
        
        if any(current_headers.values()):
            header_stats['elements_with_headers'] += 1
    
    print_header_analysis_summary(header_stats, len(elements))
    
    return header_mapping

def classify_header_level(element_text: str, element_type: str) -> str:
    """
    Klassifiziert Header-Ebene basierend auf allgemeinen Text-Eigenschaften.
    
    Args:
        element_text: Text des Elements
        element_type: Typ des Elements
        
    Returns:
        str: Header-Level ('h1', 'h2', 'h3', oder 'content')
    """
    text_stripped = element_text.strip()
    text_length = len(text_stripped)
    
    # Leere oder sehr kurze Texte sind wahrscheinlich kein sinnvoller Header
    if text_length < 3:
        return 'content'
    
    # Sehr lange Texte sind wahrscheinlich kein Header, sondern Content
    if text_length > 200:
        return 'content'
    
    # PRIMÄR: Vertraue den von Unstructured erkannten Element-Typen
    # Diese sind durch OCR und Layout-Analyse bereits gut klassifiziert
    
    if element_type == "Title":
        # Titles sind meist Hauptüberschriften, aber länge Titles können auch H2 sein
        if text_length <= 50:
            return 'h1'
        else:
            return 'h2'
    
    elif element_type == "Header":
        # Headers sind typischerweise Abschnittsüberschriften
        return 'h2'
    
    elif element_type == "SubHeader":
        # SubHeaders sind Unterabschnittsüberschriften
        return 'h3'
    
    # SEKUNDÄR: Fallback-Heuristiken für unklare Element-Typen
    
    # Texte in Großbuchstaben sind oft Überschriften
    if text_stripped.isupper() and text_length <= 100:
        if text_length <= 30:
            return 'h1'
        elif text_length <= 60:
            return 'h2'
        else:
            return 'h3'
    
    # Texte, die mit Zahlen oder Buchstaben beginnen (z.B. "1. Einleitung", "A. Grundlagen")
    import re
    if re.match(r'^[\d]+\.?\s+', text_stripped) or re.match(r'^[A-Z]\.?\s+', text_stripped):
        if text_length <= 50:
            return 'h2'
        else:
            return 'h3'
    
    # Standard: Normaler Content
    return 'content'

def print_semantic_analysis_summary(semantic_chunks: List[Dict[str, Any]], documents: List[Document]):
    """
    Gibt eine Zusammenfassung der semantischen Analyse aus.
    
    Args:
        semantic_chunks: Liste der semantischen Chunks
        documents: Liste der erstellten Documents
    """
    print(f"[SEMANTIC] {len(semantic_chunks)} semantische Abschnitte erstellt:")
    for i, chunk in enumerate(semantic_chunks):
        print(f"  - Abschnitt {i+1}: '{chunk['section']}' ({chunk['length']} Zeichen)")
    
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

def print_header_analysis_summary(header_stats: Dict[str, int], total_elements: int):
    """
    Gibt eine Zusammenfassung der Header-Analyse aus.
    
    Args:
        header_stats: Statistiken der Header-Analyse
        total_elements: Gesamtanzahl der verarbeiteten Elemente
    """
    print(f"[HEADER-ANALYSE] Analyse abgeschlossen:")
    print(f"  - H1-Überschriften erkannt: {header_stats['titles_found']}")
    print(f"  - H2/H3-Überschriften erkannt: {header_stats['headers_found']}")
    print(f"  - Elemente mit Header-Zuordnung: {header_stats['elements_with_headers']}")
    print(f"  - Gesamt-Elemente verarbeitet: {total_elements}")

def enrich_document_metadata(document: Document, additional_metadata: Dict[str, Any]) -> Document:
    """
    Reichert ein Document mit zusätzlichen Metadaten an.
    
    Args:
        document: Ursprüngliches Document
        additional_metadata: Zusätzliche Metadaten
        
    Returns:
        Document mit angereicherten Metadaten
    """
    enhanced_metadata = document.metadata.copy() if document.metadata else {}
    enhanced_metadata.update(additional_metadata)
    
    return Document(
        text=document.text,
        metadata=enhanced_metadata
    )

def validate_document_quality(documents: List[Document]) -> Dict[str, Any]:
    """
    Validiert die Qualität der erstellten Dokumente.
    
    Args:
        documents: Liste der zu validierenden Dokumente
        
    Returns:
        Dictionary mit Qualitäts-Metriken
    """
    if not documents:
        return {'valid': False, 'error': 'Keine Dokumente vorhanden'}
    
    quality_metrics = {
        'total_documents': len(documents),
        'avg_text_length': sum(len(doc.text) for doc in documents) / len(documents),
        'documents_with_pages': len([d for d in documents if d.metadata.get('page_number')]),
        'documents_with_headers': len([d for d in documents if d.metadata.get('contains_headers')]),
        'unique_sections': len(set(d.metadata.get('section_title') for d in documents if d.metadata.get('section_title'))),
        'empty_documents': len([d for d in documents if not d.text.strip()]),
        'valid': True
    }
    
    # Qualitätswarnungen
    warnings = []
    if quality_metrics['empty_documents'] > 0:
        warnings.append(f"{quality_metrics['empty_documents']} leere Dokumente gefunden")
    
    if quality_metrics['documents_with_pages'] / len(documents) < 0.5:
        warnings.append("Weniger als 50% der Dokumente haben Seitennummern")
    
    if quality_metrics['avg_text_length'] < 100:
        warnings.append("Durchschnittliche Textlänge sehr kurz (< 100 Zeichen)")
    
    quality_metrics['warnings'] = warnings
    quality_metrics['quality_score'] = calculate_quality_score(quality_metrics)
    
    return quality_metrics

def calculate_quality_score(metrics: Dict[str, Any]) -> float:
    """
    Berechnet einen Qualitäts-Score für die Dokumente (0-100).
    
    Args:
        metrics: Qualitäts-Metriken
        
    Returns:
        Qualitäts-Score zwischen 0 und 100
    """
    score = 100.0
    
    # Abzüge für Probleme
    if metrics['empty_documents'] > 0:
        score -= (metrics['empty_documents'] / metrics['total_documents']) * 20
    
    if metrics['documents_with_pages'] / metrics['total_documents'] < 0.5:
        score -= 15
    
    if metrics['avg_text_length'] < 100:
        score -= 10
    
    if metrics['unique_sections'] < 2:
        score -= 10
    
    return max(0.0, score) 